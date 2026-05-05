# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Optional
import logging
import numpy as np
import pandas as pd

from kuplift.tree_v3_global import TreeV3Global
from kuplift.node_v3_global import NodeV3Global, IncomingSplit
from kuplift.dt_decision_binary_tree_cost_v3 import DTDecisionBinaryTreeCostV3
from kuplift.dt_decision_tree_node_split_v3 import DTDecisionTreeNodeSplitV3
from kuplift.leaf_selection_strategies_v3 import (
    select_leaf_v3,
    validate_leaf_selection_strategy,
)
from kuplift.encoding_selector_v3 import select_univariate_encoder_v3
from kuplift.utils import transform_variable

logger = logging.getLogger(__name__)


class MultiTreatmentDecisionTreeV3Global:
    """
    Global-partition version:
      - fit encoder once on full train set
      - split masks at nodes are computed by applying learned partition to node raw dataset
      - raw node datasets are preserved
      - no silent fallback
    """

    def __init__(
        self,
        max_depth: int = 15,
        min_samples_leaf: int = 20,
        leaf_selection: str = "best_leaf",
        random_state: Optional[int] = None,
        cost_model=None,
        control_name=None,
        maxparts: int = 2,
        maxtreatmentgroups: Optional[int] = None,
    ):
        validate_leaf_selection_strategy(leaf_selection)

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.leaf_selection = leaf_selection
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.maxparts = 2
        self.maxtreatmentgroups = maxtreatmentgroups
        self.control_name = control_name

        self.cost_model = cost_model if cost_model is not None else DTDecisionBinaryTreeCostV3()

        self.encoder = None
        self.encoder_type = None

        self.tree: TreeV3Global | None = None
        self.tree_criterion: float = 0.0

        self.features: list[str] = []          # raw features
        self.encoded_features: list[str] = []  # informative encoded vars
        self.treatment_col_name: str | None = None
        self.target_col_name: str | None = None

        # strict metadata: encoded var -> source partition details
        self.partition_map: dict[str, dict] = {}

    @property
    def used_variable_count(self) -> int:
        return self.tree.used_variable_count if self.tree is not None else 0

    @property
    def target_modalities(self):
        return self.tree.target_modalities if self.tree is not None else []

    @property
    def treatment_modalities(self):
        return self.tree.treatment_modalities if self.tree is not None else []

    @property
    def root_node(self):
        return self.tree.root_node if self.tree is not None else None

    @property
    def internal_nodes(self):
        return self.tree.internal_nodes if self.tree is not None else []

    @property
    def leaf_nodes(self):
        return self.tree.leaf_nodes if self.tree is not None else []

    @property
    def treatment_modality_count(self) -> int:
        return self.tree.treatment_modality_count if self.tree is not None else 0

    def _build_partition_map(self) -> dict[str, dict]:
        if self.encoder is None:
            raise RuntimeError("encoder is not initialized")
        if not hasattr(self.encoder, "get_partitions"):
            raise RuntimeError("encoder does not expose get_partitions()")
        if not hasattr(self.encoder, "get_variable_type"):
            raise RuntimeError("encoder does not expose get_variable_type()")

        partitions = self.encoder.get_partitions()
        result = {}
        for var, parts in partitions.items():
            vtype = self.encoder.get_variable_type(var)
            result[var] = {
                "source_var": var,
                "source_type": vtype,
                "parts": parts,
            }
        return result

    def fit(self, data: pd.DataFrame, treatment_col, y_col) -> "MultiTreatmentDecisionTreeV3Global":
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")

        X = data.reset_index(drop=True)
        t = pd.Series(treatment_col).reset_index(drop=True)
        y = pd.Series(y_col).reset_index(drop=True)

        if len(X) != len(t) or len(X) != len(y):
            raise ValueError("data, treatment_col and y_col must have same length")

        self.treatment_col_name = getattr(t, "name", None) or "treatment"
        self.target_col_name = getattr(y, "name", None) or "target"

        self.features = list(X.columns)
        if not self.features:
            raise ValueError("No feature column available for training")

        self.encoder, encoding_info = select_univariate_encoder_v3(t)
        self.encoder_type = encoding_info.encoder_name

        if self.encoder_type == "OUE":
            X_enc = self.encoder.fit_transform(X, t, y, maxparts=self.maxparts)
        else:
            if self.maxtreatmentgroups is not None:
                X_enc = self.encoder.fit_transform(
                    X, t, y, maxparts=self.maxparts, maxtreatmentgroups=self.maxtreatmentgroups
                )
            else:
                X_enc = self.encoder.fit_transform(X, t, y, maxparts=self.maxparts)

        if not isinstance(X_enc, pd.DataFrame):
            raise RuntimeError("encoder.fit_transform must return a DataFrame")

        self.encoded_features = list(X_enc.columns)
        if not self.encoded_features:
            raise RuntimeError("No informative encoded feature returned by encoder")

        self.partition_map = self._build_partition_map()
        for f in self.encoded_features:
            if f not in self.partition_map:
                raise RuntimeError(f"Missing partition metadata for encoded feature {f!r}")

        # tree nodes keep RAW data only
        raw_train_df = X.copy()
        raw_train_df[self.treatment_col_name] = t.values
        raw_train_df[self.target_col_name] = y.values

        self.tree = TreeV3Global(
            dataset=raw_train_df,
            features=self.encoded_features,
            treatment_col_name=self.treatment_col_name,
            target_col_name=self.target_col_name,
        )

        self.cost_model.initialize(self)
        self.tree_criterion = float(self.cost_model.compute_null_tree_cost(self.tree))
        self._grow_tree()
        self.tree_criterion = float(self.cost_model.compute_total_tree_cost(self.tree))
        return self

    def _encode_single_variable_on_node(self, node: NodeV3Global, encoded_var: str) -> pd.Series:
        if encoded_var not in self.partition_map:
            raise RuntimeError(f"Missing partition metadata for {encoded_var!r}")
        info = self.partition_map[encoded_var]
        source_var = info["source_var"]
        parts = info["parts"]

        if source_var not in node.dataset.columns:
            raise RuntimeError(f"Source variable {source_var!r} missing from node dataset")

        return transform_variable(parts, node.dataset[source_var])

    def _simulate_split_on_encoded_var(self, node: NodeV3Global, encoded_var: str):
        encoded_values = self._encode_single_variable_on_node(node, encoded_var)
        observed_parts = sorted(list(pd.Series(encoded_values).dropna().unique()))
        if len(observed_parts) <= 1:
            return None
        if len(observed_parts) > 2:
            raise RuntimeError(
                f"Expected <=2 observed encoded parts for {encoded_var!r}, got {observed_parts}"
            )

        left_part = observed_parts[0]
        right_part = observed_parts[1]

        left_mask = (encoded_values == left_part)
        right_mask = (encoded_values == right_part)

        left_df = node.dataset[left_mask].copy()
        right_df = node.dataset[right_mask].copy()
        if len(left_df) == 0 or len(right_df) == 0:
            return None

        info = self.partition_map[encoded_var]
        split_var_type = info["source_type"]
        split_value = 0.5  # encoded threshold between part 0 and 1

        left_node = NodeV3Global(
            dataset=left_df,
            treatment_col_name=node.treatment_col_name,
            target_col_name=node.target_col_name,
            parent=node,
            incoming_split=IncomingSplit(var=encoded_var, op="<=", value=split_value),
        )
        right_node = NodeV3Global(
            dataset=right_df,
            treatment_col_name=node.treatment_col_name,
            target_col_name=node.target_col_name,
            parent=node,
            incoming_split=IncomingSplit(var=encoded_var, op=">", value=split_value),
        )

        n_values_cat = None
        if split_var_type == "Categorical":
            source_var = info["source_var"]
            n_values_cat = int(node.dataset[source_var].nunique(dropna=False))
            n_values_cat = max(n_values_cat, 1)

        return left_node, right_node, split_value, split_var_type, info, n_values_cat

    def _grow_tree(self) -> None:
        if self.tree is None:
            return

        depth = 0
        while depth < self.max_depth:
            candidate_leaves = [
                node for node in self.tree.leaf_nodes
                if node.sample_size >= 2 * self.min_samples_leaf
            ]
            if not candidate_leaves:
                break

            node_vs_best_cost = {}
            node_vs_best_split = {}

            for leaf in candidate_leaves:
                best_cost = None
                best_split_desc = None

                for split_var in self.tree.features:
                    simulated = self._simulate_split_on_encoded_var(leaf, split_var)
                    if simulated is None:
                        continue

                    left_node, right_node, split_value, split_var_type, source_info, n_values_cat = simulated

                    if left_node.sample_size < self.min_samples_leaf or right_node.sample_size < self.min_samples_leaf:
                        continue

                    node_split = DTDecisionTreeNodeSplitV3(
                        splittable_node=leaf,
                        split_var=split_var,
                    )
                    node_split.set_simulated_split(
                        left_node=left_node,
                        right_node=right_node,
                        split_value=split_value,
                        split_var_type=split_var_type,
                        n_values_of_categorical_split_var=n_values_cat,
                    )

                    cost = self.cost_model.compute_hypothetical_augmented_tree_cost(
                        tree=self.tree,
                        previous_cost=self.tree_criterion,
                        node_split=node_split,
                    )

                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost
                        best_split_desc = (
                            split_var, left_node, right_node, split_value, split_var_type, source_info, n_values_cat
                        )

                if best_cost is not None:
                    node_vs_best_cost[leaf] = best_cost
                    node_vs_best_split[leaf] = best_split_desc

            if not node_vs_best_cost:
                break

            selected_leaf = select_leaf_v3(
                strategy=self.leaf_selection,
                node_vs_cost=node_vs_best_cost,
                rng=self.rng,
            )
            selected_cost = node_vs_best_cost[selected_leaf]
            split_var, left_node, right_node, split_value, split_var_type, source_info, n_values_cat = node_vs_best_split[selected_leaf]

            if selected_cost >= self.tree_criterion:
                break

            self.tree.apply_split(
                node=selected_leaf,
                split_var=split_var,
                left_node=left_node,
                right_node=right_node,
                split_value=split_value,
                split_var_type=split_var_type,
                source_partition_info=source_info,
                n_values_of_categorical_split_var=n_values_cat,
            )

            self.tree_criterion = float(selected_cost)
            depth += 1

    def _descend_to_leaf(self, row: pd.Series):
        node = self.root_node
        while node is not None and not node.is_leaf:
            info = node.source_partition_info
            if info is None:
                raise RuntimeError(f"Missing source_partition_info on internal node id={node.id}")
            source_var = info["source_var"]
            parts = info["parts"]
            if source_var not in row.index:
                raise RuntimeError(f"Input row missing source variable {source_var!r}")

            encoded_idx = int(transform_variable(parts, pd.Series([row[source_var]])).iloc[0])
            go_left = (encoded_idx <= node.split_value)
            node = node.left_node if go_left else node.right_node
        return node

    def predict_leaf_id(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")

        out = []
        for _, row in X.iterrows():
            leaf = self._descend_to_leaf(row)
            out.append(leaf.id if leaf is not None else -1)
        return np.array(out, dtype=int)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")

        preds = []
        for _, row in X.iterrows():
            leaf = self._descend_to_leaf(row)
            if leaf is None:
                preds.append(None)
                continue

            treatments = list(leaf.get_treatments())
            targets = list(leaf.get_targets())

            if not treatments:
                preds.append(None)
                continue

            positive = None
            for cand in [1, "1", True, "True"]:
                if cand in targets:
                    positive = cand
                    break
            if positive is None:
                positive = targets[-1] if targets else None

            best_t = None
            best_rate = -1.0
            for t in treatments:
                total_t = 0
                pos_t = 0
                for y in targets:
                    c = int(leaf.get_count(t, y))
                    total_t += c
                    if y == positive:
                        pos_t += c
                rate = (pos_t / total_t) if total_t > 0 else 0.0
                if rate > best_rate:
                    best_rate = rate
                    best_t = t
            preds.append(best_t)

        return pd.Series(preds, index=X.index, name="prediction")

    def get_node_by_id(self, node_id: int):
        if self.tree is None:
            return None
        return self.tree.get_node_by_id(node_id)

    def get_node_path_str(self, node_id: int, separator: str = " AND ") -> str:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_node_path_str(node_id=node_id, separator=separator)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self.tree is None:
            return (
                "MultiTreatmentDecisionTreeV3Global(unfitted, "
                f"leaf_selection={self.leaf_selection}, "
                f"encoder_type={self.encoder_type})"
            )
        return (
            "MultiTreatmentDecisionTreeV3Global("
            f"encoder_type={self.encoder_type}, "
            f"leaf_selection={self.leaf_selection}, "
            f"used_variable_count={self.used_variable_count}, "
            f"treatment_modality_count={self.treatment_modality_count}, "
            f"internal_nodes={len(self.internal_nodes)}, "
            f"leaf_nodes={len(self.leaf_nodes)}, "
            f"tree_criterion={self.tree_criterion:.6f})"
        )

    def _part_to_text(self, part) -> str:
        ptype = part.part_type()
        if ptype == "Interval":
            if getattr(part, "is_missing", False):
                return "MISSING"
            lb = "-∞" if getattr(part, "is_left_open", False) else str(part.lower_bound)
            rb = "+∞" if getattr(part, "is_right_open", False) else str(part.upper_bound)
            return f"({lb}, {rb}]"
        if ptype == "Value group":
            vals = list(getattr(part, "values", []))
            return "{" + ", ".join(map(str, vals)) + "}"
        return str(part)

    def _split_rule_from_info(self, node) -> tuple[str, str]:
        info = getattr(node, "source_partition_info", None)
        if info is None:
            raise RuntimeError(f"Missing source_partition_info on node id={node.id}")

        source_var = info["source_var"]
        source_type = info["source_type"]
        parts = info["parts"]
        if parts is None or len(parts) < 2:
            raise RuntimeError(
                f"Expected at least 2 parts in source_partition_info on node id={node.id}, got {parts}"
            )

        left_txt = self._part_to_text(parts[0])
        right_txt = self._part_to_text(parts[1])

        if source_type == "Numerical":
            return f"{source_var} in {left_txt}", f"{source_var} in {right_txt}"
        elif source_type == "Categorical":
            return f"{source_var} in {left_txt}", f"{source_var} in {right_txt}"
        else:
            raise RuntimeError(
                f"Unsupported source_type={source_type!r} in source_partition_info on node id={node.id}"
            )

    def tree_to_string(self, show_path: bool = False, max_depth: int | None = None) -> str:
        if self.tree is None or self.root_node is None:
            return "MultiTreatmentDecisionTreeV3Global: tree is not fitted."

        lines: list[str] = []

        def _branch_values_for_node(node):
            if node.parent is None:
                return "ROOT"

            parent = node.parent
            left_rule, right_rule = self._split_rule_from_info(parent)

            # incoming op convention: "<=" means left, ">" means right
            if node.incoming_split.op == "<=":
                return left_rule
            if node.incoming_split.op == ">":
                return right_rule
            raise RuntimeError(
                f"Unsupported incoming split operator {node.incoming_split.op!r} for node id={node.id}"
            )

        def _human_path(node, separator: str = " AND ") -> str:
            if node.parent is None:
                return "ROOT"
            parts = []
            cur = node
            while cur is not None and cur.parent is not None:
                parts.append(_branch_values_for_node(cur))
                cur = cur.parent
            parts.reverse()
            return separator.join(parts) if parts else "ROOT"

        def _fmt_split_rule(node) -> str:
            if node.is_leaf:
                return ""
            left_rule, right_rule = self._split_rule_from_info(node)
            return f" | split_rule: left [{left_rule}], right [{right_rule}]"

        def _fmt_node(node):
            base = f"id={node.id} | type={node.type} | n={node.sample_size}"
            if not node.is_leaf:
                src = node.source_partition_info
                if src is None:
                    raise RuntimeError(f"Missing source_partition_info on node id={node.id}")
                base += f" | split={src['source_var']} ({src['source_type']})"
                base += _fmt_split_rule(node)
            else:
                treatments = list(node.get_treatments())
                targets = list(node.get_targets())
                base += f" | T={len(treatments)} | Y={len(targets)}"

            if show_path:
                base += f" | path={_human_path(node)}"
            return base

        def _walk(node, prefix: str, is_last: bool, depth: int):
            if max_depth is not None and depth > max_depth:
                return

            branch = "└── " if is_last else "├── "
            cond = _branch_values_for_node(node)
            lines.append(f"{prefix}{branch}[{cond}] {_fmt_node(node)}")

            if node.is_leaf:
                return

            children = [c for c in [node.left_node, node.right_node] if c is not None]
            if not children:
                return

            next_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                _walk(child, next_prefix, i == len(children) - 1, depth + 1)

        lines.append("Decision tree structure (Global partitions)")
        lines.append(
            f"criterion={self.tree_criterion:.6f} | "
            f"internal={len(self.internal_nodes)} | leaves={len(self.leaf_nodes)}"
        )
        lines.append(f"└── [ROOT] {_fmt_node(self.root_node)}")

        if not self.root_node.is_leaf:
            children = [c for c in [self.root_node.left_node, self.root_node.right_node] if c is not None]
            for i, child in enumerate(children):
                _walk(child, "    ", i == len(children) - 1, depth=1)

        return "\n".join(lines)

    def print_tree(self, show_path: bool = False, max_depth: int | None = None) -> None:
        print(self.tree_to_string(show_path=show_path, max_depth=max_depth))

    def tree_to_mermaid(self, max_depth: int | None = None, show_node_stats: bool = True) -> str:
        if self.tree is None or self.root_node is None:
            return 'flowchart TD\n    A["Tree is not fitted"]'

        def _escape(s: str) -> str:
            return str(s).replace('"', '\\"')

        def _node_key(node) -> str:
            return f"N{node.id}"

        def _split_rule_pair(node) -> tuple[str, str]:
            return self._split_rule_from_info(node)

        def _node_label(node) -> str:
            if not show_node_stats:
                return f"node {node.id}"

            if node.is_leaf:
                t_count = len(list(node.get_treatments()))
                y_count = len(list(node.get_targets()))
                return f"id={node.id}<br/>leaf<br/>n={node.sample_size}<br/>T={t_count}, Y={y_count}"

            src = node.source_partition_info
            if src is None:
                raise RuntimeError(f"Missing source_partition_info on node id={node.id}")

            left_rule, right_rule = _split_rule_pair(node)
            return (
                f"id={node.id}<br/>internal<br/>n={node.sample_size}"
                f"<br/>{src['source_var']} ({src['source_type']})"
                f"<br/>left: {left_rule}"
                f"<br/>right: {right_rule}"
            )

        def _edge_label(parent, child) -> str:
            left_rule, right_rule = _split_rule_pair(parent)
            if child.incoming_split.op == "<=":
                return left_rule
            if child.incoming_split.op == ">":
                return right_rule
            raise RuntimeError(
                f"Unsupported incoming split operator {child.incoming_split.op!r} for node id={child.id}"
            )

        lines = ["flowchart TD"]
        visited = set()

        def _walk(node, depth: int):
            if node is None:
                return
            if max_depth is not None and depth > max_depth:
                return

            nk = _node_key(node)
            if nk not in visited:
                visited.add(nk)
                label = _escape(_node_label(node))
                if node.is_leaf:
                    lines.append(f'    {nk}(["{label}"])')
                else:
                    lines.append(f'    {nk}["{label}"]')

            if node.is_leaf:
                return

            children = [c for c in [node.left_node, node.right_node] if c is not None]
            for child in children:
                if max_depth is not None and depth + 1 > max_depth:
                    continue

                ck = _node_key(child)
                if ck not in visited:
                    clabel = _escape(_node_label(child))
                    if child.is_leaf:
                        lines.append(f'    {ck}(["{clabel}"])')
                    else:
                        lines.append(f'    {ck}["{clabel}"]')
                    visited.add(ck)

                elabel = _escape(_edge_label(node, child))
                lines.append(f'    {nk} -->|"{elabel}"| {ck}')

                _walk(child, depth + 1)

        _walk(self.root_node, 0)
        return "\n".join(lines)
