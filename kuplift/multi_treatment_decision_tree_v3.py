# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Optional
import logging
import numpy as np
import pandas as pd

from kuplift.tree_v3 import TreeV3
from kuplift.dt_decision_binary_tree_cost_v3 import DTDecisionBinaryTreeCostV3
from kuplift.dt_decision_tree_node_split_v3 import DTDecisionTreeNodeSplitV3
from kuplift.leaf_selection_strategies_v3 import (
    select_leaf_v3,
    validate_leaf_selection_strategy,
)
from kuplift.encoding_selector_v3 import select_univariate_encoder_v3

logger = logging.getLogger(__name__)


class MultiTreatmentDecisionTreeV3:
    """
    DecisionTree V3:
      - Binary tree structure only
      - Automatic OUE/MTUE selection from treatment modality count
      - maxparts forced to 2
      - Configurable leaf selection strategy
      - Configurable cost model (injected)
      - Node dataset preserved

    Fit API (V1-style only):
      - fit(X, treatment_col, y_col, use_encoded_features=True)
        where:
          * X is a DataFrame of features
          * treatment_col is a Series (or array-like)
          * y_col is a Series (or array-like)
          * use_encoded_features controls split space:
              - True: split on encoded features (default behavior)
              - False: split on raw non-encoded features
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

        # Forced by specification
        self.maxparts = 2
        self.maxtreatmentgroups = maxtreatmentgroups
        self.control_name = control_name

        self.cost_model = (
            cost_model if cost_model is not None else DTDecisionBinaryTreeCostV3()
        )

        self.encoder = None
        self.encoder_type = None

        self.tree: TreeV3 | None = None
        self.tree_criterion: float = 0.0

        self.features: list[str] = []
        self.treatment_col_name: str | None = None
        self.target_col_name: str | None = None

        # Train/inference feature space
        self.use_encoded_features: bool = True

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

    def fit(
        self,
        data: pd.DataFrame,
        treatment_col,
        y_col,
        use_encoded_features: bool = False,
    ) -> "MultiTreatmentDecisionTreeV3":
        """
        V1-style API only:
          fit(X, treatment_col, y_col, use_encoded_features=True)
        """
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")

        logger.debug("fit(): start - n_rows=%s, n_features=%s", len(data), data.shape[1])

        X = data.reset_index(drop=True)
        t = pd.Series(treatment_col).reset_index(drop=True)
        y = pd.Series(y_col).reset_index(drop=True)

        if len(X) != len(t) or len(X) != len(y):
            raise ValueError("data, treatment_col and y_col must have same length")

        self.treatment_col_name = getattr(t, "name", None) or "treatment"
        self.target_col_name = getattr(y, "name", None) or "target"

        self.features = list(X.columns)
        if len(self.features) == 0:
            raise ValueError("No feature column available for training")

        self.use_encoded_features = bool(use_encoded_features)

        self.encoder, encoding_info = select_univariate_encoder_v3(t)
        self.encoder_type = encoding_info.encoder_name
        logger.debug(
            "fit(): encoder selected - encoder_type=%s, use_encoded_features=%s",
            self.encoder_type,
            self.use_encoded_features,
        )

        # Always fit encoder (keeps internal state coherent)
        if self.encoder_type == "OUE":
            X_enc = self.encoder.fit_transform(X, t, y, maxparts=self.maxparts)
        else:
            if self.maxtreatmentgroups is not None:
                X_enc = self.encoder.fit_transform(
                    X,
                    t,
                    y,
                    maxparts=self.maxparts,
                    maxtreatmentgroups=self.maxtreatmentgroups,
                )
            else:
                X_enc = self.encoder.fit_transform(X, t, y, maxparts=self.maxparts)

        if self.use_encoded_features:
            X_train = X_enc
            train_feature_names = list(X_enc.columns)
        else:
            X_train = X.copy()
            train_feature_names = list(X.columns)

        train_df = X_train.copy()
        train_df[self.treatment_col_name] = t.values
        train_df[self.target_col_name] = y.values

        self.tree = TreeV3(
            dataset=train_df,
            features=train_feature_names,
            treatment_col_name=self.treatment_col_name,
            target_col_name=self.target_col_name,
        )

        self.cost_model.initialize(self)
        self.tree_criterion = float(self.cost_model.compute_null_tree_cost(self.tree))
        logger.debug("fit(): initial tree_criterion=%.6f", self.tree_criterion)

        self._grow_tree()
        self.tree_criterion = float(self.cost_model.compute_total_tree_cost(self.tree))
        logger.debug(
            "fit(): done - final criterion=%.6f, internal_nodes=%s, leaf_nodes=%s",
            self.tree_criterion,
            len(self.internal_nodes),
            len(self.leaf_nodes),
        )
        return self

    def _grow_tree(self) -> None:
        if self.tree is None:
            return

        logger.debug("grow_tree(): start - max_depth=%s", self.max_depth)

        depth = 0
        while depth < self.max_depth:
            candidate_leaves = [
                node
                for node in self.tree.leaf_nodes
                if node.sample_size >= 2 * self.min_samples_leaf
            ]
            if not candidate_leaves:
                logger.debug("grow_tree(): stop - no candidate leaves at depth=%s", depth)
                break

            logger.debug(
                "grow_tree(): depth=%s, candidates=%s, current_criterion=%.6f",
                depth,
                len(candidate_leaves),
                self.tree_criterion,
            )

            node_vs_best_cost = {}
            node_vs_best_split = {}

            for leaf in candidate_leaves:
                best_cost = None
                best_split_desc = None

                for split_var in self.tree.features:
                    simulated = leaf.simulate_split(split_var)
                    if simulated is None:
                        continue

                    left_node, right_node, split_value, split_var_type = simulated

                    if left_node.sample_size < self.min_samples_leaf:
                        continue
                    if right_node.sample_size < self.min_samples_leaf:
                        continue

                    node_split = DTDecisionTreeNodeSplitV3(
                        splittable_node=leaf,
                        split_var=split_var,
                    )

                    cost = self.cost_model.compute_hypothetical_augmented_tree_cost(
                        tree=self.tree,
                        previous_cost=self.tree_criterion,
                        node_split=node_split,
                    )

                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost
                        best_split_desc = (
                            split_var,
                            left_node,
                            right_node,
                            split_value,
                            split_var_type,
                        )

                if best_cost is not None:
                    node_vs_best_cost[leaf] = best_cost
                    node_vs_best_split[leaf] = best_split_desc

            if not node_vs_best_cost:
                logger.debug("grow_tree(): stop - no valid improving split candidates")
                break

            selected_leaf = select_leaf_v3(
                strategy=self.leaf_selection,
                node_vs_cost=node_vs_best_cost,
                rng=self.rng,
            )
            selected_cost = node_vs_best_cost[selected_leaf]
            split_var, left_node, right_node, split_value, split_var_type = (
                node_vs_best_split[selected_leaf]
            )

            if selected_cost >= self.tree_criterion:
                logger.debug(
                    "grow_tree(): stop - no improvement (selected_cost=%.6f >= current=%.6f)",
                    selected_cost,
                    self.tree_criterion,
                )
                break

            self.tree.apply_split(
                node=selected_leaf,
                split_var=split_var,
                left_node=left_node,
                right_node=right_node,
                split_value=split_value,
                split_var_type=split_var_type,
            )
            logger.debug(
                "grow_tree(): split applied at depth=%s on node_id=%s var=%s type=%s new_criterion=%.6f",
                depth,
                selected_leaf.id,
                split_var,
                split_var_type,
                selected_cost,
            )

            self.tree_criterion = float(selected_cost)
            depth += 1

        logger.debug(
            "grow_tree(): end - reached depth=%s, internal_nodes=%s, leaf_nodes=%s",
            depth,
            len(self.internal_nodes),
            len(self.leaf_nodes),
        )

    def _encode_X_for_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")

        if not self.use_encoded_features:
            return X

        if self.encoder is None:
            raise RuntimeError("Model is not fitted")
        return self.encoder.transform(X)

    def _descend_to_leaf(self, row: pd.Series):
        node = self.root_node
        while node is not None and not node.is_leaf:
            var = node.split_var
            val = node.split_value
            typ = node.split_var_type

            x = row[var]
            if typ == "Numerical":
                go_left = x <= val
            else:
                go_left = x in set(val)

            node = node.left_node if go_left else node.right_node
        return node

    def predict_leaf_id(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")

        logger.debug("predict_leaf_id(): start - n_rows=%s", len(X))
        X_infer = self._encode_X_for_inference(X)
        out = []
        for _, row in X_infer.iterrows():
            leaf = self._descend_to_leaf(row)
            out.append(leaf.id if leaf is not None else -1)
        logger.debug("predict_leaf_id(): done")
        return np.array(out, dtype=int)

    def get_node_by_id(self, node_id: int):
        if self.tree is None:
            return None
        return self.tree.get_node_by_id(node_id)

    def get_node_path_str(self, node_id: int, separator: str = " AND ") -> str:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_node_path_str(node_id=node_id, separator=separator)

    def __str__(self) -> str:
        if self.tree is None:
            return (
                "MultiTreatmentDecisionTreeV3(unfitted, "
                f"leaf_selection={self.leaf_selection}, "
                f"encoder_type={self.encoder_type})"
            )
        return (
            "MultiTreatmentDecisionTreeV3("
            f"encoder_type={self.encoder_type}, "
            f"leaf_selection={self.leaf_selection}, "
            f"use_encoded_features={self.use_encoded_features}, "
            f"used_variable_count={self.used_variable_count}, "
            f"treatment_modality_count={self.treatment_modality_count}, "
            f"internal_nodes={len(self.internal_nodes)}, "
            f"leaf_nodes={len(self.leaf_nodes)}, "
            f"tree_criterion={self.tree_criterion:.6f})"
        )

    def tree_to_string(self, show_path: bool = False, max_depth: int | None = None) -> str:
        """
        Build and return an ASCII representation of the current tree
        with human-readable split conditions and paths.

        For binary encoded numerical variables (0/1):
          - left branch  => value set {0}
          - right branch => value set {1}

        More generally for numerical split threshold v:
          - left branch  => values <= v
          - right branch => values > v
        """
        if self.tree is None or self.root_node is None:
            return "MultiTreatmentDecisionTreeV3: tree is not fitted."

        lines: list[str] = []

        def _fmt_set(values, max_items: int = 5) -> str:
            vals = list(values)
            if len(vals) <= max_items:
                return "{" + ", ".join(map(str, vals)) + "}"
            head = ", ".join(map(str, vals[:max_items]))
            return "{" + head + ", ...}"

        def _branch_values_for_node(node):
            if node.parent is None:
                return "ROOT"

            inc = node.incoming_split
            parent = node.parent
            var = inc.var

            if var is None:
                return "ROOT"

            ptype = getattr(parent, "split_var_type", None)
            split_value = getattr(parent, "split_value", None)

            if ptype == "Numerical":
                if inc.op == "<=" and str(split_value) in {"0", "0.0"}:
                    return f"{var} ∈ {{0}}"
                if inc.op == ">" and str(split_value) in {"0", "0.0"}:
                    return f"{var} ∈ {{1}}"

                if inc.op == "<=":
                    return f"{var} <= {split_value}"
                if inc.op == ">":
                    return f"{var} > {split_value}"
                return f"{var} {inc.op} {inc.value}"

            if ptype == "Categorical":
                try:
                    left_group = list(split_value)
                except Exception:
                    left_group = [split_value]

                if inc.op == "<=":
                    return f"{var} ∈ {_fmt_set(left_group)}"
                if inc.op == ">":
                    return f"{var} ∉ {_fmt_set(left_group)}"
                return f"{var} {inc.op} {_fmt_set(left_group)}"

            return f"{inc.var} {inc.op} {inc.value}"

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

            var = node.split_var
            typ = node.split_var_type
            val = node.split_value

            if typ == "Numerical":
                if str(val) in {"0", "0.0"}:
                    return f" | split_rule: left {var}∈{{0}}, right {var}∈{{1}}"
                return f" | split_rule: left {var}<={val}, right {var}>{val}"

            if typ == "Categorical":
                try:
                    left_group = list(val)
                except Exception:
                    left_group = [val]
                return (
                    f" | split_rule: left {var}∈{_fmt_set(left_group)}, "
                    f"right {var}∉{_fmt_set(left_group)}"
                )

            return f" | split_value={val}"

        def _fmt_node(node):
            base = f"id={node.id} | type={node.type} | n={node.sample_size}"

            if not node.is_leaf:
                base += f" | split={node.split_var} ({node.split_var_type})"
                base += _fmt_split_rule(node)
            else:
                try:
                    treatments = list(node.get_treatments())
                    targets = list(node.get_targets())
                    base += f" | T={len(treatments)} | Y={len(targets)}"
                except Exception:
                    pass

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

        lines.append("Decision tree structure")
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
        """
        Return a Mermaid flowchart representation of the current tree.
        """
        if self.tree is None or self.root_node is None:
            return 'flowchart TD\n    A["Tree is not fitted"]'

        logger.debug("tree_to_mermaid(): start - max_depth=%s", max_depth)

        def _fmt_set(values, max_items: int = 5) -> str:
            vals = list(values)
            if len(vals) <= max_items:
                return "{" + ", ".join(map(str, vals)) + "}"
            head = ", ".join(map(str, vals[:max_items]))
            return "{" + head + ", ...}"

        def _escape(s: str) -> str:
            return str(s).replace('"', '\\"')

        def _node_key(node) -> str:
            return f"N{node.id}"

        def _node_label(node) -> str:
            if not show_node_stats:
                return f"node {node.id}"

            if node.is_leaf:
                try:
                    t_count = len(list(node.get_treatments()))
                    y_count = len(list(node.get_targets()))
                    return f"id={node.id}<br/>leaf<br/>n={node.sample_size}<br/>T={t_count}, Y={y_count}"
                except Exception:
                    return f"id={node.id}<br/>leaf<br/>n={node.sample_size}"

            var = node.split_var
            typ = node.split_var_type
            val = node.split_value

            if typ == "Numerical":
                if str(val) in {"0", "0.0"}:
                    rule = f"{var}: left={{0}}, right={{1}}"
                else:
                    rule = f"{var}: left<={val}, right>{val}"
            elif typ == "Categorical":
                try:
                    left_group = list(val)
                except Exception:
                    left_group = [val]
                rule = f"{var}: left∈{_fmt_set(left_group)}, right∉{_fmt_set(left_group)}"
            else:
                rule = f"{var}: split={val}"

            return f"id={node.id}<br/>internal<br/>n={node.sample_size}<br/>{rule}"

        def _edge_label(parent, child) -> str:
            inc = child.incoming_split
            var = inc.var
            op = inc.op
            typ = getattr(parent, "split_var_type", None)
            val = getattr(parent, "split_value", None)

            if typ == "Numerical":
                if str(val) in {"0", "0.0"}:
                    return f"{var}={0 if op == '<=' else 1}"
                return f"{var} {op} {val}"

            if typ == "Categorical":
                try:
                    left_group = list(val)
                except Exception:
                    left_group = [val]
                if op == "<=":
                    return f"{var} ∈ {_fmt_set(left_group)}"
                if op == ">":
                    return f"{var} ∉ {_fmt_set(left_group)}"
                return f"{var} {op} {_fmt_set(left_group)}"

            return f"{var} {op} {inc.value}"

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
        logger.debug("tree_to_mermaid(): done - nodes_exported=%s", len(visited))
        return "\n".join(lines)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict recommended treatment for each row.
        Output shape is aligned with input rows (same index as X).
        """
        if self.tree is None:
            raise RuntimeError("Model is not fitted")

        logger.debug("predict(): start - n_rows=%s", len(X))
        X_infer = self._encode_X_for_inference(X)

        preds = []
        for _, row in X_infer.iterrows():
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

        logger.debug("predict(): done")
        return pd.Series(preds, index=X.index, name="prediction")
