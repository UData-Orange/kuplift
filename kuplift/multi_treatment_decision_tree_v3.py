# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Optional
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
      - fit(X, treatment_col, y_col)
        where:
          * X is a DataFrame of features
          * treatment_col is a Series (or array-like)
          * y_col is a Series (or array-like)
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
    ) -> "MultiTreatmentDecisionTreeV3":
        """
        V1-style API only:
          fit(X, treatment_col, y_col)
        """
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
        if len(self.features) == 0:
            raise ValueError("No feature column available for training")

        self.encoder, encoding_info = select_univariate_encoder_v3(t)
        self.encoder_type = encoding_info.encoder_name

        if self.encoder_type == "OUE":
            X_enc = self.encoder.fit_transform(X, t, y, maxparts=self.maxparts)
        else:
            X_enc = self.encoder.fit_transform(X, t, y, maxparts=self.maxparts, maxtreatmentgroups=self.maxtreatmentgroups)

        encoded_feature_names = list(X_enc.columns)
        train_df = X_enc.copy()
        train_df[self.treatment_col_name] = t.values
        train_df[self.target_col_name] = y.values

        self.tree = TreeV3(
            dataset=train_df,
            features=encoded_feature_names,
            treatment_col_name=self.treatment_col_name,
            target_col_name=self.target_col_name,
        )

        self.cost_model.initialize(self)
        self.tree_criterion = float(self.cost_model.compute_null_tree_cost(self.tree))

        self._grow_tree()
        self.tree_criterion = float(self.cost_model.compute_total_tree_cost(self.tree))
        return self

    def _grow_tree(self) -> None:
        if self.tree is None:
            return

        depth = 0
        while depth < self.max_depth:
            candidate_leaves = [
                node
                for node in self.tree.leaf_nodes
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
                break

            self.tree.apply_split(
                node=selected_leaf,
                split_var=split_var,
                left_node=left_node,
                right_node=right_node,
                split_value=split_value,
                split_var_type=split_var_type,
            )
            self.tree_criterion = float(selected_cost)
            depth += 1

    def _encode_X_for_inference(self, X: pd.DataFrame) -> pd.DataFrame:
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
        X_enc = self._encode_X_for_inference(X)
        out = []
        for _, row in X_enc.iterrows():
            leaf = self._descend_to_leaf(row)
            out.append(leaf.id if leaf is not None else -1)
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
        """
        if self.tree is None or self.root_node is None:
            return "MultiTreatmentDecisionTreeV3: tree is not fitted."

        lines: list[str] = []

        def _human_condition_from_incoming(node) -> str:
            if node.parent is None:
                return "ROOT"

            inc = node.incoming_split
            parent = node.parent
            var = inc.var

            if var is None:
                return "ROOT"

            ptype = getattr(parent, "split_var_type", None)

            # Numerical split shown as binary parts
            if ptype == "Numerical":
                if inc.op == "<=":
                    return f"{var} ∈ part_1"
                if inc.op == ">":
                    return f"{var} ∈ part_2"
                return f"{var} ∈ part_?"

            # Categorical split
            if ptype == "Categorical":
                if inc.op == "<=":
                    return f"{var} ∈ left_group"
                if inc.op == ">":
                    return f"{var} ∈ right_group"
                return f"{var} ∈ group"

            # Fallback
            return f"{inc.var} {inc.op} {inc.value}"

        def _human_path(node, separator: str = " AND ") -> str:
            if node.parent is None:
                return "ROOT"

            parts = []
            cur = node
            while cur is not None and cur.parent is not None:
                parts.append(_human_condition_from_incoming(cur))
                cur = cur.parent
            parts.reverse()
            return separator.join(parts) if parts else "ROOT"

        def _fmt_node(node):
            base = f"id={node.id} | type={node.type} | n={node.sample_size}"

            if not node.is_leaf:
                base += f" | split={node.split_var} ({node.split_var_type})"
                if node.split_var_type == "Numerical":
                    base += " | split_rule=part_1 / part_2"
                elif node.split_var_type == "Categorical":
                    base += " | split_rule=left_group / right_group"
                else:
                    base += f" | split_value={node.split_value}"
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
            cond = _human_condition_from_incoming(node)
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
