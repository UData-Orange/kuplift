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
from kuplift.leaf_selection_strategies_v3 import select_leaf_v3, validate_leaf_selection_strategy
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
    """

    def __init__(
        self,
        max_depth: int = 3,
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

        self.cost_model = cost_model if cost_model is not None else DTDecisionBinaryTreeCostV3()

        # Learned artifacts
        self.encoder = None
        self.encoder_type = None

        self.tree: TreeV3 | None = None
        self.tree_criterion: float = 0.0

        self.features: list[str] = []
        self.treatment_col_name: str | None = None
        self.target_col_name: str | None = None

    # ------------------------------------------------------------------
    # Exposed tree-level properties (required)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        y_col: str,
    ) -> "MultiTreatmentDecisionTreeV3":
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")
        if treatment_col not in data.columns:
            raise ValueError(f"Missing treatment column: {treatment_col}")
        if y_col not in data.columns:
            raise ValueError(f"Missing target column: {y_col}")

        self.treatment_col_name = treatment_col
        self.target_col_name = y_col

        # Select features
        self.features = [c for c in data.columns if c not in {treatment_col, y_col}]
        if len(self.features) == 0:
            raise ValueError("No feature column available for training")

        # 1) Auto-select encoder (OUE/MTUE)
        self.encoder, encoding_info = select_univariate_encoder_v3(data[treatment_col])
        self.encoder_type = encoding_info.encoder_name

        # 2) Force maxparts=2 in all cases
        X = data[self.features]
        t = data[treatment_col]
        y = data[y_col]

        if self.encoder_type == "OUE":
            X_enc = self.encoder.fit_transform(
                data=X,
                treatment_col=t,
                y_col=y,
                maxparts=2,
            )
        else:
            # MTUE
            fit_kwargs = dict(
                data=X,
                treatment_col=t,
                y_col=y,
                maxparts=2,
            )
            if self.maxtreatmentgroups is not None:
                fit_kwargs["maxtreatmentgroups"] = self.maxtreatmentgroups
            X_enc = self.encoder.fit_transform(**fit_kwargs)

        # Build encoded dataset used by the tree
        encoded_feature_names = list(X_enc.columns)
        train_df = X_enc.copy()
        train_df[treatment_col] = t.values
        train_df[y_col] = y.values

        self.tree = TreeV3(
            dataset=train_df,
            features=encoded_feature_names,
            treatment_col_name=treatment_col,
            target_col_name=y_col,
        )

        # Cost model init
        self.cost_model.initialize(self)
        self.tree_criterion = float(self.cost_model.compute_null_tree_cost(self.tree))

        # Grow
        self._grow_tree()

        # Final criterion recompute for consistency
        self.tree_criterion = float(self.cost_model.compute_total_tree_cost(self.tree))
        return self

    def _grow_tree(self) -> None:
        if self.tree is None:
            return

        depth = 0
        while depth < self.max_depth:
            # Splittable leaves only
            candidate_leaves = [
                node for node in self.tree.leaf_nodes
                if node.sample_size >= 2 * self.min_samples_leaf
            ]
            if not candidate_leaves:
                break

            # For each candidate leaf, find best split variable wrt hypothetical tree cost
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

                    # leaf-size guard
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

                    # Keep realized simulation for later application
                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost
                        best_split_desc = (split_var, left_node, right_node, split_value, split_var_type)

                if best_cost is not None:
                    node_vs_best_cost[leaf] = best_cost
                    node_vs_best_split[leaf] = best_split_desc

            if not node_vs_best_cost:
                break

            # Select leaf by strategy
            selected_leaf = select_leaf_v3(
                strategy=self.leaf_selection,
                node_vs_cost=node_vs_best_cost,
                rng=self.rng,
            )
            selected_cost = node_vs_best_cost[selected_leaf]
            split_var, left_node, right_node, split_value, split_var_type = node_vs_best_split[selected_leaf]

            # Accept only improving split
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

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

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
                go_left = (x <= val)
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

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

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
