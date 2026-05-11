# kuplift/mt_random_forest.py
# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

# kuplift/mt_random_forest.py
from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd

from kuplift.mt_decision_tree import DecisionTree
from kuplift.utils import join_jt


class RandomForest:
    """
    RandomForest for uplift-style multi-treatment probabilities.

    - Each tree is a DecisionTree with leaf_selection="random"
    - Each tree is trained on all rows, but only a random subset of features
      (max_features=20 by default, or all if fewer are available)
    - predict() averages per-tree positive-class probabilities per treatment
    """

    def __init__(
        self,
        n_trees: int = 30,
        max_features: int = 20,
        random_state: Optional[int] = None,
        # DecisionTree params forwarded
        max_depth: int = 15,
        min_samples_leaf: int = 20,
        cost_model=None,
        control_name=None,
        maxparts: int = 2,
        maxtreatmentgroups: Optional[int] = None,
        local_fit_mode: str = "per_leaf",
    ):
        if n_trees <= 0:
            raise ValueError("n_trees must be >= 1")
        if max_features <= 0:
            raise ValueError("max_features must be >= 1")

        self.n_trees = int(n_trees)
        self.max_features = int(max_features)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Stored DT hyperparameters
        self.dt_params = dict(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            leaf_selection="random",  # forced by requirement
            random_state=None,        # set per-tree for diversity
            cost_model=cost_model,
            control_name=control_name,
            maxparts=maxparts,
            maxtreatmentgroups=maxtreatmentgroups,
            local_fit_mode=local_fit_mode,
        )

        # Fitted attributes
        self.trees: list[DecisionTree] = []
        self.tree_features: list[list[str]] = []
        self.features: list[str] = []
        self.treatment_col_name: Optional[str] = None
        self.target_col_name: Optional[str] = None
        self.positive_target = None
        self.treatment_modalities: list = []
        self._is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        treatment_col,
        y_col,
        positive_target=None,
    ) -> "RandomForest":
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        X = data.reset_index(drop=True)
        t = pd.Series(treatment_col).reset_index(drop=True)
        y = pd.Series(y_col).reset_index(drop=True)

        if len(X) != len(t) or len(X) != len(y):
            raise ValueError("data, treatment_col and y_col must have same length")

        self.features = list(X.columns)
        if not self.features:
            raise ValueError("No feature column available for training")

        self.treatment_col_name = getattr(t, "name", None) or "treatment"
        self.target_col_name = getattr(y, "name", None) or "target"

        self.trees = []
        self.tree_features = []

        n_available = len(self.features)
        k = min(self.max_features, n_available)

        for _ in range(self.n_trees):
            # random feature subset for this tree
            if k == n_available:
                feat_subset = list(self.features)
            else:
                feat_subset = self.rng.choice(self.features, size=k, replace=False).tolist()

            # independent random seed per tree
            tree_seed = int(self.rng.integers(0, 2**32 - 1))

            tree = DecisionTree(
                **{**self.dt_params, "random_state": tree_seed}
            )

            # Train tree on all rows but selected columns
            tree.fit(
                data=X[feat_subset].copy(),
                treatment_col=t,
                y_col=y,
                positive_target=positive_target,
            )

            self.trees.append(tree)
            self.tree_features.append(feat_subset)

        # Forest-level metadata from first tree
        first = self.trees[0]
        self.positive_target = first.positive_target
        self.treatment_modalities = list(first.treatment_modalities)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        result_type: Literal["df", "ndarray", "lists"] = "ndarray",
    ):
        if not self._is_fitted or not self.trees:
            raise RuntimeError("Model is not fitted")

        # Accept non-DataFrame input as in DecisionTree.predict_probabilities
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        else:
            # only check availability; each tree will receive its own subset
            missing = [c for c in self.features if c not in X.columns]
            if missing:
                raise ValueError(f"Input X is missing training feature columns: {missing}")

        # Aggregate probabilities
        per_tree = []
        for tree, feat_subset in zip(self.trees, self.tree_features):
            probs = tree.predict_probabilities(
                X[feat_subset],
                result_type="ndarray",
            )
            per_tree.append(probs)

        # shape: (n_trees, n_samples, n_treatments)
        stack = np.stack(per_tree, axis=0)
        mean_probs = np.mean(stack, axis=0)  # (n_samples, n_treatments)

        if result_type == "ndarray":
            return mean_probs
        if result_type == "lists":
            return mean_probs.tolist()
        if result_type == "df":
            cols = [join_jt(self.positive_target, t) for t in self.treatment_modalities]
            return pd.DataFrame(mean_probs, index=X.index, columns=cols)

        raise ValueError(f"invalid result type {result_type!r}")
