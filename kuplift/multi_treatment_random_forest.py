# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

import random
import numpy as np
import pandas as pd

from .multi_treatment_decision_tree import MultiTreatmentDecisionTree


class MultiTreatmentRandomForest:
    """
    Random forest counterpart of MultiTreatmentDecisionTree.

    Main behavior:
    - Automatic encoding backend selection is delegated to each tree.
    - Treatment grouping is constrained to binary (maxtreatmentgroups=2).
    - Supports variable selection strategy at tree/node level.
    """

    def __init__(
        self,
        n_trees=10,
        vars_subset=False,
        random_state=10,
        *,
        control_name=None,
        variable_selection_strategy="max_level",
        maxparts=None,
        maxtreatmentgroups=2,
    ):
        self.n_trees = n_trees
        self.vars_subset = vars_subset
        self.random_state = random_state

        self.control_name = control_name
        self.variable_selection_strategy = variable_selection_strategy
        self.maxparts = maxparts
        self.maxtreatmentgroups = min(2, maxtreatmentgroups if maxtreatmentgroups is not None else 2)

        self.list_of_trees = []
        self._rng = random.Random(self.random_state)

    def _sample_feature_columns(self, data: pd.DataFrame) -> list[str]:
        """
        Optionally sample a feature subset for one tree.
        """
        cols = list(data.columns)
        if not self.vars_subset:
            return cols

        m = max(1, int(np.sqrt(len(cols))))
        return self._rng.sample(cols, m)

    def fit(self, data: pd.DataFrame, treatment_col: pd.Series, y_col: pd.Series):
        """
        Fit all trees of the forest.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame")
        if len(data) == 0:
            raise ValueError("`data` must not be empty")

        self.list_of_trees = []

        for i in range(self.n_trees):
            feature_cols = self._sample_feature_columns(data)
            tree = MultiTreatmentDecisionTree(
                control_name=self.control_name,
                variable_selection_strategy=self.variable_selection_strategy,
                random_state=self.random_state + i,
                maxparts=self.maxparts,
                maxtreatmentgroups=2,  # enforce binary tree treatment groups
            )
            tree.fit(
                data[feature_cols].copy(),
                treatment_col.copy(),
                y_col.copy(),
            )
            self.list_of_trees.append(tree)

        return self

    def predict(self, X_test: pd.DataFrame, weighted_average=False):
        """
        Predict uplift for each sample by averaging tree predictions.
        """
        if not self.list_of_trees:
            raise RuntimeError("The forest must be fitted before calling predict()")

        all_preds = [np.array(tree.predict(X_test.copy())) for tree in self.list_of_trees]

        if not weighted_average:
            return np.mean(all_preds, axis=0)

        # Weighted average by inverse tree criterion (as in legacy BayesianRandomForest spirit)
        criteria = np.array([tree.tree_criterion for tree in self.list_of_trees], dtype=float)

        # Avoid division by zero if a criterion is numerically zero
        criteria = np.where(criteria == 0.0, 1e-12, criteria)
        inv = criteria.sum() / criteria
        weights = inv / inv.sum()

        return np.average(all_preds, axis=0, weights=weights)
