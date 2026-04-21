# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import pandas as pd

from .bayesian_decision_tree import BayesianDecisionTree
from .optimized_univariate_encoding import OptimizedUnivariateEncoding
from .multi_treatment_univariate_encoding import MultiTreatmentUnivariateEncoding


VariableSelectionStrategy = Literal[
    "max_level",
    "random_uniform",
    "random_weighted_by_level",
]


class MultiTreatmentDecisionTree(BayesianDecisionTree):
    """
    Decision tree supporting automatic selection between:
    - OptimizedUnivariateEncoding (2 treatments)
    - MultiTreatmentUnivariateEncoding (3+ treatments)

    It also supports per-node variable candidate selection strategies.
    """

    def __init__(
        self,
        control_name=None,
        *,
        variable_selection_strategy: VariableSelectionStrategy = "max_level",
        random_state: int | None = None,
        maxparts: int | None = None,
        maxtreatmentgroups: int = 2,
    ):
        super().__init__(control_name=control_name)

        self.variable_selection_strategy = variable_selection_strategy
        self.random_state = random_state
        self._rng = random.Random(random_state)
        self._np_rng = np.random.default_rng(random_state)

        self.maxparts = maxparts
        self.maxtreatmentgroups = min(2, maxtreatmentgroups if maxtreatmentgroups is not None else 2)

        self.encoder = None
        self.encoder_name = None
        self.levels_by_variable: dict[str, float] = {}

        # Store deterministic binary mapping for prediction consistency
        self._binary_treatment_reference = None

    def _binarize_treatment(self, treatment_col: pd.Series) -> pd.Series:
        """
        Convert a treatment column to binary 0/1 for compatibility with the
        existing binary tree internals.

        Strategy:
        - Use a deterministic reference modality as class 0
        - Map all other modalities to class 1
        """
        trt = pd.Series(treatment_col).copy()
        modalities = sorted(list(trt.dropna().unique()), key=lambda x: str(x))
        if len(modalities) < 2:
            raise ValueError("At least two treatment modalities are required")

        if self._binary_treatment_reference is None:
            self._binary_treatment_reference = modalities[0]

        ref = self._binary_treatment_reference
        return (trt != ref).astype(int)

    def fit(self, data: pd.DataFrame, treatment_col: pd.Series, y_col: pd.Series):
        """
        Fit tree with auto-selected encoding backend.

        The transformed data is fed to the inherited BayesianDecisionTree fitting routine.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame")

        unique_treatments = list(pd.Series(treatment_col).dropna().unique())
        n_treatments = len(unique_treatments)

        if n_treatments < 2:
            raise ValueError("At least two treatment modalities are required")

        if n_treatments == 2:
            self.encoder = OptimizedUnivariateEncoding()
            self.encoder_name = "OptimizedUnivariateEncoding"
            transformed = self.encoder.fit_transform(
                data,
                treatment_col.copy(),
                y_col.copy(),
                maxparts=self.maxparts,
            )
        else:
            self.encoder = MultiTreatmentUnivariateEncoding()
            self.encoder_name = "MultiTreatmentUnivariateEncoding"
            transformed = self.encoder.fit_transform(
                data,
                treatment_col.copy(),
                y_col.copy(),
                maxparts=self.maxparts,
                maxtreatmentgroups=2,  # enforce binary grouping
            )

        # Keep variable levels for per-node candidate selection
        self.levels_by_variable = dict(self.encoder.get_levels())

        # Binary treatment projection required by the current tree/node internals
        t_binary = self._binarize_treatment(treatment_col.copy())

        # Delegate to inherited Bayesian tree on encoded features
        return super().fit(transformed, t_binary, y_col.copy())

    def predict(self, X_test: pd.DataFrame):
        """
        Predict uplift values after applying the fitted encoder transform.
        """
        if self.encoder is None:
            raise RuntimeError("The model must be fitted before calling predict()")
        X_enc = self.encoder.transform(X_test.copy())
        return super().predict(X_enc)

    def _select_candidate_attributes_for_node(self, terminal_node):
        """
        Return a per-node candidate attribute list according to strategy.

        Strategies:
        - max_level: keep only the highest-level variable among node features
        - random_uniform: pick one variable uniformly
        - random_weighted_by_level: pick one variable with probability proportional to level
        """
        node_features = list(terminal_node.x.columns)
        if len(node_features) <= 1:
            return node_features

        levels = {f: float(self.levels_by_variable.get(f, 0.0)) for f in node_features}

        if self.variable_selection_strategy == "max_level":
            best = min(
                node_features,
                key=lambda f: (-levels[f], f),  # max level, deterministic tie-break
            )
            return [best]

        if self.variable_selection_strategy == "random_uniform":
            return [self._rng.choice(node_features)]

        if self.variable_selection_strategy == "random_weighted_by_level":
            weights = np.array([max(levels[f], 0.0) for f in node_features], dtype=float)
            if weights.sum() <= 0:
                # Fallback to uniform if all levels are null/non-positive
                return [self._rng.choice(node_features)]
            probs = weights / weights.sum()
            chosen_idx = int(self._np_rng.choice(len(node_features), p=probs))
            return [node_features[chosen_idx]]

        raise ValueError(
            "Unsupported variable_selection_strategy={!r}. "
            "Expected one of: 'max_level', 'random_uniform', 'random_weighted_by_level'.".format(
                self.variable_selection_strategy
            )
        )
