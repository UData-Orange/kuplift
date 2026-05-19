# kuplift/mt_random_forest.py
# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Optional, Literal
import copy
import numpy as np
import pandas as pd

from kuplift.mt_decision_tree import MultiTreatmentDecisionTree
from kuplift.utils import join_jt


class MultiTreatmentRandomForest:
    """
    MultiTreatmentRandomForest for uplift-style multi-treatment probabilities.

    - Each tree is a MultiTreatmentDecisionTree with leaf_selection="random"
    - Each tree is trained on all rows, but only a random subset of features
      (max_features=20 by default, or all if fewer are available)
    - At each split inside each tree, variables can also be sub-sampled via
      `split_max_features` (forwarded to MultiTreatmentDecisionTree)
    - predict() averages per-tree positive-class probabilities per treatment

    Notes
    -----
    - This class does not perform row bootstrap sampling by default.
      Diversity is induced through random feature subspaces and per-tree seeds.
    - Uplift output requires `control_name` to be set and present in treatment modalities.
    """

    def __init__(
        self,
        n_trees: int = 30,
        max_features: int = 20,
        random_state: Optional[int] = None,
        # MultiTreatmentDecisionTree params forwarded
        max_depth: int = 15,
        min_samples_leaf: int = 20,
        cost_model=None,
        control_name=None,
        maxparts: int = 2,
        maxtreatmentgroups: Optional[int] = None,
        local_fit_mode: str = "per_leaf",
        split_max_features: Optional[int] = None,
        max_cores = None,
        memory_limit_mb = None
    ):
        """
        Initialize the random forest.

        Parameters
        ----------
        n_trees : int, default=30
            Number of trees in the ensemble.
        max_features : int, default=20
            Number of feature columns sampled (without replacement) per tree.
        random_state : int | None, default=None
            Random seed for forest-level RNG.
        max_depth : int, default=15
            Forwarded to each MultiTreatmentDecisionTree.
        min_samples_leaf : int, default=20
            Forwarded to each MultiTreatmentDecisionTree.
        cost_model : object | None, default=None
            Cost model forwarded to each MultiTreatmentDecisionTree.
        control_name : Any, default=None
            Control treatment name used for uplift computation.
        maxparts : int, default=2
            Forwarded to each MultiTreatmentDecisionTree encoder fit.
        maxtreatmentgroups : int | None, default=None
            Forwarded to MTUE.
        local_fit_mode : {"per_leaf", "per_variable"}, default="per_leaf"
            Local fitting mode forwarded to each MultiTreatmentDecisionTree.
        split_max_features : int | None, default=None
            Number of candidate split variables sampled at each tree node expansion.
            Forwarded to each MultiTreatmentDecisionTree.
        max_cores : int | None, default=None
            Optional max cores for Khiops calls in trees.
        memory_limit_mb : int | None, default=None
            Optional memory limit for Khiops calls in trees.

        Raises
        ------
        ValueError
            If `n_trees <= 0` or `max_features <= 0`.
        """
        if n_trees <= 0:
            raise ValueError("n_trees must be >= 1")
        if max_features <= 0:
            raise ValueError("max_features must be >= 1")
        if split_max_features is not None and int(split_max_features) <= 0:
            raise ValueError("split_max_features must be >= 1 when provided")

        self.n_trees = int(n_trees)
        self.max_features = int(max_features)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.control_name = control_name

        self.max_cores = max_cores
        self.memory_limit_mb = memory_limit_mb

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
            split_max_features=split_max_features,
            max_cores=self.max_cores,
            memory_limit_mb=self.memory_limit_mb
        )

        # Fitted attributes
        self.trees: list[MultiTreatmentDecisionTree] = []
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
    ) -> "MultiTreatmentRandomForest":
        """
        Fit all trees of the forest.

        Parameters
        ----------
        data : pandas.DataFrame
            Feature matrix.
        treatment_col : array-like / pandas.Series
            Treatment column aligned with `data`.
        y_col : array-like / pandas.Series
            Target column aligned with `data`.
        positive_target : Any, default=None
            Positive target modality forwarded to each tree fit.

        Returns
        -------
        MultiTreatmentRandomForest
            Fitted estimator.

        Raises
        ------
        ValueError
            If data is empty, has no feature column, or lengths are inconsistent.
        TypeError
            If `data` is not a pandas DataFrame.
        """
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

            base_cost_model = self.dt_params.get("cost_model", None)
            tree_cost_model = None if base_cost_model is None else copy.deepcopy(base_cost_model)
            params = {**self.dt_params, "random_state": tree_seed, "cost_model": tree_cost_model}
            tree = MultiTreatmentDecisionTree(**params)

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
        self.negative_target = first.negative_target
        self.treatment_modalities = list(first.treatment_modalities)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        predict_probabilities: bool = True,
        predict_best_treatment: bool = True,
        predict_uplift: bool = True,
    ) -> pd.DataFrame:
        """
        Predict requested outputs for each sample.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Input feature matrix.
        predict_probabilities : bool, default=True
            Include class probabilities (negative then positive per treatment).
        predict_best_treatment : bool, default=True
            Include best treatment column according to maximal positive probability.
        predict_uplift : bool, default=True
            Include uplift column as max_t P(Y=positive|t) - P(Y=positive|control).

        Returns
        -------
        pandas.DataFrame
            Concatenated prediction blocks according to requested outputs.

        Raises
        ------
        RuntimeError
            If model is not fitted.
        ValueError
            If no output is requested, features are missing, or uplift cannot be computed.
        """
        if not self._is_fitted or not self.trees:
            raise RuntimeError("Model is not fitted")

        if not (predict_probabilities or predict_best_treatment or predict_uplift):
            raise ValueError("At least one output must be requested")

        # normalize input once
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        else:
            missing = [c for c in self.features if c not in X.columns]
            if missing:
                raise ValueError(f"Input X is missing training feature columns: {missing}")

        # Always compute positive probs for best_treatment/uplift
        pos_df = self.predict_probabilities(
            X,
            result_type="df",
            include_negative_probabilities=False,
        )

        blocks = []

        if predict_probabilities:
            probs_df = self.predict_probabilities(
                X,
                result_type="df",
                include_negative_probabilities=True,  # negative first, then positive
            )
            blocks.append(probs_df)

        if predict_best_treatment:
            best_idx = np.argmax(pos_df.to_numpy(), axis=1)
            best_t = [self.treatment_modalities[i] for i in best_idx]
            blocks.append(pd.DataFrame({"best_treatment": best_t}, index=X.index))

        if predict_uplift:
            if self.control_name is None:
                raise ValueError("predict_uplift=True requires MultiTreatmentRandomForest.control_name to be set")
            if self.control_name not in self.treatment_modalities:
                raise ValueError(
                    f"control_name={self.control_name!r} is not in treatment modalities: {self.treatment_modalities}"
                )

            pos_mat = pos_df.to_numpy()
            max_pos = np.max(pos_mat, axis=1)
            j_control = self.treatment_modalities.index(self.control_name)
            p_control = pos_mat[:, j_control]
            blocks.append(pd.DataFrame({"uplift": (max_pos - p_control)}, index=X.index))

        return pd.concat(blocks, axis=1)

    def predict_probabilities(
        self,
        X: pd.DataFrame,
        result_type: Literal["df", "ndarray", "lists"] = "ndarray",
        include_negative_probabilities: bool = False,
    ):
        """
        Predict treatment-wise probabilities by averaging tree outputs.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Input feature matrix.
        result_type : {"df", "ndarray", "lists"}, default="ndarray"
            Output format.
        include_negative_probabilities : bool, default=False
            If True, prepend negative probabilities as `1 - positive`.

        Returns
        -------
        pandas.DataFrame | numpy.ndarray | list
            Predicted probabilities in requested format.

        Raises
        ------
        RuntimeError
            If model is not fitted.
        ValueError
            If input features are incomplete or `result_type` is invalid.
        """
        if not self._is_fitted or not self.trees:
            raise RuntimeError("Model is not fitted")

        # Accept non-DataFrame input as in MultiTreatmentDecisionTree.predict_probabilities
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        else:
            missing = [c for c in self.features if c not in X.columns]
            if missing:
                raise ValueError(f"Input X is missing training feature columns: {missing}")

        # Aggregate positive probabilities from trees
        per_tree = []
        for tree, feat_subset in zip(self.trees, self.tree_features):
            probs = tree.predict_probabilities(
                X[feat_subset],
                result_type="ndarray",
            )  # shape: (n_samples, n_treatments) for positive target
            per_tree.append(probs)

        stack = np.stack(per_tree, axis=0)      # (n_trees, n_samples, n_treatments)
        mean_pos = np.mean(stack, axis=0)       # (n_samples, n_treatments)

        # Build DataFrame in canonical order
        pos_cols = [join_jt(self.positive_target, t) for t in self.treatment_modalities]
        pos_df = pd.DataFrame(mean_pos, index=X.index, columns=pos_cols)

        if include_negative_probabilities:
            neg_cols = [join_jt(self.negative_target, t) for t in self.treatment_modalities]
            neg_df = pd.DataFrame(1.0 - mean_pos, index=X.index, columns=neg_cols)
            result_df = pd.concat([neg_df, pos_df], axis=1)  # P(Y=0) before P(Y=1)
        else:
            result_df = pos_df

        if result_type == "df":
            return result_df
        if result_type == "ndarray":
            return result_df.to_numpy()
        if result_type == "lists":
            return result_df.to_numpy().tolist()

        raise ValueError(f"invalid result type {result_type!r}")
