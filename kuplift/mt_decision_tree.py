# kuplift/mt_decision_tree.py
# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Optional, Literal
import logging
import warnings
from pathlib import Path
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from kuplift.mt_tree import Tree
from kuplift.mt_node import Node, IncomingSplit
from kuplift.mt_decision_binary_tree_cost import DecisionBinaryTreeCost
from kuplift.mt_decision_tree_node_split import DecisionTreeNodeSplit
from kuplift.mt_leaf_selection_strategies import (
    select_leaf,
    validate_leaf_selection_strategy,
)
from kuplift.mt_encoding_selector import select_univariate_encoder
from kuplift.utils import transform_variable, join_jt

import khiops.core  # For warning handling

logger = logging.getLogger(__name__)


class DecisionTree:
    """
    Local-partition version (base implementation):
      - for each candidate split at a node, partitions are fitted on node raw dataset
      - raw node datasets are preserved
      - no silent fallback

    Local fitting modes
    -------------------
    - per_leaf (default):
        one local fit_transform() per leaf on all raw features, then reuse partitions
        for each variable of that leaf.
    - per_variable:
        one local fit_transform() per (leaf, variable) on a single-column dataset.
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
        local_fit_mode: str = "per_leaf"
    ):
        validate_leaf_selection_strategy(leaf_selection)

        if local_fit_mode not in {"per_leaf", "per_variable"}:
            raise ValueError("local_fit_mode must be 'per_leaf' or 'per_variable'")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.leaf_selection = leaf_selection
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.maxparts = 2
        self.maxtreatmentgroups = maxtreatmentgroups
        self.control_name = control_name

        self.local_fit_mode = local_fit_mode

        self.cost_model = cost_model if cost_model is not None else DecisionBinaryTreeCost()

        # selector info (OUE/MTUE family)
        self.encoder_type = None

        self.tree: Tree | None = None
        self.tree_criterion: float = 0.0

        self.features: list[str] = []
        self.treatment_col_name: str | None = None
        self.target_col_name: str | None = None

        # Cache for local partition fits:
        # per_variable key = ("var", node_id, variable_name) -> dict result for one var
        # per_leaf key     = ("leaf", node_id)              -> dict[var] = dict result
        self._local_fit_cache: dict[tuple, dict] = {}

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

    def fit(self, data: pd.DataFrame, treatment_col, y_col, positive_target = None) -> "DecisionTree":
        if data is None or len(data) == 0:
            raise ValueError("data must be a non-empty DataFrame")

        # reset local fit cache for this training run
        self._local_fit_cache = {}

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

        _, encoding_info = select_univariate_encoder(t)
        self.encoder_type = encoding_info.encoder_name

        raw_train_df = X.copy()
        raw_train_df[self.treatment_col_name] = t.values
        raw_train_df[self.target_col_name] = y.values


        self.positive_target = self._autodetect_positive_target(y) if positive_target is None else positive_target

        self.tree = Tree(
            dataset=raw_train_df,
            features=self.features,
            treatment_col_name=self.treatment_col_name,
            target_col_name=self.target_col_name,
            positive_target = self.positive_target
        )

        self.negative_target = self.tree.negative_target

        self.cost_model.initialize(self)
        self.tree_criterion = float(self.cost_model.compute_null_tree_cost(self.tree))
        self._grow_tree()
        self.tree_criterion = float(self.cost_model.compute_total_tree_cost(self.tree))
        return self

    def _new_local_encoder(self, t_series: pd.Series):
        encoder, info = select_univariate_encoder(t_series)
        return encoder, info.encoder_name

    def _fit_local_encoder(self, local_X: pd.DataFrame, local_t: pd.Series, local_y: pd.Series):
        """
        Fit local encoder on provided local_X and return (encoder, encoder_name, X_enc).
        """
        encoder, enc_name = self._new_local_encoder(local_t)

        if enc_name == "OUE":
            X_enc = encoder.fit_transform(local_X, local_t, local_y, maxparts=self.maxparts)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "".join([
                        r"^",
                        re.escape(
                            r"""Khiops ended correctly but there were minor issues:""" "\n"
                            r"""Warnings in log:""" "\n"
                        ),
                        r""".*The target variable [^ ]+ contains only one value""",
                        r"$"
                    ]),
                    UserWarning,
                    "^" + khiops.core.internals.runner.__name__ + "$"
                )
                if self.maxtreatmentgroups is not None:
                    X_enc = encoder.fit_transform(
                        local_X,
                        local_t,
                        local_y,
                        maxparts=self.maxparts,
                        maxtreatmentgroups=self.maxtreatmentgroups,
                    )
                else:
                    X_enc = encoder.fit_transform(local_X, local_t, local_y, maxparts=self.maxparts)

        if not isinstance(X_enc, pd.DataFrame):
            raise RuntimeError("local encoder.fit_transform must return a DataFrame")

        return encoder, enc_name, X_enc

    def _fit_local_partitions_for_leaf(self, node: Node) -> dict[str, dict]:
        """
        Per-leaf mode:
        one local fit on all raw features of this node, then build per-variable info map.

        Important:
        - both success and failure are cached per leaf to avoid repeated retries.
        """
        cache_key = ("leaf", int(node.id))
        cached = self._local_fit_cache.get(cache_key)
        if cached is not None:
            return cached

        missing = [v for v in self.features if v not in node.dataset.columns]
        if missing:
            raise RuntimeError(f"Missing variables in node dataset: {missing}")

        local_X = node.dataset[self.features].copy()

        # force categorical dtype once locally to avoid repeated OUE warning
        local_t = node.dataset[self.treatment_col_name].astype(object)
        local_y = node.dataset[self.target_col_name]

        # default non-informative map for all features (used for failure fallback too)
        result_map: dict[str, dict] = {v: {"is_informative": False} for v in self.features}

        try:
            encoder, enc_name, X_enc = self._fit_local_encoder(local_X, local_t, local_y)
        except Exception as e:
            # Cache failure sentinel to avoid refitting this leaf for each variable.
            self._local_fit_cache[cache_key] = result_map
            raise RuntimeError(f"could not fit: {e}")

        informative = list(X_enc.columns)

        if len(informative) == 0:
            self._local_fit_cache[cache_key] = result_map
            return result_map

        if not hasattr(encoder, "get_partition"):
            # cache as non-informative to avoid repeated failures
            self._local_fit_cache[cache_key] = result_map
            raise RuntimeError("local encoder missing get_partition()")
        if not hasattr(encoder, "get_variable_type"):
            # cache as non-informative to avoid repeated failures
            self._local_fit_cache[cache_key] = result_map
            raise RuntimeError("local encoder missing get_variable_type()")
        if not hasattr(encoder, "get_treatment_groups"):
            # strict requirement (no fallback)
            self._local_fit_cache[cache_key] = result_map
            raise RuntimeError("local encoder missing get_treatment_groups()")

        treatment_groups_full = encoder.get_treatment_groups()

        for var in informative:
            if var not in self.features:
                # safety: ignore unexpected columns
                continue

            parts = encoder.get_partition(var)
            vtype = encoder.get_variable_type(var)

            result_map[var] = {
                "is_informative": True,
                "source_var": var,
                "source_type": vtype,
                "parts": parts,
                "encoder_type": enc_name,
                "treatment_groups_full": treatment_groups_full,
            }

        self._local_fit_cache[cache_key] = result_map
        return result_map

    def _fit_local_partition_for_var(self, node: Node, var: str) -> dict:
        """
        Return local partition info for a (node, var), depending on local_fit_mode.
        """
        if var not in self.features:
            raise RuntimeError(f"Unknown training feature {var!r}")

        if self.local_fit_mode == "per_leaf":
            per_leaf_map = self._fit_local_partitions_for_leaf(node)
            return per_leaf_map.get(var, {"is_informative": False})

        # per_variable mode (legacy behavior)
        cache_key = ("var", int(node.id), str(var))
        cached = self._local_fit_cache.get(cache_key)
        if cached is not None:
            return cached

        if var not in node.dataset.columns:
            raise RuntimeError(f"Variable {var!r} missing in node dataset")

        local_X = node.dataset[[var]].copy()

        # force categorical dtype once locally to avoid repeated OUE warning
        local_t = node.dataset[self.treatment_col_name].astype(object)
        local_y = node.dataset[self.target_col_name]

        try:
            encoder, enc_name, X_enc = self._fit_local_encoder(local_X, local_t, local_y)
        except Exception as e:
            result = {"is_informative": False}
            self._local_fit_cache[cache_key] = result
            raise RuntimeError(f"could not fit: {e}")

        informative = list(X_enc.columns)
        if len(informative) == 0:
            result = {"is_informative": False}
            self._local_fit_cache[cache_key] = result
            return result

        if var not in informative:
            result = {"is_informative": False}
            self._local_fit_cache[cache_key] = result
            return result

        if not hasattr(encoder, "get_partition"):
            result = {"is_informative": False}
            self._local_fit_cache[cache_key] = result
            raise RuntimeError("local encoder missing get_partition()")
        if not hasattr(encoder, "get_variable_type"):
            result = {"is_informative": False}
            self._local_fit_cache[cache_key] = result
            raise RuntimeError("local encoder missing get_variable_type()")
        if not hasattr(encoder, "get_treatment_groups"):
            result = {"is_informative": False}
            self._local_fit_cache[cache_key] = result
            raise RuntimeError("local encoder missing get_treatment_groups()")

        parts = encoder.get_partition(var)
        vtype = encoder.get_variable_type(var)
        treatment_groups_full = encoder.get_treatment_groups()

        result = {
            "is_informative": True,
            "source_var": var,
            "source_type": vtype,
            "parts": parts,
            "encoder_type": enc_name,
            "treatment_groups_full": treatment_groups_full,
        }

        self._local_fit_cache[cache_key] = result
        return result

    @staticmethod
    def _normalize_treatment_groups(groups) -> list[tuple]:
        """
        Normalize encoder-provided groups to list[tuple[...]].
        """
        normalized = []
        if groups is None:
            return normalized
        for g in groups:
            normalized.append(tuple(g))
        return normalized

    def _extract_groups_for_var_part(self, treatment_groups_full: dict, var: str, part):
        """
        Extract groups for (var, part) from full dictionary returned by get_treatment_groups().
        """
        if not treatment_groups_full:
            return None
        if var not in treatment_groups_full:
            return None

        groups_by_part = treatment_groups_full[var]
        if part not in groups_by_part:
            return None

        return self._normalize_treatment_groups(groups_by_part[part])

    def _simulate_split_on_var_local(self, node: Node, var: str):
        try:
            info = self._fit_local_partition_for_var(node, var)
        except RuntimeError as e:
            logger.debug(
                "Local fit failed for node_id=%s, var=%s: %s",
                node.id, var, str(e)
            )
            return None

        if not info["is_informative"]:
            return None

        parts = info["parts"]
        encoded_values = transform_variable(parts, node.dataset[var])
        observed_parts = sorted(list(pd.Series(encoded_values).dropna().unique()))

        if len(observed_parts) <= 1:
            return None
        if len(observed_parts) > 2:
            raise RuntimeError(
                f"Expected <=2 observed encoded parts for local variable {var!r}, got {observed_parts}"
            )

        left_part = observed_parts[0]
        right_part = observed_parts[1]

        left_mask = (encoded_values == left_part)
        right_mask = (encoded_values == right_part)

        left_df = node.dataset[left_mask].copy()
        right_df = node.dataset[right_mask].copy()
        if len(left_df) == 0 or len(right_df) == 0:
            return None

        split_var_type = info["source_type"]
        split_value = 0.5

        left_node = Node(
            dataset=left_df,
            treatment_col_name=node.treatment_col_name,
            target_col_name=node.target_col_name,
            parent=node,
            incoming_split=IncomingSplit(var=var, op="<=", value=split_value),
        )
        right_node = Node(
            dataset=right_df,
            treatment_col_name=node.treatment_col_name,
            target_col_name=node.target_col_name,
            parent=node,
            incoming_split=IncomingSplit(var=var, op=">", value=split_value),
        )

        # Full groups for all vars/parts on this leaf dataset
        treatment_groups_full = info.get("treatment_groups_full", None)
        left_node.treatment_groups_full = treatment_groups_full
        right_node.treatment_groups_full = treatment_groups_full

        # Cost-facing groups for this split var + specific side part
        left_node.treatment_groups = self._extract_groups_for_var_part(
            treatment_groups_full=treatment_groups_full,
            var=var,
            part=parts[left_part],
        )
        right_node.treatment_groups = self._extract_groups_for_var_part(
            treatment_groups_full=treatment_groups_full,
            var=var,
            part=parts[right_part],
        )

        n_values_cat = None
        if split_var_type == "Categorical":
            n_values_cat = int(node.dataset[var].nunique(dropna=False))
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
                    simulated = self._simulate_split_on_var_local(leaf, split_var)
                    if simulated is None:
                        continue

                    left_node, right_node, split_value, split_var_type, source_info, n_values_cat = simulated
                    if left_node.sample_size < self.min_samples_leaf or right_node.sample_size < self.min_samples_leaf:
                        continue

                    node_split = DecisionTreeNodeSplit(
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

            selected_leaf = select_leaf(
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

    def get_node_by_id(self, node_id: int):
        if self.tree is None:
            return None
        return self.tree.get_node_by_id(node_id)

    def get_node_path_str(self, node_id: int, separator: str = " AND ") -> str:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_node_path_str(node_id=node_id, separator=separator)

    def get_treatment_groups_of_leaves(self, sort=None) -> pd.DataFrame:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_treatment_groups_of_leaves(sort)

    def get_target_frequencies(self, sort=None) -> pd.DataFrame:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_target_frequencies(sort)
    
    def get_target_probabilities(self, sort=None) -> pd.DataFrame:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_target_probabilities(sort)
    
    def get_uplift(self, sort=None) -> pd.DataFrame:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_uplift(sort)
    
    def node_ids_sorted_dfs(self) -> pd.Index:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.node_ids_sorted_dfs()

    def leaf_ids_sorted_dfs(self) -> pd.Index:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.leaf_ids_sorted_dfs()
    
    def node_ids_sorted_dfs_from(self, node_ids: pd.Index) -> pd.Index:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.node_ids_sorted_dfs_from(node_ids)
    
    def get_leaf_paths(self, sort=None) -> pd.Series:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        return self.tree.get_leaf_paths(sort)
    
    def predict_best_treatment(self, X: pd.DataFrame) -> pd.Series:
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

            positive = self._autodetect_positive_target(targets) if self.positive_target is None else self.positive_target

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

    def predict_probabilities(self, X: pd.DataFrame, result_type: Literal["df", "ndarray", "lists"] = "ndarray") -> pd.DataFrame:
        if self.tree is None:
            raise RuntimeError("Model is not fitted")
        # Ensure the type is correct, so the caller can pass list of lists and the like.
        if isinstance(X, pd.DataFrame):
            if isinstance(X.columns, pd.RangeIndex):
                # The columns are named using a RangeIndex if the caller passes a DataFrame without explicit column names.
                X = X.rename(pd.Series(self.features), axis="columns")
            elif set(X.columns) != set(self.features):
                raise ValueError("passed column names do not match dataset input variable names")
        else:
            # Not a DataFrame => convert to DataFrame.
            X = pd.DataFrame(X, columns=self.features)
        leaf_ids = pd.Index(self.predict_leaf_id(X))
        leaf_probabilities: pd.DataFrame = self.get_target_probabilities()[[join_jt(self.positive_target, t) for t in self.treatment_modalities]]
        result_dataframe: pd.DataFrame = leaf_probabilities[leaf_probabilities.index.isin(leaf_ids)].sort_index(key=lambda _: leaf_ids)
        match result_type:
            case "df": return result_dataframe
            case "ndarray": return result_dataframe.to_numpy()
            case "lists": return result_dataframe.to_numpy().tolist()
            case invalid: raise ValueError("invalid result type {!r}".format(invalid))
    
    def _autodetect_positive_target(self, targets):
        positive = None
        for cand in [1, "1", True, "True"]:
            if cand in targets:
                positive = cand
                break
        if positive is None:
            positive = targets[-1] if targets else None
        return positive

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self.tree is None:
            return (
                "DecisionTree(unfitted, "
                f"leaf_selection={self.leaf_selection}, "
                f"encoder_type={self.encoder_type}, "
                f"local_fit_mode={self.local_fit_mode})"
            )
        return (
            "DecisionTree("
            f"encoder_type={self.encoder_type}, "
            f"leaf_selection={self.leaf_selection}, "
            f"local_fit_mode={self.local_fit_mode}, "
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
            return "DecisionTree: tree is not fitted."

        lines: list[str] = []

        def _branch_values_for_node(node):
            if node.parent is None:
                return "ROOT"

            parent = node.parent
            left_rule, right_rule = self._split_rule_from_info(parent)

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
            enc_type = node.source_partition_info.get("encoder_type", "NA")
            return f" | split_rule: left [{left_rule}], right [{right_rule}] | local_encoder={enc_type}"

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

        lines.append("Decision tree structure (Local partitions)")
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

    def tree_to_dot(self, max_depth: int | None = None, show_node_stats: bool = True) -> str:
        """
        Export tree to Graphviz DOT format.
        """
        if self.tree is None or self.root_node is None:
            return 'digraph Tree {\n  label="Tree is not fitted";\n}'

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
                return f"id={node.id}\\nleaf\\nn={node.sample_size}\\nT={t_count}, Y={y_count}"

            src = node.source_partition_info
            if src is None:
                raise RuntimeError(f"Missing source_partition_info on node id={node.id}")

            left_rule, right_rule = _split_rule_pair(node)
            enc_type = src.get("encoder_type", "NA")
            return (
                f"id={node.id}\\ninternal\\nn={node.sample_size}"
                f"\\n{src['source_var']} ({src['source_type']})"
                f"\\nlocal encoder: {enc_type}"
                f"\\nleft: {left_rule}"
                f"\\nright: {right_rule}"
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

        lines = [
            "digraph Tree {",
            '  graph [rankdir=TB, fontsize=10, fontname="Helvetica"];',
            '  node  [shape=box, style="rounded,filled", fillcolor="#F8F8F8", color="#666666", fontname="Helvetica"];',
            '  edge  [color="#666666", fontname="Helvetica"];',
        ]

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
                    lines.append(f'  {nk} [label="{label}", shape=ellipse, fillcolor="#E8F5E9"];')
                else:
                    lines.append(f'  {nk} [label="{label}", shape=box, fillcolor="#E3F2FD"];')

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
                        lines.append(f'  {ck} [label="{clabel}", shape=ellipse, fillcolor="#E8F5E9"];')
                    else:
                        lines.append(f'  {ck} [label="{clabel}", shape=box, fillcolor="#E3F2FD"];')
                    visited.add(ck)

                elabel = _escape(_edge_label(node, child))
                lines.append(f'  {nk} -> {ck} [label="{elabel}"];')
                _walk(child, depth + 1)

        _walk(self.root_node, 0)
        lines.append("}")
        return "\n".join(lines)
    
    def tree_to_image(self, dest: Path | str | None = None, img_format: str = "png", *args, **kwargs) -> str:
        try:
            import graphviz
        except ImportError as exc:
            raise RuntimeError("'graphviz' package is required to use this function") from exc
        if dest is None:
            dest = Path.cwd() / "tree_{:%Y%m%d_%H%M%S}.{}".format(datetime.now(timezone.utc), img_format)
        return graphviz.Source(self.tree_to_dot(*args, **kwargs)).render(outfile=dest, format=img_format, cleanup=True)