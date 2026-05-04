# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable
import pandas as pd


_NODE_ID_SEQ = count(1)


@dataclass
class IncomingSplit:
    var: str | None = None
    op: str | None = None   # "<=" or ">"
    value: Any = None


class NodeV3:
    """
    Generic and robust node representation for MultiTreatmentDecisionTreeV3.

    Required public information:
      - type ("internal" | "leaf")
      - dataset (DataFrame)
      - split_var
      - split_var_type ("Numerical" | "Categorical" | None)
      - treatment_groups
      - sample_size
      - __str__()
      - get_path_str()
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        treatment_col_name: str,
        target_col_name: str,
        parent: "NodeV3 | None" = None,
        incoming_split: IncomingSplit | None = None,
    ):
        self.id: int = next(_NODE_ID_SEQ)

        self.parent = parent
        self.incoming_split = incoming_split or IncomingSplit()

        self.dataset: pd.DataFrame = dataset
        self.treatment_col_name = treatment_col_name
        self.target_col_name = target_col_name

        # Tree structure
        self.is_leaf: bool = True
        self.left_node: NodeV3 | None = None
        self.right_node: NodeV3 | None = None

        # Split metadata (for internal nodes)
        self.split_var: str | None = None
        self.split_var_type: str | None = None  # "Numerical" | "Categorical" | None
        self.split_value: Any = None

        # Optional group structure for cost model
        self.treatment_groups: list[tuple[Any, ...]] | None = None

        # Cached contingency counts
        self._counts_by_treatment_target: dict[Any, dict[Any, int]] = {}
        self._compute_counts()

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    def type(self) -> str:
        return "leaf" if self.is_leaf else "internal"

    @property
    def sample_size(self) -> int:
        return int(len(self.dataset)) if self.dataset is not None else 0

    # ------------------------------------------------------------------
    # Counts API (for cost model)
    # ------------------------------------------------------------------

    def _compute_counts(self) -> None:
        self._counts_by_treatment_target = {}
        if self.dataset is None or self.sample_size == 0:
            return

        treatments = self.dataset[self.treatment_col_name]
        targets = self.dataset[self.target_col_name]

        for t, y in zip(treatments, targets):
            if t not in self._counts_by_treatment_target:
                self._counts_by_treatment_target[t] = {}
            self._counts_by_treatment_target[t][y] = (
                self._counts_by_treatment_target[t].get(y, 0) + 1
            )

    def get_treatments(self) -> Iterable[Any]:
        return self._counts_by_treatment_target.keys()

    def get_targets(self) -> Iterable[Any]:
        target_set = set()
        for d in self._counts_by_treatment_target.values():
            target_set.update(d.keys())
        return sorted(target_set)

    def get_count(self, treatment: Any, target: Any) -> int:
        return int(self._counts_by_treatment_target.get(treatment, {}).get(target, 0))

    def get_total_count(self) -> int:
        return self.sample_size

    def get_treatment_groups(self):
        return self.treatment_groups

    # ------------------------------------------------------------------
    # Path string API
    # ------------------------------------------------------------------

    def get_path_str(self, separator: str = " AND ") -> str:
        if self.parent is None:
            return "ROOT"

        parts = []
        node = self
        while node is not None and node.parent is not None:
            inc = node.incoming_split
            if inc.var is not None and inc.op is not None:
                parts.append(f"{inc.var} {inc.op} {inc.value}")
            node = node.parent

        parts.reverse()
        return separator.join(parts) if parts else "ROOT"

    # ------------------------------------------------------------------
    # Split simulation + application
    # ------------------------------------------------------------------

    def infer_split_var_type(self, split_var: str) -> str:
        if self.dataset is None or split_var not in self.dataset.columns:
            return "Numerical"
        series = self.dataset[split_var]
        if pd.api.types.is_numeric_dtype(series):
            return "Numerical"
        return "Categorical"

    def _best_numeric_threshold(self, split_var: str):
        """
        Simple median threshold for robustness and determinism.
        """
        s = self.dataset[split_var]
        if s.isna().all():
            return None
        return float(s.median())

    def _categorical_binary_partition(self, split_var: str):
        """
        Deterministic binary partition:
          left  = first half of sorted unique values
          right = remaining values
        Returns a set for left branch.
        """
        values = list(self.dataset[split_var].dropna().unique())
        if len(values) <= 1:
            return None
        values = sorted(values, key=lambda x: str(x))
        k = max(1, len(values) // 2)
        left_values = set(values[:k])
        if len(left_values) == 0 or len(left_values) == len(values):
            return None
        return left_values

    def simulate_split(self, split_var: str):
        """
        Simulate split without mutating current node.
        Returns:
          (left_node, right_node, split_value, split_var_type)
        or None if split is not feasible.
        """
        if self.dataset is None or split_var not in self.dataset.columns:
            return None
        if self.sample_size <= 1:
            return None

        split_var_type = self.infer_split_var_type(split_var)

        if split_var_type == "Numerical":
            threshold = self._best_numeric_threshold(split_var)
            if threshold is None:
                return None

            left_df = self.dataset[self.dataset[split_var] <= threshold]
            right_df = self.dataset[self.dataset[split_var] > threshold]

            if len(left_df) == 0 or len(right_df) == 0:
                return None

            left_node = NodeV3(
                dataset=left_df.copy(),
                treatment_col_name=self.treatment_col_name,
                target_col_name=self.target_col_name,
                parent=self,
                incoming_split=IncomingSplit(var=split_var, op="<=", value=threshold),
            )
            right_node = NodeV3(
                dataset=right_df.copy(),
                treatment_col_name=self.treatment_col_name,
                target_col_name=self.target_col_name,
                parent=self,
                incoming_split=IncomingSplit(var=split_var, op=">", value=threshold),
            )
            return left_node, right_node, threshold, split_var_type

        # Categorical
        left_values = self._categorical_binary_partition(split_var)
        if left_values is None:
            return None

        mask_left = self.dataset[split_var].isin(left_values)
        left_df = self.dataset[mask_left]
        right_df = self.dataset[~mask_left]

        if len(left_df) == 0 or len(right_df) == 0:
            return None

        # For categorical split_value, store a stable sorted tuple
        split_value = tuple(sorted(left_values, key=lambda x: str(x)))

        left_node = NodeV3(
            dataset=left_df.copy(),
            treatment_col_name=self.treatment_col_name,
            target_col_name=self.target_col_name,
            parent=self,
            incoming_split=IncomingSplit(var=split_var, op="<=", value=split_value),
        )
        right_node = NodeV3(
            dataset=right_df.copy(),
            treatment_col_name=self.treatment_col_name,
            target_col_name=self.target_col_name,
            parent=self,
            incoming_split=IncomingSplit(var=split_var, op=">", value=split_value),
        )
        return left_node, right_node, split_value, split_var_type

    def apply_split(
        self,
        split_var: str,
        left_node: "NodeV3",
        right_node: "NodeV3",
        split_value: Any,
        split_var_type: str,
    ) -> None:
        """
        Mutate current node into internal node with provided children.
        """
        self.is_leaf = False
        self.split_var = split_var
        self.split_var_type = split_var_type
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"NodeV3(id={self.id}, type={self.type}, sample_size={self.sample_size}, "
            f"split_var={self.split_var!r}, split_var_type={self.split_var_type!r}, "
            f"path='{self.get_path_str()}')"
        )

    def __repr__(self) -> str:
        return self.__str__()
