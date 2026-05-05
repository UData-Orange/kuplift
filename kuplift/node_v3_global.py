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


class NodeV3Global:
    """
    Node for global-partition tree:
      - datasets are always raw values
      - split mask comes from encoded part indices
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        treatment_col_name: str,
        target_col_name: str,
        parent: "NodeV3Global | None" = None,
        incoming_split: IncomingSplit | None = None,
    ):
        self.id: int = next(_NODE_ID_SEQ)

        self.parent = parent
        self.incoming_split = incoming_split or IncomingSplit()

        self.dataset = dataset
        self.treatment_col_name = treatment_col_name
        self.target_col_name = target_col_name

        self.is_leaf = True
        self.left_node: NodeV3Global | None = None
        self.right_node: NodeV3Global | None = None

        self.split_var: str | None = None                 # encoded variable name
        self.split_var_type: str | None = None            # "Numerical" / "Categorical"
        self.split_value: Any = None                      # threshold or left-set of encoded part indices

        self.n_values_of_categorical_split_var: int | None = None
        self.treatment_groups: list[tuple[Any, ...]] | None = None

        # source partition metadata for display/debug
        self.source_partition_info: dict | None = None

        self._counts_by_treatment_target: dict[Any, dict[Any, int]] = {}
        self._compute_counts()

    @property
    def type(self) -> str:
        return "leaf" if self.is_leaf else "internal"

    @property
    def sample_size(self) -> int:
        return int(len(self.dataset)) if self.dataset is not None else 0

    def _compute_counts(self) -> None:
        self._counts_by_treatment_target = {}
        if self.dataset is None or self.sample_size == 0:
            return
        treatments = self.dataset[self.treatment_col_name]
        targets = self.dataset[self.target_col_name]
        for t, y in zip(treatments, targets):
            self._counts_by_treatment_target.setdefault(t, {})
            self._counts_by_treatment_target[t][y] = self._counts_by_treatment_target[t].get(y, 0) + 1

    def get_treatments(self) -> Iterable[Any]:
        return self._counts_by_treatment_target.keys()

    def get_targets(self) -> Iterable[Any]:
        s = set()
        for d in self._counts_by_treatment_target.values():
            s.update(d.keys())
        return sorted(s, key=lambda x: str(x))

    def get_count(self, treatment: Any, target: Any) -> int:
        return int(self._counts_by_treatment_target.get(treatment, {}).get(target, 0))

    def get_total_count(self) -> int:
        return self.sample_size

    def get_treatment_groups(self):
        return self.treatment_groups

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

    def apply_split(
        self,
        split_var: str,
        left_node: "NodeV3Global",
        right_node: "NodeV3Global",
        split_value: Any,
        split_var_type: str,
        source_partition_info: dict | None,
        n_values_of_categorical_split_var: int | None,
    ) -> None:
        self.is_leaf = False
        self.split_var = split_var
        self.split_var_type = split_var_type
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        self.source_partition_info = source_partition_info
        self.n_values_of_categorical_split_var = n_values_of_categorical_split_var

    def __str__(self) -> str:
        return (
            f"NodeV3Global(id={self.id}, type={self.type}, sample_size={self.sample_size}, "
            f"split_var={self.split_var!r}, split_var_type={self.split_var_type!r}, "
            f"path='{self.get_path_str()}')"
        )

    def __repr__(self) -> str:
        return self.__str__()
