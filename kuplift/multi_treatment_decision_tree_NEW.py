# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from typing import Literal, get_args
from dataclasses import dataclass
from random import choice, choices
import pandas
from .typealiases import VarType, Part
from .optimized_univariate_encoding import OptimizedUnivariateEncoding
from .multi_treatment_univariate_encoding import MultiTreatmentUnivariateEncoding


class MultiTreatmentDecisionTree:
    def fit(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series) -> None:
        raise NotImplementedError


NodeType = Literal["internal", "leaf"]

@dataclass
class Node:
    type: NodeType
    split_var: str
    split_var_type: VarType
    split_parts: tuple[Part, Part]
    sample_size: int
    group_count: int
    nijt: pandas.DataFrame
    children: list["Node"]


SplitVarChoiceAlgorithm = Literal["best", "random", "random_weighted"]

class Tree:
    def __init__(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series, split_var_choice_algorithm: SplitVarChoiceAlgorithm) -> None:
        if split_var_choice_algorithm not in get_args(SplitVarChoiceAlgorithm):
            raise ValueError("algorithm {!r} is not supported".format(split_var_choice_algorithm))
        self._data: pandas.DataFrame = data
        self._treatment_col: pandas.Series = treatment_col
        self._y_col: pandas.Series = y_col
        self._split_var_choice_algorithm: SplitVarChoiceAlgorithm = split_var_choice_algorithm
        self._used_variable_count: int = 0
        self._target_modalities: list[str] = sorted(self._y_col.unique())
        self._treatment_modalities: list[str] = sorted(self._treatment_col.unique())
        self._internal_nodes: list[Node] = []
        self._leaf_nodes: list[Node] = []
        self._encoder: OptimizedUnivariateEncoding | MultiTreatmentUnivariateEncoding = OptimizedUnivariateEncoding() if self.treatment_modality_count == 2 else MultiTreatmentUnivariateEncoding()

    @property
    def used_variable_count(self) -> int: return self._used_variable_count
    @property
    def target_modalities(self) -> tuple[str]: return tuple(self._target_modalities)
    @property
    def treatment_modalities(self) -> tuple[str]: return tuple(self._treatment_modalities)
    @property
    def internal_nodes(self) -> tuple[Node]: return tuple(self._internal_nodes)
    @property
    def leaf_nodes(self) -> tuple[Node]: return tuple(self._leaf_nodes)
    @property
    def treatment_modality_count(self) -> int: return len(self._treatment_modalities)

    def fit(self) -> None:
        self._encoder.fit(self._data, self._treatment_col, self._y_col, maxparts=2)
        split_var = self._choose_split_var(vars_decreasing_the_cost())
        split_var_frequencies = self._encoder.get_target_frequencies(split_var)
        root_node = Node(
            type="leaf",
            split_var=split_var,
            split_var_type=self._encoder.get_variable_type(split_var),
            split_parts=tuple(split_var_frequencies.index),
            sample_size=len(self._data),
            group_count=len(self._encoder.get_treatment_groups(split_var)),
            nijt=split_var_frequencies,
            children=[]
        )
        self._internal_nodes.append(root_node)

    def _choose_split_var(self, vars_to_choose_from: list[str]) -> str:
        match self._split_var_choice_algorithm:
            case "best":
                return next(name for name, _ in self._encoder.get_levels() if name in vars_to_choose_from)
            case "random":
                return choice(vars_to_choose_from)
            case "random_weighted":
                return choices(vars_to_choose_from, (level for name, level in self._encoder.get_levels() if name in vars_to_choose_from and level != 0))