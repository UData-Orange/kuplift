# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from typing import Literal, get_args
from dataclasses import dataclass
from random import choice, choices
import logging
import pandas
from .typealiases import VarType, Part
from .optimized_univariate_encoding import OptimizedUnivariateEncoding
from .multi_treatment_univariate_encoding import MultiTreatmentUnivariateEncoding


logger = logging.getLogger(__name__)


class MultiTreatmentDecisionTree:
    def fit(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series) -> None:
        raise NotImplementedError


Encoder = OptimizedUnivariateEncoding | MultiTreatmentUnivariateEncoding

NodeType = Literal["internal", "leaf"]

@dataclass
class Node:
    type: NodeType
    sample_size: int
    split_var: str | None = None
    split_var_type: VarType | None = None
    split_parts: tuple[Part, Part] | None = None
    group_count: int | None = None
    dataset: pandas.DataFrame | None = None
    encoder: Encoder | None = None
    left_subnode: "Node" | None = None
    right_subnode: "Node" | None = None


SplitVarChoiceAlgorithm = Literal["best", "random", "random_weighted"]

class Tree:
    def __init__(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series, split_var_choice_algorithm: SplitVarChoiceAlgorithm = "best") -> None:
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
        self._encoder_class: Encoder = OptimizedUnivariateEncoding if self.treatment_modality_count == 2 else MultiTreatmentUnivariateEncoding

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
        raise NotImplementedError

    def create_node(self, dataset: pandas.DataFrame) -> None:
        node = Node("leaf", sample_size=len(dataset))
        # Fit the dataset attached to this node.
        node.encoder = self._encoder_class()
        node.encoder.fit(dataset[self._data.columns], dataset[self._treatment_col.name], dataset[self._y_col.name], maxparts=2)
        # Find the variables that decrease the cost of the tree.
        vars_decreasing_the_tree_cost = self._vars_decreasing_the_tree_cost()
        if not vars_decreasing_the_tree_cost:
            # The cost of the tree cannot be decreased any further.
            raise NotImplementedError
        else:
            # Choose a variable to split on among the variables that decrease the cost of the tree.
            node.split_var = self._choose_split_var(vars_decreasing_the_tree_cost)
            # Get the type of the variable.
            node.split_var_type = node.encoder.get_variable_type(node.split_var)
            # Get the parts. There may be only one if no splitting could be performed.
            node.split_parts = tuple(node.encoder.get_partition(node.split_var))
            if len(node.split_parts) == 1:
                logger.debug("Could not split any further on variable %r of type %r.", node.split_var, node.split_var_type)
            else:
                left_part, right_part = node.split_parts
                node.type = "internal"
                logger.debug("Splitting on variable %r of type %r gave two parts: %s and %s.", node.split_var, node.split_var_type, left_part, right_part)
                # node.group_count = TO BE IMPLEMENTED -> clarify what is should be, as each part may have a different number of treatment groups
                # Split the dataset according to the two parts.
                left_subdataset, right_subdataset = split_dataset_of_node(node)

    def _choose_split_var(self, vars_to_choose_from: list[str]) -> str:
        match self._split_var_choice_algorithm:
            case "best":
                return next(name for name, _ in self._encoder.get_levels() if name in vars_to_choose_from)
            case "random":
                return choice(vars_to_choose_from)
            case "random_weighted":
                return choices(vars_to_choose_from, (level for name, level in self._encoder.get_levels() if name in vars_to_choose_from and level != 0))
            
    def _vars_decreasing_the_tree_cost(self) -> list[str]:
        import warnings
        warnings.warn("TODO: Implement proper algorithm.")
        return self._encoder.informative_input_variables
    
def split_dataset_of_node(node: Node) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    transformed_data_of_split_var = node.encoder.transform(node.dataset)[node.split_var]
    return (node.dataset[transformed_data_of_split_var == 0], node.dataset[transformed_data_of_split_var == 1])