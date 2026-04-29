# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from typing import Literal, get_args
from dataclasses import dataclass
from collections import deque
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
    left_part: Part | None = None
    right_part: Part | None = None
    group_count: int | None = None
    dataset: pandas.DataFrame | None = None
    encoder: Encoder | None = None
    supernode_part: Part | None = None
    path: list[Part] | None = None
    supernode: "Node" | None = None
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
        self._split_variables: list[str] = []
        self._target_modalities: list[str] = sorted(self._y_col.unique())
        self._treatment_modalities: list[str] = sorted(self._treatment_col.unique())
        self._root_node: Node | None = None
        self._internal_nodes: list[Node] = []
        self._leaf_nodes: list[Node] = []
        self._encoder_class: Encoder = OptimizedUnivariateEncoding if self.treatment_modality_count == 2 else MultiTreatmentUnivariateEncoding

    @property
    def used_variable_count(self) -> int: return len(set(self._split_variables))
    @property
    def target_modalities(self) -> tuple[str]: return tuple(self._target_modalities)
    @property
    def treatment_modalities(self) -> tuple[str]: return tuple(self._treatment_modalities)
    @property
    def root_node(self) -> Node: return self._root_node
    @property
    def internal_nodes(self) -> tuple[Node]: return tuple(self._internal_nodes)
    @property
    def leaf_nodes(self) -> tuple[Node]: return tuple(self._leaf_nodes)
    @property
    def treatment_modality_count(self) -> int: return len(self._treatment_modalities)

    def fit(self) -> None:
        logger.debug("Fitting...")
        # Create the root node.
        self._root_node, root_left_subdataset, root_right_subdataset = self.create_node(self._data.join([self._treatment_col, self._y_col]))
        # Create a deque that will contain all work to do to grow the tree as much as it decreases its cost.
        # It will only act as a simple queue.
        # Initialize the queue with the root node, which may be a leaf or an internal node.
        work_queue: deque[tuple[Node, pandas.DataFrame | None, pandas.DataFrame | None]] = deque([(self._root_node, root_left_subdataset, root_right_subdataset)])
        while work_queue:
            node, left_subdataset, right_subdataset = work_queue.popleft()
            if node.type == "internal":  # The node is internal, meaning it has two subnodes to iterate upon.
                work_queue.append(self.create_node(left_subdataset, node, node.left_part))
                work_queue.append(self.create_node(right_subdataset, node, node.right_part))
        logger.debug("Done fitting.")

    def create_node(self, dataset: pandas.DataFrame, supernode: Node | None = None, supernode_part: Part | None = None) -> tuple[Node, pandas.DataFrame | None, pandas.DataFrame | None]:
        # Create the node object.
        node = Node("leaf", sample_size=len(dataset), supernode_part=supernode_part, path=supernode.path + [supernode_part], supernode=supernode)
        # Add the node to the tree's list of leaves.
        self._leaf_nodes.append(node)
        # Fit the dataset attached to this node.
        node.encoder = self._encoder_class()
        node.encoder.fit(dataset[self._data.columns], dataset[self._treatment_col.name], dataset[self._y_col.name], maxparts=2)
        # Find the variables that decrease the cost of the tree.
        vars_decreasing_the_tree_cost = self._vars_decreasing_the_tree_cost()
        if not vars_decreasing_the_tree_cost:  # No variables can decrease the tree cost doing splits.
            logger.debug("The cost of the tree cannot be decreased any further. Stopping here for node {}.", node.path)
            # Return the node and no subdatasets.
            return node, None, None
        else:  # At least one variable can decrease the tree cost doing splits.
            # Choose a variable to split on among the variables that decrease the cost of the tree.
            node.split_var = self._choose_split_var(vars_decreasing_the_tree_cost)
            # Get the type of the variable.
            node.split_var_type = node.encoder.get_variable_type(node.split_var)
            # Get the parts. There may be only one if no splitting could be performed.
            split_parts = node.encoder.get_partition(node.split_var)
            if len(split_parts) == 1:  # Does not split.
                logger.debug("Could not split any further on variable %r of type %r.", node.split_var, node.split_var_type)
                # Return the node and no subdatasets.
                return node, None, None
            else:  # Does split.
                # Register one more occurrence of the variable as used for splitting.
                self._split_variables.append(node.split_var)
                # Set the left and right parts of the node.
                node.left_part, node.right_part = split_parts
                # As there will be two new leaves attached to it, this node becomes an internal node.
                node.type = "internal"
                # Move the node from the tree's list of leaves to the tree's list of internal nodes.
                self._internal_nodes.append(self._leaf_nodes.pop())
                logger.debug("Splitting on variable %r of type %r gave two parts: %s and %s.", node.split_var, node.split_var_type, node.left_part, node.right_part)
                # node.group_count = TO BE IMPLEMENTED -> clarify what is should be, as each part may have a different number of treatment groups
                # Split the dataset according to the two parts.
                left_subdataset, right_subdataset = split_dataset_of_node(node)
                # Return the created node and the two subdatasets.
                return node, left_subdataset, right_subdataset

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