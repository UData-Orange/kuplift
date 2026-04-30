# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from typing import Literal, Optional, get_args
from dataclasses import dataclass
from collections import deque
from random import choice, choices
import logging
import pandas
from .typealiases import VarType, Part
from .optimized_univariate_encoding import OptimizedUnivariateEncoding
from .multi_treatment_univariate_encoding import MultiTreatmentUnivariateEncoding

logger = logging.getLogger(__name__)

SplitVarChoiceAlgorithm = Literal["best", "random", "random_weighted"]

class MultiTreatmentDecisionTree:
    def __init__(self) -> None:
        self._tree: Tree | None = None

    def fit(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series, split_var_choice_algorithm: SplitVarChoiceAlgorithm = "best") -> None:
        self._tree = Tree(data, treatment_col, y_col, split_var_choice_algorithm)
        self._tree.fit()

    def __str__(self) -> str:
        return str(self._tree)

@dataclass
class Node:
    type: Literal["internal", "leaf"]
    dataset: pandas.DataFrame
    split_var: str | None = None
    split_var_type: VarType | None = None
    left_part: Part | None = None
    right_part: Part | None = None
    group_count: int | None = None
    supernode: Optional["Node"] = None
    supernode_part: Part | None = None
    path: list[Part] | None = None
    left_subnode: Optional["Node"] = None
    right_subnode: Optional["Node"] = None
    @property
    def sample_size(self) -> int:
        return len(self.dataset)
    def format_path(self) -> str:
        return " . ".join(map(str, self.path))

NodeCreationResult = tuple[Node, pandas.DataFrame | None, pandas.DataFrame | None]

class Tree:
    def __init__(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series, split_var_choice_algorithm: SplitVarChoiceAlgorithm = "best") -> None:
        if split_var_choice_algorithm not in get_args(SplitVarChoiceAlgorithm):
            raise ValueError("algorithm {!r} is not supported".format(split_var_choice_algorithm))
        self._data: pandas.DataFrame = data
        self._treatment_col: pandas.Series = treatment_col
        self._y_col: pandas.Series = y_col
        self._split_var_choice_algorithm: SplitVarChoiceAlgorithm = split_var_choice_algorithm
        self._split_vars: list[str] = []
        self._target_modalities: list[str] = sorted(self._y_col.unique())
        self._treatment_modalities: list[str] = sorted(self._treatment_col.unique())
        self._root_node: Node | None = None
        self._internal_nodes: list[Node] = []
        self._leaf_nodes: list[Node] = []
        self._encoder: OptimizedUnivariateEncoding | MultiTreatmentUnivariateEncoding = OptimizedUnivariateEncoding() if self.treatment_modality_count == 2 else MultiTreatmentUnivariateEncoding()

    @property
    def used_variable_count(self) -> int: return len(set(self._split_vars))
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
        root_result = self.create_node(self._data.join([self._treatment_col, self._y_col]))
        self._root_node, _, _ = root_result
        # Create an empty deque that will contain all work to do to grow the tree as much as it decreases its cost.
        # It will only act as a simple queue.
        internal_nodes_to_work_on: deque[NodeCreationResult] = deque()
        # Add the root node to the queue if it successfully split, that is, if it is an internal node.
        if self._root_node.type == "internal":
            internal_nodes_to_work_on.append(root_result)
        # Extract internal nodes from the queue until it is empty.
        while internal_nodes_to_work_on:
            node, left_subdataset, right_subdataset = internal_nodes_to_work_on.popleft()
            left_result = self.create_node(left_subdataset, node, node.left_part)
            node.left_subnode, _, _ = left_result
            if node.left_subnode.type == "internal":
                internal_nodes_to_work_on.append(left_result)
            right_result = self.create_node(right_subdataset, node, node.right_part)
            node.right_subnode, _, _ = right_result
            if node.right_subnode.type == "internal":
                internal_nodes_to_work_on.append(right_result)
        logger.debug("Done fitting.")

    def create_node(self, dataset: pandas.DataFrame, supernode: Node | None = None, supernode_part: Part | None = None) -> NodeCreationResult:
        # Create the node object.
        node = Node("leaf", dataset, supernode=supernode, supernode_part=supernode_part, path=["ROOT"] if supernode is None else supernode.path + [supernode_part])
        # Add the node to the tree's list of leaves.
        self._leaf_nodes.append(node)
        # Fit the dataset attached to this node.
        try:
            self._encoder.fit(dataset[self._data.columns], dataset[self._treatment_col.name], dataset[self._y_col.name], maxparts=2)
        except KeyError as exc:
            if exc.args[0] == "detailed statistics":
                logger.debug("Failed to fit the dataset attached to node %s.", node.format_path())
                return node, None, None
            else:
                raise
        # Find the variables that decrease the cost of the tree.
        vars_decreasing_the_tree_cost = self._vars_decreasing_the_tree_cost()
        if not vars_decreasing_the_tree_cost:  # No variables can decrease the tree cost doing splits.
            logger.debug("The cost of the tree cannot be decreased any further. Stopping here for node %s.", node.format_path())
            # Return the node and no subdatasets.
            return node, None, None
        else:  # At least one variable can decrease the tree cost doing splits.
            # Choose a variable to split on among the variables that decrease the cost of the tree.
            node.split_var = self._choose_split_var(vars_decreasing_the_tree_cost)
            # Get the type of the variable.
            node.split_var_type = self._encoder.get_variable_type(node.split_var)
            # Get the parts. There may be only one if no splitting could be performed.
            split_parts = self._encoder.get_partition(node.split_var)
            if len(split_parts) == 1:  # Does not split.
                logger.debug("Could not split any further on variable %r of type %r.", node.split_var, node.split_var_type)
                # Return the node and no subdatasets.
                return node, None, None
            else:  # Does split.
                # Register one more occurrence of the variable as used for splitting.
                self._split_vars.append(node.split_var)
                # Set the left and right parts of the node.
                node.left_part, node.right_part = split_parts
                # As there will be two new leaves attached to it, this node becomes an internal node.
                node.type = "internal"
                # Move the node from the tree's list of leaves to the tree's list of internal nodes.
                self._internal_nodes.append(self._leaf_nodes.pop())
                logger.debug("Splitting on variable %r of type %r gave two parts: %s and %s.", node.split_var, node.split_var_type, node.left_part, node.right_part)
                # node.group_count = TO BE IMPLEMENTED -> clarify what is should be, as each part may have a different number of treatment groups
                # Split the dataset according to the two parts.
                left_subdataset, right_subdataset = self._split_dataset_of_node(node)
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
        variables = list(set(self._encoder.informative_input_variables) - set(self._split_vars))
        logger.debug("Variables decreasing the tree cost: %s", variables)
        return variables
    
    def _split_dataset_of_node(self, node: Node) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        transformed_data_of_split_var = self._encoder.transform(node.dataset)[node.split_var]
        return (node.dataset[transformed_data_of_split_var == 0], node.dataset[transformed_data_of_split_var == 1])
    
    def __str__(self) -> str:
        if not self._root_node:
            return "Empty tree because fit() has not been called."
        result_lines = ["Tree:"]
        work_queue: deque[Node] = deque([self._root_node])
        while work_queue:
            node = work_queue.popleft()
            result_lines.append(node.format_path())
            if node.type == "internal":
                work_queue.append(node.left_subnode)
                work_queue.append(node.right_subnode)
        return "\n".join(result_lines)