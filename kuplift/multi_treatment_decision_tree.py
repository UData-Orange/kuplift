# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from typing import Literal, Optional, get_args
from dataclasses import dataclass
from collections import deque
from random import choice, choices
from itertools import count
import logging
import pandas
from .typealiases import VarType, Part
from .treatment_grouping import TreatmentGroups
from .optimized_univariate_encoding import OptimizedUnivariateEncoding
from .multi_treatment_univariate_encoding import MultiTreatmentUnivariateEncoding

logger = logging.getLogger(__name__)

SplitLeafChoiceAlgorithm = Literal["best", "random"]
SplitVarChoiceAlgorithm = Literal["best", "random", "random_weighted"]

NodeCreationResult = tuple["Node", pandas.DataFrame | None, pandas.DataFrame | None]


@dataclass
class VariableWithTreeCost:
    variable: str
    tree_cost: float


@dataclass
class Node:
    type: Literal["internal", "leaf"]
    dataset: pandas.DataFrame
    encoder: OptimizedUnivariateEncoding | MultiTreatmentUnivariateEncoding
    path: list[tuple[str, Part]]
    split_var: str | None = None
    split_var_type: VarType | None = None
    left_part: Part | None = None
    right_part: Part | None = None
    groups: TreatmentGroups | None = None
    supernode: Optional["Node"] = None
    supernode_part: Part | None = None
    left_subnode: Optional["Node"] = None
    right_subnode: Optional["Node"] = None
    _fit_performed: bool = False
    
    @property
    def sample_size(self) -> int:
        return len(self.dataset)
    
    def __str__(self) -> str:
        return "Node: " + " . ".join(["ROOT", *("{}:{}".format(split_var, part) for split_var, part in self.path)])
    
    def __hash__(self) -> int:
        return id(self)
    
    @classmethod
    def create_root(cls, dataset: pandas.DataFrame, encoder_class: "type") -> "Node":
        return cls("leaf", dataset, encoder_class(), path = [])
    
    def fit(self) -> None:
        if not self._fit_performed:
            *x_names, t_name, y_name = self.dataset.columns
            self.encoder.fit(self.dataset[x_names], self.dataset[t_name], self.dataset[y_name], maxparts=2)
            self._fit_performed = True
        else:
            logger.debug("Fitting already performed, not computing again.")

    def get_splitting_variables(self) -> list[str]:
        return [splitvar for splitvar, parts in self.encoder.get_partitions().items() if len(parts) == 2]

    def split(self, variable: str) -> tuple["Node", "Node"]:
        self.type = "internal"
        self.split_var = variable
        self.split_var_type = self.encoder.get_variable_type(variable)
        self.left_part, self.right_part = self.encoder.get_partition(variable)
        self.groups = self.encoder.get_treatment_groups(variable)
        leaves = self._add_leaves()
        logger.debug("Splitting at '%s' on %r, giving two parts %s and %s with respectively %s and %s observations...", self, variable, self.left_part, self.right_part, len(leaves[0].dataset), len(leaves[1].dataset))
        return leaves
    
    def _add_leaves(self) -> tuple["Node", "Node"]:
        left_subdataset, right_subdataset = self._split_dataset()
        self.left_subnode = self.__class__._create_leaf(left_subdataset, self.encoder.__class__, self, self.left_part)
        self.right_subnode = self.__class__._create_leaf(right_subdataset, self.encoder.__class__, self, self.right_part)
        return self.left_subnode, self.right_subnode
    
    def _split_dataset(self) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        transformed_data_of_split_var = self.encoder.transform(self.dataset)[self.split_var]
        return self.dataset[transformed_data_of_split_var == 0], self.dataset[transformed_data_of_split_var == 1]
    
    @classmethod
    def _create_leaf(cls, dataset: pandas.DataFrame, encoder_class: "type", supernode: "Node", supernode_part: Part) -> "Node":
        return cls("leaf", dataset, encoder_class(), supernode=supernode, supernode_part=supernode_part, path=supernode.path + [(supernode.split_var, supernode_part)])


class MultiTreatmentDecisionTree:
    def __init__(self) -> None:
        self._data: pandas.DataFrame | None = None
        self._treatment_col: pandas.Series | None = None
        self._y_col: pandas.Series | None = None
        self._split_leaf_choice_algorithm: SplitLeafChoiceAlgorithm | None = None
        self._split_var_choice_algorithm: SplitVarChoiceAlgorithm | None = None
        self._split_vars: set[str] = set()
        self._target_modalities: list[str] = []
        self._treatment_modalities: list[str] = []
        self._root_node: Node | None = None
        self._internal_nodes: list[Node] = []
        self._leaf_nodes: list[Node] = []
        self._encoder_class: type | None = None
        self._cost: float | None = None

    @property
    def used_variable_count(self) -> int: return len(self._split_vars)
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

    def fit(self, data: pandas.DataFrame, treatment_col: pandas.Series, y_col: pandas.Series, split_leaf_choice_algorithm: SplitLeafChoiceAlgorithm = "best", split_var_choice_algorithm: SplitVarChoiceAlgorithm = "best") -> None:
        if split_leaf_choice_algorithm not in get_args(SplitLeafChoiceAlgorithm):
            raise ValueError("algorithm {!r} is not supported".format(split_leaf_choice_algorithm))
        if split_var_choice_algorithm not in get_args(SplitVarChoiceAlgorithm):
            raise ValueError("algorithm {!r} is not supported".format(split_var_choice_algorithm))
        
        self.__init__()
        self._data = data
        self._treatment_col = treatment_col
        self._y_col = y_col
        self._split_leaf_choice_algorithm = split_leaf_choice_algorithm
        self._split_var_choice_algorithm = split_var_choice_algorithm
        self._target_modalities = sorted(self._y_col.unique())
        self._treatment_modalities = sorted(self._treatment_col.unique())
        self._encoder_class = OptimizedUnivariateEncoding if self.treatment_modality_count == 2 else MultiTreatmentUnivariateEncoding
        # Initialize the total cost of the tree.
        self._cost = 15  # TODO
        
        # Rebuild the dataset table with the columns  | X1 | X2 | ... | Xi | T | Y |
        dataset = self._data.join([self._treatment_col, self._y_col])
        # Create the root node with the entire dataset. It is a leaf for now.
        logger.debug("Creating root node with %s observations...", len(dataset))
        self._root_node = Node.create_root(dataset, self._encoder_class)
        # Add it to the previously empty list of leaves.
        self._leaf_nodes.append(self._root_node)

        # Loop as long as the total cost of the tree can be decreased.
        for i in count(1):
            logger.debug("Iteration %s", i)
            splitting_variables_by_leaf: dict[Node, VariableWithTreeCost] = {}
            # Choose a splitting variable for each leaf.
            for node in self._leaf_nodes:
                # logger.debug("Checking if node %s is a candidate for splitting...", node)
                # Perform the fitting operation on the current leaf node.
                try:
                    logger.debug("Performing fitting operation for node %s...", node)
                    node.fit()
                    logger.debug("Fitting succeeded.")
                except RuntimeError:
                    # Fitting failed => skip to the next leaf.
                    logger.debug("Fitting failed; dataset probably too small.")
                    continue
                # Get the variables that split (number of parts == 2).
                splitting_variables = node.get_splitting_variables()
                # Extract only the variables that decrease the tree cost when split upon.
                variables_decreasing_the_tree_cost: dict[str, float] = {}
                for split_var in splitting_variables:
                    # logger.debug("Checking if variable %r is a splitting candidate that decrease the cost of the tree...", split_var)
                    new_tree_cost = self._compute_new_possible_tree_cost(node, split_var)
                    if new_tree_cost < self._cost:
                        variables_decreasing_the_tree_cost[split_var] = new_tree_cost
                # If there is such a variable, the leaf is a candidate for splitting.
                if variables_decreasing_the_tree_cost:
                    # Choose a variable to split upon, according to the algorithm passed as an argument.
                    chosen_variable = self._choose_split_var(node, list(variables_decreasing_the_tree_cost))
                    # Add the leaf to the candidate collection, with splitting specified for the chosen variable only.
                    splitting_variables_by_leaf[node] = VariableWithTreeCost(chosen_variable, variables_decreasing_the_tree_cost[chosen_variable])
                else:
                    logger.debug("No variable can decrease the cost of the tree.")
            # If there are leaves that can be split, one will be chosen to become an internal node and to spawn two new leaves.
            if splitting_variables_by_leaf:
                # Choose a leaf to split, according to the algorithm passed as an argument.
                chosen_leaf = self._choose_split_leaf(splitting_variables_by_leaf)
                self._split_vars.add(splitting_variables_by_leaf[chosen_leaf].variable)
                # Split.
                new_leaves_to_add = chosen_leaf.split(splitting_variables_by_leaf[chosen_leaf].variable)
                # Move the current node from the list of leaves to the list of internal nodes.
                self._leaf_nodes.remove(chosen_leaf)
                self._internal_nodes.append(chosen_leaf)
                # Add the new leaf nodes to the list of leaves.
                self._leaf_nodes.extend(new_leaves_to_add)
                # Update the total tree cost.
                self._cost = splitting_variables_by_leaf[chosen_leaf].tree_cost
            # Otherwise, no leaf can be split to decrease the total tree cost => exit the loop.
            else:
                break

    def _choose_split_var(self, node: Node, vars_to_choose_from: list[str]) -> str:
        match self._split_var_choice_algorithm:
            case "best":
                return next(name for name, _ in node.encoder.get_levels() if name in vars_to_choose_from)
            case "random":
                return choice(vars_to_choose_from)
            case "random_weighted":
                return choices(vars_to_choose_from, (level for name, level in node.encoder.get_levels() if name in vars_to_choose_from and level != 0))[0]

    def _choose_split_leaf(self, splitting_variables_by_leaf: dict[Node, VariableWithTreeCost]) -> Node:
        match self._split_leaf_choice_algorithm:
            case "best":
                return min(splitting_variables_by_leaf, key=lambda leaf: splitting_variables_by_leaf[leaf].tree_cost)
            case "random":
                return choice(list(splitting_variables_by_leaf))

    def _compute_new_possible_tree_cost(self, split_node: Node, split_var: str) -> float:
        return self._cost - 1 if self._cost >= 0 else self._cost

    def __str__(self) -> str:
        if not self._root_node:
            return "Empty tree because fit() has not been called."
        result_lines = []
        work_queue: deque[Node] = deque([self._root_node])
        while work_queue:
            node = work_queue.popleft()
            result_lines.append(str(node))
            if node.type == "internal":
                work_queue.append(node.left_subnode)
                work_queue.append(node.right_subnode)
        return "\n".join(result_lines)