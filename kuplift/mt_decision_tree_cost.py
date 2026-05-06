# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from abc import ABC, abstractmethod


class DecisionTreeCost(ABC):
    """
    Abstract cost-model interface for multi-treatment DecisionTree.

    A concrete implementation must provide:
      - null tree cost
      - total tree cost
      - hypothetical augmented-tree cost for one candidate split
    """

    def __init__(self):
        self.total_attribute_number: int = 0
        self.class_value_number: int = 0

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def initialize(self, tree) -> None:
        """
        Initialize cost-model metadata from the current tree.
        """
        features = getattr(tree, "features", None)
        self.total_attribute_number = len(features) if features is not None else 0

        target_modalities = getattr(tree, "target_modalities", None)
        self.class_value_number = (
            len(target_modalities) if target_modalities is not None else 0
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_null_tree_cost(self, tree) -> float:
        """
        Cost of the initial tree (root-only).
        """

    @abstractmethod
    def compute_total_tree_cost(self, tree) -> float:
        """
        Total cost of the current tree.
        """

    @abstractmethod
    def compute_hypothetical_augmented_tree_cost(
        self, tree, previous_cost: float, node_split
    ) -> float:
        """
        Cost of tree after applying a hypothetical split.
        """

    # ------------------------------------------------------------------
    # Optional API
    # ------------------------------------------------------------------

    def compute_hypothetical_pruned_tree_cost(
        self, tree, previous_cost: float, internal_node
    ) -> float:
        """
        Optional pruning-cost API. Default: not implemented.
        """
        raise NotImplementedError(
            "Pruning cost is not implemented for this cost model."
        )
