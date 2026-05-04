# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations


class DTDecisionTreeNodeSplitV3:
    """
    Container for a hypothetical split evaluation in MultiTreatmentDecisionTreeV3.

    It stores:
      - the splittable node
      - the split variable name
      - the hypothetical full-tree cost if this split is applied
    """

    def __init__(self, splittable_node=None, split_var: str | None = None):
        self._splittable_node = splittable_node
        self._split_var = split_var
        self._tree_cost: float = 0.0

    # ------------------------------------------------------------------
    # Splittable node
    # ------------------------------------------------------------------

    def get_splittable_node(self):
        return self._splittable_node

    def set_splittable_node(self, node) -> None:
        self._splittable_node = node

    # ------------------------------------------------------------------
    # Split variable
    # ------------------------------------------------------------------

    def get_split_var(self) -> str | None:
        return self._split_var

    def set_split_var(self, split_var: str | None) -> None:
        self._split_var = split_var

    # ------------------------------------------------------------------
    # Tree cost
    # ------------------------------------------------------------------

    def get_tree_cost(self) -> float:
        return self._tree_cost

    def set_tree_cost(self, cost: float) -> None:
        self._tree_cost = float(cost)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        node_id = getattr(self._splittable_node, "id", None)
        return (
            f"DTDecisionTreeNodeSplitV3("
            f"node_id={node_id}, split_var={self._split_var!r}, tree_cost={self._tree_cost:.6f})"
        )

    def __repr__(self) -> str:
        return self.__str__()
