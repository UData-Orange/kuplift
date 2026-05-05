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
      - simulated split payload (children + split metadata)
      - the hypothetical full-tree cost if this split is applied
    """

    def __init__(self, splittable_node=None, split_var: str | None = None):
        self._splittable_node = splittable_node
        self._split_var = split_var
        self._tree_cost: float = 0.0

        # Simulated split payload (must be explicitly provided by caller)
        self._left_node = None
        self._right_node = None
        self._split_value = None
        self._split_var_type = None
        self._n_values_of_categorical_split_var = None

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
    # Simulated split payload
    # ------------------------------------------------------------------

    def set_simulated_split(
        self,
        left_node,
        right_node,
        split_value,
        split_var_type: str,
        n_values_of_categorical_split_var: int | None = None,
    ) -> None:
        self._left_node = left_node
        self._right_node = right_node
        self._split_value = split_value
        self._split_var_type = split_var_type
        self._n_values_of_categorical_split_var = n_values_of_categorical_split_var

    def get_left_node(self):
        return self._left_node

    def get_right_node(self):
        return self._right_node

    def get_split_value(self):
        return self._split_value

    def get_split_var_type(self):
        return self._split_var_type

    def get_n_values_of_categorical_split_var(self):
        return self._n_values_of_categorical_split_var

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
            f"node_id={node_id}, split_var={self._split_var!r}, "
            f"split_var_type={self._split_var_type!r}, tree_cost={self._tree_cost:.6f})"
        )

    def __repr__(self) -> str:
        return self.__str__()
