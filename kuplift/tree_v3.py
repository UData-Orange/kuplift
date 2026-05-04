# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Any
import pandas as pd

from kuplift.node_v3 import NodeV3


class TreeV3:
    """
    Binary tree container for MultiTreatmentDecisionTreeV3.

    Exposed properties required by spec:
      - used_variable_count
      - target_modalities
      - treatment_modalities
      - root_node
      - internal_nodes
      - leaf_nodes
      - treatment_modality_count
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        treatment_col_name: str,
        target_col_name: str,
    ):
        self.dataset = dataset
        self.features = list(features)
        self.treatment_col_name = treatment_col_name
        self.target_col_name = target_col_name

        self.feature_subset: set[str] = set()

        self.target_modalities = sorted(
            list(dataset[target_col_name].dropna().unique()), key=lambda x: str(x)
        )
        self.treatment_modalities = sorted(
            list(dataset[treatment_col_name].dropna().unique()), key=lambda x: str(x)
        )

        self.root_node = NodeV3(
            dataset=dataset.copy(),
            treatment_col_name=treatment_col_name,
            target_col_name=target_col_name,
            parent=None,
        )

        self.internal_nodes: list[NodeV3] = []
        self.leaf_nodes: list[NodeV3] = [self.root_node]

    # ------------------------------------------------------------------
    # Required metrics/properties
    # ------------------------------------------------------------------

    @property
    def used_variable_count(self) -> int:
        return len(self.feature_subset)

    @property
    def treatment_modality_count(self) -> int:
        return len(self.treatment_modalities)

    # ------------------------------------------------------------------
    # Utility methods expected by cost model
    # ------------------------------------------------------------------

    def get_instances_number(self) -> int:
        return int(len(self.dataset))

    def get_class_value_number(self) -> int:
        return int(len(self.target_modalities))

    def get_treatment_number(self) -> int:
        return int(len(self.treatment_modalities))

    def get_number_of_instances(self, target_value: Any, treatment_value: Any) -> int:
        mask = (
            (self.dataset[self.target_col_name] == target_value)
            & (self.dataset[self.treatment_col_name] == treatment_value)
        )
        return int(mask.sum())

    # ------------------------------------------------------------------
    # Tree mutation
    # ------------------------------------------------------------------

    def apply_split(
        self,
        node: NodeV3,
        split_var: str,
        left_node: NodeV3,
        right_node: NodeV3,
        split_value,
        split_var_type: str,
    ) -> None:
        """
        Apply validated binary split to a leaf node.
        """
        if node not in self.leaf_nodes:
            raise ValueError("Can only split a leaf node currently present in leaf_nodes")

        node.apply_split(
            split_var=split_var,
            left_node=left_node,
            right_node=right_node,
            split_value=split_value,
            split_var_type=split_var_type,
        )

        self.leaf_nodes.remove(node)
        self.leaf_nodes.extend([left_node, right_node])
        self.internal_nodes.append(node)
        self.feature_subset.add(split_var)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_node_by_id(self, node_id: int) -> NodeV3 | None:
        if self.root_node.id == node_id:
            return self.root_node
        for n in self.internal_nodes:
            if n.id == node_id:
                return n
        for n in self.leaf_nodes:
            if n.id == node_id:
                return n
        return None

    def get_node_path_str(self, node_id: int, separator: str = " AND ") -> str:
        node = self.get_node_by_id(node_id)
        if node is None:
            raise ValueError(f"No node found with id={node_id}")
        return node.get_path_str(separator=separator)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"TreeV3("
            f"instances={self.get_instances_number()}, "
            f"features={len(self.features)}, "
            f"used_variable_count={self.used_variable_count}, "
            f"treatment_modality_count={self.treatment_modality_count}, "
            f"internal_nodes={len(self.internal_nodes)}, "
            f"leaf_nodes={len(self.leaf_nodes)})"
        )

    def __repr__(self) -> str:
        return self.__str__()
