# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

from typing import Any
from collections import deque
import pandas as pd

from kuplift.mt_node import Node
from kuplift.utils import build_table_by_cell, join_jt, split_jt, probability


class Tree:
    def __init__(
        self,
        dataset: pd.DataFrame,
        features: list[str],  # raw features
        treatment_col_name: str,
        target_col_name: str,
        positive_target
    ):
        self.dataset = dataset
        self.features = list(features)
        self.treatment_col_name = treatment_col_name
        self.target_col_name = target_col_name
        self.positive_target = positive_target

        self.feature_subset: set[str] = set()

        self.target_modalities = sorted(list(dataset[target_col_name].dropna().unique()), key=lambda x: str(x))
        self.treatment_modalities = sorted(list(dataset[treatment_col_name].dropna().unique()), key=lambda x: str(x))

        self.negative_target = self.target_modalities[0] if self.target_modalities[0] != self.positive_target else self.target_modalities[1]

        self.root_node = Node(
            dataset=dataset.copy(),
            treatment_col_name=treatment_col_name,
            target_col_name=target_col_name,
            parent=None,
        )

        self.internal_nodes: list[Node] = []
        self.leaf_nodes: list[Node] = [self.root_node]

    @property
    def used_variable_count(self) -> int:
        return len(self.feature_subset)

    @property
    def treatment_modality_count(self) -> int:
        return len(self.treatment_modalities)

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

    def apply_split(
        self,
        node: Node,
        split_var: str,
        left_node: Node,
        right_node: Node,
        split_value,
        split_var_type: str,
        source_partition_info: dict | None,
        n_values_of_categorical_split_var: int | None,
    ) -> None:
        if node not in self.leaf_nodes:
            raise ValueError("Can only split a leaf node currently present in leaf_nodes")

        node.apply_split(
            split_var=split_var,
            left_node=left_node,
            right_node=right_node,
            split_value=split_value,
            split_var_type=split_var_type,
            source_partition_info=source_partition_info,
            n_values_of_categorical_split_var=n_values_of_categorical_split_var,
        )

        self.leaf_nodes.remove(node)
        self.leaf_nodes.extend([left_node, right_node])
        self.internal_nodes.append(node)
        self.feature_subset.add(split_var)

    def get_node_by_id(self, node_id: int):
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

    def get_treatment_groups_of_leaves(self, sort=None) -> pd.DataFrame:
        ids, paths, groups = [], [], []
        for leaf in self.leaf_nodes:
            ids.append(leaf.id)
            paths.append(leaf.get_path_str())
            groups.append(leaf.treatment_groups)
        return self._sort_nodes(pd.DataFrame(index=ids, data={"Path": paths, "Treatment groups": groups}), sort)
    
    def get_target_frequencies(self, sort=None) -> pd.DataFrame:
        def frequency(id_index, id, jt_index, jt) -> int:
            leaf = self.get_node_by_id(id)
            j, t = split_jt(jt)
            return len(leaf.dataset[(leaf.dataset[self.target_col_name].astype(str) == j) & (leaf.dataset[self.treatment_col_name].astype(str) == t)])
        return self._sort_nodes(build_table_by_cell(
            rows=[leaf.id for leaf in self.leaf_nodes],
            columns=[join_jt(j, t) for j in self.target_modalities for t in self.treatment_modalities],
            func=frequency
        ), sort)
    
    def get_target_probabilities(self, sort=None) -> pd.DataFrame:
        frequencies = self.get_target_frequencies()
        def probability_(id_index, id, jt_index, jt) -> float:
            j, t = split_jt(jt)
            other_j = str(self.positive_target if j != str(self.positive_target) else self.negative_target)
            return probability(frequencies[jt][id], frequencies[join_jt(other_j, t)][id])
        return self._sort_nodes(build_table_by_cell(
            rows=[leaf.id for leaf in self.leaf_nodes],
            columns=[join_jt(j, t) for j in self.target_modalities for t in self.treatment_modalities],
            func=probability_
        ), sort)
    
    def get_uplift(self, sort=None) -> pd.DataFrame:
        probabilities = self.get_target_probabilities()
        return self._sort_nodes(build_table_by_cell(
            rows=[leaf.id for leaf in self.leaf_nodes],
            columns=self.treatment_modalities,
            func=lambda id_index, id, t_index, t: probabilities[join_jt(self.positive_target, t)][id] - probabilities[join_jt(self.negative_target, t)][id]
        ), sort)
    
    def node_ids_sorted_dfs(self) -> pd.Index:
        nodes = deque([self.root_node])
        node_ids = []
        while nodes:
            node = nodes.popleft()
            node_ids.append(node.id)
            if node.type == "internal":
                nodes.appendleft(node.right_node)
                nodes.appendleft(node.left_node)
        return pd.Index(node_ids)

    def leaf_ids_sorted_dfs(self) -> pd.Index:
        nodes = deque([self.root_node])
        node_ids = []
        while nodes:
            node = nodes.popleft()
            if node.type == "internal":
                nodes.appendleft(node.right_node)
                nodes.appendleft(node.left_node)
            else:
                node_ids.append(node.id)
        return pd.Index(node_ids)
    
    def node_ids_sorted_dfs_from(self, node_ids: pd.Index) -> pd.Index:
        return pd.Index(node_id for node_id in self.node_ids_sorted_dfs() if node_id in node_ids)
    
    def get_leaf_paths(self, sort=None) -> pd.Series:
        ids = []
        paths = []
        for leaf in self.leaf_nodes:
            ids.append(leaf.id)
            paths.append(leaf.get_path_str())
        return self._sort_nodes(pd.Series(name="Path", index=[leaf.id for leaf in self.leaf_nodes], data=paths), sort)
        
    def _sort_nodes(self, data, sort=None):
        if sort is None:
            return data
        elif sort == "dfs":
            return data.sort_index(key=self.node_ids_sorted_dfs_from)
        else:
            raise ValueError("unsupported sorting algorithm {!r}".format(sort))