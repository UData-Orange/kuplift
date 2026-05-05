# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

import math

from kuplift.dt_decision_tree_cost_v3 import DTDecisionTreeCostV3
from kuplift.kwstat_v3 import KWStat


class DTDecisionBinaryTreeCostV3(DTDecisionTreeCostV3):
    """
    Default binary-tree cost model for MultiTreatmentDecisionTreeV3.

    Notes
    -----
    - The tree structure remains binary (left/right split).
    - This cost model is robust to multi-treatment datasets as long as node-level
      class/treatment counts can be accessed via generic NodeV3 methods.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Null tree cost (root-only)
    # ------------------------------------------------------------------

    def compute_null_tree_cost(self, tree) -> float:
        """
        Cost of a root-only tree.
        We reuse the general "total-tree" formula; with no internal node and one leaf,
        it naturally gives the null model cost.
        """
        return self.compute_total_tree_cost(tree)

    # ------------------------------------------------------------------
    # Total tree cost
    # ------------------------------------------------------------------

    def compute_total_tree_cost(self, tree) -> float:
        """
        Total cost decomposition:
          - attribute choice cost
          - internal node costs
          - leaf costs
        """
        features = getattr(tree, "features", None)
        if not features:
            return 0.0

        n_used_attributes = len(getattr(tree, "feature_subset", []) or [])
        n_internal_nodes = len(getattr(tree, "internal_nodes", []) or [])

        cost = self.compute_attribute_choice_cost(
            n_used_attribute_number=n_used_attributes,
            n_internal_node_number=n_internal_nodes,
        )

        for node in getattr(tree, "internal_nodes", []):
            cost += self.compute_internal_node_cost(tree, node)

        for node in getattr(tree, "leaf_nodes", []):
            cost += self.compute_leaf_cost(node, tree)

        return float(cost)

    # ------------------------------------------------------------------
    # Attribute choice cost
    # ------------------------------------------------------------------

    def compute_attribute_choice_cost(
        self, n_used_attribute_number: int, n_internal_node_number: int
    ) -> float:
        if self.total_attribute_number <= 0:
            return 0.0
        if n_used_attribute_number < 0:
            raise ValueError("n_used_attribute_number must be >= 0")
        if n_internal_node_number < 0:
            raise ValueError("n_internal_node_number must be >= 0")
        if n_internal_node_number < n_used_attribute_number:
            raise ValueError("n_internal_node_number must be >= n_used_attribute_number")

        if n_used_attribute_number == 0:
            return math.log(2.0)

        cost = math.log(2.0) * (1.0 + n_internal_node_number)
        cost += math.log(float(n_used_attribute_number)) * n_internal_node_number
        cost += KWStat.NaturalNumbersUniversalCodeLength(n_used_attribute_number)
        cost -= KWStat.LnFactorial(n_used_attribute_number)
        return float(cost)

    # ------------------------------------------------------------------
    # Internal node cost
    # ------------------------------------------------------------------

    def compute_internal_node_cost(self, tree, node) -> float:
        """
        Internal node coding cost:
          - one bit for internal/leaf choice term
          - split descriptor:
              * Numerical: threshold among N positions ~ ln(N+1)
              * Categorical: grouping to 2 parts via Bell number proxy
        """
        cost = math.log(2.0)

        split_var = getattr(node, "split_var", None)
        split_var_type = getattr(node, "split_var_type", None)

        # If no effective split info, keep only base term
        if split_var is None or split_var_type is None:
            return float(cost)

        n_instances = int(getattr(node, "sample_size", 0))
        if n_instances < 0:
            n_instances = 0

        if split_var_type == "Numerical":
            cost += math.log(n_instances + 1.0)
        elif split_var_type == "Categorical":
            # We estimate number of values from node dataset
            dataset = getattr(node, "dataset", None)
            if dataset is not None and split_var in dataset.columns:
                n_values = int(dataset[split_var].nunique(dropna=False))
                n_values = max(n_values, 1)
                # grouping to 2 parts
                cost += KWStat.LnBellNumber(n_values, 2)
            else:
                # cannot get number of distinct values
                raise RuntimeError("could not get number of distinct values for variable {!r}".format(split_var))
        else:
            # unknown variable type
            raise ValueError("unsupported variable type {!r}".format(split_var_type))

        return float(cost)

    # ------------------------------------------------------------------
    # Leaf cost
    # ------------------------------------------------------------------

    def compute_leaf_cost(self, node, tree=None) -> float:
        """
        Generic leaf likelihood/prior coding from treatment-target contingency table.

        NodeV3 is expected to expose:
          - get_treatments()
          - get_targets()
          - get_count(treatment, target)
          - get_total_count()
          - get_treatment_groups()  (optional; default = singleton groups)
        """
        # model-choice term
        cost = math.log(2.0)

        # Generic modalities
        treatments = list(node.get_treatments())
        targets = list(node.get_targets())
        T = len(treatments)
        J = len(targets)

        N = int(node.get_total_count())

        # Treatment groups (optional)
        groups = node.get_treatment_groups()
        if not groups:
            # default: one group per treatment
            groups = [tuple([t]) for t in treatments]
        G = len(groups)

        # group-structure coding
        if T > 0 and G > 0:
            cost += math.log(float(T))
            cost += KWStat.LnBellNumber(T, G)

        # per-group multinomial coding
        for group in groups:
            Ng = 0
            for t in group:
                for y in targets:
                    Ng += int(node.get_count(t, y))

            cost += math.log(Ng + 1.0)
            cost += KWStat.LnFactorial(Ng)

            for y in targets:
                n_event = 0
                for t in group:
                    n_event += int(node.get_count(t, y))
                cost -= KWStat.LnFactorial(n_event)

        # tiny guard (not strictly necessary)
        if N == 0:
            return math.log(2.0)

        return float(cost)

    # ------------------------------------------------------------------
    # Hypothetical augmented cost
    # ------------------------------------------------------------------

    def compute_hypothetical_augmented_tree_cost(
        self, tree, previous_cost: float, node_split
    ) -> float:
        """
        Compute hypothetical tree cost if node_split is applied.
        The candidate split must provide:
          - node_split.get_splittable_node()
          - node_split.get_split_var()
        NodeV3 is expected to support:
          - simulate_split(split_var) -> (left_node, right_node, split_value, split_var_type)
        """
        source_node = node_split.get_splittable_node()
        split_var = node_split.get_split_var()

        if source_node is None or split_var is None:
            node_split.set_tree_cost(float(previous_cost))
            return float(previous_cost)

        # Simulate split without mutating the original tree
        simulated = source_node.simulate_split(split_var)
        if simulated is None:
            node_split.set_tree_cost(float(previous_cost))
            return float(previous_cost)

        left_node, right_node, split_value, split_var_type = simulated

        # Build incremental update from previous cost:
        # + possible attribute-choice delta
        # + internal node cost (for source node converted to internal)
        # - old leaf cost (source leaf)
        # + new leaves costs
        cost = float(previous_cost)

        used_vars = set(getattr(tree, "feature_subset", []) or [])
        internal_nodes = getattr(tree, "internal_nodes", []) or []
        n_used = len(used_vars)
        n_internal = len(internal_nodes)

        if split_var not in used_vars:
            cost -= self.compute_attribute_choice_cost(n_used, n_internal)
            cost += self.compute_attribute_choice_cost(n_used + 1, n_internal + 1)
        else:
            cost += math.log(2.0)
            if n_used > 0:
                cost += math.log(float(n_used))

        # Create a lightweight proxy representing source as internal
        class _InternalProxy:
            pass

        proxy = _InternalProxy()
        proxy.split_var = split_var
        proxy.split_var_type = split_var_type
        proxy.sample_size = source_node.sample_size
        proxy.dataset = source_node.dataset

        cost += self.compute_internal_node_cost(tree, proxy)
        cost -= self.compute_leaf_cost(source_node, tree)
        cost += self.compute_leaf_cost(left_node, tree)
        cost += self.compute_leaf_cost(right_node, tree)

        node_split.set_tree_cost(float(cost))
        return float(cost)
