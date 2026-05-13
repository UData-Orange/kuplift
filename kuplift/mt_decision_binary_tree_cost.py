# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from __future__ import annotations

import math

from kuplift.mt_decision_tree_cost import MultiTreatmentDecisionTreeCost
from kuplift.mt_kwstat import KWStat


class MultiTreatmentDecisionBinaryTreeCost(MultiTreatmentDecisionTreeCost):
    """
    Default binary-tree cost model for multi-treatment DecisionTree.

    Notes
    -----
    - The tree structure remains binary (left/right split).
    - This cost model is robust to multi-treatment datasets as long as node-level
      class/treatment counts can be accessed via generic multi-treatment Node methods.
    - Hypothetical split simulation is STRICT: the candidate split must provide
      simulated children and split metadata; no hidden split recomputation here.
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
        # TODO: Re-implement.
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
            n_values = getattr(node, "n_values_of_categorical_split_var", None)
            if n_values is None:
                raise RuntimeError(
                    f"Missing n_values_of_categorical_split_var for categorical split variable {split_var!r}"
                )
            if int(n_values) <= 0:
                raise ValueError(
                    f"Invalid n_values_of_categorical_split_var={n_values} for variable {split_var!r}"
                )
            cost += KWStat.LnBellNumber(int(n_values), 2)
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

        Node is expected to expose:
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
        The candidate split MUST provide:
          - node_split.get_splittable_node()
          - node_split.get_split_var()
          - node_split.get_left_node()
          - node_split.get_right_node()
          - node_split.get_split_var_type()
          - node_split.get_n_values_of_categorical_split_var()  (if categorical)
        """
        source_node = node_split.get_splittable_node()
        split_var = node_split.get_split_var()

        if source_node is None or split_var is None:
            raise ValueError("node_split must provide splittable_node and split_var")

        left_node = node_split.get_left_node()
        right_node = node_split.get_right_node()
        split_var_type = node_split.get_split_var_type()
        n_values_cat = node_split.get_n_values_of_categorical_split_var()

        if left_node is None or right_node is None:
            raise ValueError("node_split must provide simulated left_node and right_node")
        if split_var_type is None:
            raise ValueError("node_split must provide split_var_type")
        if split_var_type == "Categorical" and n_values_cat is None:
            raise ValueError(
                "node_split must provide n_values_of_categorical_split_var for categorical splits"
            )

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
        proxy.n_values_of_categorical_split_var = n_values_cat

        cost += self.compute_internal_node_cost(tree, proxy)
        cost -= self.compute_leaf_cost(source_node, tree)
        cost += self.compute_leaf_cost(left_node, tree)
        cost += self.compute_leaf_cost(right_node, tree)

        node_split.set_tree_cost(float(cost))
        return float(cost)
