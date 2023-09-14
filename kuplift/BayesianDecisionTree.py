######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
from math import log
from .HelperFunctions import (
    log_fact,
    universal_code_natural_numbers,
)
from .Tree import _Tree


class BayesianDecisionTree(_Tree):
    """
    The BayesianDecisionTree class implements the UB-DT algorithm described in:
    Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2023, May).
    A Non-Parametric Bayesian Decision Trees for Uplift modelling. In PAKDD.

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series
        Treatment column.
    y_col : pd.Series
        Outcome column.
    control_name: int or str
        The name of the control value in the treatment column
    """

    def __init__(self, control_name=None):
        super().__init__(control_name)

    def fit(self,data, treatment_col, y_col):
        """Fit an uplift decision tree model using UB-DT

        Parameters
        ----------
        X_train : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        """
        super().__initializeVars__(data, treatment_col, y_col)
        # In case if we have a new attribute for splitting
        prob_kt_plus_one = (
            universal_code_natural_numbers(self.k_t + 1)
            - log_fact(self.k_t + 1)
            + (self.k_t + 1) * log(self.k)
        )
        prob_of_attribute_selection_among_subset_attributes_plus_one = log(
            self.k_t + 1
        ) * (len(self.internal_nodes) + 1)

        encoding_of_being_an_internal_node_plus_one = (
            self.encoding_of_being_an_internal_node + log(2)
        )

        # When splitting a node to 2 nodes, the number of leaf nodes is
        # incremented only by one, since the parent node was leaf
        # and is now internal.
        # 2 for two extra leaf nodes multiplied by 2 for w. Total = 4.
        encoding_of_being_a_leaf_node_and_containing_te_plus_two = (
            self.encoding_of_being_a_leaf_node_and_containing_te + (2 * log(2))
        )

        encoding_of_internal_and_leaves_and_w_with_extra_nodes = (
            encoding_of_being_an_internal_node_plus_one
            + encoding_of_being_a_leaf_node_and_containing_te_plus_two
        )

        i = 0
        while True:
            node_vs_best_attribute_corresponding_to_the_best_cost = {}
            node_vs_best_cost = {}
            node_vs_candidate_splits_costs = (
                {}
            )  # Dictionary containing Nodes as key and their values are another dictionary each with attribute:CostSplit

            for terminal_node in self.terminal_nodes:
                # This if condition is here to not to repeat calculations of candidate splits
                if terminal_node.candidate_splits_vs_criterion is None:
                    node_vs_candidate_splits_costs[
                        terminal_node
                    ] = (
                        terminal_node.discretize_vars_and_get_attributes_splits_costs()
                    )
                else:
                    node_vs_candidate_splits_costs[
                        terminal_node
                    ] = terminal_node.candidate_splits_vs_criterion.copy()

                if len(node_vs_candidate_splits_costs[terminal_node]) == 0:
                    continue

                # Update Costs
                for attribute in node_vs_candidate_splits_costs[terminal_node]:
                    if attribute in self.feature_subset:
                        node_vs_candidate_splits_costs[terminal_node][
                            attribute
                        ] += (
                            self.prob_kt
                            + self.prob_attribute_selection
                            + encoding_of_internal_and_leaves_and_w_with_extra_nodes
                            + self.leaf_prior
                            + self.tree_likelihood
                            + self.prior_of_internal_nodes
                        )
                    else:
                        node_vs_candidate_splits_costs[terminal_node][
                            attribute
                        ] += (
                            prob_kt_plus_one
                            + encoding_of_internal_and_leaves_and_w_with_extra_nodes
                            + prob_of_attribute_selection_among_subset_attributes_plus_one
                            + self.leaf_prior
                            + self.tree_likelihood
                            + self.prior_of_internal_nodes
                        )

                # Once costs are updated, I get the key of the minimal value
                # split for terminal_node
                key_of_the_minimal_val = min(
                    node_vs_candidate_splits_costs[terminal_node],
                    key=node_vs_candidate_splits_costs[terminal_node].get,
                )

                node_vs_best_attribute_corresponding_to_the_best_cost[
                    terminal_node
                ] = key_of_the_minimal_val
                node_vs_best_cost[
                    terminal_node
                ] = node_vs_candidate_splits_costs[terminal_node][
                    key_of_the_minimal_val
                ]

            if len(list(node_vs_best_cost)) == 0:
                break

            optimal_node_attribute_to_split_up = min(
                node_vs_best_cost, key=node_vs_best_cost.get
            )
            optimal_val = node_vs_best_cost[optimal_node_attribute_to_split_up]
            optimal_node = optimal_node_attribute_to_split_up
            optimal_attribute = (
                node_vs_best_attribute_corresponding_to_the_best_cost[
                    optimal_node_attribute_to_split_up
                ]
            )

            if optimal_val < self.tree_criterion:
                self.tree_criterion = optimal_val
                if optimal_attribute not in self.feature_subset:
                    self.feature_subset.append(optimal_attribute)
                    self.k_t += 1
                new_left_leaf, new_right_leaf = optimal_node.perform_split(
                    optimal_attribute
                )
                self.terminal_nodes.append(new_left_leaf)
                self.terminal_nodes.append(new_right_leaf)
                self.internal_nodes.append(optimal_node)
                self.terminal_nodes.remove(optimal_node)

                self.calc_criterion()
            else:
                break
