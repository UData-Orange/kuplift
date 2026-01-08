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
import numpy as np
import random
from .helperfunctions import (
    log_fact,
    universal_code_natural_numbers,
)
from .tree import _Tree


class _UpliftTreeClassifier(_Tree):
    """Private child class

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

    def __init__(self,control_name=None):
        super().__init__()

    def grow_tree(self,data, treatment_col, y_col):
        super().__initializeVars__(data.copy(), treatment_col.copy(), y_col.copy())
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
            # Dictionary containing Nodes as key and their values are another
            # dictionary each with attribute:CostSplit
            node_vs_candidate_splits_costs = {}
            for terminal_node in self.terminal_nodes:
                # This if condition is here to not to repeat calculations of
                # candidate splits
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
                list_of_attribute_splits_improving_tree_criterion = []
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

                    if (
                        node_vs_candidate_splits_costs[terminal_node][
                            attribute
                        ]
                        < self.tree_criterion
                    ):
                        list_of_attribute_splits_improving_tree_criterion.append(
                            attribute
                        )
                if len(list_of_attribute_splits_improving_tree_criterion) == 0:
                    continue
                key_of_the_minimal_val = random.choice(
                    list_of_attribute_splits_improving_tree_criterion
                )  # key_of_the_minimal_val is the attribute name

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
            optimal_node_attribute_to_split_up = random.choice(
                list(node_vs_best_cost)
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

        self.tree_criterion = (
            self.prob_kt
            + self.encoding_of_being_an_internal_node
            + self.prob_attribute_selection
            + self.prior_of_internal_nodes
            + self.encoding_of_being_a_leaf_node_and_containing_te
            + self.leaf_prior
            + self.tree_likelihood
        )


class BayesianRandomForest:
    """
    The BayesianRandomForest class implements the UB-RF algorithm described in:
    Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2023, May).
    A Non-Parametric Bayesian Decision Trees for Uplift modelling. In PAKDD.

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe containing data.
    treatment_col : pd.Series
        Treatment column.
    outcome_col : pd.Series
        Outcome column.
    n_trees : int, default 10
        Number of trees in a forest.
    vars_subset : bool, default False
        Use a random subset of the variables for each tree in the forest.
    random_state : int, default 10
        Seed used by the random number generator.
    """

    def __init__(
        self,
        n_trees=10,
        vars_subset=False,
        random_state=10,
    ):
        self.list_of_trees = []
        
        self.n_trees=n_trees

        self.vars_subset=vars_subset

        self.random_state=random_state
        random.seed(self.random_state)

        self.treatment_name='treatment'

        self.outcome_name='outcome'

    def fit(self,data,treatment_col,y_col):
        """Fit a decision tree algorithm."""
        # data.loc[:,self.treatment_name]=treatment_col
        # data.loc[:,self.outcome_name]=y_col
        # self.data = data
        
        if self.vars_subset: # Randomly select columns for the data
            cols = list(self.data.columns)
            # cols.remove(self.treatment_name)
            # cols.remove(self.outcome_name)
            cols = random.sample(cols, int(np.sqrt(len(cols))))
            # data = data[cols + [self.treatment_name, self.outcome_name]]
            data = data[cols]

        for i in range(self.n_trees):
            Tree = _UpliftTreeClassifier()
            self.list_of_trees.append(Tree)

        for tree in self.list_of_trees:
            tree.grow_tree(data.copy(),treatment_col.copy(), y_col.copy())

    def predict(self, X_test, weighted_average=False):
        """
        Predict the uplift value for each example in X_test.

        Parameters
        ----------
        X_test : pd.Dataframe
            Dataframe containing test data.
        weighted_average : bool, default False
            Give a weight for the predictions of each tree according to its cost.

        Returns
        -------
        y_pred_list : (ndarray, shape=(num_samples, 1))
            An array containing the predicted uplift for each sample.
        """
        if not weighted_average:
            list_of_preds = []

            for tree in self.list_of_trees:
                list_of_preds.append(np.array(tree.predict(X_test)))
            return np.mean(list_of_preds, axis=0)
        else:
            list_of_criterion = []
            list_of_preds = []
            for tree in self.list_of_trees:
                list_of_criterion.append(np.array(tree.tree_criterion))
                list_of_preds.append(np.array(tree.predict(X_test)))

            sum_of_criterions = sum(list_of_criterion)
            for i in range(len(list_of_criterion)):
                list_of_criterion[i] = sum_of_criterions / list_of_criterion[i]
            sum_of_weights = sum(list_of_criterion)

            list_of_weights = []
            for i in range(len(list_of_criterion)):
                list_of_weights.append(list_of_criterion[i] / sum_of_weights)
            return np.average(list_of_preds, axis=0, weights=list_of_weights)
