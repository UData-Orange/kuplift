######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import numpy as np
from math import log
from .HelperFunctions import (
    log_fact,
    universal_code_natural_numbers,
)
from .Node import _Node


class _Tree:
    """Private parent class

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series
        Treatment column.
    y_col : pd.Series
        Outcome column.
    """

    def __init__(self, data, treatment_col, y_col):  # ordered data as argument
        self.nodes_ids = 0
        self.root_node = _Node(data, treatment_col, y_col, ID=self.nodes_ids + 1)
        self.terminal_nodes = [self.root_node]
        self.internal_nodes = []

        self.k = len(list(data.columns))
        self.k_t = 1
        self.features = list(data.columns)
        self.feature_subset = []

        self.prob_kt = None
        self.encoding_of_being_an_internal_node = None
        self.prob_attribute_selection = None
        self.prior_of_internal_nodes = None
        self.encoding_of_being_a_leaf_node_and_containing_te = (
            len(self.terminal_nodes) * log(2) * 2
        )  # TE=TreatmentEffect
        self.leaf_prior = None
        self.tree_likelihood = None

        self.calc_criterion()

        self.tree_criterion = (
            self.prob_kt
            + self.encoding_of_being_an_internal_node
            + self.prob_attribute_selection
            + self.prior_of_internal_nodes
            + self.encoding_of_being_a_leaf_node_and_containing_te
            + self.leaf_prior
            + self.tree_likelihood
        )

    def calc_criterion(self):
        self.__calc_prob_kt()
        self.__calc_prior_of_internal_node()
        self.__calc_encoding()
        self.__calc_leaf_prior()
        self.__calc_tree_likelihood()

    def __calc_prob_kt(self):
        self.prob_kt = (
            universal_code_natural_numbers(self.k_t)
            - log_fact(self.k_t)
            + self.k_t * log(self.k)
        )

    def __calc_prior_of_internal_node(self):
        if len(self.internal_nodes) == 0:
            self.prior_of_internal_nodes = 0
            self.prob_attribute_selection = 0
        else:
            prior_of_internal_nodes = 0
            for internal_node in self.internal_nodes:
                prior_of_internal_nodes += internal_node.prior_of_internal_node
            self.prior_of_internal_nodes = prior_of_internal_nodes
            self.prob_attribute_selection = log(self.k_t) * len(self.internal_nodes)

    def __calc_encoding(self):
        self.encoding_of_being_a_leaf_node_and_containing_te = (
            len(self.terminal_nodes) * log(2) * 2
        )
        self.encoding_of_being_an_internal_node = len(self.internal_nodes) * log(2)

    def __calc_leaf_prior(self):
        leaf_priors = 0
        for leaf_node in self.terminal_nodes:
            leaf_priors += leaf_node.prior_leaf
        self.leaf_prior = leaf_priors

    def __calc_tree_likelihood(self):
        leaf_likelihoods = 0
        for leaf_node in self.terminal_nodes:
            leaf_likelihoods += leaf_node.likelihood_leaf
        self.tree_likelihood = leaf_likelihoods

    def __traverse_tree(self, x, node):
        if node.is_leaf == True:
            return node.average_uplift

        if x[node.attribute] <= node.split_threshold:
            return self.__traverse_tree(x, node.left_node)
        return self.__traverse_tree(x, node.right_node)

    def predict(self, X_test):
        """Predict the uplift value for each example in X_test

        Parameters
        ----------
        X_train : pd.Dataframe
            Dataframe containing feature variables.

        Returns
        -------
        y_pred_list(ndarray, shape=(num_samples, 1))
            An array containing the predicted treatment uplift for each sample.
        """
        predictions = [
            self.__traverse_tree(X_test.iloc[x], self.root_node)
            for x in range(len(X_test))
        ]
        return np.array(predictions)
