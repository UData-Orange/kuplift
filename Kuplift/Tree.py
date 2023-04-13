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
    Data_features : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series
        Treatment column.
    y_col : pd.Series
        Outcome column.
    """

    def __init__(self, data, treatmentName, outcomeName):  # ordered data as argument
        self.nodesIds = 0
        self.rootNode = _Node(data, treatmentName, outcomeName, ID=self.nodesIds + 1)
        self.terminalNodes = [self.rootNode]
        self.internalNodes = []

        self.K = len(list(data.columns))
        self.K_t = 1
        self.features = list(data.columns)
        self.feature_subset = []

        self.Prob_Kt = None
        self.EncodingOfBeingAnInternalNode = None
        self.ProbAttributeSelection = None
        self.PriorOfInternalNodes = None
        self.EncodingOfBeingALeafNodeAndContainingTE = (
            len(self.terminalNodes) * log(2) * 2
        )  # TE=TreatmentEffect
        self.LeafPrior = None
        self.TreeLikelihood = None

        self.calc_criterion()

        self.TreeCriterion = (
            self.Prob_Kt
            + self.EncodingOfBeingAnInternalNode
            + self.ProbAttributeSelection
            + self.PriorOfInternalNodes
            + self.EncodingOfBeingALeafNodeAndContainingTE
            + self.LeafPrior
            + self.TreeLikelihood
        )

    def calc_criterion(self):
        self.__calc_prob_kt()
        self.__calc_prior_of_internal_node()
        self.__calc_encoding()
        self.__calc_leaf_prior()
        self.__calc_tree_likelihood()

    def __calc_prob_kt(self):
        self.Prob_Kt = (
            universal_code_natural_numbers(self.K_t)
            - log_fact(self.K_t)
            + self.K_t * log(self.K)
        )

    def __calc_prior_of_internal_node(self):
        if len(self.internalNodes) == 0:
            self.PriorOfInternalNodes = 0
            self.ProbAttributeSelection = 0
        else:
            PriorOfInternalNodes = 0
            for internalNode in self.internalNodes:
                PriorOfInternalNodes += internalNode.PriorOfInternalNode
            self.PriorOfInternalNodes = PriorOfInternalNodes
            self.ProbAttributeSelection = log(self.K_t) * len(self.internalNodes)

    def __calc_encoding(self):
        self.EncodingOfBeingALeafNodeAndContainingTE = (
            len(self.terminalNodes) * log(2) * 2
        )
        self.EncodingOfBeingAnInternalNode = len(self.internalNodes) * log(2)

    def __calc_leaf_prior(self):
        leafPriors = 0
        for leafNode in self.terminalNodes:
            leafPriors += leafNode.PriorLeaf
        self.LeafPrior = leafPriors

    def __calc_tree_likelihood(self):
        LeafLikelihoods = 0
        for leafNode in self.terminalNodes:
            LeafLikelihoods += leafNode.LikelihoodLeaf
        self.TreeLikelihood = LeafLikelihoods

    def __traverse_tree(self, x, node):
        if node.isLeaf == True:
            return node.averageUplift

        if x[node.Attribute] <= node.SplitThreshold:
            return self.__traverse_tree(x, node.leftNode)
        return self.__traverse_tree(x, node.rightNode)

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
            self.__traverse_tree(X_test.iloc[x], self.rootNode)
            for x in range(len(X_test))
        ]
        return np.array(predictions)
