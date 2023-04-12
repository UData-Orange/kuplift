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
from .BinaryDiscretizationFunctions import umodl_binary_discretization
from .HelperFunctions import (
    log_fact,
    universal_code_natural_numbers,
    log_binomial_coefficient,
)


class _Node:
    """Private class

    Parameters
    ----------
    Data_features : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series
        Treatment column.
    y_col : pd.Series
        Outcome column.
    ID : ?, optional
        ?
    """

    def __init__(self, data, treatmentName, outcomeName, ID=None):
        # Initialize attributes
        self.id = ID
        self.data = data.copy()
        self.treatment = treatmentName
        self.output = outcomeName
        self.N = data.shape[0]
        self.Nj = data[data[self.output] == 1].shape[0]
        self.Ntj = [
            data[(data[self.treatment] == 0) & (data[self.output] == 0)].shape[0],
            data[(data[self.treatment] == 0) & (data[self.output] == 1)].shape[0],
            data[(data[self.treatment] == 1) & (data[self.output] == 0)].shape[0],
            data[(data[self.treatment] == 1) & (data[self.output] == 1)].shape[0],
        ]
        self.X = data.iloc[:, :-2].copy()
        self.T = data.iloc[:, -2].copy()
        self.Y = data.iloc[:, -1].copy()

        try:
            if (self.Ntj[2] + self.Ntj[3]) == 0:
                denum = 0.00001
            else:
                denum = self.Ntj[2] + self.Ntj[3]
            self.outcomeProbInTrt = self.Ntj[3] / denum
        except:
            self.outcomeProbInTrt = 0

        try:
            if (self.Ntj[0] + self.Ntj[1]) == 0:
                denum = 0.00001
            else:
                denum = self.Ntj[0] + self.Ntj[1]
            self.outcomeProbInCtrl = self.Ntj[1] / denum
        except:
            self.outcomeProbInCtrl = 0

        self.averageUplift = self.outcomeProbInTrt - self.outcomeProbInCtrl
        self.Attribute = None
        self.SplitThreshold = None
        self.isLeaf = True
        self.CandidateSplitsVsDataLeftDataRight = None
        self.CandidateSplitsVsCriterion = None
        self.leftNode = None
        self.rightNode = None
        self.PriorOfInternalNode = self.__calc_prior_of_internal_node()
        (
            self.PriorLeaf,
            self.LikelihoodLeaf,
            self.W,
        ) = self.__calc_prior_and_likelihood_leaf()

    def __calc_prior_of_internal_node(self):
        return log_binomial_coefficient(sum(self.Ntj) + 1, 1)

    def __calc_prior_and_likelihood_leaf(self):
        NumberOfTreatment = self.Ntj[2] + self.Ntj[3]
        NumberOfControl = self.Ntj[0] + self.Ntj[1]
        NumberOfPosOutcome = self.Ntj[1] + self.Ntj[3]
        NumberOfZeroOutcome = self.Ntj[0] + self.Ntj[2]

        LeafPrior_W0 = log_binomial_coefficient(sum(self.Ntj) + 1, 1)
        TreeLikelihood_W0 = (
            log_fact(sum(self.Ntj))
            - log_fact(NumberOfPosOutcome)
            - log_fact(NumberOfZeroOutcome)
        )

        LeafPrior_W1 = log_binomial_coefficient(
            NumberOfTreatment + 1, 1
        ) + log_binomial_coefficient(NumberOfControl + 1, 1)
        TreeLikelihood_W1 = (
            log_fact(NumberOfTreatment) - log_fact(self.Ntj[2]) - log_fact(self.Ntj[3])
        ) + (log_fact(NumberOfControl) - log_fact(self.Ntj[0]) - log_fact(self.Ntj[1]))

        if (LeafPrior_W0 + TreeLikelihood_W0) < (LeafPrior_W1 + TreeLikelihood_W1):
            W = 0
            LeafPrior = LeafPrior_W0
            TreeLikelihood = TreeLikelihood_W0
        else:
            W = 1
            LeafPrior = LeafPrior_W1
            TreeLikelihood = TreeLikelihood_W1
        return LeafPrior, TreeLikelihood, W

    def discretize_vars_and_get_attributes_splits_costs(self):
        """For this node loop on all attributes and get the optimal split for each one.

        Returns
        -------
        Dictionary of lists

        For example: return a dictionnary {age: Cost, sex: Cost}
        The cost here corresponds to
        1- the cost of this node to internal instead of leaf (CriterionToBeInternal-PriorLeaf)
        2- The combinatorial terms of the leaf prior and likelihood
        """
        features = list(self.X.columns)
        AttributeToSplitVsLeftAndRightData = {}
        for attribute in features:
            if (
                len(self.X[attribute].value_counts()) == 1
                or len(self.X[attribute].value_counts()) == 0
            ):
                continue
            DiscRes = umodl_binary_discretization(self.X, self.T, self.Y, attribute)
            if DiscRes == -1:
                continue
            dataLeft, dataRight, threshold = DiscRes[0], DiscRes[1], DiscRes[2]
            AttributeToSplitVsLeftAndRightData[attribute] = [
                dataLeft,
                dataRight,
                threshold,
            ]

        self.CandidateSplitsVsDataLeftDataRight = (
            AttributeToSplitVsLeftAndRightData.copy()
        )
        CandidateSplitsVsCriterion = self.__get_attributes_splits_costs(
            AttributeToSplitVsLeftAndRightData
        )
        self.CandidateSplitsVsCriterion = CandidateSplitsVsCriterion.copy()
        return CandidateSplitsVsCriterion.copy()

    def __get_attributes_splits_costs(self, DictOfEachAttVsEffectifs):
        # Prior of Internal node is only the combinatorial calculations
        CriterionToBeInternal = (
            self.__calc_prior_of_internal_node()
        )  # In case we split this node, it will be no more a leaf but an internal node
        NewPriorVals = CriterionToBeInternal - self.PriorLeaf - self.LikelihoodLeaf

        CandidateSplitsVsCriterion = {}
        for key in DictOfEachAttVsEffectifs:
            LeavesVal = self.__update_tree_criterion(DictOfEachAttVsEffectifs[key][:2])
            CandidateSplitsVsCriterion[key] = NewPriorVals + LeavesVal
        return CandidateSplitsVsCriterion.copy()

    def __update_tree_criterion(self, LeftAndRightData, simulate=True):
        LeavesVals = 0
        for (
            NewNodeEffectifs
        ) in LeftAndRightData:  # Loop on Left and Right candidate nodes
            L = _Node(NewNodeEffectifs, self.treatment, self.output)
            LeavesVals += L.PriorLeaf + L.LikelihoodLeaf
            del L
        return LeavesVals

    def perform_split(self, Attribute):
        if self.CandidateSplitsVsDataLeftDataRight == None:
            raise
        else:
            self.isLeaf = False
            self.leftNode = _Node(
                self.CandidateSplitsVsDataLeftDataRight[Attribute][0],
                self.treatment,
                self.output,
                ID=self.id * 2,
            )
            self.rightNode = _Node(
                self.CandidateSplitsVsDataLeftDataRight[Attribute][1],
                self.treatment,
                self.output,
                ID=self.id * 2 + 1,
            )
            self.Attribute = Attribute
            self.SplitThreshold = self.CandidateSplitsVsDataLeftDataRight[Attribute][2]
            return self.leftNode, self.rightNode


class BayesianDecisionTree:
    """
    The BayesianDecisionTree class implements the UB-DT algorithm described in:
    Rafla, M., Voisine, N., Crémilleux, B., \& Boullé, M. (2023, May). A Non-Parametric Bayesian Decision Trees for Uplift modelling. In PAKDD.
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
        self.PriorOfInternalNode = None
        self.EncodingOfBeingALeafNodeAndContainingTE = (
            len(self.terminalNodes) * log(2) * 2
        )  # TE=TreatmentEffect
        self.LeafPrior = None
        self.TreeLikelihood = None

        self.__calc_criterion()

        self.TreeCriterion = (
            self.Prob_Kt
            + self.EncodingOfBeingAnInternalNode
            + self.ProbAttributeSelection
            + self.PriorOfInternalNode
            + self.EncodingOfBeingALeafNodeAndContainingTE
            + self.LeafPrior
            + self.TreeLikelihood
        )

    def __calc_criterion(self):
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
            self.PriorOfInternalNode = 0
            self.ProbAttributeSelection = 0
        else:
            PriorOfInternalNode = 0
            for internalNode in self.internalNodes:
                PriorOfInternalNode += internalNode.PriorOfInternalNode
            self.PriorOfInternalNode = PriorOfInternalNode
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

    def fit(self):
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
        # In case if we have a new attribute for splitting
        Prob_KtPlusOne = (
            universal_code_natural_numbers(self.K_t + 1)
            - log_fact(self.K_t + 1)
            + (self.K_t + 1) * log(self.K)
        )
        ProbOfAttributeSelectionAmongSubsetAttributesPlusOne = log(self.K_t + 1) * (
            len(self.internalNodes) + 1
        )

        EncodingOfBeingAnInternalNodePlusOne = self.EncodingOfBeingAnInternalNode + log(
            2
        )

        # When splitting a node to 2 nodes, the number of leaf nodes is incremented only by one, since the parent node was leaf and is now internal.
        # 2 for two extra leaf nodes multiplied by 2 for W. Total = 4.
        EncodingOfBeingALeafNodeAndContainingTEPlusTWO = (
            self.EncodingOfBeingALeafNodeAndContainingTE + (2 * log(2))
        )

        EncodingOfInternalAndLeavesAndWWithExtraNodes = (
            EncodingOfBeingAnInternalNodePlusOne
            + EncodingOfBeingALeafNodeAndContainingTEPlusTWO
        )

        i = 0
        while True:
            NodeVsBestAttributeCorrespondingToTheBestCost = {}
            NodeVsBestCost = {}
            NodeVsCandidateSplitsCosts = (
                {}
            )  # Dictionary containing Nodes as key and their values are another dictionary each with attribute:CostSplit

            for terminalNode in self.terminalNodes:
                # This if condition is here to not to repeat calculations of candidate splits
                if terminalNode.CandidateSplitsVsCriterion == None:
                    NodeVsCandidateSplitsCosts[
                        terminalNode
                    ] = terminalNode.discretize_vars_and_get_attributes_splits_costs()
                else:
                    NodeVsCandidateSplitsCosts[
                        terminalNode
                    ] = terminalNode.CandidateSplitsVsCriterion.copy()

                if len(NodeVsCandidateSplitsCosts[terminalNode]) == 0:
                    continue

                # Update Costs
                for attribute in NodeVsCandidateSplitsCosts[terminalNode]:
                    if attribute in self.feature_subset:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute] += (
                            self.Prob_Kt
                            + self.ProbAttributeSelection
                            + EncodingOfInternalAndLeavesAndWWithExtraNodes
                            + self.LeafPrior
                            + self.TreeLikelihood
                            + self.PriorOfInternalNode
                        )
                    else:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute] += (
                            Prob_KtPlusOne
                            + EncodingOfInternalAndLeavesAndWWithExtraNodes
                            + ProbOfAttributeSelectionAmongSubsetAttributesPlusOne
                            + self.LeafPrior
                            + self.TreeLikelihood
                            + self.PriorOfInternalNode
                        )

                # Once costs are updated, I get the key of the minimal value split for terminalNode
                KeyOfTheMinimalVal = min(
                    NodeVsCandidateSplitsCosts[terminalNode],
                    key=NodeVsCandidateSplitsCosts[terminalNode].get,
                )

                NodeVsBestAttributeCorrespondingToTheBestCost[
                    terminalNode
                ] = KeyOfTheMinimalVal
                NodeVsBestCost[terminalNode] = NodeVsCandidateSplitsCosts[terminalNode][
                    KeyOfTheMinimalVal
                ]

            if len(list(NodeVsBestCost)) == 0:
                break

            OptimalNodeAttributeToSplitUp = min(NodeVsBestCost, key=NodeVsBestCost.get)
            OptimalVal = NodeVsBestCost[OptimalNodeAttributeToSplitUp]
            OptimalNode = OptimalNodeAttributeToSplitUp
            OptimalAttribute = NodeVsBestAttributeCorrespondingToTheBestCost[
                OptimalNodeAttributeToSplitUp
            ]

            if OptimalVal < self.TreeCriterion:
                self.TreeCriterion = OptimalVal
                if OptimalAttribute not in self.feature_subset:
                    self.feature_subset.append(OptimalAttribute)
                    self.K_t += 1
                NewLeftLeaf, NewRightLeaf = OptimalNode.perform_split(OptimalAttribute)
                self.terminalNodes.append(NewLeftLeaf)
                self.terminalNodes.append(NewRightLeaf)
                self.internalNodes.append(OptimalNode)
                self.terminalNodes.remove(OptimalNode)

                self.__calc_criterion()
            else:
                break
        print("Learning Finished")
        for node in self.terminalNodes:
            print("Node id ", node.id)
            print("Node outcomeProbInTrt ", node.outcomeProbInTrt)
            print("Node outcomeProbInCtrl ", node.outcomeProbInCtrl)
            print("self ntj ", node.Ntj)
        print("===============")

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
