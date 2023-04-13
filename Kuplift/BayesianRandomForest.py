######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
from math import log
import numpy as np
import random
from .HelperFunctions import (
    log_fact,
    universal_code_natural_numbers,
)
from .Tree import _Tree

random.seed(10)  # to decomment for tests


class _UpliftTreeClassifier(_Tree):
    """Private child class

    Parameters
    ----------
    Data_features : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series
        Treatment column.
    y_col : pd.Series
        Outcome column.
    """

    def __init__(self, Data_features, treatment_col, y_col):
        super().__init__(Data_features, treatment_col, y_col)

    def grow_tree(self):
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
                ListOfAttributeSplitsImprovingTreeCriterion = []
                for attribute in NodeVsCandidateSplitsCosts[terminalNode]:
                    if attribute in self.feature_subset:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute] += (
                            self.Prob_Kt
                            + self.ProbAttributeSelection
                            + EncodingOfInternalAndLeavesAndWWithExtraNodes
                            + self.LeafPrior
                            + self.TreeLikelihood
                            + self.PriorOfInternalNodes
                        )
                    else:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute] += (
                            Prob_KtPlusOne
                            + EncodingOfInternalAndLeavesAndWWithExtraNodes
                            + ProbOfAttributeSelectionAmongSubsetAttributesPlusOne
                            + self.LeafPrior
                            + self.TreeLikelihood
                            + self.PriorOfInternalNodes
                        )

                    if (
                        NodeVsCandidateSplitsCosts[terminalNode][attribute]
                        < self.TreeCriterion
                    ):
                        ListOfAttributeSplitsImprovingTreeCriterion.append(attribute)
                if len(ListOfAttributeSplitsImprovingTreeCriterion) == 0:
                    continue
                KeyOfTheMinimalVal = random.choice(
                    ListOfAttributeSplitsImprovingTreeCriterion
                )  # KeyOfTheMinimalVal is the attribute name

                NodeVsBestAttributeCorrespondingToTheBestCost[
                    terminalNode
                ] = KeyOfTheMinimalVal
                NodeVsBestCost[terminalNode] = NodeVsCandidateSplitsCosts[terminalNode][
                    KeyOfTheMinimalVal
                ]

            if len(list(NodeVsBestCost)) == 0:
                break
            OptimalNodeAttributeToSplitUp = random.choice(list(NodeVsBestCost))
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

                self.calc_criterion()
            else:
                print("WILL NEVER ENTER HERE")
                break


class BayesianRandomForest:
    """
    The BayesianRandomForest class implements the UB-RF algorithm described in:
    Rafla, M., Voisine, N., Crémilleux, B., \& Boullé, M. (2023, May). A Non-Parametric Bayesian Decision Trees for Uplift modelling. In PAKDD.

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe containing data.
    treatment_col : pd.Series
        Treatment column.
    outcome_col : pd.Series
        Outcome column.
    n_trees : int
        Number of trees in a forest.
    """

    def __init__(self, data, treatmentName, outcomeName, n_trees, NotAllVars=False):
        self.ListOfTrees = []
        self.data = data
        # Randomly select columns for the data
        if NotAllVars == True:
            cols = list(self.data.columns)
            cols.remove(treatmentName)
            cols.remove(outcomeName)
            print("cols before are ", cols)
            cols = random.sample(cols, int(np.sqrt(len(cols))))
            print("cols after are ", cols)
            self.data = self.data[cols + [treatmentName, outcomeName]]
        for i in range(n_trees):
            Tree = _UpliftTreeClassifier(self.data.copy(), treatmentName, outcomeName)
            self.ListOfTrees.append(Tree)

    def fit(self):
        """
        Fit a decision tree algorithm
        """
        for tree in self.ListOfTrees:
            tree.grow_tree()

    def predict(self, X_test):
        """
        Predict the uplift value for each example in X_test

        Parameters
        ----------
        X_test : pd.Dataframe
            Dataframe containing test data.

        Returns
        -------
        y_pred_list(ndarray, shape=(num_samples, 1))
            An array containing the predicted uplift for each sample.
        """
        ListOfPreds = []

        for tree in self.ListOfTrees:
            ListOfPreds.append(np.array(tree.predict(X_test)))
        return np.mean(ListOfPreds, axis=0)
