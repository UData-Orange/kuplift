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
from .HelperFunctions import (
    log_fact,
    universal_code_natural_numbers,
)
from .Tree import _Tree


class BayesianDecisionTree(_Tree):
    """
    The BayesianDecisionTree class implements the UB-DT algorithm described in:
    Rafla, M., Voisine, N., Crémilleux, B., \& Boullé, M. (2023, May). A Non-Parametric Bayesian Decision Trees for Uplift modelling. In PAKDD.

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

                self.calc_criterion()
            else:
                break
        print("Learning Finished")
        for node in self.terminalNodes:
            print("Node id ", node.id)
            print("Node outcomeProbInTrt ", node.outcomeProbInTrt)
            print("Node outcomeProbInCtrl ", node.outcomeProbInCtrl)
            print("self ntj ", node.Ntj)
        print("===============")
