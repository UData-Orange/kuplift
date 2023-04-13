######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
from .BinaryDiscretizationFunctions import umodl_binary_discretization
from .HelperFunctions import (
    log_fact,
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
