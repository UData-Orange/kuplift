######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
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
    data : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series
        Treatment column.
    y_col : pd.Series
        Outcome column.
    ID : int, default None
        Tree node ID.
    """

    def __init__(self, data, treatment_col, y_col, ID=None):
        # Initialize attributes
        self.id = ID
        self.treatment = treatment_col
        self.output = y_col
        self.n = data.shape[0]
        self.nj = data[data[self.output] == 1].shape[0]
        self.ntj = [
            data[(data[self.treatment] == 0) & (data[self.output] == 0)].shape[
                0
            ],
            data[(data[self.treatment] == 0) & (data[self.output] == 1)].shape[
                0
            ],
            data[(data[self.treatment] == 1) & (data[self.output] == 0)].shape[
                0
            ],
            data[(data[self.treatment] == 1) & (data[self.output] == 1)].shape[
                0
            ],
        ]
        self.x = data.iloc[:, :-2].copy()
        self.t = data.iloc[:, -2].copy()
        self.y = data.iloc[:, -1].copy()

        try:
            if (self.ntj[2] + self.ntj[3]) == 0:
                denum = 0.00001
            else:
                denum = self.ntj[2] + self.ntj[3]
            self.outcome_prob_in_trt = self.ntj[3] / denum
        except Exception:
            self.outcome_prob_in_trt = 0

        try:
            if (self.ntj[0] + self.ntj[1]) == 0:
                denum = 0.00001
            else:
                denum = self.ntj[0] + self.ntj[1]
            self.outcome_prob_in_ctrl = self.ntj[1] / denum
        except Exception:
            self.outcome_prob_in_ctrl = 0

        self.average_uplift = (
            self.outcome_prob_in_trt - self.outcome_prob_in_ctrl
        )
        self.attribute = None
        self.split_threshold = None
        self.is_leaf = True
        self.candidate_splits_vs_data_left_data_right = None
        self.candidate_splits_vs_criterion = None
        self.left_node = None
        self.right_node = None
        self.prior_of_internal_node = self.__calc_prior_of_internal_node()
        (
            self.prior_leaf,
            self.likelihood_leaf,
            self.w,
        ) = self.__calc_prior_and_likelihood_leaf()

    def __calc_prior_of_internal_node(self):
        return log_binomial_coefficient(sum(self.ntj) + 1, 1)

    def __calc_prior_and_likelihood_leaf(self):
        number_of_treatment = self.ntj[2] + self.ntj[3]
        number_of_control = self.ntj[0] + self.ntj[1]
        number_of_pos_outcome = self.ntj[1] + self.ntj[3]
        number_of_zero_outcome = self.ntj[0] + self.ntj[2]

        leaf_prior_w_zero = log_binomial_coefficient(sum(self.ntj) + 1, 1)
        tree_likelihood_w_zero = (
            log_fact(sum(self.ntj))
            - log_fact(number_of_pos_outcome)
            - log_fact(number_of_zero_outcome)
        )

        leaf_prior_w_one = log_binomial_coefficient(
            number_of_treatment + 1, 1
        ) + log_binomial_coefficient(number_of_control + 1, 1)
        tree_likelihood_w_one = (
            log_fact(number_of_treatment)
            - log_fact(self.ntj[2])
            - log_fact(self.ntj[3])
        ) + (
            log_fact(number_of_control)
            - log_fact(self.ntj[0])
            - log_fact(self.ntj[1])
        )

        if (leaf_prior_w_zero + tree_likelihood_w_zero) < (
            leaf_prior_w_one + tree_likelihood_w_one
        ):
            w = 0
            leaf_prior = leaf_prior_w_zero
            tree_likelihood = tree_likelihood_w_zero
        else:
            w = 1
            leaf_prior = leaf_prior_w_one
            tree_likelihood = tree_likelihood_w_one
        return leaf_prior, tree_likelihood, w

    def discretize_vars_and_get_attributes_splits_costs(self):
        """For this node loop on all attributes
        and get the optimal split for each one.

        Returns
        -------
        Dictionary of lists

        For example: return a dictionnary {age: Cost, sex: Cost}
        The cost here corresponds to
        1- the cost of this node to internal instead of leaf
            (criterion_to_be_internal-prior_leaf)
        2- The combinatorial terms of the leaf prior and likelihood
        """
        features = list(self.x.columns)
        attribute_to_split_vs_left_and_right_data = {}
        for attribute in features:
            if (
                len(self.x[attribute].value_counts()) == 1
                or len(self.x[attribute].value_counts()) == 0
            ):
                continue
            disc_res = umodl_binary_discretization(
                self.x, self.t, self.y, attribute
            )
            if disc_res == -1:
                continue
            data_left, data_right, threshold = (
                disc_res[0],
                disc_res[1],
                disc_res[2],
            )
            attribute_to_split_vs_left_and_right_data[attribute] = [
                data_left,
                data_right,
                threshold,
            ]

        self.candidate_splits_vs_data_left_data_right = (
            attribute_to_split_vs_left_and_right_data.copy()
        )
        candidate_splits_vs_criterion = self.__get_attributes_splits_costs(
            attribute_to_split_vs_left_and_right_data
        )
        self.candidate_splits_vs_criterion = (
            candidate_splits_vs_criterion.copy()
        )
        return candidate_splits_vs_criterion.copy()

    def __get_attributes_splits_costs(self, dict_of_each_att_vs_effectifs):
        # Prior of Internal node is only the combinatorial calculations
        # In case we split this node, it will be no more a leaf but an internal
        # node
        criterion_to_be_internal = self.__calc_prior_of_internal_node()
        new_prior_vals = (
            criterion_to_be_internal - self.prior_leaf - self.likelihood_leaf
        )

        candidate_splits_vs_criterion = {}
        for key in dict_of_each_att_vs_effectifs:
            leaves_val = self.__update_tree_criterion(
                dict_of_each_att_vs_effectifs[key][:2]
            )
            candidate_splits_vs_criterion[key] = new_prior_vals + leaves_val
        return candidate_splits_vs_criterion.copy()

    def __update_tree_criterion(self, left_and_right_data):
        leaves_vals = 0
        for (
            new_node_effectifs
        ) in left_and_right_data:  # Loop on Left and Right candidate nodes
            node = _Node(new_node_effectifs, self.treatment, self.output)
            leaves_vals += node.prior_leaf + node.likelihood_leaf
            del node
        return leaves_vals

    def perform_split(self, attribute):
        if self.candidate_splits_vs_data_left_data_right is None:
            raise
        else:
            self.is_leaf = False
            self.left_node = _Node(
                self.candidate_splits_vs_data_left_data_right[attribute][0],
                self.treatment,
                self.output,
                ID=self.id * 2,
            )
            self.right_node = _Node(
                self.candidate_splits_vs_data_left_data_right[attribute][1],
                self.treatment,
                self.output,
                ID=self.id * 2 + 1,
            )
            self.attribute = attribute
            self.split_threshold = (
                self.candidate_splits_vs_data_left_data_right[attribute][2]
            )
            del self.x
            del self.t
            del self.y
            del self.candidate_splits_vs_data_left_data_right

            return self.left_node, self.right_node
