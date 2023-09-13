######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import numpy as np
import pandas as pd
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

    def __init__(self,control_name=None):  # ordered data as argument
        self.nodes_ids = None
        self.root_node = None
        self.terminal_nodes = []
        self.internal_nodes = []

        self.k = None
        self.k_t = None
        self.features = None
        self.feature_subset = None

        self.prob_kt = None
        self.encoding_of_being_an_internal_node = None
        self.prob_attribute_selection = None
        self.prior_of_internal_nodes = None
        self.encoding_of_being_a_leaf_node_and_containing_te = None
        self.leaf_prior = None
        self.tree_likelihood = None
        
        self.summary_df = None
        
        self.treatment_name='treatment'
        self.outcome_name='outcome'
        self.control_name=control_name
        
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
            self.prob_attribute_selection = log(self.k_t) * len(
                self.internal_nodes
            )

    def __calc_encoding(self):
        self.encoding_of_being_a_leaf_node_and_containing_te = (
            len(self.terminal_nodes) * log(2) * 2
        )
        self.encoding_of_being_an_internal_node = len(
            self.internal_nodes
        ) * log(2)

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
        if node.is_leaf:
            return node.average_uplift

        if x[node.attribute] <= node.split_threshold:
            return self.__traverse_tree(x, node.left_node)
        return self.__traverse_tree(x, node.right_node)
    
    def __initializeVars__(self, data, treatment_col, y_col):
        data = data.assign(**{self.treatment_name: treatment_col.copy()})
        data = data.assign(**{self.outcome_name: y_col.copy()})
        
        #dealing with control name
        if self.control_name != None:
            trt_vals=list(data[self.treatment_name].unique())
            
            #Verify that control name is in the treatment column, else raise an exception
            if self.control_name not in trt_vals:
                raise Exception("the control name is not in the treatment column")
            data[self.treatment_name] = data[self.treatment_name].replace(self.control_name,0)

            trt_vals.remove(self.control_name)
            #the other value will be considered as the treatment
            data[self.treatment_name] = data[self.treatment_name].replace(trt_vals[0],1)
        
        self.nodes_ids = 0
        self.root_node = _Node(
            data, self.treatment_name, self.outcome_name, ID=self.nodes_ids + 1
        )
        self.terminal_nodes = [self.root_node]
        self.internal_nodes = []

        self.k = len(list(data.columns))
        self.k_t = 1
        self.features = list(data.columns)
        self.feature_subset = []

        self.encoding_of_being_a_leaf_node_and_containing_te = (
            len(self.terminal_nodes) * log(2) * 2
        )  # TE=TreatmentEffect
        
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

    def getSummary(self):
        summary_df = pd.DataFrame(
            columns=[
                "NodeId",
                "is_leaf",
                "T0Y0",
                "T0Y1",
                "T1Y0",
                "T1Y1",
                "Uplift",
                "SplittedAttribute",
                "split_threshold",
            ]
        )  # split_threshold
        for internalNode in self.internal_nodes:
            summary_df.loc[len(summary_df.index)] = [
                internalNode.id,
                internalNode.is_leaf,
                internalNode.ntj[0],
                internalNode.ntj[1],
                internalNode.ntj[2],
                internalNode.ntj[3],
                internalNode.average_uplift,
                internalNode.attribute,
                internalNode.split_threshold,
            ]
        for terminalNode in self.terminal_nodes:
            summary_df.loc[len(summary_df.index)] = [
                terminalNode.id,
                terminalNode.is_leaf,
                terminalNode.ntj[0],
                terminalNode.ntj[1],
                terminalNode.ntj[2],
                terminalNode.ntj[3],
                terminalNode.average_uplift,
                terminalNode.attribute,
                terminalNode.split_threshold,
            ]
        self.summary_df = summary_df
        return self.summary_df

    def export_tree(self, IdValue=1, numTabs=0, text_desc=""):
        def create_tabs(txt, numTabs):
            for numTab in range(numTabs):
                txt += "\t"
            return txt

        # fill the summary dataframe of the tree
        if IdValue == 1:
            self.getSummary()

        row = (
            self.summary_df[self.summary_df["NodeId"] == IdValue]
            .iloc[:1]
            .reset_index(drop=True)
            .squeeze()
        )
        #     print("row is ",type(row))
        #     print("row is ",row)
        if row["is_leaf"] == False:
            #         print(" id ",str(IdValue)," not leaf")
            text_desc = create_tabs(text_desc, numTabs)
            text_desc = (
                text_desc
                + "|--- "
                + " "
                + str(row["SplittedAttribute"])
                + " <= "
                + str(row["split_threshold"])
                + "\n"
            )
            #         print(text_desc)
            text_desc = self.export_tree(IdValue * 2, numTabs + 1, text_desc)

            text_desc = create_tabs(text_desc, numTabs)
            text_desc = (
                text_desc
                + "|--- "
                + " "
                + str(row["SplittedAttribute"])
                + " > "
                + str(row["split_threshold"])
                + "\n"
            )
            text_desc = self.export_tree(
                IdValue * 2 + 1, numTabs + 1, text_desc
            )
        else:
            text_desc = create_tabs(text_desc, numTabs)
            text_desc += "|--- Leaf \n"
            text_desc = create_tabs(text_desc, numTabs + 1)
            try:
                text_desc = (
                    text_desc
                    + "|--- "
                    + " Outcome Distribution in Treatment "
                    + str(row["T1Y1"] / (row["T1Y1"] + row["T1Y0"]))
                    + "\n"
                )
            except:
                text_desc = (
                    text_desc
                    + "|--- "
                    + " Outcome Distribution in Treatment "
                    + str(row["T1Y1"] / 0.0001)
                    + "(No treatment)\n"
                )

            text_desc = create_tabs(text_desc, numTabs + 1)
            try:
                text_desc = (
                    text_desc
                    + "|--- "
                    + " Outcome Distribution in Control "
                    + str(row["T0Y1"] / (row["T0Y1"] + row["T0Y0"]))
                    + "\n"
                )
            except:
                text_desc = (
                    text_desc
                    + "|--- "
                    + " Outcome Distribution in Control "
                    + str(row["T0Y1"] / 0.001)
                    + " (No control)\n"
                )

        return text_desc
