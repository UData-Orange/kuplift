######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Description?"""
import sys
from .HelperFunctions import preprocessData
from .UMODL_SearchAlgorithm import ExecuteGreedySearchAndPostOpt


class FeatureSelection:
    """Main class"""

    def __getTheBestVar(self, Data_features, features, treatment_col, y_col):
        """Description?

        Parameters
        ----------
        Data_features : pd.Dataframe
            Dataframe containing feature variables.
        features : ?
            ?
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.

        Returns
        -------
        Python Dictionary
            Keys are the variable names and the values are the variable importance (Sorted).

        For example: return a dictionary VarVsImportance={"age":2.2,"genre":2.3}
        """
        VarVsImportance = {}
        VarVsDisc = {}
        for feature in features:
            print("feature is ", feature)
            (
                VarVsImportance[feature],
                VarVsDisc[feature],
            ) = ExecuteGreedySearchAndPostOpt(
                Data_features[[feature, treatment_col, y_col]]
            )
        # sort the dictionary by values in ascending order
        VarVsImportance = {
            k: v for k, v in sorted(VarVsImportance.items(), key=lambda item: item[1])
        }
        return VarVsImportance

    def filter(self, Data_features, treatment_col, y_col):
        """Description?

        Parameters
        ----------
        Data_features : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.

        Returns
        -------
        Python Dictionary
            Variables names and their corresponding importance value (Sorted).
        """
        stdoutOrigin = sys.stdout
        sys.stdout = open("log.txt", "w")

        cols = list(Data_features.columns)

        cols.remove(treatment_col)
        cols.remove(y_col)
        Data_features = Data_features[cols + [treatment_col, y_col]]

        features = list(Data_features.columns[:-2])

        Data_features = preprocessData(Data_features, treatment_col, y_col)

        ListOfVarsImportance = self.__getTheBestVar(
            Data_features, features, treatment_col, y_col
        )

        sys.stdout.close()
        sys.stdout = stdoutOrigin

        return ListOfVarsImportance
