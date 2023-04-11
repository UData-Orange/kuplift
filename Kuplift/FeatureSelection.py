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
    """
    The FeatureSelection implements the feature selection algorithm 'UMODL-FS' described in:
    Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2023, March). A non-parametric bayesian approach for uplift discretization and feature selection. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2022, Grenoble, France, September 19–23, 2022, Proceedings, Part V (pp. 239-254). Cham: Springer Nature Switzerland.
    """

    def __getTheBestVar(self, data, treatment_col, y_col):
        """Description?

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing data.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.

        Returns
        -------
        dict
            A Python dictionary containing the sorted variable importance, where the keys represent the variable names and the values denote their respective importance.

        For example: return a dictionary VarVsImportance={"age":2.2,"job":2.3}
        """
        features=list(data.columns)
        features.remove(treatment_col)
        features.remove(y_col)
        
        VarVsImportance = {}
        VarVsDisc = {}
        for feature in features:
            print("feature is ", feature)
            (
                VarVsImportance[feature],
                VarVsDisc[feature],
            ) = ExecuteGreedySearchAndPostOpt(
                data[[feature, treatment_col, y_col]]
            )
        # sort the dictionary by values in ascending order
        VarVsImportance = {
            k: v for k, v in sorted(VarVsImportance.items(), key=lambda item: item[1])
        }
        return VarVsImportance

    def filter(self, data, treatment_col, y_col):
        """
        This function runs the feature selection algorithm 'UMODL-FS', ranking variables based on their importance in the given data.

        Parameters
        ----------
        data : pd.Dataframe
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

        cols = list(data.columns)

        cols.remove(treatment_col)
        cols.remove(y_col)
        data = data[cols + [treatment_col, y_col]]

        features = list(data.columns[:-2])

        data = preprocessData(data, treatment_col, y_col)

        ListOfVarsImportance = self.__getTheBestVar(
            data, treatment_col, y_col
        )

        sys.stdout.close()
        sys.stdout = stdoutOrigin

        return ListOfVarsImportance
