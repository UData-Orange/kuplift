######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import pandas as pd
from .HelperFunctions import preprocess_data
from .UMODL_SearchAlgorithm import execute_greedy_search_and_post_opt


class UnivariateEncoding:
    """Description ?"""

    def __init__(self):
        self.var_vs_disc = {}
        self.treatment_col = ""
        self.y_col = ""

    def fit_transform(self, Data_features, treatment_col, y_col):
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
        pd.Dataframe
            Pandas Dataframe that contains encoded Data_features.
        """
        self.fit(Data_features, treatment_col, y_col)
        Data_features = self.transform(Data_features)
        return Data_features

    def fit(self, Data_features, treatment_col, y_col):
        """Description?

        Parameters
        ----------
        Data_features : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        """
        self.treatment_col = treatment_col
        self.y_col = y_col

        cols = list(Data_features.columns)
        cols.remove(treatment_col)
        cols.remove(y_col)

        Data_features = Data_features[cols + [treatment_col, y_col]]
        Data_features = preprocess_data(Data_features, treatment_col, y_col)

        var_vs_importance = {}
        self.var_vs_disc = {}

        for col in cols:
            (
                var_vs_importance[col],
                self.var_vs_disc[col],
            ) = execute_greedy_search_and_post_opt(
                Data_features[[col, treatment_col, y_col]]
            )
            if len(self.var_vs_disc[col]) == 1:
                self.var_vs_disc[col] = None
            else:
                self.var_vs_disc[col] = self.var_vs_disc[col][:-1]

    def transform(self, Data_features):
        """Description?

        Parameters
        ----------
        Data_features : pd.Dataframe
            Dataframe containing feature variables.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded Data_features.
        """
        cols = list(Data_features.columns)
        cols.remove(self.treatment_col)
        cols.remove(self.y_col)
        for col in cols:
            if self.var_vs_disc[col] is None:
                Data_features.drop(col, inplace=True, axis=1)
            else:
                Data_features[col] = pd.cut(
                    Data_features[col],
                    bins=[Data_features[col].min() - 0.001]
                    + self.var_vs_disc[col]
                    + [Data_features[col].max() + 0.001],
                )
                Data_features[col] = Data_features[col].astype("category")
                Data_features[col] = Data_features[col].cat.codes
        return Data_features
