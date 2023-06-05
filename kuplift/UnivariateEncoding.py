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
    """
    The UnivariateEncoding class implements the UMODL algorithm for uplift data encoding described in:
    Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M.
    (2023, March). A non-parametric bayesian approach for uplift
    discretization and feature selection. ECML PKDD
    """

    def __init__(self):
        self.var_vs_disc = {}
        self.treatment_col = ""
        self.y_col = ""

    def fit_transform(self, data, treatment_col, y_col):
        """
        fit_transform() learns a discretisation model using UMODL and transforms the data.

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
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, y_col)
        data = self.transform(data)
        return data

    def fit(self, data, treatment_col, y_col):
        """
         fit() learns a discretisation model using the UMODL approach

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        """
        self.treatment_col = treatment_col
        self.y_col = y_col

        cols = list(data.columns)
        cols.remove(treatment_col)
        cols.remove(y_col)

        data = data[cols + [treatment_col, y_col]]
        data = preprocess_data(data, treatment_col, y_col)

        var_vs_importance = {}
        self.var_vs_disc = {}

        for col in cols:
            (
                var_vs_importance[col],
                self.var_vs_disc[col],
            ) = execute_greedy_search_and_post_opt(data[[col, treatment_col, y_col]])
            if len(self.var_vs_disc[col]) == 1:
                self.var_vs_disc[col] = None
            else:
                self.var_vs_disc[col] = self.var_vs_disc[col][:-1]

    def transform(self, data):
        """
        transform() applies the discretisation model learned by the fit() method

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        cols = list(data.columns)
        cols.remove(self.treatment_col)
        cols.remove(self.y_col)
        for col in cols:
            if self.var_vs_disc[col] is None:
                data.drop(col, inplace=True, axis=1)
            else:
                data[col] = pd.cut(
                    data[col],
                    bins=[data[col].min() - 0.001]
                    + self.var_vs_disc[col]
                    + [data[col].max() + 0.001],
                )
                data[col] = data[col].astype("category")
                data[col] = data[col].cat.codes
        return data
