######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import pandas as pd
import multiprocessing as mp
from .HelperFunctions import preprocess_data
from .UMODL_SearchAlgorithm import execute_greedy_search_and_post_opt


class UnivariateEncoding:
    """
    The UnivariateEncoding class implements the UMODL algorithm for uplift data
    encoding described in:
    Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M.
    (2023, March). A non-parametric bayesian approach for uplift
    discretization and feature selection. ECML PKDD.
    """

    def __init__(self):
        self.var_vs_disc = {}
        self.treatment_col = ""
        self.y_col = ""

    def fit_transform(
        self, data, treatment_col, y_col, parallelized=False, num_processes=5
    ):
        """
        fit_transform() learns a discretisation model using UMODL and
        transforms the data.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        parallelized : bool, default False
            Whether to run the code on several processes.
        num_processes : int, default 5
            Number of processes to use in parallel.

        Returns
        -------
        pd.Dataframe
            Pandas Dataframe that contains encoded data.
        """
        self.fit(data, treatment_col, y_col, parallelized, num_processes)
        data = self.transform(data)
        return data

    def fit(
        self, data, treatment_col, y_col, parallelized=False, num_processes=5
    ):
        """
        fit() learns a discretisation model using the UMODL approach.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        parallelized : bool, default False
            Whether to run the code on several processes.
        num_processes : int, default 5
            Number of processes to use in parallel.
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

        if parallelized:
            pool = mp.Pool(processes=num_processes)

            arguments_to_pass_in_parallel = []
            for col in cols:
                arguments_to_pass_in_parallel.append(
                    [data[[col, treatment_col, y_col]]]
                )
            list_of_tuples_feature_vs_importance = pool.map(
                execute_greedy_search_and_post_opt,
                arguments_to_pass_in_parallel,
            )
            pool.close()

            for el in list_of_tuples_feature_vs_importance:
                col = el[0]
                if len(el[2]) == 1:
                    self.var_vs_disc[col] = None
                else:
                    self.var_vs_disc[col] = el[2][:-1]

        else:
            for col in cols:
                (
                    var_vs_importance[col],
                    self.var_vs_disc[col],
                ) = execute_greedy_search_and_post_opt(
                    data[[col, treatment_col, y_col]]
                )
                if len(self.var_vs_disc[col]) == 1:
                    self.var_vs_disc[col] = None
                else:
                    self.var_vs_disc[col] = self.var_vs_disc[col][:-1]
        return self.var_vs_disc

    def transform(self, data):
        """
        transform() applies the discretisation model learned by the
        fit() method.

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
                minBoundary = min(
                    data[col].min(), self.var_vs_disc[col][0] - 0.001
                )
                maxBoundary = max(
                    data[col].max(), self.var_vs_disc[col][-1] + 0.001
                )
                data[col] = pd.cut(
                    data[col],
                    bins=[minBoundary] + self.var_vs_disc[col] + [maxBoundary],
                )
                data[col] = data[col].astype("category")
                data[col] = data[col].cat.codes
        return data
