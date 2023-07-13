######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import multiprocessing as mp
from .HelperFunctions import preprocess_data
from .UMODL_SearchAlgorithm import execute_greedy_search_and_post_opt


class FeatureSelection:
    """
    The FeatureSelection implements the feature selection algorithm 'UMODL-FS'
    described in: Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M.
    (2023, March). A non-parametric bayesian approach for uplift
    discretization and feature selection.
    In Machine Learning and Knowledge Discovery in Databases:
    European Conference, ECML PKDD 2022, Grenoble, France,
    September 19–23, 2022, Proceedings, Part V (pp. 239-254).
    Cham: Springer Nature Switzerland.
    """

    def __get_the_best_var(self, data, treatment_col, y_col):
        """
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
            A Python dictionary containing the sorted variable importance,
            where the keys represent the variable names and the values denote
            their respective importance.

        For example: return a dictionary
                    var_vs_importance={"age":2.2,"job":2.3}
        """
        features = list(data.columns)
        features.remove(treatment_col)
        features.remove(y_col)

        var_vs_importance = {}
        var_vs_disc = {}
        for feature in features:
            print("feature is ", feature)
            (
                var_vs_importance[feature],
                var_vs_disc[feature],
            ) = execute_greedy_search_and_post_opt(
                data[[feature, treatment_col, y_col]]
            )
        # sort the dictionary by values in ascending order
        var_vs_importance = {
            k: v
            for k, v in sorted(
                var_vs_importance.items(), key=lambda item: item[1]
            )
        }
        return var_vs_importance

    @staticmethod
    def get_the_best_var_parallel(args):
        """
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
            A Python dictionary containing the sorted variable importance,
            where the keys represent the variable names and the values denote
            their respective importance.

        For example: return a dictionary
                    var_vs_importance={"age":2.2,"job":2.3}
        """
        data, treatment_col, y_col = args[0], args[1], args[2]

        features = list(data.columns)
        feature = features[0]
        features.remove(treatment_col)
        features.remove(y_col)
        print("feature is ", feature)
        var_vs_importance = {}
        var_vs_disc = {}
        (
            var_vs_importance[feature],
            var_vs_disc[feature],
        ) = execute_greedy_search_and_post_opt(
            data[[feature, treatment_col, y_col]]
        )
        return (feature, var_vs_importance[feature])

    def filter(
        self, data, treatment_col, y_col, parallelized=False, num_processes=5
    ):
        """
        This function runs the feature selection algorithm 'UMODL-FS',
        ranking variables based on their importance in the given data.

        Parameters
        ----------
        data : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        parallelized : Boolean
            Whether to run the code on several processes (default = 5)
        num_processes : int
            number of processes to use in parallel (default = 5)

        Returns
        -------
        Python Dictionary
            Variables names and their corresponding importance value (Sorted).
        """
        cols = list(data.columns)

        cols.remove(treatment_col)
        cols.remove(y_col)
        data = data[cols + [treatment_col, y_col]]
        data = preprocess_data(data, treatment_col, y_col)

        if parallelized == True:
            pool = mp.Pool(processes=num_processes)

            arguments_to_pass_in_parallel = []
            for col in cols:
                arguments_to_pass_in_parallel.append(
                    [data[[col, treatment_col, y_col]], treatment_col, y_col]
                )
            list_of_tuples_feature_vs_importance = pool.map(
                FeatureSelection.get_the_best_var_parallel,
                arguments_to_pass_in_parallel,
            )
            pool.close()

            # transform tuple to dict
            list_of_tuples_feature_vs_importance = dict(
                list_of_tuples_feature_vs_importance
            )

            list_of_vars_importance = {
                k: v
                for k, v in sorted(
                    list_of_tuples_feature_vs_importance.items(),
                    key=lambda item: item[1],
                )
            }

        else:
            list_of_vars_importance = self.__get_the_best_var(
                data, treatment_col, y_col
            )

        return list_of_vars_importance
