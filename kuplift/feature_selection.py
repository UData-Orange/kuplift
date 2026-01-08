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
from .helperfunctions import preprocess_data
from .umodl_search_algorithm import execute_greedy_search_and_post_opt
import pandas as pd


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
    def __init__(self,control_name=None):
        self.treatment_name = "treatment"
        self.outcome_name = "outcome"
        self.control_name=control_name
        self.features_importance_details={}
        self.dict_vars_vs_importance={}
    
    def get_features_importance_details(self):
        '''
        After launch the feature selection approach, this function helps getting the details of a each feature. 
        How it was discretized, the intervals, the outcome denisities in each interval.
        '''
        
        # Prepare lists to capture multi-index levels
        outer_keys = []
        inner_keys = []
        values = []
        
        dictionary_of_dataframes={}
        # Loop through the dictionary to populate the lists
        for key, sublist in self.features_importance_details.items():
            for idx, val in enumerate(sublist):
                outer_keys.append(key)
                inner_keys.append(idx)
                values.append(val)
        
        # Create the multi-index
        multi_index = pd.MultiIndex.from_tuples(list(zip(outer_keys, inner_keys)), names=["Variable", "Interval"])
        
        # Convert the lists to DataFrame
        df = pd.DataFrame(values, columns=["interval", "P(Y|T=1)", "P(Y|T=0)", "uplift"], index=multi_index)
        
        df["Imp. Score"] = df.index.get_level_values(0).map(self.dict_vars_vs_importance)
        
        # Get sorted unique outer indices based on Imp. Score
        unique_outer_indices = df.groupby(level=0)['Imp. Score'].first().sort_values(ascending=False).index

        # Reindex the dataframe to rearrange based on the sorted outer indices
        df_sorted = df.reindex(unique_outer_indices, level=0)
        
        return df_sorted
    
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
        features.remove(self.treatment_name)
        features.remove(self.outcome_name)

        var_vs_importance = {}
        var_vs_disc = {}
        for feature in features:
            (
                var_vs_importance[feature],
                var_vs_disc[feature],
                col_name,
                self.features_importance_details[feature]
            ) = execute_greedy_search_and_post_opt(
                data[[feature, self.treatment_name, self.outcome_name]],get_intervals_uplift=True
            )
        # sort the dictionary by values in ascending order
        var_vs_importance = {
            k: v
            for k, v in sorted(
                var_vs_importance.items(), key=lambda item: item[1]
            )
        }
        return var_vs_importance

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
        parallelized : bool, default False
            Whether to run the code on several processes.
        num_processes : int, default 5
            Number of processes to use in parallel, 'parallelized' argument should be True.

        Returns
        -------
        Python Dictionary
            Variables names and their corresponding importance value (Sorted).
        """
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

        
        cols = list(data.columns)
        cols.remove(self.treatment_name)
        cols.remove(self.outcome_name)

        data = data[cols + [self.treatment_name, self.outcome_name]]
        data = preprocess_data(data, self.treatment_name, self.outcome_name)
        
        self.var_vs_disc = {}
        
        if parallelized:
            pool = mp.Pool(processes=num_processes)

            arguments_to_pass_in_parallel = []
            for col in cols:
                arguments_to_pass_in_parallel.append(
                    [data[[col, self.treatment_name, self.outcome_name]],True]
                )
            list_of_tuples_feature_vs_importance = pool.starmap(
                execute_greedy_search_and_post_opt,
                arguments_to_pass_in_parallel
            )
            pool.close()

            for el in list_of_tuples_feature_vs_importance:
                col = el[2]
                self.features_importance_details[col]=el[3]
                if len(el[1]) == 1:
                    self.var_vs_disc[col] = 0
                else:
                    self.var_vs_disc[col] = el[0]
            self.var_vs_disc = {
                k: v
                for k, v in sorted(
                    self.var_vs_disc.items(), key=lambda item: item[1]
                )
            }
            self.dict_vars_vs_importance=self.var_vs_disc.copy()
            return self.var_vs_disc
            

        else:
            list_of_vars_importance = self.__get_the_best_var(
                data, self.treatment_name, self.outcome_name
            )
            self.dict_vars_vs_importance=list_of_vars_importance.copy()
        
        return list_of_vars_importance
