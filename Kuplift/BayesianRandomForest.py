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

class BayesianRandomForest:
    """Main class
    
    Parameters
    ----------
    n_trees : int
        Number of trees in a forest.
    """
    def __init__(self, n_trees):
        self.n_trees = n_trees
    
    def fit(self, X_train, treatment_col, outcome_col):
        """Description?

        Parameters
        ----------
        X_train : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        outcome_col : pd.Series
            Outcome column.
        """
        return 0
    
    def predict(self, X_test):
        """Description?

        Parameters
        ----------
        X_train : pd.Dataframe
            Dataframe containing feature variables.
        
        Returns
        -------
        y_pred_list(ndarray, shape=(num_samples, 1))
            An array containing the predicted treatment uplift for each sample.
        """
        return X_test
