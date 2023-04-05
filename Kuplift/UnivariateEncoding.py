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
import pandas as pd
from HelperFunctions import preprocessData
from UMODL_SearchAlgorithm import ExecuteGreedySearchAndPostOpt 

class UnivariateEncoding:
    """Main class"""
    
    def __init__(self):
        self.VarVsDisc={}
    
    def fit_transform(self, Data_features, treatment_col , y_col):
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
        self.fit(self, Data_features, treatment_col , y_col)
        Data_features = self.transform(self, Data_features)
        return Data_features
    
    def fit(self, Data_features, treatment_col , y_col):
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
        cols=list(Data_features.columns)

        cols.remove(treatment_col)
        cols.remove(y_col)
        Data_features=Data_features[cols+[treatment_col,y_col]]
        
        Data_features=preprocessData(Data_features,treatment_col,y_col)
        
        VarVsImportance={}
        self.VarVsDisc={}

        for col in cols:
            VarVsImportance[col],self.VarVsDisc[col]=ExecuteGreedySearchAndPostOpt(Data_features[[col,treatment_col,y_col]])
            self.VarVsDisc[col]=self.VarVsDisc[col][:-1]
    
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
        cols=list(Data_features.columns)
        for col in cols:
            if len(self.VarVsDisc[col]) == 1:
                Data_features.drop(col,inplace=True,axis=1)
                Data_features.drop(col,inplace=True,axis=1)
            else:
                if Data_features[col].max()>self.VarVsDisc[col][-1]:
                    print("SOMETHING STRANGS IS HAPPENING max in train")
                Data_features[col] = pd.cut(Data_features[col], bins=[Data_features[col].min()-0.001]+self.VarVsDisc[col]+[Data_features[col].max()+0.001])

                Data_features[col] = Data_features[col].astype('category')
                Data_features[col] = Data_features[col].cat.codes
        return Data_features
    