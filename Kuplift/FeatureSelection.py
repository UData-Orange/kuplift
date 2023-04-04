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
from UMODL_SearchAlgorithm import ExecuteGreedySearchAndPostOpt 

class FeatureSelection:
    """Main class"""
    
    def __preprocessData(self, Data_features, treatment_col='segment', y_col='visit'):
        cols = Data_features.columns
        num_cols = list(Data_features._get_numeric_data().columns)

        num_cols.remove(treatment_col)
        num_cols.remove(y_col)
        for num_col in num_cols:
            if len(Data_features[num_col].value_counts())<(Data_features.shape[0]/100):
                num_cols.remove(num_col)
            else:
                Data_features[num_col] = Data_features[num_col].fillna(Data_features[num_col].mean())

        categoricalCols=list(set(cols) - set(num_cols))
        if treatment_col in categoricalCols:
            categoricalCols.remove(treatment_col)
        if y_col in categoricalCols:
            categoricalCols.remove(y_col)
        for catCol in categoricalCols:
            Data_features[catCol] = Data_features[catCol].fillna(Data_features[catCol].mode()[0])
            DictValVsUplift={}
            for val in Data_features[catCol].value_counts().index:
                dataset_slice=Data_features[Data_features[catCol]==val]
                t0j0=dataset_slice[(dataset_slice[treatment_col]==0)&(dataset_slice[y_col]==0)].shape[0]
                t0j1=dataset_slice[(dataset_slice[treatment_col]==0)&(dataset_slice[y_col]==1)].shape[0]
                t1j0=dataset_slice[(dataset_slice[treatment_col]==1)&(dataset_slice[y_col]==0)].shape[0]
                t1j1=dataset_slice[(dataset_slice[treatment_col]==1)&(dataset_slice[y_col]==1)].shape[0]

                if (t1j1+t1j0)==0:
                    UpliftInThisSlice=-1
                elif (t0j1+t0j1)==0:
                    UpliftInThisSlice=0
                else:
                    UpliftInThisSlice=(t1j1/(t1j1+t1j0))-(t0j1/(t0j1+t0j1))
                DictValVsUplift[val]=UpliftInThisSlice
            OrderedDict={k: v for k, v in sorted(DictValVsUplift.items(), key=lambda item: item[1])}
            encoded_i=0
            for k,v in OrderedDict.items():
                Data_features[catCol] = Data_features[catCol].replace([k],encoded_i)
                encoded_i+=1
        Data_features[treatment_col]=Data_features[treatment_col].astype(str)
        return Data_features
    
    # def __getImportantVariables_UMODL_ForMultiProcessing(self, data):
    #     featureImportanceAndBounds=ExecuteGreedySearchAndPostOpt(data)
    #     return featureImportanceAndBounds
    
    def __getTheBestVar(self, Data_features, features, treatment_col, y_col):
        '''
        return a dictionary where the keys are the variable names and the values are the variable importance
        for example: return a dictionary VarVsImportance={"age":2.2,"genre":2.3}
        '''
        VarVsImportance={}
        VarVsDisc={}
        for feature in features:
            print("feature is ",feature)
            # VarVsImportance[feature],VarVsDisc[feature]=self.__getImportantVariables_UMODL_ForMultiProcessing(Data_features[[feature,treatment_col,y_col]])
            VarVsImportance[feature],VarVsDisc[feature]=ExecuteGreedySearchAndPostOpt(Data_features[[feature,treatment_col,y_col]])
        
        # sort the dictionary by values in ascending order 
        VarVsImportance={k: v for k, v in sorted(VarVsImportance.items(), key=lambda item: item[1])}
        return VarVsImportance

    def filter(self, Data_features, treatment_col , y_col):
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
        stdoutOrigin=sys.stdout
        sys.stdout = open("log.txt", "w")

        cols=list(Data_features.columns)

        cols.remove(treatment_col)
        cols.remove(y_col)
        Data_features=Data_features[cols+[treatment_col,y_col]]

        features=list(Data_features.columns[:-2])

        Data_features=self.__preprocessData(Data_features,treatment_col,y_col)
        
        ListOfVarsImportance=self.__getTheBestVar(Data_features,features,treatment_col,y_col)

        sys.stdout.close()
        sys.stdout=stdoutOrigin
    
        return ListOfVarsImportance
