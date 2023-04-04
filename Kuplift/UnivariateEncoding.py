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
from UMODL_SearchAlgorithm import ExecuteGreedySearchAndPostOpt 

class UnivariateEncoding:
    """Main class"""
    
    def __preprocessData(self, df,treatmentName='segment',outcomeName='visit'):
        cols = df.columns
        num_cols = list(df._get_numeric_data().columns)

        num_cols.remove(treatmentName)
        num_cols.remove(outcomeName)
        for num_col in num_cols:
            if len(df[num_col].value_counts())<(df.shape[0]/100):
    #             print("categorical columns disguised in a numerical column")
                num_cols.remove(num_col)
            else:
                df[num_col] = df[num_col].fillna(df[num_col].mean())

        categoricalCols=list(set(cols) - set(num_cols))
        if treatmentName in categoricalCols:
            categoricalCols.remove(treatmentName)
        if outcomeName in categoricalCols:
            categoricalCols.remove(outcomeName)
    #     print("Categorical variables are  ",categoricalCols)
        for catCol in categoricalCols:
    #         print("Encoding ",catCol)
            df[catCol] = df[catCol].fillna(df[catCol].mode()[0])
            DictValVsUplift={}
            for val in df[catCol].value_counts().index:
                dataset_slice=df[df[catCol]==val]
                t0j0=dataset_slice[(dataset_slice[treatmentName]==0)&(dataset_slice[outcomeName]==0)].shape[0]
                t0j1=dataset_slice[(dataset_slice[treatmentName]==0)&(dataset_slice[outcomeName]==1)].shape[0]
                t1j0=dataset_slice[(dataset_slice[treatmentName]==1)&(dataset_slice[outcomeName]==0)].shape[0]
                t1j1=dataset_slice[(dataset_slice[treatmentName]==1)&(dataset_slice[outcomeName]==1)].shape[0]

                if (t1j1+t1j0)==0:
                    UpliftInThisSlice=-1
                elif (t0j1+t0j1)==0:
                    UpliftInThisSlice=0
                else:
                    UpliftInThisSlice=(t1j1/(t1j1+t1j0))-(t0j1/(t0j1+t0j1))
                DictValVsUplift[val]=UpliftInThisSlice
            # print("DictValVsUplift")
            # print(DictValVsUplift)
            OrderedDict={k: v for k, v in sorted(DictValVsUplift.items(), key=lambda item: item[1])}
            encoded_i=0
            for k,v in OrderedDict.items():
                df[catCol] = df[catCol].replace([k],encoded_i)
                encoded_i+=1
    #     print("df after encoding categorical variables is ",df)
        df[treatmentName]=df[treatmentName].astype(str)
        return df

    
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
        cols=list(Data_features.columns)

        cols.remove(treatment_col)
        cols.remove(y_col)
        Data_features=Data_features[cols+[treatment_col,y_col]]
        
        Data_features=self.__preprocessData(Data_features,treatment_col,y_col)
        
        VarVsImportance={}
        VarVsDisc={}

        for col in cols:
            VarVsImportance[col],VarVsDisc[col]=ExecuteGreedySearchAndPostOpt(Data_features[[col,treatment_col,y_col]])
        
        for col in cols:
            if len(VarVsDisc[col]) == 1:
                Data_features.drop(col,inplace=True,axis=1)
                Data_features.drop(col,inplace=True,axis=1)
            else:
                if Data_features[col].max()>VarVsDisc[col][-1]:
                    print("SOMETHING STRANGS IS HAPPENING max in train")
                Data_features[col] = pd.cut(Data_features[col], bins=[Data_features[col].min()-0.001]+VarVsDisc[col])

                Data_features[col] = Data_features[col].astype('category')
                Data_features[col] = Data_features[col].cat.codes
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
        return 0
    
    def transform(X):
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
        return X
