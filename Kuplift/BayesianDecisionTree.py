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
import numpy as np
from math import log
from HelperFunctions import log_fact, universal_code_natural_numbers, log_binomial_coefficient

#TODO UMODL_Discretizer !

class _Interval:
    def __init__(self, data,treatmentName,outcomeName,ID=None):
        self.id=ID
        self.data=data.copy() #ordered data
        self.treatment=treatmentName
        self.output=outcomeName
#         print("self.data is ",self.data.head())
        self.N=data.shape[0]
        self.Nj = data[data[self.output]==1].shape[0] 
        self.Ntj=[data[(data[self.treatment]==0)&(data[self.output]==0)].shape[0],
                 data[(data[self.treatment]==0)&(data[self.output]==1)].shape[0],
                 data[(data[self.treatment]==1)&(data[self.output]==0)].shape[0],
                 data[(data[self.treatment]==1)&(data[self.output]==1)].shape[0]]
        
        self.X=data.iloc[:,:-2].copy()
        self.T=data.iloc[:,-2].copy()
        self.Y=data.iloc[:,-1].copy()
        
        
        
        try:
            if (self.Ntj[2]+self.Ntj[3])==0:
                denum=0.00001
            else:
                denum=(self.Ntj[2]+self.Ntj[3])
            self.outcomeProbInTrt=(self.Ntj[3]/denum)
        except:
            self.outcomeProbInTrt=0
        try:
            if (self.Ntj[0]+self.Ntj[1])==0:
                denum=0.00001
            else:
                denum=self.Ntj[0]+self.Ntj[1]
            self.outcomeProbInCtrl=(self.Ntj[1]/denum)
        except:
            self.outcomeProbInCtrl=0
        self.averageUplift=self.outcomeProbInTrt-self.outcomeProbInCtrl
        
        self.Attribute=None
        self.SplitThreshold=None
        self.isLeaf=True
        
        self.CandidateSplitsVsDataLeftDataRight=None
        self.CandidateSplitsVsCriterion=None
        
        self.leftInterval=None
        self.rightInterval=None
        
        self.PriorOfInternalInterval=self.calcPriorOfInternalInterval()
        self.PriorLeaf,self.LikelihoodLeaf,self.W=self.calcPriorAndLikelihoodLeaf()
    
    def calcPriorOfInternalInterval(self):
        return log_binomial_coefficient(sum(self.Ntj)+1,1)
    
    def calcPriorAndLikelihoodLeaf(self):
        NumberOfTreatment=self.Ntj[2]+self.Ntj[3]
        NumberOfControl=self.Ntj[0]+self.Ntj[1]
        NumberOfPosOutcome=self.Ntj[1]+self.Ntj[3]
        NumberOfZeroOutcome=self.Ntj[0]+self.Ntj[2]

        #W=0
        LeafPrior_W0=log_binomial_coefficient(sum(self.Ntj)+1, 1)
        TreeLikelihood_W0=log_fact(sum(self.Ntj))-log_fact(NumberOfPosOutcome)-log_fact(NumberOfZeroOutcome)
        #W=1
        LeafPrior_W1=log_binomial_coefficient(NumberOfTreatment+1, 1)+log_binomial_coefficient(NumberOfControl+1, 1)
        TreeLikelihood_W1=(log_fact(NumberOfTreatment)-log_fact(self.Ntj[2])-log_fact(self.Ntj[3]))+(log_fact(NumberOfControl)-log_fact(self.Ntj[0])-log_fact(self.Ntj[1]))

        if (LeafPrior_W0+TreeLikelihood_W0)<(LeafPrior_W1+TreeLikelihood_W1):
            W=0
            LeafPrior=LeafPrior_W0
            TreeLikelihood=TreeLikelihood_W0
        else:
            W=1
            LeafPrior=LeafPrior_W1
            TreeLikelihood=TreeLikelihood_W1
        return LeafPrior,TreeLikelihood,W
    
    def DiscretizeVarsAndGetAttributesSplitsCosts(self):
        '''
        For this node loop on all attributes and get the optimal split for each one
        
        return a dictionary of lists
        {age: Cost, sex: Cost}
        The cost here corresponds to 
        1- the cost of this node to internal instead of leaf (CriterionToBeInternal-PriorLeaf)
        2- The combinatorial terms of the leaf prior and likelihood
        
        NOTE: Maybe I should save the AttributeToSplitVsLeftAndRightData in this node.
        '''
        features=list(self.X.columns)
        AttributeToSplitVsLeftAndRightData={}
#         print("features are ",features)
        for attribute in features:
            if len(self.X[attribute].value_counts())==1 or len(self.X[attribute].value_counts())==0:
                continue
            DiscRes=UMODL_Discretizer(self.X,self.T,self.Y,attribute)
            if DiscRes==-1:
                continue
            dataLeft,dataRight,threshold=DiscRes[0],DiscRes[1],DiscRes[2]
            AttributeToSplitVsLeftAndRightData[attribute]=[dataLeft,dataRight,threshold]
        
        self.CandidateSplitsVsDataLeftDataRight=AttributeToSplitVsLeftAndRightData.copy()
        CandidateSplitsVsCriterion=self.GetAttributesSplitsCosts(AttributeToSplitVsLeftAndRightData)
        self.CandidateSplitsVsCriterion=CandidateSplitsVsCriterion.copy()
        return CandidateSplitsVsCriterion.copy()
    
    def GetAttributesSplitsCosts(self,DictOfEachAttVsEffectifs):
        #Prior of Internal node is only the combinatorial calculations
        CriterionToBeInternal=self.calcPriorOfInternalInterval() #In case we split this node, it will be no more a leaf but an internal node
        NewPriorVals=CriterionToBeInternal-self.PriorLeaf-self.LikelihoodLeaf
        
        CandidateSplitsVsCriterion={}
        for key in DictOfEachAttVsEffectifs:
            LeavesVal=self.updateTreeCriterion(DictOfEachAttVsEffectifs[key][:2])#,K_subset,subsetFeatures)
            CandidateSplitsVsCriterion[key]=NewPriorVals+LeavesVal
        return CandidateSplitsVsCriterion.copy()
            

    def updateTreeCriterion(self,LeftAndRightData,simulate=True):
#         NewNodeEffectifs=[T0J0,T0J1,T1J0,T1J1]
        LeavesVals=0
        for NewIntervalEffectifs in LeftAndRightData:#Loop on Left and Right candidate nodes
            L=_Interval(NewIntervalEffectifs,self.treatment,self.output)
            LeavesVals+=(L.PriorLeaf+L.LikelihoodLeaf)
            del L
        return LeavesVals
    
    def performSplit(self,Attribute):
        if self.CandidateSplitsVsDataLeftDataRight==None:
            raise
        else:
            self.isLeaf=False
            self.leftInterval = _Interval(self.CandidateSplitsVsDataLeftDataRight[Attribute][0],self.treatment,self.output,ID=self.id*2)
            self.rightInterval = _Interval(self.CandidateSplitsVsDataLeftDataRight[Attribute][1],self.treatment,self.output,ID=self.id*2+1)
            self.Attribute = Attribute
            self.SplitThreshold=self.CandidateSplitsVsDataLeftDataRight[Attribute][2]
            return self.leftInterval,self.rightInterval
        return -1

class BayesianDecisionTree:
    """Main class"""
    
    def __init__(self, data,treatmentName,outcomeName):#ordered data as argument
        self.intervalsIds=0
        self.rootInterval=_Interval(data,treatmentName,outcomeName,ID=self.intervalsIds+1) 
        self.terminalIntervals=[self.rootInterval]
        self.internalIntervals=[]
        
        self.K=len(list(data.columns))
        self.K_t=1
        self.features=list(data.columns)
        self.feature_subset=[]

        self.Prob_Kt=None
        self.EncodingOfBeingAnInternalInterval=None
        self.ProbAttributeSelection=None
        self.PriorOfInternalIntervals=None
        self.EncodingOfBeingALeafIntervalAndContainingTE=len(self.terminalIntervals)*log(2)*2 #TE=TreatmentEffect
        self.LeafPrior=None
        self.TreeLikelihood=None
        
        
        self.calcCriterion()
        
        self.TreeCriterion=self.Prob_Kt+self.EncodingOfBeingAnInternalInterval+self.ProbAttributeSelection+self.PriorOfInternalIntervals+self.EncodingOfBeingALeafIntervalAndContainingTE+self.LeafPrior+self.TreeLikelihood

    
    def fit(self, X_train, treatment_col, outcome_col):
        """Description?

        Parameters
        ----------
        X_train : pd.Dataframe
            Dataframe containing feature variables.
        treatment_col : pd.Series
            Treatment column.
        y_col : pd.Series
            Outcome column.
        """
        #In case if we have a new attribute for splitting
        Prob_KtPlusOne=universal_code_natural_numbers(self.K_t+1)-log_fact(self.K_t+1)+(self.K_t+1)*log(self.K)
        ProbOfAttributeSelectionAmongSubsetAttributesPlusOne=log(self.K_t+1)*(len(self.internalIntervals)+1)
        
        EncodingOfBeingAnInternalIntervalPlusOne=self.EncodingOfBeingAnInternalInterval+log(2)
        
        #When splitting a node to 2 nodes, the number of leaf nodes is incremented only by one, since the parent node was leaf and is now internal.
        #2 for two extra leaf nodes multiplied by 2 for W. Total = 4.
        EncodingOfBeingALeafIntervalAndContainingTEPlusTWO=self.EncodingOfBeingALeafIntervalAndContainingTE+(2*log(2)) 
        
        EncodingOfInternalAndLeavesAndWWithExtraIntervals=EncodingOfBeingAnInternalIntervalPlusOne+EncodingOfBeingALeafIntervalAndContainingTEPlusTWO
        
        
        i=0
        while(True):
            IntervalVsBestAttributeCorrespondingToTheBestCost={}
            IntervalVsBestCost={}
            IntervalVsCandidateSplitsCosts={}#Dictionary containing Intervals as key and their values are another dictionary each with attribute:CostSplit
            IntervalVsCandidateSplitsCostsInTheInterval={}

            for terminalInterval in self.terminalIntervals:

                #This if condition is here to not to repeat calculations of candidate splits
                if terminalInterval.CandidateSplitsVsCriterion==None:
                    IntervalVsCandidateSplitsCosts[terminalInterval]=terminalInterval.DiscretizeVarsAndGetAttributesSplitsCosts()
                else:
                    IntervalVsCandidateSplitsCosts[terminalInterval]=terminalInterval.CandidateSplitsVsCriterion.copy()
                
                if len(IntervalVsCandidateSplitsCosts[terminalInterval])==0:
                    continue

                #Update Costs
                for attribute in IntervalVsCandidateSplitsCosts[terminalInterval]:
                    if attribute in self.feature_subset:
                        IntervalVsCandidateSplitsCosts[terminalInterval][attribute]+=(self.Prob_Kt
                                                                              +self.ProbAttributeSelection
                                                                              +EncodingOfInternalAndLeavesAndWWithExtraIntervals
                                                                              +self.LeafPrior+self.TreeLikelihood+self.PriorOfInternalIntervals)
                    else:
                        IntervalVsCandidateSplitsCosts[terminalInterval][attribute]+=(Prob_KtPlusOne
                                                                              +EncodingOfInternalAndLeavesAndWWithExtraIntervals
                                                                              +ProbOfAttributeSelectionAmongSubsetAttributesPlusOne
                                                                              +self.LeafPrior+self.TreeLikelihood+self.PriorOfInternalIntervals)
               
                #Once costs are updated, I get the key of the minimal value split for terminalInterval
                KeyOfTheMinimalVal=min(IntervalVsCandidateSplitsCosts[terminalInterval], key=IntervalVsCandidateSplitsCosts[terminalInterval].get)
                
                IntervalVsBestAttributeCorrespondingToTheBestCost[terminalInterval]=KeyOfTheMinimalVal
                IntervalVsBestCost[terminalInterval]=IntervalVsCandidateSplitsCosts[terminalInterval][KeyOfTheMinimalVal]
            
            if len(list(IntervalVsBestCost))==0:
                break
            
            OptimalIntervalAttributeToSplitUp=min(IntervalVsBestCost, key=IntervalVsBestCost.get)
            OptimalVal=IntervalVsBestCost[OptimalIntervalAttributeToSplitUp]
            OptimalInterval=OptimalIntervalAttributeToSplitUp
            OptimalAttribute=IntervalVsBestAttributeCorrespondingToTheBestCost[OptimalIntervalAttributeToSplitUp]
            
            if OptimalVal<self.TreeCriterion:
                self.TreeCriterion=OptimalVal
                if OptimalAttribute not in self.feature_subset:
                    self.feature_subset.append(OptimalAttribute)
                    self.K_t+=1
                NewLeftLeaf,NewRightLeaf=OptimalInterval.performSplit(OptimalAttribute)
                self.terminalIntervals.append(NewLeftLeaf)
                self.terminalIntervals.append(NewRightLeaf)
                self.internalIntervals.append(OptimalInterval)
                self.terminalIntervals.remove(OptimalInterval)
                
                self.calcCriterion()
            else:
                break
        print("Learning Finished")
        for interval in self.terminalIntervals:
            print("Interval id ",interval.id)
            print("Interval outcomeProbInTrt ",interval.outcomeProbInTrt)
            print("Interval outcomeProbInCtrl ",interval.outcomeProbInCtrl)
            print("self ntj ",interval.Ntj)
        print("===============")
        
    
    def _traverse_tree(self, x, interval):
        if interval.isLeaf==True:
            return interval.averageUplift
        
        if x[interval.Attribute] <= interval.SplitThreshold:
            return self._traverse_tree(x, interval.leftinterval)
        return self._traverse_tree(x, interval.rightinterval)
    
    
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
        predictions = [self._traverse_tree(X_test.iloc[x], self.rootInterval) for x in range(len(X_test))]
        return np.array(predictions)
