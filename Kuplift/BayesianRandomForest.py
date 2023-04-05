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
from math import log
import multiprocessing as mp
import numpy as np
import pandas as pd
import random
from HelperFunctions import log_fact, universal_code_natural_numbers, log_binomial_coefficient

#TODO UMODL_Discretizer !

class _Node:
    def __init__(self, data,treatmentName,outcomeName,ID=None):
#         print("Initializing a node")
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
#         print("self.X in node ",self.X)
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
#         self.K_t=K_t
#         self.subsetFeatures=subset_features
#         print("self.subsetFeatures is ",self.subsetFeatures)
        
        self.isLeaf=True
        
        self.CandidateSplitsVsDataLeftDataRight=None
        self.CandidateSplitsVsCriterion=None
        
        self.leftNode=None
        self.rightNode=None
        
        self.PriorOfInternalNode=self.calcPriorOfInternalNode()
        self.PriorLeaf,self.LikelihoodLeaf,self.W=self.calcPriorAndLikelihoodLeaf()
    
    def calcPriorOfInternalNode(self):
#         return log(self.K_t)+log_binomial_coefficient(sum(self.Ntj)+1,1)
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
#                 print("NULL MODEL FOUND was returned")
                continue
            dataLeft,dataRight,threshold=DiscRes[0],DiscRes[1],DiscRes[2]
            AttributeToSplitVsLeftAndRightData[attribute]=[dataLeft,dataRight,threshold]
        
        self.CandidateSplitsVsDataLeftDataRight=AttributeToSplitVsLeftAndRightData.copy()
        CandidateSplitsVsCriterion=self.GetAttributesSplitsCosts(AttributeToSplitVsLeftAndRightData)
        self.CandidateSplitsVsCriterion=CandidateSplitsVsCriterion.copy()
        return CandidateSplitsVsCriterion.copy()
    
    def GetAttributesSplitsCosts(self,DictOfEachAttVsEffectifs):
        #Prior of Internal node is only the combinatorial calculations
        CriterionToBeInternal=self.calcPriorOfInternalNode() #In case we split this node, it will be no more a leaf but an internal node
        NewPriorVals=CriterionToBeInternal-self.PriorLeaf-self.LikelihoodLeaf
        
        CandidateSplitsVsCriterion={}
        for key in DictOfEachAttVsEffectifs:
            LeavesVal=self.updateTreeCriterion(DictOfEachAttVsEffectifs[key][:2])#,K_subset,subsetFeatures)
            CandidateSplitsVsCriterion[key]=NewPriorVals+LeavesVal
        return CandidateSplitsVsCriterion.copy()
            

    def updateTreeCriterion(self,LeftAndRightData,simulate=True):
        LeavesVals=0
        for NewNodeEffectifs in LeftAndRightData:#Loop on Left and Right candidate nodes
            L=_Node(NewNodeEffectifs,self.treatment,self.output)
            LeavesVals+=(L.PriorLeaf+L.LikelihoodLeaf)
            del L
        return LeavesVals
    
    def performSplit(self,Attribute):
        if self.CandidateSplitsVsDataLeftDataRight==None:
            raise
        else:
            self.isLeaf=False
            self.leftNode = _Node(self.CandidateSplitsVsDataLeftDataRight[Attribute][0],self.treatment,self.output,ID=self.id*2)
            self.rightNode = _Node(self.CandidateSplitsVsDataLeftDataRight[Attribute][1],self.treatment,self.output,ID=self.id*2+1)
            self.Attribute = Attribute
            self.SplitThreshold=self.CandidateSplitsVsDataLeftDataRight[Attribute][2]
            return self.leftNode,self.rightNode
        return -1
            
            

# Uplift Tree Classifier
class _UpliftTreeClassifier:
    
    def __init__(self, data,treatmentName,outcomeName):#ordered data as argument

        self.nodesIds=0
        self.rootNode=_Node(data,treatmentName,outcomeName,ID=self.nodesIds+1) 
        self.terminalNodes=[self.rootNode]
        self.internalNodes=[]
        
        self.K=len(list(data.columns))
        self.K_t=1
        self.features=list(data.columns)
        self.feature_subset=[]

        self.Prob_Kt=None
        self.EncodingOfBeingAnInternalNode=None
        self.ProbAttributeSelection=None
        self.PriorOfInternalNodes=None
        self.EncodingOfBeingALeafNodeAndContainingTE=len(self.terminalNodes)*log(2)*2 #TE=TreatmentEffect
        self.LeafPrior=None
        self.TreeLikelihood=None
        
        
        self.calcCriterion()
        
        self.TreeCriterion=self.Prob_Kt+self.EncodingOfBeingAnInternalNode+self.ProbAttributeSelection+self.PriorOfInternalNodes+self.EncodingOfBeingALeafNodeAndContainingTE+self.LeafPrior+self.TreeLikelihood
        
#================================================================================================================================================
#================================================================================================================================================
    def calcCriterion(self):
        self.calcProb_kt()
        self.calcPriorOfInternalNodes()
        self.calcEncoding()
        self.calcLeafPrior()
        self.calcTreeLikelihood()
#---------------------------------------------------------------------------------------------------------------------------------------  
    def calcProb_kt(self):
        self.Prob_Kt=universal_code_natural_numbers(self.K_t)-log_fact(self.K_t)+self.K_t*log(self.K)
#---------------------------------------------------------------------------------------------------------------------------------------    
    def calcPriorOfInternalNodes(self):
        if len(self.internalNodes)==0:
            self.PriorOfInternalNodes=0
            self.ProbAttributeSelection=0
        else:
            PriorOfInternalNodes=0
            for internalNode in self.internalNodes:
                PriorOfInternalNodes+=internalNode.PriorOfInternalNode
            self.PriorOfInternalNodes=PriorOfInternalNodes
            self.ProbAttributeSelection=log(self.K_t)*len(self.internalNodes)
#---------------------------------------------------------------------------------------------------------------------------------------    
    def calcEncoding(self):
        self.EncodingOfBeingALeafNodeAndContainingTE=len(self.terminalNodes)*log(2)*2
        self.EncodingOfBeingAnInternalNode=len(self.internalNodes)*log(2)
#---------------------------------------------------------------------------------------------------------------------------------------
    def calcTreeLikelihood(self):
        LeafLikelihoods=0
        for leafNode in self.terminalNodes:
            LeafLikelihoods+=leafNode.LikelihoodLeaf
        self.TreeLikelihood=LeafLikelihoods
#---------------------------------------------------------------------------------------------------------------------------------------    
    def calcLeafPrior(self):
        leafPriors=0
        for leafNode in self.terminalNodes:
            leafPriors+=leafNode.PriorLeaf
        self.LeafPrior=leafPriors
#---------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
#================================================================================================================================================
    
    def growTree(self):
        #In case if we have a new attribute for splitting
        Prob_KtPlusOne=universal_code_natural_numbers(self.K_t+1)-log_fact(self.K_t+1)+(self.K_t+1)*log(self.K)
        ProbOfAttributeSelectionAmongSubsetAttributesPlusOne=log(self.K_t+1)*(len(self.internalNodes)+1)
        
        EncodingOfBeingAnInternalNodePlusOne=self.EncodingOfBeingAnInternalNode+log(2)
        
        #When splitting a node to 2 nodes, the number of leaf nodes is incremented only by one, since the parent node was leaf and is now internal.
        #2 for two extra leaf nodes multiplied by 2 for W. Total = 4.
        EncodingOfBeingALeafNodeAndContainingTEPlusTWO=self.EncodingOfBeingALeafNodeAndContainingTE+(2*log(2)) 
        
        EncodingOfInternalAndLeavesAndWWithExtraNodes=EncodingOfBeingAnInternalNodePlusOne+EncodingOfBeingALeafNodeAndContainingTEPlusTWO
        
        
        i=0
        while(True):
            NodeVsBestAttributeCorrespondingToTheBestCost={}
            NodeVsBestCost={}

            
            NodeVsCandidateSplitsCosts={}#Dictionary containing Nodes as key and their values are another dictionary each with attribute:CostSplit
            NodeVsCandidateSplitsCostsInTheNode={}


            for terminalNode in self.terminalNodes:
                #This if condition is here to not to repeat calculations of candidate splits
                if terminalNode.CandidateSplitsVsCriterion==None:
                    NodeVsCandidateSplitsCosts[terminalNode]=terminalNode.DiscretizeVarsAndGetAttributesSplitsCosts()
                else:
                    NodeVsCandidateSplitsCosts[terminalNode]=terminalNode.CandidateSplitsVsCriterion.copy()
                
                if len(NodeVsCandidateSplitsCosts[terminalNode])==0:
                    continue
                #Update Costs
                ListOfAttributeSplitsImprovingTreeCriterion=[]
                for attribute in NodeVsCandidateSplitsCosts[terminalNode]:
                    if attribute in self.feature_subset:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute]+=(self.Prob_Kt
                                                                              +self.ProbAttributeSelection
                                                                              +EncodingOfInternalAndLeavesAndWWithExtraNodes
                                                                              +self.LeafPrior+self.TreeLikelihood+self.PriorOfInternalNodes)
                    else:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute]+=(Prob_KtPlusOne
                                                                              +EncodingOfInternalAndLeavesAndWWithExtraNodes
                                                                              +ProbOfAttributeSelectionAmongSubsetAttributesPlusOne
                                                                              +self.LeafPrior+self.TreeLikelihood+self.PriorOfInternalNodes)
                    
                
                    if NodeVsCandidateSplitsCosts[terminalNode][attribute]<self.TreeCriterion:
                        ListOfAttributeSplitsImprovingTreeCriterion.append(attribute)
                if len(ListOfAttributeSplitsImprovingTreeCriterion)==0:
                    continue
                KeyOfTheMinimalVal=random.choice(ListOfAttributeSplitsImprovingTreeCriterion) #KeyOfTheMinimalVal is the attribute name
                
                NodeVsBestAttributeCorrespondingToTheBestCost[terminalNode]=KeyOfTheMinimalVal
                NodeVsBestCost[terminalNode]=NodeVsCandidateSplitsCosts[terminalNode][KeyOfTheMinimalVal]
            
            if len(list(NodeVsBestCost))==0:
                break
            OptimalNodeAttributeToSplitUp=random.choice(list(NodeVsBestCost))
            OptimalVal=NodeVsBestCost[OptimalNodeAttributeToSplitUp]
            OptimalNode=OptimalNodeAttributeToSplitUp
            OptimalAttribute=NodeVsBestAttributeCorrespondingToTheBestCost[OptimalNodeAttributeToSplitUp]
            
            if OptimalVal<self.TreeCriterion:

                self.TreeCriterion=OptimalVal
                if OptimalAttribute not in self.feature_subset:
                    self.feature_subset.append(OptimalAttribute)
                    self.K_t+=1
                NewLeftLeaf,NewRightLeaf=OptimalNode.performSplit(OptimalAttribute)
                self.terminalNodes.append(NewLeftLeaf)
                self.terminalNodes.append(NewRightLeaf)
                self.internalNodes.append(OptimalNode)
                self.terminalNodes.remove(OptimalNode)
                
                self.calcCriterion()
            else:
                print("WILL NEVER ENTER HERE")
                break
    def getSummary(self):
        SummaryDF=pd.DataFrame(columns=['NodeId','isLeaf','T0Y0','T0Y1','T1Y0','T1Y1','Uplift','SplittedAttribute','SplitThreshold'])#SplitThreshold
        for internalNode in self.internalNodes:
            SummaryDF.loc[len(SummaryDF.index)] = [internalNode.id,internalNode.isLeaf,internalNode.Ntj[0],internalNode.Ntj[1],internalNode.Ntj[2],internalNode.Ntj[3],internalNode.averageUplift,internalNode.Attribute,internalNode.SplitThreshold]
        for terminalNode in self.terminalNodes:
            SummaryDF.loc[len(SummaryDF.index)] = [terminalNode.id,terminalNode.isLeaf,terminalNode.Ntj[0],terminalNode.Ntj[1],terminalNode.Ntj[2],terminalNode.Ntj[3],terminalNode.averageUplift,terminalNode.Attribute,terminalNode.SplitThreshold]
        SummaryDF.to_csv("SummaryDT_UMODL.csv")
    
    def _traverse_tree(self, x, node):
        if node.isLeaf==True:
            return node.averageUplift
        
        if x[node.Attribute] <= node.SplitThreshold:
            return self._traverse_tree(x, node.leftNode)
        return self._traverse_tree(x, node.rightNode)
    
    def predict(self, X):
        predictions = [self._traverse_tree(X.iloc[x], self.rootNode) for x in range(len(X))]
        return np.array(predictions)



class BayesianRandomForest:
    """Main class
    
    Parameters
    ----------
    n_trees : int
        Number of trees in a forest.
    """
    def __init__(self, data,treatmentName,outcomeName,numberOfTrees=10,parallelized=False,NotAllVars=False):
        self.ListOfTrees=[]
        self.data=data
        #Randomly select columns for the data
        if NotAllVars==True:
            cols=list(self.data.columns)
            cols.remove(treatmentName)
            cols.remove(outcomeName)
            print("cols before are ",cols)
            cols=random.sample(cols,int(np.sqrt(len(cols))))
            print("cols after are ",cols)
            self.data=self.data[cols+[treatmentName,outcomeName]]
        self.parallelized=parallelized
        for i in range(numberOfTrees):
            Tree=_UpliftTreeClassifier(self.data.copy(),treatmentName,outcomeName)
            self.ListOfTrees.append(Tree)

    
    #Question : pourquoi ces paramètres de fonction ?
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
        if self.parallelized:
            pool = mp.Pool(processes=10)
            pool.map(self.fit_parallelized, self.ListOfTrees)
            pool.close()
        else:
            for tree in self.ListOfTrees:
                tree.growTree()
    
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
        ListOfPreds=[]
        
        if self.parallelized:
            pool = mp.Pool(processes=10)
            ListOfModelsAndX=[]
            for tree in self.ListOfTrees:
                ListOfModelsAndX.append([tree,X_test.copy()])
#             print("ListOfModelsAndX ",ListOfModelsAndX)
            ListOfPreds=pool.map(self.predict_parallelized, ListOfModelsAndX)
            pool.close()
        else:
            for tree in self.ListOfTrees:
                ListOfPreds.append(np.array(tree.predict(X_test)))
        return np.mean(ListOfPreds,axis=0)
    