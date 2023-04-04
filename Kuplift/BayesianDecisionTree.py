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

class BayesianDecisionTree:
    """Main class"""
    
    #TODO à revoir
    def __log_fact(self, n):
        """
        Compute log(fact(n))
        :param n:
        :return: value of log(fact(n))
        """
        start_counter(4)
        # print("\t\t asked for log_fact(n=%d)"%n)
        # Use approximation for large n
        if n > 1e6:
            # print('\t\t Using approximation : res=%.f' %log_fact_approx(n))
            return log_fact_approx(n)
        # computation of values, tabulation in private array
        else:
            s = len(_Log_Fact_Table)
            if n >= s:
                if s == 0:
                    _Log_Fact_Table.append(0)
                size = len(_Log_Fact_Table)
                while size <= n:
                    # print('%d<=%d' %(size,n))
                    _Log_Fact_Table.append(log(size) + _Log_Fact_Table[size - 1])
                    size = size + 1
            stop_counter(4)
            return _Log_Fact_Table[n]

    
    def __log_2_star(self, k: int):
        """
        Computes the term log_2*(k)=log_2(k) + log_2(log_2(k)) + ...  of Rissanen's code for integers
        so long as the terms are positive
        :param k:
        :return:
        """
        d_log2 = log(2.0)
        d_cost = 0.0
        d_logI = log(1.0 * k) / d_log2

        if k < 1:
            raise ValueError("Universal code is defined for natural numbers over 1")
        else:
            while d_logI > 0:
                d_cost += d_logI
                d_logI = log(d_logI) / d_log2

            return d_cost    

    
    def __universal_code_natural_numbers(self, k: int):
        """
        Compute the universal code for integers presented by Rissanen in
        'A Universal Prior for Integers and Estimation by Minimum Description Length', Rissanen 1983
        :param k:
        :return:
        """
        dC0 = 2.86511  # First value computed following the given estimation formula, as e(3)=65536 + d_log2^5 / (1-d_log2)
        d_log2 = log(2.0)

        if k < 1:
            raise ValueError("Universal code is defined for natural numbers over 1")
        else:
            # Initialize code length cost to log_2(dC0)
            d_cost = log(dC0) / d_log2

            # Add log_2*(k)
            d_cost += self.__log_2_star(k)

            # Go back to the natural log
            d_cost *= d_log2

            return d_cost  
    
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
        Prob_KtPlusOne=self.__universal_code_natural_numbers(self.K_t+1)-self.__log_fact(self.K_t+1)+(self.K_t+1)*log(self.K)
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
               
                #Once costs are updated, I get the key of the minimal value split for terminalNode
                KeyOfTheMinimalVal=min(NodeVsCandidateSplitsCosts[terminalNode], key=NodeVsCandidateSplitsCosts[terminalNode].get)
                
                NodeVsBestAttributeCorrespondingToTheBestCost[terminalNode]=KeyOfTheMinimalVal
                NodeVsBestCost[terminalNode]=NodeVsCandidateSplitsCosts[terminalNode][KeyOfTheMinimalVal]
            
            if len(list(NodeVsBestCost))==0:
                break
            
            OptimalNodeAttributeToSplitUp=min(NodeVsBestCost, key=NodeVsBestCost.get)
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
                break
        print("Learning Finished")
        for node in self.terminalNodes:
            print("Node id ",node.id)
            print("Node outcomeProbInTrt ",node.outcomeProbInTrt)
            print("Node outcomeProbInCtrl ",node.outcomeProbInCtrl)
            print("self ntj ",node.Ntj)
        print("===============")
        
    
    def _traverse_tree(self, x, node):
        if node.isLeaf==True:
            return node.averageUplift
        
        if x[node.Attribute] <= node.SplitThreshold:
            return self._traverse_tree(x, node.leftNode)
        return self._traverse_tree(x, node.rightNode)
    
    
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
        predictions = [self._traverse_tree(X_test.iloc[x], self.rootNode) for x in range(len(X_test))]
        return np.array(predictions)
