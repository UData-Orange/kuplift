import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rcParams

from operator import itemgetter
import bisect
from math import log, pi, pow, exp, lgamma, sqrt
import numpy as np
from typing import Callable
from math import ceil, floor
from operator import itemgetter
from sortedcontainers import SortedKeyList
from operator import add
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import logging
import time
import os
import pickle
import sys
import json
from equalFreq import generate_EqualFreqSteps
import equalFreq

from stats_rissanen import universal_code_natural_numbers
from stats_rissanen import log_2_star
# import MDL_Criteria

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

from scipy.special import comb
import scipy.special as sc

from operator import itemgetter
import operator
import bisect
import stats_rissanen
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)

from HelperFunctions import log_fact
from HelperFunctions import log_binomial_coefficient

import HelperFunctions


from stats_rissanen import BoundedNaturalNumbersUniversalCodeLength
import scipy.special as sc


#============================================================================================
_nb_counter = []
_start_counter = []
_start_time_counter = []
_deltatime_counter = []
_NumberOfCounters=25
for i in range(0,_NumberOfCounters):
    _nb_counter.append(0)
    _start_counter.append(False)
    _start_time_counter.append(time.time())
    _deltatime_counter.append(0)


def start_counter(i):
    _nb_counter[i]=_nb_counter[i]+1
    _start_counter[i]=True
    _start_time_counter[i]=time.time()  
    
def stop_counter(i):
    _start_counter[i]=False
    diff = time.time()  - _start_time_counter[i]
    _deltatime_counter[i] = _deltatime_counter[i]+diff
    _start_time_counter[i]=time.time()

    

#============================================================================================
def calcNullModel(dff,att,treatmentColName,outputColName):
    data=dff[[att,treatmentColName,outputColName]].values.tolist()
    
    data=SortedKeyList(data, key=itemgetter(0))
    dataNITJ=[0,0,0,0] #The frequency of treatment class
    for intervalList in data:
        dataNITJ[int((intervalList[1]*2)+intervalList[2])]+=1
    
    N_instances=dff.shape[0]
    
    
    NumberOfIndividualsWithClass1=dff[dff[outputColName]==1].shape[0]
    NumberOfIndividualsWithClass0=dff[dff[outputColName]==0].shape[0]
    
    LastTermInNullModel=log_fact(N_instances)-(log_fact(NumberOfIndividualsWithClass1)+log_fact(NumberOfIndividualsWithClass0))
    return (2*log(2))+calcCriterion(dataNITJ)
def calcCriterion(NITJ_Interval,NUllModel=False): 
    
    NITJ_Interval_sum=sum(NITJ_Interval)
    Fact_Class0Freq=HelperFunctions._Log_Fact_Table[(NITJ_Interval[0]+NITJ_Interval[2])]
    Fact_Class1Freq=HelperFunctions._Log_Fact_Table[(NITJ_Interval[1]+NITJ_Interval[3])]
    
    Fact_T0Freq=HelperFunctions._Log_Fact_Table[(NITJ_Interval[0]+NITJ_Interval[1])]
    Fact_T1Freq=HelperFunctions._Log_Fact_Table[(NITJ_Interval[2]+NITJ_Interval[3])]
    
    Fact_T0Class0Freq=HelperFunctions._Log_Fact_Table[NITJ_Interval[0]]
    Fact_T0Class1Freq=HelperFunctions._Log_Fact_Table[NITJ_Interval[1]]
    
    Fact_T1Class0Freq=HelperFunctions._Log_Fact_Table[NITJ_Interval[2]]
    Fact_T1Class1Freq=HelperFunctions._Log_Fact_Table[NITJ_Interval[3]]

    
    
    #Likelihood 1 W=0
    start_counter(0)
    likelihood_denum=0
    j=0
    likelihood_denum+=Fact_Class0Freq
    j=1
    likelihood_denum+=Fact_Class1Freq
    
#     print("NITJ_Interval_sum ",NITJ_Interval_sum)
    likelihood1_tmp=((HelperFunctions._Log_Fact_Table[NITJ_Interval_sum]-likelihood_denum))
    
    #Likelihood 2 W=1
    res_t=0
    t=0
    likelihood_denum=0
    
    j=0
    likelihood_denum+=Fact_T0Class0Freq
    j=1
    likelihood_denum+=Fact_T0Class1Freq
    
    res_t+=(Fact_T0Freq-likelihood_denum)
    t=1
    likelihood_denum=0
    
    j=0
    likelihood_denum+=Fact_T1Class0Freq
    j=1
    likelihood_denum+=Fact_T1Class1Freq

    res_t+=(Fact_T1Freq-likelihood_denum)
    
    likelihood2_tmp=res_t
    
    #Prior 1 W=0
    prior1_tmp=(log_binomial_coefficient(NITJ_Interval_sum+1,1))
    #Prior 2 W=1
    res_t=0
    
    t=0
    
    res_t_temp=log_binomial_coefficient((NITJ_Interval[2*t]+NITJ_Interval[2*t+1])+1,1)
    res_t+=res_t_temp
    t=1
    res_t_temp=log_binomial_coefficient((NITJ_Interval[2*t]+NITJ_Interval[2*t+1])+1,1)
    res_t+=res_t_temp
    
    prior2_tmp=res_t
    stop_counter(0)
    
    righMergeW=None
    if NUllModel==False:
        if (prior1_tmp+likelihood1_tmp)<(prior2_tmp+likelihood2_tmp):
            righMergeW=0
        else:
            righMergeW=1
    else:
        if (prior1_tmp+likelihood1_tmp)>(prior2_tmp+likelihood2_tmp):
            righMergeW=0
        else:
            righMergeW=1
    
    Prior1=(1-righMergeW)*prior1_tmp
    Prior2=righMergeW*prior2_tmp
    Likelihood1=(1-righMergeW)*likelihood1_tmp
    Likelihood2=(righMergeW)*likelihood2_tmp
    SumOfPriorsAndLikelihoods=Prior1+Prior2+Likelihood1+Likelihood2
    

    


    return SumOfPriorsAndLikelihoods
def splitInterval(df,colName,treatmentColName,outputColName,NullModelValue,granularite=16):#i is interval index in IntervalsList
    data=df[[colName,treatmentColName,outputColName]].values.tolist()
    
    data=SortedKeyList(data, key=itemgetter(0))
#     print("Started to split the interval")
    Count=2 #We only have two intervals
    dataNITJ=[0,0,0,0] #The frequency of treatment class
    for intervalList in data:
        dataNITJ[int((intervalList[1]*2)+intervalList[2])]+=1

    N=len(data)
    IncludingLeftBorder=True
    LeftBound=data[0][0] #The smallest value
    RightBound = data[-1][0] #The biggest value

    #Get all the unique values in the data i.e All unique values between left and right bounds
    uniqueValuesInBothIntervals = list(data.irange_key(LeftBound, RightBound,(IncludingLeftBorder,True)))
    uniqueValuesInBothIntervals=list(map(itemgetter(0),uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals=list(set(uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals.sort() #Sort the unique values
    
    Splits={}
    LeftAndRightIntervalOfSplits={}

    previousLeftInterval=[0,0,0,0]
    prevVal=None
    
    #Classical prior vs new prior of rissannen !!!
    PriorRissanen=log(2)+log_binomial_coefficient(N+1,1)+2*log(2)

    
    for val in uniqueValuesInBothIntervals:
        if (len(uniqueValuesInBothIntervals)<=1) or (val==RightBound):
            break

        if prevVal==None: #Enters here only for the first unique value
            leftSplit=list(data.irange_key(LeftBound, val,(True,True))) #Get a list of all data between LeftBound and current unique value 
            leftInterval=[0,0,0,0]
            for intervalList in leftSplit:
                leftInterval[int((intervalList[1]*2)+intervalList[2])]+=1
        else:
            leftSplit=list(data.irange_key(prevVal, val,(False,True)))
            leftInterval=[0,0,0,0]
            for intervalList in leftSplit:
                leftInterval[int((intervalList[1]*2)+intervalList[2])]+=1
            '''
            New Left Interval frequencies is the sum of the previous left interval (bounded between Smallest value and prevVal) and the 
            new left interval (bounded between prevVal and val)
            '''
            leftInterval=list(map(operator.add, previousLeftInterval, leftInterval))

        #the nitj for the right split (Which we call the rightInterval) will be the difference between the old nitj and the leftInterval
        prevVal=val
        previousLeftInterval=leftInterval.copy()
        
        '''
        The new rigt interval is the soustraction of all the data and the new left interval
        '''
        rightInterval= list(map(operator.sub, dataNITJ, leftInterval))

        #Calculate criterion manually
        start_counter(22)
        criterion1=calcCriterion(leftInterval) #prior and likelihood 
        criterion2=calcCriterion(rightInterval) #prior and likelihood 
        stop_counter(22)
        
        start_counter(24)
        #MODL value
        SplitCriterionVal_leftAndRight=PriorRissanen+criterion1+criterion2
        stop_counter(24)
        #If the MODL value is smaller than the null model value add it to the splits dictionary
        if SplitCriterionVal_leftAndRight < NullModelValue:
            if sum(rightInterval)==0:
                print("strange case")
                print('NullModelValue ',NullModelValue)
                print("SplitCriterionVal_leftAndRight ",SplitCriterionVal_leftAndRight)
            Splits[val]=SplitCriterionVal_leftAndRight
    splitDone = False
    bestSplit=None
    
    #If dictionary Splits contain value, get the minimal one
    if Splits:
        bestSplit = min(Splits, key=Splits.get) #To be optimized maybe
        leftSplit=list(data.irange_key(LeftBound, bestSplit,(True,True)))
        leftInterval=[0,0,0,0]
        for intervalList in leftSplit:
            leftInterval[int((intervalList[1]*2)+intervalList[2])]+=1
        rightInterval= list(map(operator.sub, dataNITJ, leftInterval))
        
        IndexOfLastRowInLeftData=bisect.bisect_right(df[colName].tolist(), bestSplit)    
        LeftData=df.iloc[:IndexOfLastRowInLeftData,:]
        RightData=df.iloc[IndexOfLastRowInLeftData:,:]
        return LeftData,RightData,bestSplit,Splits[bestSplit]
    else:
        SplitCriterionVal_leftAndRight=None
    
    
    return -1
        
#============================================================================================  
# Not Used  
def GranTableCreation(df,attributeToDiscretize):
    colName=attributeToDiscretize

    GranTable={}
    frequencies={}
    data=df[[colName,'T','Y']].values.tolist()
    
    data=SortedKeyList(data, key=itemgetter(0))
    
    LeftBound=data[0][0] #The smallest value
    RightBound = data[-1][0] #The biggest value

    N=df.shape[0]
    
    #Array of four elements indicating number of examples with T0J0, T0J1, T1J0, T1J1
    DataTreatmentClassFreq=[0,0,0,0]
    for intervalList in data:
        DataTreatmentClassFreq[int((intervalList[1]*2)+intervalList[2])]+=1
    
    uniqueValuesInBothIntervals = list(data.irange_key(LeftBound, RightBound,(True,True)))
    uniqueValuesInBothIntervals=list(map(itemgetter(0),uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals=list(set(uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals.sort() #Sort the unique values
    uniqueValuesInBothIntervals=np.array(uniqueValuesInBothIntervals)

    
    DataVSTreatmentClassFreq={}
    prevFreqList=[0,0,0,0]
    
    possible_thresholds_index=0
    #==========================#==========================#==========================
    '''
    After reviewing this part of the code, its serves to calculate the frequency of treatment class vals when 
    val is a threshold (the frequencies before val)
    '''
    # Note that the threshold is in the left interval.
    prevVal=None
    prevprevVal=None
    prevFreq=[0,0,0,0]
    tempFreq=[0,0,0,0]
    for intervalList in data:
        val=intervalList[0]
        if val in DataVSTreatmentClassFreq:
            tempFreq[int((intervalList[1]*2)+intervalList[2])]+=1
#             DataVSTreatmentClassFreq[val][int((intervalList[1]*2)+intervalList[2])]+=1
        else:
            if prevVal!=None and prevprevVal==None:
                DataVSTreatmentClassFreq[prevVal] = tempFreq.copy()
                prevprevVal=prevVal
                prevVal=val
            elif prevVal!=None and prevprevVal!=None:
                DataVSTreatmentClassFreq[prevVal] =[a + b for a, b in zip(tempFreq.copy(), DataVSTreatmentClassFreq[prevprevVal].copy())]
                prevprevVal=prevVal
                prevVal=val
#                 DataVSTreatmentClassFreq[val]=tempFreq.copy()+DataVSTreatmentClassFreq[prevVal].copy()
            elif prevVal==None:
                DataVSTreatmentClassFreq[val]=[0,0,0,0]
                prevVal=val
            tempFreq=[0,0,0,0]
            tempFreq[int((intervalList[1]*2)+intervalList[2])]+=1
    DataVSTreatmentClassFreq[prevVal] =[a + b for a, b in zip(tempFreq.copy(), DataVSTreatmentClassFreq[prevprevVal].copy())]

    keys_list=list(DataVSTreatmentClassFreq.keys()) #normalement keys_list doit être égal à uniqueValuesInBothIntervals

    steps=[]
    gran_max=ceil(log(N,2))
    if gran_max>12:
        gran_max=12
    for granul in range(1, gran_max):
        steps.append(2**granul)
        
    for step in steps:
        granularite=log(step,2)
        Count=2 #Number of intervals
        PriorRissanen=log(2)+stats_rissanen.BoundedNaturalNumbersUniversalCodeLength(granularite,ceil(log(N/2,2)))+stats_rissanen.BoundedNaturalNumbersUniversalCodeLength(Count-1,(2**granularite)-1)+(Count-1)*log((2**granularite)-1)-helperFunctions._Log_Fact_Table[Count-1]+Count*log(2)

        start_counter(2)
        thresholds=generate_EqualFreqSteps(np.array(df[colName]),step-1)

        stop_counter(2)
        
        start_counter(6)
        
        index_prev=0
        
        if len(list(thresholds))==0:
            continue
        for el in list(thresholds):
#             print("el is ",el)
            if el in GranTable:
#                 print("WILL CONTINUE")
                continue
            start_counter(5)
            IndexOfValInDataVSTreatmentClassFreq=bisect.bisect_left(keys_list, el)
            ValInDataVSTreatmentClassFreq=keys_list[IndexOfValInDataVSTreatmentClassFreq]
            leftInterval=DataVSTreatmentClassFreq[ValInDataVSTreatmentClassFreq]
            stop_counter(5)

            rightInterval= list(map(operator.sub, DataTreatmentClassFreq, leftInterval))

    
            
            criterion1=calcCriterion(leftInterval)
            criterion2=calcCriterion(rightInterval)
            
            GranTable[el]=[granularite,criterion1+criterion2+PriorRissanen,leftInterval]
            frequencies[el]=[granularite,criterion1+criterion2+PriorRissanen,leftInterval]
        stop_counter(6)
    GranTable=pd.DataFrame.from_dict(GranTable,orient='index',columns=["Granularite","CriterionCost","leftInterval"])
    frequencies=pd.DataFrame.from_dict(frequencies,orient='index',columns=["Granularite","CriterionCost","leftInterval"])
    GranTable.to_csv("GranTable.csv")
    frequencies.to_csv("frequencies.csv")
    try:
        ValOfTheMinCost=float(GranTable[['CriterionCost']].idxmin())
    except:
        return -1
    GranOfMinValCost=GranTable.loc[[ValOfTheMinCost]]['Granularite']
    MinCost=GranTable.loc[ValOfTheMinCost,'CriterionCost']
    NullModelVal=calcNullModel(df,attributeToDiscretize)
    if MinCost>NullModelVal:
        return -1
    
    IndexOfLastRowInLeftData=bisect.bisect_right(df[attributeToDiscretize].tolist(), ValOfTheMinCost)
    
    LeftData=df.iloc[:IndexOfLastRowInLeftData,:]
    RightData=df.iloc[IndexOfLastRowInLeftData:,:]
    return LeftData,RightData,ValOfTheMinCost
#============================================================================================    
def Exec(df,attributeToDiscretize,treatmentColName,outputColName):    
    #NULL MODEL VALUE
    NullModelValue=calcNullModel(df,attributeToDiscretize,treatmentColName,outputColName)
    return splitInterval(df,attributeToDiscretize,treatmentColName,outputColName,NullModelValue)
#============================================================================================  
#MAIN

def UMODL_BinaryDiscretization(data,T,Y,attributeToDiscretize):
    df=pd.DataFrame()
    df=data.copy()
    treatmentColName=T.name
    outputColName=Y.name
    df[treatmentColName]=T
    df[outputColName]=Y
    

    df.sort_values(by=attributeToDiscretize,inplace=True)
    
    df.reset_index(inplace=True,drop=True)
    
    log_fact(df.shape[0]+1)
    return Exec(df,attributeToDiscretize,treatmentColName,outputColName)