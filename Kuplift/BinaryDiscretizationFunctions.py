######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import bisect
from math import log
import pandas as pd
import time
from .HelperFunctions import _Log_Fact_Table, log_fact, log_binomial_coefficient
from operator import itemgetter, add, sub
from sortedcontainers import SortedKeyList

_nb_counter = []
_start_counter = []
_start_time_counter = []
_deltatime_counter = []
_NumberOfCounters = 25
for i in range(_NumberOfCounters):
    _nb_counter.append(0)
    _start_counter.append(False)
    _start_time_counter.append(time.time())
    _deltatime_counter.append(0)


def start_counter(i):
    _nb_counter[i] = _nb_counter[i] + 1
    _start_counter[i] = True
    _start_time_counter[i] = time.time()


def stop_counter(i):
    _start_counter[i] = False
    diff = time.time() - _start_time_counter[i]
    _deltatime_counter[i] = _deltatime_counter[i] + diff
    _start_time_counter[i] = time.time()


def calc_criterion(NITJ_Interval, NUllModel=False):
    NITJ_Interval_sum = sum(NITJ_Interval)
    Fact_Class0Freq = _Log_Fact_Table[(NITJ_Interval[0] + NITJ_Interval[2])]
    Fact_Class1Freq = _Log_Fact_Table[(NITJ_Interval[1] + NITJ_Interval[3])]

    Fact_T0Freq = _Log_Fact_Table[(NITJ_Interval[0] + NITJ_Interval[1])]
    Fact_T1Freq = _Log_Fact_Table[(NITJ_Interval[2] + NITJ_Interval[3])]

    Fact_T0Class0Freq = _Log_Fact_Table[NITJ_Interval[0]]
    Fact_T0Class1Freq = _Log_Fact_Table[NITJ_Interval[1]]

    Fact_T1Class0Freq = _Log_Fact_Table[NITJ_Interval[2]]
    Fact_T1Class1Freq = _Log_Fact_Table[NITJ_Interval[3]]

    # Likelihood 1 W=0
    start_counter(0)
    likelihood_denum = 0
    j = 0
    likelihood_denum += Fact_Class0Freq
    j = 1
    likelihood_denum += Fact_Class1Freq

    likelihood1_tmp = _Log_Fact_Table[NITJ_Interval_sum] - likelihood_denum

    # Likelihood 2 W=1
    res_t = 0
    t = 0
    likelihood_denum = 0

    j = 0
    likelihood_denum += Fact_T0Class0Freq
    j = 1
    likelihood_denum += Fact_T0Class1Freq

    res_t += Fact_T0Freq - likelihood_denum
    t = 1
    likelihood_denum = 0

    j = 0
    likelihood_denum += Fact_T1Class0Freq
    j = 1
    likelihood_denum += Fact_T1Class1Freq

    res_t += Fact_T1Freq - likelihood_denum

    likelihood2_tmp = res_t

    # Prior 1 W=0
    prior1_tmp = log_binomial_coefficient(NITJ_Interval_sum + 1, 1)
    # Prior 2 W=1
    res_t = 0
    t = 0
    res_t_temp = log_binomial_coefficient(
        (NITJ_Interval[2 * t] + NITJ_Interval[2 * t + 1]) + 1, 1
    )
    res_t += res_t_temp
    t = 1
    res_t_temp = log_binomial_coefficient(
        (NITJ_Interval[2 * t] + NITJ_Interval[2 * t + 1]) + 1, 1
    )
    res_t += res_t_temp

    prior2_tmp = res_t
    stop_counter(0)

    righMergeW = None
    if NUllModel == False:
        if (prior1_tmp + likelihood1_tmp) < (prior2_tmp + likelihood2_tmp):
            righMergeW = 0
        else:
            righMergeW = 1
    else:
        if (prior1_tmp + likelihood1_tmp) > (prior2_tmp + likelihood2_tmp):
            righMergeW = 0
        else:
            righMergeW = 1

    Prior1 = (1 - righMergeW) * prior1_tmp
    Prior2 = righMergeW * prior2_tmp
    Likelihood1 = (1 - righMergeW) * likelihood1_tmp
    Likelihood2 = (righMergeW) * likelihood2_tmp
    SumOfPriorsAndLikelihoods = Prior1 + Prior2 + Likelihood1 + Likelihood2
    return SumOfPriorsAndLikelihoods


def split_interval(
    df, colName, treatmentColName, outputColName, NullModelValue, granularite=16
):  # i is interval index in IntervalsList
    data = df[[colName, treatmentColName, outputColName]].values.tolist()

    data = SortedKeyList(data, key=itemgetter(0))
    Count = 2  # We only have two intervals
    dataNITJ = [0, 0, 0, 0]  # The frequency of treatment class
    for intervalList in data:
        dataNITJ[int((intervalList[1] * 2) + intervalList[2])] += 1

    N = len(data)
    IncludingLeftBorder = True
    LeftBound = data[0][0]  # The smallest value
    RightBound = data[-1][0]  # The biggest value

    # Get all the unique values in the data i.e All unique values between left and right bounds
    uniqueValuesInBothIntervals = list(
        data.irange_key(LeftBound, RightBound, (IncludingLeftBorder, True))
    )
    uniqueValuesInBothIntervals = list(map(itemgetter(0), uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals = list(set(uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals.sort()  # Sort the unique values

    Splits = {}

    previousLeftInterval = [0, 0, 0, 0]
    prevVal = None

    # Classical prior vs new prior of rissannen !!!
    PriorRissanen = log(2) + log_binomial_coefficient(N + 1, 1) + 2 * log(2)

    for val in uniqueValuesInBothIntervals:
        if (len(uniqueValuesInBothIntervals) <= 1) or (val == RightBound):
            break

        if prevVal == None:  # Enters here only for the first unique value
            leftSplit = list(
                data.irange_key(LeftBound, val, (True, True))
            )  # Get a list of all data between LeftBound and current unique value
            leftInterval = [0, 0, 0, 0]
            for intervalList in leftSplit:
                leftInterval[int((intervalList[1] * 2) + intervalList[2])] += 1
        else:
            leftSplit = list(data.irange_key(prevVal, val, (False, True)))
            leftInterval = [0, 0, 0, 0]
            for intervalList in leftSplit:
                leftInterval[int((intervalList[1] * 2) + intervalList[2])] += 1
            """
            New Left Interval frequencies is the sum of the previous left interval (bounded between Smallest value and prevVal) and the 
            new left interval (bounded between prevVal and val)
            """
            leftInterval = list(map(add, previousLeftInterval, leftInterval))

        # the nitj for the right split (Which we call the rightInterval) will be the difference between the old nitj and the leftInterval
        prevVal = val
        previousLeftInterval = leftInterval.copy()

        """
        The new rigt interval is the soustraction of all the data and the new left interval
        """
        rightInterval = list(map(sub, dataNITJ, leftInterval))

        # Calculate criterion manually
        start_counter(22)
        criterion1 = calc_criterion(leftInterval)  # prior and likelihood
        criterion2 = calc_criterion(rightInterval)  # prior and likelihood
        stop_counter(22)

        start_counter(24)
        # MODL value
        SplitCriterionVal_leftAndRight = PriorRissanen + criterion1 + criterion2
        stop_counter(24)
        # If the MODL value is smaller than the null model value add it to the splits dictionary
        if SplitCriterionVal_leftAndRight < NullModelValue:
            if sum(rightInterval) == 0:
                print("strange case")
                print("NullModelValue ", NullModelValue)
                print("SplitCriterionVal_leftAndRight ", SplitCriterionVal_leftAndRight)
            Splits[val] = SplitCriterionVal_leftAndRight
    bestSplit = None

    # If dictionary Splits contain value, get the minimal one
    if Splits:
        bestSplit = min(Splits, key=Splits.get)  # To be optimized maybe
        leftSplit = list(data.irange_key(LeftBound, bestSplit, (True, True)))
        leftInterval = [0, 0, 0, 0]
        for intervalList in leftSplit:
            leftInterval[int((intervalList[1] * 2) + intervalList[2])] += 1
        rightInterval = list(map(sub, dataNITJ, leftInterval))

        IndexOfLastRowInLeftData = bisect.bisect_right(df[colName].tolist(), bestSplit)
        LeftData = df.iloc[:IndexOfLastRowInLeftData, :]
        RightData = df.iloc[IndexOfLastRowInLeftData:, :]
        return LeftData, RightData, bestSplit, Splits[bestSplit]
    else:
        SplitCriterionVal_leftAndRight = None

    return -1


def calc_null_model(dff, att, treatmentColName, outputColName):
    data = dff[[att, treatmentColName, outputColName]].values.tolist()

    data = SortedKeyList(data, key=itemgetter(0))
    dataNITJ = [0, 0, 0, 0]  # The frequency of treatment class
    for intervalList in data:
        dataNITJ[int((intervalList[1] * 2) + intervalList[2])] += 1

    N_instances = dff.shape[0]

    NumberOfIndividualsWithClass1 = dff[dff[outputColName] == 1].shape[0]
    NumberOfIndividualsWithClass0 = dff[dff[outputColName] == 0].shape[0]

    LastTermInNullModel = log_fact(N_instances) - (
        log_fact(NumberOfIndividualsWithClass1)
        + log_fact(NumberOfIndividualsWithClass0)
    )
    return (2 * log(2)) + calc_criterion(dataNITJ)


def exec(df, attributeToDiscretize, treatmentColName, outputColName):
    NullModelValue = calc_null_model(
        df, attributeToDiscretize, treatmentColName, outputColName
    )
    return split_interval(
        df, attributeToDiscretize, treatmentColName, outputColName, NullModelValue
    )


def umodl_binary_discretization(data, T, Y, attributeToDiscretize):
    df = pd.DataFrame()
    df = data.copy()
    treatmentColName = T.name
    outputColName = Y.name
    df[treatmentColName] = T
    df[outputColName] = Y
    df.sort_values(by=attributeToDiscretize, inplace=True)
    df.reset_index(inplace=True, drop=True)
    log_fact(df.shape[0] + 1)
    return exec(df, attributeToDiscretize, treatmentColName, outputColName)
