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
from operator import add, sub, itemgetter
from sortedcontainers import SortedKeyList
from .HelperFunctions import log_fact, log_binomial_coefficient
from .BinaryDiscretizationFunctions import start_counter, stop_counter


class _Interval:
    def __init__(self, data):
        if len(data) == 3:
            self.nitj = data[0]
            self.IncludedRightFrontier = data[1]
            self.W = data[2]
            self.rightMergeDelta = None
            self.righMergeW = None
            self.Prior1 = None
            self.Prior2 = None
            self.Likelihood1 = None
            self.Likelihood2 = None
            self.next = None
            self.previous = None
            self.SumOfPriorsAndLikelihoods = None
            self.SumOfPriorsAndLikelihoodsWithZeroW = None
            self.SumOfPriorsAndLikelihoodsWithOneW = None
        else:
            self.nitj = data[0]
            self.IncludedRightFrontier = data[1]
            self.W = data[2]
            self.rightMergeDelta = data[3]
            self.righMergeW = data[4]
            self.Prior1 = data[5]
            self.Prior2 = data[6]
            self.Likelihood1 = data[7]
            self.Likelihood2 = data[8]
            self.next = data[9]
            self.previous = data[10]
            self.SumOfPriorsAndLikelihoods = data[11]
            self.SumOfPriorsAndLikelihoodsWithZeroW = None

    def getIntervalData(self):
        return [
            self.nitj,
            self.IncludedRightFrontier,
            self.W,
            self.rightMergeDelta,
            self.righMergeW,
            self.Prior1,
            self.Prior2,
            self.Likelihood1,
            self.Likelihood2,
            self.next,
            self.previous,
            self.SumOfPriorsAndLikelihoods,
        ]

    def calculatePriorsAndLikelihoods(
        self, mode="init", CalcNull1=False, CalcNull2=False
    ):
        if mode == "DeltaCalc":
            interval1_to_be_merged = self.nitj.copy()
            interval2_to_be_merged = self.next.nitj.copy()
            NITJ_Interval = list(
                map(add, interval1_to_be_merged, interval2_to_be_merged)
            )
        else:
            NITJ_Interval = self.nitj.copy()

        # Likelihood 1
        likelihood_denum = 0
        for j in range(2):
            likelihood_denum += log_fact((NITJ_Interval[j] + NITJ_Interval[j + 2]))
        likelihood1_tmp = log_fact(np.sum(NITJ_Interval)) - likelihood_denum
        # Likelihood 2
        res_t = 0
        for t in range(2):
            likelihood_denum = 0
            for j in range(2):
                likelihood_denum += log_fact(NITJ_Interval[t + t * 1 + j])
            res_t += (
                log_fact(NITJ_Interval[t + t * 1] + NITJ_Interval[t + t * 1 + 1])
                - likelihood_denum
            )
        likelihood2_tmp = res_t
        # Prior 1
        prior1_tmp = log_binomial_coefficient(np.sum(NITJ_Interval) + 1, 1)
        # Prior 2
        res_t = 0
        for t in range(2):
            res_t_temp = log_binomial_coefficient(
                (NITJ_Interval[t + t * 1] + NITJ_Interval[t + t * 1 + 1]) + 1, 1
            )
            res_t += res_t_temp
        prior2_tmp = res_t

        if mode == "DeltaCalc":
            if (prior1_tmp + likelihood1_tmp) < (prior2_tmp + likelihood2_tmp):
                self.righMergeW = 0
            else:
                self.righMergeW = 1
            SumOfPriorsAndLikelihoods = (
                (1 - self.righMergeW) * prior1_tmp
                + self.righMergeW * prior2_tmp
                + (1 - self.righMergeW) * likelihood1_tmp
                + (self.righMergeW) * likelihood2_tmp
            )

        elif mode == "MergeAndUpdate":
            if (
                ((prior1_tmp + likelihood1_tmp) < (prior2_tmp + likelihood2_tmp))
                or CalcNull1 == True
                or CalcNull2 == True
            ):
                if CalcNull2 == True:
                    self.righMergeW = 1
                else:
                    self.righMergeW = 0
            else:
                self.righMergeW = 1
            self.Prior1 = (1 - self.righMergeW) * prior1_tmp
            self.Prior2 = self.righMergeW * prior2_tmp
            self.Likelihood1 = (1 - self.righMergeW) * likelihood1_tmp
            self.Likelihood2 = (self.righMergeW) * likelihood2_tmp
            self.W = self.righMergeW
            SumOfPriorsAndLikelihoods = (
                self.Prior1 + self.Prior2 + self.Likelihood1 + self.Likelihood2
            )
            if CalcNull1 == True:
                self.SumOfPriorsAndLikelihoodsWithZeroW = SumOfPriorsAndLikelihoods
            if CalcNull2 == True:
                self.SumOfPriorsAndLikelihoodsWithOneW = SumOfPriorsAndLikelihoods
            else:
                self.SumOfPriorsAndLikelihoods = SumOfPriorsAndLikelihoods

        elif mode == "init":
            SumOfPriorsAndLikelihoods = (
                ((1 - self.W) * prior1_tmp)
                + (self.W * prior2_tmp)
                + ((1 - self.W) * likelihood1_tmp)
                + ((self.W) * likelihood2_tmp)
            )
            self.SumOfPriorsAndLikelihoods = SumOfPriorsAndLikelihoods
            self.Prior1 = (1 - self.W) * prior1_tmp
            self.Prior2 = self.W * prior2_tmp
            self.Likelihood1 = (1 - self.W) * likelihood1_tmp
            self.Likelihood2 = (self.W) * likelihood2_tmp
        return SumOfPriorsAndLikelihoods


class DLL:
    def __init__(self):
        self.head = None
        self.tail = None
        self.N = None
        self.MODL_value = None
        self.count = 0
        self.I = self.count

    # append to the end of the list
    def append(self, listData):
        if self.head == None:
            self.head = _Interval(listData)
            self.tail = self.head
            self.count += 1
            self.I += 1
            return
        self.tail.next = _Interval(listData)
        self.tail.next.previous = self.tail
        self.tail = self.tail.next
        self.count += 1
        self.I += 1

    def insert(self, interval, index):
        nitj = interval.nitj
        IncludedRightFrontier = interval.IncludedRightFrontier
        W_value = interval.W

        if (index > self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        if index == self.count:
            self.append([nitj, IncludedRightFrontier, W_value])
            return

        if index == 0:
            self.head.previous = _Interval([nitj, IncludedRightFrontier, W_value])
            self.head.previous.next = self.head
            self.head = self.head.previous
            self.count += 1
            self.I += 1
            return

        start = self.head
        for _ in range(index):
            start = start.next
        start.previous.next = _Interval([nitj, IncludedRightFrontier, W_value])
        start.previous.next.previous = start.previous
        start.previous.next.next = start
        start.previous = start.previous.next
        self.count += 1
        self.I += 1
        return

    def removeInterval(self, interval):
        if interval == self.head:
            self.head = self.head.next
            self.head.previous = None
            self.count -= 1
            self.I -= 1
            return

        if interval == self.tail:
            self.tail = self.tail.previous
            self.tail.next = None
            self.count -= 1
            self.I -= 1
            return

        interval.previous.next, interval.next.previous = (
            interval.next,
            interval.previous,
        )
        self.count -= 1
        self.I -= 1
        return

    def remove(self, index):
        if (index >= self.count) | (index < 0):
            raise ValueError(f"Index out of range: {index}, size: {self.count}")

        if index == 0:
            self.head = self.head.next
            self.head.previous = None
            self.count -= 1
            self.I -= 1
            return

        if index == (self.count - 1):
            self.tail = self.tail.previous
            self.tail.next = None
            self.count -= 1
            self.I -= 1
            return

        start = self.head
        for i in range(index):
            start = start.next
        start.previous.next, start.next.previous = start.next, start.previous
        self.count -= 1
        self.I -= 1
        return

    def size(self):
        return self.count

    def getNth(self, index):
        current = self.head  # Initialise temp
        count = 0  # Index of current interval

        # Loop while end of linked list is not reached
        while current:
            if count == index:
                return current
            count += 1
            current = current.next
        return 0

    def listprint(self, interval):
        while interval is not None:
            print("=======================================")
            print("=======================================")
            print("NITJ ", interval.nitj)
            print("Frontier ", interval.IncludedRightFrontier)
            print("W value ", interval.W)
            print("Sum of priors and likelihoods ", interval.SumOfPriorsAndLikelihoods)
            print("interval.Prior1 ", interval.Prior1)
            print("interval.Prior2 ", interval.Prior2)
            print("interval.Likelihood 1 ", interval.Likelihood1)
            print("interval.Likelihood 2 ", interval.Likelihood2)
            print("=======================================")
            print("=======================================\n")
            interval = interval.next

    def getDiscretizationCriterionValue(self):
        start = self.head
        summations = 0
        while start:
            summations += start.SumOfPriorsAndLikelihoods
            start = start.next
        summations = (
            summations
            + log_binomial_coefficient(self.N + self.count - 1, self.count - 1)
            + self.count * log(2)
            + log(self.N)
        )
        self.MODL_value = summations
        return summations

    def getSortedListOfAddressAndRightMergeValue(self, interval):
        AddressAndVal = []  # list of lists
        while interval:
            rightMergeVal = interval.rightMergeDelta
            AddressAndVal.append((rightMergeVal, interval))
            interval = interval.next
        AddressAndVal = sorted(AddressAndVal, key=itemgetter(0))
        return sorted(AddressAndVal, key=itemgetter(0))

    def getDiscretizationInfo(self):
        start = self.head
        IntervalBounds = []
        ListOfWs = []
        while start:
            IntervalBounds.append(start.IncludedRightFrontier)
            ListOfWs.append(start.W)
            start = start.next

        return [self.MODL_value, self.count, IntervalBounds, ListOfWs]


def createElementaryDiscretization(dll, data):
    """
    params
    data : list of lists, each internal list contains [data value, treatment value, y value]

    returns
    1- a list of lists. Each internal list represents an interval and contains Effectifs of T0J0, T0J1, T1J0, T1J1 respectively.
    2- frontier value per interval
    3- W : list containing the Wi value for each interval, the initial discretization has Wi = 0 for all i
    """
    start_counter(0)
    # This is a list of lists, each internal list represents an interval and contains Effectifs of T0J0, T0J1, T1J0, T1J1 respectively
    prev = None
    i = -1
    for intervalList in data:
        if intervalList[0] != prev:
            if i != -1:
                dll.tail.calculatePriorsAndLikelihoods()
            dll.append([[0, 0, 0, 0], intervalList[0], 0])
            prev = intervalList[0]
            i += 1
        dll.tail.nitj[int((intervalList[1] * 2) + intervalList[2])] += 1
    dll.tail.calculatePriorsAndLikelihoods()
    stop_counter(0)

    # calculating whole criterion
    return dll, dll.getDiscretizationCriterionValue(), dll.size()


def criterion_delta_for_one_adjacent_interval_merge(
    dll, LeftIntervalNode, indexPassed=True, mode="DeltaCalc"
):
    start_counter(1)
    if indexPassed:
        LeftIntervalNode = dll.getNth(LeftIntervalNode)

    RightIntervalNode = LeftIntervalNode.next
    if RightIntervalNode == None:
        # it means the left interval node is the last node and cannot be merged
        LeftIntervalNode.rightMergeDelta = -1
        return
    OldLeftIntervalNodeCriterion = LeftIntervalNode.SumOfPriorsAndLikelihoods

    LeftIntervalNodeCriterion = LeftIntervalNode.calculatePriorsAndLikelihoods(mode)

    RightIntervalNodeCriterion = RightIntervalNode.SumOfPriorsAndLikelihoods

    newCriterionValue = (
        dll.MODL_value
        - RightIntervalNodeCriterion
        - OldLeftIntervalNodeCriterion
        - log_binomial_coefficient(dll.N + dll.I - 1, dll.I - 1)
        - ((dll.I) * log(2))
        + log_binomial_coefficient(dll.N + dll.I - 2, dll.I - 2)
        + ((dll.I - 1) * log(2))
        + LeftIntervalNodeCriterion
    )

    LeftIntervalNode.rightMergeDelta = dll.MODL_value - newCriterionValue
    stop_counter(1)


def compute_criterion_delta_for_all_possible_merges(dll):
    start_counter(2)
    size = dll.size()
    for i in range(size):
        criterion_delta_for_one_adjacent_interval_merge(dll, i)
    stop_counter(2)


def greedySearch(best_merges, Intervals, N):
    start_counter(3)
    for step in range(N):
        if len(best_merges) > 0:
            best_merge_tuple = best_merges.pop()
        else:
            break
        if best_merge_tuple[0] >= 0:
            IntervalToBeMerged = best_merge_tuple[1]
            IntervalRightOfTheMerge = IntervalToBeMerged.next

            interval1_to_be_merged = IntervalToBeMerged.nitj.copy()
            interval2_to_be_merged = IntervalRightOfTheMerge.nitj.copy()
            start_counter(6)

            MergedIntervals = list(
                map(add, interval1_to_be_merged, interval2_to_be_merged)
            )
            stop_counter(6)

            IntervalToBeMerged.nitj = MergedIntervals
            IntervalToBeMerged.IncludedRightFrontier = (
                IntervalRightOfTheMerge.IncludedRightFrontier
            )

            OldLeftIntervalNodeCriterion = IntervalToBeMerged.SumOfPriorsAndLikelihoods
            OldRightIntervalNodeCriterion = (
                IntervalRightOfTheMerge.SumOfPriorsAndLikelihoods
            )

            LeftIntervalNodeCriterion = (
                IntervalToBeMerged.calculatePriorsAndLikelihoods(mode="MergeAndUpdate")
            )  # it will update W, Priors, lkelihoods and SumOfPriorsAndLikelihoods
            Intervals.MODL_value = (
                Intervals.MODL_value
                - OldRightIntervalNodeCriterion
                - OldLeftIntervalNodeCriterion
                - log_binomial_coefficient(
                    Intervals.N + Intervals.I - 1, Intervals.I - 1
                )
                - ((Intervals.I) * log(2))
                + log_binomial_coefficient(
                    Intervals.N + Intervals.I - 2, Intervals.I - 2
                )
                + ((Intervals.I - 1) * log(2))
                + LeftIntervalNodeCriterion
            )

            IntervalRight_to_new_interval = IntervalToBeMerged.next
            IntervalLeft_to_new_interval = IntervalToBeMerged.previous

            best_merges.remove(
                (IntervalRightOfTheMerge.rightMergeDelta, IntervalRightOfTheMerge)
            )
            Intervals.removeInterval(IntervalRightOfTheMerge)

            if IntervalRight_to_new_interval == None:  # last Interval
                best_merges.remove(
                    (
                        IntervalLeft_to_new_interval.rightMergeDelta,
                        IntervalLeft_to_new_interval,
                    )
                )
                criterion_delta_for_one_adjacent_interval_merge(
                    Intervals, IntervalLeft_to_new_interval, indexPassed=False
                )
                best_merges.add(
                    (
                        IntervalLeft_to_new_interval.rightMergeDelta,
                        IntervalLeft_to_new_interval,
                    )
                )
            elif IntervalLeft_to_new_interval == None:
                criterion_delta_for_one_adjacent_interval_merge(
                    Intervals, IntervalToBeMerged, indexPassed=False
                )
                best_merges.add(
                    (IntervalToBeMerged.rightMergeDelta, IntervalToBeMerged)
                )
            else:
                best_merges.remove(
                    (
                        IntervalLeft_to_new_interval.rightMergeDelta,
                        IntervalLeft_to_new_interval,
                    )
                )
                criterion_delta_for_one_adjacent_interval_merge(
                    Intervals, IntervalLeft_to_new_interval, indexPassed=False
                )
                best_merges.add(
                    (
                        IntervalLeft_to_new_interval.rightMergeDelta,
                        IntervalLeft_to_new_interval,
                    )
                )
                criterion_delta_for_one_adjacent_interval_merge(
                    Intervals, IntervalToBeMerged, indexPassed=False
                )
                best_merges.add(
                    (IntervalToBeMerged.rightMergeDelta, IntervalToBeMerged)
                )
        else:
            break
    stop_counter(3)


def merge(interval, Intervals, NumberOfMerges=1):
    NeighboursToMerge = [interval]
    MergedIntervals = interval.nitj.copy()
    SumOfOldPriorsAndLikelihoods = interval.SumOfPriorsAndLikelihoods
    for i in range(NumberOfMerges):
        lastInterval = NeighboursToMerge[-1].next
        NeighboursToMerge.append(lastInterval)
        MergedIntervals = list(
            map(add, NeighboursToMerge[-1].nitj.copy(), MergedIntervals)
        )
        SumOfOldPriorsAndLikelihoods += NeighboursToMerge[-1].SumOfPriorsAndLikelihoods

    interval.nitj = MergedIntervals
    interval.IncludedRightFrontier = NeighboursToMerge[-1].IncludedRightFrontier
    # NOW WE HAVE TO SEARCH for the old values of the sum of prior and likelihoods !!!!
    LeftIntervalNodeCriterion = interval.calculatePriorsAndLikelihoods(
        mode="MergeAndUpdate"
    )  # it will update W, Priors, lkelihoods and SumOfPriorsAndLikelihoods
    Intervals.MODL_value = (
        Intervals.MODL_value
        - SumOfOldPriorsAndLikelihoods
        - log_binomial_coefficient(Intervals.N + Intervals.I - 1, Intervals.I - 1)
        - ((Intervals.I) * log(2))
        + log_binomial_coefficient(
            Intervals.N + Intervals.I - NumberOfMerges - 1,
            Intervals.I - NumberOfMerges - 1,
        )
        + ((Intervals.I - NumberOfMerges) * log(2))
        + LeftIntervalNodeCriterion
    )

    for i in range(
        1, len(NeighboursToMerge)
    ):  # Note the first element is the current interval that we are merging, no need to remove it!
        Intervals.removeInterval(NeighboursToMerge[i])


def splitInterval(interval, Intervals, data, i):  # i is interval index in IntervalsList
    if interval == Intervals.head:
        IncludingLeftBorder = True
        LeftBound = data[0][0]
    else:
        IncludingLeftBorder = False
        LeftBound = interval.previous.IncludedRightFrontier
    RightBound = interval.IncludedRightFrontier
    uniqueValuesInBothIntervals = list(
        data.irange_key(LeftBound, RightBound, (IncludingLeftBorder, True))
    )
    uniqueValuesInBothIntervals = list(map(itemgetter(0), uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals = list(set(uniqueValuesInBothIntervals))
    uniqueValuesInBothIntervals.sort()
    Splits = {}
    LeftAndRightIntervalOfSplits = {}
    previousLeftInterval = [0, 0, 0, 0]
    prevVal = None

    for val in uniqueValuesInBothIntervals:
        if prevVal == None:
            leftSplit = list(
                data.irange_key(LeftBound, val, (IncludingLeftBorder, True))
            )
            leftInterval = [0, 0, 0, 0]
            for intervalList in leftSplit:
                leftInterval[int((intervalList[1] * 2) + intervalList[2])] += 1
        else:
            leftSplit = list(data.irange_key(prevVal, val, (False, True)))
            leftInterval = [0, 0, 0, 0]
            for intervalList in leftSplit:
                leftInterval[int((intervalList[1] * 2) + intervalList[2])] += 1
            leftInterval = list(map(add, previousLeftInterval, leftInterval))

        prevVal = val
        previousLeftInterval = leftInterval

        rightInterval = list(map(sub, interval.nitj, leftInterval))

        LeftInterval = _Interval([leftInterval, val, 0])
        RightInterval = _Interval([rightInterval, interval.IncludedRightFrontier, 0])

        criterion1 = LeftInterval.calculatePriorsAndLikelihoods(mode="MergeAndUpdate")
        criterion2 = RightInterval.calculatePriorsAndLikelihoods(mode="MergeAndUpdate")

        SplitCriterionVal_leftAndRight = (
            Intervals.MODL_value
            - interval.SumOfPriorsAndLikelihoods
            - log_binomial_coefficient(Intervals.N + Intervals.I - 1, Intervals.I - 1)
            - ((Intervals.I) * log(2))
            + criterion1
            + criterion2
            + log_binomial_coefficient(Intervals.N + Intervals.I, Intervals.I)
            + ((Intervals.I + 1) * log(2))
        )

        if SplitCriterionVal_leftAndRight < Intervals.MODL_value:
            Splits[val] = SplitCriterionVal_leftAndRight
            LeftAndRightIntervalOfSplits[val] = [leftInterval, rightInterval]
    splitDone = False
    bestSplit = None
    if Splits:
        bestSplit = min(Splits, key=Splits.get)  # To be optimized maybe
        leftInterval = LeftAndRightIntervalOfSplits[bestSplit][0]
        rightInterval = LeftAndRightIntervalOfSplits[bestSplit][1]

        LeftInterval = interval
        rightBoundOfTheRightInterval = interval.IncludedRightFrontier
        Intervals.insert(
            _Interval([rightInterval, rightBoundOfTheRightInterval, 0]), i + 1
        )
        RightInterval = Intervals.getNth(i + 1)

        LeftInterval.nitj = leftInterval
        LeftInterval.IncludedRightFrontier = bestSplit

        LeftInterval.calculatePriorsAndLikelihoods(
            mode="MergeAndUpdate"
        )  # it will update W, Priors, lkelihoods and SumOfPriorsAndLikelihoods
        RightInterval.calculatePriorsAndLikelihoods(
            mode="MergeAndUpdate"
        )  # it will update W, Priors, lkelihoods and SumOfPriorsAndLikelihoods

        Intervals.MODL_value = Splits[bestSplit]
        splitDone = True
    return splitDone, bestSplit, Intervals


def PostOptimizationToBeRepeated(Intervals, data, i=0):
    data = SortedKeyList(data, key=itemgetter(0))
    interval = Intervals.head
    OlD_MODL_CRITERION_VALUE = Intervals.MODL_value
    CriterionValueAfterOptimization = None
    LoopingCount = 0
    # Merge Split
    while True:
        LoopingCount += 1
        i = 0
        while True:
            interval = Intervals.getNth(i)
            if interval == 0:
                break
            SplitDone, SplitVal, Intervals = splitInterval(interval, Intervals, data, i)
            if SplitDone:
                i += 2
            else:
                i += 1

        i = 0
        while True:
            interval = Intervals.getNth(i)
            if interval == 0:
                break
            if interval == Intervals.tail:
                break
            if interval.next == None:
                # Arrived to the most left interval
                break

            ################################################
            OldSplitVal = interval.IncludedRightFrontier
            ####MERGESPLIT###############
            merge(interval, Intervals, 1)

            # Merge finished successfully
            SplitAfterMergeDone, SplitVal, Intervals = splitInterval(
                interval, Intervals, data, i
            )
            ################################################
            if SplitAfterMergeDone and SplitVal != OldSplitVal:
                i += 2
            else:
                i += 1

        # MergeMegeSplit
        i = 0
        while True:
            interval = Intervals.getNth(i)

            if interval == 0:
                break
            if interval == Intervals.tail:
                break
            if interval.next == None:
                # Arrived to the most left interval
                break
            if interval.next.next == None:
                break

            OriginalIntervalsList = copyList(Intervals)
            ################################################
            ####MERGESPLIT###############
            merge(interval, Intervals, 2)

            OriginalIntervalsListAfterMerge = copyList(Intervals)

            # Merge finished successfully
            SplitAfterMergeDone, SplitVal, Intervals = splitInterval(
                interval, Intervals, data, i
            )
            ################################################
            if (
                Intervals.MODL_value < OriginalIntervalsList.MODL_value
                and Intervals.MODL_value < OriginalIntervalsListAfterMerge.MODL_value
            ):
                i += 2
            elif (
                OriginalIntervalsList.MODL_value
                < OriginalIntervalsListAfterMerge.MODL_value
            ):
                Intervals = copyList(OriginalIntervalsList)
                i += 3
            else:
                Intervals = copyList(OriginalIntervalsListAfterMerge)
                i += 1

        CriterionValueAfterOptimization = Intervals.MODL_value

        if round(CriterionValueAfterOptimization, 5) < round(
            OlD_MODL_CRITERION_VALUE, 5
        ):
            OlD_MODL_CRITERION_VALUE = CriterionValueAfterOptimization
            continue
        else:
            break
    return (
        Intervals.size(),
        Intervals.MODL_value,
        Intervals,
        Intervals.getDiscretizationInfo(),
    )


def copyList(DLL_to_be_copied):
    newList = DLL()  # head of the new list
    newList.N = DLL_to_be_copied.N
    newList.MODL_value = DLL_to_be_copied.MODL_value
    current = DLL_to_be_copied.head  # used to iterate over the original list

    while current:
        newList.append(current.getIntervalData())
        current = current.next

    return newList


def CalculateFeatureLevel(Intervals, method="ED"):
    interval = Intervals.head
    AbsoluteSum = 0

    if Intervals.I == 1:
        return 0

    while interval:
        nitj = interval.nitj
        nit0j0 = nitj[0]
        nit0j1 = nitj[1]
        nit1j0 = nitj[2]
        nit1j1 = nitj[3]
        Ni = nit0j0 + nit0j1 + nit1j0 + nit1j1

        try:
            piYT1 = nit1j1 / (nit1j1 + nit1j0)
        except:
            piYT1 = 0
        try:
            piYT0 = nit0j1 / (nit0j1 + nit0j0)
        except:
            piYT0 = 0
        if method == "ED":
            AbsoluteSum += (((piYT1) - (piYT0)) ** 2) * Ni / Intervals.N  # ED
        elif method == "Chi":
            if piYT0 < 0.1**6:
                piYT0 = 0.1**6
            AbsoluteSum += ((((piYT1) - (piYT0)) ** 2) / piYT0) * Ni / Intervals.N
        elif method == "KL":
            if piYT0 < 0.1**6:
                piYT0 = 0.1**6
            elif piYT0 > 1 - 0.1**6:
                piYT0 = 1 - 0.1**6
            AbsoluteSum += ((piYT1) * log(piYT1 / piYT0)) * Ni / Intervals.N
        interval = interval.next
    return AbsoluteSum


def ExecuteGreedySearchAndPostOpt(df):
    treatmentCol_name = df.columns[1]
    y_name = df.columns[2]

    df[treatmentCol_name] = df[treatmentCol_name].astype(int)
    df[y_name] = df[y_name].astype(int)

    df = df.values.tolist()
    df = sorted(df, key=itemgetter(0))
    Intervals = DLL()
    Intervals.N = len(df)

    Intervals, Intervals.MODL_value, I = createElementaryDiscretization(
        Intervals, df
    )  # Elementary discretization

    compute_criterion_delta_for_all_possible_merges(
        Intervals
    )  # Compute the cost of all possible merges of two adjacent intervals

    best_merges = Intervals.getSortedListOfAddressAndRightMergeValue(
        Intervals.head
    )  # Get all the costs of 'all possible merges of two adjacent intervals' sorted
    best_merges = SortedKeyList(best_merges, key=itemgetter(0))

    # Start greedy search
    greedySearch(best_merges, Intervals, Intervals.N)

    # Post Optimization steps
    IntervalsNUM, umodl_val, Intervals, Info = PostOptimizationToBeRepeated(
        Intervals, df
    )

    Bounds = Info[2]
    FeatureLevel_ED = CalculateFeatureLevel(Intervals)

    return [FeatureLevel_ED, Bounds]
