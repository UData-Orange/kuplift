######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import numpy as np
from math import log
from operator import add, sub, itemgetter
from sortedcontainers import SortedKeyList
from .helperfunctions import log_fact, log_binomial_coefficient
from .binary_discretization_functions import start_counter, stop_counter


class _Interval:
    def __init__(self, data):
        if len(data) == 3:
            self.nitj = data[0]
            self.included_right_frontier = data[1]
            self.w = data[2]
            self.right_merge_delta = None
            self.right_merge_w = None
            self.prior_one = None
            self.prior_two = None
            self.likelihood_one = None
            self.likelihood_two = None
            self.next = None
            self.previous = None
            self.sum_of_priors_and_likelihoods = None
            self.sum_of_priors_and_likelihoods_with_zero_w = None
            self.sum_of_priors_and_likelihoods_with_one_w = None
        else:
            self.nitj = data[0]
            self.included_right_frontier = data[1]
            self.w = data[2]
            self.right_merge_delta = data[3]
            self.right_merge_w = data[4]
            self.prior_one = data[5]
            self.prior_two = data[6]
            self.likelihood_one = data[7]
            self.likelihood_two = data[8]
            self.next = data[9]
            self.previous = data[10]
            self.sum_of_priors_and_likelihoods = data[11]
            self.sum_of_priors_and_likelihoods_with_zero_w = None

    def get_interval_data(self):
        return [
            self.nitj,
            self.included_right_frontier,
            self.w,
            self.right_merge_delta,
            self.right_merge_w,
            self.prior_one,
            self.prior_two,
            self.likelihood_one,
            self.likelihood_two,
            self.next,
            self.previous,
            self.sum_of_priors_and_likelihoods,
        ]

    def calculate_priors_and_likelihoods(
        self, mode="init", calc_null_one=False, calc_null_two=False
    ):
        if mode == "DeltaCalc":
            interval_one_to_be_merged = self.nitj.copy()
            interval_two_to_be_merged = self.next.nitj.copy()
            nitj_interval = list(
                map(add, interval_one_to_be_merged, interval_two_to_be_merged)
            )
        else:
            nitj_interval = self.nitj.copy()

        # Likelihood 1
        likelihood_denum = 0
        for j in range(2):
            likelihood_denum += log_fact(
                (nitj_interval[j] + nitj_interval[j + 2])
            )
        likelihood_one_tmp = log_fact(np.sum(nitj_interval)) - likelihood_denum
        # Likelihood 2
        res_t = 0
        for t in range(2):
            likelihood_denum = 0
            for j in range(2):
                likelihood_denum += log_fact(nitj_interval[t + t * 1 + j])
            res_t += (
                log_fact(
                    nitj_interval[t + t * 1] + nitj_interval[t + t * 1 + 1]
                )
                - likelihood_denum
            )
        likelihood_two_tmp = res_t
        # Prior 1
        prior_one_tmp = log_binomial_coefficient(np.sum(nitj_interval) + 1, 1)
        # Prior 2
        res_t = 0
        for t in range(2):
            res_t_temp = log_binomial_coefficient(
                (nitj_interval[t + t * 1] + nitj_interval[t + t * 1 + 1]) + 1,
                1,
            )
            res_t += res_t_temp
        prior_two_tmp = res_t

        if mode == "DeltaCalc":
            if (prior_one_tmp + likelihood_one_tmp) < (
                prior_two_tmp + likelihood_two_tmp
            ):
                self.right_merge_w = 0
            else:
                self.right_merge_w = 1
            sum_of_priors_and_likelihoods = (
                (1 - self.right_merge_w) * prior_one_tmp
                + self.right_merge_w * prior_two_tmp
                + (1 - self.right_merge_w) * likelihood_one_tmp
                + (self.right_merge_w) * likelihood_two_tmp
            )

        elif mode == "MergeAndUpdate":
            if (
                (
                    (prior_one_tmp + likelihood_one_tmp)
                    < (prior_two_tmp + likelihood_two_tmp)
                )
                or calc_null_one
                or calc_null_two
            ):
                if calc_null_two:
                    self.right_merge_w = 1
                else:
                    self.right_merge_w = 0
            else:
                self.right_merge_w = 1
            self.prior_one = (1 - self.right_merge_w) * prior_one_tmp
            self.prior_two = self.right_merge_w * prior_two_tmp
            self.likelihood_one = (1 - self.right_merge_w) * likelihood_one_tmp
            self.likelihood_two = (self.right_merge_w) * likelihood_two_tmp
            self.w = self.right_merge_w
            sum_of_priors_and_likelihoods = (
                self.prior_one
                + self.prior_two
                + self.likelihood_one
                + self.likelihood_two
            )
            if calc_null_one:
                self.sum_of_priors_and_likelihoods_with_zero_w = (
                    sum_of_priors_and_likelihoods
                )
            if calc_null_two:
                self.sum_of_priors_and_likelihoods_with_one_w = (
                    sum_of_priors_and_likelihoods
                )
            else:
                self.sum_of_priors_and_likelihoods = (
                    sum_of_priors_and_likelihoods
                )

        elif mode == "init":
            sum_of_priors_and_likelihoods = (
                ((1 - self.w) * prior_one_tmp)
                + (self.w * prior_two_tmp)
                + ((1 - self.w) * likelihood_one_tmp)
                + ((self.w) * likelihood_two_tmp)
            )
            self.sum_of_priors_and_likelihoods = sum_of_priors_and_likelihoods
            self.prior_one = (1 - self.w) * prior_one_tmp
            self.prior_two = self.w * prior_two_tmp
            self.likelihood_one = (1 - self.w) * likelihood_one_tmp
            self.likelihood_two = (self.w) * likelihood_two_tmp
        return sum_of_priors_and_likelihoods


class _DLL:
    def __init__(self):
        self.head = None
        self.tail = None
        self.n = None
        self.modl_value = None
        self.count = 0
        self.i = self.count

    # append to the end of the list
    def append(self, listData):
        if self.head is None:
            self.head = _Interval(listData)
            self.tail = self.head
            self.count += 1
            self.i += 1
            return
        self.tail.next = _Interval(listData)
        self.tail.next.previous = self.tail
        self.tail = self.tail.next
        self.count += 1
        self.i += 1

    def insert(self, interval, index):
        nitj = interval.nitj
        included_right_frontier = interval.included_right_frontier
        W_value = interval.w

        if (index > self.count) | (index < 0):
            raise ValueError(
                f"Index out of range: {index}, size: {self.count}"
            )

        if index == self.count:
            self.append([nitj, included_right_frontier, W_value])
            return

        if index == 0:
            self.head.previous = _Interval(
                [nitj, included_right_frontier, W_value]
            )
            self.head.previous.next = self.head
            self.head = self.head.previous
            self.count += 1
            self.i += 1
            return

        start = self.head
        for _ in range(index):
            start = start.next
        start.previous.next = _Interval(
            [nitj, included_right_frontier, W_value]
        )
        start.previous.next.previous = start.previous
        start.previous.next.next = start
        start.previous = start.previous.next
        self.count += 1
        self.i += 1
        return

    def remove_interval(self, interval):
        if interval == self.head:
            self.head = self.head.next
            self.head.previous = None
            self.count -= 1
            self.i -= 1
            return

        if interval == self.tail:
            self.tail = self.tail.previous
            self.tail.next = None
            self.count -= 1
            self.i -= 1
            return

        interval.previous.next, interval.next.previous = (
            interval.next,
            interval.previous,
        )
        self.count -= 1
        self.i -= 1
        return

    def remove(self, index):
        if (index >= self.count) | (index < 0):
            raise ValueError(
                f"Index out of range: {index}, size: {self.count}"
            )

        if index == 0:
            self.head = self.head.next
            self.head.previous = None
            self.count -= 1
            self.i -= 1
            return

        if index == (self.count - 1):
            self.tail = self.tail.previous
            self.tail.next = None
            self.count -= 1
            self.i -= 1
            return

        start = self.head
        for i in range(index):
            start = start.next
        start.previous.next, start.next.previous = start.next, start.previous
        self.count -= 1
        self.i -= 1
        return

    def size(self):
        return self.count

    def get_nth(self, index):
        current = self.head  # Initialise temp
        count = 0  # Index of current interval

        # Loop while end of linked list is not reached
        while current:
            if count == index:
                return current
            count += 1
            current = current.next
        return 0

    def get_discretization_criterion_value(self):
        start = self.head
        summations = 0
        while start:
            summations += start.sum_of_priors_and_likelihoods
            start = start.next
        summations = (
            summations
            + log_binomial_coefficient(self.n + self.count - 1, self.count - 1)
            + self.count * log(2)
            + log(self.n)
        )
        self.modl_value = summations
        return summations

    def get_sorted_list_of_address_and_right_merge_value(self, interval):
        address_and_val = []  # list of lists
        while interval:
            rightMergeVal = interval.right_merge_delta
            address_and_val.append((rightMergeVal, interval))
            interval = interval.next
        address_and_val = sorted(address_and_val, key=itemgetter(0))
        return sorted(address_and_val, key=itemgetter(0))

    def get_discretization_info(self):
        start = self.head
        interval_bounds = []
        list_of_ws = []
        while start:
            interval_bounds.append(start.included_right_frontier)
            list_of_ws.append(start.w)
            start = start.next

        return [self.modl_value, self.count, interval_bounds, list_of_ws]


def create_elementary_discretization(dll, data):
    """
    params
    data : list of lists, each internal list contains [data value, treatment value, y value]

    returns
    1- a list of lists. Each internal list represents an interval and contains Effectifs of T0J0, T0J1, T1J0, T1J1 respectively.
    2- frontier value per interval
    3- w : list containing the Wi value for each interval, the initial discretization has Wi = 0 for all i
    """
    start_counter(0)
    # This is a list of lists, each internal list represents an interval and contains Effectifs of T0J0, T0J1, T1J0, T1J1 respectively
    prev = None
    i = -1
    for interval_list in data:
        if interval_list[0] != prev:
            if i != -1:
                dll.tail.calculate_priors_and_likelihoods()
            dll.append([[0, 0, 0, 0], interval_list[0], 0])
            prev = interval_list[0]
            i += 1
        dll.tail.nitj[int((interval_list[1] * 2) + interval_list[2])] += 1
    dll.tail.calculate_priors_and_likelihoods()
    stop_counter(0)

    # calculating whole criterion
    return dll, dll.get_discretization_criterion_value(), dll.size()


def criterion_delta_for_one_adjacent_interval_merge(
    dll, left_interval_node, index_passed=True, mode="DeltaCalc"
):
    start_counter(1)
    if index_passed:
        left_interval_node = dll.get_nth(left_interval_node)

    right_interval_node = left_interval_node.next
    if right_interval_node is None:
        # it means the left interval node is the last node and cannot be merged
        left_interval_node.right_merge_delta = -1
        return
    old_left_interval_node_criterion = (
        left_interval_node.sum_of_priors_and_likelihoods
    )

    left_interval_node_criterion = (
        left_interval_node.calculate_priors_and_likelihoods(mode)
    )

    right_interval_node_criterion = (
        right_interval_node.sum_of_priors_and_likelihoods
    )

    new_criterion_value = (
        dll.modl_value
        - right_interval_node_criterion
        - old_left_interval_node_criterion
        - log_binomial_coefficient(dll.n + dll.i - 1, dll.i - 1)
        - ((dll.i) * log(2))
        + log_binomial_coefficient(dll.n + dll.i - 2, dll.i - 2)
        + ((dll.i - 1) * log(2))
        + left_interval_node_criterion
    )

    left_interval_node.right_merge_delta = dll.modl_value - new_criterion_value
    stop_counter(1)


def compute_criterion_delta_for_all_possible_merges(dll):
    start_counter(2)
    size = dll.size()
    for i in range(size):
        criterion_delta_for_one_adjacent_interval_merge(dll, i)
    stop_counter(2)


def greedy_search(best_merges, intervals, N):
    start_counter(3)
    for step in range(N):
        if len(best_merges) > 0:
            best_merge_tuple = best_merges.pop()
        else:
            break
        if best_merge_tuple[0] >= 0:
            interval_to_be_merged = best_merge_tuple[1]
            interval_right_of_the_merge = interval_to_be_merged.next

            interval_one_to_be_merged = interval_to_be_merged.nitj.copy()
            interval_two_to_be_merged = interval_right_of_the_merge.nitj.copy()
            start_counter(6)

            merged_intervals = list(
                map(add, interval_one_to_be_merged, interval_two_to_be_merged)
            )
            stop_counter(6)

            interval_to_be_merged.nitj = merged_intervals
            interval_to_be_merged.included_right_frontier = (
                interval_right_of_the_merge.included_right_frontier
            )

            old_left_interval_node_criterion = (
                interval_to_be_merged.sum_of_priors_and_likelihoods
            )
            old_right_interval_node_criterion = (
                interval_right_of_the_merge.sum_of_priors_and_likelihoods
            )

            left_interval_node_criterion = interval_to_be_merged.calculate_priors_and_likelihoods(
                mode="MergeAndUpdate"
            )  # it will update w, Priors, lkelihoods and sum_of_priors_and_likelihoods
            intervals.modl_value = (
                intervals.modl_value
                - old_right_interval_node_criterion
                - old_left_interval_node_criterion
                - log_binomial_coefficient(
                    intervals.n + intervals.i - 1, intervals.i - 1
                )
                - ((intervals.i) * log(2))
                + log_binomial_coefficient(
                    intervals.n + intervals.i - 2, intervals.i - 2
                )
                + ((intervals.i - 1) * log(2))
                + left_interval_node_criterion
            )

            interval_right_to_new_interval = interval_to_be_merged.next
            interval_left_to_new_interval = interval_to_be_merged.previous

            best_merges.remove(
                (
                    interval_right_of_the_merge.right_merge_delta,
                    interval_right_of_the_merge,
                )
            )
            intervals.remove_interval(interval_right_of_the_merge)

            if interval_right_to_new_interval is None:  # last Interval
                best_merges.remove(
                    (
                        interval_left_to_new_interval.right_merge_delta,
                        interval_left_to_new_interval,
                    )
                )
                criterion_delta_for_one_adjacent_interval_merge(
                    intervals,
                    interval_left_to_new_interval,
                    index_passed=False,
                )
                best_merges.add(
                    (
                        interval_left_to_new_interval.right_merge_delta,
                        interval_left_to_new_interval,
                    )
                )
            elif interval_left_to_new_interval is None:
                criterion_delta_for_one_adjacent_interval_merge(
                    intervals, interval_to_be_merged, index_passed=False
                )
                best_merges.add(
                    (
                        interval_to_be_merged.right_merge_delta,
                        interval_to_be_merged,
                    )
                )
            else:
                best_merges.remove(
                    (
                        interval_left_to_new_interval.right_merge_delta,
                        interval_left_to_new_interval,
                    )
                )
                criterion_delta_for_one_adjacent_interval_merge(
                    intervals,
                    interval_left_to_new_interval,
                    index_passed=False,
                )
                best_merges.add(
                    (
                        interval_left_to_new_interval.right_merge_delta,
                        interval_left_to_new_interval,
                    )
                )
                criterion_delta_for_one_adjacent_interval_merge(
                    intervals, interval_to_be_merged, index_passed=False
                )
                best_merges.add(
                    (
                        interval_to_be_merged.right_merge_delta,
                        interval_to_be_merged,
                    )
                )
        else:
            break
    stop_counter(3)


def merge(interval, intervals, number_of_merges=1):
    neighbours_to_merge = [interval]
    merged_intervals = interval.nitj.copy()
    sum_of_old_priors_and_likelihoods = interval.sum_of_priors_and_likelihoods
    for i in range(number_of_merges):
        last_interval = neighbours_to_merge[-1].next
        neighbours_to_merge.append(last_interval)
        merged_intervals = list(
            map(add, neighbours_to_merge[-1].nitj.copy(), merged_intervals)
        )
        sum_of_old_priors_and_likelihoods += neighbours_to_merge[
            -1
        ].sum_of_priors_and_likelihoods

    interval.nitj = merged_intervals
    interval.included_right_frontier = neighbours_to_merge[
        -1
    ].included_right_frontier
    # NOW WE HAVE TO SEARCH for the old values of the sum of prior and likelihoods !!!!
    left_interval_node_criterion = interval.calculate_priors_and_likelihoods(
        mode="MergeAndUpdate"
    )  # it will update w, Priors, lkelihoods and sum_of_priors_and_likelihoods
    intervals.modl_value = (
        intervals.modl_value
        - sum_of_old_priors_and_likelihoods
        - log_binomial_coefficient(
            intervals.n + intervals.i - 1, intervals.i - 1
        )
        - ((intervals.i) * log(2))
        + log_binomial_coefficient(
            intervals.n + intervals.i - number_of_merges - 1,
            intervals.i - number_of_merges - 1,
        )
        + ((intervals.i - number_of_merges) * log(2))
        + left_interval_node_criterion
    )

    for i in range(
        1, len(neighbours_to_merge)
    ):  # Note the first element is the current interval that we are merging, no need to remove it!
        intervals.remove_interval(neighbours_to_merge[i])


def split_interval(
    interval, intervals, data, i
):  # i is interval index in intervalsList
    if interval == intervals.head:
        including_left_border = True
        left_bound = data[0][0]
    else:
        including_left_border = False
        left_bound = interval.previous.included_right_frontier
    right_bound = interval.included_right_frontier
    unique_values_in_both_intervals = list(
        data.irange_key(left_bound, right_bound, (including_left_border, True))
    )
    unique_values_in_both_intervals = list(
        map(itemgetter(0), unique_values_in_both_intervals)
    )
    unique_values_in_both_intervals = list(
        set(unique_values_in_both_intervals)
    )
    unique_values_in_both_intervals.sort()
    splits = {}
    left_and_right_interval_of_splits = {}
    previous_left_interval = [0, 0, 0, 0]
    prev_val = None

    for val in unique_values_in_both_intervals:
        if prev_val is None:
            left_split = list(
                data.irange_key(left_bound, val, (including_left_border, True))
            )
            left_interval = [0, 0, 0, 0]
            for interval_list in left_split:
                left_interval[
                    int((interval_list[1] * 2) + interval_list[2])
                ] += 1
        else:
            left_split = list(data.irange_key(prev_val, val, (False, True)))
            left_interval = [0, 0, 0, 0]
            for interval_list in left_split:
                left_interval[
                    int((interval_list[1] * 2) + interval_list[2])
                ] += 1
            left_interval = list(
                map(add, previous_left_interval, left_interval)
            )

        prev_val = val
        previous_left_interval = left_interval

        right_interval = list(map(sub, interval.nitj, left_interval))

        Left_interval = _Interval([left_interval, val, 0])
        Right_interval = _Interval(
            [right_interval, interval.included_right_frontier, 0]
        )

        criterion_one = Left_interval.calculate_priors_and_likelihoods(
            mode="MergeAndUpdate"
        )
        criterion_two = Right_interval.calculate_priors_and_likelihoods(
            mode="MergeAndUpdate"
        )

        split_criterion_val_left_and_right = (
            intervals.modl_value
            - interval.sum_of_priors_and_likelihoods
            - log_binomial_coefficient(
                intervals.n + intervals.i - 1, intervals.i - 1
            )
            - ((intervals.i) * log(2))
            + criterion_one
            + criterion_two
            + log_binomial_coefficient(intervals.n + intervals.i, intervals.i)
            + ((intervals.i + 1) * log(2))
        )

        if split_criterion_val_left_and_right < intervals.modl_value:
            splits[val] = split_criterion_val_left_and_right
            left_and_right_interval_of_splits[val] = [
                left_interval,
                right_interval,
            ]
    split_done = False
    best_split = None
    if splits:
        best_split = min(splits, key=splits.get)  # To be optimized maybe
        left_interval = left_and_right_interval_of_splits[best_split][0]
        right_interval = left_and_right_interval_of_splits[best_split][1]

        Left_interval = interval
        right_bound_of_the_right_interval = interval.included_right_frontier
        intervals.insert(
            _Interval([right_interval, right_bound_of_the_right_interval, 0]),
            i + 1,
        )
        Right_interval = intervals.get_nth(i + 1)

        Left_interval.nitj = left_interval
        Left_interval.included_right_frontier = best_split

        Left_interval.calculate_priors_and_likelihoods(
            mode="MergeAndUpdate"
        )  # it will update w, Priors, lkelihoods and sum_of_priors_and_likelihoods
        Right_interval.calculate_priors_and_likelihoods(
            mode="MergeAndUpdate"
        )  # it will update w, Priors, lkelihoods and sum_of_priors_and_likelihoods

        intervals.modl_value = splits[best_split]
        split_done = True
    return split_done, best_split, intervals


def post_optimization_to_be_repeated(intervals, data, i=0):
    data = SortedKeyList(data, key=itemgetter(0))
    interval = intervals.head
    old_modl_criterion_value = intervals.modl_value
    criterion_value_after_optimization = None
    looping_count = 0
    # Merge Split
    while True:
        looping_count += 1
        i = 0
        while True:
            interval = intervals.get_nth(i)
            if interval == 0:
                break
            split_done, split_val, intervals = split_interval(
                interval, intervals, data, i
            )
            if split_done:
                i += 2
            else:
                i += 1

        i = 0
        while True:
            interval = intervals.get_nth(i)
            if interval == 0:
                break
            if interval == intervals.tail:
                break
            if interval.next is None:
                # Arrived to the most left interval
                break

            old_split_val = interval.included_right_frontier
            merge(interval, intervals, 1)
            # Merge finished successfully
            split_after_merge_done, split_val, intervals = split_interval(
                interval, intervals, data, i
            )

            if split_after_merge_done and split_val != old_split_val:
                i += 2
            else:
                i += 1

        # MergeMegeSplit
        i = 0
        while True:
            interval = intervals.get_nth(i)

            if interval == 0:
                break
            if interval == intervals.tail:
                break
            if interval.next is None:
                # Arrived to the most left interval
                break
            if interval.next.next is None:
                break

            original_intervals_list = copy_list(intervals)
            merge(interval, intervals, 2)
            original_intervals_list_after_merge = copy_list(intervals)

            # Merge finished successfully
            split_after_merge_done, split_val, intervals = split_interval(
                interval, intervals, data, i
            )

            if (
                intervals.modl_value < original_intervals_list.modl_value
                and intervals.modl_value
                < original_intervals_list_after_merge.modl_value
            ):
                i += 2
            elif (
                original_intervals_list.modl_value
                < original_intervals_list_after_merge.modl_value
            ):
                intervals = copy_list(original_intervals_list)
                i += 3
            else:
                intervals = copy_list(original_intervals_list_after_merge)
                i += 1

        criterion_value_after_optimization = intervals.modl_value

        if round(criterion_value_after_optimization, 5) < round(
            old_modl_criterion_value, 5
        ):
            old_modl_criterion_value = criterion_value_after_optimization
            continue
        else:
            break
    return (
        intervals.size(),
        intervals.modl_value,
        intervals,
        intervals.get_discretization_info(),
    )


def copy_list(dll_to_be_copied):
    new_list = _DLL()  # head of the new list
    new_list.n = dll_to_be_copied.n
    new_list.modl_value = dll_to_be_copied.modl_value
    current = dll_to_be_copied.head  # used to iterate over the original list

    while current:
        new_list.append(current.get_interval_data())
        current = current.next

    return new_list


def calculate_feature_level(intervals, method="ED",get_intervals_uplift=True, min_val_in_var='-inf',max_val_in_var='+inf'):
    interval = intervals.head
    absolute_sum = 0
    
    intervals_vs_uplift=[] #a list of lists, each internal list contains the right boundary and the uplift
    
    if intervals.i == 1:
        if get_intervals_uplift:
            intervals_vs_uplift.append(["] "+str(min_val_in_var)+", " +str(max_val_in_var)+" ]",0,0,0])
            return 0,intervals_vs_uplift
        return 0
    
    previous_frontier=str(round(min_val_in_var,4))
    while interval:
        nitj = interval.nitj
        nit0j0 = nitj[0]
        nit0j1 = nitj[1]
        nit1j0 = nitj[2]
        nit1j1 = nitj[3]
        Ni = nit0j0 + nit0j1 + nit1j0 + nit1j1

        try:
            piYT1 = nit1j1 / (nit1j1 + nit1j0)
        except Exception:
            piYT1 = 0
        try:
            piYT0 = nit0j1 / (nit0j1 + nit0j0)
        except Exception:
            piYT0 = 0
        
        if get_intervals_uplift:
            interval_boundaries = "] "+previous_frontier+" , "+str(round(interval.included_right_frontier,4))+" ]"
            intervals_vs_uplift.append([interval_boundaries,round(piYT1,4),round(piYT0,4),round(round(piYT1,4)-round(piYT0,4),4)])
            previous_frontier=str(round(interval.included_right_frontier,4))
            
        if method == "ED":
            absolute_sum += (((piYT1) - (piYT0)) ** 2) * Ni / intervals.n  # ED
        elif method == "Chi":
            if piYT0 < 0.1**6:
                piYT0 = 0.1**6
            absolute_sum += (
                ((((piYT1) - (piYT0)) ** 2) / piYT0) * Ni / intervals.n
            )
        elif method == "KL":
            if piYT0 < 0.1**6:
                piYT0 = 0.1**6
            elif piYT0 > 1 - 0.1**6:
                piYT0 = 1 - 0.1**6
            absolute_sum += ((piYT1) * log(piYT1 / piYT0)) * Ni / intervals.n
        interval = interval.next
    
    if get_intervals_uplift:
        return absolute_sum,intervals_vs_uplift
    else:
        return absolute_sum


def execute_greedy_search_and_post_opt(df,get_intervals_uplift=True):
    col_name=df.columns[0]
    min_val_in_col=df[col_name].min() # A not very clean approach to add the boundaries in the final details dataframe using the calculate_feature_level()
    max_val_in_var=df[col_name].max()
    treatment_col_name = df.columns[1]
    y_name = df.columns[2]

    df = df.astype({treatment_col_name:'int'})
    df = df.astype({y_name:'int'})

    df = df.values.tolist()
    df = sorted(df, key=itemgetter(0))
    intervals = _DLL()
    intervals.n = len(df)

    intervals, intervals.modl_value, w = create_elementary_discretization(
        intervals, df
    )  # Elementary discretization

    compute_criterion_delta_for_all_possible_merges(
        intervals
    )  # Compute the cost of all possible merges of two adjacent intervals

    best_merges = intervals.get_sorted_list_of_address_and_right_merge_value(
        intervals.head
    )  # Get all the costs of 'all possible merges of two adjacent intervals' sorted
    best_merges = SortedKeyList(best_merges, key=itemgetter(0))

    # Start greedy search
    greedy_search(best_merges, intervals, intervals.n)

    # Post Optimization steps
    (
        intervals_num,
        umodl_val,
        intervals,
        info,
    ) = post_optimization_to_be_repeated(intervals, df)

    bounds = info[2]
    
    if get_intervals_uplift:
        feature_level_ed, intervals_vs_uplift = calculate_feature_level(intervals,get_intervals_uplift=get_intervals_uplift, min_val_in_var=round(min_val_in_col,4),max_val_in_var=round(max_val_in_var,4))
        return [feature_level_ed, bounds,col_name, intervals_vs_uplift]
    else:
        feature_level_ed = calculate_feature_level(intervals,get_intervals_uplift=False)
        return [feature_level_ed, bounds,col_name]

    
