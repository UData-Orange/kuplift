######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
import bisect
from math import log
import pandas as pd
import time
from .HelperFunctions import (
    _log_fact_table,
    log_fact,
    log_binomial_coefficient,
)
from operator import itemgetter, add, sub
from sortedcontainers import SortedKeyList

_nb_counter = []
_start_counter = []
_start_time_counter = []
_deltatime_counter = []
_number_of_counters = 25
for i in range(_number_of_counters):
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


def calc_criterion(nitj_interval, null_model=False):
    nitj_interval_sum = sum(nitj_interval)
    Fact_Class0Freq = _log_fact_table[(nitj_interval[0] + nitj_interval[2])]
    Fact_Class1Freq = _log_fact_table[(nitj_interval[1] + nitj_interval[3])]

    Fact_T0Freq = _log_fact_table[(nitj_interval[0] + nitj_interval[1])]
    Fact_T1Freq = _log_fact_table[(nitj_interval[2] + nitj_interval[3])]

    Fact_T0Class0Freq = _log_fact_table[nitj_interval[0]]
    Fact_T0Class1Freq = _log_fact_table[nitj_interval[1]]

    Fact_T1Class0Freq = _log_fact_table[nitj_interval[2]]
    Fact_T1Class1Freq = _log_fact_table[nitj_interval[3]]

    # Likelihood 1 w=0
    start_counter(0)
    likelihood_denum = 0
    j = 0
    likelihood_denum += Fact_Class0Freq
    j = 1
    likelihood_denum += Fact_Class1Freq

    likelihood_one_tmp = _log_fact_table[nitj_interval_sum] - likelihood_denum

    # Likelihood 2 w=1
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

    likelihood_two_tmp = res_t

    # Prior 1 w=0
    prior_one_tmp = log_binomial_coefficient(nitj_interval_sum + 1, 1)
    # Prior 2 w=1
    res_t = 0
    t = 0
    res_t_temp = log_binomial_coefficient(
        (nitj_interval[2 * t] + nitj_interval[2 * t + 1]) + 1, 1
    )
    res_t += res_t_temp
    t = 1
    res_t_temp = log_binomial_coefficient(
        (nitj_interval[2 * t] + nitj_interval[2 * t + 1]) + 1, 1
    )
    res_t += res_t_temp

    prior_two_tmp = res_t
    stop_counter(0)

    right_merge_w = None
    if not null_model:
        if (prior_one_tmp + likelihood_one_tmp) < (
            prior_two_tmp + likelihood_two_tmp
        ):
            right_merge_w = 0
        else:
            right_merge_w = 1
    else:
        if (prior_one_tmp + likelihood_one_tmp) > (
            prior_two_tmp + likelihood_two_tmp
        ):
            right_merge_w = 0
        else:
            right_merge_w = 1

    prior_one = (1 - right_merge_w) * prior_one_tmp
    prior_two = right_merge_w * prior_two_tmp
    likelihood_one = (1 - right_merge_w) * likelihood_one_tmp
    likelihood_two = (right_merge_w) * likelihood_two_tmp
    sum_of_priors_and_likelihoods = (
        prior_one + prior_two + likelihood_one + likelihood_two
    )
    return sum_of_priors_and_likelihoods


def split_interval(
    df,
    col_name,
    treatment_col_name,
    output_col_name,
    null_model_value,
    granularite=16,
):  # i is interval index in IntervalsList
    data = df[[col_name, treatment_col_name, output_col_name]].values.tolist()

    data = SortedKeyList(data, key=itemgetter(0))
    data_nitj = [0, 0, 0, 0]  # The frequency of treatment class
    for interval_list in data:
        data_nitj[int((interval_list[1] * 2) + interval_list[2])] += 1

    N = len(data)
    including_left_border = True
    left_bound = data[0][0]  # The smallest value
    right_bound = data[-1][0]  # The biggest value

    # Get all the unique values in the data i.e All unique values between left
    # and right bounds
    unique_values_in_both_intervals = list(
        data.irange_key(left_bound, right_bound, (including_left_border, True))
    )
    unique_values_in_both_intervals = list(
        map(itemgetter(0), unique_values_in_both_intervals)
    )
    unique_values_in_both_intervals = list(
        set(unique_values_in_both_intervals)
    )
    unique_values_in_both_intervals.sort()  # Sort the unique values

    splits = {}

    previous_left_interval = [0, 0, 0, 0]
    prev_val = None

    # Classical prior vs new prior of rissannen !!!
    prior_rissanen = log(2) + log_binomial_coefficient(N + 1, 1) + 2 * log(2)

    for val in unique_values_in_both_intervals:
        if (len(unique_values_in_both_intervals) <= 1) or (val == right_bound):
            break

        if prev_val is None:  # Enters here only for the first unique value
            # Get a list of data between left_bound and current unique value
            left_split = list(data.irange_key(left_bound, val, (True, True)))
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
            """
            New Left Interval frequencies is the sum of the previous left
            interval (bounded between Smallest value and prev_val) and the
            new left interval (bounded between prev_val and val)
            """
            left_interval = list(
                map(add, previous_left_interval, left_interval)
            )

        # the nitj for the right split (Which we call the right_interval)
        # will be the difference between the old nitj and the left_interval
        prev_val = val
        previous_left_interval = left_interval.copy()

        """
        The new rigt interval is the soustraction of all the data
        and the new left interval
        """
        right_interval = list(map(sub, data_nitj, left_interval))

        # Calculate criterion manually
        start_counter(22)
        criterion_one = calc_criterion(left_interval)  # prior and likelihood
        criterion_two = calc_criterion(right_interval)  # prior and likelihood
        stop_counter(22)

        # MODL value
        start_counter(24)
        split_criterion_val_left_and_right = (
            prior_rissanen + criterion_one + criterion_two
        )
        stop_counter(24)
        # If the MODL value is smaller than the null model value add it
        # to the splits dictionary
        if split_criterion_val_left_and_right < null_model_value:
            splits[val] = split_criterion_val_left_and_right
    best_split = None

    # If dictionary splits contain value, get the minimal one
    if splits:
        best_split = min(splits, key=splits.get)  # To be optimized maybe
        left_split = list(
            data.irange_key(left_bound, best_split, (True, True))
        )
        left_interval = [0, 0, 0, 0]
        for interval_list in left_split:
            left_interval[int((interval_list[1] * 2) + interval_list[2])] += 1
        right_interval = list(map(sub, data_nitj, left_interval))

        index_of_last_row_in_left_data = bisect.bisect_right(
            df[col_name].tolist(), best_split
        )
        left_data = df.iloc[:index_of_last_row_in_left_data, :]
        right_data = df.iloc[index_of_last_row_in_left_data:, :]
        return left_data, right_data, best_split, splits[best_split]
    else:
        split_criterion_val_left_and_right = None
    return -1


def calc_null_model(dff, att, treatment_col_name, output_col_name):
    data = dff[[att, treatment_col_name, output_col_name]].values.tolist()
    data = SortedKeyList(data, key=itemgetter(0))
    data_nitj = [0, 0, 0, 0]  # The frequency of treatment class
    for interval_list in data:
        data_nitj[int((interval_list[1] * 2) + interval_list[2])] += 1
    return (2 * log(2)) + calc_criterion(data_nitj)


def exec(df, attributeToDiscretize, treatment_col_name, output_col_name):
    null_model_value = calc_null_model(
        df, attributeToDiscretize, treatment_col_name, output_col_name
    )
    return split_interval(
        df,
        attributeToDiscretize,
        treatment_col_name,
        output_col_name,
        null_model_value,
    )


def umodl_binary_discretization(data, T, Y, attributeToDiscretize):
    df = pd.DataFrame()
    df = data.copy()
    treatment_col_name = T.name
    output_col_name = Y.name
    df[treatment_col_name] = T
    df[output_col_name] = Y
    df.sort_values(by=attributeToDiscretize, inplace=True)
    df.reset_index(inplace=True, drop=True)
    log_fact(df.shape[0] + 1)
    return exec(df, attributeToDiscretize, treatment_col_name, output_col_name)
