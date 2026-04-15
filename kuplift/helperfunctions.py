# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from math import log, inf
from itertools import count
import string
import random
import khiops.core
from .helperclasses import IntervalPartition, ValGrpPartition

_log_fact_table = []


def log_fact(n):
    """Compute log(fact(n)).

    Parameters
    ----------
    n : int
        Number on which to do the calculation.

    Returns
    -------
    float
        Value of log(fact(n)).
    """
    # Use approximation for large n
    if n > 1e6:
        raise ValueError("Value of n is too large")
    # computation of values, tabulation in private array
    else:
        s = len(_log_fact_table)
        if n >= s:
            if s == 0:
                _log_fact_table.append(0)
            size = len(_log_fact_table)
            while size <= n:
                _log_fact_table.append(log(size) + _log_fact_table[size - 1])
                size = size + 1
        return _log_fact_table[n]


def log_2_star(k):
    """Computes the term log_2*(k)=log_2(k) + log_2(log_2(k)) + ...
    of Rissanen's code for integers so long as the terms are positive.

    Parameters
    ----------
    k : int
        Number on which to do the calculation.

    Returns
    -------
    float
        Value of the result.
    """
    d_log2 = log(2.0)
    d_cost = 0.0
    d_logI = log(1.0 * k) / d_log2

    if k < 1:
        raise ValueError(
            "Universal code is defined for natural numbers over 1"
        )
    else:
        while d_logI > 0:
            d_cost += d_logI
            d_logI = log(d_logI) / d_log2
        return d_cost


def log_binomial_coefficient(n, k):
    """Computes the log of the binomial coefficient (n
                                                     k)
    (log of the total number of combinations of k elements from n).

    Parameters
    ----------
    n : int
        Total number of elements.
    k : int
        Number of selected elements.

    Returns
    -------
    float
        Value of the result.
    """
    # start_counter(5)
    nf = log_fact(n)
    kf = log_fact(k)
    nkf = log_fact(n - k)
    # stop_counter(5)
    return (nf - nkf) - kf


def universal_code_natural_numbers(k):
    """Compute the universal code for integers presented by Rissanen in
    'A Universal Prior for Integers and Estimation by Minimum Description
    Length', Rissanen 1983.

    Parameters
    ----------
    k : int

    Returns
    -------
    float
        Value of the result.
    """
    dC0 = 2.86511  # First value computed following the given estimation
    # formula, as e(3)=65536 + d_log2^5 / (1-d_log2)
    d_log2 = log(2.0)

    if k < 1:
        raise ValueError(
            "Universal code is defined for natural numbers over 1"
        )
    else:
        d_cost = log(dC0) / d_log2  # Initialize code length cost to log_2(dC0)
        d_cost += log_2_star(k)  # Add log_2*(k)
        d_cost *= d_log2  # Go back to the natural log
        return d_cost


def preprocess_data(data, treatment_col="segment", y_col="visit"):
    """preprocess data

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series, default "segment"
        Treatment column.
    y_col : pd.Series, default "visit"
        Outcome column.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe that contains encoded data.
    """
    cols = data.columns
    num_cols = list(data._get_numeric_data().columns)

    if treatment_col in num_cols:
        num_cols.remove(treatment_col)
    if y_col in num_cols:
        num_cols.remove(y_col)

    num_col_index = 0
    while num_col_index < len(num_cols):
        num_col = num_cols[num_col_index]
        if len(data[num_col].value_counts()) < 1000:
            num_cols.remove(num_col)
        else:
            data[num_col] = data[num_col].fillna(data[num_col].min() - 1)
            num_col_index = num_col_index + 1

    categorical_cols = list(set(cols) - set(num_cols))
    if treatment_col in categorical_cols:
        categorical_cols.remove(treatment_col)
    if y_col in categorical_cols:
        categorical_cols.remove(y_col)
    for cat_col in categorical_cols:
        data[cat_col] = data[cat_col].fillna("NAN_VAL")
        dict_val_vs_uplift = {}
        for val in data[cat_col].value_counts().index:
            if val == "NAN_VAL":
                continue
            dataset_slice = data[data[cat_col] == val]
            t0j0 = dataset_slice[
                (dataset_slice[treatment_col] == 0)
                & (dataset_slice[y_col] == 0)
            ].shape[0]
            t0j1 = dataset_slice[
                (dataset_slice[treatment_col] == 0)
                & (dataset_slice[y_col] == 1)
            ].shape[0]
            t1j0 = dataset_slice[
                (dataset_slice[treatment_col] == 1)
                & (dataset_slice[y_col] == 0)
            ].shape[0]
            t1j1 = dataset_slice[
                (dataset_slice[treatment_col] == 1)
                & (dataset_slice[y_col] == 1)
            ].shape[0]

            if (t1j1 + t1j0) == 0:
                uplift_in_this_slice = -1
            elif (t0j1 + t0j1) == 0:
                uplift_in_this_slice = 0
            else:
                uplift_in_this_slice = (t1j1 / (t1j1 + t1j0)) - (
                    t0j1 / (t0j1 + t0j1)
                )
            dict_val_vs_uplift[val] = uplift_in_this_slice
        ordered_dict = {
            k: v
            for k, v in sorted(
                dict_val_vs_uplift.items(), key=lambda item: item[1]
            )
        }

        data[cat_col] = data[cat_col].replace(["NAN_VAL"], -1)
        encoded_i = 0
        for k, v in ordered_dict.items():
            data[cat_col] = data[cat_col].replace([k], encoded_i)
            encoded_i += 1
    data[treatment_col] = data[treatment_col].astype(str)
    return data


def partition_to_rule(partition: list[khiops.core.PartInterval | khiops.core.PartValueGroup], variable: khiops.core.Variable) -> khiops.core.Rule:
    """Format a partition as a Khiops rule string.
    
    Parameters
    ----------
    partition
        A partition to render as Khiops Rule.
    variable: khiops.core.Variable
        A variable used in the rule.
    
    Returns
    -------
    khiops.core.Rule
        A rule.
    """
    if not partition:
        raise ValueError("the partition must contain at least one part")
    match partition[0].part_type():
        case "Interval":
            return khiops.core.Rule(
                "IntervalId",
                khiops.core.Rule(
                    "IntervalBounds",
                    *(interval.upper_bound for interval in partition if not interval.is_missing and not interval.is_right_open)
                ),
                variable
            )
        case "Value group":
            return khiops.core.Rule(
                "GroupId",
                khiops.core.Rule(
                    "ValueGroups",
                    *(
                        khiops.core.Rule("ValueGroup", *(group.values + ([" * "] if group.is_default_part else [])))
                        for group in partition
                    )
                ),
                variable)
        case unsupported_part_type:
            raise ValueError("unsupported part type {!r}".format(unsupported_part_type))


def random_name(charset=string.ascii_letters + string.digits, length=16, *, prefix="", suffix="", maxtries=1000, check=None):
    for i in count(1):
        name = "{}{}{}".format(prefix, "".join(random.choices(charset, k=length)), suffix)
        if check(name):
            return name
        if i == maxtries:
            raise RuntimeError("failed to find a 'check'-passing name after {} attempt(s)".format(i))
        

def random_varname(dictionary: khiops.core.Dictionary, prefix=""):
    return random_name(prefix=prefix, check=lambda name: dictionary.get_variable(name) is None)