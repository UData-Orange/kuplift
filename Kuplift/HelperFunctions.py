######################################################################################
# Copyright (c) 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Kuplift - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
from math import log

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
        raise ValueError("Universal code is defined for natural numbers over 1")
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
    dC0 = 2.86511  # First value computed following the given estimation formula, as e(3)=65536 + d_log2^5 / (1-d_log2)
    d_log2 = log(2.0)

    if k < 1:
        raise ValueError("Universal code is defined for natural numbers over 1")
    else:
        d_cost = log(dC0) / d_log2  # Initialize code length cost to log_2(dC0)
        d_cost += log_2_star(k)  # Add log_2*(k)
        d_cost *= d_log2  # Go back to the natural log
        return d_cost


def preprocess_data(Data_features, treatment_col="segment", y_col="visit"):
    """Description?

    Parameters
    ----------
    Data_features : pd.Dataframe
        Dataframe containing feature variables.
    treatment_col : pd.Series, optional
        Treatment column.
    y_col : pd.Series, optional
        Outcome column.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe that contains encoded Data_features.
    """
    cols = Data_features.columns
    num_cols = list(Data_features._get_numeric_data().columns)

    num_cols.remove(treatment_col)
    num_cols.remove(y_col)
    for num_col in num_cols:
        if len(Data_features[num_col].value_counts()) < (Data_features.shape[0] / 100):
            num_cols.remove(num_col)
        else:
            Data_features[num_col] = Data_features[num_col].fillna(
                Data_features[num_col].mean()
            )

    categorical_cols = list(set(cols) - set(num_cols))
    if treatment_col in categorical_cols:
        categorical_cols.remove(treatment_col)
    if y_col in categorical_cols:
        categorical_cols.remove(y_col)
    for cat_col in categorical_cols:
        Data_features[cat_col] = Data_features[cat_col].fillna(
            Data_features[cat_col].mode()[0]
        )
        dict_val_vs_uplift = {}
        for val in Data_features[cat_col].value_counts().index:
            dataset_slice = Data_features[Data_features[cat_col] == val]
            t0j0 = dataset_slice[
                (dataset_slice[treatment_col] == 0) & (dataset_slice[y_col] == 0)
            ].shape[0]
            t0j1 = dataset_slice[
                (dataset_slice[treatment_col] == 0) & (dataset_slice[y_col] == 1)
            ].shape[0]
            t1j0 = dataset_slice[
                (dataset_slice[treatment_col] == 1) & (dataset_slice[y_col] == 0)
            ].shape[0]
            t1j1 = dataset_slice[
                (dataset_slice[treatment_col] == 1) & (dataset_slice[y_col] == 1)
            ].shape[0]

            if (t1j1 + t1j0) == 0:
                uplift_in_this_slice = -1
            elif (t0j1 + t0j1) == 0:
                uplift_in_this_slice = 0
            else:
                uplift_in_this_slice = (t1j1 / (t1j1 + t1j0)) - (t0j1 / (t0j1 + t0j1))
            dict_val_vs_uplift[val] = uplift_in_this_slice
        ordered_dict = {
            k: v
            for k, v in sorted(dict_val_vs_uplift.items(), key=lambda item: item[1])
        }
        encoded_i = 0
        for k, v in ordered_dict.items():
            Data_features[cat_col] = Data_features[cat_col].replace([k], encoded_i)
            encoded_i += 1
    Data_features[treatment_col] = Data_features[treatment_col].astype(str)
    return Data_features
