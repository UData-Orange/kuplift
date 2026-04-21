# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

import numpy as np
import pandas as pd
import pytest

from kuplift import MultiTreatmentRandomForest


class _FakeTree:
    """Minimal tree-like object for forest predict tests."""

    def __init__(self, preds, criterion):
        self._preds = np.array(preds, dtype=float)
        self.tree_criterion = float(criterion)

    def predict(self, X_test):
        # Return deterministic predictions regardless of input
        return self._preds


@pytest.fixture
def small_features_df():
    return pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4],
            "x2": [1, 2, 3, 4],
            "x3": [10, 20, 30, 40],
            "x4": [7, 7, 8, 9],
        }
    )


def test_init_clamps_maxtreatmentgroups_to_2():
    """Constructor must enforce binary treatment grouping upper bound."""
    rf = MultiTreatmentRandomForest(maxtreatmentgroups=999)
    assert rf.maxtreatmentgroups == 2


def test_sample_feature_columns_without_subset_returns_all_columns(small_features_df):
    """If vars_subset=False, all feature columns should be returned."""
    rf = MultiTreatmentRandomForest(vars_subset=False, random_state=123)
    cols = rf._sample_feature_columns(small_features_df)
    assert set(cols) == set(small_features_df.columns)
    assert len(cols) == len(small_features_df.columns)


def test_sample_feature_columns_with_subset_returns_sqrt_count(small_features_df):
    """If vars_subset=True, number of selected columns should be max(1, floor(sqrt(p)))."""
    rf = MultiTreatmentRandomForest(vars_subset=True, random_state=123)
    cols = rf._sample_feature_columns(small_features_df)
    expected_count = max(1, int(np.sqrt(len(small_features_df.columns))))
    assert len(cols) == expected_count
    assert len(set(cols)) == len(cols)  # no duplicates
    assert set(cols).issubset(set(small_features_df.columns))


def test_predict_raises_if_not_fitted():
    """predict must fail if no trees are available."""
    rf = MultiTreatmentRandomForest()
    X = pd.DataFrame({"x1": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        rf.predict(X)


def test_predict_unweighted_average():
    """Unweighted predict should return the mean of all tree predictions."""
    rf = MultiTreatmentRandomForest()
    rf.list_of_trees = [
        _FakeTree([0.0, 1.0, 2.0], criterion=10.0),
        _FakeTree([1.0, 2.0, 3.0], criterion=20.0),
        _FakeTree([2.0, 3.0, 4.0], criterion=30.0),
    ]
    X = pd.DataFrame({"x1": [1, 2, 3]})

    preds = rf.predict(X, weighted_average=False)
    expected = np.array([1.0, 2.0, 3.0], dtype=float)

    assert isinstance(preds, np.ndarray)
    np.testing.assert_allclose(preds, expected, rtol=1e-12, atol=1e-12)


def test_predict_weighted_average():
    """Weighted predict should use inverse-criterion style weights."""
    rf = MultiTreatmentRandomForest()
    rf.list_of_trees = [
        _FakeTree([1.0, 1.0, 1.0], criterion=1.0),
        _FakeTree([3.0, 3.0, 3.0], criterion=3.0),
    ]
    X = pd.DataFrame({"x1": [1, 2, 3]})

    preds = rf.predict(X, weighted_average=True)

    # Weight computation in class:
    # inv_i = sum(criteria) / criterion_i => [4/1, 4/3] = [4, 1.333...]
    # normalized weights => [0.75, 0.25]
    expected = 0.75 * np.array([1.0, 1.0, 1.0]) + 0.25 * np.array([3.0, 3.0, 3.0])

    np.testing.assert_allclose(preds, expected, rtol=1e-12, atol=1e-12)


def test_predict_weighted_handles_zero_criterion():
    """Weighted predict should remain finite when a tree criterion is zero."""
    rf = MultiTreatmentRandomForest()
    rf.list_of_trees = [
        _FakeTree([1.0, 2.0], criterion=0.0),   # edge case
        _FakeTree([3.0, 4.0], criterion=2.0),
    ]
    X = pd.DataFrame({"x1": [1, 2]})

    preds = rf.predict(X, weighted_average=True)

    assert isinstance(preds, np.ndarray)
    assert np.isfinite(preds).all()
    assert len(preds) == 2


def test_fit_raises_on_invalid_data_type():
    """fit must raise TypeError if data is not a DataFrame."""
    rf = MultiTreatmentRandomForest()
    with pytest.raises(TypeError):
        rf.fit(data=[1, 2, 3], treatment_col=pd.Series([0, 1, 0]), y_col=pd.Series([0, 1, 1]))


def test_fit_raises_on_empty_dataframe():
    """fit must raise ValueError if data is empty."""
    rf = MultiTreatmentRandomForest()
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        rf.fit(empty_df, treatment_col=pd.Series(dtype=int), y_col=pd.Series(dtype=int))
