# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

import numpy as np
import pandas as pd
import pytest

from kuplift import MultiTreatmentRandomForest


@pytest.fixture
def tiny_mt_dataset():
    X = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "x2": [10, 11, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15],
            "x3": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )
    t = pd.Series(["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"], name="treatment")
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="target")
    return X, t, y


def test_init_validates_n_trees_and_max_features():
    with pytest.raises(ValueError):
        MultiTreatmentRandomForest(n_trees=0)

    with pytest.raises(ValueError):
        MultiTreatmentRandomForest(max_features=0)


def test_fit_raises_on_invalid_data_type():
    rf = MultiTreatmentRandomForest(n_trees=2, max_features=2, random_state=0)
    with pytest.raises(TypeError):
        rf.fit(data=[1, 2, 3], treatment_col=pd.Series([0, 1, 0]), y_col=pd.Series([0, 1, 1]))


def test_fit_raises_on_empty_dataframe():
    rf = MultiTreatmentRandomForest(n_trees=2, max_features=2, random_state=0)
    with pytest.raises(ValueError):
        rf.fit(pd.DataFrame(), treatment_col=pd.Series(dtype=object), y_col=pd.Series(dtype=int))


def test_fit_raises_on_length_mismatch():
    rf = MultiTreatmentRandomForest(n_trees=2, max_features=2, random_state=0)
    X = pd.DataFrame({"x1": [1, 2, 3]})
    t = pd.Series(["A", "B"])      # len 2
    y = pd.Series([0, 1, 0])       # len 3
    with pytest.raises(ValueError):
        rf.fit(X, t, y)


def test_predict_raises_if_not_fitted():
    rf = MultiTreatmentRandomForest()
    X = pd.DataFrame({"x1": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        rf.predict(X)


def test_fit_then_predict_probabilities_shapes(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    rf = MultiTreatmentRandomForest(
        n_trees=3,
        max_features=2,
        random_state=0,
        max_depth=2,
        min_samples_leaf=2,
    )
    rf.fit(X, t, y)

    out_df = rf.predict_probabilities(X, result_type="df", include_negative_probabilities=False)
    out_np = rf.predict_probabilities(X, result_type="ndarray", include_negative_probabilities=False)
    out_ls = rf.predict_probabilities(X, result_type="lists", include_negative_probabilities=False)

    assert isinstance(out_df, pd.DataFrame)
    assert isinstance(out_np, np.ndarray)
    assert isinstance(out_ls, list)

    assert out_df.shape[0] == len(X)
    assert out_np.shape[0] == len(X)
    assert len(out_ls) == len(X)


def test_predict_probabilities_invalid_result_type_raises(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    rf = MultiTreatmentRandomForest(
        n_trees=2,
        max_features=2,
        random_state=0,
        max_depth=2,
        min_samples_leaf=2,
    )
    rf.fit(X, t, y)

    with pytest.raises(ValueError):
        rf.predict_probabilities(X, result_type="bad_type")


def test_predict_dataframe_outputs_blocks(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    rf = MultiTreatmentRandomForest(
        n_trees=3,
        max_features=2,
        random_state=0,
        max_depth=2,
        min_samples_leaf=2,
        control_name="A",
    )
    rf.fit(X, t, y)

    pred = rf.predict(
        X,
        predict_probabilities=True,
        predict_best_treatment=True,
        predict_uplift=True,
    )
    assert isinstance(pred, pd.DataFrame)
    assert pred.shape[0] == len(X)
    assert "best_treatment" in pred.columns
    assert "uplift" in pred.columns


def test_predict_uplift_requires_control_name(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    rf = MultiTreatmentRandomForest(
        n_trees=2,
        max_features=2,
        random_state=0,
        max_depth=2,
        min_samples_leaf=2,
        control_name=None,
    )
    rf.fit(X, t, y)

    with pytest.raises(ValueError):
        rf.predict(
            X,
            predict_probabilities=False,
            predict_best_treatment=False,
            predict_uplift=True,
        )


def test_predict_uplift_control_name_must_exist(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    rf = MultiTreatmentRandomForest(
        n_trees=2,
        max_features=2,
        random_state=0,
        max_depth=2,
        min_samples_leaf=2,
        control_name="DOES_NOT_EXIST",
    )
    rf.fit(X, t, y)

    with pytest.raises(ValueError):
        rf.predict(
            X,
            predict_probabilities=False,
            predict_best_treatment=False,
            predict_uplift=True,
        )
