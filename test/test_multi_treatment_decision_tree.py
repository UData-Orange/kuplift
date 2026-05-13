# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

import numpy as np
import pandas as pd
import pytest

from kuplift import MultiTreatmentDecisionTree


@pytest.fixture
def tiny_mt_dataset():
    # Dataset simple, multi-traitement (3 modalités), target binaire
    X = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "x2": [10, 11, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15],
        }
    )
    t = pd.Series(["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"], name="treatment")
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="target")
    return X, t, y


def test_fit_raises_on_empty_data():
    model = MultiTreatmentDecisionTree()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame(), pd.Series(dtype=object), pd.Series(dtype=int))


def test_fit_raises_on_length_mismatch():
    model = MultiTreatmentDecisionTree()
    X = pd.DataFrame({"x1": [1, 2, 3]})
    t = pd.Series(["A", "B"])       # longueur 2
    y = pd.Series([0, 1, 0])        # longueur 3
    with pytest.raises(ValueError):
        model.fit(X, t, y)


def test_fit_sets_basic_attributes(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    model = MultiTreatmentDecisionTree(max_depth=2, min_samples_leaf=2, random_state=0)
    model.fit(X, t, y)

    assert model.tree is not None
    assert set(model.features) == set(X.columns)
    assert model.treatment_col_name == "treatment"
    assert model.target_col_name == "target"
    assert model.encoder_type in {"OUE", "MTUE"}  # ici MTUE attendu (3 modalités), mais on garde souple


def test_predict_leaf_id_shape(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    model = MultiTreatmentDecisionTree(max_depth=2, min_samples_leaf=2, random_state=0)
    model.fit(X, t, y)

    leaf_ids = model.predict_leaf_id(X)
    assert isinstance(leaf_ids, np.ndarray)
    assert leaf_ids.shape[0] == len(X)
    assert np.issubdtype(leaf_ids.dtype, np.integer)


def test_predict_probabilities_result_types(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    model = MultiTreatmentDecisionTree(max_depth=2, min_samples_leaf=2, random_state=0)
    model.fit(X, t, y)

    out_df = model.predict_probabilities(X, result_type="df")
    out_np = model.predict_probabilities(X, result_type="ndarray")
    out_ls = model.predict_probabilities(X, result_type="lists")

    assert isinstance(out_df, pd.DataFrame)
    assert isinstance(out_np, np.ndarray)
    assert isinstance(out_ls, list)

    assert out_df.shape[0] == len(X)
    assert out_np.shape[0] == len(X)
    assert len(out_ls) == len(X)


def test_predict_probabilities_invalid_result_type_raises(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    model = MultiTreatmentDecisionTree(max_depth=2, min_samples_leaf=2, random_state=0)
    model.fit(X, t, y)

    with pytest.raises(ValueError):
        model.predict_probabilities(X, result_type="bad_type")


def test_predict_best_treatment_returns_series(tiny_mt_dataset):
    X, t, y = tiny_mt_dataset
    model = MultiTreatmentDecisionTree(max_depth=2, min_samples_leaf=2, random_state=0)
    model.fit(X, t, y)

    pred = model.predict_best_treatment(X)
    assert isinstance(pred, pd.Series)
    assert len(pred) == len(X)


def test_methods_raise_when_unfitted():
    model = MultiTreatmentDecisionTree()
    X = pd.DataFrame({"x1": [1, 2, 3]})

    with pytest.raises(RuntimeError):
        model.predict_leaf_id(X)

    with pytest.raises(RuntimeError):
        model.predict_probabilities(X)

    with pytest.raises(RuntimeError):
        model.predict_best_treatment(X)
