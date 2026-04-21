# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

import numpy as np
import pandas as pd
import pytest

from kuplift import MultiTreatmentDecisionTree


def test_init_clamps_maxtreatmentgroups_to_2():
    """The constructor must enforce an upper bound of 2 treatment groups."""
    model = MultiTreatmentDecisionTree(maxtreatmentgroups=999)
    assert model.maxtreatmentgroups == 2


def test_binarize_treatment_returns_int_binary():
    """_binarize_treatment must return integer values in {0,1}."""
    model = MultiTreatmentDecisionTree()

    t = pd.Series(["A", "B", "C", "A", "C", "B"], name="segment")
    out = model._binarize_treatment(t)

    assert sorted(out.unique().tolist()) == [0, 1]
    assert np.issubdtype(out.dtype, np.integer)


def test_binarize_treatment_is_deterministic_on_first_reference():
    """
    First call should pick a deterministic reference treatment (sorted by str),
    then map reference->0 and others->1.
    """
    model = MultiTreatmentDecisionTree()

    t = pd.Series(["B", "A", "C", "A"], name="segment")
    out = model._binarize_treatment(t)

    # sorted modalities by str => "A" reference
    expected = pd.Series([1, 0, 1, 0], name="segment")
    pd.testing.assert_series_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_binarize_treatment_reuses_same_reference_across_calls():
    """Once initialized, binary reference treatment must stay stable across calls."""
    model = MultiTreatmentDecisionTree()

    t1 = pd.Series(["B", "A", "C"], name="segment")
    _ = model._binarize_treatment(t1)
    ref = model._binary_treatment_reference

    t2 = pd.Series(["A", "B", "C", "A", "B"], name="segment")
    out2 = model._binarize_treatment(t2)

    assert ref is not None
    expected2 = (t2 != ref).astype(int)
    pd.testing.assert_series_equal(out2.reset_index(drop=True), expected2.reset_index(drop=True))


def test_strategy_max_level_selects_highest_level():
    """max_level must select exactly the highest-level feature."""
    model = MultiTreatmentDecisionTree(variable_selection_strategy="max_level")
    model.levels_by_variable = {"x1": 0.5, "x2": 3.0, "x3": 1.1}

    class NodeLike:
        def __init__(self):
            self.x = pd.DataFrame(columns=["x1", "x2", "x3"])

    selected = model._select_candidate_attributes_for_node(NodeLike())
    assert selected == ["x2"]


def test_strategy_random_uniform_selects_single_existing_feature():
    """random_uniform must return exactly one feature from node columns."""
    model = MultiTreatmentDecisionTree(
        variable_selection_strategy="random_uniform",
        random_state=42,
    )
    model.levels_by_variable = {"x1": 0.0, "x2": 0.0, "x3": 0.0}

    class NodeLike:
        def __init__(self):
            self.x = pd.DataFrame(columns=["x1", "x2", "x3"])

    selected = model._select_candidate_attributes_for_node(NodeLike())
    assert len(selected) == 1
    assert selected[0] in {"x1", "x2", "x3"}


def test_strategy_random_weighted_prefers_positive_weight():
    """With a single positive weight, random_weighted_by_level should always pick it."""
    model = MultiTreatmentDecisionTree(
        variable_selection_strategy="random_weighted_by_level",
        random_state=123,
    )
    model.levels_by_variable = {"x1": 0.0, "x2": 5.0, "x3": 0.0}

    class NodeLike:
        def __init__(self):
            self.x = pd.DataFrame(columns=["x1", "x2", "x3"])

    draws = [model._select_candidate_attributes_for_node(NodeLike())[0] for _ in range(20)]
    assert set(draws) == {"x2"}


def test_strategy_random_weighted_fallback_uniform_when_all_zero():
    """If all levels are <= 0, weighted strategy must fallback to uniform draw."""
    model = MultiTreatmentDecisionTree(
        variable_selection_strategy="random_weighted_by_level",
        random_state=7,
    )
    model.levels_by_variable = {"x1": 0.0, "x2": 0.0, "x3": 0.0}

    class NodeLike:
        def __init__(self):
            self.x = pd.DataFrame(columns=["x1", "x2", "x3"])

    selected = model._select_candidate_attributes_for_node(NodeLike())
    assert len(selected) == 1
    assert selected[0] in {"x1", "x2", "x3"}


def test_invalid_strategy_raises_value_error():
    """Unsupported variable selection strategy must raise ValueError."""
    model = MultiTreatmentDecisionTree(variable_selection_strategy="not_supported")
    model.levels_by_variable = {"x1": 1.0}

    class NodeLike:
        def __init__(self):
            self.x = pd.DataFrame(columns=["x1", "x2"])

    with pytest.raises(ValueError):
        model._select_candidate_attributes_for_node(NodeLike())


def test_select_candidate_with_single_feature_returns_it():
    """If node has a single feature, strategy must return that feature directly."""
    model = MultiTreatmentDecisionTree(variable_selection_strategy="max_level")
    model.levels_by_variable = {"x1": 123.0}

    class NodeLike:
        def __init__(self):
            self.x = pd.DataFrame(columns=["x1"])

    selected = model._select_candidate_attributes_for_node(NodeLike())
    assert selected == ["x1"]
