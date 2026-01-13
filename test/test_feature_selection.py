# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

from kuplift import FeatureSelection


def test_filter(test_dataframe):
    fs = FeatureSelection()

    features = list(test_dataframe.columns[:-2])
    important_vars = fs.filter(test_dataframe[features], test_dataframe["segment"], test_dataframe["visit"])
    assert important_vars == {
        "x2_informative": 0,
        "x4_informative": 0,
        "x6_informative": 0,
        "x7_informative": 0,
        "x8_informative": 0,
        "x11_irrelevant": 0,
        "x12_irrelevant": 0,
        "x13_irrelevant": 0,
        "x14_irrelevant": 0,
        "x15_irrelevant": 0,
        "x16_irrelevant": 0,
        "x17_irrelevant": 0,
        "x18_irrelevant": 0,
        "x19_irrelevant": 0,
        "x20_irrelevant": 0,
        "x21_irrelevant": 0,
        "x22_irrelevant": 0,
        "x23_irrelevant": 0,
        "x24_irrelevant": 0,
        "x25_irrelevant": 0,
        "x26_irrelevant": 0,
        "x27_irrelevant": 0,
        "x28_irrelevant": 0,
        "x29_irrelevant": 0,
        "x30_irrelevant": 0,
        "x1_informative": 0.010107186288141222,
        "x3_informative": 0.010230804170840865,
        "x5_informative": 0.010579495308748063,
        "x10_informative": 0.010998393563985725,
        "x9_informative": 0.011337824503076499,
        "x35_uplift_increase": 0.014427997839683255,
        "x34_uplift_increase": 0.015052299093348126,
        "x33_uplift_increase": 0.015064719742149577,
        "x32_uplift_increase": 0.016302396142469408,
        "x31_uplift_increase": 0.017155249895371802,
        "x36_uplift_increase": 0.018097298951444143,
    }
