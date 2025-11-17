import pandas as pd
import pytest


@pytest.fixture()
def test_dataframe():
    df = pd.read_csv("data/zenodo_sample.csv")
    return df

@pytest.fixture()
def test_dataframe_with_categorical_variable():
    return pd.read_csv("data/data_uplift_missing.csv")
