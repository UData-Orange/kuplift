import pandas as pd
import pytest


@pytest.fixture()
def test_dataframe():
    df = pd.read_csv("data/zenodo_sample.csv")
    return df
