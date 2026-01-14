"""Test configuration

Terminology
-----------
- df: dataframe
- catvar: categorical variable
- opt_ue: optimized univariate encoding
"""

import pandas as pd
import pytest
from kuplift import OptimizedUnivariateEncoding

@pytest.fixture
def test_dataframe():
    df = pd.read_csv("data/zenodo_sample.csv")
    return df

@pytest.fixture
def df_with_catvar():
    return pd.read_csv("data/data_uplift_missing.csv")

@pytest.fixture
def opt_ue_with_catvar(df_with_catvar):
    df = df_with_catvar.astype({"VAR2": object, "CIBLE": object})
    ue = OptimizedUnivariateEncoding()
    ue.fit(df[df.columns[:-2]], df["TRAITEMENT"], df["CIBLE"])
    return ue
