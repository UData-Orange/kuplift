import os
import sys, os
import pandas as pd

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + "/../")

from Kuplift.UnivariateEncoding import UnivariateEncoding

df = pd.read_csv("data/zenodo_sample.csv")


def test_fit_transform():
    ue = UnivariateEncoding()
    encoded_data = ue.fit_transform(df, "segment", "visit")
    # assert encoded_data == []
