import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from kuplift.FeatureSelection import FeatureSelection
from kuplift.BayesianDecisionTree import BayesianDecisionTree
from kuplift.BayesianRandomForest import BayesianRandomForest
from kuplift.UnivariateEncoding import UnivariateEncoding

# change the booleans of the classes to be tested
testUnvariateEncoding = True

df = pd.read_csv("data/zenodo_sample.csv")
print(df.head())

stdoutOrigin = sys.stdout
sys.stdout = open("OutputResults.txt", "w")


if testUnvariateEncoding:
    df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)

    ue = UnivariateEncoding()
    ue.fit(df_train, "segment", "visit")
    encoded_data = ue.transform(df_test)
    list_encoded_data = encoded_data.values.tolist()
    print("list_encoded_data ", list_encoded_data)

sys.stdout.close()
sys.stdout = stdoutOrigin
