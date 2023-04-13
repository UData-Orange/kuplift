import sys, os
import pandas as pd

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + "/../")

from Kuplift.FeatureSelection import FeatureSelection
from Kuplift.BayesianDecisionTree import BayesianDecisionTree
from Kuplift.BayesianRandomForest import BayesianRandomForest
from Kuplift.UnivariateEncoding import UnivariateEncoding

# change the booleans of the classes to be tested
test_feature_selection = False
test_bayesian_decision_tree = False
test_bayesian_random_forest = True
test_unvariate_encoding = False

df = pd.read_csv("data/zenodo_sample.csv")
print(df.head())

stdout_origin = sys.stdout
sys.stdout = open("OutputResults.txt", "w")

if test_feature_selection:
    fs = FeatureSelection()
    importantvars = fs.filter(df, "segment", "visit")
    print(importantvars)

if test_bayesian_decision_tree:
    features = list(df.columns[:-2])
    tree = BayesianDecisionTree(df, "segment", "visit")
    tree.fit()
    preds = tree.predict(df[features])
    print(list(preds))

if test_bayesian_random_forest:
    features = list(df.columns[:-2])
    forest = BayesianRandomForest(df, "segment", "visit", 4)
    forest.fit()
    preds = forest.predict(df[features])
    print(list(preds))

if test_unvariate_encoding:
    ue = UnivariateEncoding()
    encoded_data = ue.fit_transform(df, "segment", "visit")
    print("encoded_data ", encoded_data)

sys.stdout.close()
sys.stdout = stdout_origin
