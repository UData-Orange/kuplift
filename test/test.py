import sys, os
import pandas as pd

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

from Kuplift.FeatureSelection import FeatureSelection
from Kuplift.BayesianDecisionTree import BayesianDecisionTree
from Kuplift.BayesianRandomForest import BayesianRandomForest
from Kuplift.UnivariateEncoding import UnivariateEncoding

# change the booleans of the classes to be tested
testFeatureSelection = True
testBayesianDecisionTree = False
testBayesianRandomForest = False
testUnvariateEncoding = False

df = pd.read_csv("data/zenodo_sample.csv")
print(df.head())

stdoutOrigin=sys.stdout
sys.stdout = open("OutputResults.txt", "w")

if testFeatureSelection:
    fs = FeatureSelection()
    importantvars = fs.filter(df, "segment", "visit")
    print(importantvars)

if testBayesianDecisionTree:
    features = list(df.columns[:-2])
    tree = BayesianDecisionTree(df, "segment", "visit")
    tree.fit()
    preds = tree.predict(df[features])
    print(list(preds))

if testBayesianRandomForest:
    features = list(df.columns[:-2])
    forest = BayesianRandomForest(df, "segment", "visit", 4)
    forest.fit()
    preds = forest.predict(df[features])
    print(list(preds))

if testUnvariateEncoding:
    ue = UnivariateEncoding()
    encoded_data=ue.fit_transform(df, "segment", "visit")
    print('encoded_data ',encoded_data)
    
sys.stdout.close()
sys.stdout=stdoutOrigin    
