import logging
from kuplift import MultiTreatmentDecisionTree
import pandas

LOGLEVEL = logging.INFO
LOGLEVEL = logging.DEBUG
DATASET_PATH = "./data/dataset.csv"
TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"

if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)
    dataset = pandas.read_csv(DATASET_PATH)
    tree = MultiTreatmentDecisionTree()
    tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    print(tree.predict(dataset[dataset.columns[:-2]]))