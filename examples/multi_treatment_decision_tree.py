import logging
from kuplift import MultiTreatmentDecisionTree
import pandas

LOGLEVEL = logging.INFO
LOGLEVEL = logging.DEBUG

TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"

if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)

    # # 2 treatments
    # dataset = pandas.read_csv("./data/data_uplift_missing.csv").astype({"VAR2": object, TREATMENT_NAME: object, TARGET_NAME: object})
    dataset = pandas.read_csv("./data/data_uplift_missing.csv").astype({TREATMENT_NAME: object, TARGET_NAME: object})
    tree = MultiTreatmentDecisionTree()
    tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    # print(tree.predict(dataset[dataset.columns[:-2]]))

    # # 4 treatments
    # dataset = pandas.read_csv("./data/dataset.csv")
    # tree = MultiTreatmentDecisionTree()
    # tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    # print(tree.predict(dataset[dataset.columns[:-2]]))