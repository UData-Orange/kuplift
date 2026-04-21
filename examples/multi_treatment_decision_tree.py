import logging
from kuplift import MultiTreatmentDecisionTree
import pandas

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 2 treatments
    dataset = pandas.read_csv("./data/data_uplift_missing.csv")
    tree = MultiTreatmentDecisionTree()
    tree.fit(dataset[dataset.columns[:-2]], dataset["TRAITEMENT"], dataset["CIBLE"])
    print(tree.predict(dataset[dataset.columns[:-2]]))

    # 4 treatments
    dataset = pandas.read_csv("./data/dataset.csv")
    tree = MultiTreatmentDecisionTree()
    tree.fit(dataset[dataset.columns[:-2]], dataset["TRAITEMENT"], dataset["CIBLE"])
    print(tree.predict(dataset[dataset.columns[:-2]]))