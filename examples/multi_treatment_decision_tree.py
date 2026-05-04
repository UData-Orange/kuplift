import logging
from kuplift import MultiTreatmentDecisionTreeV3
import pandas

logging.getLogger("kuplift.multi_treatment_univariate_encoding").setLevel(logging.WARNING)

LOGLEVEL = logging.INFO
LOGLEVEL = logging.DEBUG

TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"

if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)

    # # 2 treatments
    dataset = pandas.read_csv("./data/data_uplift_missing.csv").astype({"VAR2": object, TREATMENT_NAME: object, TARGET_NAME: object})
    tree = MultiTreatmentDecisionTreeV3()
    tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    tree.print_tree(show_path=True)
    print()
    print()
    # print(tree.predict(dataset[dataset.columns[:-2]]))

    # # 4 treatments
    dataset = pandas.read_csv("./data/dataset.csv").astype({"VAR2": object, TREATMENT_NAME: object, TARGET_NAME: object})
    tree = MultiTreatmentDecisionTreeV3()
    tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    tree.print_tree(show_path=True)
    # print(tree.predict(dataset[dataset.columns[:-2]]))