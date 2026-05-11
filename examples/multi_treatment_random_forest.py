import logging
from kuplift import RandomForest
import pandas

logging.getLogger("kuplift.mt_univariate_encoding").setLevel(logging.WARNING)

LOGLEVEL = logging.INFO
# LOGLEVEL = logging.DEBUG

TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"
CATEGORICAL_VAR_NAME = "VAR2"

if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)

    # 2 treatments
    dataset = pandas.read_csv("./data/data_uplift_missing.csv").astype({CATEGORICAL_VAR_NAME: object, TREATMENT_NAME: object, TARGET_NAME: object})
    forest = RandomForest(n_trees=5, max_depth=5)
    forest.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])

    # print()
    # print()

    # # 4 treatments
    # dataset = pandas.read_csv("./data/dataset.csv").astype({CATEGORICAL_VAR_NAME: object, TREATMENT_NAME: object, TARGET_NAME: object})
    # forest = RandomForest(n_trees=5, max_depth=5)
    # forest.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])



    # dataset = pandas.read_csv("./data/synthetic_10k.txt", sep="\t").astype({"T": object, "Y": object})
    # forest = RandomForest(n_trees=5, max_depth=5)
    # forest.fit(dataset[dataset.columns[:-2]], dataset["T"], dataset["Y"])