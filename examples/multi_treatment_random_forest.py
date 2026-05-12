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
    forest = RandomForest(n_trees=5, max_depth=5, control_name="T0")
    forest.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    print(forest.predict(pandas.DataFrame([[0.2, 0.4, 0.2, 0.3], [0.6, 0.8, 0.1, 0.1]], columns=["VAR1", "VAR2", "VAR3", "VAR4"])))

    # 4 treatments
    dataset = pandas.read_csv("./data/dataset.csv").astype({CATEGORICAL_VAR_NAME: object, TREATMENT_NAME: object, TARGET_NAME: object})
    forest = RandomForest(n_trees=5, max_depth=5, control_name="T0")
    forest.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    print(forest.predict(pandas.DataFrame([[0.2, "B", 0.4, 0.2, 0.3], [0.6, "A", 0.8, 0.1, 0.1]], columns=["VAR1", "VAR2", "VAR3", "VAR4", "VAR5"])))


    # dataset = pandas.read_csv("./data/synthetic_10k.txt", sep="\t").astype({"T": object, "Y": object})
    # forest = RandomForest(n_trees=5, max_depth=5)
    # forest.fit(dataset[dataset.columns[:-2]], dataset["T"], dataset["Y"])