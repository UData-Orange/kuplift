import logging
from kuplift import DecisionTree
import pandas

logging.getLogger("kuplift.mt_univariate_encoding").setLevel(logging.WARNING)

LOGLEVEL = logging.INFO
# LOGLEVEL = logging.DEBUG

TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"
CATEGORICAL_VAR_NAME = "VAR2"

# You can print `tree.tree_to_mermaid()` to stdout and copy-paste the output to https://mermaid.live/ to see the tree as a diagram.
# `tree.print_tree(show_path=True)` also works but it prints the tree to the console and the result is less readable.

if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)

    # # 2 treatments
    # dataset = pandas.read_csv("./data/data_uplift_missing.csv").astype({CATEGORICAL_VAR_NAME: object, TREATMENT_NAME: object, TARGET_NAME: object})
    # tree = DecisionTree()
    # tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    # # print(tree.tree_to_mermaid())
    # tree.print_tree(show_path=True)
    # print(tree.predict_best_treatment(dataset[dataset.columns[:-2]]))

    # print()
    # print()

    # # 4 treatments
    # dataset = pandas.read_csv("./data/dataset.csv").astype({CATEGORICAL_VAR_NAME: object, TREATMENT_NAME: object, TARGET_NAME: object})
    # tree = DecisionTree()
    # tree.fit(dataset[dataset.columns[:-2]], dataset[TREATMENT_NAME], dataset[TARGET_NAME])
    # # print(tree.tree_to_mermaid())
    # tree.print_tree(show_path=True)
    # print(tree.predict_best_treatment(dataset[dataset.columns[:-2]]))



    dataset = pandas.read_csv("./data/synthetic_10k.txt", sep="\t").astype({"T": object, "Y": object})
    tree = DecisionTree()
    tree.fit(dataset[dataset.columns[:-2]], dataset["T"], dataset["Y"])
    # print(tree.tree_to_mermaid())
    tree.print_tree(show_path=True)