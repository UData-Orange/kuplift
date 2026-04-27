from kuplift import *
import pandas as pd
import logging


LOGLEVEL = logging.INFO
# LOGLEVEL = logging.DEBUG
DATASET_PATH = "./data/data_uplift_missing.csv"
TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"
TARGET1 = 1
REF_TREATMENT = "T0"


if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)
    df = pd.read_csv(DATASET_PATH)
    ue = OptimizedUnivariateEncoding()
    ue.fit(df[df.columns[:-2]], df[TREATMENT_NAME], df[TARGET_NAME])

    print("Input variables:", ue.input_variables)
    print("Informative input variables:", ue.informative_input_variables)
    print("Non-informative input variables:", ue.noninformative_input_variables)
    print("Treatment name:", ue.treatment_name)
    print("Target name:", ue.target_name)
    print("Treatments:", ue.treatment_modalities)
    print("Targets:", ue.target_modalities)
    print("Target-treatment pairs:", ue.target_treatment_pairs)
    print("Input variable levels:", ue.get_levels())
    
    for var in ue.informative_input_variables:
        print("\n[Details of variable {!r}]".format(var))
        print("Level:", ue.get_level(var))
        print("Partition:", ", ".join(map(str, ue.get_partition(var))))
        print("Target frequencies:")
        print(ue.get_target_frequencies(var))
        print("Target probabilities:")
        print(ue.get_target_probabilities(var))
        print("Uplift:")
        print(ue.get_uplift(TARGET1, REF_TREATMENT, var))