from kuplift import *
import pandas as pd
import logging


LOGLEVEL = logging.INFO
# LOGLEVEL = logging.DEBUG
DATASET_PATH = "./data/dataset.csv"
TREATMENT_NAME = "TRAITEMENT"
TARGET_NAME = "CIBLE"
TARGET1 = 1
REF_TREATMENT = "T0"
OUTPUT_DIR = None
# OUTPUT_DIR = "./OUTPUT"


if __name__ == "__main__":
    logging.basicConfig(level=LOGLEVEL)
    df = pd.read_csv(DATASET_PATH)
    ue = MultiTreatmentUnivariateEncoding()
    ue.fit(df[df.columns[:-2]], df[TREATMENT_NAME], df[TARGET_NAME], outputdir=OUTPUT_DIR)

    print("Input variables:", ue.input_variables)
    print("Informative input variables:", ue.informative_input_variables)
    print("Non-informative input variables:", ue.noninformative_input_variables)
    print("Treatment name:", ue.treatment_name)
    print("Target name:", ue.target_name)
    print("Treatments:", ue.treatment_modalities)
    print("Targets:", ue.target_modalities)
    print("Target-treatment pairs:", ue.target_treatment_pairs)
    print("Input variable levels:", ue.get_levels())
    for var, groups_by_parts in ue.get_treatment_groups().items():
        for part, groups in groups_by_parts.items():
            print("Treatment groups for variable {} and part {}: [{}]".format(var, part, ", ".join(str(group) for group in groups)))
    
    for var in ue.informative_input_variables:
        print("\n[Details of variable {!r}]".format(var))
        print("Level:", ue.get_level(var))
        print("Partition:", ", ".join(map(str, ue.get_partition(var))))
        for part, groups in ue.get_treatment_groups(var).items():
            print("Treatment groups for part {}: [{}]".format(part, ", ".join(str(group) for group in groups)))
        print("Target frequencies:")
        print(ue.get_target_frequencies(var))
        print("Target probabilities:")
        print(ue.get_target_probabilities(var))
        print("Uplift:")
        print(ue.get_uplift(TARGET1, REF_TREATMENT, var))
        print("Target probabilities with groups:")
        print(ue.get_target_probabilities_of_treatment_groups(var))
        print("Uplift with groups.")
        print(ue.get_uplift_of_treatment_groups(TARGET1, REF_TREATMENT, var))