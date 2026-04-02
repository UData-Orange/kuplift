# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Multi-treatment univariate encoding example code.

Executing this code will generate a model from the data file and print various
statistics.
"""

import pandas
import kuplift
import logging
from pprint import pprint  # So that the user does not have to import it.

logger = logging.getLogger(__name__)


def yesno(question):
    return input("%s  y/Y/yes/YES to accept, anything else to decline.\n> " % question) in ["y", "Y", "yes", "YES"]


def main():
    global ue  # Make it available for study after execution of this function.
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    ue = kuplift.MultiTreatmentUnivariateEncoding()
    # df = pandas.read_csv("/home/user1/testfiles/kuplift/random_dataset.csv")
    df = pandas.read_csv("/home/user1/testfiles/kuplift/dataset_multivar_with_random.csv")
    ue.fit(df[df.columns[:-2]], df["TREATMENT"], df["TARGET"], outputdir="/home/user1/test")
    print()
    print("Model generation complete. MultiTreatmentUnivariateEncoding instance available under the name 'ue'. 'pprint.pprint' is already imported.")
    print()
    print("Variable levels:")
    print(ue.get_levels())
    print()
    print("Input variables:")
    print(ue.input_variables)
    print()
    print("Informative input variables:")
    print(ue.informative_input_variables)
    print()
    print("Non-informative input variables:")
    print(ue.noninformative_input_variables)
    print()
    print("Treatment column name:")
    print(ue.treatment_name)
    print()
    print("Target column name:")
    print(ue.target_name)
    print()
    print("Treatment modalities:")
    print(*ue.treatment_modalities, sep=", ")
    print()
    print("Target modalities:")
    print(*ue.target_modalities, sep=", ")
    print()
    print("Target-treatment pairs:")
    print(*ue.target_treatment_pairs, sep=", ")
    VARNAME = "VARIABLE1"
    if VARNAME in ue.informative_input_variables:
        print()
        print("Treatment groups for variable '{}':".format(VARNAME))
        for part, groups in ue.treatment_groups[VARNAME].items():
            print("  - Part:", part)
            print("    Groups:")
            for group in groups:
                print("      -", group)
        print()
        if yesno("Compute target probabilities? This may take a long time."):
            print("Target probabilities for variable '{}':".format(VARNAME))
            print(ue.get_target_probabilities(VARNAME))
        print()
        if yesno("Compute uplift? This may take a long time."):
            print("Uplift for variable '{}':".format(VARNAME))
            print(ue.get_uplift(1, 0, VARNAME))


if __name__ == "__main__":
    main()