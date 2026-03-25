# Copyright (c) 2026 Orange. All rights reserved.
# This software is distributed under the MIT License, the text of which is available
# at https://spdx.org/licenses/MIT.html or see the "LICENSE" file for more details.

"""Optimized univariate encoding example code.

Executing this code will generate a model from the data file and print various
statistics.
"""

import pandas
import kuplift
from pprint import pprint  # So that the user does not have to import it.


def main():
    global ue  # Make it available for study after execution of this function.
    ue = kuplift.OptimizedUnivariateEncoding()
    df = pandas.read_csv("data/data_uplift_missing.csv").astype({"VAR2": object, "CIBLE": object})
    ue.fit(df[df.columns[:-2]], df["TRAITEMENT"], df["CIBLE"])
    print()
    print("Model generation complete. OptimizedUnivariateEncoding instance available under the name 'ue'. 'pprint.pprint' is already imported.")
    print()
    print()
    print("Variable levels:")
    print(ue.get_levels())
    print()
    print()
    VARNAME = "VAR1"
    print("Target probabilities for variable '{}':".format(VARNAME))
    print(ue.get_target_probabilities(VARNAME))
    print()
    print()
    print("Uplift for variable '{}':".format(VARNAME))
    print(ue.get_uplift(1, "T0", VARNAME))


if __name__ == "__main__":
    main()
