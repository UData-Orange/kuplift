"""Optimized univariate encoding example code.

Executing this code will generate a model from the data file and print various
statistics.
"""

import pandas
import kuplift

def main():
    global ue
    ue = kuplift.OptimizedUnivariateEncoding()
    df = pandas.read_csv("data/data_uplift_missing.csv").astype({"VAR2": object, "CIBLE": object})
    ue.fit(df[df.columns[:-2]], df["TRAITEMENT"], df["CIBLE"])
    print("\nModel generation complete. OptimizedUnivariateEncoding instance available under the name 'ue'.")
    print("At any of the following steps, press Enter to continue or type 'stop' then Enter to stop at a given step (useful in inspection mode).")
    if input("\nPress Enter to display the variable levels...") == "stop": return
    print(ue.get_levels())
    if input("\nPress Enter to display the target probabilities for variable 'VAR1'...") == "stop": return
    print(ue.get_target_probabilities("VAR1"))
    if input("\nPress Enter to display the uplift for variable 'VAR1'...") == "stop": return
    print(ue.get_uplift(1, "T0", "VAR1"))


if __name__ == "__main__":
    main()
