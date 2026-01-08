"""Optimized univariate encoding example code.

Executing this code will generate a model from the data file and print various
statistics.
"""

import pandas as pd
import kuplift as kp
df = pd.read_csv("data/data_uplift_missing.csv").astype({"VAR2": object, "CIBLE": object})
data = df[df.columns[:-2]]
treatment_col = df["TRAITEMENT"]
target_col = df["CIBLE"]
ue = kp.OptimizedUnivariateEncoding()
ue.fit(data, treatment_col, target_col)
print("\nModel generation complete.")
input("\nPress enter to display the variable levels...")
print(ue.get_levels())
input("\nPress enter to display the target probabilities for variable 'VAR1'...")
print(ue.get_target_probabilities("VAR1"))
input("\nPress enter to display the uplift for variable 'VAR1'...")
print(ue.get_uplift(1, "T0", "VAR1"))
