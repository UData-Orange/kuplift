"""Optimized univariate encoding example code.

Executing this code will generate a model from the data file.
From this point you can call other methods of the 'OptimizedUnivariateEncoding'
class, such as 'get_target_probabilities' and 'get_uplift'.
See the comments at the end of the code for examples of what you can do.
"""

import pandas as pd
import kuplift as kp
df = pd.read_csv("data/data_uplift_missing.csv").astype({"VAR2": object, "CIBLE": object})
data = df[df.columns[:-2]]
treatment_col = df["TRAITEMENT"]
target_col = df["CIBLE"]
ue = kp.OptimizedUnivariateEncoding()
ue.fit(data, treatment_col, target_col)
# >>> ue.get_target_probability("VAR1")
# >>> ue.get_uplift(1, "T0", "VAR1")
