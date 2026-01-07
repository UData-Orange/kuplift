"""Optimized univariate encoding example code.

This code assumes your current working directory is the toplevel of this package,
so as to be able to access the sample data file without the need to adapt its path.

Executing this code will generate a model from the data file.

You can use this example code by following one of these ways:
    - Calling the module using the python executable in "inspection" mode:
      `python -i .\examples\optimized_univariate_encoding.py`
    - Importing everything from this module in a running interpreter like this:
      `from examples.optimized_univariate_encoding import *`
    - Copying/pasting the code into a Python interpreter.

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
