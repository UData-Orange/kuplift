## kuplift package

<p  align="center">
  <img src="https://raw.githubusercontent.com/UData-Orange/kuplift/main/docs/source/logo.png" width="310" />
</p>


kuplift is a _Python_ package that provides a series of uplift modeling methods based on recent research work. kuplift allows users to easily use the following algorithms:

1. Encoding data using a discretization method for treatment effect (uplift) modeling called _UMODL_.

2. OptimizedUnivariateEncoding is an optimized version of the umodl algorithm written in C++ for continuous and categorical variables.
    
3. Variable selection for uplift modeling with _UMODL-FS_.
    
4. Learning a Bayesian decision tree model for uplift modeling with _UB-DT_.
    
5. Learning a random forest model for uplift modeling with _UB-RF_.


### How to install

~~~ console
$ pip install kuplift
~~~


### User Guide

~~~ python
import kuplift as kp
import pandas as pd

df = pd.read_csv("data.csv")

# Make sure the dtype of all categorical variables is object
df = df.astype({"some_categorical_variable": object})

variables = list(df.columns[:-2])  # Last two columns are treatment and target columns

# Univariate variable transformation
ue = kp.UnivariateEncoding()
encoded_data = ue.fit_transform(df[variables], df["treatment"], df["target"])

# Univariate variable transformation optimized through the use of the C++ implementation of umodl
oue = kp.OptimizedUnivariateEncoding()
encoded_data = oue.fit_transform(df[variables], df["treatment"], df["target"])

# Feature selection
fs = kp.FeatureSelection()
important_vars = fs.filter(df[variables], df["treatment"], df["target"])

# Uplift Bayesian Decision Tree
tree = kp.BayesianDecisionTree()
tree.fit(df[variables], df["treatment"], df["target"])
preds = tree.predict(df[variables])

# Uplift Bayesian Random Forest
forest = kp.BayesianRandomForest(n_trees=4)
forest.fit(df[variables], df["treatment"], df["target"])
preds = forest.predict(df[variables])
~~~


### Examples

You can find examples in the [examples](./examples) directory.


### Documentation

Refer to the documentation at https://udata-orange.github.io/kuplift/


### Credits
kuplift has been developed at Orange Labs.


### Current contributors

Mina Rafla

Nicolas Voisine


### References

Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2022, September). A non-parametric bayesian approach for uplift discretization and feature selection. **_ECML PKDD 2022_**

Rafla, M., Voisine, N., & Crémilleux, B. (2023, May). Parameter-free Bayesian decision trees for uplift modeling. **_PAKDD 2023_**
