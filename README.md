
## kuplift package

<p  align="center">
  <img src="https://raw.githubusercontent.com/UData-Orange/kuplift/main/docs/source/logo.png" width="310" />
</p>


kuplift is a _Python_ package that provides a series of uplift modeling methods based on recent research work. kuplift allows users to easily use the following algorithms:

1. Encoding data using a discretization method for treatment effect (uplift) modeling called _UMODL_.
    
2.  Variable selection for uplift modeling with _UMODL-FS_.
    
3. Learning a Bayesian decision tree model for uplift modeling with _UB-DT_.
    
4.  Learning a random forest model for uplift modeling with _UB-RF_.

**How to install**:

```python
pip install kuplift
```

**User Guide**:

```python
import kuplift as kp
import pandas as pd

df = pd.read_csv("dataname.csv")

# Univariate variable transformation:
ue = kp.UnivariateEncoding()
encoded_data = ue.fit_transform(df, "treatment", "outcome")

# Feature selection
fs = kp.FeatureSelection()
important_vars = fs.filter(df, "treatment", "outcome")

# Uplift Bayesian Decision Tree
tree = kp.BayesianDecisionTree(df, "treatment", "outcome")
tree.fit()
preds = tree.predict(df[column_names])

# Uplift Bayesian Random Forest
forest = kp.BayesianRandomForest(df, "treatment", "outcome", nb_trees)
forest.fit()
preds = forest.predict(df[features])
```

**Documentation**:

Refer to the documentation at https://udata-orange.github.io/kuplift/

**Credits**:
kuplift has been developed at Orange Labs.

Current contributors:

Mina Rafla
Nicolas Voisine


**References**:

Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2023, March). A non-parametric bayesian approach for uplift discretization and feature selection. **_ECML PKDD 2022_**

Rafla, M., Voisine, N., & Crémilleux, B. (2023, May). Parameter-free Bayesian decision trees for uplift modeling. **_PAKDD 2023_**
