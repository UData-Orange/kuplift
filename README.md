
## Bibliothèque Kuplift


Kuplift is a _Python_ package that provides a series of uplift modeling methods based on recent research work. Kuplift allows users to easily use the following algorithms:

1. Encoding data using a discretization method for treatment effect (uplift) modeling called _UMODL_[^fn1]
    
2.  Variable selection for uplift modeling with _UMODL-FS_.[^fn1]
    
3. Learning a Bayesian decision tree model for uplift modeling with _UB-DT_.[^fn2]
    
4.  Learning a random forest model for uplift modeling with _UB-RF_.[^fn2]

**How to install**:

```python
 DRAFT: pip install Kuplift
```

**User Guide**:

```python
import pandas as pd

df = pd.read_csv("dataname.csv")

# Univariate encoding:
ue=UnivariateEncoding()

encoded_data=ue.fit_transform(df, "treatment", "outcome")

# Feature selection
fs=FeatureSelection()

important_vars=fs.filter(df, "treatment", "outcome")

# Decision trees
Tree=BayesianDecisionTree(df, "treatment", "outcome")

Tree.fit()

preds=Tree.predict(df[column_names])

# Random forests
forest=BayesianRandomForest(df, "treatment", "outcome", Nb_trees)

forest.fit()

preds=forest.predict(df[features])
```



[^fn1]: Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2023, March). A non-parametric bayesian approach for uplift discretization and feature selection. **_ECML PKDD 2022 (rang A)_**

[^fn2]: Rafla, M., Voisine, N., & Crémilleux, B. (2023, May). Parameter-free Bayesian decision trees for uplift modeling. **_PAKDD 2023 (rang A)_**
