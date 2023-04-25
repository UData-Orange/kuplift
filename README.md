## Bibliothèque Kuplift


Kuplift is a Python package that provides a series of uplift modeling methods based on recent research work. Kuplift allows users to easily use the following algorithms:

    1- Encoding data using a discretization method for treatment effect (uplift) modeling called UMODL [^1].
    2- Variable selection for treatment effect modeling using UMODL-FS [^1].
    3- Learning a decision tree model for treatment effect modeling with UB-DT approach [^2]
    4- Learning a decision forest model for treatment effect modeling with UB-RF approach [^2]

**User Guide**:

```python
import pandas as pd

df = pd.read_csv("dataname.csv")

# Discrétisation univariée:
ue=UnivariateEncoding()

encoded_data=ue.fit_transform(df, "treatment", "outcome")

# Sélection de variables
fs=FeatureSelection()

important_vars=fs.filter(df, "treatment", "outcome")

# Arbre de décisions
Tree=BayesianDecisionTree(df, "treatment", "outcome")

Tree.fit()

preds=Tree.predict(df[column_names])

# Forêt d'arbres
forest=BayesianRandomForest(df, "treatment", "outcome", Nb_trees)

forest.fit()

preds=forest.predict(df[features])
```



[^1]: Rafla, M., Voisine, N., Crémilleux, B., & Boullé, M. (2023, March). A non-parametric bayesian approach for uplift discretization and feature selection. **_ECML PKDD 2022 (rang A)_**

[^2]: Rafla, M., Voisine, N., & Crémilleux, B. (2023, May). Parameter-free Bayesian decision trees for uplift modeling. **_PAKDD 2023 (rang A)_**

