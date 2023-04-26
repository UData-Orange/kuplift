# Bibliothèque Kuplift


Kuplift est un package _Python_ qui fournit une série de méthodes de modélisation d'uplift basées sur les travaux de recherches récentes. Kuplift permet aux utilisateurs d'utiliser facilement les algorithmes suivants :

1.  Encodage des données selon une méthode de discrétisation des variables pour la modélisation de l'effet de traitement (uplift) [1]
    
2.  Sélection des variables pour la modélisation de l'effet de traitement [1]
    
3.  Apprentissage d'un modèle d'arbre de décision pour la modélisation de l'effet de traitement. [2]
    
4.  Apprentissage d'un modèle de forêt d'arbres de décision pour la modélisation de l'effet de traitement. [2]

**Guide d'utilisateur**:

```python
import pandas as pd

df = pd.read_csv("dataname.csv")

# Discrétisation univariée:
ue = UnivariateEncoding()

encoded_data = ue.fit_transform(df, "treatment", "outcome")

# Sélection de variables
fs = FeatureSelection()

important_vars = fs.filter(df, "treatment", "outcome")

# Arbre de décisions
tree = BayesianDecisionTree(df, "treatment", "outcome")

tree.fit()

preds = tree.predict(df[column_names])

# Forêt d'arbres
forest = BayesianRandomForest(df, "treatment", "outcome", nb_trees)

forest.fit()

preds = forest.predict(df[features])
```
