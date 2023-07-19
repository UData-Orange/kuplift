Get started
-----------

Installation
============
Go to the `Pypi site <https://pypi.org/project/kuplift/>`_ to download and 
install the package.

Or use the command : ::
    
    pip install kuplift


Requirements
============
`Numpy <https://pypi.org/project/numpy/>`_ >= 1.18.5

`Pandas <https://pypi.org/project/pandas/>`_ >= 0.24.1

`Sortedcontainers <https://pypi.org/project/sortedcontainers/>`_ >= 2.4.0

Quick start
============

Code sample : ::

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

Licence
=======
License
See the ``LICENSE.md`` file of this repository or in the -info directory of the 
python package installation (you can find it with ``pip show -f kuplift``)

Credits
=======
kuplift has been developed at `Orange Labs <https://hellofuture.orange.com/en/artificial-intelligence/>`_.

**Current contributors:**


Mina Rafla

Nicolas Voisine
