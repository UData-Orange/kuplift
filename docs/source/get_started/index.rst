Get started
===========

Installation
------------
Go to the `PyPI site <https://pypi.org/project/kuplift/>`_ to download and 
install the package.

Or use the command : ::
    
    pip install kuplift

Quick start
-----------

Code example : ::

    import kuplift as kp
    import pandas as pd

    df = pd.read_csv("data.csv")

    # Make sure the dtype of all categorical variables is object
    df = df.astype({"some_categorical_variable": object})

    variables = df[:-2]  # Last two columns are treatment and target columns

    # Univariate variable transformation
    ue = kp.UnivariateEncoding()
    encoded_data = ue.fit_transform(df[variables], df["treatment"], df["target"])

    # Univariate variable transformation optimized through the use of umodl
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

License
-------
See the ``LICENSE.md`` file of this repository or in the -info directory of the 
python package installation (you can find it with ``pip show -f kuplift``)

Credits
-------
kuplift has been developed at `Orange Labs <https://hellofuture.orange.com/en/artificial-intelligence/>`_.

**Current contributors:**

Mina Rafla

Nicolas Voisine
