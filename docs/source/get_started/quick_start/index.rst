Quick start
-----------

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
