Tasks To Do
===========

Multi-treatment univariate encoding
-----------------------------------

- [x] Allow grouping by treatments in `get_target_probabilities` and `get_uplift`.
- [x] Fix `transform`.
- [x] `get_target_probabilities` must compute probabilities for all j|t.
- [x] In *helperfunctions.py* -> `preprocess_data`: fix computations (`t0j1 + t0j1` -> `t0j1 + t0j0`).
- [ ] Fix MultiTreatmentUnivariateEncoding output tables (transpose them).
- [ ] Fix behaviour when dataset is too small and all levels are 0.
- [ ] Merge files (replace files + fix imports) after verification:
  - *bayesian_decision_tree_v2.py* into *bayesian_decision_tree.py*;
  - *tree_v2.py* into *tree.py*;
  - *node_v2.py* into *node.py*.
- [ ] See if `fix_valuegroups` is still needed with Khiops v11.


Decision Tree
-------------

- [ ] Create a new decision tree algorithm based on the BayesianDecisionTree class but using either OptimizedUnivariateEncoding or MultiTreatmentUnivariateEncoding.
  The new class must be named `MultiTreatmentDecisionTree`.
  The UnivariateEncoding class must be chosen automatically depending on the number of treatments in the dataset:
  - 2 treatments => OptimizedUnivariateEncoding
  - 3 treatments or more => MultiTreatmentUnivariateEncoding
  The decision tree must be binary. As such, any use of MultiTreatmentUnivariateEncoding must limit the number of treatment groups to 2.
  The choice of the variable is random with equal chances for all variables.

- [ ] Allow the choice of the variable to be configured to any of these algorithms:
  - Always choose the variable with the highest level.
  - Choose randomly with equal chances for all variables.
  - Choose randomly with chances proportional to the levels of the variables.