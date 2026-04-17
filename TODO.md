Tasks To Do
===========

Multi-treatment univariate encoding
-----------------------------------

- [ ] Clarify how the "1" target, that is, the value in the target column when the treatment "works", should be specified.
  Currently, it is a parameter of the `get_uplift` function but not of the `get_target_probabilities` function!
  The `get_target_probabilities` function works because the target modalities are hardcoded inside of this function (`0` and `1`).
  Because the `get_uplift` function works with the probabilities returned by `get_target_probabilities`, no target modality is hardcoded
  inside of this function, but the parameter is also not used.
  Maybe the parameter should be moved to the `get_target_probabilities` function?
  Or maybe the "1" target should be specified upon `MultiTreatmentUnivariateEncoding` class instantiation?
- [x] Allow grouping by treatments in `get_target_probabilities` and `get_uplift`


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