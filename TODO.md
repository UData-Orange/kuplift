Tasks To Do
===========

Multi-treatment univariate encoding
-----------------------------------

- [x] Allow grouping by treatments in `get_target_probabilities` and `get_uplift`.
- [x] Fix `transform`.
- [ ] Verify/fix behaviour when dataset is too small and all levels are 0.
- [ ] Clarify how the "1" target, that is, the value in the target column when the treatment "works", should be specified.
  Currently, it is a parameter of the `get_uplift` function but not of the `get_target_probabilities` function!
  The `get_target_probabilities` function works because the target modalities are hardcoded inside of this function (`0` and `1`).
  Because the `get_uplift` function works with the probabilities returned by `get_target_probabilities`, no target modality is hardcoded
  inside of this function, but the parameter is also not used.
  Maybe the parameter should be moved to the `get_target_probabilities` function?
  Or maybe the "1" target should be specified upon `MultiTreatmentUnivariateEncoding` class instantiation?
- [ ] See if `fix_valuegroups` is still needed with Khiops v11.
- [ ] In *helperfunctions.py* -> `preprocess_data`: verify computations.
  The code of this function contains the pattern `t0j1 + t0j1` twice.
  Possible consequences:
  - The computed uplift to sort catergories may be wrong.
  - The sort in `ordered_dict` to encode modalities may become incorrect.
  - The transformation of categorical variables may impact negatively the learning process.
  Two fix proposals are in *preprocess_data_fix_proposal.py*.


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