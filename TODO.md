Tasks To Do
===========

Decision Tree
-------------

- [ ] Create a new decision tree algorithm based on the BayesianDecisionTree class but using either OptimizedUnivariateEncoding or MultiTreatmentUnivariateEncoding.
  The new class must be named MultiTreatmentDecisionTree.
  The UnivariateEncoding class must be chosen automatically depending on the number of treatments in the dataset:
  - 2 treatments => OptimizedUnivariateEncoding
  - 3 treatments or more => MultiTreatmentUnivariateEncoding
  The decision tree must be binary. As such, any use of MultiTreatmentUnivariateEncoding must limit the number of treatment groups to 2.
  The choice of the variable is random with equal chances for all variables.

- [ ] Allow the choice of the variable to be configured to any of these algorithms:
  - Always choose the variable with the highest level.
  - Choose randomly with equal chances for all variables.
  - Choose randomly with chances proportional to the levels of the variables.