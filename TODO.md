Tasks To Do
===========

In `kuplift.multi_treatment_univariate_encoding.MultiTreatmentUnivariateEncoding`
---------------------------------------------------------------------------------

- [x] Fix categorical variables considered numerical when they are of numerical types.
- [x] Fix filter variable rule with categorical variable.


Elsewhere
---------

### Decision Tree

- [ ] Refactor and modify decision tree algorithms to use either OptimizedUnivariateEncoding or MultiTreatmentUnivariateEncoding.
  The UnivariateEncoding class must be chosen automatically depending on the number of treatments in the dataset:
  - 2 treatments => OptimizedUnivariateEncoding
  - 3 treatments or more => MultiTreatmentUnivariateEncoding
  The decision tree must be binary. As such, any use of MultiTreatmentUnivariateEncoding must limit the number of treatment groups to 2.
  The choice of the variable is random with equal chances for all variables.

- [ ] Allow the choice of the variable to be configured to any of these algorithms:
  - Always choose the variable with the highest level.
  - Choose randomly with equal chances for all variables.
  - Choose randomly with chances proportional to the levels of the variables.