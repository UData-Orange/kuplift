# Changelog

All notable changes to this project will be documented in this file.

## [0.0.15] - 2026-01-14

- Set minimal version of *umodl* to *0.0.28*.
- Additions in class kuplift.optimized_univariate_encoding.OptimizedUnivariateEncoding:
    - New method `get_level` to get level of a single variable
    - New properties `informative_input_variables` and `noninformative_input_variables`
- Change in documentation generation: fetch package version from pyproject.toml to avoid duplicate version specification

## [0.0.14] - 2026-01-13

- Update copyright year and fix license reference in source files, for consistency with the LICENSE file

## [0.0.13] - 2026-01-09

- Fix OptimizedUnivariateEncoding.get_target_probabilities for categorical variables
- Add uplift graph in notebook

## [0.0.12] - 2026-01-09

- Fix computations in OptimizedUnivariateEncoding:
    - get_target_probabilities
    - get_uplift

## [0.0.11] - 2026-01-08

- Rename modules to prevent their names from being shadowed by the class names.
- Add OptimizedUnivariateEncoding example usage as a Jupyter notebook.
- Constraint Khiops version < 11 to avoid a warning. This may change in the future but will require more work.
- Split Python code example into multiple steps to allow interrupting it in inspection mode (python -i examples/...).
- Do not throw an exception if there is a MISSING value in an informative numerical variable column and there is no dedicated 'MISSING interval'. Instead, return 0 as if the MISSING value was part of the first interval.
- Improve OptimizedUnivariateEncoding class documentation.

## [0.0.10] - 2026-01-08

- Add TargetTreatmentPair class
- Document OptimizedUnivariateEncoding's attributes
- Add methods in OptimizedUnivariateEncoding:
    - get_levels
    - get_target_frequencies
    - get_uplift

## [0.0.9] - 2025-09-18

- Re-enable support for Python 3.9

## [0.0.8] - 2025-09-18

- Fix calling kuplift.OptimizedUnivariateEncoding's fit and fit_transform methods when the maxpartnumber argument is not specified.
- Fix code example in both the README.md and the documentation
- Fix reST syntax errors in the documentation

## [0.0.7] - 2025-09-18

- Add a new maxpartnumber parameter to the OptimizedUnivariateEncoding fit and fit_transform functions to allow customization of the maximal number of intervals/groups

## [0.0.6] - 2025-09-18

- Add optimized univariate encoding using umodl which is written in C++
- Modernize CI workflows to avoid direct use of setup.py and to publish Python packages using OIDC (Trusted Publishers)
- Update the list of supported Python versions

## [0.0.5] - 2023-09-15

- Interface modifications similar to causalml

## [0.0.4] - 2023-07-20

- minor correction
- README.md

## [0.0.3] - 2023-07-18

- add parallelization
- correct warnings
- detail documentation
