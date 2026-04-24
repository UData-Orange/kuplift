Welcome to kuplift's documentation!
===================================

Description
-----------

It's a **User Parameter-free Bayesian Framework for Uplift Modeling**.

How to use kuplift ?
--------------------

.. toctree::

  Get started <get_started/index>

.. toctree::

  Documentation <documentation/index>
  
.. toctree::

  References <references/index>

A note on the terminology used in the source code
-------------------------------------------------

The source code uses short names such as *i*, *j* or *t*. Here are their definitions:
- *i*: part (interval for a numerical variable or value group for a categorical variable);
- *j*: target (outcome);
- *t*: treatment;
- *g*: group of treatments;
- *N*: number of observations (frequency);
- *P*: probability.

The source code also refers to tables, represented by `pandas.DataFrame` and named using these short names.
For example: *N_ijt*, *P_ijg*, *Uplift_ig*".
Explanation of the example names:
- *N_ijt*: Each value in the table is a number of observations. One DataFrame column contains the values for one part (*i*). One DataFrame row contains the values for one target-treatment pair (*jt*).
- *P_ijg*: Each value in the table is a probability. One DataFrame column contains the values for one part (*i*). One DataFrame row contains the values for one target-treatmentgroup pair (*jg*).
- *Uplift_ig*: Each value in the table is an uplift. One DataFrame column contains the values for one part (*i*). One DataFrame row contains the values for one treatment group (*g*).