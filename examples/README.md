Example code
============

This directory contains examples of code to show how you can use this package.

Running the examples
====================

There are two types of examples in this directory:
- **Python modules**
- **Jupyter notebooks**.

Running the Python modules
--------------------------

> Some of the Python modules may assume your current working directory is the toplevel of this package (where the *pyproject.toml* file resides),
> so as to be able to access the sample data files without any modification of their paths in the example code.

The following shows how to run a Python example module from a shell:
~~~ console
$ python examples/multi_treatment_univariate_encoding.py
~~~


Running the Jupyter notebooks
-----------------------------

Here are two ways to run the Jupyter notebooks:
- installing *Jupyter Lab* and using your web browser as a client to connect to the local Jupyter Lab server
- installing only *ipykernel* and using another client, that may be embedded into your text editor.

### Using Jupyter Lab

You can create a virtual environment to keep kuplift and ipykernel/jupyterlab, together with their dependencies, isolated
from the rest of your system. For example, to use Jupyter Lab:
~~~ console
$ # Create the virtual environment:
$ python -m venv .venv
$ # Activate the virtual environment:
$ source .venv/bin/activate
$ # Install kuplift and jupyterlab (jupyterlab is in the `notebook` optional dependency group):
$ pip install kuplift[notebook]
$ # Open the just-installed Jupyter web application inside a browser:
$ jupyter lab
$ # Look at you web browser, Jupyter Lab should be ready to use!
~~~