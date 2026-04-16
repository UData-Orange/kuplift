Example code
============

This directory contains examples of code to show how you can use this package.

Running the examples
====================

There are two types of examples:
- Python modules
- Jupyter notebooks.

Running the Python modules
--------------------------

> Some of the Python modules may assume your current working directory is the toplevel of this package (where the pyproject.toml file resides),
> so as to be able to access the sample data files without any modification of their paths in the example code.

The following shows how to run a Python example module from a shell.
~~~ bash
python examples/multi_treatment_univariate_encoding.py
~~~


Running the Jupyter notebooks
-----------------------------

Here are two ways to run the Jupyter notebooks:
- Installing Jupyter Lab and using your web browser as a client to connect to the local Jupyter Lab server
- Installing only ipykernel and using another client, such as the one built-in Visual Studio Code
  (in that case, you also need to install the `Jupyter` extension of VS Code)

### Using Jupyter Lab

You can create a virtual environment to keep kuplift and ipykernel/jupyterlab, together with their dependencies, isolated
from the rest of your system. For example, to use Jupyter Lab:
~~~ console
$ python -m venv .venv  # Create the virtual environment
$ source .venv/bin/activate  # Activate the virtual environment
$ pip install kuplift jupyterlab  # Install kuplift and jupyterlab
$ jupyter lab  # Open the just-installed Jupyter web application inside a browser
$ # Look at you web browser, Jupyter Lab should be ready to use!
~~~

> You can also install jupyterlab separately using *pipx*: `pipx install jupyterlab`. That way it will not be in the same virtual environment kuplift lives in. To execute: `jupyter-lab`.

### Using the Jupyter extension of Visual Studio Code

If you want to use the Jupyter extension of Visual Studio Code, installing `ipykernel` instead of `jupyterlab` would
suffice, and VS Code might ask you to do that for you if you clicked on a Jupyter notebook file (extension .ipynb).
Make sure VS Code installs ipykernel into the same environment as the one into which you installed kuplift.
Then, just opening a Jupyter notebook file into VS Code should display the notebook client instead of the actual text
content of the file (this is really just a JSON file).
