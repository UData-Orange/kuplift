# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kuplift"
copyright = "2025, Orange"
author = "Orange"
release = "0.0.8"

# Be strict about any broken references
nitpicky = True

# To avoid using qualifiers like :class: to reference objects within the same context
default_role = "obj"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
]

## Numpydoc extension config
numpydoc_show_class_members = False

## Autodoc extension config
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,
}

## Intersphinx extension config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

## Autosummary extension config
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and directories to
# ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ["_templates", "_build", "Thumbs.db", ".DS_Store"]

# HTML Theme
# Theme colors and fonts come from https://brand.orange.com
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#FFF0E2",
        "font-stack": "Helvetica Neue, sans-serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#CC6100",
        "font-stack": "Helvetica Neue, sans-serif",
    },
}
html_title = f"<h6><center>{project} {release}</center></h6>"
html_logo = "./logo.png"

# HTML static pages
html_static_path = []
html_css_files = [
    "css/custom.css",
]
