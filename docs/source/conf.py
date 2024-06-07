# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Financial Data Science'
copyright = '2022-2024, Terence Lim'
author = 'Terence Lim'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# workaround for sphinx docstring duplicate object description
napoleon_google_docstring = True  # Enable parsing of Google-style pydocs.
napoleon_use_ivar = True  # to correctly handle Attributes header in class pydocs

templates_path = ['_templates']
exclude_patterns = []

# Sort members by type
autodoc_member_order = 'groupwise'
autodoc_default_options = {
    'special-members': '__getitem__, __call__', #'__init__, __call__',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, '/home/terence/Dropbox/github/investment-data-science/finds')
