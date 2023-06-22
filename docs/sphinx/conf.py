# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

from sphinx.ext import autodoc

project = "PyroclastMPM"
copyright = "2023, Retief Lubbe"
author = "Retief Lubbe"


sys.path.insert(0, os.path.abspath("../../python/pybind/"))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx.ext.graphviz",  # enable graphs for breathe
]

templates_path = ["_templates"]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"


# auto summary

autosummary_generate = True
autoclass_content = "both"
html_show_sourcelink = False
autodoc_inherit_docstrings = True
set_type_checking_flag = True
autodoc_default_flags = ["members"]


templates_path = ["_templates"]

html_context = {"default_mode": "dark"}

## breathe
breathe_default_members = ("members", "undoc-members")
breathe_implementation_filename_extensions = [".cpp"]
breathe_order_parameters_first = True

breathe_projects = {"pyroclastmpm": "../doxygen/xml/"}
breathe_default_project = "pyroclastmpm"


# coverage
coverage_show_missing_items = True


# napoleon
numpydoc_show_class_members = False
numpydoc_xref_param_type = True


add_module_names = False


# https://stackoverflow.com/questions/46279030/how-can-i-prevent-sphinx-from-listing-object-as-a-base-class


class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`pybind11_object`":
            return
        super().add_line(line, source, *lineno)


autodoc.ClassDocumenter = MockedClassDocumenter
