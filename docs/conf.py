# Configuration file for the Sphinx documentation builder.
#
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

project = "PQF"
copyright = "2024, RobertLD"
author = "RobertLD"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_use_modindex = True
html_static_path = ["_static"]
html_theme_options = {
    "prev_next_buttons_location": "bottom",
    "display_version": True,
    "logo_only": False,
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
}


autosummary_generate = True

autodoc_default_options = {
    "inherited-members": None,
}
autodoc_typehints = "none"
