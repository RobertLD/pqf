# Configuration file for the Sphinx documentation builder.
#
import sys
import os
import pkgutil

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath("../.."))
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
    "sphinx.ext.napoleon",
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
autodoc_typehints = "both"

# generate_rst.py


def generate_autosummary_rst(package_name):
    submodules = list_submodules(package_name)
    with open("index.rst", "w") as f:
        f.write(f"{package_name} Documentation\n")
        f.write("=================\n\n")
        f.write(".. rubric:: Modules\n\n")
        f.write(".. autosummary::\n")
        f.write("    :recursive:\n")
        f.write("    :toctree: generated\n\n")
        for module in submodules:
            f.write(f"    {module}\n")


def list_submodules(package_name):
    package = __import__(package_name, fromlist=[""])
    return [
        name
        for _, name, _ in pkgutil.iter_modules(package.__path__, package_name + ".")
    ]


generate_autosummary_rst("pqf")
