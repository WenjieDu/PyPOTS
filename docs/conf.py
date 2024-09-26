# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import os
import sys

from sphinx.ext.napoleon.docstring import NumpyDocstring

try:
    sys.path.insert(0, os.path.abspath(".."))
except IndexError:
    pass

import pypots

# -- Project information -----------------------------------------------------
project = "PyPOTS"
author = "Wenjie Du"
date_now = datetime.datetime.now()
copyright = f"{date_now.year}, {author}"

# The full version, including alpha/beta/rc tags
release = pypots.__version__

# -- General configuration ---------------------------------------------------
# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # enables Sphinx to parse both NumPy and Google style docstrings, otherwise no Param/Returns
    "sphinx_autodoc_typehints",  # enables generating type hints automatically in docstrings
    "sphinxcontrib.bibtex",
    "sphinxcontrib.gtagjs",
]

# configs for sphinx.ext.autodoc
# set the order of the members in the documentation
autodoc_member_order = "bysource"

napoleon_use_param = True  # enables parsing parameters in docstrings

# configs for sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "torch": ("https://pytorch.org/docs/main/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# configs for sphinx.ext.imgmath
imgmath_font_size = 16  # set the font size for math equations as 18, the original is 12
imgmath_image_format = "svg"  # using SVG because svg is much clearer than png
# # imgmath_dvipng_args = ["-gamma", "2", "-D", "20000", "-bg", "#F5F5F5"]  # configurations for png format

# configs for sphinxcontrib-bibtex
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# configs for sphinxcontrib-gtagjs
gtagjs_ids = [
    "G-HT18SK09XE",  # PyPOTS docs site
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for PDF output -------------------------------------------------
latex_engine = "xelatex"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for a list of builtin themes.
html_theme = "furo"

html_context = {
    "last_updated": f"{date_now.year}/{date_now.month}/{date_now.day}",
}
# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_favicon = (
    "https://raw.githubusercontent.com/PyPOTS/pypots.github.io/main/static/figs/pypots_logos/PyPOTS/logo_FFBG.svg"
)

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}


# the code below is for fixing the display of `Attributes` heading,
# refer to https://github.com/ivadomed/ivadomed/issues/315
# adds extensions to the Napoleon NumpyDocstring class
def parse_keys_section(self, section):
    return self._format_fields("Keys", self._consume_fields())


NumpyDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields("Attributes", self._consume_fields())


NumpyDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields("Class Attributes", self._consume_fields())


NumpyDocstring._parse_class_attributes_section = parse_class_attributes_section


def patched_parse(self):
    self._sections["keys"] = self._parse_keys_section
    self._sections["class attributes"] = self._parse_class_attributes_section
    self._unpatched_parse()


NumpyDocstring._unpatched_parse = NumpyDocstring._parse
NumpyDocstring._parse = patched_parse
