#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Catalyst documentation build configuration file.
#
# This file is execfiled with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from sphinx.highlighting import lexers

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(""))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.join(os.getcwd(), "doc"))

# In Catalyst, modules are in a different directory, thus we have to
# add paths to sys.path here to fix the issue.
sys.path.insert(0, os.path.join(os.getcwd(), "frontend"))

# For obtaining all relevant C++ source files
currdir = Path(__file__).resolve().parent
PROJECT_SOURCE_DIR = currdir.parent

# -------------------------------------------------------------------------
CPP_SOURCE_DIRS = [
    # PROJECT_SOURCE_DIR.joinpath("mlir").joinpath("include"),
    PROJECT_SOURCE_DIR.joinpath("runtime").joinpath("include"),
]
CPP_EXCLUDE_DIRS = []


def obtain_cpp_files():
    script_path = PROJECT_SOURCE_DIR.joinpath("bin/cpp-files.py")

    if not script_path.exists():
        print("The project directory structure is corrupted.")
        sys.exit(1)

    file_list_total = []
    for CPP_SOURCE_DIR in CPP_SOURCE_DIRS:
        exclude_dirs = [CPP_SOURCE_DIR.joinpath(exclude_dir) for exclude_dir in CPP_EXCLUDE_DIRS]

        print(f"CPP_SOURCE_DIR: {CPP_SOURCE_DIR}")

        p = subprocess.run(
            [str(script_path), "--header-only", CPP_SOURCE_DIR, "--exclude-dirs", *exclude_dirs],
            capture_output=True,
        )
        file_list = json.loads(p.stdout)
        file_list_total.extend(
            "../" + str(Path(f).relative_to(PROJECT_SOURCE_DIR)) for f in file_list
        )

    return file_list_total


CPP_FILES = obtain_cpp_files()
print(CPP_FILES)


# -------------------------------------------------------------------------
# Mock out all modules that aren't required for compiling of documentation
class Mock(MagicMock):
    __name__ = "foo"

    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = [
    "mlir_quantum",
    "mlir_quantum.runtime",
    "mlir_quantum.runtime.np_to_memref",
    "mlir_quantum.dialects",
    "mlir_quantum.dialects.arith",
    "mlir_quantum.dialects.tensor",
    "mlir_quantum.dialects.scf",
    "mlir_quantum.dialects.quantum",
    "mlir_quantum.dialects.gradient",
    "mlir_quantum.dialects.catalyst",
    "mlir_quantum.dialects.mitigation",
    "mlir_quantum.compiler_driver",
    "pybind11",
    "cudaq",
]

mock = Mock()
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.3"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "exhale",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxext.opengraph",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_tabs.tabs",
    "m2r2",
]

intersphinx_mapping = {"https://docs.pennylane.ai/en/stable/": None}

# add the custom MLIR Lexer
from .MLIRLexer import MLIRLexer
lexers['mlir'] = MLIRLexer(startinline=True)

# OpenGraph Metadata
ogp_use_first_image = True  # set to False for autocards
ogp_image = "_static/catalyst_illustration.jpg"  # comment for autocards

ogp_social_cards = {
    "image": "_static/catalyst_illustration.jpg",
    "enable": True,
    "site_url": "https://docs.pennylane.ai/projects/catalyst",
    "line_color": "#03b2ff",
}

# The base URL with a proper language and version.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

autosummary_generate = True
autosummary_imported_members = False
automodapi_toctreedirnm = "code/api"
automodsumm_inherited_members = True

# Breathe extension
breathe_projects = {"Catalyst": "./doxyoutput/xml"}
breathe_default_project = "Catalyst"

mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"
)

# nbsphinx settings
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "examples/.ipynb_checkpoints",
    "*-checkpoint.ipynb",
]
nbsphinx_execute = "never"
nbsphinx_epilog = """
.. raw:: html

    <div class="sphx-glr-download-link-note admonition note">
        <div class="sphx-glr-download">
            <p><a href="../{{env.docname}}.ipynb" class="reference download">Download Python</a></p>
        </div>

        <div class="sphx-glr-download">
            <p><a href="../{{env.docname}}.ipynb" class="reference download">Download notebook</a></p>
        </div>
    </div>
    <div id="tutorial-type">doc/{{ env.doc2path(env.docname, base=None) }}</div>
"""
nbsphinx_requirejs_path = ""

# Exhale extension
# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Overview",
    "doxygenStripFromPath": "..",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": ("INPUT = " + " ".join(CPP_FILES) + "\nEXCLUDE_SYMBOLS = std::* "),
}

# Add any paths that contain templates here, relative to this directory.

templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Catalyst"
copyright = "2023, Xanadu Quantum Technologies"
author = "Xanadu Inc."

add_module_names = False

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

import catalyst  # pylint: disable=wrong-import-position

release = catalyst.__version__

# The short X.Y version.
version = re.match(r"^(\d+\.\d+)", release).expand(r"\1")

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- html theme ---------------------------------------------------------
html_theme = "pennylane"

# html theme options (see theme.conf for more information)
html_theme_options = {
    "navbar_name": "Catalyst",
    "navbar_active_link": 3,
    "google_analytics_tracking_id": "G-C480Z9JL0D",
    "extra_copyrights": [
        "TensorFlow, the TensorFlow logo, and any related marks are trademarks " "of Google Inc."
    ],
    "toc_overview": True,
    "github_repo": "PennyLaneAI/catalyst",
}

edit_on_github_project = "PennyLaneAI/catalyst"
edit_on_github_branch = "main/doc"

# ============================================================

# the order in which autodoc lists the documented members
autodoc_member_order = "bysource"

# inheritance_diagram graphviz attributes
inheritance_node_attrs = {"color": "lightskyblue1", "fillcolor": "lightskyblue1", "style": "filled"}

# autodoc_default_flags = ['members']
autosummary_generate = True
