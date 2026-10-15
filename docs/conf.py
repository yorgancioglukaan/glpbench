"""Sphinx configuration for the GLPBench documentation."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glpbench import __version__  # noqa: E402

project = "GLPBench"
author = "Kaan Yorgancioglu"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
myst_enable_extensions = ["colon_fence"]

exclude_patterns = ["_build", "migration_baseline.md"]
templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
