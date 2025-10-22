import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

project = 'Tunable Kernel Nulling'
author = 'Auto-generated'
year = datetime.now().year

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'index.rst', 'README.md']

# Use the modern Furo theme (lightweight and pleasant for API docs)
html_theme = 'furo'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Pour une génération complète (CI), installez les dépendances listées dans
# requirements-docs.txt afin d'éviter d'avoir à moquer les imports.
autodoc_mock_imports = ['numba', 'matplotlib', 'scipy', 'git']

# Napoleon settings (use numpy docstring style by default)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# myst-parser options (allow LaTeX math and some extensions)
myst_enable_extensions = [
    'amsmath', 'colon_fence', 'deflist', 'dollarmath', 'fieldlist',
    'html_admonition', 'html_image', 'linkify', 'replacements',
    'smartquotes', 'strikethrough', 'substitution', 'tasklist'
]
