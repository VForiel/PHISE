import os
import sys
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

project = 'Tunable Kernel Nulling'
author = 'Auto-generated'
year = datetime.now().year

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Pour une génération complète (CI), installez les dépendances listées dans
# requirements-docs.txt afin d'éviter d'avoir à moquer les imports.
autodoc_mock_imports = ['numba', 'matplotlib', 'scipy']
