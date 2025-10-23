from .classes import *
from .modules import *

import os

# Get version from pyproject.toml
try:
    import toml
    pyproject_file = os.path.join(os.path.dirname(__file__), '../..', 'pyproject.toml')
    with open(pyproject_file, 'r') as f:
        pyproject = toml.load(f)
    __version__ = pyproject['project']['version']
except Exception as e:
    print("❌ Error: Could not retrieve version information.")
    print(f"ℹ️ {e}")

# Try to get current commit (if in a git repo)
try:
    import subprocess
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    __version__ += f"+{commit[:7]}"
except Exception:
    pass