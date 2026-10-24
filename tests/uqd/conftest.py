import os
import sys

# Make the repository root importable (src.uqd, src.cfg) and this directory
# importable (fakes) regardless of where pytest is launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
