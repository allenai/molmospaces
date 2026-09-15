"""Put this checkout's repo root first on sys.path. Import before anything else.

Two things need it. `fetchman` is not an installed package (pyproject ships
only `molmo_spaces*`), so it resolves only via the repo root. And `pip install
-e` may well point `molmo_spaces` at a *different* clone -- python puts the
script's directory on sys.path, not the working directory, so site-packages
otherwise wins over the checkout these scripts live in.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[:1] != [_REPO_ROOT]:
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)
