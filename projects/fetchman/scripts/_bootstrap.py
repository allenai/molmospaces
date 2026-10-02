"""Put the repo root and `projects/` first on sys.path. Import before anything else.

Three things need it. `fetchman` is not an installed package (pyproject ships
only `molmo_spaces*`), so it resolves only via `projects/` being on sys.path
(it lives at `projects/fetchman`, and imports it as the top-level `fetchman`).
The repo root is needed too, for `molmo_spaces` imports. And `pip install -e`
may well point `molmo_spaces` at a *different* clone -- python puts the
script's directory on sys.path, not the working directory, so site-packages
otherwise wins over the checkout these scripts live in.
"""

import sys
from pathlib import Path

_PROJECTS_DIR = str(Path(__file__).resolve().parents[2])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
# Inserted in reverse so `projects/` ends up first: a stray top-level
# `fetchman/` at the repo root must not shadow `projects/fetchman`.
for _path in (_REPO_ROOT, _PROJECTS_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
