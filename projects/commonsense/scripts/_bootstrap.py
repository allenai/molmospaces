"""Put the repo root and `projects/` first on sys.path. Import before anything else.

Same as projects/fetchman/scripts/_bootstrap.py. `commonsense` is not an installed
package (pyproject ships only `molmo_spaces*`), so it resolves only via `projects/`
being on sys.path. The repo root goes first too, so an editable `molmo_spaces`
install pointing at a different clone does not shadow this checkout.
"""

import sys
from pathlib import Path

_PROJECTS_DIR = str(Path(__file__).resolve().parents[2])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
# Inserted in reverse so `projects/` ends up first.
for _path in (_REPO_ROOT, _PROJECTS_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
