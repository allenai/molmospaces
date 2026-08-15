import functools
import os
import re
from pathlib import Path

# Use molmo_spaces' own resolved ASSETS_DIR (its ResourceManager-managed
# cache) rather than g1_molmo's -- confirmed to already contain the same
# robots/g1/g1_dex.xml and scenes/procthor-10k-val/val_0.xml this port needs
# (byte-identical g1_dex.xml, verified by direct diff against g1_molmo's
# copy). Falls back to G1_MOLMO_ASSETS_DIR / a g1_molmo checkout if
# molmo_spaces' own constants module isn't importable for some reason, or if
# a specific asset (e.g. a grasp source) turns out to be missing there.
try:
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
except ImportError:
    ASSETS_DIR: Path = Path(
        os.environ.get("G1_MOLMO_ASSETS_DIR", str(Path.home() / "code/g1_molmo/molmospaces/assets"))
    )

# Grasp files default to assets/grasps but can live anywhere via env var
_grasps_env = os.environ.get("MOLMOSPACES_GRASPS_DIR", "")
GRASPS_DIR: Path = Path(_grasps_env).expanduser() if _grasps_env else ASSETS_DIR / "grasps"


@functools.cache
def grasp_source_dir(source: str) -> Path:
    """grasps/<source>, also accepting the raw molmospaces_resources cache
    layout which nests a version dir (grasps/<source>/<YYYYMMDD>/<ID>/)."""
    d = GRASPS_DIR / source if source else GRASPS_DIR
    if d.is_dir():
        versions = []
        for k in d.iterdir():
            if not (k.is_dir() and re.fullmatch(r"\d{8}", k.name)):
                versions = []
                break
            versions.append(k)
        if versions:
            return max(versions, key=lambda k: k.name)
    return d
