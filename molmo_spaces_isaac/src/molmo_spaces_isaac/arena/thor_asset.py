"""Create Isaac Lab Arena Object instances from MolmoSpaces THOR USD assets.

Set MOLMO_ARENA_THOR_PREFER_FLAT_USD=1 to prefer {id}/{id}.usda over {id}_mesh/{id}_mesh.usda
(useful when instanced mesh layouts cause rigid-body / render disagreement).
"""

from __future__ import annotations

import os
from pathlib import Path

from molmo_spaces_isaac import MOLMO_SPACES_ISAAC_BASE_DIR
from molmo_spaces_isaac.utils.common import PACKAGE_ROOT, load_thor_assets_metadata

# Default version subdir when using MOLMO_ISAAC_ASSETS_ROOT (e.g. molmospaces_isaac layout)
THOR_DEFAULT_VERSION = "20260128"


def get_thor_assets_root() -> Path:
    """THOR USD root: MOLMO_THOR_USD_DIR, or MOLMO_ISAAC_ASSETS_ROOT/objects/thor/..., or PACKAGE_ROOT/assets/usd/objects/thor."""
    direct = os.environ.get("MOLMO_THOR_USD_DIR")
    if direct:
        return Path(direct).expanduser().resolve()
    root = os.environ.get("MOLMO_ISAAC_ASSETS_ROOT")
    if root:
        base = Path(root).expanduser().resolve()
        versioned = base / "objects" / "thor" / "thor" / THOR_DEFAULT_VERSION
        if versioned.is_dir():
            return versioned
        # Fallback: objects/thor without version
        flat = base / "objects" / "thor"
        if flat.is_dir():
            return flat
    return PACKAGE_ROOT / "assets" / "usd" / "objects" / "thor"


# Arena imports (optional at module load so rest of package works without Arena)
try:
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from molmo_spaces_isaac.arena.arena_collision_objects import get_arena_object_class

    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False
    ObjectType = None
    Pose = None
    get_arena_object_class = None  # type: ignore[misc, assignment]


def _thor_usd_candidates(asset_id: str, base: Path) -> list[Path]:
    """Ordered list of possible USD paths under base.

    Default: ``{id}_mesh/{id}_mesh.usda`` then ``{id}/{id}.usda`` (ms-download layout).
    Set ``MOLMO_ARENA_THOR_PREFER_FLAT_USD=1`` to try the non-mesh layout first (often fewer
    instance/prototype issues with Isaac Lab rigid-body + rendering).
    """
    mesh = base / f"{asset_id}_mesh" / f"{asset_id}_mesh.usda"
    flat = base / asset_id / f"{asset_id}.usda"
    prefer_flat = (os.environ.get("MOLMO_ARENA_THOR_PREFER_FLAT_USD") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return [flat, mesh] if prefer_flat else [mesh, flat]


def get_thor_usd_path(asset_id: str, assets_dir: Path | None = None) -> Path:
    """Return the USD path for a THOR asset. Tries several roots (ms-download layouts)."""
    if assets_dir is None:
        assets_dir = get_thor_assets_root()

    search_roots: list[Path] = [assets_dir]
    # Flat `objects/thor` given but objects live under `objects/thor/thor/20260128/` (common ms-download layout)
    nested = assets_dir / "thor" / THOR_DEFAULT_VERSION
    if nested.is_dir() and nested not in search_roots:
        search_roots.append(nested)
    # If assets_dir is already versioned, also try parent flat `objects/thor` (symlink / mixed installs)
    if assets_dir.name == THOR_DEFAULT_VERSION and assets_dir.parent.name == "thor":
        flat_sibling = assets_dir.parent.parent
        if flat_sibling.is_dir() and flat_sibling not in search_roots:
            search_roots.append(flat_sibling)

    canonical = _thor_usd_candidates(asset_id, assets_dir)[0]
    for root in search_roots:
        for candidate in _thor_usd_candidates(asset_id, root):
            if candidate.is_file():
                return candidate
    return canonical


def create_thor_object_for_arena(
    asset_id: str,
    instance_name: str | None = None,
    initial_pose: Pose | None = None,
    assets_dir: Path | None = None,
    metadata_path: Path | None = None,
    usd_path_override: Path | str | None = None,
):
    """Create an Arena Object for a THOR asset (RIGID or ARTICULATION) for Scene(assets=[...]).

    Matches :func:`molmo_spaces_isaac.arena.objaverse_asset.create_objaverse_object_for_arena`:
    plain Arena ``Object`` with default spawn cfg (no extra PhysX schema overrides on the root).
    """
    if not _ARENA_AVAILABLE or get_arena_object_class is None:
        raise ImportError(
            "isaaclab_arena is required for Arena. "
            "Install from source (see Isaac Lab Arena documentation) and ensure it is on PYTHONPATH."
        )
    if usd_path_override is not None:
        usd_path = Path(usd_path_override)
    else:
        if assets_dir is None:
            assets_dir = get_thor_assets_root()
        usd_path = get_thor_usd_path(asset_id, assets_dir)
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"THOR asset USD not found: {usd_path}. "
            "Set MOLMO_ISAAC_ASSETS_ROOT or MOLMO_THOR_USD_DIR, or run: ms-download --type usd --install-dir assets/usd --assets thor"
        )
    if metadata_path is None:
        thor_root = assets_dir if assets_dir is not None else get_thor_assets_root()
        in_dir = thor_root / "usd_assets_metadata.json"
        if in_dir.is_file():
            metadata_path = in_dir
        elif (PACKAGE_ROOT / "usd_assets_metadata.json").is_file():
            metadata_path = PACKAGE_ROOT / "usd_assets_metadata.json"
        else:
            metadata_path = MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "usd_assets_metadata.json"
    metadata = load_thor_assets_metadata(metadata_path)
    if asset_id not in metadata:
        raise KeyError(
            f"Asset id '{asset_id}' not in metadata at {metadata_path}. "
            "Check usd_assets_metadata.json."
        )
    info = metadata[asset_id]
    object_type = ObjectType.ARTICULATION if info.articulated else ObjectType.RIGID
    name = instance_name if instance_name else asset_id
    arena_object_cls = get_arena_object_class()
    return arena_object_cls(
        name=name,
        prim_path=None,
        usd_path=usd_path.as_posix(),
        object_type=object_type,
        initial_pose=initial_pose,
    )
