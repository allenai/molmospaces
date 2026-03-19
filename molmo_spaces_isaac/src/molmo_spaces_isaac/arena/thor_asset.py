"""Create Isaac Lab Arena Object instances from MolmoSpaces THOR USD assets."""

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
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.pose import Pose

    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False
    Object = None
    ObjectType = None
    Pose = None


def get_thor_usd_path(asset_id: str, assets_dir: Path | None = None) -> Path:
    """Return the USD path for a THOR asset (e.g. Apple_1 -> .../Apple_1_mesh/Apple_1_mesh.usda)."""
    if assets_dir is None:
        assets_dir = get_thor_assets_root()
    mesh_dir = assets_dir / f"{asset_id}_mesh"
    base_name = f"{asset_id}_mesh"
    return mesh_dir / f"{base_name}.usda"


def get_rigid_body_relative_path(usd_path: Path) -> str:
    """Path from object root to RigidBodyAPI prim (default_prim_name/rel). Empty if pxr unavailable."""
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        return ""

    stage = Usd.Stage.Open(usd_path.as_posix())
    if not stage:
        return ""
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        return ""
    root_path = default_prim.GetPath()
    default_prim_name = default_prim.GetName()
    for prim in _iter_prims(default_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            p = prim.GetPath()
            if p == root_path:
                return ""
            # Path relative to default prim
            rel = str(p).replace(str(root_path), "", 1).lstrip("/")
            # Include default prim name so path matches post-spawn hierarchy: name/default_prim/rel
            return f"{default_prim_name}/{rel}" if rel else default_prim_name
    return ""


def _iter_prims(prim):
    out = [prim]
    for c in prim.GetChildren():
        out.extend(_iter_prims(c))
    return out


def create_thor_object_for_arena(
    asset_id: str,
    instance_name: str | None = None,
    initial_pose: Pose | None = None,
    assets_dir: Path | None = None,
    metadata_path: Path | None = None,
    usd_path_override: Path | str | None = None,
):
    """Create an Arena Object for a THOR asset (RIGID or ARTICULATION) for Scene(assets=[...])."""
    if not _ARENA_AVAILABLE:
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
    return Object(
        name=name,
        prim_path=None,
        usd_path=usd_path.as_posix(),
        object_type=object_type,
        initial_pose=initial_pose,
    )
