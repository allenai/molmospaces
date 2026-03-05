"""Create Isaac Lab Arena Object instances from MolmoSpaces Objaverse USD assets."""

from __future__ import annotations

import os
from pathlib import Path

# Default version subdir when using MOLMO_ISAAC_ASSETS_ROOT (e.g. molmospaces_isaac layout)

OBJAVERSE_DEFAULT_VERSION = "20260128"

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


def get_objaverse_usd_root() -> Path:
    """Objaverse USD root: MOLMO_OBJAVERSE_USD_DIR, or MOLMO_ISAAC_ASSETS_ROOT/objects/objaverse, or ~/.molmospaces/..."""
    raw = os.environ.get("MOLMO_OBJAVERSE_USD_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    root = os.environ.get("MOLMO_ISAAC_ASSETS_ROOT")
    if root:
        base = Path(root).expanduser().resolve()
        obja = base / "objects" / "objaverse"
        if obja.is_dir():
            versioned = obja / OBJAVERSE_DEFAULT_VERSION
            if versioned.is_dir():
                return versioned
            return obja
    default = Path.home() / ".molmospaces" / "usd" / "assets" / "usd" / "objects" / "objaverse"
    return default.resolve() if default.exists() else default


def get_objaverse_usd_path(asset_id: str, assets_dir: Path | None = None) -> Path | None:
    """Return the USD path for an Objaverse asset, or None if not found.

    Tries: assets_dir / obja_<asset_id> / obja_<asset_id>.usda; if assets_dir
    is a versioned dir (e.g. .../objaverse/20260128), also tries parent dir.
    """
    if assets_dir is None:
        assets_dir = get_objaverse_usd_root()
    else:
        assets_dir = Path(assets_dir).resolve()
    base = f"obja_{asset_id}"
    candidates = [assets_dir / f"obja_{asset_id}" / f"{base}.usda"]
    if assets_dir.name == OBJAVERSE_DEFAULT_VERSION and assets_dir.parent.is_dir():
        candidates.append(assets_dir.parent / f"obja_{asset_id}" / f"{base}.usda")
    for p in candidates:
        if p.is_file():
            return p
    return None


def create_objaverse_object_for_arena(
    asset_id: str,
    instance_name: str | None = None,
    initial_pose: "Pose | None" = None,
    assets_dir: Path | None = None,
):
    """Create an Arena Object (RIGID) for an Objaverse asset for Scene(assets=[...])."""
    if not _ARENA_AVAILABLE:
        raise ImportError(
            "isaaclab_arena is required for Objaverse in Arena. "
            "Install from source (see Isaac Lab Arena documentation)."
        )
    usd_path = get_objaverse_usd_path(asset_id, assets_dir)
    if usd_path is None or not usd_path.is_file():
        root = assets_dir if assets_dir is not None else get_objaverse_usd_root()
        tried = root / f"obja_{asset_id}" / f"obja_{asset_id}.usda"
        raise FileNotFoundError(
            f"Objaverse USD not found: {tried}. "
            f"Set MOLMO_OBJAVERSE_USD_DIR to the dir containing obja_<id> subdirs, or ensure assets_root/objects/objaverse/ has obja_<id>/obja_<id>.usda."
        )
    name = instance_name if instance_name else f"obja_{asset_id}"
    return Object(
        name=name,
        prim_path=None,
        usd_path=usd_path.as_posix(),
        object_type=ObjectType.RIGID,
        initial_pose=initial_pose,
    )
