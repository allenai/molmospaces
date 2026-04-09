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
        for obja in (base / "objects" / "objaverse", base / "objaverse"):
            if obja.is_dir():
                versioned = obja / OBJAVERSE_DEFAULT_VERSION
                if versioned.is_dir():
                    return versioned
                return obja
    default = Path.home() / ".molmospaces" / "usd" / "assets" / "usd" / "objects" / "objaverse"
    return default.resolve() if default.exists() else default


def get_objaverse_usd_path(
    asset_id: str, assets_dir: Path | None = None, _tried: list | None = None
) -> Path | None:
    """Return the USD path for an Objaverse asset, or None if not found.

    Tries: assets_dir / obja_<id> / obja_<id>.usda; if assets_dir is versioned
    (e.g. .../objaverse/20260128) also tries parent; if assets_dir is
    .../objaverse also tries .../objaverse/20260128/obja_<id>/....
    """
    if assets_dir is None:
        assets_dir = get_objaverse_usd_root()
    else:
        assets_dir = Path(assets_dir).resolve()
    base = f"obja_{asset_id}"
    stem = assets_dir / f"obja_{asset_id}" / f"{base}.usda"
    candidates: list[Path] = [stem]
    if assets_dir.name == OBJAVERSE_DEFAULT_VERSION and assets_dir.parent.is_dir():
        candidates.append(assets_dir.parent / f"obja_{asset_id}" / f"{base}.usda")
    elif (assets_dir / OBJAVERSE_DEFAULT_VERSION).is_dir():
        candidates.append(assets_dir / OBJAVERSE_DEFAULT_VERSION / f"obja_{asset_id}" / f"{base}.usda")
    for p in candidates:
        if _tried is not None:
            _tried.append(p)
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
    tried: list[Path] = []
    usd_path = get_objaverse_usd_path(asset_id, assets_dir, _tried=tried)
    if usd_path is None or not usd_path.is_file():
        tried_str = ", ".join(str(p) for p in tried) if tried else "(none)"
        raise FileNotFoundError(
            f"Objaverse USD not found for id {asset_id}. Tried: {tried_str}. "
            "The benchmark may use a different Objaverse subset than your assets_root. "
            "Add obja_<id>/obja_<id>.usda under assets_root/objects/objaverse (or .../objaverse/20260128), "
            "or set MOLMO_OBJAVERSE_USD_DIR to a directory that contains it."
        )
    name = instance_name if instance_name else f"obja_{asset_id}"
    return Object(
        name=name,
        prim_path=None,
        usd_path=usd_path.as_posix(),
        object_type=ObjectType.RIGID,
        initial_pose=initial_pose,
    )
