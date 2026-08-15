import functools

from molmo_spaces.g1_molmo_port import GRASPS_DIR, grasp_source_dir
from molmo_spaces.utils.constants.object_constants import THOR_PICKUP_OBJECTS_LOWERCASE

# Shared by Scene (attach prefix) and G1Env (ObjectManager's is_excluded uses
# `env.config.robot_config.robot_namespace` as a substring check) so both stay
# in sync with a single literal.
ROBOT_PREFIX = "robot_0/"

# Byte-for-byte the same 69-category list as molmo_spaces' own canonical
# THOR_PICKUP_OBJECTS_LOWERCASE (verified via a full element-by-element diff)
# -- this g1_molmo_port fork previously carried its own copy with one typo
# ("toiletpaperused up", a stray literal space) that the canonical list
# doesn't have. Aliased under gold's own name since is_pickup_type() and
# scene.py both reference THOR_PICKUP_TYPES.
THOR_PICKUP_TYPES = THOR_PICKUP_OBJECTS_LOWERCASE

# Categories excluded from THOR_PICKUP_TYPES — too soft / large / deformable to grasp well.
THOR_PICKUP_BLACKLIST = [
    "pillow",
]


# Maps a substring of a body/geom name to a THOR material category. Used by the
# scene texture randomizer to pick category-appropriate textures (walls get wall
# textures, floors get floor textures, etc.). The right-hand sides must match
# keys in objects/thor/material-database.json.
SCENE_TEXTURE_CATEGORIES = {
    "wall": "Wall",
    "backsplash": "Wall",
    "ceiling": "Wall",
    "floor": "Floor",
    "room": "Floor",  # ProcTHOR floor body name
    "counter": "CounterTop",
    "countertop": "CounterTop",
    "island": "Table",  # upstream convention
    "table": "Table",
    "desk": "Table",
    "shelf": "Table",
    "shelving": "Table",
    "doorway": "Doorway",
    "door": "Doorway",
    "drawer": "Doorway",
    "cabinet": "Doorway",
    "handle": "Doorway",
}


def classify_scene_geom(body_name: str, geom_name: str) -> str | None:
    """Return the THOR category for a scene geom based on substring matches in
    its body name or geom name, or None if it doesn't match any architectural
    keyword. Body name takes precedence (procthor names are more reliable)."""
    b = (body_name or "").lower()
    g = (geom_name or "").lower()
    for kw, cat in SCENE_TEXTURE_CATEGORIES.items():
        if kw in b or kw in g:
            return cat
    return None


def is_pickup_type(category):
    cat = category.lower().replace("_", "")
    if any(cat.startswith(t) or t.startswith(cat) for t in THOR_PICKUP_BLACKLIST):
        return False
    return any(cat.startswith(t) or t.startswith(cat) for t in THOR_PICKUP_TYPES)


_VALID_GRASP_UIDS_CACHE = GRASPS_DIR / ".valid_uids.txt"


@functools.lru_cache(maxsize=1)
def _valid_grasp_uids() -> frozenset:
    """Asset IDs with a *_grasps_filtered.npz on disk. Cached at assets/grasps/.valid_uids.txt
    (delete to force a re-scan after adding new grasps)."""
    base = GRASPS_DIR
    if not base.is_dir():
        return frozenset()
    if _VALID_GRASP_UIDS_CACHE.exists():
        with open(_VALID_GRASP_UIDS_CACHE) as f:
            return frozenset(line.strip() for line in f if line.strip())
    uids = set()
    for sub in (
        grasp_source_dir(""),
        grasp_source_dir("droid"),
        grasp_source_dir("droid_objaverse"),
    ):
        if not sub.is_dir():
            continue
        for d in sub.iterdir():
            if not d.is_dir():
                continue
            if (d / f"{d.name}_grasps_filtered.npz").exists():
                uids.add(d.name)
    try:
        with open(_VALID_GRASP_UIDS_CACHE, "w") as f:
            for u in sorted(uids):
                f.write(u + "\n")
    except OSError:
        pass
    return frozenset(uids)


def has_valid_grasp(asset_id: str) -> bool:
    """True iff a grasp transform file exists for this asset_id."""
    if not asset_id:
        return False
    return asset_id in _valid_grasp_uids()


def joint_grasp_path(thor_object_name: str, thor_joint_name: str = ""):
    """Filesystem path for a per-joint grasp file. Mirrors upstream
    molmospaces layout: `grasps/<obj_thor>/<joint_thor>_grasps_filtered.npz`,
    with `grasps/droid/<obj_thor>/...` as fallback. Returns `None` if no file
    exists. If `thor_joint_name` is empty, falls back to the convention
    `<obj_thor>_joint` (works for objects whose primary joint is named that)."""
    if not thor_object_name:
        return None
    jn = thor_joint_name or f"{thor_object_name}_joint"
    candidates = [
        grasp_source_dir("") / thor_object_name / f"{jn}_grasps_filtered.npz",
        grasp_source_dir("droid") / thor_object_name / f"{jn}_grasps_filtered.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def has_joint_grasps(thor_object_name: str, thor_joint_name: str = "") -> bool:
    return joint_grasp_path(thor_object_name, thor_joint_name) is not None
