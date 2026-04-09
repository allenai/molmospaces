"""Convert MolmoSpaces benchmark episode (JSON/EpisodeSpec) to Arena scene + task. Pick only; THOR and Objaverse (require_thor_only=False)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Matches molmo_spaces_isaac downloader SOURCE_TO_VERSION["scenes"]["usd"]["ithor"]
_DEFAULT_ITHOR_USD_SCENES_VERSION = "20260121"

# Optional: use molmo_spaces EpisodeSpec when available
try:
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    _HAS_MOLMO_SPACES = True
except ImportError:
    _HAS_MOLMO_SPACES = False
    EpisodeSpec = None

log = logging.getLogger(__name__)

# Pose in benchmark: [x, y, z, qw, qx, qy, qz]
# THOR assets in Isaac often need 90° around X for correct up-axis (same as viewer)
THOR_UP_AXIS_QUAT_WXYZ = (0.70710678, 0.70710678, 0.0, 0.0)


@dataclass
class ArenaEpisodeSpec:
    """Arena-ready spec for pick: scene (MolmoSpaces USD or fallback background_key), objects, pickup_name, robot_base_pose, success threshold."""

    background_key: str
    objects: list[tuple[str, str, list[float], str]]  # (name, asset_id, pose_7, "thor"|"objaverse")
    pickup_name: str
    robot_base_pose: list[float]
    task_type: str  # "pick"
    succ_pos_threshold: float = 0.12  # lift height >= this for pick success
    scene_usd_path: Path | None = None  # When set, use this MolmoSpaces scene as environment (same as benchmark)
    robot_init_joint_pos: dict[str, float] | None = None  # Isaac Panda joint_pos from episode robot.init_qpos (arm+gripper)


def _quat_mul_wxyz(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Multiply two quaternions (w, x, y, z). Result = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_rotate_vector_wxyz(q_wxyz: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate vector v by unit quaternion q (w,x,y,z). Returns q * (0,v) * conj(q)."""
    w, x, y, z = q_wxyz
    tw = -x * v[0] - y * v[1] - z * v[2]
    tx = w * v[0] + y * v[2] - z * v[1]
    ty = w * v[1] - x * v[2] + z * v[0]
    tz = w * v[2] + x * v[1] - y * v[0]
    return (
        -tw * x + tx * w - ty * z + tz * y,
        -tw * y + tx * z + ty * w - tz * x,
        -tw * z - tx * y + ty * x + tz * w,
    )


def _pose_7_world_to_robot_frame(pose_7_world: list[float], robot_base_pose_7: list[float]) -> list[float]:
    """Transform pose_7 from world to robot-centric frame (Franka at origin in Arena)."""
    if len(robot_base_pose_7) < 7:
        return list(pose_7_world)
    r_x, r_y, r_z = robot_base_pose_7[0], robot_base_pose_7[1], robot_base_pose_7[2]
    r_q = (robot_base_pose_7[3], robot_base_pose_7[4], robot_base_pose_7[5], robot_base_pose_7[6])
    o_x, o_y, o_z = pose_7_world[0], pose_7_world[1], pose_7_world[2]
    o_q = (pose_7_world[3], pose_7_world[4], pose_7_world[5], pose_7_world[6])
    inv_r_q = (r_q[0], -r_q[1], -r_q[2], -r_q[3])
    delta = (o_x - r_x, o_y - r_y, o_z - r_z)
    pos_robot = _quat_rotate_vector_wxyz(inv_r_q, delta)
    rot_robot = _quat_mul_wxyz(inv_r_q, o_q)
    return [pos_robot[0], pos_robot[1], pos_robot[2], rot_robot[0], rot_robot[1], rot_robot[2], rot_robot[3]]


def _inv_pose_7_wxyz(pose_7: list[float]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Inverse of pose [x,y,z,qw,qx,qy,qz]. Returns (position_xyz, rotation_wxyz) for the inverse transform."""
    if len(pose_7) < 7:
        return ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    rx, ry, rz = pose_7[0], pose_7[1], pose_7[2]
    r_q = (pose_7[3], pose_7[4], pose_7[5], pose_7[6])
    inv_q = (r_q[0], -r_q[1], -r_q[2], -r_q[3])
    pos_inv = _quat_rotate_vector_wxyz(inv_q, (-rx, -ry, -rz))
    return ((pos_inv[0], pos_inv[1], pos_inv[2]), inv_q)


def init_qpos_to_panda_joint_pos(robot_block: dict | None) -> dict[str, float] | None:
    """Map episode ``robot.init_qpos`` (arm, gripper, base) to Isaac Franka ``ArticulationCfg.init_state.joint_pos`` keys.

    Expects ``arm`` length 7 and optional ``gripper`` length 1–2 (sets ``panda_finger_joint.*`` to match Arena Franka's default key).
    Returns None if disabled via env or arm is missing/short.
    """
    if (os.environ.get("MOLMO_ARENA_IGNORE_EPISODE_INIT_QPOS") or "").strip().lower() in ("1", "true", "yes"):
        return None
    if not robot_block or not isinstance(robot_block, dict):
        return None
    iq = robot_block.get("init_qpos") or {}
    if not isinstance(iq, dict):
        return None
    arm = iq.get("arm")
    if not arm or not isinstance(arm, (list, tuple)) or len(arm) < 7:
        return None
    out: dict[str, float] = {}
    for i in range(7):
        out[f"panda_joint{i + 1}"] = float(arm[i])
    grip = iq.get("gripper") or []
    if isinstance(grip, (list, tuple)) and len(grip) >= 2:
        g = (float(grip[0]) + float(grip[1])) * 0.5
    elif isinstance(grip, (list, tuple)) and len(grip) == 1:
        g = float(grip[0])
    else:
        g = 0.04
    g = max(0.0, min(0.04, g))
    out["panda_finger_joint.*"] = g
    return out


def _ithor_floorplan_usd(scenes_root: Path, house_index: int) -> Path | None:
    """Find scene.usda for ithor FloorPlan{house_index}_physics under ms-download layouts."""
    root = Path(scenes_root).resolve()
    fp_dir = f"FloorPlan{int(house_index)}_physics"
    for ext in ("scene.usda", "scene.usdc"):
        for base in (root / "ithor", root / "scenes" / "ithor"):
            p = base / fp_dir / ext
            if p.is_file():
                return p
        # USD layout: scenes/ithor/ithor/<version>/FloorPlanN_physics/
        nested = root / "scenes" / "ithor" / "ithor"
        if nested.is_dir():
            env_ver = (os.environ.get("MOLMO_ITHOR_SCENES_VERSION") or "").strip()
            version_order: list[Path] = []
            if env_ver:
                v = nested / env_ver
                if v.is_dir():
                    version_order.append(v)
            def_ver = nested / _DEFAULT_ITHOR_USD_SCENES_VERSION
            if def_ver.is_dir() and def_ver not in version_order:
                version_order.append(def_ver)
            for v in sorted((p for p in nested.iterdir() if p.is_dir()), key=lambda x: x.name, reverse=True):
                if v not in version_order:
                    version_order.append(v)
            for vdir in version_order:
                p = vdir / fp_dir / ext
                if p.is_file():
                    return p
    return None


def resolve_episode_scene_usd_path(episode: dict, scenes_root: Path | None) -> Path | None:
    """Resolve episode (house_index, scene_dataset, data_split) to the MolmoSpaces scene USD path. Returns None if not found."""
    if scenes_root is None or not Path(scenes_root).is_dir():
        return None
    root = Path(scenes_root).resolve()
    house_index = episode.get("house_index")
    scene_dataset = (episode.get("scene_dataset") or "").strip()
    data_split = (episode.get("data_split") or "val").strip()
    if house_index is None:
        return None
    hi = int(house_index)
    if scene_dataset.lower() == "ithor":
        return _ithor_floorplan_usd(root, hi)
    # procthor-style: scenes_root/{dataset}-{split}/{split}_{house_idx}/scene.usda or _physics
    dataset_split = f"{scene_dataset}-{data_split}"
    for folder in (f"{data_split}_{hi}_physics", f"{data_split}_{hi}"):
        for ext in ("scene.usda", "scene.usdc"):
            p = root / dataset_split / folder / ext
            if p.is_file():
                return p
    return None


def _pose_7_to_arena_pose(pose_7: list[float], apply_thor_up_axis: bool = True) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert [x,y,z,qw,qx,qy,qz] to (position_xyz, rotation_wxyz). Optionally apply 90° X for THOR up-axis."""
    x, y, z = pose_7[0], pose_7[1], pose_7[2]
    qw, qx, qy, qz = pose_7[3], pose_7[4], pose_7[5], pose_7[6]
    if not apply_thor_up_axis:
        return ((x, y, z), (qw, qx, qy, qz))
    # 90° around X: quat (w,x,y,z) = (cos(45°), sin(45°), 0, 0)
    import math
    half = 0.5 * math.sqrt(2)
    rx_w, rx_x = half, half
    # quat multiply: result = rx * q_bench (rx applied first in world)
    rw = rx_w * qw - rx_x * qx
    rx = rx_w * qx + rx_x * qw
    ry = rx_w * qy + rx_x * qz
    rz = rx_w * qz - rx_x * qy
    return ((x, y, z), (rw, rx, ry, rz))


def _path_to_thor_asset_id(rel_path: str) -> str | None:
    """Extract THOR asset id from path (e.g. objects/thor/Bowl_27.xml -> Bowl_27), or None."""
    p = Path(rel_path)
    parts = p.parts
    if len(parts) >= 3 and parts[0] == "objects" and parts[1] == "thor":
        return p.stem
    return None


def _path_to_objaverse_asset_id(rel_path: str) -> str | None:
    """Extract Objaverse UUID from path (objects/objaverse/<uuid>/<uuid>.xml -> <uuid>), or None."""
    p = Path(rel_path)
    parts = p.parts
    if len(parts) >= 3 and parts[0] == "objects" and parts[1] == "objaverse":
        # objects/objaverse/<uuid>/<uuid>.xml -> <uuid>
        return p.parent.name
    return None


def _objaverse_uuid_from_object_name(name: str) -> str | None:
    """Extract 32-char hex Objaverse UUID from object name (e.g. egg_<uuid>_1_0_0 -> <uuid>), or None."""
    import re
    # 32 hexadecimal characters (Objaverse UUID)
    match = re.search(r"([0-9a-f]{32})", name.lower())
    if match:
        return match.group(1)
    return None


def episode_dict_to_arena_spec(
    episode: dict,
    *,
    background_key: str = "kitchen",
    require_thor_only: bool = True,
    scenes_root: Path | None = None,
) -> ArenaEpisodeSpec | None:
    """Build ArenaEpisodeSpec from episode dict. Pick only (PickTask); pick-and-place benchmarks are not supported. If scenes_root is set, resolves the episode's MolmoSpaces scene USD."""
    task = episode.get("task") or {}
    task_cls = task.get("task_cls") or ""
    task_type = (task.get("task_type") or "").strip().lower()
    if not task_type:
        if "PickTask" in task_cls and "PickAndPlace" not in task_cls:
            task_type = "pick"
        else:
            log.debug("Unsupported task_cls for Arena (pick only): %s", task_cls)
            return None
    if task_type != "pick":
        log.debug("Unsupported task_type for Arena (pick only): %s", task_type)
        return None

    sm = episode.get("scene_modifications") or {}
    added_objects = sm.get("added_objects") or {}
    object_poses = sm.get("object_poses") or {}

    pickup_obj_name = task.get("pickup_obj_name")
    if not pickup_obj_name:
        log.warning("Episode task missing pickup_obj_name")
        return None
    def get_pose(name: str, fallback_pose_key: str | None) -> list[float] | None:
        if name in object_poses:
            p = object_poses[name]
            if len(p) >= 7:
                return list(p[:7])
        if fallback_pose_key and task.get(fallback_pose_key) and len(task.get(fallback_pose_key)) >= 7:
            return list(task[fallback_pose_key][:7])
        return None

    # Resolve pickup object: from added_objects (path) or fallback: pose + Objaverse UUID from name
    pickup_asset_id = None
    pickup_source = "thor"
    pickup_pose = get_pose(pickup_obj_name, "pickup_obj_start_pose")

    if pickup_obj_name in added_objects:
        pickup_path = added_objects[pickup_obj_name]
        pickup_asset_id = _path_to_thor_asset_id(pickup_path)
        if not pickup_asset_id:
            pickup_asset_id = _path_to_objaverse_asset_id(pickup_path)
            pickup_source = "objaverse"
    else:
        log.debug("Pickup object '%s' not in added_objects, trying Objaverse UUID from name", pickup_obj_name)
        if pickup_pose and not require_thor_only:
            pickup_asset_id = _objaverse_uuid_from_object_name(pickup_obj_name)
            if pickup_asset_id:
                pickup_source = "objaverse"

    if not pickup_asset_id:
        log.debug(
            "Pickup object '%s' not in added_objects and could not infer THOR/Objaverse id",
            pickup_obj_name,
        )
        return None
    if require_thor_only and pickup_source != "thor":
        log.debug("Pickup object is Objaverse but require_thor_only=True")
        return None
    if not pickup_pose:
        log.warning("No pose for pickup object '%s'", pickup_obj_name)
        return None

    # Build list of objects to spawn: (name, asset_id, pose_7, source)
    objects: list[tuple[str, str, list[float], str]] = []
    pickup_arena_name = pickup_obj_name.replace("/", "_")
    objects.append((pickup_arena_name, pickup_asset_id, pickup_pose, pickup_source))

    robot_base_pose = list(task.get("robot_base_pose") or [0.0] * 7)[:7]
    if len(robot_base_pose) < 7:
        robot_base_pose = (robot_base_pose + [0.0] * 7)[:7]

    succ_pos_threshold = float(task.get("succ_pos_threshold", 0.01))
    scene_usd_path = resolve_episode_scene_usd_path(episode, scenes_root)
    robot_init_joint_pos = init_qpos_to_panda_joint_pos(episode.get("robot"))

    return ArenaEpisodeSpec(
        background_key=background_key,
        scene_usd_path=scene_usd_path,
        objects=objects,
        pickup_name=pickup_arena_name,
        robot_base_pose=robot_base_pose,
        task_type=task_type,
        succ_pos_threshold=succ_pos_threshold,
        robot_init_joint_pos=robot_init_joint_pos,
    )


def episode_spec_to_arena_spec(
    episode_spec: "EpisodeSpec",
    *,
    background_key: str = "kitchen",
    require_thor_only: bool = True,
) -> ArenaEpisodeSpec | None:
    """Build ArenaEpisodeSpec from MolmoSpaces EpisodeSpec (Pydantic). Requires molmo_spaces; uses model_dump()."""
    if not _HAS_MOLMO_SPACES or EpisodeSpec is None:
        raise ImportError("molmo_spaces is required for episode_spec_to_arena_spec. Install molmo_spaces.")
    return episode_dict_to_arena_spec(
        episode_spec.model_dump(),
        background_key=background_key,
        require_thor_only=require_thor_only,
    )
