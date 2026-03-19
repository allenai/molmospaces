"""Build an Isaac Lab Arena environment from an ArenaEpisodeSpec (from MolmoSpaces benchmark).

Uses the same environment as MolmoSpaces when possible: if the spec has scene_usd_path (resolved
from the episode's house_index, scene_dataset, data_split), that scene USD is loaded as the
background and object poses are applied in robot frame so the layout matches the benchmark.
Otherwise falls back to an Arena background by background_key.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

from molmo_spaces_isaac.arena.episode_to_arena import (
    ArenaEpisodeSpec,
    _inv_pose_7_wxyz,
    _pose_7_to_arena_pose,
    _pose_7_world_to_robot_frame,
)
from molmo_spaces_isaac.arena.molmospaces_pick_and_place_task import MolmoSpacesPickAndPlaceTask
from molmo_spaces_isaac.arena.objaverse_asset import create_objaverse_object_for_arena
from molmo_spaces_isaac.arena.thor_asset import create_thor_object_for_arena

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

try:
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False
    Object = None
    ObjectType = None


def build_arena_env_from_episode_spec(
    spec: ArenaEpisodeSpec,
    *,
    env_name: str = "molmospaces_arena_benchmark",
    embodiment_key: str = "franka",
    enable_cameras: bool = True,
    cli_args_list: list[str] | None = None,
    thor_assets_dir: Path | None = None,
    thor_metadata_path: Path | None = None,
    objaverse_assets_dir: Path | None = None,
    episode_length_s: float | None = None,
):
    """Build a registered Arena env from ArenaEpisodeSpec. Uses the episode's MolmoSpaces scene USD when spec.scene_usd_path is set; otherwise Arena background by background_key. Returns (env, env_builder)."""
    if not _ARENA_AVAILABLE:
        raise ImportError(
            "isaaclab_arena is required. Install from source (see Isaac Lab Arena documentation)."
        )
    asset_registry = AssetRegistry()
    robot_base_pose_7 = getattr(spec, "robot_base_pose", None) or [0.0] * 7
    if len(robot_base_pose_7) < 7:
        robot_base_pose_7 = (robot_base_pose_7 + [0.0] * 7)[:7]

    scene_usd = getattr(spec, "scene_usd_path", None)
    if scene_usd is not None and Path(scene_usd).is_file():
        pos_xyz, rot_wxyz = _inv_pose_7_wxyz(robot_base_pose_7)
        scene_pose = Pose(position_xyz=pos_xyz, rotation_wxyz=rot_wxyz)
        background = Object(
            name="molmospaces_scene",
            prim_path=None,
            usd_path=str(Path(scene_usd).resolve()),
            object_type=ObjectType.BASE,
            initial_pose=scene_pose,
        )
        log.info("Using MolmoSpaces scene %s as environment", scene_usd)
    else:
        background = asset_registry.get_asset_by_name(spec.background_key)()

    scene_assets: list = [background]
    pick_object = None
    place_object = None

    pick_start_z: float | None = None
    for item in spec.objects:
        # (name, asset_id, pose_7, source) with source "thor" or "objaverse"
        arena_name = item[0]
        asset_id = item[1]
        pose_7_world = item[2]
        pose_7 = _pose_7_world_to_robot_frame(pose_7_world, robot_base_pose_7)
        source = item[3] if len(item) >= 4 else "thor"
        pos_xyz, rot_wxyz = _pose_7_to_arena_pose(pose_7, apply_thor_up_axis=(source == "thor"))
        pose = Pose(position_xyz=pos_xyz, rotation_wxyz=rot_wxyz)
        if source == "objaverse":
            obj = create_objaverse_object_for_arena(
                asset_id,
                instance_name=arena_name,
                initial_pose=pose,
                assets_dir=objaverse_assets_dir,
            )
        else:
            obj = create_thor_object_for_arena(
                asset_id,
                instance_name=arena_name,
                initial_pose=pose,
                assets_dir=thor_assets_dir,
                metadata_path=thor_metadata_path,
            )
        scene_assets.append(obj)
        if arena_name == spec.pickup_name:
            pick_object = obj
            pick_start_z = float(pos_xyz[2])
        if spec.place_name and arena_name == spec.place_name:
            place_object = obj

    if pick_object is None:
        raise ValueError(
            f"Pick object '{spec.pickup_name}' not found in spec.objects. "
            f"Names: {[item[0] for item in spec.objects]}"
        )

    if place_object is None:
        place_object = asset_registry.get_asset_by_name("blue_sorting_bin")()
        place_object.set_initial_pose(
            Pose(position_xyz=(0.5, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )
        scene_assets.append(place_object)
        log.info("Using Arena blue_sorting_bin as place destination (pick-only or non-THOR receptacle)")

    scene = Scene(assets=scene_assets)
    success_radius = getattr(spec, "succ_pos_threshold", 0.12)
    task_type = getattr(spec, "task_type", "pick_and_place")
    place_start_pose_7: list[float] | None = None
    if spec.place_name:
        for item in spec.objects:
            if item[0] == spec.place_name:
                pose_7_world = item[2]
                pose_7 = _pose_7_world_to_robot_frame(pose_7_world, robot_base_pose_7)
                place_start_pose_7 = list(pose_7)
                break
    task = MolmoSpacesPickAndPlaceTask(
        pick_up_object=pick_object,
        destination_location=place_object,
        background_scene=background,
        success_radius=success_radius,
        episode_length_s=episode_length_s,
        use_molmospaces_success=True,
        place_receptacle_start_pose_7=place_start_pose_7,
        max_place_receptacle_pos_displacement=getattr(
            spec, "max_place_receptacle_pos_displacement", 0.1
        ),
        max_place_receptacle_rot_displacement=getattr(
            spec, "max_place_receptacle_rot_displacement", 0.785
        ),
        supported_fallback_thres=getattr(spec, "supported_fallback_thres", 0.01),
        default_extent=0.06,
        task_type=task_type,
        pick_start_z=pick_start_z,
        pick_lift_threshold_m=success_radius if task_type == "pick" else 0.01,
    )
    embodiment = asset_registry.get_asset_by_name(embodiment_key)(enable_cameras=enable_cameras)
    env_spec = IsaacLabArenaEnvironment(
        name=env_name,
        embodiment=embodiment,
        scene=scene,
        task=task,
    )
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    parser = get_isaaclab_arena_cli_parser()
    cli_args = parser.parse_args(cli_args_list or [])
    env_builder = ArenaEnvBuilder(env_spec, cli_args)
    env = env_builder.make_registered()
    return env, env_builder
