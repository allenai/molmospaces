"""Build an Isaac Lab Arena environment from an ArenaEpisodeSpec (from MolmoSpaces benchmark).

Uses the same environment as MolmoSpaces when possible: if the spec has scene_usd_path (resolved
from the episode's house_index, scene_dataset, data_split), that scene USD is loaded as the
background and object poses are applied in robot frame so the layout matches the benchmark.
Otherwise falls back to an Arena background by background_key.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path
from molmo_spaces_isaac.arena.episode_to_arena import (
    ArenaEpisodeSpec,
    _inv_pose_7_wxyz,
    _pose_7_to_arena_pose,
    _pose_7_world_to_robot_frame,
)
from molmo_spaces_isaac.arena.molmospaces_pick_task import MolmoSpacesPickTask
from molmo_spaces_isaac.arena.objaverse_asset import create_objaverse_object_for_arena
from molmo_spaces_isaac.arena.thor_asset import create_thor_object_for_arena, get_thor_usd_path

log = logging.getLogger(__name__)

try:
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    from molmo_spaces_isaac.arena.arena_collision_objects import get_arena_object_class

    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False
    ObjectType = None
    get_arena_object_class = None  # type: ignore[misc, assignment]


def _apply_franka_episode_init_qpos(embodiment, embodiment_key: str, spec: ArenaEpisodeSpec) -> None:
    """Merge episode ``robot.init_qpos`` (via spec.robot_init_joint_pos) into Franka ``scene_config.robot.init_state.joint_pos``."""
    if embodiment_key != "franka":
        return
    jp = getattr(spec, "robot_init_joint_pos", None)
    if not jp:
        return
    sc = getattr(embodiment, "scene_config", None)
    if sc is None or not hasattr(sc, "robot"):
        return
    robot_cfg = sc.robot
    if robot_cfg is None or not hasattr(robot_cfg, "init_state"):
        return
    existing = getattr(robot_cfg.init_state, "joint_pos", None) or {}
    try:
        merged = {**dict(existing), **jp}
        robot_cfg.init_state.joint_pos = merged
        print(
            "[molmospaces_arena] Applied episode robot.init_qpos to Franka (panda_joint1–7 + gripper).",
            flush=True,
        )
    except Exception as e:
        log.warning("Could not apply episode init_qpos to Franka: %s", e)


def _apply_franka_joint_pos_control(embodiment) -> bool:
    """Switch Franka arm action from IK delta-EEF to absolute joint position (8D). Returns True on success."""
    try:
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    except ImportError:
        log.warning("Could not import JointPositionActionCfg; keeping IK delta-EEF control.")
        return False
    action_cfg = getattr(embodiment, "action_config", None)
    if action_cfg is None or not hasattr(action_cfg, "arm_action"):
        action_cfg = getattr(getattr(embodiment, "env_cfg", None), "actions", None)
        if action_cfg is None or not hasattr(action_cfg, "arm_action"):
            return False
    action_cfg.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=1.0,
        use_default_offset=False,
    )
    print("[molmospaces_arena] Franka: switched to joint position control (8D actions).", flush=True)
    return True


def _apply_franka_abs_joint_pos_obs(embodiment) -> bool:
    """Replace joint_pos_rel obs with absolute joint_pos. Returns True on success."""
    try:
        import isaaclab.envs.mdp as mdp_isaac_lab
        from isaaclab.managers import ObservationTermCfg as ObsTerm
    except ImportError:
        log.warning("Could not import isaaclab mdp; keeping joint_pos_rel obs.")
        return False
    obs_cfg = getattr(embodiment, "observation_config", None)
    if obs_cfg is None:
        return False
    policy_cfg = getattr(obs_cfg, "policy", None)
    if policy_cfg is None or not hasattr(policy_cfg, "joint_pos"):
        return False
    policy_cfg.joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos)
    print("[molmospaces_arena] Franka: switched joint_pos obs to absolute (radians).", flush=True)
    return True


def _attach_franka_droid_cameras(embodiment, img_height: int = 224, img_width: int = 224) -> bool:
    """Attach DROID-style wrist + exo cameras to the Arena Franka.

    Mirrors FrankaDroidCameraSystem from molmo_spaces. Obs keys: camera_obs.wrist_cam_rgb, camera_obs.exo_cam_rgb.
    Returns True on success, False if isaaclab is unavailable.
    """
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass
    except ImportError:
        log.warning("isaaclab not available; skipping FrankaCameraCfg attachment.")
        return False

    _wrist_fl = 19.3  # ~57° HFOV (DROID D405)
    _exo_fl = 14.7   # ~71° HFOV

    @configclass
    class FrankaDroidCameraCfg:
        wrist_cam: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCam",
            update_period=0.0,
            height=img_height,
            width=img_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_wrist_fl,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.05),
                rot=(0.0, 0.7071, 0.7071, 0.0),  # 180° X + 90° Z roll correction
                convention="opengl",
            ),
        )

        exo_cam: CameraCfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0/ExoCam",
            update_period=0.0,
            height=img_height,
            width=img_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_exo_fl,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                # DROID offset is (0.1, 0.57, 0.66) from fr3_link0, but in DROID the robot
                # sits on a ~0.9 m stand. In Isaac, panda_link0 is at floor z=0, so we add
                # 0.9 m to Z to match the physical camera height above the workspace.
                pos=(0.1, 0.57, 1.56),
                rot=(-0.3633, -0.1241, 0.4263, 0.8191),  # wxyz from FrankaDroidCameraSystem
                convention="opengl",
            ),
        )

    embodiment.camera_config = FrankaDroidCameraCfg()
    print(
        "[molmospaces_arena] Attached FrankaDroidCameraCfg "
        f"(wrist_cam + exo_cam, {img_width}x{img_height}) to Franka embodiment.",
        flush=True,
    )
    return True


def _replicate_scene_to_all_envs(stage, scene_usd_path: str, n_envs: int) -> int:
    """Copy env_0's molmospaces_scene USD reference to env_1..N-1 (grid cloner skips BASE assets)."""
    try:
        from pxr import UsdGeom
    except ImportError:
        log.warning("pxr not available; cannot replicate scene to other envs.")
        return 0

    src_path = "/World/envs/env_0/molmospaces_scene"
    src_prim = stage.GetPrimAtPath(src_path)
    if not src_prim.IsValid():
        log.warning("env_0 molmospaces_scene not found; cannot replicate.")
        return 0

    local_xform = UsdGeom.Xformable(src_prim).GetLocalTransformation()

    n_added = 0
    n_replaced = 0
    for env_idx in range(1, n_envs):
        dst_path = f"/World/envs/env_{env_idx}/molmospaces_scene"
        existing = stage.GetPrimAtPath(dst_path)
        if existing.IsValid():
            # The @clone decorator in spawn_from_usd (called with {ENV_REGEX_NS}/molmospaces_scene)
            # tries to clone env_0's scene to env_1..N-1 during make_registered(). For the
            # multi-sublayer iTHOR USD, the clone produces prims with inconsistent world-space
            # transforms (scene appears at wrong position / "middle of nowhere") and un-patched
            # physics attributes. Remove and replace with our own USD reference.
            stage.RemovePrim(dst_path)
            n_replaced += 1
        dst_prim = stage.DefinePrim(dst_path, "Xform")
        dst_prim.GetReferences().AddReference(scene_usd_path)
        UsdGeom.Xformable(dst_prim).MakeMatrixXform().Set(local_xform)
        n_added += 1

    if n_added or n_replaced:
        print(
            f"[molmospaces_arena] Replicated molmospaces_scene to {n_added} env(s)"
            + (f" (replaced {n_replaced} bad clone(s))" if n_replaced else "")
            + ".",
            flush=True,
        )
    return n_added


def _patch_ithor_scene_physics_runtime(n_envs: int = 1, deactivate_object_types: list[str] | None = None) -> None:
    """Post-init iTHOR physics fixes: make rigid bodies kinematic, disable visual mesh collisions,
    deactivate conflicting decoration objects, and lock articulation joints."""
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics
    except ImportError:
        log.warning("omni.usd / pxr not available; skipping iTHOR scene physics patch.")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        log.warning("No USD stage; skipping iTHOR scene physics patch.")
        return

    n_kinematic = 0
    n_vis_disabled = 0
    n_deactivated = 0
    n_joints_locked = 0
    for env_idx in range(n_envs):
        scene_root = f"/World/envs/env_{env_idx}/molmospaces_scene"
        scene_prim = stage.GetPrimAtPath(scene_root)
        if not scene_prim.IsValid():
            continue
        # Usd.PrimRange sees prims added post-init (env_1..N-1 scenes); stage.Traverse() doesn't.
        for prim in Usd.PrimRange(scene_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
                n_kinematic += 1
            if "_visual_" in prim.GetName() and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                n_vis_disabled += 1
            if deactivate_object_types:
                prim_name_lower = prim.GetName().lower()
                if any(prim_name_lower.startswith(obj_type) for obj_type in deactivate_object_types):
                    prim.SetActive(False)
                    n_deactivated += 1
            stiffness_attr = prim.GetAttribute("drive:angular:physics:stiffness")
            if stiffness_attr and stiffness_attr.IsValid():
                stiffness_attr.Set(1e6)
                n_joints_locked += 1

    msg = (
        f"[molmospaces_arena] Scene physics patched: {n_kinematic} rigid bodies → kinematic, "
        f"{n_vis_disabled} visual mesh collisions → disabled, "
        f"{n_joints_locked} articulation joints → locked"
    )
    if n_deactivated:
        msg += f", {n_deactivated} conflicting scene decoration prim(s) → deactivated ({deactivate_object_types})"
    print(msg + ".", flush=True)



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
    scene_extra_translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_joint_pos_control: bool = False,
    num_envs: int = 1,
    env_spacing: float | None = None,
):
    """Build a registered Arena env from ArenaEpisodeSpec. Uses the episode's MolmoSpaces scene USD when spec.scene_usd_path is set; otherwise Arena background by background_key. Returns (env, env_builder)."""
    if not _ARENA_AVAILABLE or get_arena_object_class is None:
        raise ImportError(
            "isaaclab_arena is required. Install from source (see Isaac Lab Arena documentation)."
        )
    asset_registry = AssetRegistry()
    arena_object_cls = get_arena_object_class()
    robot_base_pose_7 = getattr(spec, "robot_base_pose", None) or [0.0] * 7
    if len(robot_base_pose_7) < 7:
        robot_base_pose_7 = (robot_base_pose_7 + [0.0] * 7)[:7]

    scene_usd = getattr(spec, "scene_usd_path", None)
    if scene_usd is not None and Path(scene_usd).is_file():
        rb = robot_base_pose_7
        pos_near_zero = all(abs(float(rb[i])) < 1e-5 for i in range(3))
        quat_identity = abs(float(rb[3]) - 1.0) < 1e-5 and all(abs(float(rb[i])) < 1e-5 for i in range(4, 7))
        if pos_near_zero and quat_identity:
            log.warning(
                "Episode robot_base_pose is at origin with identity rotation, but a MolmoSpaces scene USD "
                "is loaded. House origins rarely match the robot stand; the Franka may sit inside geometry "
                "(e.g. an island). Use real benchmark episodes with correct robot_base_pose, or "
                "--scene_extra_xyz to shift the scene root."
            )
        pos_xyz, rot_wxyz = _inv_pose_7_wxyz(robot_base_pose_7)
        ex = scene_extra_translation_xyz
        fine_z = float((os.environ.get("MOLMO_ARENA_SCENE_FINE_Z") or "0").strip() or "0")
        # --align_scene_floor_z_zero: skip the robot_base_pose Z contribution; use only
        # scene_extra_xyz.z + MOLMO_ARENA_SCENE_FINE_Z.  XY and yaw still follow robot_base_pose.
        if (os.environ.get("MOLMO_ARENA_SCENE_Z_ALIGN_WORLD_ZERO") or "").strip() in ("1", "true", "yes"):
            scene_z = ex[2] + fine_z
        else:
            scene_z = pos_xyz[2] + ex[2] + fine_z
        pos_xyz = (pos_xyz[0] + ex[0], pos_xyz[1] + ex[1], scene_z)
        scene_pose = Pose(position_xyz=pos_xyz, rotation_wxyz=rot_wxyz)
        background = arena_object_cls(
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
    pick_start_z: float | None = None
    for item in spec.objects:
        # (name, asset_id, pose_7, source) with source "thor" or "objaverse"
        arena_name = item[0]
        asset_id = item[1]
        pose_7_world = item[2]
        # World (episode) → robot frame: p_robot = inv(R_base) * (p_world - t_base); q_robot = inv(q_base)*q_world.
        pose_7 = list(_pose_7_world_to_robot_frame(pose_7_world, robot_base_pose_7))
        source = item[3] if len(item) >= 4 else "thor"
        if source == "thor" and arena_name == spec.pickup_name:
            z_extra = float((os.environ.get("MOLMO_ARENA_PICK_Z_EXTRA") or "0").strip() or "0")
            if z_extra != 0.0:
                pose_7[2] = float(pose_7[2]) + z_extra
        # THOR: +90° X on quaternion only (mesh up-axis); position xyz unchanged.
        pos_xyz, rot_wxyz = _pose_7_to_arena_pose(pose_7, apply_thor_up_axis=(source == "thor"))
        pose = Pose(position_xyz=pos_xyz, rotation_wxyz=rot_wxyz)
        if source == "thor" and arena_name == spec.pickup_name:
            usd_p = get_thor_usd_path(asset_id, thor_assets_dir)
            print(
                f"[molmospaces_arena] Pick '{asset_id}': USD path={usd_p} exists={usd_p.is_file()} "
                f"world_xyz={tuple(round(float(pose_7_world[i]), 4) for i in range(3))} "
                f"arena_spawn_xyz={tuple(round(float(pos_xyz[i]), 4) for i in range(3))}",
                flush=True,
            )
            if not usd_p.is_file():
                raise FileNotFoundError(
                    f"THOR pickup USD missing for '{asset_id}': {usd_p}. "
                    "Pass --assets_root to your ms-download install (with --assets thor), or set MOLMO_ISAAC_ASSETS_ROOT / MOLMO_THOR_USD_DIR."
                )
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

    if pick_object is None:
        raise ValueError(
            f"Pick object '{spec.pickup_name}' not found in spec.objects. "
            f"Names: {[item[0] for item in spec.objects]}"
        )

    scene = Scene(assets=scene_assets)
    success_radius = getattr(spec, "succ_pos_threshold", 0.01)
    task = MolmoSpacesPickTask(
        pick_up_object=pick_object,
        background_scene=background,
        episode_length_s=episode_length_s,
        pick_start_z=pick_start_z,
        pick_lift_threshold_m=success_radius,
    )
    embodiment = asset_registry.get_asset_by_name(embodiment_key)(enable_cameras=enable_cameras)
    if use_joint_pos_control and embodiment_key == "franka":
        ok_ctrl = _apply_franka_joint_pos_control(embodiment)
        ok_obs = _apply_franka_abs_joint_pos_obs(embodiment)
        if not ok_ctrl or not ok_obs:
            raise RuntimeError(
                "use_joint_pos_control=True but could not patch Arena Franka "
                f"(action_cfg found={ok_ctrl}, obs_cfg found={ok_obs}). "
                "Check that Arena Franka exposes 'action_config.arm_action' and "
                "'observation_config.policy.joint_pos' (or update the attribute paths in "
                "build_from_spec._apply_franka_joint_pos_control / _apply_franka_abs_joint_pos_obs)."
            )
    _apply_franka_episode_init_qpos(embodiment, embodiment_key, spec)
    if enable_cameras and embodiment_key == "franka":
        _attach_franka_droid_cameras(embodiment)
    _ev = getattr(embodiment, "event_config", None)
    if _ev is not None and hasattr(_ev, "randomize_franka_joint_state"):
        _ev.randomize_franka_joint_state = None
        print("[molmospaces_arena] Disabled randomize_franka_joint_state for eval.", flush=True)
    env_spec = IsaacLabArenaEnvironment(
        name=env_name,
        embodiment=embodiment,
        scene=scene,
        task=task,
    )
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    parser = get_isaaclab_arena_cli_parser()
    effective_cli = list(cli_args_list or [])
    if "--num_envs" not in effective_cli and num_envs != 1:
        effective_cli += ["--num_envs", str(num_envs)]
    cli_args = parser.parse_args(effective_cli)
    env_builder = ArenaEnvBuilder(env_spec, cli_args)
    import gymnasium as _gym
    _env_name, _env_cfg = env_builder.build_registered()
    if env_spacing is not None:
        _env_cfg.scene.env_spacing = float(env_spacing)
        print(f"[molmospaces_arena] env_spacing set to {env_spacing} m.", flush=True)
    env = _gym.make(_env_name, cfg=_env_cfg).unwrapped
    if num_envs > 1 and scene_usd is not None and Path(scene_usd).is_file():
        try:
            import omni.usd as _ousd
            _stage = _ousd.get_context().get_stage()
            if _stage is not None:
                _replicate_scene_to_all_envs(_stage, str(Path(scene_usd).resolve()), num_envs)
        except Exception as e:
            log.warning("Could not replicate iTHOR scene to additional envs: %s", e)
    pickup_base = spec.pickup_name.split("_")[0].lower()
    _patch_ithor_scene_physics_runtime(n_envs=num_envs, deactivate_object_types=[pickup_base])
    return env, env_builder
