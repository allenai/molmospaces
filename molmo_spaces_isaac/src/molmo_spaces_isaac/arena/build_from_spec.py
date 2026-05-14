"""Build an Isaac Lab Arena environment from an ArenaEpisodeSpec (from MolmoSpaces benchmark).

Uses the same environment as MolmoSpaces when possible: if the spec has scene_usd_path (resolved
from the episode's house_index, scene_dataset, data_split), that scene USD is loaded as the
background and object poses are applied in robot frame so the layout matches the benchmark.
Otherwise falls back to an Arena background by background_key.
"""

from __future__ import annotations

import logging
import math
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
from molmo_spaces_isaac.arena.scene_object_reference import SceneRigidObjectReference
from molmo_spaces_isaac.arena.thor_asset import (
    create_thor_object_for_arena,
    get_thor_usd_path,
    get_usd_up_axis,
    should_apply_thor_up_axis_correction,
)

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
    """Apply episode ``robot.init_qpos`` to both robot init state and Arena's reset event.

    Arena's Franka/DROID embodiments install an ``init_franka_arm_pose`` reset event that overwrites
    ``scene_config.robot.init_state.joint_pos`` during ``env.reset()``. Patch both places so the
    initial robot state matches the MolmoSpaces episode.
    """
    if embodiment_key != "franka" and not str(embodiment_key).startswith("droid"):
        return
    jp = getattr(spec, "robot_init_joint_pos", None)
    if not jp:
        return
    grip = float(jp.get("panda_finger_joint.*", jp.get("finger_joint", 0.0)))
    sc = getattr(embodiment, "scene_config", None)
    robot_cfg = getattr(sc, "robot", None) if sc is not None else None
    existing = getattr(getattr(robot_cfg, "init_state", None), "joint_pos", None) or {}
    merged = dict(existing)
    if str(embodiment_key).startswith("droid"):
        for i in range(7):
            key = f"panda_joint{i + 1}"
            if key in jp:
                merged[key] = jp[key]
        merged["finger_joint"] = grip
    else:
        merged.update(jp)
    try:
        if robot_cfg is not None and hasattr(robot_cfg, "init_state"):
            robot_cfg.init_state.joint_pos = merged
        print(
            "[molmospaces_arena] Applied episode robot.init_qpos to robot init_state.",
            flush=True,
        )
    except Exception as e:
        log.warning("Could not apply episode init_qpos to robot init_state: %s", e)

    ev = getattr(embodiment, "event_config", None)
    init_event = getattr(ev, "init_franka_arm_pose", None) if ev is not None else None
    params = getattr(init_event, "params", None)
    if not isinstance(params, dict) or "default_pose" not in params:
        return
    try:
        default_pose = [float(x) for x in list(params["default_pose"])]
        for i in range(7):
            key = f"panda_joint{i + 1}"
            if key in jp and i < len(default_pose):
                default_pose[i] = float(jp[key])
        if str(embodiment_key).startswith("droid"):
            # DROID reset vector order: 7 Panda joints, then Robotiq finger_joint and mimic joints.
            if len(default_pose) > 7:
                default_pose[7] = grip
        else:
            # Franka reset vector order: 7 Panda joints + two panda finger joints.
            if len(default_pose) > 7:
                default_pose[7] = grip
            if len(default_pose) > 8:
                default_pose[8] = grip
        params["default_pose"] = default_pose
        print(
            "[molmospaces_arena] Patched init_franka_arm_pose reset event with episode robot.init_qpos.",
            flush=True,
        )
    except Exception as e:
        log.warning("Could not patch init_franka_arm_pose reset event: %s", e)


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


def _apply_droid_joint_velocity_control(embodiment) -> bool:
    """Switch DROID arm action to joint velocity control for OpenPI DROID checkpoints."""
    try:
        from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
    except ImportError:
        log.warning("Could not import JointVelocityActionCfg; keeping existing DROID action control.")
        return False
    action_cfg = getattr(embodiment, "action_config", None)
    if action_cfg is None or not hasattr(action_cfg, "arm_action"):
        return False
    action_cfg.arm_action = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )
    sc = getattr(embodiment, "scene_config", None)
    robot_cfg = getattr(sc, "robot", None) if sc is not None else None
    actuators = getattr(robot_cfg, "actuators", None) if robot_cfg is not None else None
    if isinstance(actuators, dict):
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            actuator = actuators.get(actuator_name)
            if actuator is not None and hasattr(actuator, "stiffness"):
                actuator.stiffness = 0.0
    print(
        "[molmospaces_arena] DROID: switched to joint velocity control "
        "(pi05_droid action space; Panda position stiffness disabled).",
        flush=True,
    )
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


def _attach_franka_droid_cameras(embodiment, img_height: int = 352, img_width: int = 624) -> bool:
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

    img_height = int((os.environ.get("MOLMO_ARENA_DROID_CAMERA_HEIGHT") or str(img_height)).strip())
    img_width = int((os.environ.get("MOLMO_ARENA_DROID_CAMERA_WIDTH") or str(img_width)).strip())

    # Match MolmoSpaces' raw DROID camera renders. OpenPI receives these after
    # resize-with-padding to 224x224.
    _wrist_fl = _vertical_fov_to_focal_length(52.0)
    _exo_fl = _vertical_fov_to_focal_length(71.0)

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
                # Legacy Franka fallback: keep the exo camera at the MolmoSpaces
                # world-height target. The DROID embodiment path below uses the
                # calibrated robot root plus a local camera offset instead.
                pos=(0.1, 0.57, 1.24),
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


def _vertical_fov_to_focal_length(fov_degrees: float, vertical_aperture: float = 3.024) -> float:
    """Convert MuJoCo-style vertical FOV degrees to Isaac camera focal length."""
    fov = max(1.0, min(179.0, float(fov_degrees)))
    return float(vertical_aperture) / (2.0 * math.tan(math.radians(fov) * 0.5))


def _camera_spec_by_name(camera_specs: list[dict] | None, names: tuple[str, ...]) -> dict | None:
    for spec in camera_specs or []:
        if isinstance(spec, dict) and spec.get("name") in names:
            return spec
    return None


def _quat_mul_wxyz(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_rotate_vector_wxyz(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    inv = (q[0], -q[1], -q[2], -q[3])
    rotated = _quat_mul_wxyz(_quat_mul_wxyz(q, (0.0, *v)), inv)
    return (rotated[1], rotated[2], rotated[3])


def _patch_droid_molmospaces_cameras(
    embodiment,
    img_height: int = 352,
    img_width: int = 624,
    camera_specs: list[dict] | None = None,
) -> bool:
    """Patch Arena's DROID cameras toward MolmoSpaces eval camera poses.

    MolmoSpaces DROID evals define the shoulder camera at (0.1, 0.57, 0.66) from fr3_link0.
    Arena's DROID USD root frame does not line up with MuJoCo's fr3_link0 frame, so the
    default camera Z is paired with the calibrated DROID root mount below to preserve the
    same world camera height.
    """
    if (os.environ.get("MOLMO_ARENA_DROID_SKIP_CAMERA_PATCH") or "").strip().lower() in ("1", "true", "yes"):
        print("[molmospaces_arena] Keeping Arena DROID camera defaults for this run.", flush=True)
        return False

    cam_cfg = getattr(embodiment, "camera_config", None)
    if cam_cfg is None:
        return False

    img_height = int((os.environ.get("MOLMO_ARENA_DROID_CAMERA_HEIGHT") or str(img_height)).strip())
    img_width = int((os.environ.get("MOLMO_ARENA_DROID_CAMERA_WIDTH") or str(img_width)).strip())

    def _set_camera(
        name: str,
        pos,
        rot,
        focal_length: float | None = None,
        f_stop: float | None = None,
        prim_path: str | None = None,
    ) -> bool:
        cam = getattr(cam_cfg, name, None)
        if cam is None:
            return False
        if prim_path is not None and hasattr(cam, "prim_path"):
            cam.prim_path = prim_path
        off = getattr(cam, "offset", None)
        if off is not None:
            off.pos = tuple(pos)
            off.rot = tuple(rot)
            off.convention = "opengl"
        if hasattr(cam, "height"):
            cam.height = img_height
        if hasattr(cam, "width"):
            cam.width = img_width
        spawn = getattr(cam, "spawn", None)
        if spawn is not None:
            if focal_length is not None and hasattr(spawn, "focal_length"):
                spawn.focal_length = focal_length
            if hasattr(spawn, "horizontal_aperture"):
                spawn.horizontal_aperture = 5.376
            if hasattr(spawn, "vertical_aperture"):
                spawn.vertical_aperture = 3.024
            if f_stop is not None and hasattr(spawn, "f_stop"):
                spawn.f_stop = f_stop
        return True

    exo_spec = _camera_spec_by_name(camera_specs, ("droid_shoulder_light_randomization", "exo_camera_1"))
    wrist_spec = _camera_spec_by_name(camera_specs, ("wrist_camera_zed_mini", "wrist_camera"))
    exo_fov = float(exo_spec.get("fov", 71.0)) if exo_spec else 71.0
    wrist_fov = float(wrist_spec.get("fov", 52.0)) if wrist_spec else 52.0
    exo_fl = float(
        (os.environ.get("MOLMO_ARENA_DROID_EXO_FOCAL_LENGTH") or str(_vertical_fov_to_focal_length(exo_fov))).strip()
    )
    wrist_fl = float(
        (os.environ.get("MOLMO_ARENA_DROID_WRIST_FOCAL_LENGTH") or str(_vertical_fov_to_focal_length(wrist_fov))).strip()
    )

    def _tuple_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        values = tuple(float(part.strip()) for part in raw.replace(",", " ").split())
        if len(values) != len(default):
            raise ValueError(f"{name} must contain {len(default)} floats, got {len(values)}: {raw!r}")
        return values

    if exo_spec and isinstance(exo_spec.get("camera_offset"), (list, tuple)):
        exo_offset = tuple(float(x) for x in exo_spec["camera_offset"][:3])
        # Arena's flattened DROID root is not MuJoCo's fr3_link0. Keep the
        # calibrated camera-height delta while preserving per-episode XY jitter.
        z_bias = float((os.environ.get("MOLMO_ARENA_DROID_CAMERA_Z_BIAS") or "0.135").strip() or "0.135")
        exo_pos = (exo_offset[0], exo_offset[1], exo_offset[2] + z_bias)
    else:
        exo_z = float((os.environ.get("MOLMO_ARENA_DROID_EXO_Z") or "0.795").strip() or "0.795")
        exo_pos = (0.1, 0.57, exo_z)
    if exo_spec and isinstance(exo_spec.get("camera_quaternion"), (list, tuple)):
        exo_rot = tuple(float(x) for x in exo_spec["camera_quaternion"][:4])
    else:
        exo_rot = (-0.3633, -0.1241, 0.4263, 0.8191)

    left_ok = _set_camera(
        "external_camera",
        exo_pos,
        exo_rot,
        focal_length=exo_fl,
    )
    _set_camera(
        "external_camera_2",
        (exo_pos[0], -abs(exo_pos[1]), exo_pos[2]),
        (0.8190819, -0.42629058, 0.12409726, -0.36329197),
        focal_length=exo_fl,
    )
    # Match MolmoSpaces' DROID wrist_camera_zed_mini. The Arena DROID USD uses a
    # different Robotiq camera attachment frame, so this local transform keeps the
    # wrist image in the same gripper-relative frame used by the MuJoCo eval.
    wrist_frame_q = (-0.5, 0.5, -0.5, 0.5)
    if wrist_spec and isinstance(wrist_spec.get("camera_offset"), (list, tuple)):
        wrist_offset = tuple(float(x) for x in wrist_spec["camera_offset"][:3])
        wrist_pos_default = _quat_rotate_vector_wxyz(wrist_frame_q, wrist_offset)
        wrist_x_bias = float((os.environ.get("MOLMO_ARENA_DROID_WRIST_X_BIAS") or "-0.01817").strip() or "-0.01817")
        wrist_z_bias = float((os.environ.get("MOLMO_ARENA_DROID_WRIST_Z_BIAS") or "0.0").strip() or "0.0")
        wrist_pos_default = (
            wrist_pos_default[0] + wrist_x_bias,
            wrist_pos_default[1],
            wrist_pos_default[2] + wrist_z_bias,
        )
    else:
        wrist_pos_default = (0.00405, -0.018015, -0.070726)
    if wrist_spec and isinstance(wrist_spec.get("camera_quaternion"), (list, tuple)):
        wrist_q = tuple(float(x) for x in wrist_spec["camera_quaternion"][:4])
        wrist_rot_default = _quat_mul_wxyz(wrist_frame_q, wrist_q)
    else:
        wrist_rot_default = (0.40693, -0.604224, -0.591482, 0.345645)
    wrist_pos = _tuple_env("MOLMO_ARENA_DROID_WRIST_POS", wrist_pos_default)
    wrist_rot = _tuple_env("MOLMO_ARENA_DROID_WRIST_ROT", wrist_rot_default)
    wrist_prim_path = (os.environ.get("MOLMO_ARENA_DROID_WRIST_PRIM_PATH") or "").strip() or None
    _set_camera(
        "wrist_camera",
        wrist_pos,
        wrist_rot,
        focal_length=wrist_fl,
        f_stop=0.0,
        prim_path=wrist_prim_path,
    )
    if left_ok:
        print(
            "[molmospaces_arena] Patched DROID cameras from MolmoSpaces episode specs "
            f"({img_width}x{img_height}, exo_fov={exo_fov:.2f}, wrist_fov={wrist_fov:.2f}).",
            flush=True,
        )
    return left_ok


def _apply_droid_molmospaces_mount_pose(embodiment) -> tuple[float, float, float] | None:
    """Place Arena's DROID root so the same qpos matches MolmoSpaces/MuJoCo TCP pose."""
    disable = (
        os.environ.get("MOLMO_ARENA_DROID_DISABLE_MOUNT_POSE")
        or os.environ.get("MOLMO_ARENA_DROID_DISABLE_MOUNT_HEIGHT")
        or ""
    )
    if disable.strip().lower() in ("1", "true", "yes"):
        return None
    # MuJoCo's fr3_link0 mount is 0.58 m above robot_base_pose, but Arena's flattened
    # DROID USD root frame is offset differently. 0.445 m matches the MuJoCo TCP height
    # for the same Franka joint pose while still removing Arena's visual stand prop.
    mount_x = float((os.environ.get("MOLMO_ARENA_DROID_MOUNT_X") or "0.0").strip() or "0.0")
    mount_y = float((os.environ.get("MOLMO_ARENA_DROID_MOUNT_Y") or "0.0").strip() or "0.0")
    mount_z = float((os.environ.get("MOLMO_ARENA_DROID_MOUNT_Z") or "0.445").strip() or "0.445")
    sc = getattr(embodiment, "scene_config", None)
    robot_cfg = getattr(sc, "robot", None) if sc is not None else None
    init_state = getattr(robot_cfg, "init_state", None) if robot_cfg is not None else None
    if init_state is None:
        return None
    pos = list(getattr(init_state, "pos", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    while len(pos) < 3:
        pos.append(0.0)
    pos[0] = mount_x
    pos[1] = mount_y
    pos[2] = mount_z
    init_state.pos = tuple(float(v) for v in pos[:3])
    print(
        "[molmospaces_arena] Set DROID robot root to "
        f"xyz=({mount_x:.4f}, {mount_y:.4f}, {mount_z:.4f}) m "
        "for MolmoSpaces/MuJoCo TCP-pose parity.",
        flush=True,
    )
    return mount_x, mount_y, mount_z


def _remove_droid_arena_stand_for_molmospaces(embodiment) -> bool:
    """Remove Arena's decorative DROID stand so the robot matches MolmoSpaces.

    MolmoSpaces' ``franka_droid`` MuJoCo model exposes ``robot_0/fr3_link0`` at
    its own mount frame and does not include a separate stand body. Arena's DROID
    embodiment adds ``Robot_Stand`` as a convenience scene prop, which makes
    visual/policy parity worse for these benchmark scenes. The actual Arena root
    height is calibrated separately against MuJoCo TCP height.
    """
    if (os.environ.get("MOLMO_ARENA_DROID_KEEP_STAND") or "").strip().lower() in ("1", "true", "yes"):
        return False
    sc = getattr(embodiment, "scene_config", None)
    if sc is None or not hasattr(sc, "stand"):
        return False
    sc.stand = None
    print("[molmospaces_arena] Removed Arena DROID Robot_Stand to match MolmoSpaces franka_droid.", flush=True)
    return True


def _patch_droid_molmospaces_gripper_actuators(embodiment) -> bool:
    """Use a Robotiq actuator set that actually drives the closing finger chain.

    Arena's DROID config actuates only ``finger_joint``. The flattened Robotiq USD also exposes
    inner finger joints and passive mimic joints. Isaac Lab's own Franka+Robotiq config adds those
    joints to the actuator map so a closed driver command produces a parallel grasp instead of a
    good-looking driver value with weak/no pad contact.
    """
    try:
        from isaaclab.actuators import ImplicitActuatorCfg
    except ImportError:
        log.warning("Could not import ImplicitActuatorCfg; keeping Arena DROID gripper actuators.")
        return False

    sc = getattr(embodiment, "scene_config", None)
    robot_cfg = getattr(sc, "robot", None) if sc is not None else None
    if robot_cfg is None:
        return False
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return False

    patched = dict(actuators)
    patched.pop("gripper", None)
    patched.update(
        {
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=1650.0,
                velocity_limit_sim=10.0,
                stiffness=17.0,
                damping=0.02,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=0.2,
                damping=0.001,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
                effort_limit_sim=1.0,
                velocity_limit_sim=10.0,
                stiffness=0.0,
                damping=0.0,
            ),
        }
    )
    robot_cfg.actuators = patched
    print(
        "[molmospaces_arena] Patched DROID Robotiq actuators for parallel finger-chain grasping.",
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


def _patch_ithor_scene_physics_runtime(
    n_envs: int = 1,
    deactivate_object_types: list[str] | None = None,
    dynamic_scene_object_names: list[str] | None = None,
) -> None:
    """Post-init iTHOR physics fixes: make rigid bodies kinematic, disable visual mesh collisions,
    deactivate conflicting decoration objects, and lock articulation joints."""
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema
    except ImportError:
        log.warning("omni.usd / pxr not available; skipping iTHOR scene physics patch.")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        log.warning("No USD stage; skipping iTHOR scene physics patch.")
        return

    n_kinematic = 0
    n_dynamic = 0
    n_mass_patched = 0
    n_vis_disabled = 0
    n_deactivated = 0
    n_joints_locked = 0
    dynamic_names = set(dynamic_scene_object_names or [])
    pickup_mass = float((os.environ.get("MOLMO_ARENA_PICKUP_MASS") or "0.2").strip() or "0.2")

    def _is_dynamic_pickup_prim(prim) -> bool:
        if not dynamic_names:
            return False
        path = str(prim.GetPath())
        return any(prim.GetName() == name or f"/Geometry/{name}" in path for name in dynamic_names)

    for env_idx in range(n_envs):
        scene_root = f"/World/envs/env_{env_idx}/molmospaces_scene"
        scene_prim = stage.GetPrimAtPath(scene_root)
        if not scene_prim.IsValid():
            continue
        # Usd.PrimRange sees prims added post-init (env_1..N-1 scenes); stage.Traverse() doesn't.
        for prim in Usd.PrimRange(scene_prim):
            is_dynamic_pickup = _is_dynamic_pickup_prim(prim)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                if is_dynamic_pickup:
                    rb_api.GetKinematicEnabledAttr().Set(False)
                    try:
                        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                            contact_api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPath())
                        else:
                            contact_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                        contact_api.CreateThresholdAttr().Set(0.0)
                    except Exception as e:
                        log.warning("Could not enable contact reporting on dynamic pickup prim %s: %s", prim.GetPath(), e)
                    try:
                        from pxr import Gf

                        mass_api = UsdPhysics.MassAPI(prim)
                        if not mass_api:
                            mass_api = UsdPhysics.MassAPI.Apply(prim)
                        mass_api.GetMassAttr().Set(pickup_mass)
                        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                        # A small positive inertia avoids imported collider sentinels such as -inf COM / zero inertia
                        # from destabilizing the dynamic pickup.
                        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.001, 0.001, 0.001))
                        n_mass_patched += 1
                    except Exception as e:
                        log.warning("Could not patch mass on dynamic pickup prim %s: %s", prim.GetPath(), e)
                    n_dynamic += 1
                else:
                    rb_api.GetKinematicEnabledAttr().Set(True)
                    n_kinematic += 1
            if is_dynamic_pickup and prim.HasAPI(UsdPhysics.MassAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    from pxr import Gf

                    mass_api = UsdPhysics.MassAPI(prim)
                    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    if mass_api.GetDiagonalInertiaAttr().Get() in (None, (0, 0, 0)):
                        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.001, 0.001, 0.001))
                    n_mass_patched += 1
                except Exception:
                    pass
            if "_visual_" in prim.GetName() and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                n_vis_disabled += 1
            if deactivate_object_types:
                prim_name_lower = prim.GetName().lower()
                if not is_dynamic_pickup and any(
                    prim_name_lower.startswith(obj_type) for obj_type in deactivate_object_types
                ):
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
    if dynamic_names:
        msg += f", {n_dynamic} scene pickup rigid body/bodies kept dynamic"
    if n_mass_patched:
        msg += f", {n_mass_patched} dynamic pickup mass/inertia attr(s) patched"
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
    use_joint_velocity_control: bool = False,
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
    dynamic_scene_object_names: list[str] = []
    for item in spec.objects:
        # (name, asset_id/scene_prim, pose_7, source) with source "thor", "objaverse", or "scene"
        arena_name = item[0]
        asset_id = item[1]
        pose_7_world = item[2]
        # World (episode) → robot frame: p_robot = inv(R_base) * (p_world - t_base); q_robot = inv(q_base)*q_world.
        pose_7 = list(_pose_7_world_to_robot_frame(pose_7_world, robot_base_pose_7))
        source = item[3] if len(item) >= 4 else "thor"
        if source == "scene" and not (scene_usd is not None and Path(scene_usd).is_file()):
            raise ValueError(
                f"Scene pickup '{asset_id}' requires a resolved scene USD, but spec.scene_usd_path={scene_usd!r}. "
                "Pass --scenes_root/--assets_root pointing at the Isaac-ready USD scene root "
                "(for this workspace, /home/horde/molmo-proj/assets or /home/horde/molmo-proj/assets/usd)."
            )
        thor_usd_path = get_thor_usd_path(asset_id, thor_assets_dir) if source == "thor" else None
        apply_thor_up_axis = bool(
            thor_usd_path is not None and should_apply_thor_up_axis_correction(thor_usd_path)
        )
        if source == "thor" and arena_name == spec.pickup_name:
            z_extra = float((os.environ.get("MOLMO_ARENA_PICK_Z_EXTRA") or "0").strip() or "0")
            if z_extra != 0.0:
                pose_7[2] = float(pose_7[2]) + z_extra
        # THOR object meshes keep the original local frame; apply the correction unless explicitly overridden.
        pos_xyz, rot_wxyz = _pose_7_to_arena_pose(
            pose_7,
            apply_thor_up_axis=bool(source == "thor" and apply_thor_up_axis),
        )
        pose = Pose(position_xyz=pos_xyz, rotation_wxyz=rot_wxyz)
        if source == "scene" and arena_name == spec.pickup_name:
            print(
                f"[molmospaces_arena] Pick existing scene object '{asset_id}': "
                f"world_xyz={tuple(round(float(pose_7_world[i]), 4) for i in range(3))} "
                f"arena_xyz={tuple(round(float(pos_xyz[i]), 4) for i in range(3))}",
                flush=True,
            )
        elif source == "thor" and arena_name == spec.pickup_name:
            usd_p = thor_usd_path if thor_usd_path is not None else get_thor_usd_path(asset_id, thor_assets_dir)
            print(
                f"[molmospaces_arena] Pick '{asset_id}': USD path={usd_p} exists={usd_p.is_file()} "
                f"upAxis={get_usd_up_axis(usd_p) or 'unknown'} apply_up_axis={apply_thor_up_axis} "
                f"world_xyz={tuple(round(float(pose_7_world[i]), 4) for i in range(3))} "
                f"arena_spawn_xyz={tuple(round(float(pos_xyz[i]), 4) for i in range(3))}",
                flush=True,
            )
            if not usd_p.is_file():
                raise FileNotFoundError(
                    f"THOR pickup USD missing for '{asset_id}': {usd_p}. "
                    "Pass --assets_root to your ms-download install (with --assets thor), or set MOLMO_ISAAC_ASSETS_ROOT / MOLMO_THOR_USD_DIR."
                )
        if source == "scene":
            obj = SceneRigidObjectReference(
                name=arena_name,
                scene_object_name=asset_id,
                initial_pose=pose,
            )
            dynamic_scene_object_names.append(asset_id)
        elif source == "objaverse":
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
    if str(embodiment_key).startswith("droid"):
        _remove_droid_arena_stand_for_molmospaces(embodiment)
        _apply_droid_molmospaces_mount_pose(embodiment)
        _patch_droid_molmospaces_gripper_actuators(embodiment)
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
    if use_joint_velocity_control and str(embodiment_key).startswith("droid"):
        ok_ctrl = _apply_droid_joint_velocity_control(embodiment)
        if not ok_ctrl:
            raise RuntimeError(
                "use_joint_velocity_control=True but could not patch Arena DROID "
                "action_config.arm_action to JointVelocityActionCfg."
            )
    _apply_franka_episode_init_qpos(embodiment, embodiment_key, spec)
    img_width, img_height = 624, 352
    img_resolution = getattr(spec, "img_resolution", None)
    if img_resolution and len(img_resolution) >= 2:
        img_width, img_height = int(img_resolution[0]), int(img_resolution[1])
    if enable_cameras and embodiment_key == "franka":
        _attach_franka_droid_cameras(embodiment, img_height=img_height, img_width=img_width)
    elif enable_cameras and str(embodiment_key).startswith("droid"):
        _patch_droid_molmospaces_cameras(
            embodiment,
            img_height=img_height,
            img_width=img_width,
            camera_specs=getattr(spec, "camera_specs", None),
        )
    _ev = getattr(embodiment, "event_config", None)
    if _ev is not None and hasattr(_ev, "randomize_franka_joint_state"):
        _rand = _ev.randomize_franka_joint_state
        _params = getattr(_rand, "params", None)
        if isinstance(_params, dict):
            _params["mean"] = 0.0
            _params["std"] = 0.0
            print(
                "[molmospaces_arena] Set randomize_franka_joint_state std=0 so reset writes episode qpos deterministically.",
                flush=True,
            )
        else:
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
    if scene_usd is not None and Path(scene_usd).is_file():
        _env_cfg.scene.filter_collisions = False
        try:
            _env_cfg.scene.molmospaces_scene.spawn.activate_contact_sensors = True
            print(
                "[molmospaces_arena] Enabled contact reporting on imported iTHOR scene rigid bodies.",
                flush=True,
            )
        except Exception as e:
            log.warning("Could not enable imported-scene contact reporting before env creation: %s", e)
        print(
            "[molmospaces_arena] Disabled Isaac Lab env collision filtering for the referenced iTHOR scene.",
            flush=True,
        )
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
    deactivate_object_types = [] if dynamic_scene_object_names else [pickup_base]
    _patch_ithor_scene_physics_runtime(
        n_envs=num_envs,
        deactivate_object_types=deactivate_object_types,
        dynamic_scene_object_names=dynamic_scene_object_names,
    )
    return env, env_builder
