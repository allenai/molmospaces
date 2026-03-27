import functools
import glob
import json
import logging
import random
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesObject, create_mlspaces_body
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.robot_views.abstract import RobotView
from molmo_spaces.utils.asset_names import get_thor_name
from molmo_spaces.utils.pose import pos_quat_to_pose_mat
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)


@functools.lru_cache(
    maxsize=10000
)  # should correspond to ~1.5MB - increase if necessary, TODO(rose) adjust this later when tuning throughput
def has_grasp_folder(object_name: str) -> bool:
    """Check if grasp files exist for the given object (for pick tasks).

    Results are cached to avoid repeated filesystem operations.

    NOTE: This only checks file existence. Use has_valid_grasp_file() for a more
    thorough check that validates the file actually contains transforms.
    """
    grasp_files = [
        ASSETS_DIR / f"grasps/droid/{object_name}/{object_name}_grasps_filtered.npz",
        ASSETS_DIR / f"grasps/droid/{object_name}/{object_name}_joint_grasps_filtered.npz",
        ASSETS_DIR / f"grasps/droid_objaverse/{object_name}/{object_name}_grasps_filtered.npz",
        ASSETS_DIR / f"grasps/rum/{object_name}/{object_name}_grasps_filtered.json",
    ]

    return any(grasp_file.exists() for grasp_file in grasp_files)


@functools.lru_cache(maxsize=10000)
def has_valid_grasp_file(object_name: str, min_grasps: int = 1) -> bool:
    """Check if valid grasp files exist for the given object with actual transforms.

    This is more expensive than has_grasp_folder() as it loads and validates the file,
    but ensures load_grasps_for_object() will succeed.

    Args:
        object_name: The object identifier (asset_id or thor name)
        min_grasps: Minimum number of transforms required (default 1)

    Returns:
        True if a valid grasp file with sufficient transforms exists
    """
    grasp_files = [
        ASSETS_DIR / f"grasps/droid/{object_name}/{object_name}_grasps_filtered.npz",
        ASSETS_DIR / f"grasps/droid_objaverse/{object_name}/{object_name}_grasps_filtered.npz",
        ASSETS_DIR / f"grasps/rum/{object_name}/{object_name}_grasps_filtered.json",
    ]

    for grasp_file in grasp_files:
        if not grasp_file.exists():
            continue
        try:
            grasp_file_str = grasp_file.as_posix()
            if grasp_file_str.endswith(".npz"):
                npz_data = np.load(grasp_file)
                transforms = npz_data.get("transforms", [])
            elif grasp_file_str.endswith(".json"):
                with open(grasp_file, "r") as f:
                    json_data = json.load(f)
                    transforms = json_data.get("transforms", [])
            else:
                continue
            if len(transforms) >= min_grasps:
                return True
        except Exception:
            continue
    return False


@functools.lru_cache(maxsize=10000)  # should correspond to ~1.5MB - increase if necessary
def has_joint_grasp_file(object_name: str, joint_name: str) -> bool:
    """Check if joint grasp files exist for the given object and joint."""
    grasp_files = [
        ASSETS_DIR / f"grasps/droid/{object_name}/{joint_name}_grasps_filtered.npz",
        ASSETS_DIR / f"grasps/rum/{object_name}/{joint_name}_grasps_filtered.json",
    ]

    for grasp_file in grasp_files:
        grasp_file_str = grasp_file.as_posix()
        if grasp_file_str.endswith(".npz"):
            if not grasp_file.exists():
                return False
            npz_data = np.load(grasp_file)
            transforms = npz_data.get("transforms", [])
        elif grasp_file_str.endswith(".json"):
            if not grasp_file.exists():
                return False
            with open(grasp_file, "r") as f:
                json_data = json.load(f)
                transforms = json_data.get("transforms", [])
        else:
            raise ValueError(f"Invalid grasp file: {grasp_file}")
        if (
            len(transforms) > 100
        ):  # at least 100 grasps. NOTE(yejin):Attempt to filter out inside drawers like in Fridges
            return True
    return False


def load_grasps_for_object(object_name, num_grasps=50):
    """Load grasps for a specific object."""

    grasp_files = {
        ASSETS_DIR / f"grasps/droid/{object_name}/{object_name}_grasps_filtered.npz": "droid",
        ASSETS_DIR
        / f"grasps/droid_objaverse/{object_name}/{object_name}_grasps_filtered.npz": "droid",
        ASSETS_DIR / f"grasps/rum/{object_name}/{object_name}_grasps_filtered.json": "rum",
    }

    combined_transforms = []
    for filename, _gripper in grasp_files.items():
        try:
            filename_str = filename.as_posix()
            if filename_str.endswith(".npz"):
                npz_data = np.load(filename)
                transforms = npz_data.get("transforms", [])
            elif filename_str.endswith(".json"):
                with open(filename, "r") as f:
                    json_data = json.load(f)
                    transforms = json_data.get("transforms", [])
        except FileNotFoundError:
            continue

        if len(transforms) > 0:
            combined_transforms.extend(transforms)
            log.info(f"Loading grasps from: {filename} got {len(transforms)}")
            break

    if len(combined_transforms) == 0:
        log.info(f"No grasp transformations found for {object_name}")
        raise ValueError(f"Failed to find grasp file for: {object_name}")

    grasps = []
    if len(combined_transforms) <= num_grasps:
        selected_grasps = combined_transforms
    else:
        selected_grasps = random.sample(combined_transforms, num_grasps)

    selected_grasps = np.array(selected_grasps)
    grasps = selected_grasps
    return _gripper, grasps


def load_grasps_for_object_per_joint(
    object_name,
    joint_name,
    num_grasps=50,
    grasp_dir=Path(ASSETS_DIR) / "grasps",
    gripper="droid",
):
    """Load grasps for a specific object."""
    combined_transforms = []

    # "droid" or "rum" folder(s)
    source_to_pattern = dict(
        droid=grasp_dir.as_posix() + "/droid*/" + object_name + "/" + f"{joint_name}*_filtered.npz",
        rum=grasp_dir.as_posix() + "/rum/" + object_name + "/" + f"{joint_name}*_filtered.json",
        any_npz=grasp_dir.as_posix() + "/*/" + object_name + "/" + f"{joint_name}*_filtered.npz",
    )

    pattern = str(Path(source_to_pattern[gripper]).resolve())
    grasp_files = glob.glob(pattern)
    assert len(grasp_files) <= 1, f"Expected up to 1 grasp file, got {len(grasp_files)}"
    for filename in grasp_files:
        log.info(f"Loading grasps from: {filename}")
        try:
            if filename.endswith(".json"):
                with open(filename, "r") as f:
                    json_data = json.load(f)
                    transforms = json_data.get("transforms", [])
            elif filename.endswith(".npz"):
                npz_data = np.load(filename)
                transforms = npz_data.get("transforms", [])
            combined_transforms.extend(transforms)
        except Exception as e:
            log.error(f"Error: Failed to load grasp file for: {filename} : {e}")

    if not combined_transforms:
        log.warning(f"No grasp transformations found for {object_name}")
        return None, []

    grasps = np.array(combined_transforms)
    gripper = filename.split("/")[-3]  # (path) / rum / (object_name) / (joint name.json) -> rum
    return gripper, grasps  # , joint_info["parent_body"]


def get_grasp_collision_body_name(grasp_idx: int) -> str:
    return f"grasp_collision_{grasp_idx}"


def add_grasp_collision_bodies(
    spec: mujoco.MjSpec,
    num_grasps: int,
    grasp_width: float,
    grasp_length: float,
    grasp_height: float,
    grasp_base_pos: np.ndarray,
):
    """Add grasp collision bodies to the scene."""
    for i in range(num_grasps):
        # init grasp bodies in the sky (below the ground causes collision with the floor)
        grasp_body = spec.worldbody.add_body(
            name=get_grasp_collision_body_name(i),
            pos=[0, 0, 10],
            gravcomp=1.0,
        )
        grasp_body.add_freejoint()

        geom_kwargs = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=[0, 0, 1, 1],
            group=3,
            contype=0,
            conaffinity=0b1111,
        )

        base_geom = grasp_body.add_geom(**geom_kwargs)
        base_geom.size[0] = grasp_height / 2
        base_geom.fromto[:3] = np.array([0, -grasp_width / 2, 0]) + grasp_base_pos
        base_geom.fromto[3:] = np.array([0, grasp_width / 2, 0]) + grasp_base_pos

        finger1_geom = grasp_body.add_geom(**geom_kwargs)
        finger1_geom.size[0] = grasp_height / 2
        finger1_geom.fromto[:3] = np.array([0, -grasp_width / 2, 0]) + grasp_base_pos
        finger1_geom.fromto[3:] = np.array([0, -grasp_width / 2, grasp_length]) + grasp_base_pos

        finger2_geom = grasp_body.add_geom(**geom_kwargs)
        finger2_geom.size[0] = grasp_height / 2
        finger2_geom.fromto[:3] = np.array([0, grasp_width / 2, 0]) + grasp_base_pos
        finger2_geom.fromto[3:] = np.array([0, grasp_width / 2, grasp_length]) + grasp_base_pos


def get_noncolliding_grasp_mask(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    grasp_poses_world: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    n_grasps = len(grasp_poses_world)
    grasp_bodies = [
        create_mlspaces_body(mj_data, get_grasp_collision_body_name(i)) for i in range(batch_size)
    ]
    start_poses = [body.pose.copy() for body in grasp_bodies]
    grasp_body_ids = set(body.body_id for body in grasp_bodies)

    try:
        colliding_grasp_mask = np.zeros(n_grasps, dtype=bool)
        for i in range(0, n_grasps, batch_size):
            grasp_bid_to_idx = {}
            n_grasps_in_batch = min(batch_size, n_grasps - i)
            for j in range(i, i + n_grasps_in_batch):
                grasp_body = grasp_bodies[j - i]
                grasp_body.pose = grasp_poses_world[j]
                grasp_bid_to_idx[grasp_body.body_id] = j
            for j in range(n_grasps_in_batch, len(grasp_bodies)):
                grasp_bodies[j].pose = start_poses[j]

            mujoco.mj_kinematics(mj_model, mj_data)
            mujoco.mj_collision(mj_model, mj_data)
            for contact in mj_data.contact:
                bid1 = mj_model.geom_bodyid[contact.geom1]
                bid2 = mj_model.geom_bodyid[contact.geom2]
                if bid1 in grasp_body_ids or bid2 in grasp_body_ids:
                    grasp_bid = bid1 if bid1 in grasp_body_ids else bid2
                    other_bid = bid2 if grasp_bid == bid1 else bid1
                    assert other_bid not in grasp_body_ids

                    grasp_idx = grasp_bid_to_idx[grasp_bid]
                    colliding_grasp_mask[grasp_idx] = True

        return ~colliding_grasp_mask
    finally:
        # move the grasp bodies back out of the way
        for body, pose in zip(grasp_bodies, start_poses):
            body.pose = pose
        mujoco.mj_fwdPosition(mj_model, mj_data)


def get_all_grasp_poses(
    policy,
    pickup_obj: MlSpacesObject,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Get all available grasp poses for the pickup object without sampling or filtering.

    Args:
        policy: The policy object containing task and environment information
        pickup_obj: The object to grasp

    Returns:
        tuple: (all_grasp_poses, gripper_type, object_pose) where:
            - all_grasp_poses: Array of all grasp poses in world frame with shape (N, 4, 4)
            - gripper_type: String indicating gripper type ("rum" or "droid")
            - object_pose: 4x4 transformation matrix of the object pose
    """
    model = policy.task.env.current_model
    data = policy.task.env.current_data
    scene_metadata = policy.task.env.current_scene_metadata

    # Load grasps based on task type
    if policy.config.task_type in [
        "pick",
        "pick_and_place",
        "pick_and_place_next_to",
        "pick_and_place_color",
    ]:
        thor_dict = scene_metadata["objects"].get(pickup_obj.name, None) if scene_metadata else None
        thor_name = thor_dict["asset_id"] if thor_dict else get_thor_name(model, pickup_obj)

        gripper, cached_grasps = load_grasps_for_object(thor_name, num_grasps=int(1e6))
        if len(cached_grasps) == 0:
            raise ValueError(f"No grasps found for {thor_name}")

        object_pose = pos_quat_to_pose_mat(pickup_obj.position, pickup_obj.quat)

    elif policy.config.task_type in ["open", "close"]:
        _joint_name = pickup_obj.joint_names[policy.config.task_config.joint_index]
        joint_name = scene_metadata["objects"][pickup_obj.name]["name_map"]["joints"][_joint_name]
        category_base_name = scene_metadata["objects"][pickup_obj.name]["asset_id"]

        gripper, cached_grasps = load_grasps_for_object_per_joint(
            category_base_name, joint_name, num_grasps=int(1e6)
        )
        if len(cached_grasps) == 0:
            raise ValueError(f"No grasps found for {pickup_obj.name}")

        # Get joint body pose
        joint_body_id = model.joint(_joint_name).bodyid[0]
        object_position = data.xpos[joint_body_id]
        object_quat = data.xquat[joint_body_id]
        object_pose = pos_quat_to_pose_mat(object_position, object_quat)
    else:
        raise ValueError(f"Invalid task type {policy.config.task_type}")

    # Transform grasps to gripper TCP
    if gripper == "rum":
        ROT_Z_90 = pos_quat_to_pose_mat(
            [0, 0, 0], R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
        )
        RUM_BASE_TCP = pos_quat_to_pose_mat(np.array([0.0, 0, 0.12]), [1, 0, 0, 0])
        GRIP_BASE_TCP = RUM_BASE_TCP @ ROT_Z_90
    elif gripper == "droid":
        GRIP_BASE_TCP = np.eye(4)

    # Convert all cached grasps to world frame
    grasp_poses_world = object_pose @ cached_grasps @ GRIP_BASE_TCP

    # Add flipped grasp poses (symmetric around 180 degree rotation around Z)
    flipped_grasp_poses_world = grasp_poses_world.copy()
    flipped_grasp_poses_world[..., :3, :3] = (
        flipped_grasp_poses_world[..., :3, :3]
        @ R.from_euler("z", 180, degrees=True).as_matrix()[None]
    )

    # Concatenate original and flipped grasps
    all_grasp_poses = np.concatenate([grasp_poses_world, flipped_grasp_poses_world])

    log.info(f"Loaded {len(all_grasp_poses)} total grasp poses (including flipped versions)")

    return all_grasp_poses, gripper, object_pose


def compute_grasp_pose(
    policy,
    pickup_obj: MlSpacesObject,
    robot_view: RobotView,
    check_collision: bool,
    n_collision_checks: int,
    collision_batch_size: int,
    check_ik: bool,
    n_ik_checks: int,
    ik_batch_size: int,
    pos_cost_weight: float = 1.0,
    rot_cost_weight: float = 0.01,
    vertical_cost_weight: float = 2.0,
    horizontal_cost_weight: float = 0,
    com_dist_cost_weight: float = 8.0,
) -> np.ndarray:
    model = policy.task.env.current_model
    data = policy.task.env.current_data

    # Get all grasp poses
    grasp_poses_world, gripper, object_pose = get_all_grasp_poses(policy, pickup_obj)

    # Get gripper TCP transform for visualization
    if gripper == "rum":
        ROT_Z_90 = pos_quat_to_pose_mat(
            [0, 0, 0], R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
        )
        RUM_BASE_TCP = pos_quat_to_pose_mat(np.array([0.0, 0, 0.12]), [1, 0, 0, 0])
        GRIP_BASE_TCP = RUM_BASE_TCP @ ROT_Z_90
    elif gripper == "droid":
        GRIP_BASE_TCP = np.eye(4)

    # get the current TCP position
    tcp_pose_arr = policy.task.sensor_suite.sensors["tcp_pose"].get_observation(
        policy.task._env, policy.task
    )
    tcp_pose = pos_quat_to_pose_mat(tcp_pose_arr[0:3], tcp_pose_arr[3:7])
    tcp_pose_world = policy.task._env.current_robot.robot_view.base.pose @ tcp_pose
    tcp_pose_inv = np.linalg.inv(tcp_pose_world)

    dist_tcp = tcp_pose_inv @ grasp_poses_world  # shape (N,4,4)
    dists_tcp_p = np.linalg.norm(dist_tcp[:, :3, 3], axis=1)
    dist_tcp_o = R.from_matrix(dist_tcp[:, :3, :3]).magnitude() * 180 / np.pi

    dists_up = grasp_poses_world[:, 2, 2]  # range = [-1, 1]

    dists_com = np.linalg.norm((np.linalg.inv(object_pose) @ grasp_poses_world)[:, :3, 3], axis=1)

    # Cost for horizontal orientation: 0 = perfectly horizontal (z-axis parallel to XY plane), 1 = vertical
    # Lower cost = more horizontal, so we want to minimize this
    # Use squared term to more strongly penalize vertical orientations
    dists_xy_parallel = np.abs(dists_up) ** 2

    dist_total = (
        pos_cost_weight * dists_tcp_p
        + rot_cost_weight * dist_tcp_o
        + vertical_cost_weight * dists_up
        + horizontal_cost_weight * dists_xy_parallel
        + com_dist_cost_weight * dists_com
    )
    close_grasp_ids = np.argsort(dist_total, kind="stable")  # weight positions and orientations
    close_grasp_ids = close_grasp_ids[:n_collision_checks]

    if check_collision:
        with Timer() as collision_check_time:
            noncolliding_grasp_mask = get_noncolliding_grasp_mask(
                model,
                data,
                grasp_poses_world[close_grasp_ids],
                collision_batch_size,
            )

        log.info(
            f"Collision-checked {len(close_grasp_ids)} grasps in {collision_check_time.value:.3f}s, found {np.sum(noncolliding_grasp_mask)} non-colliding grasps"
        )
    else:
        noncolliding_grasp_mask = np.ones(len(close_grasp_ids), dtype=bool)

    noncolliding_close_grasp_ids = close_grasp_ids[noncolliding_grasp_mask]
    colliding_close_grasp_ids = close_grasp_ids[~noncolliding_grasp_mask]

    # check the feasibility of the grasp pose
    if check_ik:
        found_feasible = False
        grasp_idx = None
        grasp_pose_world = None
        n_checks_done = 0

        if noncolliding_grasp_mask.any():
            with Timer() as ik_check_time:
                for i in range(0, n_ik_checks, ik_batch_size):
                    grasps = grasp_poses_world[noncolliding_close_grasp_ids[i : i + ik_batch_size]]
                    n_checks_done += len(grasps)
                    feasible_mask = policy.check_feasible_ik(grasps)
                    if np.any(feasible_mask).item():
                        found_feasible = True
                        feasible_grasps = grasps[feasible_mask]
                        for idx in range(len(feasible_grasps)):
                            if policy.check_feasible_ik(
                                feasible_grasps[idx]
                            ):  # NOTE(yejin): for some reason, 0th index sometime returns false
                                grasp_pose_world = feasible_grasps[idx]
                                grasp_idx = noncolliding_close_grasp_ids[i : i + ik_batch_size][
                                    feasible_mask
                                ][idx]
                                break

            log.info(
                f"Feasibility-checked {n_checks_done} grasps in {ik_check_time.value:.3f}s, found feasible grasp: {found_feasible}"
            )
    else:
        found_feasible = True
        grasp_idx = noncolliding_close_grasp_ids[0]
        grasp_pose_world = grasp_poses_world[grasp_idx]

    # debug by adding tcp locations to interactive viewer
    view_poses = False
    if view_poses:
        view_noncolliding = True
        view_colliding = False
        policy._show_poses(np.array([tcp_pose]), style="tcp")  # red
        if view_noncolliding and len(noncolliding_close_grasp_ids) > 0:
            policy._show_poses(
                grasp_poses_world[noncolliding_close_grasp_ids[:3]],
                style="tcp",
                color=(0, 1, 0, 1),  # green
            )
        if view_colliding and len(colliding_close_grasp_ids) > 0:
            policy._show_poses(
                grasp_poses_world[colliding_close_grasp_ids[:10]],
                style="tcp",
                color=(1, 0, 0, 1),  # red
            )
        if gripper == "rum" and grasp_pose_world is not None:
            policy._show_poses(
                np.array([grasp_pose_world @ np.linalg.inv(GRIP_BASE_TCP)]),
                style="RUM",
                color=(0, 0, 1, 1),
            )  # blue
        if policy.task.viewer:
            policy.task.viewer.sync()

    if not found_feasible:
        raise ValueError("No feasible grasp found")

    original_grasp_idx = grasp_idx % (len(grasp_poses_world) // 2)
    log.info(
        f"Feasible grasp found {grasp_idx} (originally {original_grasp_idx}): w/ {dists_tcp_p[grasp_idx]:.3f}[m] {dist_tcp_o[grasp_idx]:.3f}[deg]"
    )

    log.debug("\n[POSE DEBUG] Computing target poses:")
    log.debug(f"  - Robot base position: {robot_view.base.pose[:3, 3]}")
    log.debug(f"  - TCP position: {tcp_pose_arr}")
    log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")

    return grasp_pose_world
