import logging
import random
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesArticulationObject, MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import (
    ASSETS_DIR,
    OBJECT_LIBRARY_TO_GRASP_LIBRARIES,
    USER_GRASP_LIBRARIES,
)
from molmo_spaces.utils.lazy_loading_utils import locate_uid_package, get_user_grasp_library_index


log = logging.getLogger(__name__)


def get_grasp_libraries_for_object(uid: str) -> list[str]:
    package, _, _ = locate_uid_package(uid)
    if package not in OBJECT_LIBRARY_TO_GRASP_LIBRARIES:
        return []
    return list(OBJECT_LIBRARY_TO_GRASP_LIBRARIES[package])


def _filter_grasp_libraries_for_object(
    uid: str, grasp_libraries: list[str] | None = None
) -> list[str]:
    available_grasp_libraries = get_grasp_libraries_for_object(uid)
    if grasp_libraries is None:
        grasp_libraries = available_grasp_libraries
    else:
        available_grasp_libraries = set(available_grasp_libraries)
        grasp_libraries = [
            library for library in grasp_libraries if library in available_grasp_libraries
        ]

    return grasp_libraries


def get_grasp_path(uid: str, grasp_libraries: list[str] | None = None) -> Path | None:
    grasp_libraries = _filter_grasp_libraries_for_object(uid, grasp_libraries)

    for library in grasp_libraries:
        if library in USER_GRASP_LIBRARIES:
            grasp_library_dir = USER_GRASP_LIBRARIES[library]
            grasp_library_index = get_user_grasp_library_index(grasp_library_dir)
            robot_name = library.split("/", 1)[-1]
            grasp_file = grasp_library_index.grasp_paths.get(robot_name, {}).get(uid, None)
            if grasp_file is not None:
                grasp_file = grasp_library_dir / grasp_file
        else:
            grasp_file = ASSETS_DIR / f"grasps/{library}/{uid}/{uid}_grasps_filtered.npz"

        if grasp_file is not None and grasp_file.exists():
            return grasp_file

    return None


def get_joint_grasp_path(
    uid: str, joint_name: str, grasp_libraries: list[str] | None = None
) -> Path | None:
    grasp_libraries = _filter_grasp_libraries_for_object(uid, grasp_libraries)

    for library in grasp_libraries:
        if library in USER_GRASP_LIBRARIES:
            grasp_library_dir = USER_GRASP_LIBRARIES[library]
            grasp_library_index = get_user_grasp_library_index(grasp_library_dir)
            robot_name = library.split("/", 1)[-1]
            grasp_file = (
                grasp_library_index.articulated_grasp_paths.get(robot_name, {})
                .get(uid, {})
                .get(joint_name, None)
            )
            if grasp_file is not None:
                grasp_file = grasp_library_dir / grasp_file
        else:
            grasp_file = ASSETS_DIR / f"grasps/droid/{uid}/{joint_name}_grasps_filtered.npz"

        if grasp_file is not None and grasp_file.exists():
            return grasp_file

    return None


def load_object_grasps(
    uid: str, grasp_libraries: list[str] | None = None, num_grasps: int = 50
) -> np.ndarray:
    grasp_path = get_grasp_path(uid, grasp_libraries)
    if grasp_path is None:
        raise ValueError(f"No grasp file found for {uid}")

    # TODO: wire up this method (and below) into demonstrators

    npz_data = np.load(grasp_path)
    transforms: np.ndarray = npz_data["transforms"]
    if len(transforms) <= num_grasps:
        return transforms
    else:
        idxs = random.sample(range(len(transforms)), num_grasps)
        return transforms[idxs]


def load_object_joint_grasps(
    uid: str, joint_name: str, grasp_libraries: list[str] | None = None, num_grasps: int = 50
) -> np.ndarray:
    grasp_path = get_joint_grasp_path(uid, joint_name, grasp_libraries)
    if grasp_path is None:
        raise ValueError(f"No joint grasp file found for {uid}/{joint_name}")

    npz_data = np.load(grasp_path)
    transforms: np.ndarray = npz_data["transforms"]
    if len(transforms) <= num_grasps:
        return transforms
    else:
        idxs = random.sample(range(len(transforms)), num_grasps)
        return transforms[idxs]


def flip_grasps(grasps: np.ndarray) -> np.ndarray:
    flip = np.eye(4)
    flip[:3, :3] = R.from_euler("z", 180, degrees=True).as_matrix()
    return grasps @ flip


def get_pickup_grasps(env: CPUMujocoEnv, obj: MlSpacesObject) -> np.ndarray:
    scene_metadata = env.current_scene_metadata
    if scene_metadata is None:
        raise ValueError(f"Could not load grasps for object {obj.name}: No scene metadata found!")
    if obj.name not in scene_metadata["objects"]:
        raise ValueError(f"Could not load grasps for object {obj.name}: Object not found in scene metadata!")

    asset_id: str = scene_metadata["objects"][obj.name]["asset_id"]
    grasps = load_object_grasps(asset_id, num_grasps=int(1e6))
    if len(grasps) == 0:
        raise ValueError(f"No grasps found for {obj.name}")

    grasps_world = obj.pose @ grasps
    all_grasp_poses = np.concatenate([grasps_world, flip_grasps(grasps_world)])

    log.info(f"Loaded {len(all_grasp_poses)} total grasp poses (including flipped versions)")
    return all_grasp_poses


def get_joint_grasps(env: CPUMujocoEnv, obj: MlSpacesArticulationObject, joint_idx: int) -> np.ndarray:
    scene_metadata = env.current_scene_metadata
    if scene_metadata is None:
        raise ValueError(f"Could not load grasps for object {obj.name}: No scene metadata found!")
    if obj.name not in scene_metadata["objects"]:
        raise ValueError(f"Could not load grasps for object {obj.name}: Object not found in scene metadata!")

    joint_name: str = obj.joint_names[joint_idx]
    asset_joint_name = scene_metadata["objects"][obj.name]["name_map"]["joints"][joint_name]
    asset_id = scene_metadata["objects"][obj.name]["asset_id"]

    grasps = load_object_joint_grasps(asset_id, asset_joint_name, num_grasps=int(1e6))
    if len(grasps) == 0:
        raise ValueError(f"No grasps found for {obj.name}/{joint_name}")

    joint_bodyid = env.current_model.joint(joint_name).bodyid.item()
    joint_body_pose = np.eye(4)
    joint_body_pose[:3, 3] = env.current_data.xpos[joint_bodyid]
    joint_body_pose[:3, :3] = env.current_data.xmat[joint_bodyid].reshape(3, 3)

    grasps_world = joint_body_pose @ grasps
    all_grasp_poses = np.concatenate([grasps_world, flip_grasps(grasps_world)])
    log.info(f"Loaded {len(all_grasp_poses)} total grasp poses (including flipped versions)")
    return all_grasp_poses
