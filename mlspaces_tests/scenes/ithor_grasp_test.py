"""
A script to test if the assets are pickable.

This script implements a complete grasping test pipeline:
[x] 1. Load grasps for the graspable objects
[x] 2. Sample one grasp
[x] 3. Place the gripper at the grasp pose
[x] 4. Pick up the object
[x] 5. Check if the object is picked up
[ ] 6. Drop the object
[x] 7. Repeat 2-6 for multiple objects

[X] 8. Repeaet 1-7 for multiple scenes
[x] 9. multi processing for faster testing

[ ] 10. Teset the scene with ProcTHOR house
"""

import traceback
import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

# Configure logging format but don't set level yet
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")

import datetime
import glob
import json
import random
import re

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add the parent directory to the path to import from molmo_spaces
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molmo_spaces.data_generation.recorder import RGBRecorder
from molmo_spaces.editor.constants import ALL_PICKUP_TYPES_THOR
from molmo_spaces.editor.thor_model_editor import ThorMjModelEditor
from molmo_spaces.env.arena.arena_utils import load_env_with_objects
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

MAX_GRASPS_RETRIES = 3  # 1 #3 #10
ROBOT_TYPE = "RUM"  #
PICK_HEIGHT_THRESHOLD = 0.1  # 0.01 m


DATE_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

ITHOR = False  # False
ProcTHOR = True


def extract_base_object_names(model, metadata: dict):
    """Extract base object names from the model."""
    object_body_map = {}
    base_names = set()
    object_pattern = re.compile(r"^([A-Za-z]+_\d+)(_.*)?$")

    objects = metadata.get("objects", [])
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        object_dict = objects.get(body_name, None)
        if object_dict:
            asset_id = object_dict.get("asset_id", None)
            object_body_map[asset_id] = body_name
            base_names.add(asset_id)
        """
        if body_name:
            if ITHOR:
                match = object_pattern.match(body_name)
                if match:
                    base_name = match.group(1)
                    base_names.add(base_name)
                    object_body_map[base_name] = body_name
            if ProcTHOR:
                name_end = body_name.split("|")[-1]
                match = re.search(r"^\d+_([A-Za-z_]+[\d_]+)", name_end)
                if match:
                    base_name = match.group(1).strip("_")
                    base_names.add(base_name)
                    object_body_map[base_name] = body_name
         """
    return list(base_names), object_body_map


def find_objects_with_grasps(
    base_object_names,
    grasp_dir="assets/grasps",
):
    """Find objects that have grasp data available."""
    objects_with_grasps = []

    for obj_name in base_object_names:
        pattern = f"{grasp_dir}/*/{obj_name}/{obj_name}_*filtered.npz"
        grasp_files = glob.glob(pattern)
        if grasp_files:
            objects_with_grasps.append(obj_name)
            continue

    return objects_with_grasps


def load_grasps_for_object(object_name, num_grasps=50, grasp_dir="assets/grasps"):
    """Load grasps for a specific object."""
    combined_transforms = []
    pattern = f"{grasp_dir}/*/{object_name}/{object_name}_*filtered.npz"
    # pattern = f"{grasp_dir}/{object_name}/{object_name}_*filtered.json"
    grasp_files = glob.glob(pattern)

    if not grasp_files:
        base_name = object_name.split("_")[0]
        object_number = object_name.split("_")[1]
        pattern = f"{grasp_dir}/{object_name}/{base_name}_{object_number}_*filtered.npz"
        # pattern = f"{grasp_dir}/{object_name}/{base_name}_{object_number}_*filtered.json"
        grasp_files = glob.glob(pattern)

    for filename in grasp_files:
        print(f"Loading grasps from: {filename}")
        try:
            with open(filename, "r") as f:
                npz_data = np.load(filename)
                transforms = npz_data.get("transforms", [])
                # transforms = json.load(f)
                # transforms = transforms.get("transforms", [])
                combined_transforms.extend(transforms)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not combined_transforms:
        print(f"No grasp transformations found for {object_name}")
        return []

    grasps = []
    if len(combined_transforms) <= num_grasps:
        selected_transforms = combined_transforms
    else:
        selected_transforms = random.sample(combined_transforms, num_grasps)

    for transform_data in selected_transforms:
        transform = np.array(transform_data)
        pos = transform[:3, 3]
        quat = R.from_matrix(transform[:3, :3]).as_quat(scalar_first=True)
        grasps.append((pos, quat))

    print(len(grasps))
    return grasps


def find_matching_body_names(model, base_name):
    """Find body names that match the base object name."""
    matching_bodies = []
    pattern = re.compile(f"^{base_name}(_|$)")

    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and pattern.match(body_name):
            matching_bodies.append(body_name)

    return matching_bodies


class GraspTestEnvironment:
    """Manages the grasping test environment and operations."""

    def __init__(self, scene_path: str, grasps_per_object: int = 50) -> None:
        self.scene_path = scene_path
        self.grasps_per_object = grasps_per_object
        self.model = None
        self.data = None
        self.object_grasps = {}
        self.object_body_map = {}
        self.grasp_head_ids = []
        self.viewer = None

        self.linear_pregrasp_to_grasp = []

        self.failed_grasp_objects = []
        self.failed_pick_objects = []
        self.success_grasp_objects = []
        self.success_pick_objects = []
        self.place_try = -1  # starting index for place try

        # Setup logging - level is controlled by root logger
        self.logger = logging.getLogger(__name__)
        self.recorders = []

    def setup_recorders(self, save_images: bool = False) -> None:
        self.renderer = MjOpenGLRenderer(model=self.model, device_id=None)  # device_id)
        self.recorders = [
            # RGBRecorder(
            #    period_ms=100,
            #    camera_name="robot_0/follower",  # "robot_0/egocentric", # rum specific
            #    renderer=self.renderer,
            #    save_video=True,
            #    save_images=save_images,
            # ),
            # RGBRecorder(
            #    period_ms=100,
            #    camera_name="robot_0/egocentric",  # "robot_0/egocentric", # rum specific
            #    renderer=self.renderer,
            #    save_video=True,
            #    save_images=save_images,
            # ),
            RGBRecorder(
                period_ms=100,
                camera_name="robot_0/wrist_camera",  # "robot_0/egocentric", # rum specific
                renderer=self.renderer,
                save_video=True,
                save_images=save_images,
            ),
        ]

        self.save_dir = f"debug/grasp_test_videos/{DATE_TIME}/{os.path.basename(self.scene_path)}"

    def plot_3d_frame(self, ax, position, quaternion, label, color, scale=0.1) -> None:
        """
        Plot a 3D coordinate frame at the given position and orientation.

        Args:
            ax: matplotlib 3D axis
            position: 3D position [x, y, z]
            quaternion: quaternion [w, x, y, z]
            label: label for the frame
            color: color for the frame
            scale: scale factor for the frame size
        """
        # Convert quaternion to rotation matrix
        # Note: quaternion format is [w, x, y, z]
        rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])

        # Define the coordinate frame vectors
        origin = np.array(position)
        x_axis = rotation.apply(np.array([scale, 0, 0]))
        y_axis = rotation.apply(np.array([0, scale, 0]))
        z_axis = rotation.apply(np.array([0, 0, scale]))

        # Plot the coordinate frame
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="red",
            alpha=0.8,
            length=scale,
            arrow_length_ratio=0.3,
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="green",
            alpha=0.8,
            length=scale,
            arrow_length_ratio=0.3,
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            alpha=0.8,
            length=scale,
            arrow_length_ratio=0.3,
        )

        # Add a small sphere at the origin
        ax.scatter(origin[0], origin[1], origin[2], c=color, s=100, alpha=0.8)

        # Add text label
        ax.text(
            origin[0] + scale * 0.1,
            origin[1] + scale * 0.1,
            origin[2] + scale * 0.1,
            label,
            fontsize=10,
            color=color,
            weight="bold",
        )

    def load_scene(self) -> bool:
        """Load the scene and initialize the environment."""
        try:
            self.logger.info(f"Loading scene from: {self.scene_path}")

            # Check if scene file exists
            if not os.path.exists(self.scene_path):
                self.logger.error(f"Scene file not found: {self.scene_path}")
                return False

            # Load the model and data
            self.model, _ = load_env_with_objects(self.scene_path)
            self.data = mujoco.MjData(self.model)

            self.logger.info(f"Successfully loaded scene with {self.model.nbody} bodies")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load scene: {e}")
            return False

    def load_object_grasps(self, metadata: dict) -> bool:
        """Load grasps for all available objects."""
        try:
            base_object_names, object_body_map = extract_base_object_names(self.model, metadata)
            objects_with_grasps = find_objects_with_grasps(base_object_names)

            # Map objects to body names
            for obj_base_name in objects_with_grasps:
                self.object_body_map[obj_base_name] = object_body_map[obj_base_name]
            #    matching_bodies = find_matching_body_names(self.model, obj_base_name)
            #    if matching_bodies:
            #        self.object_body_map[obj_base_name] = matching_bodies[0]
            #        self.logger.info(f"Mapped {obj_base_name} to {matching_bodies[0]}")
            #    else:
            #        self.logger.warning(f"No matching body found for {obj_base_name}")

            # Load grasps for each object
            available_objects = []
            for obj_base_name, actual_body_name in self.object_body_map.items():
                grasps = load_grasps_for_object(obj_base_name, self.grasps_per_object)
                if grasps:
                    self.object_grasps[obj_base_name] = grasps
                    available_objects.append(obj_base_name)
                    self.logger.info(
                        f"Loaded {len(grasps)} grasps for {obj_base_name} (body: {actual_body_name})"
                    )

            if not available_objects:
                self.logger.warning("No objects with grasps found")
                return False

            self.logger.info(f"Successfully loaded grasps for {len(available_objects)} objects")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load object grasps: {e}")
            return False

    def start_viewer(self) -> bool:
        """Start the MuJoCo passive viewer."""
        try:
            self.logger.info("Starting MuJoCo passive viewer...")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.logger.info("Viewer started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start viewer: {e}")
            return False

    def stop_viewer(self) -> None:
        """Stop the MuJoCo viewer."""
        if self.viewer:
            try:
                self.viewer.close()
                self.logger.info("Viewer closed")
            except Exception as e:
                self.logger.error(f"Error closing viewer: {e}")
            finally:
                self.viewer = None

    def sample_grasp(
        self, object_name: str, index: int = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample a random grasp for the specified object."""
        if object_name not in self.object_grasps:
            self.logger.warning(f"No grasps available for {object_name}")
            return None

        grasps = self.object_grasps[object_name]
        if not grasps:
            return None

        if index is None:
            # Sample a random grasp
            grasp_idx = np.random.randint(0, len(grasps))
        else:
            if index >= len(grasps):
                # self.logger.warning(f"Index {index} is greater than the number of grasps for {object_name}")
                return None

            grasp_idx = index
            # if index == 0:
            ##    # reorder grasps randomly
            #    random.shuffle(grasps)

        selected_grasp = grasps[grasp_idx]

        self.logger.info(f"Randomly selected grasp {grasp_idx + 1}/{len(grasps)} for {object_name}")

        return selected_grasp

    def tcp_to_base_grasp_pose(
        self, tcp_grasp_pose: tuple[np.ndarray, np.ndarray], gripper_length: float = 0.13
    ):
        # tcp is gripper tip center
        # base is robot base

        pos, quat = tcp_grasp_pose
        transform = np.eye(4)
        transform[:3, 3] = pos
        transform[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()

        transform = transform @ np.eye(4)
        transform[:3, 3] = transform[:3, 3] - gripper_length * transform[:3, 2]

        base_pos, base_rot = transform[:3, 3], transform[:3, :3]

        apply_z_rotation = True
        if apply_z_rotation:
            transform = R.from_matrix(base_rot) * R.from_euler("z", 90, degrees=True)
            base_quat = transform.as_quat(scalar_first=True)
        else:
            base_quat = R.from_matrix(base_rot).as_quat(scalar_first=True)

        return base_pos, base_quat

    def orient_grasp_pose_based_on_object(
        self, object_name: str, grasp_pose: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Orient the grasp pose based on the object."""
        if object_name not in self.object_body_map:
            self.logger.error(f"Object {object_name} not found in body map")
            return None, None

        obj_body_name = self.object_body_map[object_name]
        pos, quat = grasp_pose

        # Get object's current position and orientation
        object_pos = self.data.body(obj_body_name).xpos.copy()
        object_quat = self.data.body(obj_body_name).xquat.copy()

        # Convert object quaternion to rotation matrix
        object_rotation = R.from_quat(
            [object_quat[1], object_quat[2], object_quat[3], object_quat[0]]
        )

        # Calculate grasp position in world coordinates
        grasp_pos_world = object_pos + object_rotation.apply(pos)

        # Calculate grasp rotation in world coordinates
        grasp_rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        grasp_rot_world = object_rotation * grasp_rotation
        grasp_quat_world = grasp_rot_world.as_quat(scalar_first=True)

        return grasp_pos_world, grasp_quat_world

    def get_pregrasp_pose(
        self, object_name: str, grasp_pose: tuple[np.ndarray, np.ndarray], distance=0.03
    ):
        starting_quat = grasp_pose[1]
        rotation = R.from_quat(
            [starting_quat[1], starting_quat[2], starting_quat[3], starting_quat[0]]
        )
        starting_position = grasp_pose[0] + rotation.apply(np.array([0, 0, -distance]))

        plot = False
        if plot:
            # plot grasp pose and pregrasp pose with 3D coordinate frames
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot grasp pose frame
            self.plot_3d_frame(ax, grasp_pose[0], grasp_pose[1], "grasp", "red", 0.15)

            # Plot pregrasp pose frame
            self.plot_3d_frame(ax, starting_position, starting_quat, "pregrasp", "blue", 0.15)

            # Draw a line connecting grasp and pregrasp poses
            ax.plot(
                [grasp_pose[0][0], starting_position[0]],
                [grasp_pose[0][1], starting_position[1]],
                [grasp_pose[0][2], starting_position[2]],
                "k--",
                alpha=0.5,
                linewidth=1,
            )

            # Add distance annotation
            distance_text = f"Distance: {distance:.3f}m"
            mid_point = (grasp_pose[0] + starting_position) / 2
            ax.text(
                mid_point[0],
                mid_point[1],
                mid_point[2],
                distance_text,
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

            # Plot trajectory path if available
            if hasattr(self, "linear_pregrasp_to_grasp") and self.linear_pregrasp_to_grasp:
                trajectory_positions = [pose[0] for pose in self.linear_pregrasp_to_grasp]
                if trajectory_positions:
                    trajectory_positions = np.array(trajectory_positions)
                    ax.plot(
                        trajectory_positions[:, 0],
                        trajectory_positions[:, 1],
                        trajectory_positions[:, 2],
                        "orange",
                        alpha=0.6,
                        linewidth=2,
                        label="Trajectory",
                    )

            # Add labels and title
            ax.set_xlabel("X (m)", fontsize=12)
            ax.set_ylabel("Y (m)", fontsize=12)
            ax.set_zlabel("Z (m)", fontsize=12)
            ax.set_title(
                "Grasp and Pregrasp Poses with 3D Coordinate Frames", fontsize=14, fontweight="bold"
            )

            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])

            # Add grid
            ax.grid(True, alpha=0.3)

            # Set axis limits with some padding
            all_positions = np.vstack([grasp_pose[0], starting_position])
            min_vals = all_positions.min(axis=0) - 0.1
            max_vals = all_positions.max(axis=0) + 0.1
            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
            ax.set_zlim(min_vals[2], max_vals[2])

            # Add legend
            from matplotlib.patches import FancyArrowPatch

            # Create custom legend
            red_patch = FancyArrowPatch((0, 0), (0, 0), color="red", alpha=0.8, linewidth=2)
            green_patch = FancyArrowPatch((0, 0), (0, 0), color="green", alpha=0.8, linewidth=2)
            blue_patch = FancyArrowPatch((0, 0), (0, 0), color="blue", alpha=0.8, linewidth=2)

            ax.legend(
                [red_patch, green_patch, blue_patch],
                ["X-axis", "Y-axis", "Z-axis"],
                loc="upper right",
            )

            plt.tight_layout()
            plt.show()

        n_linear_steps = 5
        linear_pregrasp_to_grasp = []

        position = starting_position
        quat = starting_quat
        orig_distance = distance
        for _i in range(n_linear_steps + 1):
            distance -= orig_distance / n_linear_steps
            position = starting_position + rotation.apply(np.array([0, 0, -distance]))
            linear_pregrasp_to_grasp.append((position, quat))
        linear_pregrasp_to_grasp.append((grasp_pose[0], grasp_pose[1]))

        self.linear_pregrasp_to_grasp = []
        for pos, quat in linear_pregrasp_to_grasp:
            self.linear_pregrasp_to_grasp.append(
                self.orient_grasp_pose_based_on_object(object_name, (pos, quat))
            )

        if plot:
            # plot the linear pregrasp to grasp trajectory
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            for pos, quat in self.linear_pregrasp_to_grasp:
                self.plot_3d_frame(ax, pos, quat, "pregrasp", "blue", 0.15)
            plt.show()

        return self.linear_pregrasp_to_grasp[0][0], self.linear_pregrasp_to_grasp[0][1]

    def is_collision_free(self, world_grasp_pos, world_grasp_quat) -> bool:
        # check with new data
        data = mujoco.MjData(self.model)

        # gripper_id = self.model.body("robot_0/").id
        gripper_qpos_start = self.model.joint("robot_0/").qposadr[0]  # dofadr[0]
        data.qpos[gripper_qpos_start : gripper_qpos_start + 3] = world_grasp_pos
        data.qpos[gripper_qpos_start + 3 : gripper_qpos_start + 7] = world_grasp_quat
        mujoco.mj_step(self.model, data)
        if self.viewer:
            self.viewer.sync()

        # check collision of gripper
        for n in range(data.ncon):
            geom1 = data.contact[n].geom1
            body1 = self.model.geom(geom1).bodyid[0]
            root_body1 = self.model.body(body1).rootid[0]
            if root_body1 == self.model.body("robot_0/").id:
                self.logger.info(f"Collision detected between {geom1} and {data.contact[n].geom2}")
                return False
            geom2 = data.contact[n].geom2
            body2 = self.model.geom(geom2).bodyid[0]
            root_body2 = self.model.body(body2).rootid[0]
            if root_body2 == self.model.body("robot_0/").id:
                self.logger.info(f"Collision detected between {geom2} and {data.contact[n].geom1}")
                return False

        return True

    def place_gripper_at_grasp(
        self, object_name: str, grasp_pose: tuple[np.ndarray, np.ndarray], pregrasp: bool = False
    ) -> bool:
        """Place the gripper at the grasp pose."""

        world_grasp_pos, world_grasp_quat = self.orient_grasp_pose_based_on_object(
            object_name, grasp_pose
        )
        if world_grasp_pos is None:
            return False

        if not pregrasp and not self.is_collision_free(world_grasp_pos, world_grasp_quat):
            return False

        if pregrasp:
            # PRE GRASP CHECK
            world_grasp_pos, world_grasp_quat = self.get_pregrasp_pose(object_name, grasp_pose)
            if not self.is_collision_free(world_grasp_pos, world_grasp_quat):
                return False

        # If all good, place the gripper at the grasp pose
        self.data.mocap_pos[0] = world_grasp_pos
        self.data.mocap_quat[0] = world_grasp_quat

        # gripper_id = self.model.body("robot_0/").id
        gripper_qpos_start = self.model.joint("robot_0/").qposadr[0]  # dofadr[0]
        self.data.qpos[gripper_qpos_start : gripper_qpos_start + 3] = world_grasp_pos
        self.data.qpos[gripper_qpos_start + 3 : gripper_qpos_start + 7] = world_grasp_quat
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

        return True

    def get_object_mass_sum_of_geoms(self, object_name: str) -> float:
        """Get the mass of the object."""
        object_body_name = self.object_body_map[object_name]
        return self.model.body(object_body_name).subtreemass[0]

    def grasp_object(self, object_name: str, pregrasp: bool = False) -> bool:
        """Grasp and pick the object."""
        # reset data per test
        mujoco.mj_resetData(self.model, self.data)

        # settle physics
        mujoco.mj_step(self.model, self.data, nstep=100)
        if self.viewer:
            self.viewer.sync()

        place_gripper = False
        start_time = time.time()
        while not place_gripper and self.place_try < self.grasps_per_object:
            self.place_try += 1
            # Sample a grasp
            if pregrasp:
                grasp_pose = self.sample_grasp(object_name, self.place_try)
            else:
                grasp_pose = self.sample_grasp(object_name, self.place_try)

            if not grasp_pose:
                self.logger.debug(f"Could not sample grasp for {object_name}")
                continue

            grasp_pose = self.tcp_to_base_grasp_pose(grasp_pose, gripper_length=0.155)

            if not self.place_gripper_at_grasp(object_name, grasp_pose, pregrasp):
                self.logger.debug(f"Could not place gripper for {object_name}")
                continue
            else:
                place_gripper = True

        end_time = time.time()

        if not place_gripper:
            self.logger.debug(
                f"Could not place gripper for {object_name}. Time taken: {end_time - start_time}"
            )
            return False

        self.logger.info(f"Time taken to grasp object {object_name}: {end_time - start_time}")
        return True

    def pick_object(self, object_name: str, pregrasp: bool = False):
        # Initial object position
        object_start_pos = self.data.body(self.object_body_map[object_name]).xpos.copy()

        if pregrasp:
            # Move down the gripper towards self.grasp_pose
            assert self.linear_pregrasp_to_grasp is not None
            for pos, quat in self.linear_pregrasp_to_grasp:
                self.data.mocap_pos[0] = pos
                self.data.mocap_quat[0] = quat
                mujoco.mj_step(self.model, self.data, nstep=500)
                if self.viewer:
                    self.viewer.sync()
                if self.recorders is not None:
                    for recorder in self.recorders:
                        recorder(self.data)

        # Simulate and visualize for a few steps - CLOSE THE GRIPPER
        for _step in range(10):
            self.data.ctrl[0] = 255  # -0.8
            mujoco.mj_step(self.model, self.data, nstep=500)  # 50
            if self.viewer:
                self.viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # Simulate and visualize for a few steps - MOVE UP THE GRIPPER
        for _step in range(20):
            self.data.mocap_pos[0] = self.data.mocap_pos[0] + np.array([0, 0, 0.05])
            mujoco.mj_step(self.model, self.data, nstep=50)
            if self.viewer:
                self.viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # Check if object is picked (simplified)
        object_final_pos = self.data.body(self.object_body_map[object_name]).xpos.copy()
        is_picked = object_final_pos[2] - object_start_pos[2] > PICK_HEIGHT_THRESHOLD
        self.logger.info(f"Object {object_name} picked: {is_picked}")

        return is_picked

    def get_object_mass_llm(self, object_name: str) -> float:
        """Get the mass of the object using LLM."""
        from tests.ithor_object_mass_est import ithor_estimated_masses

        for key, value in ithor_estimated_masses.items():
            if key.lower() in object_name.lower():
                return value
        return 0.0

    def run_grasping_test(self, num_iterations: int = None, pregrasp: bool = False) -> None:
        """Run the complete grasping test with visualization."""
        available_objects = list(self.object_grasps.keys())

        if num_iterations is None:
            num_iterations = len(available_objects)
        self.logger.info(f"Starting grasping test with {num_iterations} iterations")

        if not available_objects:
            self.logger.error("No objects available for testing")
            return

        # Start the viewer
        if not self.start_viewer():
            self.logger.error("Failed to start viewer, running test without visualization")
            self.run_grasping_test_no_viewer(num_iterations)
            return

        try:
            # shuffle the objects
            random.shuffle(available_objects)

            for iteration in range(num_iterations):
                self.setup_recorders()

                self.logger.info(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

                # Randomly select an object
                object_name = available_objects[iteration]
                self.logger.info(f"Testing object: {object_name}")

                # Get object mass
                object_mass = self.get_object_mass_sum_of_geoms(object_name)
                llm_object_mass = self.get_object_mass_llm(object_name)
                self.logger.info(
                    f"Object {object_name} mass: {object_mass}, llm mass: {llm_object_mass}"
                )

                n_tries = 0
                is_grasped = False
                is_picked = False
                self.place_try = -1
                while n_tries < MAX_GRASPS_RETRIES:
                    n_tries += 1
                    # Grasp and pick object
                    is_grasped = self.grasp_object(object_name, pregrasp)
                    if is_grasped:
                        self.success_grasp_objects.append(object_name)
                        is_picked = self.pick_object(object_name, pregrasp)
                        if is_picked:
                            n_tries = MAX_GRASPS_RETRIES

                if not is_grasped:
                    self.failed_grasp_objects.append(object_name)
                else:
                    self.success_grasp_objects.append(object_name)

                if not is_picked:
                    self.failed_pick_objects.append(object_name)
                else:
                    self.success_pick_objects.append(object_name)

                # DEBUG: save only failed
                if is_grasped and not is_picked and self.recorders is not None:
                    # if is_grasped and self.recorders is not None:
                    for recorder in self.recorders:
                        recorder.save(
                            os.path.join(self.save_dir, f"{object_name}"), save_video=True
                        )

                # Check if viewer is still running
                if self.viewer and not self.viewer.is_running():
                    self.logger.info("Viewer closed by user, stopping test")
                    break

        except KeyboardInterrupt:
            self.logger.info("Test interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during test: {e}")
        finally:
            self.stop_viewer()

        self.logger.info("Grasping test completed")

    def run_grasping_test_no_viewer(
        self, num_iterations: int = None, pregrasp: bool = False
    ) -> None:
        """Run the grasping test without visualization (fallback)."""
        self.logger.info(f"Running test without viewer for {num_iterations} iterations")
        available_objects = list(self.object_grasps.keys())

        if num_iterations is None:
            num_iterations = len(available_objects)

        # shuffle the objects
        random.shuffle(available_objects)

        for iteration in range(num_iterations):
            self.setup_recorders()

            self.logger.info(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # Randomly select an object
            object_name = available_objects[iteration]
            self.logger.info(f"Testing object: {object_name}")

            # Get object mass
            object_mass = self.get_object_mass_sum_of_geoms(object_name)
            self.logger.info(f"Object {object_name} mass: {object_mass}")

            is_picked = False
            is_grasped = False
            n_tries = 0
            self.place_try = -1
            while not is_picked and n_tries < MAX_GRASPS_RETRIES:
                n_tries += 1
                # Grasp and pick object
                is_grasped = self.grasp_object(object_name, pregrasp)
                if is_grasped:
                    is_picked = self.pick_object(object_name, pregrasp)
                    if is_picked:
                        break

            if not is_grasped:
                self.failed_grasp_objects.append(object_name)
            else:
                self.success_grasp_objects.append(object_name)

            if not is_picked:
                self.failed_pick_objects.append(object_name)
            else:
                self.success_pick_objects.append(object_name)

            # DEBUG: save only failed grasps
            # if is_grasped and not is_picked and self.recorders is not None:
            if is_grasped and self.recorders is not None:
                for recorder in self.recorders:
                    recorder.save(os.path.join(self.save_dir, f"{object_name}"), save_video=True)

        self.logger.info("Grasping test completed (no viewer)")


def create_map(scene: str) -> bool:
    """Create the map of the scene."""
    map_path = scene.replace(".xml", "_map.png")
    if ITHOR:
        map = iTHORMap.from_mj_model_path(scene, px_per_m=200, agent_radius=0.25, device_id=None)
        map.save(map_path)
    elif ProcTHOR:
        map = ProcTHORMap.from_mj_model_path(scene, px_per_m=200, agent_radius=0.25, device_id=None)
        map.save(map_path)
    return True


def add_robot_to_scene(
    scene: str,
    gripper_scene_path: str,
    robot_path: str = "assets/robots/franka_droid/robotiq_2f85_v4/2f85.xml",
) -> bool:
    """Add the robot to the scene."""
    # load the map of the scene
    map_path = scene.replace(".xml", "_map.png")
    map_scene = scene
    if not os.path.exists(map_path):
        print(f"Error: Map file not found: {map_path}")
        # create the map
        create_map(map_scene)

    map = None
    if ITHOR:
        map = iTHORMap.load(map_path)
    elif ProcTHOR:
        map = ProcTHORMap.load(map_path)
    else:
        raise ValueError(f"Unsupported scene type: {scene}")

    # sample a random free point from the map
    free_points = map.get_free_points()
    random_free_point = free_points[np.random.randint(0, len(free_points))]
    pos = random_free_point
    pos[2] = 0.45

    editor = ThorMjModelEditor.from_xml_path(scene)
    editor.set_options()
    editor.set_size(size=5000)
    editor.set_compiler()
    editor.add_robot(xml_path=robot_path, pos=pos, quat=[1, 0, 0, 0])
    editor.add_mocap_body(
        name="target_ee_pose",
        gripper_weld=True,
        gripper_name="robot_0/",
        pos=pos,
        quat=[1, 0, 0, 0],
    )
    editor.save_xml(save_path=gripper_scene_path)
    return True


def run_scene(
    scene: str,
    grasps_per_object: int,
    iterations: int,
    verbose: bool,
    no_viewer: bool,
    pregrasp: bool,
):
    """Run a single scene test. This function is designed to be multiprocessing-safe."""
    try:
        # Add the robot to the scene
        # gripper_scene_path = scene.replace(".xml", f"_{ROBOT_TYPE}_robot.xml")
        gripper_scene_path = scene.replace(".xml", "_robot.xml")
        if not os.path.exists(gripper_scene_path):
            add_robot_to_scene(scene, gripper_scene_path)
            print(f"Robot added to scene: {gripper_scene_path}")

        # Create and run the grasping test environment
        env = GraspTestEnvironment(gripper_scene_path, grasps_per_object)

        # Load the scene
        if not env.load_scene():
            print(f"Error: Failed to load scene: {scene}")
            return {
                "failed_grasp_objects": [],
                "failed_pick_objects": [],
                "success_grasp_objects": [],
                "success_pick_objects": [],
                "error": f"Failed to load scene: {scene}",
            }

        # Load object grasps
        metadata = get_scene_metadata(gripper_scene_path)

        if not env.load_object_grasps(metadata):
            print(f"Error: Failed to load object grasps for scene: {scene}")
            return {
                "failed_grasp_objects": [],
                "failed_pick_objects": [],
                "success_grasp_objects": [],
                "success_pick_objects": [],
                "error": f"Failed to load object grasps for scene: {scene}",
            }

        # Run the grasping test
        if no_viewer:
            env.run_grasping_test_no_viewer(iterations, pregrasp)
        else:
            env.run_grasping_test(iterations, pregrasp)

        return {
            "failed_grasp_objects": env.failed_grasp_objects,
            "failed_pick_objects": env.failed_pick_objects,
            "success_grasp_objects": env.success_grasp_objects,
            "success_pick_objects": env.success_pick_objects,
        }

    except Exception as e:
        print(f"Error processing scene {scene}: {e}")
        traceback.print_exc()

        return {
            "failed_grasp_objects": [],
            "failed_pick_objects": [],
            "success_grasp_objects": [],
            "success_pick_objects": [],
            "error": f"Exception in scene {scene}: {str(e)}",
        }


def main() -> None:
    """Main function to run the grasping test."""
    parser = argparse.ArgumentParser(
        description="Test grasping functionality for objects in a scene"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="assets/scenes/ithor/FloorPlan1_physics_mesh.xml",
        help="Path to the scene XML file",
    )
    parser.add_argument(
        "--grasps_per_object",
        type=int,
        default=1000,  # 50, # 2000 is all grasps
        help="Number of grasps per object (default: 50)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of test iterations (default: 3)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--no-viewer", action="store_true", help="Run without the MuJoCo viewer (headless mode)"
    )
    parser.add_argument("--pregrasp", action="store_true", help="Enable pregrasp")
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Enable multiprocessing for faster execution (only works with --no-viewer)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to use for multiprocessing (default: number of CPU cores)",
    )

    args = parser.parse_args()
    scene = args.scene
    grasps_per_object = args.grasps_per_object
    iterations = args.iterations
    verbose = args.verbose
    no_viewer = args.no_viewer
    pregrasp = args.pregrasp
    multiprocessing = args.multiprocessing
    num_processes = args.num_processes

    # Validate multiprocessing arguments
    if multiprocessing and not no_viewer:
        print("Warning: Multiprocessing requires --no-viewer flag. Disabling multiprocessing.")
        multiprocessing = False

    if multiprocessing:
        if num_processes is None:
            num_processes = cpu_count() // 2
        print(f"Multiprocessing enabled with {num_processes} processes")

    # Set logging level
    if verbose:
        # Enable all logging by setting root logger to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        # Also enable the environment logger
        logging.getLogger("molmo_spaces").setLevel(logging.DEBUG)
        # Enable the GraspTestEnvironment logger
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        print("Verbose logging enabled - you should see detailed log messages")
    else:
        # Disable all logging by setting root logger to CRITICAL + 1
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        print("Logging disabled - use --verbose to see detailed log messages")

    results = {}
    if not scene.endswith("_mesh.xml") and not scene.endswith(".xml"):
        scene_dir = scene
        # get all the scenes in the assets/scenes/ithor_081525 folder
        scenes = [
            # f for f in os.listdir(scene_dir) if f.endswith("_mesh.xml") or f.endswith("_ceiling.xml")
            f
            for f in os.listdir(scene_dir)
            if f.endswith("_mesh.xml") or f.endswith(".xml") and not f.endswith("_ceiling.xml")
        ]

        if multiprocessing and len(scenes) > 1:
            # Use multiprocessing for multiple scenes
            print(
                f"Processing {len(scenes)} scenes with multiprocessing using {num_processes} processes..."
            )

            # Prepare arguments for multiprocessing
            scene_args = []
            for scene_file in scenes:
                scene_path = os.path.join(scene_dir, scene_file)
                scene_args.append(
                    (scene_path, grasps_per_object, iterations, verbose, no_viewer, pregrasp)
                )

            # Run scenes in parallel with progress tracking
            with Pool(processes=num_processes) as pool:
                scene_results = pool.starmap(run_scene, scene_args)

            # Combine results and check for errors
            for i, scene_file in enumerate(scenes):
                results[scene_file] = scene_results[i]
                if "error" in scene_results[i]:
                    print(f"Warning: Scene {scene_file} had an error: {scene_results[i]['error']}")
                else:
                    print(
                        f"Completed scene {scene_file}: {len(scene_results[i]['success_pick_objects'])} successful picks"
                    )

        else:
            # Sequential processing
            for scene_file in scenes:
                print(f"Processing scene: {scene_file}")
                scene_path = os.path.join(scene_dir, scene_file)
                results[scene_file] = run_scene(
                    scene_path, grasps_per_object, iterations, verbose, no_viewer, pregrasp
                )
    else:
        results[scene] = run_scene(
            scene, grasps_per_object, iterations, verbose, no_viewer, pregrasp
        )

    # aggregate results
    all_results = {
        "failed_grasp_objects": [],
        "failed_pick_objects": [],
        "success_grasp_objects": [],
        "success_pick_objects": [],
    }

    # Track scenes with errors
    scenes_with_errors = []

    for scene, scene_results in results.items():
        if "error" in scene_results:
            scenes_with_errors.append(scene)
            print(f"Skipping results from scene {scene} due to error")
            continue

        all_results["failed_grasp_objects"].extend(scene_results["failed_grasp_objects"])
        all_results["failed_pick_objects"].extend(scene_results["failed_pick_objects"])
        all_results["success_grasp_objects"].extend(scene_results["success_grasp_objects"])
        all_results["success_pick_objects"].extend(scene_results["success_pick_objects"])

    if scenes_with_errors:
        print(f"\nWarning: {len(scenes_with_errors)} scenes had errors and were skipped:")
        for scene in scenes_with_errors:
            print(f"  - {scene}")

    print(f"Failed grasp objects: {all_results['failed_grasp_objects']}")
    print(f"Failed pick objects: {all_results['failed_pick_objects']}")
    print(f"Success grasp objects: {all_results['success_grasp_objects']}")
    print(f"Success pick objects: {all_results['success_pick_objects']}")

    if multiprocessing and len(scenes) > 1:
        print("\nMultiprocessing summary:")
        print(f"  - Total scenes processed: {len(scenes)}")
        print(f"  - Scenes with errors: {len(scenes_with_errors)}")
        print(f"  - Successful scenes: {len(scenes) - len(scenes_with_errors)}")
        print(f"  - Processes used: {num_processes}")

    print("\nGrasping test completed successfully!")

    # save results to a json file
    save_dir = f"debug/grasp_test_videos/{DATE_TIME}/"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    # Set multiprocessing start method for Linux compatibility
    if sys.platform.startswith("linux"):
        mp.set_start_method("spawn", force=True)
    main()
