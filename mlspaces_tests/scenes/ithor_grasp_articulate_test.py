"""
A script to test if the assets are openable.

This script implements a following grasp and open test pipeline:
[ ] 1. Load grasps for the graspable articulable objects
[ ] 2. Sample one grasp
[ ] 3. Place the gripper at the grasp pose
[ ] 4. Grasp the object
[ ] 5. Open the object
[ ] 6. Check if the object is opened
[ ] 7. Repeat 2-6 for multiple objects
[ ] 8. Repeat 1-7 for multiple scenes
[ ] 9. multi processing for faster testing
"""

import argparse
import datetime
import glob
import json
import logging
import os
import random
import re
import time

import mujoco
import numpy as np

from molmo_spaces.utils.articulation_utils import (
    gather_joint_info,
    step_circular_path,
    step_linear_path,
)
from ithor_grasp_test import GraspTestEnvironment, add_robot_to_scene
from scipy.spatial.transform import Rotation as R

from molmo_spaces.editor.constants import ALL_ARTICULATION_TYPES_THOR

EXTRA_ARTICULATION_TYPES_THOR = ALL_ARTICULATION_TYPES_THOR + [
    "cabinet",
    "drawer",
    "oven",
    "dishwasher",
    "showerdoor",
    "RoboTHOR",
]


MAX_GRASPS_RETRIES = 10  # 100 #5 #3 1
OPEN_THRESHOLD = 0.667  # 0.7

DATE_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

DEBUG_SAVE_JUST_FIRST_FRAME = False


ITHOR = True  # True False
ProcTHOR = not ITHOR


def load_grasps_for_object_per_joint(
    object_name, joint_name, num_grasps=50, grasp_dir="assets/grasps"
):
    """Load grasps for a specific object."""
    combined_transforms = []
    pattern = f"{grasp_dir}/{object_name}/{joint_name}*_filtered.json"
    grasp_files = glob.glob(pattern)
    # all folder names in grasp_dir
    # print(pattern)
    assert len(grasp_files) <= 1, f"Expected up to 1 grasp file, got {len(grasp_files)}"
    for filename in grasp_files:
        print(f"Loading grasps from: {filename}")
        try:
            with open(filename, "r") as f:
                json_data = json.load(f)
                # transforms = json_data.get("transforms", [])
                # transforms = json_data.get("root_transforms", [])
                transforms = json_data.get("parent_transform", [])
                combined_transforms.extend(transforms)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not combined_transforms:
        print(f"No grasp transformations found for {object_name}")
        return []  # , None

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
    return grasps  # , joint_info["parent_body"]


def extract_base_articulate_object_names(model):
    """Extract base object names from the model."""
    base_names = set()
    object_body_map = {}
    # object_pattern = re.compile(r"^([A-Za-z]+_[A-Za-z0-9]+)(_.*)?$")
    # object_pattern = re.compile(r"^([A-Za-z]+_[A-Za-z0-9]+)_([A-Za-z0-9]+)(_.*)?$")

    for i in range(model.njnt):
        joint_type = model.joint(i).type
        joint_name = model.joint(i).name
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue

        body_id = model.joint(i).bodyid
        # Extract the first element if it's an array
        if hasattr(body_id, "__len__") and len(body_id) > 0:
            body_id = body_id[0]

        root_body_id = model.body(body_id).rootid
        model.body(body_id).parentid
        # Extract the first element if it's an array
        if hasattr(root_body_id, "__len__") and len(root_body_id) > 0:
            root_body_id = root_body_id[0]

        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)
        joint_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(body_name)

        if ITHOR:
            # all but last element
            base_name = "_".join(body_name.split("_")[:-1])
            print(base_name)

            category_name = body_name.split("_")[0]
            if category_name.lower() == "robot":
                continue
            if not any(
                category_name.lower() in category.lower()
                for category in EXTRA_ARTICULATION_TYPES_THOR
            ):
                continue

            base_names.add(base_name)
            if base_name not in object_body_map:
                object_body_map[base_name] = {}
            object_body_map[base_name][joint_name] = joint_body_name

        if ProcTHOR:
            joint_name = model.joint(i).name
            name_end = joint_name.split("|")[-1]
            match = re.search(r"^\d+_([A-Za-z_]+[\d_]+)", name_end)
            if match:
                base_name = match.group(1).strip("_")

                category_name = base_name.split("_")[0]
                if category_name.lower() == "robot":
                    continue
                if not any(
                    category_name.lower() in category.lower()
                    for category in EXTRA_ARTICULATION_TYPES_THOR
                ):
                    continue

                base_names.add(base_name)
                if base_name not in object_body_map:
                    object_body_map[base_name] = {}
                object_body_map[base_name][joint_name] = joint_body_name  # joint_body_name

    return list(base_names), object_body_map


class GraspArticulateTestEnvironment(GraspTestEnvironment):
    def __init__(self, scene_path: str, grasps_per_object: int = 100) -> None:
        super().__init__(scene_path, grasps_per_object)  # pyright: ignore[reportUnknownMemberType]

    def load_object_grasps(self) -> bool:
        """Load grasps for all available objects."""
        try:
            base_object_names, object_body_map = extract_base_articulate_object_names(self.model)  # pyright: ignore[reportUnknownVariableType]
            # objects_with_grasps = find_objects_with_grasps(base_object_names, "assets/grasps_articulate")  # pyright: ignore[reportUnknownVariableType]

            # Map objects to body names
            self.object_body_map = object_body_map

            # Load grasps for each object
            available_objects = []
            for obj_base_name, actual_joint_to_body_name in self.object_body_map.items():
                joint_names = actual_joint_to_body_name.keys()
                for joint_name in joint_names:
                    actual_body_name = actual_joint_to_body_name[joint_name]

                    root_body_id = self.model.body(actual_body_name).rootid
                    root_body_name = self.model.body(root_body_id).name

                    joint_type = self.model.joint(joint_name).type
                    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                        continue
                    if "handle" in joint_name.lower():  # ignore door handles
                        continue

                    shorten_joint_name = joint_name.replace(root_body_name + "_", "", 1)
                    print(
                        f"Shorten joint name: {shorten_joint_name} {actual_body_name} {joint_name}"
                    )
                    grasps = load_grasps_for_object_per_joint(
                        obj_base_name,
                        shorten_joint_name,
                        self.grasps_per_object,
                        grasp_dir="assets/grasps_articulate",
                    )

                    if grasps:
                        if obj_base_name not in self.object_grasps:
                            self.object_grasps[obj_base_name] = {}
                        self.object_grasps[obj_base_name][joint_name] = grasps

                        available_objects.append(obj_base_name)
                        self.logger.info(
                            f"Loaded {len(grasps)} grasps for {obj_base_name} (body: {actual_body_name}) (joint: {joint_name})"
                        )
                    else:
                        self.logger.warning(
                            f"No grasps found for {obj_base_name} (body: {actual_body_name}) (joint: {joint_name})"
                        )
                        self.failed_grasp_objects.append((obj_base_name, joint_name))

            if not available_objects:
                self.logger.warning("No objects with grasps found")
                return False

            self.logger.info(f"Successfully loaded grasps for {len(available_objects)} objects")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load object grasps: {e}")
            return False

    def orient_grasp_pose_based_on_object(
        self, object_name: str, grasp_pose: tuple[np.ndarray, np.ndarray], joint_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Orient the grasp pose based on the object."""
        if object_name not in self.object_body_map:
            self.logger.error(f"Object {object_name} not found in body map")
            return None, None

        obj_body_name = self.object_body_map[object_name][joint_name]
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

    def sample_grasp(
        self, object_name: str, joint_name: str, index: int = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample a random grasp for the specified object."""
        if object_name not in self.object_grasps:
            self.logger.warning(f"No grasps available for {object_name}")
            return None

        grasps = self.object_grasps[object_name][joint_name]
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

    def get_pregrasp_pose(
        self,
        object_name: str,
        grasp_pose: tuple[np.ndarray, np.ndarray],
        joint_name: str,
        distance=0.025,
    ):
        starting_quat = grasp_pose[1]
        rotation = R.from_quat(
            [starting_quat[1], starting_quat[2], starting_quat[3], starting_quat[0]]
        )
        starting_position = grasp_pose[0] + rotation.apply(np.array([0, 0, -distance]))

        n_linear_steps = 5
        linear_pregrasp_to_grasp = []

        position = starting_position
        quat = starting_quat
        orig_distance = distance
        for _i in range(n_linear_steps + 1):
            distance -= orig_distance / n_linear_steps
            position = starting_position + rotation.apply(np.array([0, 0, -distance]))
            linear_pregrasp_to_grasp.append((position, quat))
        # linear_pregrasp_to_grasp.append((grasp_pose[0], grasp_pose[1]))

        self.linear_pregrasp_to_grasp = []
        for pos, quat in linear_pregrasp_to_grasp:
            self.linear_pregrasp_to_grasp.append(
                self.orient_grasp_pose_based_on_object(object_name, (pos, quat), joint_name)
            )

        return self.linear_pregrasp_to_grasp[0][0], self.linear_pregrasp_to_grasp[0][1]

    def place_gripper_at_pregrasp(
        self,
        object_name: str,
        grasp_pose: tuple[np.ndarray, np.ndarray],
        joint_name: str,
        pregrasp: bool = False,
    ) -> bool:
        """Place the gripper at the grasp pose."""
        # PRE GRASP CHECK
        distance = 0.3 if DEBUG_SAVE_JUST_FIRST_FRAME else 0.025
        world_grasp_pos, world_grasp_quat = self.get_pregrasp_pose(
            object_name, grasp_pose, joint_name, distance=distance
        )
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

    def grasp_object(self, object_name: str, joint_name: str, pregrasp: bool = False) -> bool:
        """Grasp and pick the object."""
        # reset data per test
        mujoco.mj_resetData(self.model, self.data)

        # settle physics
        mujoco.mj_step(self.model, self.data, nstep=10)
        if self.viewer:
            self.viewer.sync()

        place_gripper = False
        start_time = time.time()
        while not place_gripper and self.place_try < self.grasps_per_object:
            self.place_try += 1
            # Sample a grasp
            grasp_pose = self.sample_grasp(object_name, joint_name, self.place_try)
            # grasp_pose = self.tcp_to_base_grasp_pose(grasp_pose, gripper_length=0.0) # 0.13 for Robotiq tcp to base. From RUM base to robotiq base

            if not grasp_pose:
                self.logger.debug(f"Could not sample grasp for {object_name}")
                continue

            if not self.place_gripper_at_pregrasp(object_name, grasp_pose, joint_name, pregrasp):
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

    def pull_object(self, object_name: str, joint_name: str, pregrasp: bool = False):
        # Move down the gripper towards self.grasp_pose
        assert self.linear_pregrasp_to_grasp is not None
        for pos, quat in self.linear_pregrasp_to_grasp:
            self.data.mocap_pos[0] = pos
            self.data.mocap_quat[0] = quat
            mujoco.mj_step(self.model, self.data, nstep=100)
            if self.viewer:
                self.viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # Simulate and visualize for a few steps - CLOSE THE GRIPPER
        for _step in range(10):
            self.data.ctrl[0] = 255  # -0.8
            mujoco.mj_step(self.model, self.data, nstep=100)  # 50
            if self.viewer:
                self.viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # update joint pos after stepping in
        joint_info = gather_joint_info(self.model, self.data, joint_name)

        # Initial object position
        object_start_pos = self.data.qpos[joint_info["joint_qpos_adr"]].copy()

        # Simulate and visualize for a few steps -pULL BACK THE GRIPPER EITHER LINEARE OR ARC MOTION
        if joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_HINGE:
            path = step_circular_path(
                current_pos=self.data.mocap_pos[0],
                current_quat=self.data.mocap_quat[0],
                joint_info=joint_info,
                n_waypoints=80,
            )
        elif joint_info["joint_type"] == mujoco.mjtJoint.mjJNT_SLIDE:
            # self.linear_pregrasp_to_grasp.reverse()
            max_range = joint_info["max_range"]
            normalize_dir_axis = (
                self.linear_pregrasp_to_grasp[-1][0] - self.linear_pregrasp_to_grasp[0][0]
            ) / np.linalg.norm(
                self.linear_pregrasp_to_grasp[-1][0] - self.linear_pregrasp_to_grasp[0][0]
            )
            end_pos = self.linear_pregrasp_to_grasp[-1][0] + normalize_dir_axis * max_range

            path = step_linear_path(
                to_handle_dist=end_pos - self.linear_pregrasp_to_grasp[-1][0],
                current_pos=self.data.mocap_pos[0],
                current_quat=self.data.mocap_quat[0],
                step_size=0.005,
                is_reverse=True,
            )
        else:
            raise ValueError(f"Unknown joint type: {joint_info['joint_type']}")

        assert self.linear_pregrasp_to_grasp is not None
        # reverse the linear pregrasp to grasp
        # self.linear_pregrasp_to_grasp.reverse()
        positions = path["mocap_pos"]
        quaternions = path["mocap_quat"]
        for pos, quat in zip(positions, quaternions):
            self.data.mocap_pos[0] = pos
            self.data.mocap_quat[0] = quat
            mujoco.mj_step(self.model, self.data, nstep=100)  # 500
            if self.viewer:
                self.viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # Check if object is picked (simplified)
        object_final_pos = self.data.qpos[joint_info["joint_qpos_adr"]].copy()
        max_range = joint_info["max_range"]
        # if already opened, subtract the max range
        if object_start_pos > object_final_pos:
            # joint_range = joint_info["joint_range"]
            max_range = np.abs(object_start_pos)
        else:
            max_range = max_range - np.abs(object_start_pos)
        is_pulled = np.abs(object_final_pos - object_start_pos) / max_range >= OPEN_THRESHOLD
        return is_pulled

    def run_grasping_test_no_viewer(
        self, num_iterations: int = None, pregrasp: bool = False
    ) -> None:
        """Run the grasping test without visualization (fallback)."""
        self.logger.info(f"Running test without viewer for {num_iterations} iterations")
        available_objects = list(self.object_grasps.keys())

        # shuffle the objects
        random.shuffle(available_objects)

        for iteration in range(len(available_objects)):
            self.logger.info(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # Randomly select an object
            object_name = available_objects[iteration]
            for joint_name in self.object_grasps[object_name]:
                self.setup_recorders(save_images=DEBUG_SAVE_JUST_FIRST_FRAME)
                self.logger.info(f"Testing object: {object_name} (joint: {joint_name})")

                is_picked = False
                is_grasped = False
                n_tries = 0
                self.place_try = -1
                while not is_picked and n_tries < MAX_GRASPS_RETRIES:
                    n_tries += 1
                    # Grasp and pick object
                    is_grasped = self.grasp_object(object_name, joint_name, pregrasp=True)
                    if is_grasped:
                        if DEBUG_SAVE_JUST_FIRST_FRAME:
                            if self.recorders is not None:
                                for recorder in self.recorders:
                                    recorder(self.data)
                            is_picked = True
                        else:
                            is_picked = self.pull_object(object_name, joint_name, pregrasp=True)
                        if is_picked:
                            break

                if not is_grasped:
                    self.failed_grasp_objects.append((object_name, joint_name))
                else:
                    self.success_grasp_objects.append((object_name, joint_name))

                if not is_picked:
                    self.failed_pick_objects.append((object_name, joint_name))
                else:
                    self.success_pick_objects.append((object_name, joint_name))

                # DEBUG: save only failed grasps
                # if is_grasped and not is_picked and self.recorders is not None:
                if is_grasped and self.recorders is not None:
                    for recorder in self.recorders:
                        recorder.save(os.path.join(self.save_dir, f"{joint_name}"), save_video=True)
                if DEBUG_SAVE_JUST_FIRST_FRAME:
                    for recorder in self.recorders:
                        print(len(recorder.data))
                        recorder.save(
                            os.path.join(self.save_dir, f"{joint_name}"), save_video=False
                        )

        self.logger.info("Grasping test completed (no viewer)")


def run_scene(gripper_scene_path: str, num_grasps: int):
    logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(gripper_scene_path):
        scene_path = gripper_scene_path.replace("_robot.xml", ".xml")
        if not os.path.exists(scene_path):
            print(f"Scene path {scene_path} does not exist")
            return
        # add_robot_to_scene(scene_path, gripper_scene_path)
        add_robot_to_scene(
            scene_path,
            gripper_scene_path,
            robot_path="assets/robots/franka_droid/robotiq_2f85_v4/model.xml",
        )
        print(f"Robot added to scene: {gripper_scene_path}")

    # load scene
    success_info = {
        "failed_grasp_objects": [],
        "failed_pick_objects": [],
        "success_grasp_objects": [],
        "success_pick_objects": [],
    }
    env = GraspArticulateTestEnvironment(gripper_scene_path, num_grasps)
    env.load_scene()
    if env.load_object_grasps():
        env.run_grasping_test_no_viewer()

        success_info = {
            "failed_grasp_objects": env.failed_grasp_objects,
            "failed_pick_objects": env.failed_pick_objects,
            "success_grasp_objects": env.success_grasp_objects,
            "success_pick_objects": env.success_pick_objects,
        }
        print(success_info)
        success_rate = len(env.success_pick_objects) / (
            len(env.success_grasp_objects) + len(env.failed_grasp_objects)
        )
        if not os.path.exists(env.save_dir):
            os.makedirs(env.save_dir, exist_ok=True)
        with open(f"{env.save_dir}/success_info_{success_rate:.3f}.json", "w") as f:
            json.dump(success_info, f)

    return success_info


def main() -> None:
    """Main function to run the grasping test."""
    parser = argparse.ArgumentParser(
        description="Test grasping functionality for objects in a scene"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="assets/scenes/ithor_082625/FloorPlan1_physics_mesh.xml",
        help="Path to the scene XML file",
    )
    parser.add_argument(
        "--num_grasps", type=int, default=1000, help="Number of grasps per object (default: 100)"
    )
    args = parser.parse_args()
    num_grasps = args.num_grasps
    scene_path = args.scene

    results = {}
    if not scene_path.endswith("_mesh.xml") and not scene_path.endswith(".xml"):
        scene_dir = scene_path
        if ITHOR:
            scenes = [f for f in os.listdir(scene_dir) if f.endswith("_mesh.xml")]
            scenes = sorted(scenes, key=lambda x: int(x.split("FloorPlan")[1].split("_")[0]))

        elif ProcTHOR:
            scenes = [f for f in os.listdir(scene_dir) if f.endswith("_ceiling.xml")]
        print(f"Processing {len(scenes)} scenes")
        # sort scenes by name FloorPlan{i}

        for scene in scenes:
            gripper_scene_path = scene.replace(".xml", "_robot.xml")
            print(f"Processing scene: {scene}")
            results[scene] = run_scene(os.path.join(scene_dir, gripper_scene_path), num_grasps)
    else:
        # gripper_scene_path = scene_path.replace(".xml", "_robot.xml")
        gripper_scene_path = scene_path.replace(".xml", "_robotiq.xml")
        print(f"Processing scene: {scene_path}")
        results[scene_path] = run_scene(gripper_scene_path, num_grasps)

    ## Aggregate results
    all_results = {
        "failed_grasp_objects": [],
        "failed_pick_objects": [],
        "success_grasp_objects": [],
        "success_pick_objects": [],
    }

    for scene, scene_results in results.items():
        all_results["failed_grasp_objects"].extend(scene_results["failed_grasp_objects"])
        all_results["failed_pick_objects"].extend(scene_results["failed_pick_objects"])
        all_results["success_grasp_objects"].extend(scene_results["success_grasp_objects"])
        all_results["success_pick_objects"].extend(scene_results["success_pick_objects"])

    final_success_rate = len(all_results["success_pick_objects"]) / (
        len(all_results["success_grasp_objects"]) + len(all_results["failed_grasp_objects"])
    )
    print("-" * 100)
    print("Final results:")
    print("-" * 100)
    print(f"Final success rate: {final_success_rate:.3f}")
    print(f"Final success count: {len(all_results['success_pick_objects'])}")
    print(
        f"Final total count: {len(all_results['success_grasp_objects']) + len(all_results['failed_grasp_objects'])}"
    )
    print("-" * 100)

    with open(
        f"debug/grasp_test_videos/{DATE_TIME}/results_{final_success_rate:.3f}.json", "w"
    ) as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    main()
