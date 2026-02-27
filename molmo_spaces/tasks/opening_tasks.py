import copy
import logging
from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import Door
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.env.object_manager import MlSpacesArticulationObject
from molmo_spaces.env.rby1_sensors import get_rby1_door_opening_sensors
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class OpeningTask(PickTask):
    """Opening task implementation

    Types: Cabinet, Drawer, Oven, Dishwasher, Showerdoor, etc
    """

    def __init__(
        self,
        env: BaseMujocoEnv,
        exp_config: MlSpacesExpConfig,
        sensor_suite: SensorSuite | None = None,
    ) -> None:
        super().__init__(env, exp_config, sensor_suite)
        self.exp_config = exp_config

        self.articulation_objects = self._get_articulation_objects()

        if self.config.task_type == "close":
            assert len(self.articulation_objects) == 1

    def get_task_description(self) -> str:
        pickup_obj_name = self.config.task_config.referral_expressions["pickup_obj_name"]
        return f"Open the {pickup_obj_name}"

    def _get_articulation_objects(self):
        articulation_objects = []
        for n in range(self._env.n_batch):
            om = self._env.object_managers[n]
            articulation_object = om.get_object_by_name(self.config.task_config.pickup_obj_name)

            if self.exp_config.task_config.any_inst_of_category:
                cat = om.get_annotation_category(articulation_object)
                candidates = om.get_objects_of_type(cat)

                filtered_candidates = []
                for candidate in candidates:
                    # NOTE(yejin): sometimes non jointed objects get picked up
                    if isinstance(candidate, MlSpacesArticulationObject):
                        filtered_candidates.append(candidate)
                articulation_objects.append(filtered_candidates)
            else:
                if isinstance(articulation_object, MlSpacesArticulationObject):
                    articulation_objects.append([articulation_object])

        return articulation_objects

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig) -> SensorSuite:
        """Create a sensor suite from configuration using the centralized get_core_sensors function."""
        from molmo_spaces.env.sensors import get_core_sensors

        sensors = get_core_sensors(config)
        return SensorSuite(sensors)

    def judge_success(self) -> bool:
        success = self.get_reward()[0] >= self.config.task_config.task_success_threshold
        return success

    def get_reward(self) -> np.ndarray:
        """Calculate reward for each environment in the batch.

        Returns:
            0 to 1 reward based on the percentage of the object that is opened.

        Note:
            Assumes closed position is always at 0. Works for both:
            - [0, 1.57] range (closed at 0, open at 1.57)
            - [-1.57, 0] range (closed at 0, open at -1.57)
        """
        rewards_envs = np.zeros(self._env.n_batch)

        if self.config.task_type == "open":
            for n in range(self._env.n_batch):
                reward_cand = []
                for articulation_object in self.articulation_objects[n]:
                    for j in range(articulation_object.njoints):
                        current_joint_state = articulation_object.get_joint_position(j)
                        # Closed position is always 0, so distance from 0 is the opening amount
                        # abs() handles both positive [0, 1.57] and negative [-1.57, 0] ranges
                        _joint_range = articulation_object.get_joint_range(j)
                        joint_range_float = np.abs(_joint_range[1] - _joint_range[0])
                        percent_open = np.abs(current_joint_state) / joint_range_float
                        joint_type = articulation_object.get_joint_type(j)
                        if joint_type == mujoco.mjtJoint.mjJNT_FREE or (
                            percent_open == np.inf and joint_range_float == 0
                        ):
                            continue
                        reward_cand.append(percent_open)
                if len(reward_cand) > 0:
                    rewards_envs[n] = max(reward_cand)
                else:
                    rewards_envs[n] = 0.0

        elif self.config.task_type == "close":
            # for close, assume only one object is open
            for n in range(self._env.n_batch):
                assert len(self.articulation_objects) == 1
                articulation_object = self.articulation_objects[n][0]
                current_joint_state = articulation_object.get_joint_position(
                    self.config.task_config.joint_index
                )
                # Closed position is always 0, so distance from 0 is the opening amount
                # abs() handles both positive [0, 1.57] and negative [-1.57, 0] ranges
                _joint_range = articulation_object.get_joint_range(
                    self.config.task_config.joint_index
                )
                joint_range_float = np.abs(_joint_range[1] - _joint_range[0])
                percent_open = np.abs(current_joint_state) / joint_range_float
                rewards_envs[n] = percent_open

            # negate the quanity
            rewards_envs = 1 - rewards_envs
        else:
            raise ValueError(f"Invalid task type: {self.config.task_type}")
        
        return rewards_envs

    def get_info(self) -> list[dict[str, Any]]:
        """Get additional metrics for each environment."""
        metrics = []

        for i in range(self._env.n_batch):
            metrics.append(
                {
                    # TODO(max): just saving the first candidates joint pos for now.
                    "joint_position": self.articulation_objects[i][0].get_joint_position(
                        self.config.task_config.joint_index
                    ),
                    "success": self.judge_success(),
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics


class DoorOpeningTask(BaseMujocoTask):
    """Door opening task implementation."""

    def __init__(
        self,
        env: BaseMujocoEnv,
        exp_config: MlSpacesExpConfig,
        sensor_suite: SensorSuite | None = None,
    ) -> None:
        # Create sensor suite if not provided and sensors are enabled
        if sensor_suite is None and exp_config.task_config.use_sensors:
            sensor_suite = self._create_sensor_suite_from_config(exp_config)

        super().__init__(env, exp_config, sensor_suite)
        self.exp_config = exp_config

        # Get door object in the environment
        self.door_object = Door(exp_config.task_config.door_body_name, env.mj_datas[0])

        # Task state tracking
        self.door_opened = False

        # Initialize task state
        self.current_door_joint_state = (
            self.exp_config.task_config.articulated_joint_reset_state.copy()
        )
        self.door_object.set_joint_position(
            self.door_object.get_hinge_joint_index(), float(self.current_door_joint_state[0])
        )

        # Reset robot and set to base pose
        for robot in self.env.robots:
            robot.reset()
            robot.set_world_pose(self.exp_config.task_config.robot_base_pose)

        # Run a mujoco forward to update the environment
        mujoco.mj_forward(self.env.mj_model, self.env.mj_datas[0])

        # Check which handle to use in this task episode (Doors have handles on both sides)
        self._use_other_side_handle = False  # Initially choose the default handle
        other_side_handle_chosen = (
            self.check_if_use_flip_side_handle()
        )  # check if default handle is on the other side of the door
        self._use_other_side_handle = other_side_handle_chosen
        # Check if we are pushing or pulling the door for this task episode
        self._is_pushing_door = self.check_if_pushing_door()

    def get_task_description(self) -> str:
        return "Push the door open" if self._is_pushing_door else "Pull the door open"

    def _create_sensor_suite_from_config(self, exp_config: MlSpacesExpConfig) -> SensorSuite:
        """Create a sensor suite from configuration using the RBY1-specific sensor function."""
        sensors = get_rby1_door_opening_sensors(exp_config)
        return SensorSuite(sensors)

    def register_policy(self, policy) -> None:
        """Register a policy with the task and reset the task after policy initialization.

        For door opening tasks, we need to reset the task after the policy is initialized
        to avoid EGL context errors.
        """
        super().register_policy(policy)
        self.reset()

    def get_reward(self) -> np.ndarray:
        """Calculate the reward for the current state."""
        rewards = np.zeros(self.env.n_batch)

        # Reward for opening the door (sparse)
        rewards[0] = self.exp_config.task_config.door_open_reward * self.judge_success()

        return rewards

    def step(
        self, actions: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Override step to add door opening percentage logging."""
        # Call parent step method
        observation, reward, terminated, truncated, info = super().step(actions)

        # Log door opening percentage
        percent_open = self.get_door_open_percentage()

        log.debug(f"Step {self.episode_step_count}: Door open percentage: {percent_open:.2%} ")

        return observation, reward, terminated, truncated, info

    def is_terminal(self) -> np.ndarray:
        """Check if the task has reached a terminal state."""
        terminal = np.zeros(self.env.n_batch, dtype=bool)

        # Task is terminal if successful OR if done action was received
        # (Note: difference between terminal and timed_out)
        terminal[0] = self.judge_success() or self._done_action_received

        return terminal

    def get_door_open_percentage(self) -> float:
        """
        Get the current door opening percentage (0.0 = closed, 1.0 = fully open).

        Returns:
            Float between 0.0 and 1.0 representing the door opening percentage
        """
        # Update door state to get current joint state

        # Get current joint state from door object
        current_joint_state = self.door_object.get_joint_position(
            self.door_object.get_hinge_joint_index()
        )
        self.current_door_joint_state = current_joint_state

        # Calculate percentage
        joint_range = self.exp_config.task_config.articulated_joint_range
        percent_open = (self.current_door_joint_state - joint_range[0]) / (
            joint_range[1] - joint_range[0]
        )

        # Clip to [0, 1]
        return float(np.clip(percent_open, 0.0, 1.0))

    def judge_success(self) -> bool:
        """Judge if the task is successful.

        Uses the maximum door opening percentage reached during the episode,
        not the current percentage (since the door might close after being opened).
        """
        # Use maximum door opening reached during episode for success determination
        if self.get_door_open_percentage() >= self.exp_config.task_config.door_openness_threshold:
            self.door_opened = True

        else:
            self.door_opened = False

        return self.door_opened

    def get_door_joint_position(self) -> np.ndarray:
        """Get the position of the door joint."""
        return self.door_object.get_joint_anchor_position(self.door_object.get_hinge_joint_index())

    def get_door_handle_position(self) -> np.ndarray:
        """Get the position of the door handle in world frame."""
        return self.door_object.get_handle_pose()[:3]

    def get_door_handle_orientation(self) -> np.ndarray:
        """Get the orientation of the door handle in world frame."""
        handle_quat = self.door_object.get_handle_pose()[3:]
        handle_rot_mat = R.from_quat(handle_quat, scalar_first=True).as_matrix()
        return handle_rot_mat

    def get_door_handle_extents(self) -> np.ndarray:
        """Get the extents of the door handle bounding box in world frame."""
        bbox_half_lengths = self.door_object.get_handle_bboxes_array()[
            0
        ][
            3:
        ]  # (x,y,z half-lengths) - AABB is [center, size], so [0][3:] gets the half-sizes of the first handle
        return bbox_half_lengths * 2  # (x,y,z half-lengths * 2)

    def get_door_surface_normal_from_geometry(self) -> np.ndarray:
        """Get door surface normal by analyzing door geometry (more robust method).

        Ensures the normal points toward the robot's current position.
        """
        # Get door position and handle position
        door_handle_pos = self.get_door_handle_position()
        door_joint_pos = self.get_door_joint_position()

        # Get current robot position
        robot_pos = self.get_current_robot_position()

        # Calculate the vector from hinge to handle (this gives us the door's "width" direction)
        hinge_to_handle = door_handle_pos - door_joint_pos
        hinge_to_handle[2] = 0  # Keep only horizontal direction

        # The surface normal should be perpendicular to the hinge-to-handle vector
        # and point away from the door's front face
        # We'll use the cross product with the vertical (Z) axis
        vertical = np.array([0, 0, 1])
        surface_normal = np.cross(hinge_to_handle, vertical)

        # Normalize the result
        surface_normal = surface_normal / np.linalg.norm(surface_normal)

        # Ensure the normal points toward the robot
        # Calculate vector from door handle to robot
        door_to_robot = robot_pos - door_handle_pos
        door_to_robot[2] = 0  # Keep only horizontal direction

        # If the dot product is negative, flip the normal to point toward robot
        if np.dot(surface_normal, door_to_robot) < 0:
            surface_normal = -surface_normal

        return surface_normal

    def get_current_robot_position(self) -> np.ndarray:
        """Get the current robot position in world coordinates."""
        # Get robot position from the first robot in the environment
        robot_view = self.env.robots[0].robot_view
        base_pose = robot_view.base.pose
        return base_pose[:3, 3]  # Extract position (x, y, z)

    def get_target_ee_pose(
        self,
        current_arm: str = "left",
        maintain_orientation: bool = False,
        offset_distance: float = 0.0,
        articulate_deltas: list[float] | None = None,
        return_both_poses: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Get the target ee pose for grasping or articulating the door handle.
        Args:
            offset_distance: The distance to offset the grasp position from the door handle position, Eg. for pre-grasping the handle.
            articulate_deltas: The deltas to articulate the door joints. If None, don't articulate the door joints.
            return_both_poses: If True, returns both symmetric poses (original and 180-degree flipped). If False, returns only the original pose.
        Returns:
            The target ee pose for grasping or articulating the door handle.
            If return_both_poses=True, returns a tuple of (original_pose, flipped_pose).
        """

        # Using just one handle for now
        if articulate_deltas is not None:
            # Create a copy of the door object to make the new target pose
            door_obj_copy = copy.deepcopy(self.door_object)
            door_obj_copy.set_joint_position(
                door_obj_copy.get_hinge_joint_index(),
                self.current_door_joint_state + articulate_deltas[0],
            )

            door_handle_position = door_obj_copy.get_handle_pose()[:3]
            door_handle_rot_mat = R.from_quat(
                door_obj_copy.get_handle_pose()[3:], scalar_first=True
            ).as_matrix()
        else:
            # Get handle positions and orientations
            door_handle_position = self.get_door_handle_position()
            door_handle_rot_mat = self.get_door_handle_orientation()

        # Get ee pose from handle pose
        target_ee_pos, target_ee_rot_mat, bbox_offset = self.handle_pose_to_ee_pose(
            door_handle_position, door_handle_rot_mat
        )

        # Add offset distance to be at the edge of the handle (Also used for eg. for adding a pre-grasping distance to the handle)
        offset_distance += bbox_offset
        # optionally also add tcp offset distance to move the tcp further / closer to the handle
        offset_distance += self.exp_config.task_config.additional_tcp_offset_distance
        target_ee_tf = np.eye(4)
        target_ee_tf[:3, 3] = target_ee_pos
        target_ee_tf[:3, :3] = target_ee_rot_mat
        offset_tf_mat = np.eye(4)
        offset_tf_mat[:3, 3] = np.array(
            [offset_distance, 0.0, 0.0]
        )  # offset is along handle i.e. X axis (forward)
        target_ee_tf_after_offset = target_ee_tf @ offset_tf_mat  # post-multiply X-offset transform

        target_ee_pos = target_ee_tf_after_offset[:3, 3]

        # Add any additional robot TCP rotation offset
        target_ee_rot = R.from_matrix(target_ee_rot_mat) * R.from_matrix(
            self.exp_config.task_config.additional_tcp_rotation_offset_mat
        )

        if current_arm == "left":
            quat = np.array([0, -0.707107, 0.707107, 0])
            target_ee_rot = target_ee_rot * R.from_quat(quat, scalar_first=True)
        if current_arm == "right":
            quat = np.array([0, 0.707107, 0.707107, 0])
            target_ee_rot = target_ee_rot * R.from_quat(quat, scalar_first=True)

        # Set as target_ee_pose
        target_ee_quat = target_ee_rot.as_quat(scalar_first=True)
        target_ee_pose = np.concatenate([target_ee_pos, target_ee_quat])

        # If requested, generate the symmetric pose (180-degree rotation around gripper approach axis)
        if return_both_poses:
            flip_rotation = R.from_euler("z", np.pi)
            target_ee_rot_flipped = target_ee_rot * flip_rotation
            target_ee_quat_flipped = target_ee_rot_flipped.as_quat(scalar_first=True)
            target_ee_pose_flipped = np.concatenate([target_ee_pos, target_ee_quat_flipped])
            if self.exp_config.task_config.viz_target_ee:
                self.env.mj_datas[0].mocap_pos[0] = target_ee_pose[:3]
                self.env.mj_datas[0].mocap_quat[0] = target_ee_pose[3:]
            return target_ee_pose, target_ee_pose_flipped
        if maintain_orientation:
            current_ee_pose = (
                self.env.robots[0]
                .robot_view.get_move_group(f"{current_arm}_arm")
                .leaf_frame_to_world
            )
            current_ee_rot = R.from_matrix(current_ee_pose[:3, :3])
            target_ee_pose = np.concatenate(
                [target_ee_pos, current_ee_rot.as_quat(scalar_first=True)]
            )

        # Optional: Visualize target ee pose
        if self.exp_config.task_config.viz_target_ee:
            self.env.mj_datas[0].mocap_pos[0] = target_ee_pose[:3]
            self.env.mj_datas[0].mocap_quat[0] = target_ee_pose[3:]
        return target_ee_pose

    def get_target_head_pose(self, target_pos: np.ndarray) -> np.ndarray:
        robot_view = self.env.robots[0].robot_view

        # Get head position in world frame
        head_pose_to_world = robot_view.get_move_group("head").leaf_frame_to_world
        head_pos = head_pose_to_world[:3, 3]

        base_pose_to_world = robot_view.base.pose

        base_rot_matrix = base_pose_to_world[:3, :3]

        # Calculate look vector in world frame
        look_vector_world = target_pos - head_pos

        # Transform to robot base frame
        look_vector_base = base_rot_matrix.T @ look_vector_world

        horizontal_distance = np.sqrt(look_vector_base[0] ** 2 + look_vector_base[1] ** 2)

        current_head_joints = robot_view.get_move_group("head").joint_pos
        current_pan = current_head_joints[0]

        if horizontal_distance < 1e-6:
            desired_pan = current_pan
            desired_tilt = np.pi / 2 if look_vector_base[2] > 0 else -np.pi / 2
        else:
            # Calculate pan and tilt in BASE frame
            desired_pan = np.arctan2(look_vector_base[1], look_vector_base[0])
            desired_tilt = np.arctan2(-look_vector_base[2], horizontal_distance)

        # Smooth angle wrapping
        pan_diff = desired_pan - current_pan
        pan_diff = np.mod(pan_diff + np.pi, 2 * np.pi) - np.pi
        desired_pan = current_pan + pan_diff

        # Clip to joint limits
        head_move_group = robot_view.get_move_group("head")
        joint_limits = head_move_group.joint_pos_limits
        desired_pan = np.clip(desired_pan, joint_limits[0, 0], joint_limits[0, 1])
        desired_tilt = np.clip(desired_tilt, joint_limits[1, 0], joint_limits[1, 1])

        return np.array([desired_pan, desired_tilt])

    def handle_pose_to_ee_pose(
        self, handle_position: np.ndarray, handle_rot_mat: np.ndarray
    ) -> tuple:
        """
        Compute an ee pose from a handle pose.
        Adapted from https://github.com/allenai/mujoco-thor/blob/thor-assets/scripts/calibration/ithor_artiuclate_test.py#L552
        Also return the offset distance to be added to the ee pose to be at the edge of the handle (using the handle bbox).
        """
        # Assumption: VISUAL GEOM Z Axis is usually along the handle (front to back). X-axis is up/right. Y is left/up.

        # 1. Compute target Y axis direction
        y_axis = handle_rot_mat @ np.array([0, 1, 0])  # target Y is same as handle Y axis

        # 2. Compute target X axis direction
        handle_neg_z_axis = handle_rot_mat @ np.array([0, 0, -1])
        # Use the handle's negative Z-axis as X-axis reference
        x_axis = handle_neg_z_axis

        # 3. Compute target Z axis direction as orthogonal to X and Y
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        # Form ee rotation matrix with columns: [X, Y, Z]
        ee_rot_mat = np.column_stack((x_axis, y_axis, z_axis))

        # Get the distance by which the pose has to be offset to be at the edge of the handle (using the handle bbox)
        bbox_half_lengths = self.door_object.get_handle_bboxes_array()[
            0
        ][
            3:
        ]  # (x,y,z half-lengths) - AABB is [center, size], so [0][3:] gets the half-sizes of the first handle
        # offset along original z axis since that is now along the x axis (along the handle front to back)
        bbox_offset = -bbox_half_lengths[2]

        # Edge case: if handle is a vertical handle
        # If X axis is pointing up or down, add an in-frame yaw by 90 degrees
        if x_axis[2] > 0.99:
            ee_rot_mat = (
                R.from_matrix(ee_rot_mat) * R.from_euler("Z", 90, degrees=True)
            ).as_matrix()
            # now we are using the original y axis instead as the x axis (along the handle front to back)
            bbox_offset = -bbox_half_lengths[1]
        elif x_axis[2] < -0.99:
            ee_rot_mat = (
                R.from_matrix(ee_rot_mat) * R.from_euler("Z", -90, degrees=True)
            ).as_matrix()
            # now we are using the original y axis instead as the x axis (along the handle front to back)
            bbox_offset = -bbox_half_lengths[1]

        # If we are using the other side handle, flip the ee pose (about Yaw axis)
        if self._use_other_side_handle:
            ee_rot_mat = (
                R.from_matrix(ee_rot_mat) * R.from_euler("Z", 180, degrees=True)
            ).as_matrix()

        ee_pos = handle_position.copy()

        return ee_pos, ee_rot_mat, bbox_offset

    def check_if_use_flip_side_handle(self) -> bool:
        """
        Check if the handle chosen is on the other side of the door, we will then need to flip the target ee poses
        """
        # get current robot base position
        robot_base_pos = self.env.robots[0].get_world_pose_tf_mat()[:3, 3]
        # compute ee_pose for the handle at a small offset distance (5cms) closer from the handle
        offset_distance = -0.05
        target_ee_pose = self.get_target_ee_pose(offset_distance=offset_distance)
        target_ee_pos = target_ee_pose[:3]
        # get dist between target ee pose and robot base position and handle position and robot base position
        dist_ee_pose = np.linalg.norm(target_ee_pos - robot_base_pos)
        dist_handle_pose = np.linalg.norm(self.get_door_handle_position() - robot_base_pos)
        # We have chosen the opposite handle, we need to flip the ee poses during this task episode
        return dist_ee_pose > dist_handle_pose

    def check_if_pushing_door(self) -> bool:
        """
        Check if the door is being pushed or pulled for this task episode.
        """
        # get current robot base position
        robot_base_pos = self.env.robots[0].get_world_pose_tf_mat()[:3, 3]
        # compute ee_pose for the handle at a small articulation (0.1radians) of the door
        articulate_deltas = [0.5]
        target_ee_pose = self.get_target_ee_pose(articulate_deltas=articulate_deltas)
        target_ee_pos = target_ee_pose[:3]
        # get dist between target ee pose and robot base position and handle position and robot base position
        dist_ee_pose = np.linalg.norm(target_ee_pos - robot_base_pos)
        dist_handle_pose = np.linalg.norm(self.get_door_handle_position() - robot_base_pos)
        if dist_ee_pose > dist_handle_pose:
            log.info("Pushing the door")
            return True  # We are pushing the door in this task episode
        else:
            log.info("Pulling the door")
            return False  # We are pulling the door in this task episode
