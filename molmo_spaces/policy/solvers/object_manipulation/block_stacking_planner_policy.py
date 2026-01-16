import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MjThorObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    MoveSegment,
    MoveSequence,
)
from molmo_spaces.tasks.util_samplers.grasp_sampler import TopDownGraspPoseSampler
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

log = logging.getLogger(__name__)


class FrankaTopDownGraspPoseSampler(TopDownGraspPoseSampler):
    """Top-down grasp pose sampler for Franka robot."""

    def __init__(self) -> None:
        # Define a simple top-down orientation: gripper pointing straight down (Z-axis points down)
        # Using Euler angles: rotate 180 degrees around Y-axis to point Z down
        top_down_rotation = R.from_euler("y", 180, degrees=True).as_matrix()
        super().__init__(top_down_rotation)


class BlockStackingPlannerPolicy(BaseObjectManipulationPlannerPolicy):
    def _compute_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view
        print(
            f"[DEBUG block_stacking _compute_trajectory] START: qpos = {robot_view.get_qpos_dict()}"
        )
        target_poses = self._compute_target_poses()
        print(
            f"[DEBUG block_stacking _compute_trajectory] AFTER _compute_target_poses: qpos = {robot_view.get_qpos_dict()}"
        )

        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        return [
            GripperAction(robot_view, True, 0.0),
            MoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                [
                    MoveSegment(
                        name="pregrasp",
                        start_pose=start_ee_pose,
                        end_pose=target_poses["pregrasp"],
                        speed=self.policy_config.speed_fast,
                    ),
                    MoveSegment(
                        name="grasp",
                        start_pose=target_poses["pregrasp"],
                        end_pose=target_poses["grasp"],
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(robot_view, False, self.policy_config.gripper_close_duration),
            MoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                is_holding_object=True,
                move_segments=[
                    MoveSegment(
                        name="lift",
                        start_pose=target_poses["grasp"],
                        end_pose=target_poses["lift"],
                        speed=self.policy_config.speed_slow,
                    ),
                    MoveSegment(
                        name="preplace",
                        start_pose=target_poses["lift"],
                        end_pose=target_poses["preplace"],
                        speed=self.policy_config.speed_fast,
                    ),
                    MoveSegment(
                        name="place",
                        start_pose=target_poses["preplace"],
                        end_pose=target_poses["place"],
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(robot_view, True, self.policy_config.gripper_open_duration),
            MoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                [
                    MoveSegment(
                        name="retreat",
                        start_pose=target_poses["place"],
                        end_pose=target_poses["postplace"],
                        speed=self.policy_config.speed_fast,
                    )
                ],
            ),
        ]

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        task_config = self.config.task_config
        target_poses = {}

        robot_view = self.task.env.current_robot.robot_view

        # Get the two blocks: block_to_pick and block_base (receptacle)
        pickup_obj: MjThorObject = self.task.env.get_object_by_name(task_config.pickup_obj_name)
        base_block: MjThorObject = self.task.env.get_object_by_name(
            task_config.place_receptacle_name
        )

        # Use TopDownGraspPoseSampler to get grasp pose for the block to pick up
        grasp_sampler = FrankaTopDownGraspPoseSampler()
        grasp_sampler.set_target(pickup_obj)
        grasp_pos, grasp_quat = grasp_sampler.sample(robot_view.base.pose)

        # Convert to 4x4 pose matrix
        grasp_pose_world = np.eye(4)
        grasp_pose_world[:3, 3] = grasp_pos
        grasp_pose_world[:3, :3] = R.from_quat(grasp_quat, scalar_first=True).as_matrix()

        log.debug(f"  - Block to pick (p): {pickup_obj.position}")
        log.debug(f"  - Base block (p): {base_block.position}")
        log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")

        print("Checking feasible ik for grasp pose")
        if not self.check_feasible_ik(grasp_pose_world):
            log.debug("  - L IK FAILED for grasp pose!")
            log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            # raise ValueError("IK failed for grasp pose")

        target_poses["grasp"] = grasp_pose_world

        # Pregrasp pose - above the grasp position
        pregrasp_pose = grasp_pose_world.copy()
        pregrasp_pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp_pose[:3, 2]

        print("Checking feasible ik for pregrasp pose")
        if not self.check_feasible_ik(pregrasp_pose):
            log.debug("  - L IK FAILED for pregrasp pose!")
            log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {pregrasp_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            # raise ValueError("IK failed for pregrasp pose")

        target_poses["pregrasp"] = pregrasp_pose

        # Get base block bounding box to calculate place height
        base_block_aabb_center, base_block_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, base_block.object_id
        )
        base_block_top_z = base_block_aabb_center[2] + base_block_aabb_size[2] / 2

        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)

        # Lift pose - above base block
        lift_pose = grasp_pose_world.copy()
        lift_pose[2, 3] = (
            base_block_top_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )

        print("Checking feasible ik for lift pose")
        if not self.check_feasible_ik(lift_pose):
            log.debug("  - L IK FAILED for lift pose!")
            # raise ValueError("IK failed for lift pose")

        target_poses["lift"] = lift_pose

        # Preplace pose - horizontally centered over base block, above its top surface
        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = base_block.position[:2]
        preplace_pose[2, 3] = (
            base_block_top_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )
        # Offset the EE to ensure the pickup object is in the middle of the base block
        preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position

        print("Checking feasible ik for preplace pose")
        if not self.check_feasible_ik(preplace_pose):
            log.error("  - ❌ IK FAILED for preplace pose!")
            current_ee_pose = robot_view.get_move_group(
                robot_view.get_gripper_movegroup_ids()[0]
            ).leaf_frame_to_world
            log.error(f"  - Current EE pose:\n{current_ee_pose}")
            log.error(f"  - Target preplace pose:\n{preplace_pose}")
            log.error(
                f"  - Position distance: {np.linalg.norm(preplace_pose[:3, 3] - current_ee_pose[:3, 3]):.3f}m"
            )
            log.error(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            # raise ValueError("IK failed for preplace pose")
        target_poses["preplace"] = preplace_pose

        # Place pose - on top of base block
        place_pose = preplace_pose.copy()
        place_pose[2, 3] = base_block_top_z + pickup_obj_clearance_offset
        print("Checking feasible ik for place pose")
        if not self.check_feasible_ik(place_pose):
            log.debug("  - L IK FAILED for place pose!")
            # raise ValueError("IK failed for place pose")
        target_poses["place"] = place_pose

        # Postplace pose - retreat from place
        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]
        target_poses["postplace"] = postplace_pose

        # debug
        visualize_poses = True
        if visualize_poses and self.task.viewer is not None:
            self._show_poses(np.stack(list(target_poses.values()), axis=0), style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()
                # Pause and wait for user to examine poses
                # import time
                # log.info("Viewer paused - examine the poses. Press Ctrl+C to continue...")
                # try:
                #     while True:
                #         time.sleep(0.1)
                #         self.task.viewer.sync()
                # except KeyboardInterrupt:
                #     log.info("Continuing...")

        return target_poses
