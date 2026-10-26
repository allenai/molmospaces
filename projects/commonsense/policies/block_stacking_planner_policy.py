import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.tasks.task import BaseMujocoTask
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

    def sample(self, base_pose: np.ndarray, mode: str = "global", **kwargs):
        """Override to accept a 4x4 pose matrix directly (Franka has no .base_pose attr)."""
        from molmo_spaces.utils.linalg_utils import relative_to_global_transform

        pose = relative_to_global_transform(
            self._heuristics_based_on_center_bbox(base_pose), base_pose
        )
        position = pose[:3, 3]
        quaternion = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        return position, quaternion


class BlockStackingPlannerPolicy(BaseObjectManipulationPlannerPolicy):
    """Planner that stacks N blocks in order: block_2 on block_1, block_3 on block_2, etc.

    After each pick-place cycle completes, re-plans for the next block pair
    using live block positions (since blocks move during simulation).
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        self._block_names = config.task_config.block_names
        # Index of the block currently being picked (1-based into block_names)
        # e.g. stack_index=1 means picking block_names[1] onto block_names[0]
        self._stack_index = 1
        super().__init__(config, task)

    def reset(self, reset_retries: bool = True):
        self._stack_index = 1
        super().reset(reset_retries)

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        action = super().get_action(info)

        if action.get("done") and self._stack_index < len(self._block_names) - 1:
            # Current pick-place cycle completed; advance to next block pair
            self._stack_index += 1
            pickup_name = self._block_names[self._stack_index]
            base_name = self._block_names[self._stack_index - 1]
            log.info(
                f"[BLOCK STACKING] Completed stacking up to block {self._stack_index}. "
                f"Now picking '{pickup_name}' to place on '{base_name}'."
            )

            # Re-compute trajectory for the next block pair
            self.action_primitives = self._compute_trajectory()
            self.action_idx = 0
            for ap in self.action_primitives:
                ap.reset()

            # Remove done flag and return noop for this step
            action = self.robot_view.get_noop_ctrl_dict()

        return action

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view

        pickup_name = self._block_names[self._stack_index]
        base_name = self._block_names[self._stack_index - 1]
        log.info(
            f"[BLOCK STACKING] Computing trajectory: pick '{pickup_name}', "
            f"place on '{base_name}' (step {self._stack_index}/{len(self._block_names) - 1})"
        )

        target_poses = self._compute_target_poses(pickup_name, base_name)

        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        return [
            GripperAction(robot_view, True, 0.0),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                [
                    TCPMoveSegment(
                        name="pregrasp",
                        start_pose=start_ee_pose,
                        end_pose=target_poses["pregrasp"],
                        speed=self.policy_config.speed_fast,
                    ),
                    TCPMoveSegment(
                        name="grasp",
                        start_pose=target_poses["pregrasp"],
                        end_pose=target_poses["grasp"],
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(robot_view, False, self.policy_config.gripper_close_duration),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                is_holding_object=True,
                move_segments=[
                    TCPMoveSegment(
                        name="lift",
                        start_pose=target_poses["grasp"],
                        end_pose=target_poses["lift"],
                        speed=self.policy_config.speed_slow,
                    ),
                    TCPMoveSegment(
                        name="preplace",
                        start_pose=target_poses["lift"],
                        end_pose=target_poses["preplace"],
                        speed=self.policy_config.speed_fast,
                    ),
                    TCPMoveSegment(
                        name="place",
                        start_pose=target_poses["preplace"],
                        end_pose=target_poses["place"],
                        speed=self.policy_config.speed_slow,
                    ),
                ],
            ),
            GripperAction(robot_view, True, self.policy_config.gripper_open_duration),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                [
                    TCPMoveSegment(
                        name="retreat",
                        start_pose=target_poses["place"],
                        end_pose=target_poses["postplace"],
                        speed=self.policy_config.speed_fast,
                    )
                ],
            ),
        ]

    def _compute_target_poses(self, pickup_name: str, base_name: str) -> dict[str, np.ndarray]:
        robot_view = self.task.env.current_robot.robot_view

        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(pickup_name)
        base_block: MlSpacesObject = om.get_object_by_name(base_name)

        # Use TopDownGraspPoseSampler to get grasp pose for the block to pick up
        grasp_sampler = FrankaTopDownGraspPoseSampler()
        grasp_sampler.set_target(pickup_obj)
        grasp_pos, grasp_quat = grasp_sampler.sample(robot_view.base.pose)

        # Convert to 4x4 pose matrix
        grasp_pose_world = np.eye(4)
        grasp_pose_world[:3, 3] = grasp_pos
        grasp_pose_world[:3, :3] = R.from_quat(grasp_quat, scalar_first=True).as_matrix()

        log.debug(f"  - Block to pick '{pickup_name}' (p): {pickup_obj.position}")
        log.debug(f"  - Base block '{base_name}' (p): {base_block.position}")
        log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")

        # Pregrasp pose - above the grasp position
        pregrasp_pose = grasp_pose_world.copy()
        pregrasp_pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp_pose[:3, 2]

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

        # Preplace pose - horizontally centered over base block, above its top surface
        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = base_block.position[:2]
        preplace_pose[2, 3] = (
            base_block_top_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )
        # Offset the EE to ensure the pickup object is in the middle of the base block
        preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position

        # Place pose - held block's bottom lands 1cm above the base block's top
        # surface instead of exactly touching it. The extra clearance avoids
        # interpenetration from gripper sag / IK tracking error; the block
        # falls the last cm and settles via physics.
        place_pose = preplace_pose.copy()
        place_pose[2, 3] = (
            base_block_top_z + self.policy_config.place_z_clearance + pickup_obj_clearance_offset
        )

        # Postplace pose - retreat from place
        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        # Check IK feasibility for all poses
        pose_names = ["pregrasp", "grasp", "lift", "preplace", "place", "postplace"]
        poses = [
            pregrasp_pose,
            grasp_pose_world,
            lift_pose,
            preplace_pose,
            place_pose,
            postplace_pose,
        ]
        ik_results = {name: self.check_feasible_ik(pose) for name, pose in zip(pose_names, poses)}
        failed = [name for name, ok in ik_results.items() if not ok]

        if failed:
            log.warning(
                f"IK FAILED for: {', '.join(failed)}\n"
                f"  Picking '{pickup_name}' -> placing on '{base_name}'\n"
                f"  Pregrasp pos:  {pregrasp_pose[:3, 3]}\n"
                f"  Grasp pos:     {grasp_pose_world[:3, 3]}\n"
                f"  Lift pos:      {lift_pose[:3, 3]}\n"
                f"  Preplace pos:  {preplace_pose[:3, 3]}\n"
                f"  Place pos:     {place_pose[:3, 3]}\n"
                f"  Postplace pos: {postplace_pose[:3, 3]}\n"
                f"  Robot base:    {robot_view.base.pose[:3, 3]}\n"
                f"  Height diffs: pregrasp={pregrasp_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"grasp={grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"lift={lift_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"preplace={preplace_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"place={place_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"postplace={postplace_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )

            if self.task.viewer is not None:
                pose_colors = {
                    "pregrasp": (0, 1, 0, 1),  # green
                    "grasp": (1, 0, 0, 1),  # red
                    "lift": (0, 0, 1, 1),  # blue
                    "preplace": (1, 1, 0, 1),  # yellow
                    "place": (1, 0, 1, 1),  # magenta
                    "postplace": (0, 1, 1, 1),  # cyan
                }
                for name, pose in zip(pose_names, poses):
                    self._show_poses(np.array([pose]), style="tcp", color=pose_colors[name])
                self.task.viewer.sync()

            raise ValueError(f"IK failed for {', '.join(failed)} pose(s)")

        target_poses = {}
        target_poses["pregrasp"] = pregrasp_pose
        target_poses["grasp"] = grasp_pose_world
        target_poses["lift"] = lift_pose
        target_poses["preplace"] = preplace_pose
        target_poses["place"] = place_pose
        target_poses["postplace"] = postplace_pose

        if self.task.viewer is not None:
            self._show_poses(np.stack(list(target_poses.values()), axis=0), style="tcp")
            self.task.viewer.sync()

        return target_poses
