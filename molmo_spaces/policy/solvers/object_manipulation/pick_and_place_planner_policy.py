import logging
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    JointMoveSegment,
    JointMoveSequence,
    NoopAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.utils.grasp_sample import (
    compute_grasp_pose,
    get_all_grasp_poses,
    get_noncolliding_grasp_mask,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class HoldCurrentControlAction(ActionPrimitive):
    """Hold the current commanded controls for a fixed duration."""

    def __init__(self, robot_view, duration: float) -> None:
        super().__init__(robot_view, duration)
        self._action = None

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
        return self.elapsed_time() >= self.duration

    def get_current_action(self) -> dict:
        if self._action is None:
            self._action = self.robot_view.get_ctrl_dict()
        return self._action


class PickAndPlacePlannerPolicy(BaseObjectManipulationPlannerPolicy):
    def _compute_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view
        target_poses = self._compute_target_poses()

        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world

        return [
            GripperAction(robot_view, True, 0.0),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
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
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
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
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
                    TCPMoveSegment(
                        name="retreat",
                        start_pose=target_poses["place"],
                        end_pose=target_poses["postplace"],
                        speed=self.policy_config.speed_fast,
                    )
                ],
            ),
            JointMoveSequence(
                robot_view,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                move_segments=[
                    JointMoveSegment(
                        name="go_home",
                        start_qpos=None,
                        end_qpos=self.config.robot_config.init_qpos,
                        duration_s=4.0,
                    )
                ],
            ),
            HoldCurrentControlAction(robot_view, 2.0),
        ]

    def _get_grasp_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
        robot_view,
        task_config: PickAndPlaceTaskConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        randomize_pregrasp = False
        if randomize_pregrasp:
            # Random height variations
            pregrasp_height_offset = np.random.uniform(
                -self.policy_config.pregrasp_height_noise,
                self.policy_config.pregrasp_height_noise,
            )
            postgrasp_height_offset = np.random.uniform(
                -self.policy_config.postgrasp_height_noise,
                self.policy_config.postgrasp_height_noise,
            )
        else:
            pregrasp_height_offset = 0.0
            postgrasp_height_offset = 0.0

        pregrasp_pose = grasp_pose_world.copy()
        # Pregrasp pose - above the pickup object with randomization
        pregrasp_pose[:3, 3] -= (
            self.policy_config.pregrasp_z_offset + pregrasp_height_offset
        ) * pregrasp_pose[:3, 2]

        log.debug(f"  - obj_start (p): {pickup_obj.position}")
        log.debug(f"  - obj_start (t): {task_config.pickup_obj_start_pose}")
        log.debug(f"  - obj_end (t): {task_config.pickup_obj_goal_pose}")
        log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")

        if not self.check_feasible_ik(pregrasp_pose):
            log.debug("  - ❌ IK FAILED for pregrasp pose!")
            log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {pregrasp_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            raise ValueError("IK failed for pregrasp pose")

        log.debug(f"  - Grasp pose position: {grasp_pose_world[:3, 3]}")
        log.debug(
            f"  - Grasp height above robot base: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
        )
        if not self.check_feasible_ik(grasp_pose_world):
            log.debug("  - ❌ IK FAILED for grasp pose!")
            log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            raise ValueError("IK failed for grasp pose")

        # Lift pose - above grasp position
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)
        lift_pose = grasp_pose_world.copy()
        lift_pose[2, 3] = (
            receptacle_top_z
            + pickup_obj_clearance_offset
            + self.policy_config.place_z_offset
            + postgrasp_height_offset
        )

        if not self.check_feasible_ik(lift_pose):
            log.debug("  - ❌ IK FAILED for lift pose!")
            raise ValueError("IK failed for lift pose")

        return pregrasp_pose, grasp_pose_world, lift_pose

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)

        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = place_receptacle.position[:2]
        preplace_pose[2, 3] = (
            receptacle_top_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )
        # offset the EE to ensure the pickup object is in the middle of the receptacle
        preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position
        if not self.check_feasible_ik(preplace_pose):
            log.debug("  - ❌ IK FAILED for preplace pose!")
            raise ValueError("IK failed for preplace pose")

        place_pose = preplace_pose.copy()
        place_pose[2, 3] = receptacle_top_z + pickup_obj_clearance_offset
        if not self.check_feasible_ik(place_pose):
            log.debug("  - ❌ IK FAILED for place pose!")
            raise ValueError("IK failed for place pose")

        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        return preplace_pose, place_pose, postplace_pose

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        task_config = self.config.task_config
        assert isinstance(task_config, PickAndPlaceTaskConfig)
        target_poses = {}

        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle: MlSpacesObject = om.get_object_by_name(task_config.place_receptacle_name)

        grasp_pose_world = compute_grasp_pose(
            self,
            pickup_obj,
            robot_view,
            check_collision=self.policy_config.filter_colliding_grasps,
            n_collision_checks=self.policy_config.grasp_collision_max_grasps,
            collision_batch_size=self.policy_config.grasp_collision_batch_size,
            check_ik=self.policy_config.filter_feasible_grasps,
            n_ik_checks=self.policy_config.grasp_feasibility_max_grasps,
            ik_batch_size=self.policy_config.grasp_feasibility_batch_size,
            pos_cost_weight=self.policy_config.grasp_pos_cost_weight,
            rot_cost_weight=self.policy_config.grasp_rot_cost_weight,
            vertical_cost_weight=self.policy_config.grasp_vertical_cost_weight,
            com_dist_cost_weight=self.policy_config.grasp_com_dist_cost_weight,
        )

        pregrasp_pose, grasp_pose, lift_pose = self._get_grasp_poses(
            grasp_pose_world=grasp_pose_world,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
            robot_view=robot_view,
            task_config=task_config,
        )
        target_poses["pregrasp"] = pregrasp_pose
        target_poses["grasp"] = grasp_pose
        target_poses["lift"] = lift_pose

        preplace_pose, place_pose, postplace_pose = self._get_placement_poses(
            grasp_pose_world=grasp_pose_world,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
        )
        target_poses["preplace"] = preplace_pose
        target_poses["place"] = place_pose
        target_poses["postplace"] = postplace_pose

        if self.policy_config.debug_poses and self.task.viewer is not None:
            self._show_poses(np.stack(list(target_poses.values()), axis=0), style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()

        return target_poses


class UnitreeG1RightArmPickAndPlacePlannerPolicy(PickAndPlacePlannerPolicy):
    """Pick-and-place planner variant that solves IK through the G1 right arm only."""

    def reset(self, reset_retries: bool = True):
        self._g1_failure_diag_steps: list[dict] = []
        self._g1_last_target_pose: np.ndarray | None = None
        self._g1_initial_tcp_pose = self._current_g1_tcp_pose()
        self._g1_failure_diagnostics_logged = False
        self._g1_failure_reason_override: str | None = None
        self._g1_failure_phase_override: str | None = None
        self._g1_bypass_feasible_ik = False
        self._g1_unfiltered_grasp_fallback_used = False
        self._g1_unfiltered_grasp_fallback_reason: str | None = None
        self._g1_selected_grasp_debug: dict | None = None
        self._g1_grasp_selector_debug: dict | None = None
        self._g1_last_ik_debug: dict | None = None
        self._g1_failure_hold_action: NoopAction | None = None
        self._g1_terminal_failure_pending = False
        self._g1_grip_diag_samples: list[dict] = []
        self._g1_grip_diag_last_log_time = -np.inf
        self._g1_grip_diag_last_key: tuple | None = None
        super().reset(reset_retries=reset_retries)
        if self._g1_failure_reason_override:
            self._populate_failed_trajectory_target_poses()

    def _record_g1_selected_grasp_debug(self, grasp_debug: dict) -> None:
        self._g1_selected_grasp_debug = grasp_debug

    def _populate_failed_trajectory_target_poses(self) -> None:
        tcp_pose = self._current_g1_tcp_pose()
        self.target_poses.update(
            {
                "pregrasp": tcp_pose,
                "grasp": tcp_pose,
                "lift": tcp_pose,
                "preplace": tcp_pose,
                "place": tcp_pose,
                "postplace": tcp_pose,
            }
        )

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        try:
            if getattr(self.policy_config, "g1_pick_lift_only", False):
                return self._compute_g1_pick_lift_only_trajectory()
            return self._compute_g1_pick_and_place_trajectory()
        except ValueError as exc:
            reason = self._classify_trajectory_build_failure(exc)
            if reason is None:
                raise
            self._g1_failure_reason_override = reason
            self._g1_failure_phase_override = "grasp_selection"
            log.info("[G1_DIAG] trajectory_build_failure=%s error=%r", reason, str(exc))
            return [
                NoopAction(
                    self.robot_view,
                    getattr(self.policy_config, "diagnostic_failure_hold_duration_s", 2.0),
                )
            ]

    def _g1_postgrasp_hold_action(self, robot_view) -> HoldCurrentControlAction | None:
        hold_duration = float(getattr(self.policy_config, "g1_postgrasp_hold_duration", 0.0))
        if hold_duration <= 0.0:
            return None
        return HoldCurrentControlAction(robot_view, hold_duration)

    def _compute_g1_pick_and_place_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view
        target_poses = self._compute_target_poses()
        held_object_speed = float(
            getattr(self.policy_config, "g1_held_object_speed", self.policy_config.speed_slow)
        )

        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world

        postgrasp_hold = self._g1_postgrasp_hold_action(robot_view)
        trajectory: list[ActionPrimitive] = [
            GripperAction(robot_view, True, 0.0),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
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
        ]
        if postgrasp_hold is not None:
            trajectory.append(postgrasp_hold)
        trajectory.extend(
            [
                TCPMoveSequence(
                    robot_view,
                    self._tcp_to_jp_fn,
                    self.policy_config.move_settle_time,
                    is_holding_object=True,
                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                    tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                    tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                    move_segments=[
                        TCPMoveSegment(
                            name="lift",
                            start_pose=target_poses["grasp"],
                            end_pose=target_poses["lift"],
                            speed=held_object_speed,
                        ),
                        TCPMoveSegment(
                            name="preplace",
                            start_pose=target_poses["lift"],
                            end_pose=target_poses["preplace"],
                            speed=held_object_speed,
                        ),
                        TCPMoveSegment(
                            name="place",
                            start_pose=target_poses["preplace"],
                            end_pose=target_poses["place"],
                            speed=held_object_speed,
                        ),
                    ],
                ),
                GripperAction(robot_view, True, self.policy_config.gripper_open_duration),
                TCPMoveSequence(
                    robot_view,
                    self._tcp_to_jp_fn,
                    self.policy_config.move_settle_time,
                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                    tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                    tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                    move_segments=[
                        TCPMoveSegment(
                            name="retreat",
                            start_pose=target_poses["place"],
                            end_pose=target_poses["postplace"],
                            speed=self.policy_config.speed_fast,
                        )
                    ],
                ),
                JointMoveSequence(
                    robot_view,
                    self.policy_config.move_settle_time,
                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                    move_segments=[
                        JointMoveSegment(
                            name="go_home",
                            start_qpos=None,
                            end_qpos=self.config.robot_config.init_qpos,
                            duration_s=4.0,
                        )
                    ],
                ),
                HoldCurrentControlAction(robot_view, 2.0),
            ]
        )
        return trajectory

    def _compute_g1_pick_lift_only_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view
        target_poses = self._compute_target_poses()
        held_object_speed = float(
            getattr(self.policy_config, "g1_held_object_speed", self.policy_config.speed_slow)
        )

        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world

        postgrasp_hold = self._g1_postgrasp_hold_action(robot_view)
        trajectory: list[ActionPrimitive] = [
            GripperAction(robot_view, True, 0.0),
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                move_segments=[
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
        ]
        if postgrasp_hold is not None:
            trajectory.append(postgrasp_hold)
        trajectory.extend(
            [
                TCPMoveSequence(
                    robot_view,
                    self._tcp_to_jp_fn,
                    self.policy_config.move_settle_time,
                    is_holding_object=True,
                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                    tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                    tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                    move_segments=[
                        TCPMoveSegment(
                            name="lift",
                            start_pose=target_poses["grasp"],
                            end_pose=target_poses["lift"],
                            speed=held_object_speed,
                        )
                    ],
                ),
                HoldCurrentControlAction(robot_view, 2.0),
            ]
        )
        return trajectory

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        if getattr(self.policy_config, "g1_online_grasp_selector", False):
            return self._compute_g1_online_grasp_target_poses()

        try:
            return super()._compute_target_poses()
        except ValueError as exc:
            if (
                str(exc) != "No feasible grasp found"
                or not self.policy_config.record_unfiltered_attempt_on_no_feasible_grasp
            ):
                raise

            self._g1_unfiltered_grasp_fallback_used = True
            self._g1_unfiltered_grasp_fallback_reason = "no_feasible_g1_grasp"
            log.info(
                "[G1_DIAG] no_feasible_g1_grasp; recording one unfiltered grasp attempt"
            )
            self._g1_bypass_feasible_ik = True
            try:
                return super()._compute_target_poses()
            finally:
                self._g1_bypass_feasible_ik = False

    def _classify_trajectory_build_failure(self, exc: ValueError) -> str | None:
        message = str(exc)
        if message == "No feasible grasp found":
            return "no_feasible_g1_grasp"
        if message == "No G1 tabletop grasp candidate found":
            return "no_g1_tabletop_grasp_candidate"
        if message == "IK failed for preplace pose":
            return "no_feasible_g1_preplace"
        if message == "IK failed for place pose":
            return "no_feasible_g1_place"
        return None

    def _compute_g1_online_grasp_target_poses(self) -> dict[str, np.ndarray]:
        task_config = self.config.task_config
        assert isinstance(task_config, PickAndPlaceTaskConfig)

        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle: MlSpacesObject = om.get_object_by_name(task_config.place_receptacle_name)

        grasp_pose_world = self._select_g1_tabletop_grasp_pose(
            pickup_obj,
            place_receptacle,
            robot_view,
        )
        target_poses = self._build_g1_candidate_target_poses(
            grasp_pose_world,
            pickup_obj,
            place_receptacle,
        )

        if self.policy_config.debug_poses and self.task.viewer is not None:
            self._show_poses(np.stack(list(target_poses.values()), axis=0), style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()

        return target_poses

    def _build_g1_candidate_target_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> dict[str, np.ndarray]:
        grasp_pose_world = grasp_pose_world.copy()
        grasp_pose_world = self._g1_apply_grasp_inward_offset(
            grasp_pose_world,
            pickup_obj,
        )
        grasp_pose_world = self._g1_level_grasp_orientation(grasp_pose_world)
        table_top_z = self._g1_tabletop_top_z()
        min_grasp_z = table_top_z + float(
            getattr(self.policy_config, "g1_grasp_table_clearance", 0.0)
        )
        if np.isfinite(min_grasp_z):
            grasp_pose_world[2, 3] = max(grasp_pose_world[2, 3], min_grasp_z)
        grasp_pose_world = self._g1_center_grasp_lateral(
            grasp_pose_world,
            pickup_obj,
        )

        pregrasp_pose = grasp_pose_world.copy()
        pregrasp_pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp_pose[:3, 2]

        model = self.task.env.current_data.model
        data = self.task.env.current_data
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            model,
            data,
            place_receptacle.object_id,
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            model,
            data,
            pickup_obj.object_id,
        )
        pickup_obj_top_z = pickup_obj_aabb_center[2] + pickup_obj_aabb_size[2] / 2
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)
        min_pregrasp_z = max(
            pregrasp_pose[2, 3],
            grasp_pose_world[2, 3]
            + float(getattr(self.policy_config, "g1_pregrasp_min_vertical_lift", 0.0)),
            pickup_obj_top_z
            + float(getattr(self.policy_config, "g1_pregrasp_object_clearance", 0.0)),
        )
        pregrasp_pose[2, 3] = min_pregrasp_z

        lift_pose = grasp_pose_world.copy()
        lift_pose[2, 3] = (
            receptacle_top_z
            + pickup_obj_clearance_offset
            + self.policy_config.place_z_offset
        )
        lift_pose = self._g1_raise_pose_for_carried_object_bottom_clearance(
            lift_pose,
            grasp_pose_world,
            pickup_obj_bottom_z,
            receptacle_top_z,
        )

        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = place_receptacle.position[:2]
        preplace_pose[2, 3] = (
            receptacle_top_z
            + pickup_obj_clearance_offset
            + self.policy_config.place_z_offset
        )
        preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position
        preplace_pose = self._g1_raise_pose_for_carried_object_bottom_clearance(
            preplace_pose,
            grasp_pose_world,
            pickup_obj_bottom_z,
            receptacle_top_z,
        )

        place_pose = preplace_pose.copy()
        place_pose[2, 3] = receptacle_top_z + pickup_obj_clearance_offset

        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        return {
            "pregrasp": pregrasp_pose,
            "grasp": grasp_pose_world,
            "lift": lift_pose,
            "preplace": preplace_pose,
            "place": place_pose,
            "postplace": postplace_pose,
        }

    def _g1_raise_pose_for_carried_object_bottom_clearance(
        self,
        pose: np.ndarray,
        grasp_pose_world: np.ndarray,
        pickup_obj_bottom_z: float,
        receptacle_top_z: float,
    ) -> np.ndarray:
        clearance = float(getattr(self.policy_config, "g1_place_travel_object_clearance", 0.0))
        if clearance <= 0.0:
            return pose

        raised_pose = pose.copy()
        carried_bottom_z = raised_pose[2, 3] + pickup_obj_bottom_z - grasp_pose_world[2, 3]
        min_bottom_z = receptacle_top_z + clearance
        if carried_bottom_z < min_bottom_z:
            raised_pose[2, 3] += min_bottom_z - carried_bottom_z
        return raised_pose

    def _g1_center_grasp_lateral(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
    ) -> np.ndarray:
        if not getattr(self.policy_config, "g1_center_grasp_lateral", False):
            return grasp_pose_world

        pose = grasp_pose_world.copy()
        object_pos = np.asarray(pickup_obj.position[:3], dtype=float)
        object_in_grasp = pose[:3, :3].T @ (object_pos - pose[:3, 3])
        lateral_offset = float(object_in_grasp[1])
        lateral_offset *= float(
            getattr(self.policy_config, "g1_grasp_lateral_centering_scale", 1.0)
        )
        max_offset = float(
            getattr(self.policy_config, "g1_grasp_lateral_centering_max_offset", np.inf)
        )
        if np.isfinite(max_offset):
            lateral_offset = float(np.clip(lateral_offset, -max_offset, max_offset))

        pose[:3, 3] += lateral_offset * pose[:3, 1]
        return pose

    def _g1_level_grasp_orientation(self, grasp_pose_world: np.ndarray) -> np.ndarray:
        if not getattr(self.policy_config, "g1_level_grasp_orientation", False):
            return grasp_pose_world

        pose = grasp_pose_world.copy()
        z_axis = pose[:3, 2]
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm <= 1e-6:
            return pose

        z_axis = z_axis / z_norm
        vertical_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(z_axis[2]), -1.0, 1.0))))
        max_tilt_deg = float(
            getattr(self.policy_config, "g1_grasp_level_max_tilt_deg", np.inf)
        )
        if np.isfinite(max_tilt_deg) and vertical_tilt_deg > max_tilt_deg:
            return pose

        target_z = np.array([0.0, 0.0, 1.0 if z_axis[2] >= 0.0 else -1.0])
        x_axis = pose[:3, 0]
        x_axis = x_axis - target_z * float(np.dot(x_axis, target_z))
        if np.linalg.norm(x_axis) <= 1e-6:
            x_axis = pose[:3, 1] - target_z * float(np.dot(pose[:3, 1], target_z))
        if np.linalg.norm(x_axis) <= 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(target_z, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        pose[:3, :3] = np.column_stack([x_axis, y_axis, target_z])
        return pose

    def _g1_apply_grasp_inward_offset(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
    ) -> np.ndarray:
        inward_offset = float(
            getattr(self.policy_config, "g1_grasp_inward_xy_offset", 0.0)
        )
        if inward_offset <= 0.0:
            return grasp_pose_world

        offset_pose = grasp_pose_world.copy()
        object_xy = np.asarray(pickup_obj.position[:2], dtype=float)
        grasp_xy = offset_pose[:2, 3]
        object_delta_xy = object_xy - grasp_xy
        object_dist_xy = float(np.linalg.norm(object_delta_xy))
        if object_dist_xy <= 1e-6:
            return offset_pose

        offset_pose[:2, 3] += object_delta_xy / object_dist_xy * min(
            inward_offset,
            object_dist_xy,
        )
        return offset_pose

    def _g1_tabletop_top_z(self) -> float:
        task_sampler_config = getattr(getattr(self, "config", None), "task_sampler_config", None)
        table_body_name = getattr(task_sampler_config, "table_body_name", None)
        if not table_body_name:
            return -np.inf

        model = self.task.env.current_data.model
        data = self.task.env.current_data
        try:
            table_body_id = model.body(table_body_name).id
        except KeyError:
            return -np.inf

        table_center, table_size = body_aabb(model, data, table_body_id, visual_only=False)
        return float(table_center[2] + table_size[2] / 2)

    def _g1_pregrasp_debug(
        self,
        pregrasp_pose: np.ndarray,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
    ) -> dict[str, float]:
        model = self.task.env.current_data.model
        data = self.task.env.current_data
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            model,
            data,
            pickup_obj.object_id,
        )
        pickup_obj_top_z = pickup_obj_aabb_center[2] + pickup_obj_aabb_size[2] / 2
        object_in_grasp = grasp_pose_world[:3, :3].T @ (
            np.asarray(pickup_obj.position[:3], dtype=float) - grasp_pose_world[:3, 3]
        )
        raw_grasp_pose = getattr(self, "_g1_selected_raw_grasp_pose", None)
        table_top_z = self._g1_tabletop_top_z()
        return {
            "object_top_z": float(pickup_obj_top_z),
            "table_top_z": float(table_top_z),
            "raw_grasp_z": float(raw_grasp_pose[2, 3]) if raw_grasp_pose is not None else None,
            "grasp_z": float(grasp_pose_world[2, 3]),
            "grasp_vertical_axis_z": float(grasp_pose_world[2, 2]),
            "grasp_vertical_tilt_deg": float(
                np.degrees(np.arccos(np.clip(abs(grasp_pose_world[2, 2]), -1.0, 1.0)))
            ),
            "object_local_y_in_grasp_m": float(object_in_grasp[1]),
            "grasp_table_clearance_m": float(grasp_pose_world[2, 3] - table_top_z)
            if np.isfinite(table_top_z)
            else None,
            "pregrasp_z": float(pregrasp_pose[2, 3]),
            "pregrasp_above_grasp_m": float(pregrasp_pose[2, 3] - grasp_pose_world[2, 3]),
            "pregrasp_object_clearance_m": float(pregrasp_pose[2, 3] - pickup_obj_top_z),
        }

    def _copy_qpos_dict(self, qpos: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {move_group_id: np.asarray(values).copy() for move_group_id, values in qpos.items()}

    def _g1_eval_candidate_target_poses(
        self,
        target_poses: dict[str, np.ndarray],
        initial_qpos: dict[str, np.ndarray] | None = None,
        base_pose: np.ndarray | None = None,
    ) -> dict:
        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        phase_order = (
            ["pregrasp", "grasp", "lift", "preplace", "place"]
            if getattr(self.policy_config, "g1_grasp_require_all_pick_place_phases", True)
            else ["pregrasp", "grasp", "lift"]
        )

        live_qpos = self._copy_qpos_dict(robot_view.get_qpos_dict())
        live_base_pose = robot_view.base.pose.copy()
        qpos = self._copy_qpos_dict(initial_qpos or live_qpos)
        if base_pose is None:
            base_pose = live_base_pose

        phase_results = {}
        try:
            for phase in phase_order:
                result = kinematics.diagnose_ik(
                    gripper_mg_id,
                    target_poses[phase],
                    ["right_arm"],
                    qpos,
                    base_pose,
                    max_iter=250,
                )
                phase_results[phase] = result
                if not result["success"]:
                    return {
                        "success": False,
                        "failed_phase": phase,
                        "phase_results": phase_results,
                    }
                qpos = self._copy_qpos_dict(result["qpos"])

            return {
                "success": True,
                "failed_phase": None,
                "phase_results": phase_results,
                "final_qpos": qpos,
            }
        finally:
            robot_view.base.pose = live_base_pose
            robot_view.set_qpos_dict(live_qpos)

    def _g1_contact_quality_from_contacts(self, contacts: list[dict]) -> dict:
        robot_namespace = self.config.robot_config.robot_namespace
        object_contacts = [
            contact
            for contact in contacts
            if contact["class"] == "robot_pickup_object"
        ]
        pad_contacts = [
            contact
            for contact in object_contacts
            if self._g1_has_right_fingertip_pad_contact(contact)
        ]
        non_pad_object_contacts = [
            contact
            for contact in object_contacts
            if not self._g1_has_right_fingertip_pad_contact(contact)
        ]
        finger_link_object_contacts = [
            contact
            for contact in non_pad_object_contacts
            if any(
                body_name.startswith(f"{robot_namespace}right_dex1_finger_link")
                for _geom_name, body_name in self._g1_robot_contact_endpoints(contact)
            )
        ]
        other_robot_object_contacts = [
            contact
            for contact in non_pad_object_contacts
            if contact not in finger_link_object_contacts
        ]
        table_contacts = [
            contact
            for contact in contacts
            if contact["class"] == "robot_table"
        ]
        bin_contacts = [
            contact
            for contact in contacts
            if contact["class"] == "robot_bin"
        ]
        pad_geom_names = sorted(
            {
                geom_name
                for contact in pad_contacts
                for geom_name, _body_name in self._g1_robot_contact_endpoints(contact)
                if "right_dex1_fingertip_pad" in geom_name
            }
        )
        return {
            "object_contact_count": len(object_contacts),
            "pad_contact_count": len(pad_contacts),
            "pad_geom_count": len(pad_geom_names),
            "pad_geom_names": pad_geom_names,
            "non_pad_object_contacts": non_pad_object_contacts,
            "finger_link_object_contacts": finger_link_object_contacts,
            "other_robot_object_contacts": other_robot_object_contacts,
            "table_contacts": table_contacts,
            "bin_contacts": bin_contacts,
        }

    def _g1_grip_diagnostics(self, contacts: list[dict] | None = None) -> dict:
        robot_view = self.task.env.current_robot.robot_view
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        gripper = robot_view.get_gripper(gripper_mg_id)
        data = self.task.env.current_data
        model = self.task.env.current_model
        actuator_ids = np.asarray(getattr(gripper, "_actuator_ids", []), dtype=int)
        actuator_force = (
            data.actuator_force[actuator_ids].copy()
            if actuator_ids.size and hasattr(data, "actuator_force")
            else np.array([])
        )
        actuator_range = (
            model.actuator_forcerange[actuator_ids].copy()
            if actuator_ids.size
            else np.empty((0, 2))
        )
        force_limit = (
            np.maximum(np.abs(actuator_range[:, 0]), np.abs(actuator_range[:, 1]))
            if actuator_range.size
            else np.array([])
        )
        force_fraction = (
            np.divide(
                np.abs(actuator_force),
                np.maximum(force_limit, 1e-9),
                out=np.zeros_like(actuator_force, dtype=float),
                where=force_limit > 0.0,
            )
            if actuator_force.size
            else np.array([])
        )
        contacts = self._collect_contact_diagnostics() if contacts is None else contacts
        contact_quality = self._g1_contact_quality_from_contacts(contacts)
        ctrl = gripper.ctrl.copy()
        joint_pos = gripper.joint_pos.copy()
        return {
            "sim_time": float(data.time),
            "phase": self.get_phase(),
            "ctrl": np.round(ctrl, 5).tolist(),
            "joint_pos": np.round(joint_pos, 5).tolist(),
            "ctrl_minus_joint_pos": np.round(ctrl - joint_pos, 5).tolist(),
            "inter_finger_dist": float(gripper.inter_finger_dist),
            "actuator_force": np.round(actuator_force, 3).tolist(),
            "actuator_force_fraction": np.round(force_fraction, 3).tolist(),
            "pad_geom_count": int(contact_quality["pad_geom_count"]),
            "pad_geom_names": contact_quality["pad_geom_names"],
            "pad_contact_count": int(contact_quality["pad_contact_count"]),
            "finger_link_contact_count": len(contact_quality["finger_link_object_contacts"]),
            "other_robot_object_contact_count": len(contact_quality["other_robot_object_contacts"]),
            "table_contact_count": len(contact_quality["table_contacts"]),
            "bin_contact_count": len(contact_quality["bin_contacts"]),
        }

    def _record_g1_grip_diagnostic_sample(self) -> None:
        if not self._failure_diagnostics_enabled():
            return
        phase = self.get_phase()
        if phase not in {"gripper-close", "lift", "preplace", "place"}:
            return

        sample = self._g1_grip_diagnostics()
        self._g1_grip_diag_samples.append(sample)
        self._g1_grip_diag_samples = self._g1_grip_diag_samples[-240:]

        log_key = (
            sample["phase"],
            sample["pad_geom_count"],
            sample["finger_link_contact_count"],
            sample["other_robot_object_contact_count"],
            sample["bin_contact_count"],
        )
        log_interval_s = 0.5
        should_log = (
            log_key != self._g1_grip_diag_last_key
            or sample["sim_time"] - self._g1_grip_diag_last_log_time >= log_interval_s
        )
        if not should_log:
            return

        self._g1_grip_diag_last_key = log_key
        self._g1_grip_diag_last_log_time = sample["sim_time"]
        log.info(
            "[G1_GRIP] phase=%s ctrl=%s q=%s ctrl_minus_q=%s dist=%.5f "
            "force=%s force_frac=%s pads=%s finger_links=%s other=%s bin=%s",
            sample["phase"],
            sample["ctrl"],
            sample["joint_pos"],
            sample["ctrl_minus_joint_pos"],
            sample["inter_finger_dist"],
            sample["actuator_force"],
            sample["actuator_force_fraction"],
            sample["pad_geom_names"],
            sample["finger_link_contact_count"],
            sample["other_robot_object_contact_count"],
            sample["bin_contact_count"],
        )

    def _g1_open_grasp_contact_quality(
        self,
        qpos: dict[str, np.ndarray],
        base_pose: np.ndarray,
    ) -> dict:
        robot_view = self.task.env.current_robot.robot_view
        live_qpos = self._copy_qpos_dict(robot_view.get_qpos_dict())
        live_base_pose = robot_view.base.pose.copy()
        model = self.task.env.current_model
        data = self.task.env.current_data

        try:
            robot_view.base.pose = base_pose
            robot_view.set_qpos_dict(qpos)
            mujoco.mj_forward(model, data)
            contacts = self._collect_contact_diagnostics()
        finally:
            robot_view.base.pose = live_base_pose
            robot_view.set_qpos_dict(live_qpos)
            mujoco.mj_forward(model, data)

        quality = self._g1_contact_quality_from_contacts(contacts)
        quality.update(
            {
                "closed_probe": False,
                "object_shift_m": 0.0,
                "failure_reason": self._g1_grasp_contact_quality_failure(quality),
            }
        )
        return quality

    def _g1_mujoco_state_snapshot(self) -> dict:
        data = self.task.env.current_data
        return {
            "time": float(data.time),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "qacc": data.qacc.copy(),
            "qacc_warmstart": data.qacc_warmstart.copy(),
            "ctrl": data.ctrl.copy(),
            "act": data.act.copy(),
            "mocap_pos": data.mocap_pos.copy(),
            "mocap_quat": data.mocap_quat.copy(),
        }

    def _g1_restore_mujoco_state(self, snapshot: dict) -> None:
        model = self.task.env.current_model
        data = self.task.env.current_data
        data.time = snapshot["time"]
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.qacc[:] = snapshot["qacc"]
        data.qacc_warmstart[:] = snapshot["qacc_warmstart"]
        data.ctrl[:] = snapshot["ctrl"]
        data.act[:] = snapshot["act"]
        data.mocap_pos[:] = snapshot["mocap_pos"]
        data.mocap_quat[:] = snapshot["mocap_quat"]
        mujoco.mj_forward(model, data)

    def _g1_closed_grasp_contact_quality(
        self,
        qpos: dict[str, np.ndarray],
        base_pose: np.ndarray,
        pickup_obj: MlSpacesObject,
    ) -> dict:
        robot_view = self.task.env.current_robot.robot_view
        model = self.task.env.current_model
        data = self.task.env.current_data
        snapshot = self._g1_mujoco_state_snapshot()
        settle_steps = max(
            int(getattr(self.policy_config, "g1_closed_grasp_settle_steps", 25)),
            0,
        )

        try:
            robot_view.base.pose = base_pose
            robot_view.set_qpos_dict(qpos)
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            gripper = robot_view.get_gripper(gripper_mg_id)
            gripper.set_gripper_ctrl_open(False)
            gripper.joint_pos = gripper.ctrl.copy()
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            data.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(model, data)
            object_start_pos = pickup_obj.position.copy()
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            object_shift_m = float(np.linalg.norm(pickup_obj.position - object_start_pos))
            contacts = self._collect_contact_diagnostics()
            quality = self._g1_contact_quality_from_contacts(contacts)
            quality.update(
                {
                    "closed_probe": True,
                    "object_shift_m": object_shift_m,
                    "settle_steps": settle_steps,
                    "failure_reason": self._g1_closed_grasp_contact_quality_failure(
                        quality,
                        object_shift_m,
                    ),
                }
            )
            return quality
        finally:
            self._g1_restore_mujoco_state(snapshot)

    def _g1_candidate_grasp_contact_quality(
        self,
        qpos: dict[str, np.ndarray],
        base_pose: np.ndarray,
        pickup_obj: MlSpacesObject,
    ) -> dict:
        open_quality = self._g1_open_grasp_contact_quality(qpos, base_pose)
        if open_quality["failure_reason"]:
            return open_quality
        if getattr(self.policy_config, "g1_closed_grasp_quality_enabled", True):
            return self._g1_closed_grasp_contact_quality(qpos, base_pose, pickup_obj)
        return open_quality

    def _g1_grasp_contact_quality_failure(self, quality: dict) -> str | None:
        if (
            getattr(self.policy_config, "g1_reject_open_grasp_object_contact", True)
            and quality["object_contact_count"] > 0
        ):
            return "open_grasp_object_contact"
        if (
            getattr(self.policy_config, "g1_reject_grasp_table_contact", True)
            and quality["table_contacts"]
        ):
            return "table_contact"
        if (
            getattr(self.policy_config, "g1_reject_non_fingertip_grasp_object_contact", True)
            and quality["non_pad_object_contacts"]
        ):
            return "non_fingertip_object_contact"
        if (
            getattr(self.policy_config, "g1_require_fingertip_pad_grasp_contact", True)
            and quality["pad_contact_count"] == 0
        ):
            return "missing_fingertip_pad_contact"
        return None

    def _g1_grasp_quality_rejection_key(self, quality: dict) -> str:
        if quality["closed_probe"]:
            return "closed_grasp_quality"
        if quality["failure_reason"] == "open_grasp_object_contact":
            return "open_grasp_contact"
        return "grasp_contact"

    def _g1_closed_grasp_contact_quality_failure(
        self,
        quality: dict,
        object_shift_m: float,
    ) -> str | None:
        if (
            getattr(self.policy_config, "g1_reject_grasp_table_contact", True)
            and quality["table_contacts"]
        ):
            return "table_contact"
        if quality["bin_contacts"]:
            return "bin_contact"
        if quality["other_robot_object_contacts"]:
            return "non_fingertip_object_contact"
        min_pad_geom_count = int(
            getattr(self.policy_config, "g1_closed_grasp_min_pad_geom_count", 2)
        )
        if quality["pad_geom_count"] < min_pad_geom_count:
            return "missing_closed_fingertip_pad_contact"
        max_object_shift_m = float(
            getattr(self.policy_config, "g1_closed_grasp_max_object_shift_m", 0.02)
        )
        if object_shift_m > max_object_shift_m:
            return "closed_grasp_object_shift"
        return None

    def _g1_grasp_contact_score_penalty(self, quality: dict | None) -> float:
        if not quality:
            return 0.0
        if quality.get("closed_probe"):
            min_pad_geom_count = int(
                getattr(self.policy_config, "g1_closed_grasp_min_pad_geom_count", 2)
            )
            missing_pad_count = max(min_pad_geom_count - quality["pad_geom_count"], 0)
            return missing_pad_count * float(
                getattr(self.policy_config, "g1_closed_grasp_penalty_per_missing_pad", 1.0)
            )
        if quality["pad_geom_count"] >= 2:
            return 0.0
        return float(getattr(self.policy_config, "g1_grasp_single_pad_contact_penalty", 0.0))

    def _g1_closed_grasp_quality_debug(self, grasp_idx: int, quality: dict) -> dict:
        return {
            "grasp_idx": int(grasp_idx),
            "failure_reason": quality["failure_reason"],
            "pad_geom_count": int(quality["pad_geom_count"]),
            "pad_geom_names": quality["pad_geom_names"],
            "object_shift_m": float(quality["object_shift_m"]),
            "finger_link_contact_count": len(quality["finger_link_object_contacts"]),
            "other_robot_object_contact_count": len(quality["other_robot_object_contacts"]),
            "table_contact_count": len(quality["table_contacts"]),
            "bin_contact_count": len(quality["bin_contacts"]),
        }

    def _g1_robot_contact_endpoints(self, contact_info: dict) -> list[tuple[str, str]]:
        robot_namespace = self.config.robot_config.robot_namespace
        robot_endpoints = []
        if str(contact_info["root1_name"]).startswith(robot_namespace):
            robot_endpoints.append(
                (str(contact_info["geom1_name"]), str(contact_info["body1_name"]))
            )
        if str(contact_info["root2_name"]).startswith(robot_namespace):
            robot_endpoints.append(
                (str(contact_info["geom2_name"]), str(contact_info["body2_name"]))
            )
        return robot_endpoints

    def _g1_is_allowed_grasp_object_contact(self, contact_info: dict) -> bool:
        return self._g1_has_right_fingertip_pad_contact(contact_info)

    def _g1_has_right_fingertip_pad_contact(self, contact_info: dict) -> bool:
        return any(
            "right_dex1_fingertip_pad" in geom_name
            for geom_name, _body_name in self._g1_robot_contact_endpoints(contact_info)
        )

    def _g1_joint_score_terms(
        self,
        grasp_pose_world: np.ndarray,
        phase_results: dict[str, dict],
        initial_qpos: dict[str, np.ndarray],
    ) -> dict[str, float]:
        right_arm_group = self.robot_view.get_move_group("right_arm")
        limits = right_arm_group.joint_pos_limits
        ranges = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)
        initial_right_arm = np.asarray(initial_qpos["right_arm"])

        min_normalized_margin = 0.5
        joint_motion = 0.0
        n_solutions = 0
        for result in phase_results.values():
            qpos = result.get("qpos")
            if not qpos:
                continue
            right_arm_qpos = np.asarray(qpos["right_arm"])
            normalized_margin = np.minimum(
                right_arm_qpos - limits[:, 0],
                limits[:, 1] - right_arm_qpos,
            ) / ranges
            min_normalized_margin = min(
                min_normalized_margin,
                float(np.min(normalized_margin)),
            )
            joint_motion += float(
                np.linalg.norm(right_arm_qpos - initial_right_arm)
                / np.sqrt(max(len(initial_right_arm), 1))
            )
            n_solutions += 1

        if n_solutions:
            joint_motion /= n_solutions
        joint_margin_cost = 1.0 - np.clip(min_normalized_margin / 0.5, 0.0, 1.0)
        topdown_cost = float(np.clip(grasp_pose_world[2, 2] + 1.0, 0.0, 2.0))
        return {
            "joint_margin_cost": float(joint_margin_cost),
            "joint_motion_cost": float(joint_motion),
            "topdown_cost": topdown_cost,
            "min_normalized_joint_margin": float(min_normalized_margin),
        }

    def _select_g1_tabletop_grasp_pose(
        self,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
        robot_view,
    ) -> np.ndarray:
        model = self.task.env.current_model
        data = self.task.env.current_data
        grasp_poses_world, _, object_pose = get_all_grasp_poses(self, pickup_obj)
        original_grasp_count = len(grasp_poses_world) // 2

        tcp_pose_world = self._current_g1_tcp_pose()
        dist_tcp = np.linalg.inv(tcp_pose_world) @ grasp_poses_world
        dists_tcp_p = np.linalg.norm(dist_tcp[:, :3, 3], axis=1)
        dists_tcp_o = R.from_matrix(dist_tcp[:, :3, :3]).magnitude() * 180 / np.pi
        dists_up = grasp_poses_world[:, 2, 2]
        dists_com = np.linalg.norm(
            (np.linalg.inv(object_pose) @ grasp_poses_world)[:, :3, 3],
            axis=1,
        )
        base_cost = (
            self.policy_config.grasp_pos_cost_weight * dists_tcp_p
            + self.policy_config.grasp_rot_cost_weight * dists_tcp_o
            + self.policy_config.grasp_vertical_cost_weight * dists_up
            + self.policy_config.grasp_com_dist_cost_weight * dists_com
        )

        eligible_mask = np.ones(len(grasp_poses_world), dtype=bool)
        if getattr(self.policy_config, "g1_ignore_flipped_grasps", True):
            eligible_mask &= np.arange(len(grasp_poses_world)) < original_grasp_count

        max_tcp_rot_deg = float(
            getattr(self.policy_config, "g1_grasp_max_tcp_rot_deg", np.inf)
        )
        tcp_rotation_mask = np.zeros(len(grasp_poses_world), dtype=bool)
        if np.isfinite(max_tcp_rot_deg):
            tcp_eligible_mask = eligible_mask.copy()
            eligible_mask &= dists_tcp_o <= max_tcp_rot_deg
            tcp_rotation_mask = tcp_eligible_mask & ~eligible_mask
        eligible_ids = np.where(eligible_mask)[0]

        candidate_limit = min(
            int(getattr(self.policy_config, "g1_grasp_candidate_limit", 256)),
            len(eligible_ids),
        )
        candidate_ids = eligible_ids[np.argsort(base_cost[eligible_ids], kind="stable")][
            :candidate_limit
        ]

        if self.policy_config.filter_colliding_grasps:
            noncolliding_mask = get_noncolliding_grasp_mask(
                model,
                data,
                grasp_poses_world[candidate_ids],
                self.policy_config.grasp_collision_batch_size,
            )
        else:
            noncolliding_mask = np.ones(len(candidate_ids), dtype=bool)

        collision_free_ids = candidate_ids[noncolliding_mask]
        ik_eval_limit = min(
            int(getattr(self.policy_config, "g1_grasp_ik_eval_limit", 128)),
            len(collision_free_ids),
        )
        eval_ids = collision_free_ids[:ik_eval_limit]
        rejection_counts = {
            "flipped": (
                int(len(grasp_poses_world) - original_grasp_count)
                if getattr(self.policy_config, "g1_ignore_flipped_grasps", True)
                else 0
            ),
            "tcp_rotation": int(np.count_nonzero(tcp_rotation_mask)),
            "collision": int(len(candidate_ids) - len(collision_free_ids)),
            "pregrasp": 0,
            "grasp": 0,
            "lift": 0,
            "preplace": 0,
            "place": 0,
            "grasp_contact": 0,
            "open_grasp_contact": 0,
            "closed_grasp_quality": 0,
        }
        initial_qpos = self._copy_qpos_dict(robot_view.get_qpos_dict())
        base_pose = robot_view.base.pose.copy()
        feasible_candidates = []
        partial_candidates = []
        phase_progress_rank = {
            "pregrasp": 0,
            "grasp": 1,
            "lift": 2,
            "preplace": 3,
            "place": 4,
        }
        closed_grasp_quality_failures = []

        for grasp_idx in eval_ids:
            grasp_pose_world = grasp_poses_world[grasp_idx]
            target_poses = self._build_g1_candidate_target_poses(
                grasp_pose_world,
                pickup_obj,
                place_receptacle,
            )
            eval_result = self._g1_eval_candidate_target_poses(
                target_poses,
                initial_qpos,
                base_pose,
            )
            if not eval_result["success"]:
                failed_phase = eval_result["failed_phase"]
                rejection_counts[failed_phase] = rejection_counts.get(failed_phase, 0) + 1
                if phase_progress_rank.get(failed_phase, 0) > phase_progress_rank["grasp"]:
                    grasp_quality = None
                    grasp_result = eval_result["phase_results"].get("grasp")
                    if (
                        grasp_result
                        and grasp_result.get("success")
                        and grasp_result.get("qpos")
                    ):
                        grasp_quality = self._g1_candidate_grasp_contact_quality(
                            grasp_result["qpos"],
                            base_pose,
                            pickup_obj,
                        )
                        if grasp_quality["failure_reason"]:
                            rejection_key = self._g1_grasp_quality_rejection_key(grasp_quality)
                            rejection_counts[rejection_key] += 1
                            reason_key = f"{rejection_key}:{grasp_quality['failure_reason']}"
                            rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1
                            if grasp_quality["closed_probe"]:
                                closed_grasp_quality_failures.append(
                                    self._g1_closed_grasp_quality_debug(
                                        grasp_idx,
                                        grasp_quality,
                                    )
                                )
                            continue
                    g1_terms = self._g1_joint_score_terms(
                        grasp_pose_world,
                        eval_result["phase_results"],
                        initial_qpos,
                    )
                    progress_penalty = 10.0 - phase_progress_rank.get(failed_phase, 0)
                    total_score = (
                        progress_penalty
                        + float(base_cost[grasp_idx])
                        + self.policy_config.g1_grasp_joint_margin_weight
                        * g1_terms["joint_margin_cost"]
                        + self.policy_config.g1_grasp_joint_motion_weight
                        * g1_terms["joint_motion_cost"]
                        + self.policy_config.g1_grasp_topdown_weight * g1_terms["topdown_cost"]
                        + self._g1_grasp_contact_score_penalty(grasp_quality)
                    )
                    partial_candidates.append(
                        {
                            "grasp_idx": int(grasp_idx),
                            "grasp_pose_world": grasp_pose_world.copy(),
                            "target_poses": target_poses,
                            "score": total_score,
                            "failed_phase": failed_phase,
                            "score_terms": {
                                "base_cost": float(base_cost[grasp_idx]),
                                "tcp_pos_cost_m": float(dists_tcp_p[grasp_idx]),
                                "tcp_rot_cost_deg": float(dists_tcp_o[grasp_idx]),
                                "vertical_axis_z": float(dists_up[grasp_idx]),
                                "com_dist_cost_m": float(dists_com[grasp_idx]),
                                "progress_penalty": progress_penalty,
                                "is_flipped": bool(grasp_idx >= original_grasp_count),
                                "grasp_pad_contact_count": (
                                    int(grasp_quality["pad_contact_count"])
                                    if grasp_quality
                                    else 0
                                ),
                                "grasp_pad_geom_count": (
                                    int(grasp_quality["pad_geom_count"])
                                    if grasp_quality
                                    else 0
                                ),
                                "grasp_contact_penalty": self._g1_grasp_contact_score_penalty(
                                    grasp_quality
                                ),
                                "closed_grasp_object_shift_m": (
                                    float(grasp_quality["object_shift_m"])
                                    if grasp_quality
                                    else 0.0
                                ),
                                "closed_grasp_failure_reason": (
                                    grasp_quality["failure_reason"]
                                    if grasp_quality
                                    else None
                                ),
                                **g1_terms,
                            },
                            "pregrasp_debug": self._g1_pregrasp_debug(
                                target_poses["pregrasp"],
                                target_poses["grasp"],
                                pickup_obj,
                            ),
                        }
                    )
                continue

            grasp_quality = self._g1_candidate_grasp_contact_quality(
                eval_result["phase_results"]["grasp"]["qpos"],
                base_pose,
                pickup_obj,
            )
            if grasp_quality["failure_reason"]:
                rejection_key = self._g1_grasp_quality_rejection_key(grasp_quality)
                rejection_counts[rejection_key] += 1
                reason_key = f"{rejection_key}:{grasp_quality['failure_reason']}"
                rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1
                if grasp_quality["closed_probe"]:
                    closed_grasp_quality_failures.append(
                        self._g1_closed_grasp_quality_debug(
                            grasp_idx,
                            grasp_quality,
                        )
                    )
                continue

            g1_terms = self._g1_joint_score_terms(
                grasp_pose_world,
                eval_result["phase_results"],
                initial_qpos,
            )
            total_score = (
                float(base_cost[grasp_idx])
                + self.policy_config.g1_grasp_joint_margin_weight
                * g1_terms["joint_margin_cost"]
                + self.policy_config.g1_grasp_joint_motion_weight
                * g1_terms["joint_motion_cost"]
                + self.policy_config.g1_grasp_topdown_weight * g1_terms["topdown_cost"]
                + self._g1_grasp_contact_score_penalty(grasp_quality)
            )
            feasible_candidates.append(
                {
                    "grasp_idx": int(grasp_idx),
                    "grasp_pose_world": grasp_pose_world.copy(),
                    "target_poses": target_poses,
                    "score": total_score,
                    "score_terms": {
                        "base_cost": float(base_cost[grasp_idx]),
                        "tcp_pos_cost_m": float(dists_tcp_p[grasp_idx]),
                        "tcp_rot_cost_deg": float(dists_tcp_o[grasp_idx]),
                        "vertical_axis_z": float(dists_up[grasp_idx]),
                        "com_dist_cost_m": float(dists_com[grasp_idx]),
                        "is_flipped": bool(grasp_idx >= original_grasp_count),
                        "grasp_pad_contact_count": int(grasp_quality["pad_contact_count"]),
                        "grasp_pad_geom_count": int(grasp_quality["pad_geom_count"]),
                        "grasp_contact_penalty": self._g1_grasp_contact_score_penalty(
                            grasp_quality
                        ),
                        "closed_grasp_object_shift_m": float(
                            grasp_quality["object_shift_m"]
                        ),
                        "closed_grasp_failure_reason": grasp_quality["failure_reason"],
                        **g1_terms,
                    },
                    "pregrasp_debug": self._g1_pregrasp_debug(
                        target_poses["pregrasp"],
                        target_poses["grasp"],
                        pickup_obj,
                    ),
                }
            )

        selector_summary = {
            "grasp_count": int(len(grasp_poses_world)),
            "eligible_count": int(len(eligible_ids)),
            "candidate_count": int(candidate_limit),
            "collision_free_count": int(len(collision_free_ids)),
            "ik_tested_count": int(len(eval_ids)),
            "ik_feasible_count": int(len(feasible_candidates)),
            "partial_candidate_count": int(len(partial_candidates)),
            "rejection_counts": rejection_counts,
            "closed_grasp_quality_failures": closed_grasp_quality_failures[:8],
        }
        self._g1_grasp_selector_debug = selector_summary

        if not feasible_candidates:
            if (
                partial_candidates
                and getattr(
                    self.policy_config,
                    "g1_record_partial_attempt_on_no_full_grasp_candidate",
                    True,
                )
            ):
                selected = min(partial_candidates, key=lambda candidate: candidate["score"])
                log.info(
                    "[G1_DIAG] reason=no_full_g1_tabletop_grasp_candidate_using_partial "
                    "failed_phase=%s summary=%s",
                    selected["failed_phase"],
                    selector_summary,
                )
                return self._finalize_g1_selected_grasp(
                    selected,
                    selector_summary,
                    grasp_poses_world,
                    candidate_ids,
                    dists_tcp_p,
                    dists_tcp_o,
                    partial=True,
                )
            log.info("[G1_DIAG] reason=no_g1_tabletop_grasp_candidate summary=%s", selector_summary)
            raise ValueError("No G1 tabletop grasp candidate found")

        selected = min(feasible_candidates, key=lambda candidate: candidate["score"])
        return self._finalize_g1_selected_grasp(
            selected,
            selector_summary,
            grasp_poses_world,
            candidate_ids,
            dists_tcp_p,
            dists_tcp_o,
            partial=False,
        )

    def _finalize_g1_selected_grasp(
        self,
        selected: dict,
        selector_summary: dict,
        grasp_poses_world: np.ndarray,
        candidate_ids: np.ndarray,
        dists_tcp_p: np.ndarray,
        dists_tcp_o: np.ndarray,
        partial: bool,
    ) -> np.ndarray:
        original_grasp_count = len(grasp_poses_world) // 2
        original_grasp_idx = selected["grasp_idx"] % original_grasp_count
        selected_is_flipped = selected["grasp_idx"] >= original_grasp_count
        selected_summary = {
            **selector_summary,
            "selected_grasp_idx": selected["grasp_idx"],
            "selected_original_grasp_idx": int(original_grasp_idx),
            "selected_is_flipped": bool(selected_is_flipped),
            "selected_score": float(selected["score"]),
            "selected_score_terms": selected["score_terms"],
            "selected_pregrasp": selected.get("pregrasp_debug"),
            "selected_partial_candidate": partial,
            "selected_partial_failed_phase": selected.get("failed_phase"),
        }
        self._g1_grasp_selector_debug = selected_summary
        self._record_g1_selected_grasp_debug(
            {
                "grasp_idx": selected["grasp_idx"],
                "original_grasp_idx": int(original_grasp_idx),
                "is_flipped": bool(selected_is_flipped),
                "grasp_pose_world": selected["grasp_pose_world"].copy(),
                "top_candidate_ids": candidate_ids[
                    : getattr(self.policy_config, "g1_ik_debug_top_k_grasps", 5)
                ]
                .astype(int)
                .tolist(),
                "top_candidate_poses": grasp_poses_world[
                    candidate_ids[
                        : getattr(self.policy_config, "g1_ik_debug_top_k_grasps", 5)
                    ]
                ].copy(),
                "selected_pos_cost_m": float(dists_tcp_p[selected["grasp_idx"]]),
                "selected_rot_cost_deg": float(dists_tcp_o[selected["grasp_idx"]]),
                "selected_pregrasp": selected.get("pregrasp_debug"),
                "selector_summary": selected_summary,
            }
        )
        log.info("[G1_GRASP] selector_summary=%s", selected_summary)
        return selected["grasp_pose_world"]

    def _current_g1_tcp_pose(self) -> np.ndarray:
        return (
            self.robot_view.get_move_group(self.robot_view.get_gripper_movegroup_ids()[0])
            .leaf_frame_to_world.copy()
        )

    def _failure_diagnostics_enabled(self) -> bool:
        return bool(getattr(self.policy_config, "enable_failure_diagnostics", False))

    def _g1_ik_debug_enabled(self) -> bool:
        return bool(getattr(self.policy_config, "g1_ik_debug", False))

    def _names_for_contact(self, geom_id: int) -> dict[str, str | int]:
        model = self.task.env.current_model
        body_id = model.geom_bodyid[geom_id]
        root_body_id = model.body_rootid[body_id]
        return {
            "geom_id": int(geom_id),
            "geom_name": model.geom(geom_id).name,
            "body_id": int(body_id),
            "body_name": model.body(body_id).name,
            "root_body_id": int(root_body_id),
            "root_body_name": model.body(root_body_id).name,
        }

    def _contact_has_name(self, contact_info: dict, needle: str) -> bool:
        if not needle:
            return False
        needle = needle.lower()
        return any(
            needle in str(contact_info[key]).lower()
            for key in (
                "geom1_name",
                "geom2_name",
                "body1_name",
                "body2_name",
                "root1_name",
                "root2_name",
            )
        )

    def _contact_has_exact_name(self, contact_info: dict, needle: str) -> bool:
        if not needle:
            return False
        return any(
            needle == str(contact_info[key])
            for key in (
                "geom1_name",
                "geom2_name",
                "body1_name",
                "body2_name",
                "root1_name",
                "root2_name",
            )
        )

    def _classify_contact(self, contact_info: dict) -> str:
        robot_namespace = self.config.robot_config.robot_namespace
        robot1 = str(contact_info["root1_name"]).startswith(robot_namespace)
        robot2 = str(contact_info["root2_name"]).startswith(robot_namespace)
        if not (robot1 or robot2):
            return "other"
        if robot1 and robot2:
            return "robot_self"

        task_sampler_config = self.config.task_sampler_config
        task_config = self.config.task_config
        if self._contact_has_exact_name(contact_info, getattr(task_sampler_config, "table_body_name", "")):
            return "robot_table"
        if self._contact_has_exact_name(contact_info, getattr(task_sampler_config, "table_geom_name", "")):
            return "robot_table"
        if self._contact_has_name(contact_info, getattr(task_sampler_config, "place_receptacle_name", "")):
            return "robot_bin"
        if self._contact_has_name(contact_info, getattr(task_config, "pickup_obj_name", "")):
            return "robot_pickup_object"
        return "other"

    def _collect_contact_diagnostics(self) -> list[dict]:
        data = self.task.env.current_data
        contact_infos = []
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.dist > 0.0:
                continue
            geom1 = self._names_for_contact(contact.geom1)
            geom2 = self._names_for_contact(contact.geom2)
            contact_info = {
                "contact_id": int(contact_id),
                "dist": float(contact.dist),
                "geom1_name": geom1["geom_name"],
                "geom2_name": geom2["geom_name"],
                "body1_name": geom1["body_name"],
                "body2_name": geom2["body_name"],
                "root1_name": geom1["root_body_name"],
                "root2_name": geom2["root_body_name"],
            }
            contact_info["class"] = self._classify_contact(contact_info)
            if contact_info["class"] != "other":
                contact_infos.append(contact_info)
        return contact_infos

    def _record_failure_diagnostic_step(
        self,
        target_pose: np.ndarray,
        ik_succeeded: bool,
    ) -> None:
        if not self._failure_diagnostics_enabled():
            return

        tcp_pose = self._current_g1_tcp_pose()
        target_pos_err = float(np.linalg.norm(tcp_pose[:3, 3] - target_pose[:3, 3]))
        self._g1_failure_diag_steps.append(
            {
                "step_index": len(self._g1_failure_diag_steps),
                "sim_time": float(self.task.env.current_data.time),
                "phase": self.get_phase(),
                "ik_succeeded": bool(ik_succeeded),
                "sequential_ik_failures": int(self.sequential_ik_failures),
                "tcp_pos": np.round(tcp_pose[:3, 3], 4).tolist(),
                "target_pos": np.round(target_pose[:3, 3], 4).tolist(),
                "target_pos_err": target_pos_err,
                "contacts": self._collect_contact_diagnostics(),
            }
        )

    def _pickup_object_diagnostics(self) -> dict:
        task_config = self.config.task_config
        pickup_obj_name = getattr(task_config, "pickup_obj_name", None)
        if not pickup_obj_name:
            return {"pickup_obj_name": None}

        diagnostics = {
            "pickup_obj_name": pickup_obj_name,
            "pickup_obj_start_pose": getattr(task_config, "pickup_obj_start_pose", None),
            "place_receptacle_name": getattr(task_config, "place_receptacle_name", None),
        }
        try:
            diagnostics["task_description"] = self.task.get_obs_scene().get("task_description")
        except Exception:
            diagnostics["task_description"] = None
        try:
            om = self.task.env.object_managers[self.task.env.current_batch_index]
            pickup_obj = om.get_object_by_name(pickup_obj_name)
            center, size = body_aabb(
                self.task.env.current_model,
                self.task.env.current_data,
                pickup_obj.body_id,
                visual_only=False,
            )
            max_extent = float(np.max(size))
            volume = float(np.prod(size))
            diagnostics.update(
                {
                    "pickup_obj_pos": np.round(pickup_obj.position, 4).tolist(),
                    "pickup_obj_aabb_center": np.round(center, 4).tolist(),
                    "pickup_obj_aabb_size": np.round(size, 4).tolist(),
                    "pickup_obj_max_extent_m": max_extent,
                    "pickup_obj_volume_m3": volume,
                    "object_size_outlier": bool(
                        max_extent
                        > getattr(
                            self.policy_config,
                            "diagnostic_large_object_max_extent_m",
                            0.18,
                        )
                        or volume
                        > getattr(self.policy_config, "diagnostic_large_object_volume_m3", 0.004)
                    ),
                }
            )
        except Exception as exc:
            diagnostics["pickup_obj_error"] = repr(exc)
        return diagnostics

    def _tcp_tracking_diagnostics(self) -> dict | None:
        if self.action_idx >= len(self.action_primitives):
            return None
        action_primitive = self.action_primitives[self.action_idx]
        if not isinstance(action_primitive, TCPMoveSequence) or action_primitive.move_seg_idx is None:
            return None

        phase = action_primitive.get_current_phase()
        target_pose = action_primitive.get_current_target_pose()
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        gripper = self.robot_view.get_gripper(gripper_mg_id)
        target_from_tcp = np.linalg.inv(gripper.leaf_frame_to_world) @ target_pose
        pos_err = float(np.linalg.norm(target_from_tcp[:3, 3]))
        rot_err = float(R.from_matrix(target_from_tcp[:3, :3]).magnitude())
        pos_threshold, rot_threshold = self._tcp_tracking_thresholds_for_phase(phase)
        return {
            "phase": phase,
            "tcp_tracking_pos_error": pos_err,
            "tcp_tracking_rot_error": rot_err,
            "tcp_tracking_pos_threshold": pos_threshold,
            "tcp_tracking_rot_threshold": rot_threshold,
            "tcp_tracking_position_failed": bool(pos_err > pos_threshold),
            "tcp_tracking_rotation_failed": bool(rot_err > rot_threshold),
        }

    def _tcp_tracking_thresholds_for_phase(self, phase: str) -> tuple[float, float]:
        if phase == "pregrasp":
            return (
                float(getattr(self.policy_config, "pregrasp_tcp_pos_err_threshold", 0.1)),
                float(getattr(self.policy_config, "pregrasp_tcp_rot_err_threshold", np.inf)),
            )
        return (
            float(self.policy_config.tcp_pos_err_threshold),
            float(self.policy_config.tcp_rot_err_threshold),
        )

    def _failure_reason_guess(
        self,
        had_ik_failure: bool,
        had_robot_collision: bool,
        tracking_diagnostics: dict | None,
    ) -> str:
        if self._g1_last_ik_debug and self._g1_last_ik_debug.get("root_cause"):
            return self._g1_last_ik_debug["root_cause"]
        if self._g1_failure_reason_override:
            return self._g1_failure_reason_override
        if had_robot_collision and had_ik_failure:
            return "ik_failure_with_robot_contact"
        if had_robot_collision:
            return "robot_collision"
        if had_ik_failure:
            return "unreachable_or_poor_grasp_pose"
        if tracking_diagnostics:
            position_failed = tracking_diagnostics["tcp_tracking_position_failed"]
            rotation_failed = tracking_diagnostics["tcp_tracking_rotation_failed"]
            if position_failed and rotation_failed:
                return "tcp_pose_tracking_error"
            if position_failed:
                return "tcp_position_tracking_error"
            if rotation_failed:
                return "tcp_orientation_tracking_error"
        return "planner_failure"

    def _log_failure_diagnostics(self) -> None:
        if (
            not self._failure_diagnostics_enabled()
            or getattr(self, "_g1_failure_diagnostics_logged", False)
        ):
            return
        self._g1_failure_diagnostics_logged = True

        steps = getattr(self, "_g1_failure_diag_steps", [])
        contacts_by_class: dict[str, int] = {}
        for step in steps:
            for contact in step["contacts"]:
                contacts_by_class[contact["class"]] = contacts_by_class.get(contact["class"], 0) + 1

        robot_contact_classes = {
            "robot_table",
            "robot_bin",
            "robot_pickup_object",
            "robot_self",
        }
        had_robot_collision = any(contacts_by_class.get(k, 0) > 0 for k in robot_contact_classes)
        had_ik_failure = any(not step["ik_succeeded"] for step in steps)

        current_tcp_pose = self._current_g1_tcp_pose()
        initial_tcp_pose = getattr(self, "_g1_initial_tcp_pose", current_tcp_pose)
        tcp_displacement = float(
            np.linalg.norm(current_tcp_pose[:3, 3] - initial_tcp_pose[:3, 3])
        )
        failure_before_motion = bool(tcp_displacement < 0.02 and had_ik_failure)
        tracking_diagnostics = self._tcp_tracking_diagnostics()

        summary = {
            **self._pickup_object_diagnostics(),
            "phase_at_failure": self._g1_failure_phase_override or self.get_phase(),
            "failure_reason_guess": self._failure_reason_guess(
                had_ik_failure,
                had_robot_collision,
                tracking_diagnostics,
            ),
            "unfiltered_grasp_fallback_used": self._g1_unfiltered_grasp_fallback_used,
            "unfiltered_grasp_fallback_reason": self._g1_unfiltered_grasp_fallback_reason,
            "grasp_selector": self._g1_grasp_selector_debug,
            "failure_before_motion": failure_before_motion,
            "tcp_displacement_m": tcp_displacement,
            "tcp_tracking": tracking_diagnostics,
            "sequential_ik_failures": int(self.sequential_ik_failures),
            "contact_counts_by_class": contacts_by_class,
            "last_target_pose": (
                np.round(pose_mat_to_7d(self._g1_last_target_pose), 4).tolist()
                if self._g1_last_target_pose is not None
                else None
            ),
            "current_tcp_pose": np.round(pose_mat_to_7d(current_tcp_pose), 4).tolist(),
            "num_diagnostic_steps": len(steps),
            "held_grip_diagnostics": getattr(self, "_g1_grip_diag_samples", [])[-12:],
        }
        log.info("[G1_DIAG] failure_summary=%s", summary)
        self._write_g1_failure_summary_debug_row(summary, contacts_by_class)

        for step in steps:
            if step["contacts"] or not step["ik_succeeded"]:
                log.info(
                    "[G1_DIAG] step=%s phase=%s ik=%s ik_fails=%s target_err=%.4f "
                    "tcp=%s target=%s contacts=%s",
                    step["step_index"],
                    step["phase"],
                    step["ik_succeeded"],
                    step["sequential_ik_failures"],
                    step["target_pos_err"],
                    step["tcp_pos"],
                    step["target_pos"],
                    step["contacts"],
                )

    def _handle_failure(self) -> dict[str, np.ndarray | bool]:
        self._log_failure_diagnostics()
        hold_duration = float(
            getattr(self.policy_config, "diagnostic_failure_hold_duration_s", 0.0)
        )
        if (
            self._failure_diagnostics_enabled()
            and self.retry_count >= self.policy_config.max_retries
            and hold_duration > 0.0
        ):
            self._g1_terminal_failure_pending = True
            if self._g1_failure_hold_action is None:
                log.info("[G1_DIAG] holding failed attempt for %.2fs before done", hold_duration)
                self._g1_failure_hold_action = NoopAction(self.robot_view, hold_duration)
            if not self._g1_failure_hold_action.execute():
                return self._g1_failure_hold_action.get_current_action()

        return super()._handle_failure()

    def _check_for_failures(self) -> bool:
        self._record_g1_grip_diagnostic_sample()
        if self._g1_terminal_failure_pending:
            return True
        if self._g1_failure_reason_override:
            if self.action_idx >= len(self.action_primitives):
                return True
            action_primitive = self.action_primitives[self.action_idx]
            if action_primitive.start_time is None:
                return False
            return action_primitive.elapsed_time() >= action_primitive.duration
        if self.action_idx < len(self.action_primitives):
            action_primitive = self.action_primitives[self.action_idx]
            if (
                isinstance(action_primitive, TCPMoveSequence)
                and action_primitive.start_time is not None
                and action_primitive.get_current_phase() == "pregrasp"
            ):
                tracking_diagnostics = self._tcp_tracking_diagnostics()
                if tracking_diagnostics is None:
                    return False
                return bool(
                    tracking_diagnostics["tcp_tracking_position_failed"]
                    or tracking_diagnostics["tcp_tracking_rotation_failed"]
                )
        return super()._check_for_failures()

    def _tcp_to_jp_fn(self, mg_id: str, target_pose: np.ndarray) -> dict[str, np.ndarray]:
        kinematics = self.task.env.current_robot.kinematics

        jp = kinematics.ik(
            mg_id,
            target_pose,
            ["right_arm"],
            self.robot_view.get_qpos_dict(),
            self.robot_view.base.pose,
        )
        self._g1_last_target_pose = target_pose.copy()

        action = self.robot_view.get_ctrl_dict()
        if jp is not None:
            self.sequential_ik_failures = 0
            action["right_arm"] = jp["right_arm"]
            self._record_failure_diagnostic_step(target_pose, True)
        else:
            self.sequential_ik_failures += 1
            log.info(f"IK failed, holding current position, fails:{self.sequential_ik_failures}")
            self._record_failure_diagnostic_step(target_pose, False)
            self._run_g1_ik_debug(mg_id, target_pose)
            if self.sequential_ik_failures >= self.policy_config.max_sequential_ik_failures:
                log.info("Too many sequential IK failures, triggering retry.")
                return self._handle_failure()

        return action

    def _run_diagnostic_ik(
        self,
        mg_id: str,
        target_pose: np.ndarray,
        unlocked_groups: list[str],
        max_iter: int = 250,
    ) -> dict:
        return self.task.env.current_robot.kinematics.diagnose_ik(
            mg_id,
            target_pose,
            unlocked_groups,
            self.robot_view.get_qpos_dict(),
            self.robot_view.base.pose,
            max_iter=max_iter,
        )

    def _candidate_pose_for_phase(self, candidate_grasp_pose: np.ndarray) -> np.ndarray:
        pose = candidate_grasp_pose.copy()
        if self.get_phase() == "pregrasp":
            pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pose[:3, 2]
        return pose

    def _summarize_ik_result(self, result: dict) -> str:
        status = "ok" if result.get("success") else "fail"
        pos_err = result.get("final_pos_error_norm")
        rot_err = result.get("final_rot_error_norm")
        clamps = result.get("clamp_event_count", 0)
        singular = result.get("singular_iterations", 0)
        unavailable = result.get("unavailable_groups") or []
        parts = [status]
        if pos_err is not None:
            parts.append(f"p={pos_err:.3f}")
        if rot_err is not None:
            parts.append(f"r={rot_err:.3f}")
        if clamps:
            parts.append(f"clamp={clamps}")
        if singular:
            parts.append(f"sing={singular}")
        if unavailable:
            parts.append(f"missing={','.join(unavailable)}")
        return " ".join(parts)

    def _target_distance_diagnostics(self, target_pose: np.ndarray) -> dict:
        base_pos = self.robot_view.base.pose[:3, 3]
        target_pos = target_pose[:3, 3]
        diagnostics = {
            "target_z": float(target_pos[2]),
            "target_base_distance": float(np.linalg.norm(target_pos - base_pos)),
        }
        try:
            shoulder_pos = self.task.env.current_data.body(
                f"{self.config.robot_config.robot_namespace}right_shoulder_pitch_link"
            ).xpos
            diagnostics["target_shoulder_distance"] = float(np.linalg.norm(target_pos - shoulder_pos))
        except KeyError:
            diagnostics["target_shoulder_distance"] = None
        return diagnostics

    def _ik_debug_root_cause(self, debug: dict, contacts: list[dict]) -> str:
        contact_classes = {contact["class"] for contact in contacts}
        if contact_classes & {"robot_table", "robot_bin", "robot_pickup_object"}:
            return "table/object_collision"
        if "robot_self" in contact_classes:
            return "robot_self_collision"

        right_arm = debug["variants"]["right_arm"]
        waist_right_arm = debug["variants"]["waist_right_arm"]
        base_waist_right_arm = debug["variants"]["base_waist_right_arm"]
        relaxed_orientation = debug["variants"]["relaxed_orientation"]
        higher_z = debug["variants"]["higher_z"]

        if not right_arm.get("success") and waist_right_arm.get("success"):
            return "needs_waist"
        if not right_arm.get("success") and relaxed_orientation.get("success"):
            return "orientation_constrained"
        if not right_arm.get("success") and higher_z.get("success"):
            return "orientation_constrained"
        if not right_arm.get("success") and any(
            candidate.get("success") for candidate in debug.get("top_k_grasp_results", [])
        ):
            return "orientation_constrained"
        if right_arm.get("clamp_event_count", 0) > 0:
            return "joint_limit"
        if (
            right_arm.get("singular_iterations", 0) > 0
            or (right_arm.get("min_singular_value") or 1.0) < 1e-4
        ):
            return "singularity"
        if not right_arm.get("success") and base_waist_right_arm.get("success"):
            return "object_too_far"
        return "object_too_far"

    def _g1_ik_debug_table_path(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "g1_ik_diagnostics.md"

    def _ensure_g1_ik_debug_table(self) -> Path:
        table_path = self._g1_ik_debug_table_path()
        if not table_path.exists():
            table_path.write_text(
                "| episode_object | phase | grasp_id | target_z | base_dist | shoulder_dist | "
                "right_arm | waist_right_arm | base_waist_right_arm | relaxed_orientation | "
                "higher_z | top_k_success | contacts | root_cause |\n"
                "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | --- |\n"
            )
        return table_path

    def _write_g1_ik_debug_table(self, debug: dict) -> None:
        table_path = self._ensure_g1_ik_debug_table()
        selected_grasp = debug.get("selected_grasp") or {}
        distances = debug["distances"]
        variants = debug["variants"]
        top_k_success = sum(1 for item in debug.get("top_k_grasp_results", []) if item.get("success"))
        contacts = ",".join(sorted({contact["class"] for contact in debug["contacts"]})) or "none"
        row = (
            f"| {debug['pickup_obj_name']} | {debug['phase']} | "
            f"{selected_grasp.get('grasp_idx', 'n/a')} | "
            f"{distances['target_z']:.3f} | {distances['target_base_distance']:.3f} | "
            f"{(distances['target_shoulder_distance'] or float('nan')):.3f} | "
            f"{self._summarize_ik_result(variants['right_arm'])} | "
            f"{self._summarize_ik_result(variants['waist_right_arm'])} | "
            f"{self._summarize_ik_result(variants['base_waist_right_arm'])} | "
            f"{self._summarize_ik_result(variants['relaxed_orientation'])} | "
            f"{self._summarize_ik_result(variants['higher_z'])} | "
            f"{top_k_success} | {contacts} | {debug['root_cause']} |\n"
        )
        with table_path.open("a") as f:
            f.write(row)

    def _write_g1_failure_summary_debug_row(
        self,
        summary: dict,
        contacts_by_class: dict[str, int],
    ) -> None:
        if not self._g1_ik_debug_enabled() or self._g1_last_ik_debug is not None:
            return

        table_path = self._ensure_g1_ik_debug_table()
        target_pose = self._g1_last_target_pose if self._g1_last_target_pose is not None else self._current_g1_tcp_pose()
        distances = self._target_distance_diagnostics(target_pose)
        selected_grasp = self._g1_selected_grasp_debug or {}
        contacts = ",".join(sorted(contacts_by_class)) or "none"
        row = (
            f"| {summary.get('pickup_obj_name')} | {summary.get('phase_at_failure')} | "
            f"{selected_grasp.get('grasp_idx', 'n/a')} | "
            f"{distances['target_z']:.3f} | {distances['target_base_distance']:.3f} | "
            f"{(distances['target_shoulder_distance'] or float('nan')):.3f} | "
            "n/a | n/a | n/a | n/a | n/a | 0 | "
            f"{contacts} | {summary.get('failure_reason_guess')} |\n"
        )
        with table_path.open("a") as f:
            f.write(row)

    def _run_g1_ik_debug(self, mg_id: str, target_pose: np.ndarray) -> None:
        if not self._g1_ik_debug_enabled():
            return

        current_tcp_pose = self._current_g1_tcp_pose()
        relaxed_orientation_pose = target_pose.copy()
        relaxed_orientation_pose[:3, :3] = current_tcp_pose[:3, :3]
        higher_z_pose = target_pose.copy()
        higher_z_pose[2, 3] += getattr(self.policy_config, "g1_ik_debug_higher_z_offset", 0.05)

        variants = {
            "right_arm": self._run_diagnostic_ik(mg_id, target_pose, ["right_arm"]),
            "waist_right_arm": self._run_diagnostic_ik(
                mg_id,
                target_pose,
                ["waist", "right_arm"],
            ),
            "base_waist_right_arm": self._run_diagnostic_ik(
                mg_id,
                target_pose,
                ["base", "waist", "right_arm"],
            ),
            "relaxed_orientation": self._run_diagnostic_ik(
                mg_id,
                relaxed_orientation_pose,
                ["right_arm"],
            ),
            "higher_z": self._run_diagnostic_ik(mg_id, higher_z_pose, ["right_arm"]),
        }

        top_k_grasp_results = []
        selected_grasp = self._g1_selected_grasp_debug or {}
        candidate_poses = selected_grasp.get("top_candidate_poses")
        candidate_ids = selected_grasp.get("top_candidate_ids") or []
        if candidate_poses is not None:
            for candidate_id, candidate_pose in zip(candidate_ids, candidate_poses, strict=False):
                candidate_target = self._candidate_pose_for_phase(candidate_pose)
                result = self._run_diagnostic_ik(mg_id, candidate_target, ["right_arm"], max_iter=150)
                result["candidate_id"] = int(candidate_id)
                top_k_grasp_results.append(result)

        contacts = self._collect_contact_diagnostics()
        debug = {
            "pickup_obj_name": getattr(self.config.task_config, "pickup_obj_name", None),
            "phase": self.get_phase(),
            "selected_grasp": {
                key: value
                for key, value in selected_grasp.items()
                if key not in {"grasp_pose_world", "top_candidate_poses"}
            },
            "target_pose": np.round(pose_mat_to_7d(target_pose), 6).tolist(),
            "current_tcp_pose": np.round(pose_mat_to_7d(current_tcp_pose), 6).tolist(),
            "right_arm_qpos": np.round(self.robot_view.get_move_group("right_arm").joint_pos, 6).tolist(),
            "distances": self._target_distance_diagnostics(target_pose),
            "variants": variants,
            "top_k_grasp_results": top_k_grasp_results,
            "contacts": contacts,
        }
        debug["root_cause"] = self._ik_debug_root_cause(debug, contacts)
        self._g1_last_ik_debug = debug

        log.info(
            "[G1_IK_DEBUG] object=%s phase=%s grasp=%s root_cause=%s right_arm=%s "
            "waist_right_arm=%s base_waist_right_arm=%s relaxed=%s higher_z=%s contacts=%s",
            debug["pickup_obj_name"],
            debug["phase"],
            debug["selected_grasp"].get("grasp_idx"),
            debug["root_cause"],
            self._summarize_ik_result(variants["right_arm"]),
            self._summarize_ik_result(variants["waist_right_arm"]),
            self._summarize_ik_result(variants["base_waist_right_arm"]),
            self._summarize_ik_result(variants["relaxed_orientation"]),
            self._summarize_ik_result(variants["higher_z"]),
            contacts,
        )
        self._write_g1_ik_debug_table(debug)

    def check_feasible_ik(self, pose: np.ndarray) -> bool:
        if self._g1_bypass_feasible_ik:
            if pose.ndim > 2:
                return np.ones(pose.shape[0], dtype=bool)
            return True

        if not self.policy_config.filter_feasible_grasps:
            if pose.ndim > 2:
                return np.ones(pose.shape[0], dtype=bool)
            return True

        robot_view = self.task.env.current_robot.robot_view
        kinematics = self.task.env.current_robot.kinematics
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]

        if pose.ndim > 2:
            return np.array(
                [
                    kinematics.ik(
                        gripper_mg_id,
                        single_pose,
                        ["right_arm"],
                        robot_view.get_qpos_dict(),
                        robot_view.base.pose,
                        max_iter=100,
                    )
                    is not None
                    for single_pose in pose
                ],
                dtype=bool,
            )

        assert pose.shape == (4, 4)
        jp_dict = kinematics.ik(
            gripper_mg_id,
            pose,
            ["right_arm"],
            robot_view.get_qpos_dict(),
            robot_view.base.pose,
            max_iter=100,
        )
        return jp_dict is not None
