import logging
import time

import numpy as np

from molmo_spaces.configs import PickTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_pickup_grasps

log = logging.getLogger(__name__)


SCENE_SETTLE_DURATION = 5.0  # seconds to wait for scene to settle before planning


class PickPlannerPolicy(BaseObjectManipulationPlannerPolicy):
    def __init__(self, config, task):
        super().__init__(config, task)
        self._settling = False
        self._settle_start_time = None
        self._has_settled = False

    def reset(self, reset_retries: bool = True):
        if self._has_settled or SCENE_SETTLE_DURATION <= 0:
            # Scene already settled or no settle needed: use normal path
            self._has_settled = True
            super().reset(reset_retries=reset_retries)
        else:
            # First reset: defer trajectory computation until scene settles
            if not self.ik_warmed_up:
                from molmo_spaces.utils.profiler_utils import Timer

                with Timer() as warmup_time:
                    self.task.env.current_robot.parallel_kinematics.warmup_ik(
                        self.policy_config.grasp_feasibility_batch_size
                    )
                self.ik_warmed_up = True
                log.info(f"Warmed up parallel IK solver in {warmup_time.value:.3f}s")

            self.action_primitives = []
            self.action_idx = 0
            if reset_retries:
                self.retry_count = 0
            self.target_poses = {
                "pregrasp": np.eye(4),
                "grasp": np.eye(4),
                "lift": np.eye(4),
            }
            self._settling = True
            self._settle_start_time = None
            log.info(f"[PICK PLANNER] Waiting {SCENE_SETTLE_DURATION}s for scene to settle")

    def get_phase(self) -> str:
        if self._settling:
            return "gripper-open"
        return super().get_phase()

    def get_action(self, info):
        if self._settling:
            robot_view = self.task.env.current_robot.robot_view
            if self._settle_start_time is None:
                self._settle_start_time = robot_view.mj_data.time
            elapsed = robot_view.mj_data.time - self._settle_start_time
            if elapsed < SCENE_SETTLE_DURATION:
                return robot_view.get_noop_ctrl_dict()
            # Settling done — compute the real trajectory via normal reset
            self._settling = False
            self._has_settled = True
            log.info("[PICK PLANNER] Scene settled, computing trajectory")
            super().reset(reset_retries=True)
        return super().get_action(info)

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
                    )
                ],
            ),
        ]

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        """Compute all target poses for the pick-and-place sequence."""
        task_config = self.config.task_config
        assert isinstance(task_config, PickTaskConfig)
        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)

        candidate_grasps = get_pickup_grasps(
            self.task.env, pickup_obj, grasp_libraries=self.policy_config.grasp_libraries
        )
        grasp_pose_world = select_grasp_pose(
            self.task.env,
            candidate_grasps,
            pickup_obj.pose,
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

        target_poses = {}

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
        pregrasp_pose[:3, 3] += np.array(
            [0, 0, self.policy_config.pregrasp_z_offset + pregrasp_height_offset]
        )

        log.debug(f"  - obj_start (p): {pickup_obj.position}")
        log.debug(f"  - obj_start (t): {task_config.pickup_obj_start_pose}")
        log.debug(f"  - obj_end (t): {task_config.pickup_obj_goal_pose}")
        log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")

        pregrasp_ik_ok = self.check_feasible_ik(pregrasp_pose)
        grasp_ik_ok = self.check_feasible_ik(grasp_pose_world)

        # Lift pose - above grasp position
        lift_pose = grasp_pose_world.copy()
        lift_pose[:3, 3] += np.array(
            [0, 0, self.policy_config.postgrasp_z_offset + postgrasp_height_offset]
        )
        lift_ik_ok = self.check_feasible_ik(lift_pose)

        # If any IK failed and viewer is available, visualize and pause
        if not (pregrasp_ik_ok and grasp_ik_ok and lift_ik_ok):
            failed = []
            if not pregrasp_ik_ok:
                failed.append("pregrasp")
            if not grasp_ik_ok:
                failed.append("grasp")
            if not lift_ik_ok:
                failed.append("lift")

            log.warning(
                f"IK FAILED for: {', '.join(failed)}\n"
                f"  Pregrasp pos: {pregrasp_pose[:3, 3]}\n"
                f"  Grasp pos:    {grasp_pose_world[:3, 3]}\n"
                f"  Lift pos:     {lift_pose[:3, 3]}\n"
                f"  Robot base:   {robot_view.base.pose[:3, 3]}\n"
                f"  Height diffs: pregrasp={pregrasp_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"grasp={grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m, "
                f"lift={lift_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )

            if self.task.viewer is not None:
                # Visualize all poses: green=pregrasp, red=grasp, blue=lift
                self._show_poses(np.array([pregrasp_pose]), style="tcp", color=(0, 1, 0, 1))
                self._show_poses(np.array([grasp_pose_world]), style="tcp", color=(1, 0, 0, 1))
                self._show_poses(np.array([lift_pose]), style="tcp", color=(0, 0, 1, 1))
                self.task.viewer.sync()
                log.warning(
                    "Viewer paused — pan around to inspect poses. "
                    "Green=pregrasp, Red=grasp, Blue=lift. "
                    "Close the viewer window to continue."
                )
                while self.task.viewer.is_running():
                    time.sleep(0.1)

            raise ValueError(f"IK failed for {', '.join(failed)} pose(s)")

        target_poses["pregrasp"] = pregrasp_pose
        target_poses["grasp"] = grasp_pose_world
        target_poses["lift"] = lift_pose

        log.info(f"Planning completed. w/ {len(target_poses)} steps\n")

        return target_poses
