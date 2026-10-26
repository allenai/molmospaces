import logging
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    JointMoveSequence,
    NoopAction,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.tasks.packing_task import PackingTask
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

log = logging.getLogger(__name__)


class PackingPlannerPolicy(PickAndPlacePlannerPolicy):
    """Packs multiple objects (clutter + original) into a box sequentially."""

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        task_config = config.task_config
        if task_config.packing_object_names:
            self._packing_object_names = task_config.packing_object_names
        else:
            self._packing_object_names = [task_config.pickup_obj_name]
        self._current_object_index = 0
        self._use_dummy_place_target = not isinstance(task_config, PickAndPlaceTaskConfig)
        log.info(
            f"[PACKING PLANNER] Will pack {len(self._packing_object_names)} objects: {self._packing_object_names}"
            f" (dummy_place_target={self._use_dummy_place_target})"
        )

    def _is_last_object(self) -> bool:
        return self._current_object_index >= len(self._packing_object_names) - 1

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Center the place pose at the receptacle's AABB center (geometric middle
        # of the box) rather than the body origin, which for many box assets sits
        # at a corner or other non-central reference and causes the object to
        # land near the box wall instead of the middle.
        preplace_pose, place_pose, postplace_pose = super()._get_placement_poses(
            grasp_pose_world=grasp_pose_world,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
        )
        place_receptacle_aabb_center, _ = body_aabb(
            self.task.env.current_data.model,
            self.task.env.current_data,
            place_receptacle.object_id,
        )
        delta_xy = place_receptacle_aabb_center[:2] - place_receptacle.position[:2]
        preplace_pose[:2, 3] += delta_xy
        place_pose[:2, 3] += delta_xy
        postplace_pose[:2, 3] += delta_xy

        # Raise drop height to clear opened box flaps and reduce flap collisions.
        packing_place_z_bump = 0.20
        preplace_pose[2, 3] += packing_place_z_bump
        place_pose[2, 3] += packing_place_z_bump
        postplace_pose[2, 3] += packing_place_z_bump
        return preplace_pose, place_pose, postplace_pose

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        trajectory = super()._compute_trajectory()
        if not self._is_last_object():
            # Strip go_home and noop for intermediate objects to save time
            trajectory = [
                p for p in trajectory if not isinstance(p, (JointMoveSequence, NoopAction))
            ]
            log.info("[PACKING PLANNER] Skipping go_home + noop for intermediate object")
        return trajectory

    def _compute_target_poses(self) -> dict[str, np.ndarray]:
        """Compute target poses, using a nearby offset for pick tasks without a receptacle."""
        if not self._use_dummy_place_target:
            return super()._compute_target_poses()

        task_config = self.config.task_config
        target_poses = {}

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

        # Compute pickup object geometry
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)

        # Compute place position: move the object away from the original pick target
        # Direction is from original pick object → current object, extended outward
        original_pick_obj = om.get_object_by_name(self.task._original_pickup_obj_name)
        direction = pickup_obj.position[:2] - original_pick_obj.position[:2]
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-3:
            direction = direction / dir_norm
        else:
            # Objects are co-located, pick a random direction
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])

        surface_z = pickup_obj_bottom_z  # same surface the object is sitting on

        # --- Grasp poses (pregrasp, grasp, lift) ---
        pregrasp_pose = grasp_pose_world.copy()
        pregrasp_pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp_pose[:3, 2]

        lift_pose = grasp_pose_world.copy()
        lift_pose[2, 3] = (
            surface_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )

        # --- Placement poses: try increasing distances until IK passes ---
        placement_pose_names = {"preplace", "place", "postplace"}
        pose_names = ["pregrasp", "grasp", "lift", "preplace", "place", "postplace"]
        best_distance = None

        for distance in [0.10, 0.12, 0.15, 0.18, 0.20]:
            place_xy = pickup_obj.position[:2] + distance * direction

            preplace_pose = grasp_pose_world.copy()
            preplace_pose[:2, 3] = place_xy
            preplace_pose[2, 3] = (
                surface_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
            )
            preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position

            place_pose = preplace_pose.copy()
            place_pose[2, 3] = surface_z + pickup_obj_clearance_offset

            postplace_pose = place_pose.copy()
            postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

            poses = [
                pregrasp_pose,
                grasp_pose_world,
                lift_pose,
                preplace_pose,
                place_pose,
                postplace_pose,
            ]
            ik_results = {
                name: self.check_feasible_ik(pose) for name, pose in zip(pose_names, poses)
            }
            failed = [name for name, ok in ik_results.items() if not ok]

            if not failed:
                best_distance = distance
                log.info(
                    f"[PACKING PLANNER] Place target: {distance:.2f}m from "
                    f"'{task_config.pickup_obj_name}' away from '{self.task._original_pickup_obj_name}'"
                )
                break
            elif all(f in placement_pose_names for f in failed):
                log.debug(
                    f"[PACKING PLANNER] Placement IK failed at {distance:.2f}m, trying further"
                )
                continue
            else:
                # Grasp/lift poses failed — distance won't help
                break

        if best_distance is None:
            log.warning(
                f"IK FAILED for: {', '.join(failed)}\n"
                + "\n".join(f"  {n}: {p[:3, 3]}" for n, p in zip(pose_names, poses))
            )
            raise ValueError(f"IK failed for {', '.join(failed)} pose(s)")

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

    def _skip_already_packed(self):
        """Skip past any objects that are already in the receptacle."""
        if not isinstance(self.task, PackingTask):
            return
        already_packed = self.task.objects_in_receptacle
        while self._current_object_index < len(self._packing_object_names):
            obj_name = self._packing_object_names[self._current_object_index]
            if obj_name in already_packed:
                log.info(f"[PACKING PLANNER] Skipping '{obj_name}' — already in receptacle")
                self._current_object_index += 1
            else:
                break

    def reset(self, reset_retries: bool = True):
        # Skip objects already in the receptacle (e.g. unintentionally picked up together)
        self._skip_already_packed()
        if self._current_object_index >= len(self._packing_object_names):
            log.info("[PACKING PLANNER] All objects already packed, nothing to do")
            self.action_primitives = []
            self.action_idx = 0
            return

        # Log packing status
        remaining = self._packing_object_names[self._current_object_index :]
        already_packed = (
            self.task.objects_in_receptacle if isinstance(self.task, PackingTask) else set()
        )
        log.info(
            f"[PACKING PLANNER] Full list: {self._packing_object_names} | "
            f"Already packed: {already_packed} | "
            f"Remaining: {remaining}"
        )

        # Point pickup_obj_name at the current object so _compute_target_poses uses it
        self.config.task_config.pickup_obj_name = self._packing_object_names[
            self._current_object_index
        ]
        obj_name = self.config.task_config.pickup_obj_name
        max_grasp_resamples = self.policy_config.max_retries
        for attempt in range(max_grasp_resamples + 1):
            try:
                log.info(
                    f"[PACKING PLANNER] Reset for object {self._current_object_index + 1}/{len(self._packing_object_names)}: "
                    f"'{obj_name}' (grasp attempt {attempt + 1}/{max_grasp_resamples + 1})"
                )
                super().reset(reset_retries=reset_retries)
                return
            except ValueError as e:
                log.warning(f"[PACKING PLANNER] Grasp/IK failed for '{obj_name}': {e}")
                if attempt < max_grasp_resamples:
                    log.info(f"[PACKING PLANNER] Resampling grasp for '{obj_name}'...")
                    continue
                # All attempts exhausted — skip this object
                log.warning(
                    f"[PACKING PLANNER] Skipping '{obj_name}' after {max_grasp_resamples + 1} grasp attempts"
                )
                self._advance_to_next_object()

    def _advance_to_next_object(self):
        """Skip current object and reset for the next one, or mark done."""
        self._current_object_index += 1
        if self._current_object_index < len(self._packing_object_names):
            next_name = self._packing_object_names[self._current_object_index]
            log.info(f"[PACKING PLANNER] Advancing to '{next_name}'")
            self.reset(reset_retries=True)
        else:
            log.info("[PACKING PLANNER] No more objects to pack")
            # Set up a minimal trajectory so get_action returns done immediately
            self.action_primitives = []
            self.action_idx = 0

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        action = super().get_action(info)

        if action.get("done"):
            self._current_object_index += 1
            if self._current_object_index < len(self._packing_object_names):
                prev_name = self._packing_object_names[self._current_object_index - 1]
                next_name = self._packing_object_names[self._current_object_index]
                log.info(
                    f"[PACKING PLANNER] Finished '{prev_name}' ({self._current_object_index}/{len(self._packing_object_names)}), "
                    f"advancing to '{next_name}'"
                )
                self.reset(reset_retries=True)
                return self.robot_view.get_noop_ctrl_dict()
            else:
                log.info(
                    f"[PACKING PLANNER] All {len(self._packing_object_names)} objects attempted"
                )

        return action
