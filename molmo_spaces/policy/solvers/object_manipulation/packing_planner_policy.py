import logging
from typing import Any

from molmo_spaces.configs.abstract_exp_config import MjThorExpConfig
from molmo_spaces.configs.task_configs import PackingTaskConfig
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

log = logging.getLogger(__name__)


class PackingPlannerPolicy(PickAndPlacePlannerPolicy):
    """Packs multiple objects (clutter + original) into a box sequentially."""

    def __init__(self, config: MjThorExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        task_config = config.task_config
        if isinstance(task_config, PackingTaskConfig) and task_config.packing_object_names:
            self._packing_object_names = task_config.packing_object_names
        else:
            self._packing_object_names = [task_config.pickup_obj_name]
        self._current_object_index = 0
        log.info(
            f"[PACKING PLANNER] Will pack {len(self._packing_object_names)} objects: {self._packing_object_names}"
        )

    def _is_last_object(self) -> bool:
        return self._current_object_index >= len(self._packing_object_names) - 1

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        trajectory = super()._compute_trajectory()
        if not self._is_last_object():
            # Strip go_home and noop for intermediate objects to save time
            trajectory = [
                p for p in trajectory if not isinstance(p, (JointMoveSequence, NoopAction))
            ]
            log.info("[PACKING PLANNER] Skipping go_home + noop for intermediate object")
        return trajectory

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
