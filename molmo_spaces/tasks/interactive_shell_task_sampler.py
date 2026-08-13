"""Task sampler for InteractiveShellTask.

Copies the scene-setup pattern from `PickAndPlaceMultiTaskSampler` (load a house,
place the robot, add receptacles) but instead of pre-planning a fixed LLM-generated
action sequence, hands back an `InteractiveShellTask` that lets a human drive the
robot live via `nav_to`/`pick`/`pick_and_place`/`open_object`/`close_object`.
"""

import logging

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.interactive_shell_task import InteractiveShellTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler

log = logging.getLogger(__name__)


class InteractiveShellTaskSampler(PickAndPlaceTaskSampler):
    """Sets up a scene exactly like `PickAndPlaceTaskSampler`, then hands off to a shell."""

    def _filter_place_target(self, env, pickup_obj_name, place_target_name) -> bool:
        """Skip AbstractPickAndPlaceObjectTargetTaskSampler's "pickup object must be
        smaller than the place target" check.

        That constraint exists for real pick-and-place tasks (can't place a bowl
        inside a smaller bowl), but sampling here still runs the full pick-and-place
        pipeline even when the user only intends to call pick() -- which needs a
        pickup object, not a valid place target. Skipping it matters in particular
        for pickup_types restricted to one category (e.g. ["Bowl"]): the sampled
        place target then often ends up being another instance of the same
        category, similarly sized, which always fails this check and makes
        sampling fail near-100% of the time regardless of scene/seed.
        """
        return True

    def _sample_task(self, env: CPUMujocoEnv) -> InteractiveShellTask:
        self._configure_pick_and_place(env)
        task = InteractiveShellTask(env, self.config)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task
