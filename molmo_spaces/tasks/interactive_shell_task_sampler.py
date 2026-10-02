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
from molmo_spaces.tasks.pick_task_sampler_g1 import PickTaskSamplerG1

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


class InteractiveShellPickTaskSampler(PickTaskSamplerG1):
    """Pick-only shell sampler: sets up a scene like `PickTaskSamplerG1`, then
    hands off to the shell. The G1 counterpart of `InteractiveShellTaskSampler`.

    The pick-and-place variant stages three candidate place receptacles into
    every scene, and for a single-category `pickup_types` (e.g. ["Bowl"]) the
    one it selects is another instance of that same category -- a bowl to place
    a bowl in. That target is never used: `InteractiveShellTask.pick_and_place`
    builds its own `PickAndPlaceTaskConfig` from the receptacle the caller
    names, and does not read the sampled one. The other two candidates are
    parked on a staging floor 35m above the house and never touched.

    So this sampler drops the staging entirely. It costs the scene 5 bodies,
    52 geoms and 3 free joints (21 qpos) less, removes the place-target
    selection from every reset, and makes a natively built scene structurally
    identical to the FetchMan port's -- which is what the gold-parity
    comparison needs. `pick_and_place()` is unaffected and still places onto
    any object in the house.
    """

    def _task_cls(self) -> type[InteractiveShellTask]:
        return InteractiveShellTask

    def _sample_task(self, env: CPUMujocoEnv) -> InteractiveShellTask:
        task = super()._sample_task(env)
        log.info(f"Sampled task '{task.get_task_description()}'")
        return task


# Every sampler that hands off to InteractiveShellTask.run_shell(). scripts/
# datagen/run_pipeline.py dispatches on this rather than on one class, so a
# config may pick either shell sampler.
INTERACTIVE_SHELL_SAMPLERS: tuple[type, ...] = (
    InteractiveShellTaskSampler,
    InteractiveShellPickTaskSampler,
)
