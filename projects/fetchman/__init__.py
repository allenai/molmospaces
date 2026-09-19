"""FetchMan (g1_molmo) env / task / task-sampler port.

Everything this package needed from its own package-level shim (ASSETS_DIR,
GRASPS_DIR, grasp_source_dir) now comes from molmo_spaces proper --
molmo_spaces_constants.ASSETS_DIR and utils/grasps.py's fetchman_* helpers, and
gold's pre-compile MjSpec edits from env/scene/spec_ops.py, which the
native scene build shares.

The task and task sampler implement molmo_spaces' own abstractions
(BaseMujocoTask / BaseMujocoTaskSampler) while keeping gold's sampling verbatim
-- see fetchman/tasks/task_g1ms.py and pick_task_sampler_g1ms.py.

See fetchman/scripts/check_gold_parity.py for what is left here and why it
cannot be deleted yet.
"""
