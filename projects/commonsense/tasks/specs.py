"""Benchmark schema specs for the CommonSense tasks.

Registered with the JSON benchmark eval by commonsense.tasks (register_json_eval_task),
which also adds them to benchmark_schema.ALL_TASK_SPEC_CLASSES so their fields are
copied into the task config.
"""

from pydantic import Field

from molmo_spaces.evaluation.benchmark_schema import PickTaskSpec


class MugBallPickTaskSpec(PickTaskSpec):
    """Task-specific parameters for the mug-ball (object permanence) pick task.

    Exists solely to carry `scene_settle_duration`. The mug-ball sampler sets it
    imperatively at datagen time (mug_ball_pick_task_sampler.py) and it was never part
    of any TaskSpec, so it did not survive serialization into the benchmark JSON. At
    eval it therefore fell back to the task-config default of 0.0, which disables the
    whole settle block in MugBallPickTask.step(): the scene is not given time to settle
    (the policy acts while the mugs are still falling) AND pickup_obj_start_pose /
    pickup_obj_goal_pose are never refreshed to the settled poses.

    Declaring it here fixes existing benchmarks without regenerating them, because
    apply_task_spec_to_config() runs model_validate() on the stored task dict and
    model_dump() fills in schema defaults for absent fields.
    """

    # Seconds of forced no-op actions while the mugs fall. Matches the value the
    # sampler used at datagen time.
    scene_settle_duration: float = 5.0


class BlockSupportTaskSpec(PickTaskSpec):
    """Task-specific parameters for block-support / block-stacking tasks.

    The blocks themselves are procedural primitives (see
    ``SceneModificationsSpec.primitive_objects``) keyed by the names in
    ``block_names``. ``pickup_obj_name`` and ``place_receptacle_name`` are
    inherited from PickTaskSpec and point at entries in ``block_names``.
    """

    place_receptacle_name: str | None = None
    # Ordered list of block body names (e.g. ["block_1", "block_2", ...]).
    block_names: list[str] = Field(default_factory=list)
