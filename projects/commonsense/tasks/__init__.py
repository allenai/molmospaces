from commonsense.configs.commonsense_task_configs import (
    BlockSupportTaskConfig,
    MugBallPickTaskConfig,
    SemanticGraspPickTaskConfig,
)
from commonsense.tasks.block_support_task import BlockSupportTask
from commonsense.tasks.mug_ball_pick_task import MugBallPickTask
from commonsense.tasks.semantic_grasp_pick_task import SemanticGraspPickTask
from molmo_spaces.evaluation.benchmark_schema import (
    BlockSupportTaskSpec,
    MugBallPickTaskSpec,
    PickTaskSpec,
)
from molmo_spaces.tasks.json_eval_task_sampler import register_json_eval_task

register_json_eval_task(
    BlockSupportTask, "block_stacking", BlockSupportTaskConfig, BlockSupportTaskSpec
)
# Not PickTaskSpec: MugBallPickTaskSpec adds scene_settle_duration, without which
# MugBallPickTask.step() skips its settle window and never refreshes the object
# poses to their post-fall values.
register_json_eval_task(
    MugBallPickTask, "mug_ball_pick", MugBallPickTaskConfig, MugBallPickTaskSpec
)
register_json_eval_task(
    SemanticGraspPickTask, "semantic_grasp_pick", SemanticGraspPickTaskConfig, PickTaskSpec
)

__all__ = ["BlockSupportTask", "MugBallPickTask", "SemanticGraspPickTask"]
