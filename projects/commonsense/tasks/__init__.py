from commonsense.configs.commonsense_task_configs import (
    BlockSupportTaskConfig,
    MugBallPickTaskConfig,
    SemanticGraspPickTaskConfig,
)
from commonsense.configs.semantic_pick_prompts import get_semantic_pick_prompt
from commonsense.tasks.block_support_task import BlockSupportTask
from commonsense.tasks.mug_ball_pick_task import MugBallPickTask
from commonsense.tasks.semantic_grasp_pick_task import SemanticGraspPickTask
from commonsense.tasks.specs import BlockSupportTaskSpec, MugBallPickTaskSpec
from molmo_spaces.evaluation.benchmark_schema import PickTaskSpec
from molmo_spaces.tasks.json_eval_task_sampler import register_json_eval_task
from molmo_spaces.utils.prompt_samplers import PromptSamplerLearnedPolicy

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

# prompt_level ablation for learned policies (pi / dreamzero) on semantic_grasp_pick.
PromptSamplerLearnedPolicy.TASK_PROMPT_BUILDERS["semantic_grasp_pick"] = get_semantic_pick_prompt

__all__ = ["BlockSupportTask", "MugBallPickTask", "SemanticGraspPickTask"]
