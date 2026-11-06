"""JSON benchmark eval configs for the CommonSense planner policies."""

from commonsense.configs.commonsense_policy_configs import PackingPlannerPolicyConfig

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


@register_config("PackingPlannerEvalConfig")
class PackingPlannerEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PackingPlannerPolicyConfig = PackingPlannerPolicyConfig()
    task_horizon: int = 1500
    policy_dt_ms: float = 200.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
