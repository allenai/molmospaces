"""
Eval configuration classes for SynthVLA learned policy evaluation.

IMPORTANT: These configs are EXAMPLES of how to set up evaluation configs for use
with JSON benchmarks via molmo_spaces.evaluation.run_evaluation(). The anticipated
pattern is that users will create their own eval configs in their own repositories,
import run_evaluation from molmo_spaces.evaluation, and pass their config to it.

Example usage from an external repo:
    from molmo_spaces.evaluation import run_evaluation
    from my_repo.configs import MyPolicyEvalConfig

    results = run_evaluation(
        eval_config_cls=MyPolicyEvalConfig,
        benchmark_dir="/path/to/benchmark",
        checkpoint_path="/path/to/checkpoint",
    )

Eval configs provide:
- Robot config (factories for instantiation, gravcomp settings)
- Policy config (checkpoint path, camera names, action spec)
- Timing parameters (policy_dt_ms, ctrl_dt_ms, sim_dt_ms)

Episode-specific data (init_qpos, robot_base_pose, cameras, object_poses, task config)
comes from the JSON benchmark files, not from these configs. The benchmark JSON
is strictly authoritative for episode initialization.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs_baselines import (
    CAPPolicyConfig,
    DreamZeroPolicyConfig,
    PiPolicyConfig,
    TeleopPolicyConfig,
)
from molmo_spaces.configs.robot_configs import FrankaCAPRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_configs import BaseMujocoTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
)
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class JsonBenchmarkEvalConfig(MlSpacesExpConfig):
    """
    Minimal base config for JSON benchmark evaluation.

    This config is designed for use ONLY with JSON benchmarks. It provides
    the minimal infrastructure needed to run a learned policy against a
    benchmark where all episode-specific data (task type, cameras, robot poses,
    object poses, etc.) comes from the benchmark JSON.

    Subclass this and provide:
    - robot_config: Robot configuration for instantiation
    - policy_config: Your learned policy configuration

    DO NOT provide task_sampler_config or task_config - those are placeholders
    that will be overridden by the benchmark. If you accidentally try to use
    this config for data generation (not evaluation), it will fail because
    the task sampler/config are minimal stubs.

    Example:
        class MyPolicyBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
            robot_config = FrankaRobotConfig()
            policy_config = MyPolicyConfig(checkpoint_path="/path/to/ckpt")
    """

    # Required infrastructure - subclasses must provide robot_config and policy_config

    # Timing parameters - can be overridden per-policy as needed
    num_envs: int = 1
    num_workers: int = 1
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int = 500

    # Viewer config (usually disabled for eval)
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }

    # These are overridden by benchmark - provide placeholders to satisfy base class
    # DO NOT rely on these values; the benchmark JSON is authoritative.
    task_type: str = "pick"  # Overridden per-episode from benchmark
    scene_dataset: str = "procthor-10k"  # Overridden per-episode from benchmark
    data_split: str = "val"  # Overridden per-episode from benchmark
    camera_config: None = None  # Overridden per-episode from benchmark

    # Minimal stubs - these exist only to satisfy the base class.
    # JsonEvalTaskSampler replaces these entirely with benchmark data.
    # Note: task_sampler_class must be a valid class (not None) since pipeline.py
    # instantiates a worker-level task sampler. JsonEvalRunner overrides the per-episode
    # task sampler via get_episode_task_sampler, so this worker-level sampler is unused.
    task_sampler_config: BaseMujocoTaskSamplerConfig = BaseMujocoTaskSamplerConfig(
        task_sampler_class=BaseMujocoTaskSampler,
        house_inds=[0],  # Dummy value, overridden by JsonEvalRunner from benchmark
        samples_per_house=1,
        task_batch_size=1,
        max_tasks=10000,
        load_robot_from_file=True,
    )
    task_config: BaseMujocoTaskConfig = BaseMujocoTaskConfig(task_cls=None)

    # Output config
    output_dir: Path = Path("eval_output")
    use_wandb: bool = False
    wandb_project: str = "mlspaces-benchmark-eval"
    filter_for_successful_trajectories: bool = False

    @property
    def tag(self) -> str:
        return "json_benchmark_eval"


class PiPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PiPolicyConfig = PiPolicyConfig()
    policy_dt_ms: float = 100.0  # Match your model's expected control rate

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class CAPPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaCAPRobotConfig = FrankaCAPRobotConfig()
    policy_config: CAPPolicyConfig = CAPPolicyConfig()
    policy_dt_ms: float = 500.0  # Match your model's expected control rate

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class TeleopPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: TeleopPolicyConfig = TeleopPolicyConfig()
    policy_dt_ms: float = 40  # More responsive for teleoperation

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


class DreamZeroPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DreamZeroPolicyConfig = DreamZeroPolicyConfig()
    policy_dt_ms: float = 100.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
