"""
Example configuration for Franka pick-and-place data generation using the extracted task sampler.
This shows how the scene randomization functionality from the reference script has been
properly integrated into the modular task sampler architecture.
"""

from __future__ import annotations

from typing import Any

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import (
    CameraSystemConfig,
    FrankaRandomizedD405D455CameraSystem,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig, PickPlannerPolicyConfig
from molmo_spaces.configs.task_configs import AllTaskConfigs, PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    PickTaskSamplerConfig,
)
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.profiler_utils import Profiler


class PickBaseConfig(MlSpacesExpConfig):
    # --- Experiment-level config parameters ---
    num_envs: int = 1
    use_passive_viewer: bool = False
    viewer_camera: None = None
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }
    policy_dt_ms: float = 66.0  # ~15hz
    ctrl_dt_ms: float = 2.0  # control time step
    sim_dt_ms: float = 2.0  # simulation time step
    seed: int | None = None  # Random seed for task sampling (if None, generates random seed)
    task_horizon: int | None = 500  # Maximum number of steps per episode (if None, no time limit)

    # --- Data generation settings ---
    num_workers: int = 1
    profile: bool = True
    wandb_project: str | None = "molmo-spaces-data-generation"

    # --- Task type configuration ---
    task_type: str = "pick"

    # --- ProcTHOR dataset configuration ---
    scene_dataset: str = "procthor-10k"
    data_split: str = "train"

    camera_config: CameraSystemConfig | None = FrankaRandomizedD405D455CameraSystem()

    task_sampler_config: BaseMujocoTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler
    )

    task_config: AllTaskConfigs = PickTaskConfig(task_cls=PickTask)

    policy_config: BasePolicyConfig = PickPlannerPolicyConfig()

    def _init_policy_config(self) -> BasePolicyConfig:
        return self.policy_config

    def model_post_init(self, _context: Any) -> None:
        super().model_post_init(_context)

        try:
            self.policy_config = self._init_policy_config()
        except RuntimeError as e:
            # Check if this is a CUDA/GPU-related error
            error_msg = str(e)
            if "NVIDIA" in error_msg or "CUDA" in error_msg or "GPU" in error_msg:
                # No GPU available - this is expected on manager nodes that just coordinate jobs
                # Policy config will be initialized later on worker nodes that have GPUs
                print(
                    f"Warning: Skipping policy config initialization due to missing GPU: {error_msg}"
                )
                self.policy_config = None  # pyright: ignore[reportAttributeAccessIssue] # ty: ignore
            else:
                raise

        # Auto-create profiler instance if profiling is enabled
        if self.profile and self.profiler is None:
            self.profiler = Profiler()

    @property
    def tag(self) -> str:
        return "franka_pick_datagen"
