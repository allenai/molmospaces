"""
Example configuration for RBY1 navigation to object data generation using the extracted task sampler.
This shows how the scene randomization functionality from the reference script has been
properly integrated into the modular task sampler architecture.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, RBY1MjcfCameraSystem
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig, BasePolicyConfig
from molmo_spaces.configs.task_configs import AllTaskConfigs, NavToObjTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    NavToObjTaskSamplerConfig,
)
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler
from molmo_spaces.utils.profiler_utils import Profiler


class NavToObjBaseConfig(MlSpacesExpConfig):
    """Base configuration for navigation to object data generation tasks."""

    # NOTE: will not work if used directly. Subclass examples in data_generation/configs.py

    # Experiment-level config parameters ---
    num_envs: int = 1
    use_passive_viewer: bool = False
    viewer_camera: None = None
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": np.array([0.0, 0.0, 0.5]),
    }
    policy_dt_ms: float = 200.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    task_horizon: int | None = 500
    record_videos: bool = False

    # --- Data generation settings ---
    num_threads: int = 1
    profile: bool = True

    # --- Task type configuration ---
    task_type: str = "nav_to_obj"  # Task type: nav_to_obj

    # --- ProcTHOR dataset configuration ---

    scene_dataset: str = "procthor-10k"
    data_split: str = "train"

    camera_config: CameraSystemConfig | None = RBY1MjcfCameraSystem()
    """Camera configuration - using new unified camera system"""

    task_sampler_config: BaseMujocoTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler
    )
    """Task sampler configuration (imported from task_sampler_configs.py)"""

    task_config: AllTaskConfigs = NavToObjTaskConfig(task_cls=NavToObjTask)

    task_config_preset: NavToObjTaskConfig | None = None

    policy_config: BasePolicyConfig = AStarNavToObjPolicyConfig()

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

        if self.profile and self.profiler is None:
            self.profiler = Profiler()

    @property
    def tag(self) -> str:
        return "nav_to_obj_datagen"

    # class SavedEpisode(Config):
    #     camera_config: RBY1MjcfCameraSystem | None = None  # Configuration for cameras and sensors
    #     robot_config: RBY1Config | None = None  # Configuration for the robot
    #     task_config: NavToObjTaskConfig | None = None  # Configuration for tasks
    #     task_cls_str: str | None = None
