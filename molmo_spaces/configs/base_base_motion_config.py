"""Base-motion evaluation config: how well a robot moves from one object to another.

Reuses NavToObjTask/NavToObjTaskSampler entirely -- the only difference from plain
nav_to_obj is start_near_object_probability=100, which makes the sampler place the
robot near a randomly chosen scene object (2-5m from the nav target by default)
instead of near the nav target itself. See NavToObjTaskSampler._sample_start_object_near.
"""

from __future__ import annotations

from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig
from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler


class BaseMotionBaseConfig(NavToObjBaseConfig):
    """Base configuration for base-motion (object-to-object locomotion) evaluation."""

    # NOTE: will not work if used directly. Subclass examples in data_generation/configs.py

    task_type: str = "base_motion"

    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler,
        start_near_object_probability=0,
        sample_trajectory_obstacles=True,
        num_trajectory_obstacles=10,
    )

    @property
    def tag(self) -> str:
        return "base_motion_datagen"
