"""Configs for common sense benchmark tasks."""

from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig


class BlockSupportTaskSamplerConfig(PickTaskSamplerConfig):
    """Configuration for block support task sampler.
    The block support task sampler extends the PickTaskSampler to place a red
    support cube near a randomly selected graspable object in the scene. The
    cube becomes the target pickup object for the task.
    """

    task_sampler_class: type | None = None  # Will be set to BlockSupportTaskSampler

    # Disable cluttering since we focus on the support cube
    clutter_scene_around_target_object: bool = False
    max_robot_to_block_dist: float = 0.7


class MugBallPickTaskSamplerConfig(PickTaskSamplerConfig):
    """Configuration for the mug-ball pick task sampler."""

    task_sampler_class: type | None = None  # Will be set to MugBallPickTaskSampler

    clutter_scene_around_target_object: bool = False
    max_robot_to_block_dist: float = 0.7


class SemanticGraspPickTaskSamplerConfig(PickTaskSamplerConfig):
    """Configuration for the semantic grasp pick task sampler."""

    task_sampler_class: type | None = None  # Will be set to SemanticGraspPickTaskSampler

    clutter_scene_around_target_object: bool = False
    pickup_types: list[str] = [
        # "ButterKnife", "Cup", "Fork", "Knife", "Ladle", "Mug", "Pan", "Spoon",
        "Pan",
    ]
