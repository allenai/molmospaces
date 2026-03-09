"""Configs for common sense benchmark tasks."""

import math

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

    # Robot z offset (higher than default to support multi-block stacking)
    robot_object_z_offset: float = -0.58
    robot_object_z_offset_random_min: float = 0.0
    robot_object_z_offset_random_max: float = 0.0

    # Robot placement radius (max distance from robot base to each block)
    base_pose_sampling_radius_range: tuple[float, float] = (0.0, 0.7)
    robot_placement_rotation_range_rad: float = math.radians(20)

    # Number of blocks to use (sampled uniformly from this range, inclusive)
    num_blocks_range: tuple[int, int] = (3, 3)

    # Block placement distances (relative to a reference graspable object)
    min_block_placement_dist: float = 0.05
    max_block_placement_dist: float = 0.15
    max_block_placement_tries: int = 100


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
