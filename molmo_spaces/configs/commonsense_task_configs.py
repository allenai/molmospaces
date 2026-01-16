"""Task configuration classes for commonsense benchmark tasks."""

from molmo_spaces.configs.task_configs import PickTaskConfig


class BlockSupportTaskConfig(PickTaskConfig):
    """Configuration for block support task.
    The block support task involves picking up a support cube that is placed
    near a graspable object in the scene.
    """

    task_cls: type | None = None  # Will be set to BlockSupportTask
    place_receptacle_name: str | None = None
