"""Task configuration classes for commonsense benchmark tasks."""

from molmo_spaces.configs.task_configs import PickTaskConfig


class BlockSupportTaskConfig(PickTaskConfig):
    """Configuration for block support task.
    The block support task involves picking up a support cube that is placed
    near a graspable object in the scene.
    """

    task_cls: type | None = None  # Will be set to BlockSupportTask
    place_receptacle_name: str | None = None
    block_names: list[str] = []  # Ordered list of block body names (block_1, block_2, ...)


class MugBallPickTaskConfig(PickTaskConfig):
    """Configuration for the mug-ball pick task.
    Two mugs are placed upside-down; one covers a ball. The robot must pick the
    correct mug.
    """

    task_cls: type | None = None  # Will be set to MugBallPickTask


class SemanticGraspPickTaskConfig(PickTaskConfig):
    """Configuration for the semantic grasp pick task.
    The robot must pick up an object using a semantically correct grasp
    (e.g., a pan by its handle). Uses KNN majority vote against pre-classified
    grasp data to determine correctness.
    """

    task_cls: type | None = None  # Will be set to SemanticGraspPickTask
    k_nearest_grasps: int = 5  # k for KNN majority vote
    require_no_receptacle_contact: bool = False  # If True, require object is not in contact with receptacle (can be unreliable for wide objects)
