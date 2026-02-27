"""Experiment configuration classes for commonsense benchmark tasks."""

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    AllCameraSystems,
    FrankaRandomizedD405D455CameraSystem,
)
from molmo_spaces.configs.commonsense_task_configs import BlockSupportTaskConfig
from molmo_spaces.configs.commonsense_task_configs import (
    BlockSupportTaskConfig,
    MugBallPickTaskConfig,
    SemanticGraspPickTaskConfig,
)
from molmo_spaces.configs.commonsense_task_sampler_configs import (
    BlockSupportTaskSamplerConfig,
    MugBallPickTaskSamplerConfig,
    SemanticGraspPickTaskSamplerConfig,
)
from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.tasks.commonsense_samplers.block_support_task_sampler import (
    BlockSupportTaskSampler,
)
from molmo_spaces.tasks.commonsense_samplers.mug_ball_pick_task_sampler import (
    MugBallPickTaskSampler,
)
from molmo_spaces.tasks.commonsense_samplers.semantic_grasp_pick_task_sampler import (
    SemanticGraspPickTaskSampler,
)
from molmo_spaces.tasks.commonsense_tasks.block_support_task import BlockSupportTask
from molmo_spaces.tasks.commonsense_tasks.mug_ball_pick_task import MugBallPickTask
from molmo_spaces.tasks.commonsense_tasks.semantic_grasp_pick_task import SemanticGraspPickTask


class BlockSupportConfig(PickBaseConfig):
    """Configuration for block support task data generation.
    The block support task involves:
    1. A red support cube (2cm x 2cm x 2cm) is placed near a random graspable object
    2. The robot must pick up the support cube
    3. The goal is to lift the cube above its starting position
    This configuration extends PickBaseConfig and uses BlockSupportTask and
    BlockSupportTaskSampler for task-specific functionality.
    """

    task_type: str = "block_support"

    scene_dataset: str = "procthor-objaverse-debug"  # Name of the scene dataset to load

    # Task sampler configuration - uses BlockSupportTaskSampler
    task_sampler_config: BlockSupportTaskSamplerConfig = BlockSupportTaskSamplerConfig(
        task_sampler_class=BlockSupportTaskSampler,
    )

    # Task configuration - uses BlockSupportTask
    task_config: BlockSupportTaskConfig = BlockSupportTaskConfig(task_cls=BlockSupportTask)

    # Camera configuration - inherited from PickBaseConfig
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    # Policy configuration - inherited from PickBaseConfig
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()

    @property
    def tag(self) -> str:
        return "franka_block_support_datagen"

    class SavedEpisode(Config):
        """Serializable configuration snapshot for block support tasks."""

        camera_config: AllCameraSystems | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: BlockSupportTaskConfig | None = None
        task_cls_str: str | None = None


class MugBallPickConfig(PickBaseConfig):
    """Configuration for mug-ball pick task data generation.

    Two iThor mugs are placed upside-down on a counter. One covers a ball,
    the other covers nothing. The robot must pick the correct mug.
    """

    task_type: str = "mug_ball_pick"

    scene_dataset: str = "procthor-objaverse-debug"

    task_sampler_config: MugBallPickTaskSamplerConfig = MugBallPickTaskSamplerConfig(
        task_sampler_class=MugBallPickTaskSampler,
    )

    task_config: MugBallPickTaskConfig = MugBallPickTaskConfig(task_cls=MugBallPickTask)

    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()

    @property
    def tag(self) -> str:
        return "franka_mug_ball_pick_datagen"

    class SavedEpisode(Config):
        """Serializable configuration snapshot for mug-ball pick tasks."""

        camera_config: AllCameraSystems | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: MugBallPickTaskConfig | None = None
        task_cls_str: str | None = None


class SemanticGraspPickConfig(PickBaseConfig):
    """Configuration for semantic grasp pick task data generation.

    The robot must pick up an object using a semantically correct grasp
    (e.g., a pan by its handle). Success requires both lifting the object
    and grasping it at a functionally appropriate location.
    """

    task_type: str = "semantic_grasp_pick"

    scene_dataset: str = "procthor-objaverse-debug"

    task_sampler_config: SemanticGraspPickTaskSamplerConfig = SemanticGraspPickTaskSamplerConfig(
        task_sampler_class=SemanticGraspPickTaskSampler,
    )

    task_config: SemanticGraspPickTaskConfig = SemanticGraspPickTaskConfig(
        task_cls=SemanticGraspPickTask,
    )

    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig(
        postgrasp_z_offset=0.20,  # 15cm lift to ensure clear separation from surface
    )

    @property
    def tag(self) -> str:
        return "franka_semantic_grasp_pick_datagen"

    class SavedEpisode(Config):
        """Serializable configuration snapshot for semantic grasp pick tasks."""

        camera_config: AllCameraSystems | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: SemanticGraspPickTaskConfig | None = None
        task_cls_str: str | None = None
