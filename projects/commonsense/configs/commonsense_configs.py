"""Experiment configuration classes for commonsense benchmark tasks."""

from pathlib import Path

import numpy as np
from commonsense.configs.commonsense_policy_configs import (
    BlockStackingPolicyConfig,
    PackingPlannerPolicyConfig,
)
from commonsense.configs.commonsense_task_configs import (
    BlockSupportTaskConfig,
    MugBallPickTaskConfig,
    SemanticGraspPickTaskConfig,
)
from commonsense.configs.commonsense_task_sampler_configs import (
    BlockSupportTaskSamplerConfig,
    MugBallPickTaskSamplerConfig,
    SemanticGraspPickTaskSamplerConfig,
)
from commonsense.tasks.block_support_task import BlockSupportTask
from commonsense.tasks.block_support_task_sampler import (
    BlockSupportTaskSampler,
)
from commonsense.tasks.mug_ball_pick_task import MugBallPickTask
from commonsense.tasks.mug_ball_pick_task_sampler import (
    MugBallPickTaskSampler,
)
from commonsense.tasks.packing_task_sampler import (
    CommonSensePackingTaskSampler,
    CommonSensePackingTaskSamplerConfig,
)
from commonsense.tasks.semantic_grasp_pick_task import SemanticGraspPickTask
from commonsense.tasks.semantic_grasp_pick_task_sampler import (
    SemanticGraspPickTaskSampler,
)

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    AllCameraSystems,
    FrankaDroidCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig, PickPlannerPolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_sampler_configs import BaseMujocoTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS


class CommonSenseFrankaRobotConfig(FrankaRobotConfig):
    """Franka config for CommonSense benchmark datagen.

    Uses a raised-elbow home pose (clears tall clutter and open box flaps) and
    disables initial-pose noise so every episode starts from the exact same arm
    configuration. Kept as an override so the shared FrankaRobotConfig defaults
    stay unchanged for other users.
    """

    init_qpos: dict[str, list[float]] = {
        "arm": [0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        # Arm noise disabled (was [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175] rad,
        # Jacobian-weighted to keep TCP displacement <=10cm).
        "arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


@register_config("BlockSupportConfig")
class BlockSupportConfig(PickBaseConfig):
    """Configuration for block support task data generation.
    The block support task involves:
    1. A red support cube (2cm x 2cm x 2cm) is placed near a random graspable object
    2. The robot must pick up the support cube
    3. The goal is to lift the cube above its starting position
    This configuration extends PickBaseConfig and uses BlockSupportTask and
    BlockSupportTaskSampler for task-specific functionality.
    """

    task_type: str = "block_stacking"
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "block_stacking"

    # Oracle-terminate the rollout as soon as the success condition fires
    # so flicker at the end of a barely-successful trajectory can't flip
    # the outcome back to failure (see pipeline.run_single_rollout).
    end_on_success: bool = True

    scene_dataset: str = "procthor-objaverse-debug"  # Name of the scene dataset to load

    # Droid setup (matches run_pipeline.py's `--robot droid` branch). Without
    # an explicit robot_config here, distributed launchers like
    # manager_multi_machine_sqs_beaker.py — which don't take a `--robot` flag —
    # would leave robot_config=None and crash in setup_robot_scene with
    # `'NoneType' object has no attribute 'name'`.
    robot_config: CommonSenseFrankaRobotConfig = CommonSenseFrankaRobotConfig()

    # Task sampler configuration - uses BlockSupportTaskSampler
    task_sampler_config: BlockSupportTaskSamplerConfig = BlockSupportTaskSamplerConfig(
        task_sampler_class=BlockSupportTaskSampler,
    )

    # Task configuration - uses BlockSupportTask
    task_config: BlockSupportTaskConfig = BlockSupportTaskConfig(task_cls=BlockSupportTask)

    # Camera configuration - inherited from PickBaseConfig
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    # Block-stacking task needs the dedicated stacking planner, not the pick
    # planner inherited from PickBaseConfig. Locally, run_pipeline.py sets this
    # explicitly; distributed launchers instantiate the config as-is, so the
    # default must be correct here too.
    policy_config: BlockStackingPolicyConfig = BlockStackingPolicyConfig()

    @property
    def tag(self) -> str:
        return "franka_block_support_datagen"

    class SavedEpisode(Config):
        """Serializable configuration snapshot for block support tasks."""

        camera_config: AllCameraSystems | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: BlockSupportTaskConfig | None = None
        task_cls_str: str | None = None


@register_config("MugBallPickConfig")
class MugBallPickConfig(PickBaseConfig):
    """Configuration for mug-ball pick task data generation.

    Two iThor mugs are placed upside-down on a counter. One covers a ball,
    the other covers nothing. The robot must pick the correct mug.
    """

    task_type: str = "mug_ball_pick"
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "mug_ball_pick"

    # Oracle-terminate the rollout as soon as the success condition fires
    # so flicker at the end of a barely-successful trajectory can't flip
    # the outcome back to failure (see pipeline.run_single_rollout).
    end_on_success: bool = True

    scene_dataset: str = "procthor-objaverse-debug"

    # Droid setup (matches run_pipeline.py's `--robot droid` branch). Without
    # an explicit robot_config here, distributed launchers like
    # manager_multi_machine_sqs_beaker.py — which don't take a `--robot` flag —
    # would leave robot_config=None and crash in setup_robot_scene with
    # `'NoneType' object has no attribute 'name'`.
    robot_config: CommonSenseFrankaRobotConfig = CommonSenseFrankaRobotConfig()

    task_sampler_config: MugBallPickTaskSamplerConfig = MugBallPickTaskSamplerConfig(
        task_sampler_class=MugBallPickTaskSampler,
    )

    task_config: MugBallPickTaskConfig = MugBallPickTaskConfig(task_cls=MugBallPickTask)

    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem(
        img_resolution=(1280, 720),
    )

    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig(
        # Upside-down mug grips are weak; at the 8cm default the mug
        # routinely slips and only clears the 1cm success threshold by a
        # fraction of a mm. Lifting to 15cm buys enough margin that a
        # partial slip still leaves the mug well above threshold.
        postgrasp_z_offset=0.15,
        phase_timeout=20.0,
    )

    @property
    def tag(self) -> str:
        return "franka_mug_ball_pick_datagen"

    class SavedEpisode(Config):
        """Serializable configuration snapshot for mug-ball pick tasks."""

        camera_config: AllCameraSystems | None = None
        robot_config: BaseRobotConfig | None = None
        task_config: MugBallPickTaskConfig | None = None
        task_cls_str: str | None = None


@register_config("SemanticGraspPickConfig")
class SemanticGraspPickConfig(PickBaseConfig):
    """Configuration for semantic grasp pick task data generation.

    The robot must pick up an object using a semantically correct grasp
    (e.g., a pan by its handle). Success requires both lifting the object
    and grasping it at a functionally appropriate location.
    """

    task_type: str = "semantic_grasp_pick"
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "semantic_grasp_pick"

    # Oracle-terminate the rollout as soon as the success condition fires
    # so flicker at the end of a barely-successful trajectory can't flip
    # the outcome back to failure (see pipeline.run_single_rollout).
    end_on_success: bool = True

    scene_dataset: str = "procthor-objaverse-debug"

    # Droid setup (matches run_pipeline.py's `--robot droid` branch). Without
    # an explicit robot_config here, distributed launchers like
    # manager_multi_machine_sqs_beaker.py — which don't take a `--robot` flag —
    # would leave robot_config=None and crash in setup_robot_scene with
    # `'NoneType' object has no attribute 'name'`.
    robot_config: CommonSenseFrankaRobotConfig = CommonSenseFrankaRobotConfig()

    task_sampler_config: SemanticGraspPickTaskSamplerConfig = SemanticGraspPickTaskSamplerConfig(
        task_sampler_class=SemanticGraspPickTaskSampler,
    )

    task_config: SemanticGraspPickTaskConfig = SemanticGraspPickTaskConfig(
        task_cls=SemanticGraspPickTask,
    )

    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem(
        img_resolution=(1280, 720),
    )

    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig(
        postgrasp_z_offset=0.20,  # 15cm lift to ensure clear separation from surface
        phase_timeout=20.0,
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


@register_config("FrankaPackingDroidDataGenConfig")
class FrankaPackingDroidDataGenConfig(PackingDataGenConfig):
    """Data generation config for Franka packing task with DROID-style fixed cameras.

    Mirrors FrankaPickDroidDataGenConfig but for the packing task. Bakes in
    robot_config + camera_config so distributed launchers like
    manager_multi_machine_sqs_beaker.py - which don't take a `--robot` flag -
    get a fully populated config instead of crashing in setup_robot_scene with
    `'NoneType' object has no attribute 'name'`. Also bakes in the packing
    planner policy and the CommonSense packing sampler (robot placed near both the
    object and the box, scene cluttered, every clutter object packed).
    """

    task_sampler_config: BaseMujocoTaskSamplerConfig = CommonSensePackingTaskSamplerConfig(
        task_sampler_class=CommonSensePackingTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=20,
    )
    robot_config: BaseRobotConfig = CommonSenseFrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    policy_config: BasePolicyConfig = PackingPlannerPolicyConfig(place_z_offset=0.05)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "packing_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_packing_droid_datagen"
