from pathlib import Path

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import FrankaDroidCameraSystem
from molmo_spaces.configs.policy_configs import (
    PackingPlannerPolicyConfig,
    PickAndPlacePlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import BaseRobotConfig, CommonSenseFrankaRobotConfig
from molmo_spaces.configs.task_configs import PackingTaskConfig
from molmo_spaces.configs.task_sampler_configs import PackingTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.packing_task import PackingTask
from molmo_spaces.tasks.packing_task_sampler import PackingTaskSampler
from molmo_spaces.utils.constants.object_constants import PICK_AND_PLACE_OBJECTS


@register_config("PackingDataGenConfig")
class PackingDataGenConfig(PickBaseConfig):
    task_type: str = "packing"
    num_workers: int = 1
    task_sampler_config: PackingTaskSamplerConfig = PackingTaskSamplerConfig(
        task_sampler_class=PackingTaskSampler,
        pickup_types=PICK_AND_PLACE_OBJECTS,
        samples_per_house=20,
    )
    task_config: PackingTaskConfig = PackingTaskConfig(task_cls=PackingTask)
    policy_config: PickAndPlacePlannerPolicyConfig = PickAndPlacePlannerPolicyConfig()


@register_config("FrankaPackingDroidDataGenConfig")
class FrankaPackingDroidDataGenConfig(PackingDataGenConfig):
    """Data generation config for Franka packing task with DROID-style fixed cameras.

    Mirrors FrankaPickDroidDataGenConfig but for the packing task. Bakes in
    robot_config + camera_config so distributed launchers like
    manager_multi_machine_sqs_beaker.py - which don't take a `--robot` flag -
    get a fully populated config instead of crashing in setup_robot_scene with
    `'NoneType' object has no attribute 'name'`. Also bakes in the packing
    planner policy (run_pipeline.py sets this explicitly via
    `--task_type packing`; distributed launchers instantiate the config as-is).
    """

    robot_config: BaseRobotConfig = CommonSenseFrankaRobotConfig()
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    policy_config: PackingPlannerPolicyConfig = PackingPlannerPolicyConfig(place_z_offset=0.05)
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "packing_droid_v1"

    @property
    def tag(self) -> str:
        return "franka_packing_droid_datagen"
