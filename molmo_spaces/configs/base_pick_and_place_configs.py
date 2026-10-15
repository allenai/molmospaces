from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig, PickAndPlacePlannerPolicyConfig
from molmo_spaces.configs.task_configs import AllTaskConfigs, PickAndPlaceTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
)
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler


class PickAndPlaceDataGenConfig(PickBaseConfig):
    task_type: str = "pick_and_place"
    num_workers: int = 1
    task_sampler_config: BaseMujocoTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        pickup_types=[],
        samples_per_house=20,
    )
    task_config: AllTaskConfigs = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)
    policy_config: BasePolicyConfig = PickAndPlacePlannerPolicyConfig()

    # class SavedEpisode(Config):
    #     camera_config: AllCameraSystems | None = None  # Configuration for cameras and sensors
    #     robot_config: FrankaRobotConfig | None = None  # Configuration for the robot
    #     task_config: PickAndPlaceTaskConfig | None = None  # Configuration for tasks
    #     task_cls_str: str | None = None
