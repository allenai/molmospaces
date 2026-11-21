from pathlib import Path

import pytest

from molmo_spaces.configs.base_pick_and_place_configs import PickAndPlaceDataGenConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, FrankaDroidCameraSystem
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    PickAndPlaceTaskSamplerConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler


class FrankaDroidPickAndPlaceTestConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: CameraSystemConfig | None = FrankaDroidCameraSystem()
    output_dir: Path = Path(__file__).parent / "test_output" / "franka_droid_pick_and_place_test"

    num_envs: int = 1
    seed: int | None = 42
    num_workers: int = 4

    scene_dataset: str = "procthor-10k"
    data_split: str = "train"

    task_sampler_config: BaseMujocoTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        house_inds=[1],
        episodes_per_batch=4,
        samples_per_house=4,
    )

    @property
    def tag(self) -> str:
        return "franka_droid_pick_and_place_test"


@pytest.mark.parametrize(
    ["split", "house"],
    [
        ("train", 1),
        ("train", 101),
        # ("train", 1001),
    ],
)
def test_datagen_franka_droid_pick_and_place(split: str, house: int):
    exp_config = FrankaDroidPickAndPlaceTestConfig()
    exp_config.data_split = split
    exp_config.task_sampler_config.house_inds = [house]

    runner = ParallelRolloutRunner(exp_config)
    success_count, total_count = runner.run()

    assert success_count == total_count
