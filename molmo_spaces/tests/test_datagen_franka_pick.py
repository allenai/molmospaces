import shutil
from pathlib import Path

import pytest

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig, FrankaDroidCameraSystem
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    PickTaskSamplerConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output" / "franka_droid_pick_test"


class FrankaDroidPickTestConfig(PickBaseConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: CameraSystemConfig | None = FrankaDroidCameraSystem()
    output_dir: Path = TEST_OUTPUT_DIR

    filter_for_successful_trajectories: bool = False

    num_envs: int = 1
    seed: int | None = 42
    num_workers: int = 4

    scene_dataset: str = "procthor-10k"
    data_split: str = "train"

    task_sampler_config: BaseMujocoTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        house_inds=[1],
        samples_per_house=4,
        episodes_per_batch=4,
    )

    @property
    def tag(self) -> str:
        return "franka_droid_pick_test"


@pytest.mark.parametrize(
    ["split", "house"],
    [
        ("train", 1),
        ("train", 11),
        ("train", 101),
        ("train", 1001),
    ],
)
def test_datagen_franka_droid_pick(split: str, house: int):
    exp_config = FrankaDroidPickTestConfig()
    exp_config.data_split = split
    exp_config.task_sampler_config.house_inds = [house]

    runner = ParallelRolloutRunner(exp_config)
    success_count, total_count = runner.run()

    assert total_count > 0, (
        "It seems the runner didn't execute. Remove the old test_output folder for a clean run"
    )
    success_rate = success_count / total_count

    assert success_rate >= 0.5, (
        "Must at least get 50% success rate for the test to be considered successfull"
    )


@pytest.fixture(scope="module", autouse=True)
def setup_fn():
    if TEST_OUTPUT_DIR.is_dir():
        shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)

    print("TEST STARTED !!!!!")

    yield

    print("TEST FINISHED !!!!!")
