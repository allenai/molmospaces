import os
import shutil
import warnings
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
from molmo_spaces.tests.utils import get_success_counts_from_run, push_datagen_results_to_wandb
from molmo_spaces.utils.mp_logging import _human_log_level_to_int

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output" / "franka_droid_pick_and_place_test"
TEST_HOUSES_SPLIT = ["train", "train", "train", "train"]
TEST_HOUSES_INDICES = [1, 11, 21, 31]


class FrankaDroidPickAndPlaceTestConfig(PickAndPlaceDataGenConfig):
    robot_config: BaseRobotConfig = FrankaRobotConfig()
    camera_config: CameraSystemConfig | None = FrankaDroidCameraSystem()
    output_dir: Path = TEST_OUTPUT_DIR

    num_envs: int = 1
    seed: int | None = 42
    num_workers: int = 4

    filter_for_successful_trajectories: bool = False

    scene_dataset: str = "procthor-10k"
    data_split: str = "train"

    log_level: str = "none"

    task_sampler_config: BaseMujocoTaskSamplerConfig = PickAndPlaceTaskSamplerConfig(
        task_sampler_class=PickAndPlaceTaskSampler,
        house_inds=[1],
        samples_per_house=4,
        episodes_per_batch=4,
    )

    @property
    def tag(self) -> str:
        return "franka_droid_pick_and_place_test"


@pytest.mark.parametrize(
    ["split", "house"], [(split, idx) for split, idx in zip(TEST_HOUSES_SPLIT, TEST_HOUSES_INDICES)]
)
def test_datagen_franka_droid_pick_and_place(num_workers: int, split: str, house: int):
    exp_config = FrankaDroidPickAndPlaceTestConfig()
    exp_config.num_workers = num_workers
    exp_config.data_split = split
    exp_config.task_sampler_config.house_inds = [house]

    runner = ParallelRolloutRunner(exp_config)
    runner.logger.setLevel(_human_log_level_to_int(exp_config.log_level))

    success_count, total_count = runner.run()
    success_count_run, total_count_run = get_success_counts_from_run(exp_config.output_dir, house)

    assert total_count > 0 and total_count_run > 0, (
        "It seems the runner didn't execute. Remove the old test_output folder for a clean run"
    )
    success_rate = success_count / total_count
    success_rate_run = success_count_run / total_count_run

    assert success_rate >= 0.5 or success_rate_run >= 0.5, (
        "Must at least get 50% success rate for the test to be considered successfull"
    )


@pytest.fixture(scope="module", autouse=True)
def setup_fn(pytestconfig):
    capture_manager = pytestconfig.pluginmanager.getplugin("capturemanager")
    push_wandb = pytestconfig.getoption("--push-wandb")
    if push_wandb and not os.environ.get("WANDB_API_KEY"):
        warnings.warn(
            "--push-wandb was given but WANDB_API_KEY is not set (e.g. PR from a fork); "
            "skipping the wandb push",
            stacklevel=1,
        )
        push_wandb = False

    if TEST_OUTPUT_DIR.is_dir():
        shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)

    yield

    if push_wandb:
        capture_manager.suspend_global_capture(in_=True)
        success_rates = push_datagen_results_to_wandb(TEST_OUTPUT_DIR, TEST_HOUSES_INDICES)

        for idx, house_id in enumerate(TEST_HOUSES_INDICES):
            print(f"Success-rate(house-{house_id}) = {success_rates[idx] * 100} %")

        capture_manager.resume_global_capture()
