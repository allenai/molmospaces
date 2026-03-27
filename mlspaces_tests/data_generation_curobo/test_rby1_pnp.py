"""
Tests for RBY1 pick and place data generation with CuRobo planner.

TODO: Future test improvements
- Test observation determinism across multiple resets
- Test action space validation (valid ranges)
- Test reward calculation correctness
- Test terminal/truncated conditions
- Test randomization when enabled (currently disabled for determinism)
- Test multi-environment batching (currently only single env)
- Verify trajectory data quality in integration tests (not just file existence)
- Test CuRobo planner-specific behavior (collision avoidance, path quality)
"""

import datetime
import os
from pathlib import Path

import numpy as np
import pytest

from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.utils.test_utils import (
    run_task_for_steps_with_observations,
    verify_and_compare_camera_observations,
    verify_and_compare_camera_observations_after_steps,
    verify_video_fps,
)

TEST_DATA_DIR = get_resource_manager().symlink_dir / "test_data" / "rby1_pnp"
TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
DEBUG_IMAGES_DIR = Path(__file__).resolve().parent / "test_debug_images"


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Set up environment variables for all tests."""
    yield


@pytest.fixture(scope="session")
def config():
    """Create test-specific config instance (shared across all tests)."""
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        RBY1PickAndPlaceDataGenConfig,
    )
    from molmo_spaces.utils.profiler_utils import Profiler

    config = RBY1PickAndPlaceDataGenConfig()
    config.seed = 0
    config.task_horizon = 6
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    config.profiler = Profiler()  # Manually create profiler since profile was set after init
    config.task_sampler_config.samples_per_house = 1
    config.task_sampler_config.house_inds = [0]
    config.num_workers = 1
    config.filter_for_successful_trajectories = False
    config.robot_config.action_noise_config.enabled = False
    config.policy_config.batch_size = 1

    # Use local CuroboPlanner instead of remote gRPC server
    config.policy_config.server_urls = []

    return config


@pytest.fixture(scope="session")
def task_sampler(config):
    """Create and initialize task sampler once for all tests (expensive initialization)."""
    task_sampler_class = config.task_sampler_config.task_sampler_class
    task_sampler = task_sampler_class(config)
    task_sampler.reset()
    return task_sampler


@pytest.fixture(scope="session")
def task(task_sampler):
    """Sample task once for all tests (expensive operation)."""
    task = task_sampler.sample_task()
    return task


@pytest.fixture(scope="session")
def policy_results(config, task):
    """Run policy once for all tests (expensive operation)."""
    # Reset task to initial state
    task.reset()

    # Instantiate policy with config and task
    policy_config = config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(config, task)
    policy.reset()

    # Run policy for 10 steps and get both qpos and observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(task, policy, num_steps=10, profiler=config.profiler)
    )

    return {
        "initial_qpos": initial_qpos,
        "final_qpos": final_qpos,
        "initial_obs_dict": initial_obs_dict,
        "final_obs_dict": final_obs_dict,
        "arm_side": policy.arm_side,
    }


def test_imports():
    # fmt: off
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.policy.solvers.object_manipulation.curobo_pick_and_place_planner_policy import (
        CuroboPickAndPlacePlannerPolicy,
    )
    from molmo_spaces.tasks.pick_and_place_task import (
        PickAndPlaceTask,
    )
    from molmo_spaces.tasks.pick_and_place_task_sampler import (
        PickAndPlaceTaskSampler,
    )
    # fmt: on
    assert PickAndPlaceTaskSampler is not None
    assert PickAndPlaceTask is not None
    assert CuroboPickAndPlacePlannerPolicy is not None
    assert ParallelRolloutRunner is not None


def test_task_sampler(config, task_sampler, task):
    """Test task sampler state after sampling."""
    from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

    # Check base class state after sampling
    assert task_sampler._samples_per_current_house == 1  # Already sampled one task
    assert task_sampler._house_iterator_index == -1  # Stays at -1 until samples_per_house exhausted

    # Verify the sampled task is the correct type
    assert task is not None
    assert isinstance(task, PickAndPlaceTask), f"Expected PickAndPlaceTask, got {type(task)}"

    # Verify task has expected attributes
    assert hasattr(task, "env"), "Task should have env attribute"
    assert hasattr(task, "sensor_suite"), "Task should have sensor_suite attribute"


def test_task_observations(task):
    obs = task.reset()  # Returns (observation, info) tuple

    # RBY1 has head_camera and wrist cameras
    expected_cameras = ["head_camera", "wrist_camera_l", "wrist_camera_r"]

    # Verify observations match saved test data
    # RBY1 uses resolution (640, 480) which means width=640, height=480
    # numpy shape is (height, width, channels) = (480, 640, 3)
    verify_and_compare_camera_observations(
        obs=obs,
        sensor_suite=task.sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="rby1_pnp_obs_",
        expected_cameras=expected_cameras,
        debug_images_dir=DEBUG_IMAGES_DIR / "rby1_pnp",
        debug_prefix="pnp_obs",
        expected_shape=(1024, 576, 3),
        atol=1e-3,
        rtol=1e-3,
    )


def test_policy_determinism(policy_results):
    """Test that the policy executes deterministically and produces expected outputs."""
    initial_qpos = policy_results["initial_qpos"]
    final_qpos = policy_results["final_qpos"]

    # Verify we have the expected number of joints for RBY1
    # RBY1 has: 3 holobase + 6 torso + 7 left_arm + 7 right_arm + 2 left_gripper + 2 right_gripper + 2 head = 29 DOF
    assert len(initial_qpos) == 29, f"Expected 29 joints for RBY1, got {len(initial_qpos)}"
    assert len(final_qpos) == 29, f"Expected 29 joints for RBY1, got {len(final_qpos)}"

    # Verify joints actually moved (relaxed threshold for cross-platform consistency)
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.0001), "Joints should have moved after policy execution"

    # Load and compare against saved test data for regression testing
    test_data_path = TEST_DATA_DIR / "rby1_pnp_policy_final_qpos.npy"
    expected_final_qpos = np.load(test_data_path)

    # Compare with angular wrapping for base yaw (index 2)
    diff = final_qpos - expected_final_qpos
    diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi
    assert np.all(np.abs(diff) < 5e-2), (
        f"Final joint positions don't match expected values.\nExpected: {expected_final_qpos}\nGot: {final_qpos}"
    )


def test_observations_after_steps(task, policy_results):
    """Test that observations remain deterministic after running policy steps."""
    initial_qpos = policy_results["initial_qpos"]
    final_qpos = policy_results["final_qpos"]
    initial_obs_dict = policy_results["initial_obs_dict"]
    final_obs_dict = policy_results["final_obs_dict"]

    # Verify joints moved (relaxed threshold for cross-platform consistency)
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.0001), "Joints should have moved after policy execution"

    # RBY1 has head_camera, wrist cameras, and follower cameras
    expected_cameras = ["head_camera", "wrist_camera_l", "wrist_camera_r"]

    # Ignore wonky follower cameras, plus head and wrist cameras that are too sensitive
    # Still check camera_follower which should be stable
    # Also ignore depth from the inactive arm's wrist camera (it just faces the ceiling)
    arm_side = policy_results["arm_side"]
    inactive_side = "l" if arm_side == "right" else "r"
    ignore_cameras = [
        "camera_thirdview_follower_1",
        "camera_thirdview_follower_2",
        f"wrist_camera_{inactive_side}",
        f"wrist_camera_{inactive_side}_depth",
    ]

    # Verify observations match saved test data AND that they changed from initial observations
    # RBY1 uses resolution (640, 480) which means width=640, height=480
    # numpy shape is (height, width, channels) = (480, 640, 3)
    verify_and_compare_camera_observations_after_steps(
        obs_dict=final_obs_dict,
        sensor_suite=task.sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="rby1_pnp_after_steps_",
        expected_cameras=expected_cameras,
        initial_obs_dict=initial_obs_dict,
        debug_images_dir=DEBUG_IMAGES_DIR / "rby1_pnp",
        debug_prefix="pnp_after_steps",
        expected_shape=(1024, 576, 3),
        atol=1e-3,
        rtol=1e-3,
        ignore_cameras=ignore_cameras,
        ssim_threshold=0.83,
    )


def test_integration(config):
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config.output_dir = TEST_OUTPUT_DIR / "rby1_pnp_output" / timestamp

    runner = ParallelRolloutRunner(config)

    _, total_count = runner.run()
    assert total_count == 1
    assert os.path.isdir(config.output_dir)
    for idx in config.task_sampler_config.house_inds:
        house_dir = config.output_dir / f"house_{idx}"
        assert house_dir.is_dir()
        assert (house_dir / "trajectories_batch_1_of_1.h5").is_file()
        expected_fps = 1000.0 / config.policy_dt_ms
        verify_video_fps(house_dir, expected_fps)
