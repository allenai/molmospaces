"""
Tests for RUM (floating gripper) pick data generation.

TODO: Future test improvements
- Test observation determinism across multiple resets
- Test action space validation (valid ranges)
- Test reward calculation correctness
- Test terminal/truncated conditions
- Test randomization when enabled (currently disabled for determinism)
- Test multi-environment batching (currently only single env)
- Verify trajectory data quality in integration tests (not just file existence)
"""

import datetime
from pathlib import Path

import numpy as np
import pytest

from mlspaces_tests.data_generation.config import RUMPickTestConfig
from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.utils.test_utils import (
    run_task_for_steps_with_observations,
    verify_and_compare_camera_observations,
    verify_and_compare_camera_observations_after_steps,
    verify_video_fps,
)

TEST_DATA_DIR = get_resource_manager().symlink_dir / "test_data" / "rum_pick"
TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
DEBUG_IMAGES_DIR = Path(__file__).resolve().parent / "test_debug_images"


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Set up environment variables for all tests."""
    yield


@pytest.fixture(scope="session")
def rum_config():
    """Create test-specific config instance for RUM (shared across all tests)."""
    config = RUMPickTestConfig()
    # Test overrides
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    return config


@pytest.fixture(scope="session")
def rum_task_sampler(rum_config):
    """Create and initialize task sampler once for all RUM tests (expensive initialization)."""
    task_sampler_config = rum_config.task_sampler_config
    task_sampler_class = task_sampler_config.task_sampler_class
    task_sampler = task_sampler_class(rum_config)
    task_sampler.reset()
    return task_sampler


@pytest.fixture(scope="session")
def rum_task(rum_task_sampler):
    """Sample task once for all RUM tests (expensive operation)."""
    task = rum_task_sampler.sample_task()
    return task


@pytest.fixture(scope="session")
def rum_policy_results(rum_config, rum_task):
    """Run policy once for all RUM tests (expensive operation)."""
    # Reset task to initial state
    rum_task.reset()

    # Instantiate policy with config and task
    policy_config = rum_config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(rum_config, rum_task)
    policy.reset()

    # Run policy for 10 steps and get both qpos and observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(
            rum_task, policy, num_steps=10, profiler=rum_config.profiler
        )
    )

    return {
        "initial_qpos": initial_qpos,
        "final_qpos": final_qpos,
        "initial_obs_dict": initial_obs_dict,
        "final_obs_dict": final_obs_dict,
    }


def test_imports():
    # fmt: off
    from mlspaces_tests.data_generation.config import RUMPickTestConfig
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
        PickPlannerPolicy,
    )
    from molmo_spaces.tasks.pick_task import (
        PickTask,
    )
    from molmo_spaces.tasks.pick_task_sampler import (
        PickTaskSampler,
    )
    # fmt: on
    assert RUMPickTestConfig is not None
    assert PickTaskSampler is not None
    assert PickTask is not None
    assert PickPlannerPolicy is not None
    assert ParallelRolloutRunner is not None


def test_rum_policy(rum_config):
    """Test that the policy can be instantiated and has required methods."""
    from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
        PickPlannerPolicy,
    )

    policy_config = rum_config.policy_config
    policy_cls = policy_config.policy_cls
    assert policy_cls == PickPlannerPolicy

    # Verify policy config has expected attributes
    assert hasattr(policy_config, "pregrasp_z_offset")
    assert hasattr(policy_config, "grasp_z_offset")
    assert hasattr(policy_config, "place_z_offset")
    assert hasattr(policy_config, "end_z_offset")


def test_rum_task_sampler(rum_config, rum_task_sampler, rum_task):
    """Test task sampler with RUM configuration."""
    from molmo_spaces.tasks.pick_task import PickTask

    # Check base class state after reset (use private attributes which are actually updated)
    assert rum_task_sampler._samples_per_current_house == 1  # Already sampled one task
    assert (
        rum_task_sampler._house_iterator_index == -1
    )  # Stays at -1 until samples_per_house exhausted

    # Verify the sampled task is the correct type
    assert rum_task is not None
    assert isinstance(rum_task, PickTask), f"Expected PickTask, got {type(rum_task)}"

    # Verify task has expected attributes
    assert hasattr(rum_task, "env"), "Task should have env attribute"
    assert hasattr(rum_task, "sensor_suite"), "Task should have sensor_suite attribute"

    # Verify task configuration matches expected values from config
    assert rum_task.config.camera_config.img_resolution == (624, 352)
    assert rum_task.config.ctrl_dt_ms == 2.0  # Uses default control timestep from base config


def test_rum_task_observations(rum_task):
    """Test task observations with RUM configuration."""
    obs = rum_task.reset()
    sensor_suite = rum_task.sensor_suite

    # RUM uses randomized cameras: 2 exo cameras only
    # (wrist_cam from FrankaRandomizedCameraSystem doesn't exist in RUM MJCF)
    expected_cameras = ["exo_camera_1", "exo_camera_2"]

    verify_and_compare_camera_observations(
        obs=obs,
        sensor_suite=sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="rum_pick_obs_",
        expected_cameras=expected_cameras,
        debug_images_dir=DEBUG_IMAGES_DIR / "rum_pick",
        debug_prefix="rum_obs",
        expected_shape=(624, 352, 3),
    )


def test_rum_policy_determinism(rum_policy_results):
    """Test that the policy executes deterministically and produces expected outputs."""
    initial_qpos = rum_policy_results["initial_qpos"]
    final_qpos = rum_policy_results["final_qpos"]

    # Verify we have the expected number of joints for RUM (7 base DOF + 2 gripper = 9)
    assert len(initial_qpos) == 9, (
        f"Expected 9 joints (7 base + 2 gripper), got {len(initial_qpos)}"
    )
    assert len(final_qpos) == 9, f"Expected 9 joints (7 base + 2 gripper), got {len(final_qpos)}"

    # Verify joints actually moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # Load and compare against saved test data for regression testing
    test_data_path = TEST_DATA_DIR / "rum_pick_policy_final_qpos.npy"
    expected_final_qpos = np.load(test_data_path)

    # Allow small tolerance for floating point differences
    # Base DOF (7 DOF for floating base)
    assert np.allclose(final_qpos[:7], expected_final_qpos[:7], atol=1e-1), (
        f"Final base positions don't match expected values.\nExpected: {expected_final_qpos[:7]}\nGot: {final_qpos[:7]}"
    )

    # Gripper positions might vary slightly more due to control
    assert np.allclose(final_qpos[7:], expected_final_qpos[7:], atol=1e-3), (
        f"Final gripper positions don't match expected values.\nExpected: {expected_final_qpos[7:]}\nGot: {final_qpos[7:]}"
    )


def test_rum_policy_observations_after_steps(rum_task, rum_policy_results):
    """Test that observations remain deterministic after running policy steps."""
    initial_qpos = rum_policy_results["initial_qpos"]
    final_qpos = rum_policy_results["final_qpos"]
    initial_obs_dict = rum_policy_results["initial_obs_dict"]
    final_obs_dict = rum_policy_results["final_obs_dict"]

    # Verify joints moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # RUM uses randomized cameras: 2 exo cameras only
    expected_cameras = ["exo_camera_1", "exo_camera_2"]

    # Verify observations match saved test data AND that they changed from initial observations
    verify_and_compare_camera_observations_after_steps(
        obs_dict=final_obs_dict,
        sensor_suite=rum_task.sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="rum_pick_after_steps_",
        expected_cameras=expected_cameras,
        initial_obs_dict=initial_obs_dict,
        debug_images_dir=DEBUG_IMAGES_DIR / "rum_pick",
        debug_prefix="rum_after_steps",
        ignore_cameras=["wrist_camera"],  # Wrist camera too sensitive
        expected_shape=(624, 352, 3),
    )


def test_rum_integration(rum_config):
    """Integration test for ParallelRolloutRunner with RUM pick task."""
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rum_config.output_dir = TEST_OUTPUT_DIR / "rum_pick_output" / str(timestamp)

    runner = ParallelRolloutRunner(rum_config)

    # Run the pipeline
    success_count, total_count = runner.run()
    assert total_count == 1, f"Expected to run 1 task, ran {total_count}"
    assert rum_config.output_dir.is_dir()
    for idx in rum_config.task_sampler_config.house_inds:
        house_dir = rum_config.output_dir / f"house_{idx}"
        assert house_dir.is_dir()
        assert (house_dir / "trajectories_batch_1_of_1.h5").is_file()
        expected_fps = 1000.0 / rum_config.policy_dt_ms
        verify_video_fps(house_dir, expected_fps)
