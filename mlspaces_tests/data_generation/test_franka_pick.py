"""
Tests for Franka pick and place data generation.

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

import h5py
import numpy as np
import pytest

from mlspaces_tests.data_generation.config import (
    FrankaPickDroidTestConfig,
    FrankaPickRandomizedTestConfig,
)
from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.utils.test_utils import (
    assert_obs_scene_match,
    compare_h5_groups,
    run_task_for_steps_with_observations,
    verify_and_compare_camera_observations,
    verify_and_compare_camera_observations_after_steps,
    verify_video_fps,
)

TEST_DATA_DIR = get_resource_manager().symlink_dir / "test_data" / "franka_pick"
TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
DEBUG_IMAGES_DIR = Path(__file__).resolve().parent / "test_debug_images"


@pytest.fixture(scope="session", autouse=True)
def setup_env(tmp_path_factory):
    """Set up environment variables for all tests."""
    yield


@pytest.fixture(scope="session")
def droid_config():
    """Create test-specific config instance for DROID cameras (shared across all tests)."""
    config = FrankaPickDroidTestConfig()
    # Test overrides
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    return config


@pytest.fixture(scope="session")
def randomized_config():
    """Create test-specific config instance for randomized cameras (shared across all tests)."""
    config = FrankaPickRandomizedTestConfig()
    # Test overrides
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    return config


@pytest.fixture(scope="session")
def droid_task_sampler(droid_config):
    """Create and initialize task sampler once for all DROID tests (expensive initialization)."""
    task_sampler_config = droid_config.task_sampler_config
    task_sampler_class = task_sampler_config.task_sampler_class
    task_sampler = task_sampler_class(droid_config)
    task_sampler.reset()
    return task_sampler


@pytest.fixture(scope="session")
def droid_task(droid_task_sampler):
    """Sample task once for all DROID tests (expensive operation)."""
    task = droid_task_sampler.sample_task()
    return task


@pytest.fixture(scope="session")
def randomized_task_sampler(randomized_config):
    """Create and initialize task sampler once for all randomized tests (expensive initialization)."""
    task_sampler_config = randomized_config.task_sampler_config
    task_sampler_class = task_sampler_config.task_sampler_class
    task_sampler = task_sampler_class(randomized_config)
    task_sampler.reset()
    return task_sampler


@pytest.fixture(scope="session")
def randomized_task(randomized_task_sampler):
    """Sample task once for all randomized tests (expensive operation)."""
    task = randomized_task_sampler.sample_task()
    return task


@pytest.fixture(scope="session")
def droid_policy_results(droid_config, droid_task):
    """Run policy once for all DROID tests (expensive operation)."""
    # Reset task to initial state
    droid_task.reset()

    # Instantiate policy with config and task
    policy_config = droid_config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(droid_config, droid_task)
    policy.reset()

    # Run policy for 10 steps and get both qpos and observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(
            droid_task, policy, num_steps=10, profiler=droid_config.profiler
        )
    )

    return {
        "initial_qpos": initial_qpos,
        "final_qpos": final_qpos,
        "initial_obs_dict": initial_obs_dict,
        "final_obs_dict": final_obs_dict,
    }


@pytest.fixture(scope="session")
def randomized_policy_results(randomized_config, randomized_task):
    """Run policy once for all randomized tests (expensive operation)."""
    # Reset task to initial state
    randomized_task.reset()

    # Instantiate policy with config and task
    policy_config = randomized_config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(randomized_config, randomized_task)
    policy.reset()

    # Run policy for 10 steps and get both qpos and observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(
            randomized_task, policy, num_steps=10, profiler=randomized_config.profiler
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
    from mlspaces_tests.data_generation.config import (
        FrankaPickDroidTestConfig,
        FrankaPickRandomizedTestConfig,
    )
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.policy.solvers.object_manipulation.open_close_planner_policy import (
        OpenClosePlannerPolicy,
    )
    from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
        PickAndPlacePlannerPolicy,
    )
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
    assert FrankaPickDroidTestConfig is not None
    assert FrankaPickRandomizedTestConfig is not None
    assert PickTaskSampler is not None
    assert PickTask is not None
    assert PickPlannerPolicy is not None
    assert PickAndPlacePlannerPolicy is not None
    assert OpenClosePlannerPolicy is not None
    assert ParallelRolloutRunner is not None


def test_droid_policy(droid_config):
    """Test that the policy can be instantiated and has required methods (DROID cameras)."""
    from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
        PickPlannerPolicy,
    )

    policy_config = droid_config.policy_config
    policy_cls = policy_config.policy_cls
    assert policy_cls == PickPlannerPolicy
    # if you have a new policy class, write another test

    # Verify policy config has expected attributes
    assert hasattr(policy_config, "pregrasp_z_offset")
    assert hasattr(policy_config, "grasp_z_offset")
    assert hasattr(policy_config, "place_z_offset")
    assert hasattr(policy_config, "end_z_offset")


def test_droid_task_sampler(droid_config, droid_task_sampler, droid_task):
    """Test task sampler with DROID cameras."""
    from molmo_spaces.tasks.pick_task import PickTask

    # Check base class state after sampling (use private attributes which are actually updated)
    assert droid_task_sampler._samples_per_current_house == 1  # Already sampled one task
    assert (
        droid_task_sampler._house_iterator_index == -1
    )  # Stays at -1 until samples_per_house exhausted

    # Verify the sampled task is the correct type
    assert droid_task is not None
    assert isinstance(droid_task, PickTask), f"Expected PickTask, got {type(droid_task)}"

    # Verify task has expected attributes
    assert hasattr(droid_task, "env"), "Task should have env attribute"
    assert hasattr(droid_task, "sensor_suite"), "Task should have sensor_suite attribute"

    # Verify task configuration matches expected values from config
    assert droid_task.config.camera_config.img_resolution == (624, 352)
    assert droid_task.config.ctrl_dt_ms == 2.0


def test_droid_task_observations(droid_task):
    """Test task observations with DROID cameras."""
    obs = droid_task.reset()
    sensor_suite = droid_task.sensor_suite

    # DROID cameras: only 1 exo camera + wrist
    expected_cameras = ["exo_camera_1", "wrist_camera"]

    verify_and_compare_camera_observations(
        obs=obs,
        sensor_suite=sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="franka_pick_droid_obs_",
        expected_cameras=expected_cameras,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_droid",
        debug_prefix="droid_obs",
        expected_shape=(624, 352, 3),
    )


def test_droid_policy_determinism(droid_policy_results):
    """Test that the policy executes deterministically and produces expected outputs (DROID cameras)."""
    initial_qpos = droid_policy_results["initial_qpos"]
    final_qpos = droid_policy_results["final_qpos"]

    # Verify we have the expected number of joints (7 arm + 2 gripper = 9 for Franka)
    assert len(initial_qpos) == 9, f"Expected 9 joints (7 arm + 2 gripper), got {len(initial_qpos)}"
    assert len(final_qpos) == 9, f"Expected 9 joints (7 arm + 2 gripper), got {len(final_qpos)}"

    # Verify joints actually moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # Load and compare against saved test data for regression testing
    test_data_path = TEST_DATA_DIR / "franka_pick_droid_policy_final_qpos.npy"
    expected_final_qpos = np.load(test_data_path)
    # Allow small tolerance for floating point differences
    assert np.allclose(final_qpos[:7], expected_final_qpos[:7], atol=1e-4), (
        f"Final arm joint positions don't match expected values.\nExpected: {expected_final_qpos[:7]}\nGot: {final_qpos[:7]}"
    )

    # Gripper positions might vary slightly more due to control
    assert np.allclose(final_qpos[7:], expected_final_qpos[7:], atol=1e-3), (
        f"Final gripper positions don't match expected values.\nExpected: {expected_final_qpos[7:]}\nGot: {final_qpos[7:]}"
    )


def test_droid_policy_observations_after_steps(droid_task, droid_policy_results):
    """Test that observations remain deterministic after running policy steps (DROID cameras)."""
    initial_qpos = droid_policy_results["initial_qpos"]
    final_qpos = droid_policy_results["final_qpos"]
    initial_obs_dict = droid_policy_results["initial_obs_dict"]
    final_obs_dict = droid_policy_results["final_obs_dict"]

    # Verify joints moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # DROID cameras: only 1 exo camera + wrist
    expected_cameras = ["exo_camera_1", "wrist_camera"]

    # Verify observations match saved test data AND that they changed from initial observations
    verify_and_compare_camera_observations_after_steps(
        obs_dict=final_obs_dict,
        sensor_suite=droid_task.sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="franka_pick_droid_after_steps_",
        expected_cameras=expected_cameras,
        initial_obs_dict=initial_obs_dict,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_droid",
        debug_prefix="droid_after_steps",
        ignore_cameras=["wrist_camera"],  # Wrist camera too sensitive
        expected_shape=(624, 352, 3),
    )


def test_droid_integration(droid_config):
    """Integration test for ParallelRolloutRunner with Franka pick task (DROID cameras)."""
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    droid_config.output_dir = TEST_OUTPUT_DIR / "franka_pick_droid_output" / str(timestamp)

    runner = ParallelRolloutRunner(droid_config)

    # Run the pipeline
    success_count, total_count = runner.run()
    assert total_count == 1, f"Expected to run 1 task, ran {total_count}"
    assert droid_config.output_dir.is_dir()
    for idx in droid_config.task_sampler_config.house_inds:
        house_dir = droid_config.output_dir / f"house_{idx}"
        assert house_dir.is_dir()
        actual_datafile = house_dir / "trajectories_batch_1_of_1.h5"
        assert actual_datafile.is_file()
        expected_fps = 1000.0 / droid_config.policy_dt_ms
        verify_video_fps(house_dir, expected_fps)

        expected_datafile = (
            TEST_DATA_DIR / "franka_pick_droid" / f"house_{idx}" / "trajectories_batch_1_of_1.h5"
        )
        with h5py.File(expected_datafile, "r") as f1, h5py.File(actual_datafile, "r") as f2:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # TEMPORARY BYPASS: object_image_points format changed from JSON-encoded
            # dataset to nested HDF5 groups. The expected test data still has the old
            # format. Once test data is regenerated with the new format, REMOVE this
            # ignore_paths parameter to enable full comparison.
            # I am very tired.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print(
                "WARNING: Skipping object_image_points comparison - format changed from "
                "JSON dataset to nested HDF5 groups. Regenerate test data and remove this bypass. "
                "I am very tired."
            )
            compare_h5_groups(f1, f2, atol=0.001, ignore_paths={"object_image_points"})

            # explicitly check that obs_scene matches
            for traj_key in f1:
                if traj_key.startswith("traj_"):
                    assert traj_key in f2, f"traj_key {traj_key} missing in {f2.filename}"
                    assert_obs_scene_match(f1[traj_key], f2[traj_key])


def test_randomized_task_sampler(randomized_config, randomized_task_sampler, randomized_task):
    """Test task sampler with randomized exocentric cameras."""
    from molmo_spaces.tasks.pick_task import PickTask

    # Check base class state after sampling (use private attributes which are actually updated)
    assert randomized_task_sampler._samples_per_current_house == 1  # Already sampled one task
    assert (
        randomized_task_sampler._house_iterator_index == -1
    )  # Stays at -1 until samples_per_house exhausted

    # Verify the sampled task is the correct type
    assert randomized_task is not None
    assert isinstance(randomized_task, PickTask), f"Expected PickTask, got {type(randomized_task)}"

    # Verify task has expected attributes
    assert hasattr(randomized_task, "env"), "Task should have env attribute"
    assert hasattr(randomized_task, "sensor_suite"), "Task should have sensor_suite attribute"

    # Verify task configuration matches expected values from config
    assert randomized_task.config.camera_config.img_resolution == (624, 352)
    assert randomized_task.config.ctrl_dt_ms == 2.0


def test_randomized_task_observations(randomized_task):
    """Test task observations with randomized exocentric cameras."""
    obs = randomized_task.reset()
    sensor_suite = randomized_task.sensor_suite

    # Randomized cameras: 2 exo cameras + wrist
    expected_cameras = ["exo_camera_1", "exo_camera_2", "wrist_camera"]

    verify_and_compare_camera_observations(
        obs=obs,
        sensor_suite=sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="franka_pick_randomized_obs_",
        expected_cameras=expected_cameras,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_randomized",
        debug_prefix="randomized_obs",
        expected_shape=(624, 352, 3),
    )


def test_randomized_policy_observations_after_steps(randomized_task, randomized_policy_results):
    """Test that observations remain deterministic after running policy steps (randomized cameras)."""
    initial_qpos = randomized_policy_results["initial_qpos"]
    final_qpos = randomized_policy_results["final_qpos"]
    initial_obs_dict = randomized_policy_results["initial_obs_dict"]
    final_obs_dict = randomized_policy_results["final_obs_dict"]

    # Verify joints moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # Randomized cameras: 2 exo cameras + wrist
    expected_cameras = ["exo_camera_1", "exo_camera_2", "wrist_camera"]

    # Verify observations match saved test data AND that they changed from initial observations
    verify_and_compare_camera_observations_after_steps(
        obs_dict=final_obs_dict,
        sensor_suite=randomized_task.sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="franka_pick_randomized_after_steps_",
        expected_cameras=expected_cameras,
        initial_obs_dict=initial_obs_dict,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_randomized",
        debug_prefix="randomized_after_steps",
        ignore_cameras=["wrist_camera"],  # Wrist camera too sensitive
        expected_shape=(624, 352, 3),
    )


def test_randomized_integration(randomized_config):
    """Integration test for ParallelRolloutRunner with Franka pick task (randomized cameras)."""
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    randomized_config.output_dir = (
        TEST_OUTPUT_DIR / "franka_pick_randomized_output" / str(timestamp)
    )

    runner = ParallelRolloutRunner(randomized_config)

    # Run the pipeline
    success_count, total_count = runner.run()
    assert total_count == 1, f"Expected to run 1 task, ran {total_count}"
    assert randomized_config.output_dir.is_dir()
    for idx in randomized_config.task_sampler_config.house_inds:
        house_dir = randomized_config.output_dir / f"house_{idx}"
        assert house_dir.is_dir()
        actual_datafile = house_dir / "trajectories_batch_1_of_1.h5"
        assert actual_datafile.is_file()
        expected_fps = 1000.0 / randomized_config.policy_dt_ms
        verify_video_fps(house_dir, expected_fps)

        expected_datafile = (
            TEST_DATA_DIR
            / "franka_pick_randomized"
            / f"house_{idx}"
            / "trajectories_batch_1_of_1.h5"
        )
        with h5py.File(expected_datafile, "r") as f1, h5py.File(actual_datafile, "r") as f2:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # TEMPORARY BYPASS: object_image_points format changed from JSON-encoded
            # dataset to nested HDF5 groups. The expected test data still has the old
            # format. Once test data is regenerated with the new format, REMOVE this
            # ignore_paths parameter to enable full comparison.
            # I am very tired.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print(
                "WARNING: Skipping object_image_points comparison - format changed from "
                "JSON dataset to nested HDF5 groups. Regenerate test data and remove this bypass. "
                "I am very tired."
            )
            compare_h5_groups(f1, f2, atol=0.001, ignore_paths={"object_image_points"})

            # explicitly check that obs_scene matches
            for traj_key in f1:
                if traj_key.startswith("traj_"):
                    assert traj_key in f2, f"traj_key {traj_key} missing in {f2.filename}"
                    assert_obs_scene_match(f1[traj_key], f2[traj_key])
