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

import numpy as np
import pytest

from mlspaces_tests.data_generation.config import (
    FrankaPickAndPlaceDroidTestConfig,
    FrankaPickAndPlaceGoProD405D455TestConfig,
)
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
    FrankaGoProD405D455CameraSystem,
)
from molmo_spaces.molmo_spaces_constants import get_resource_manager
from molmo_spaces.utils.depth_utils import (
    DEPTH_MAX,
    DEPTH_MIN,
    decode_depth_from_rgb,
    encode_depth_to_rgb,
    print_depth_stats,
    visualize_depth_error,
    visualize_depth_image,
)
from molmo_spaces.utils.test_utils import (
    run_task_for_steps_with_observations,
    verify_and_compare_camera_observations,
    verify_and_compare_camera_observations_after_steps,
    verify_video_fps,
)

TEST_DATA_DIR = get_resource_manager().symlink_dir / "test_data" / "franka_pick_and_place"
TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "test_output"
DEBUG_IMAGES_DIR = Path(__file__).resolve().parent / "test_debug_images"


@pytest.fixture(scope="session", autouse=True)
def setup_env(tmp_path_factory):
    """Set up environment variables for all tests."""
    yield


@pytest.fixture(scope="session")
def droid_config():
    """Create test-specific config instance for DROID cameras (shared across all tests)."""
    config = FrankaPickAndPlaceDroidTestConfig()
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
def gopro_config():
    """Create test-specific config instance for GoPro/D405/D455 cameras (shared across all tests)."""
    config = FrankaPickAndPlaceGoProD405D455TestConfig()
    # Test overrides
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    return config


@pytest.fixture(scope="session")
def gopro_task_sampler(gopro_config):
    """Create and initialize task sampler once for all GoPro tests (expensive initialization)."""
    task_sampler_config = gopro_config.task_sampler_config
    task_sampler_class = task_sampler_config.task_sampler_class
    task_sampler = task_sampler_class(gopro_config)
    task_sampler.reset()
    return task_sampler


@pytest.fixture(scope="session")
def gopro_task(gopro_task_sampler):
    """Sample task once for all GoPro tests (expensive operation)."""
    task = gopro_task_sampler.sample_task()
    return task


@pytest.fixture(scope="session")
def gopro_policy_results(gopro_config, gopro_task):
    """Run policy once for all GoPro tests (expensive operation)."""
    # Reset task to initial state
    gopro_task.reset()

    # Instantiate policy with config and task
    policy_config = gopro_config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(gopro_config, gopro_task)
    policy.reset()

    # Run policy for 10 steps and get both qpos and observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(
            gopro_task, policy, num_steps=10, profiler=gopro_config.profiler
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
        FrankaPickAndPlaceDroidTestConfig,
    )
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
    from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
        PickAndPlacePlannerPolicy,
    )
    from molmo_spaces.tasks.pick_and_place_task import (
        PickAndPlaceTask,
    )
    from molmo_spaces.tasks.pick_and_place_task_sampler import (
        PickAndPlaceTaskSampler,
    )
    # fmt: on
    assert FrankaPickAndPlaceDroidTestConfig is not None
    assert PickAndPlaceTaskSampler is not None
    assert PickAndPlaceTask is not None
    assert PickAndPlacePlannerPolicy is not None
    assert ParallelRolloutRunner is not None


def test_droid_policy(droid_config):
    """Test that the policy can be instantiated and has required methods (DROID cameras)."""
    from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
        PickAndPlacePlannerPolicy,
    )

    policy_config = droid_config.policy_config
    policy_cls = policy_config.policy_cls
    assert policy_cls == PickAndPlacePlannerPolicy
    # if you have a new policy class, write another test

    # Verify policy config has expected attributes
    assert hasattr(policy_config, "pregrasp_z_offset")
    assert hasattr(policy_config, "grasp_z_offset")
    assert hasattr(policy_config, "place_z_offset")
    assert hasattr(policy_config, "end_z_offset")


def test_droid_task_sampler(droid_config, droid_task_sampler, droid_task):
    """Test task sampler with DROID cameras."""
    from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

    assert isinstance(droid_config.camera_config, FrankaDroidCameraSystem)
    # Check base class state after sampling (use private attributes which are actually updated)
    assert droid_task_sampler._samples_per_current_house == 1  # Already sampled one task
    assert (
        droid_task_sampler._house_iterator_index == -1
    )  # Stays at -1 until samples_per_house exhausted

    # Verify the sampled task is the correct type
    assert droid_task is not None
    assert isinstance(droid_task, PickAndPlaceTask), (
        f"Expected PickAndPlaceTask, got {type(droid_task)}"
    )

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
        test_data_prefix="franka_pick_and_place_droid_obs_",
        expected_cameras=expected_cameras,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_and_place_droid",
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
    test_data_path = TEST_DATA_DIR / "franka_pick_and_place_droid_policy_final_qpos.npy"
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
        test_data_prefix="franka_pick_and_place_droid_after_steps_",
        expected_cameras=expected_cameras,
        initial_obs_dict=initial_obs_dict,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_and_place_droid",
        debug_prefix="droid_after_steps",
        ignore_cameras=["wrist_camera"],  # Wrist camera too sensitive
        expected_shape=(624, 352, 3),
    )


def test_droid_integration(droid_config):
    """Integration test for ParallelRolloutRunner with Franka pick and place task (DROID cameras)."""
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    droid_config.output_dir = (
        TEST_OUTPUT_DIR / "franka_pick_and_place_droid_output" / str(timestamp)
    )

    runner = ParallelRolloutRunner(droid_config)

    # Run the pipeline
    success_count, total_count = runner.run()
    assert total_count == 1, f"Expected to run 1 task, ran {total_count}"
    assert droid_config.output_dir.is_dir()
    for idx in droid_config.task_sampler_config.house_inds:
        house_dir = droid_config.output_dir / f"house_{idx}"
        assert house_dir.is_dir()
        assert (house_dir / "trajectories_batch_1_of_1.h5").is_file()
        expected_fps = 1000.0 / droid_config.policy_dt_ms
        verify_video_fps(house_dir, expected_fps)


# ============================================================================
# GoPro/D405/D455 Camera System Tests - Initial observations (before policy)
# ============================================================================
# NOTE: These tests must run BEFORE depth tests that use gopro_policy_results
# to avoid test interference from policy execution modifying task state


def test_gopro_task_sampler(gopro_config, gopro_task_sampler, gopro_task):
    """Test task sampler with GoPro/D405/D455 cameras."""
    from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

    assert isinstance(gopro_config.camera_config, FrankaGoProD405D455CameraSystem)
    # Check base class state after sampling (use private attributes which are actually updated)
    assert gopro_task_sampler._samples_per_current_house == 1  # Already sampled one task
    assert (
        gopro_task_sampler._house_iterator_index == -1
    )  # Stays at -1 until samples_per_house exhausted

    # Verify the sampled task is the correct type
    assert gopro_task is not None
    assert isinstance(gopro_task, PickAndPlaceTask), (
        f"Expected PickAndPlaceTask, got {type(gopro_task)}"
    )

    # Verify task has expected attributes
    assert hasattr(gopro_task, "env"), "Task should have env attribute"
    assert hasattr(gopro_task, "sensor_suite"), "Task should have sensor_suite attribute"

    # Verify task configuration matches expected values from config
    assert gopro_task.config.camera_config.img_resolution == (640, 480)
    assert gopro_task.config.ctrl_dt_ms == 2.0


def test_gopro_task_observations(gopro_task):
    """Test task observations with GoPro/D405/D455 cameras."""
    obs = gopro_task.reset()
    sensor_suite = gopro_task.sensor_suite

    # GoPro/D405/D455 cameras: 2 exo cameras + wrist
    expected_cameras = ["exo_camera_1", "exo_camera_2", "wrist_camera"]

    verify_and_compare_camera_observations(
        obs=obs,
        sensor_suite=sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="franka_pick_and_place_gopro_obs_",
        expected_cameras=expected_cameras,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_and_place_gopro",
        debug_prefix="gopro_obs",
        expected_shape=(640, 480, 3),
    )


# ============================================================================
# Depth Encoding Tests - These use gopro_policy_results (runs policy)
# ============================================================================


def test_depth_encoding_accuracy():
    """Test accuracy of 16-bit depth encoding/decoding with known values."""
    # Create test depth values across the encoding range
    test_depths = np.array(
        [
            0.05,  # Minimum (DEPTH_MIN)
            0.1,  # Close
            0.5,  # Mid-range
            0.55,  # Maximum (DEPTH_MAX)
        ],
        dtype=np.float32,
    )

    # Create a test image with these depth values
    h = 1
    depth_image = np.tile(test_depths, (h, 1))

    # Encode to RGB
    rgb_encoded = encode_depth_to_rgb(depth_image)

    # Decode back to metric depth
    decoded_depths = decode_depth_from_rgb(rgb_encoded)[0, :]

    # Verify accuracy
    # With 16-bit precision over 0.5m range (0.05-0.55m), we should have ~0.0076mm precision
    max_error = np.max(np.abs(decoded_depths - test_depths))
    print("\nDepth encoding test (16-bit RG channels):")
    print(f"  Encoding range: {DEPTH_MIN}m - {DEPTH_MAX}m ({DEPTH_MAX - DEPTH_MIN}m)")
    print(f"  Original depths: {test_depths}")
    print(f"  Decoded depths:  {decoded_depths}")
    print(f"  Maximum error:   {max_error * 1000:.6f} mm")
    print(f"  Theoretical precision: ~0.0076mm (16-bit over {DEPTH_MAX - DEPTH_MIN}m range)")

    # Assert sub-millimeter accuracy (16-bit gives ~0.0076mm, well below 0.01mm)
    assert max_error < 0.00001, (
        f"Depth encoding error {max_error * 1000:.6f}mm exceeds 0.01mm threshold"
    )


def test_depth_mp4_roundtrip_accuracy(gopro_task, gopro_policy_results):
    """Test that depth survives MP4 encoding/decoding within D405 camera accuracy spec.

    Optimized for Intel RealSense D405 camera:
    - D405 actual spec: 7cm - 50cm range, ±1.4% at 20cm = ±2.8mm
    - Encoding range: 5cm - 55cm (extended for margin → 7.6μm precision with 16-bit RG)
    - Codec: libx264rgb with CRF=18 (high-quality lossy, RGB pixel format)

    Test verifies compression errors stay within camera's natural accuracy.
    """
    import tempfile

    from molmo_spaces.utils.depth_utils import load_depth_video, save_depth_video

    # Get depth frames from policy results (initial and final observations after 10 steps)
    initial_obs_dict = gopro_policy_results["initial_obs_dict"]
    final_obs_dict = gopro_policy_results["final_obs_dict"]

    # Get depth observations
    assert "wrist_camera_depth" in initial_obs_dict, (
        "Wrist camera depth not in initial observations"
    )
    assert "wrist_camera_depth" in final_obs_dict, "Wrist camera depth not in final observations"

    initial_depth = initial_obs_dict["wrist_camera_depth"]
    final_depth = final_obs_dict["wrist_camera_depth"]

    # Validate format
    assert initial_depth.ndim == 2, "Depth should be 2D (H, W)"
    assert initial_depth.dtype == np.float32, "Depth should be float32"
    assert final_depth.ndim == 2, "Depth should be 2D (H, W)"
    assert final_depth.dtype == np.float32, "Depth should be float32"

    # Create video with both frames (tests inter-frame compression)
    depth_frames = np.stack([initial_depth, final_depth], axis=0)  # (2, H, W)

    print("\nUsing real depth data from policy execution (2 frames with gripper motion)")
    print(f"Depth range across frames: [{depth_frames.min():.3f}m, {depth_frames.max():.3f}m]")
    print_depth_stats(initial_depth, "Initial Wrist Camera Depth")
    print_depth_stats(final_depth, "Final Wrist Camera Depth (after 10 steps)")

    # Save to MP4 and reload all frames
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/test_depth_wrist_policy.mp4"

        # Use production depth video saving function
        save_depth_video(depth_frames, video_path, fps=10)

        # Read back all frames using production depth video loading function
        # This ensures proper RGB pixel format handling (no YUV conversion)
        decoded_frames = load_depth_video(video_path)

        assert decoded_frames.shape[0] == 2, f"Expected 2 frames, got {decoded_frames.shape[0]}"

    decoded_initial = decoded_frames[0]
    decoded_final = decoded_frames[1]

    # Create masks for valid pixels (within encoding range in original)
    valid_mask_initial = (initial_depth >= DEPTH_MIN) & (initial_depth <= DEPTH_MAX)
    valid_mask_final = (final_depth >= DEPTH_MIN) & (final_depth <= DEPTH_MAX)

    # Clip original depth to encoding range for fair comparison
    # (encoding clips values, so we need to compare against what was actually encoded)
    initial_depth_clipped = np.clip(initial_depth, DEPTH_MIN, DEPTH_MAX)
    final_depth_clipped = np.clip(final_depth, DEPTH_MIN, DEPTH_MAX)

    # Calculate errors for both frames (compare decoded vs clipped original)
    error_initial = np.abs(decoded_initial - initial_depth_clipped)
    error_final = np.abs(decoded_final - final_depth_clipped)

    # Calculate statistics ONLY on valid (non-clipped) pixels
    mean_error_initial = np.mean(error_initial[valid_mask_initial])
    max_error_initial = np.max(error_initial[valid_mask_initial])
    p95_error_initial = np.percentile(error_initial[valid_mask_initial], 95)

    mean_error_final = np.mean(error_final[valid_mask_final])
    max_error_final = np.max(error_final[valid_mask_final])
    p95_error_final = np.percentile(error_final[valid_mask_final], 95)

    mean_error = (mean_error_initial + mean_error_final) / 2
    max_error = max(max_error_initial, max_error_final)
    p95_error = (p95_error_initial + p95_error_final) / 2

    # Count valid vs clipped pixels
    num_valid_initial = np.sum(valid_mask_initial)
    num_clipped_initial = np.sum(~valid_mask_initial)
    num_valid_final = np.sum(valid_mask_final)
    num_clipped_final = np.sum(~valid_mask_final)

    print("\nMP4 lossy (CRF 18) round-trip depth errors:")
    print(
        f"  Encoding range: {DEPTH_MIN * 1000:.0f}mm - {DEPTH_MAX * 1000:.0f}mm (D405 spec: 70-500mm, extended for margin)"
    )
    print(
        f"  Valid pixels: initial={num_valid_initial:,} ({num_valid_initial / error_initial.size * 100:.1f}%), final={num_valid_final:,} ({num_valid_final / error_final.size * 100:.1f}%)"
    )
    print(
        f"  Clipped pixels: initial={num_clipped_initial:,} ({num_clipped_initial / error_initial.size * 100:.1f}%), final={num_clipped_final:,} ({num_clipped_final / error_final.size * 100:.1f}%)"
    )
    print("\n  Compression errors (valid pixels only):")
    print(
        f"    Mean: {mean_error * 1000:.3f} mm (initial: {mean_error_initial * 1000:.3f} mm, final: {mean_error_final * 1000:.3f} mm)"
    )
    print(
        f"    95th percentile: {p95_error * 1000:.3f} mm (initial: {p95_error_initial * 1000:.3f} mm, final: {p95_error_final * 1000:.3f} mm)"
    )
    print(
        f"    Max: {max_error * 1000:.3f} mm (initial: {max_error_initial * 1000:.3f} mm, final: {max_error_final * 1000:.3f} mm)"
    )

    # Save visualizations for both frames (use clipped versions for fair comparison)
    visualize_depth_image(
        initial_depth_clipped,
        "Original Wrist Depth - Initial (Clipped to 5-55cm encoding range)",
        save_path=DEBUG_IMAGES_DIR / "depth_mp4_roundtrip_original_initial.png",
    )
    visualize_depth_image(
        decoded_initial,
        "Decoded Wrist Depth - Initial After MP4 (CRF 18)",
        save_path=DEBUG_IMAGES_DIR / "depth_mp4_roundtrip_decoded_initial.png",
    )

    # Visualize error/delta for initial frame
    visualize_depth_error(
        initial_depth_clipped,
        decoded_initial,
        error_initial,
        "Compression Error - Initial Frame",
        save_path=DEBUG_IMAGES_DIR / "depth_mp4_roundtrip_error_initial.png",
    )

    visualize_depth_image(
        final_depth_clipped,
        "Original Wrist Depth - Final (Clipped to 5-55cm encoding range)",
        save_path=DEBUG_IMAGES_DIR / "depth_mp4_roundtrip_original_final.png",
    )
    visualize_depth_image(
        decoded_final,
        "Decoded Wrist Depth - Final After MP4 (CRF 18)",
        save_path=DEBUG_IMAGES_DIR / "depth_mp4_roundtrip_decoded_final.png",
    )

    # Visualize error/delta for final frame
    visualize_depth_error(
        final_depth_clipped,
        decoded_final,
        error_final,
        "Compression Error - Final Frame",
        save_path=DEBUG_IMAGES_DIR / "depth_mp4_roundtrip_error_final.png",
    )

    # Assert compression maintains accuracy within D405 camera spec
    # D405 spec: ±1.4% at 20cm = ±2.8mm natural camera error
    # Use 95th percentile instead of max (max can have outliers at clipping boundaries)
    assert mean_error < 0.003, (
        f"Mean MP4 compression error {mean_error * 1000:.3f}mm exceeds 3mm threshold. "
        f"This exceeds the D405 camera's natural accuracy (±2.8mm at 20cm). "
        f"Check codec settings or depth range."
    )
    assert p95_error < 0.006, (
        f"95th percentile MP4 compression error {p95_error * 1000:.3f}mm exceeds 5mm threshold. "
        f"Most pixels should have sub-millimeter accuracy. "
        f"Max error was {max_error * 1000:.1f}mm (outliers at clipping boundaries are expected)."
    )

    # Warn if scene has too much clipping (indicates wrong depth range for test data)
    clipped_fraction = (num_clipped_initial + num_clipped_final) / (
        error_initial.size + error_final.size
    )
    if clipped_fraction > 0.3:
        print(
            f"\nWARNING: {clipped_fraction * 100:.1f}% of pixels clipped (outside 5-55cm encoding range)."
        )
        print(
            f"  This test scene has depth range [{depth_frames.min() * 1000:.0f}mm, {depth_frames.max() * 1000:.0f}mm]"
        )
        print(
            "  D405 spec: 70-500mm. Consider using test scenes closer to this range for better validation."
        )


def test_droid_camera_config_no_depth(droid_config):
    """Test that droid config does NOT have depth enabled (RGB only)."""
    assert isinstance(droid_config.camera_config, FrankaDroidCameraSystem)

    # DROID config should NOT have depth cameras
    depth_cameras = [cam for cam in droid_config.camera_config.cameras if cam.record_depth]
    assert len(depth_cameras) == 0, "DROID config should not have depth cameras (RGB only)"

    # Verify we have the expected RGB cameras
    assert len(droid_config.camera_config.cameras) == 2, "Should have 2 cameras (wrist + exo)"


# ============================================================================
# GoPro/D405/D455 Camera System Tests - Policy and post-policy tests
# ============================================================================


def test_gopro_policy_determinism(gopro_policy_results):
    """Test that the policy executes deterministically and produces expected outputs (GoPro cameras)."""
    initial_qpos = gopro_policy_results["initial_qpos"]
    final_qpos = gopro_policy_results["final_qpos"]

    # Verify we have the expected number of joints (7 arm + 2 gripper = 9 for Franka)
    assert len(initial_qpos) == 9, f"Expected 9 joints (7 arm + 2 gripper), got {len(initial_qpos)}"
    assert len(final_qpos) == 9, f"Expected 9 joints (7 arm + 2 gripper), got {len(final_qpos)}"

    # Verify joints actually moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # Load and compare against saved test data for regression testing
    test_data_path = TEST_DATA_DIR / "franka_pick_and_place_gopro_policy_final_qpos.npy"
    expected_final_qpos = np.load(test_data_path)
    # Allow small tolerance for floating point differences
    assert np.allclose(final_qpos[:7], expected_final_qpos[:7], atol=1e-4), (
        f"Final arm joint positions don't match expected values.\nExpected: {expected_final_qpos[:7]}\nGot: {final_qpos[:7]}"
    )

    # Gripper positions might vary slightly more due to control
    assert np.allclose(final_qpos[7:], expected_final_qpos[7:], atol=1e-3), (
        f"Final gripper positions don't match expected values.\nExpected: {expected_final_qpos[7:]}\nGot: {final_qpos[7:]}"
    )


def test_gopro_policy_observations_after_steps(gopro_task, gopro_policy_results):
    """Test that observations remain deterministic after running policy steps (GoPro cameras)."""
    initial_qpos = gopro_policy_results["initial_qpos"]
    final_qpos = gopro_policy_results["final_qpos"]
    initial_obs_dict = gopro_policy_results["initial_obs_dict"]
    final_obs_dict = gopro_policy_results["final_obs_dict"]

    # Verify joints moved
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 0.001), "Joints should have moved after policy execution"

    # GoPro/D405/D455 cameras: 2 exo cameras + wrist
    expected_cameras = ["exo_camera_1", "exo_camera_2", "wrist_camera"]

    # Verify observations match saved test data AND that they changed from initial observations
    verify_and_compare_camera_observations_after_steps(
        obs_dict=final_obs_dict,
        sensor_suite=gopro_task.sensor_suite,
        test_data_dir=TEST_DATA_DIR,
        test_data_prefix="franka_pick_and_place_gopro_after_steps_",
        expected_cameras=expected_cameras,
        initial_obs_dict=initial_obs_dict,
        debug_images_dir=DEBUG_IMAGES_DIR / "franka_pick_and_place_gopro",
        debug_prefix="gopro_after_steps",
        ignore_cameras=["wrist_camera"],  # Wrist camera too sensitive
        expected_shape=(640, 480, 3),
    )


def test_gopro_integration(gopro_config):
    """Integration test for ParallelRolloutRunner with Franka pick and place task (GoPro cameras)."""
    from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gopro_config.output_dir = (
        TEST_OUTPUT_DIR / "franka_pick_and_place_gopro_output" / str(timestamp)
    )

    runner = ParallelRolloutRunner(gopro_config)

    # Run the pipeline
    success_count, total_count = runner.run()
    assert total_count == 1, f"Expected to run 1 task, ran {total_count}"
    assert gopro_config.output_dir.is_dir()
    for idx in gopro_config.task_sampler_config.house_inds:
        assert (gopro_config.output_dir / f"house_{idx}").is_dir()
        assert (gopro_config.output_dir / f"house_{idx}" / "trajectories_batch_1_of_1.h5").is_file()


def test_gopro_depth_camera_config(gopro_config):
    """Test that gopro config has depth enabled on wrist camera (D405)."""
    assert isinstance(gopro_config.camera_config, FrankaGoProD405D455CameraSystem)

    # Verify we have 3 cameras total
    assert len(gopro_config.camera_config.cameras) == 3, "Should have 3 cameras (wrist + 2 exo)"

    # Verify exactly one camera has depth enabled (wrist camera)
    depth_cameras = [cam for cam in gopro_config.camera_config.cameras if cam.record_depth]
    assert len(depth_cameras) == 1, "Should have exactly one depth camera"

    # Verify it's the wrist camera (D405)
    wrist_depth = depth_cameras[0]
    assert "wrist" in wrist_depth.name.lower(), "Depth camera should be wrist camera (D405)"


def test_gopro_depth_observations(gopro_task):
    """Test task observations with depth camera (wrist camera only for GoPro config)."""
    obs = gopro_task.reset()
    sensor_suite = gopro_task.sensor_suite

    # Extract observation dict
    assert obs is not None
    assert isinstance(obs, tuple) and len(obs) == 2
    obs_list, _ = obs
    assert len(obs_list) == 1
    obs_dict = obs_list[0]

    # Find depth sensors - should be exactly one (wrist_camera_depth)
    depth_sensors = [s for s in sensor_suite.sensors if s.endswith("_depth")]
    assert len(depth_sensors) == 1, "Should have exactly one depth sensor (wrist_camera_depth)"
    assert depth_sensors[0] == "wrist_camera_depth", "Depth sensor should be wrist_camera_depth"

    # Check depth observation exists and has correct format
    depth_sensor = depth_sensors[0]
    assert depth_sensor in obs_dict, f"Depth sensor {depth_sensor} not in observations"
    depth_obs = obs_dict[depth_sensor]

    # Depth should be raw float32 (H, W)
    assert depth_obs.ndim == 2, f"{depth_sensor} should be 2D (H, W)"
    assert depth_obs.dtype == np.float32, f"{depth_sensor} should be float32"

    # Print detailed stats and visualize
    print_depth_stats(depth_obs, f"{depth_sensor} (wrist camera)")
    visualize_depth_image(
        depth_obs,
        f"{depth_sensor} - Initial Observation",
        save_path=DEBUG_IMAGES_DIR / f"gopro_{depth_sensor}_initial.png",
    )

    # Verify depth range
    depth_min = depth_obs.min()
    depth_max = depth_obs.max()

    # Depth should be within reasonable range
    assert depth_min >= 0, f"Min depth {depth_min} should be non-negative"
    assert depth_max < 10.0, f"Max depth {depth_max} seems unreasonably large"


def test_gopro_depth_determinism(gopro_task, gopro_policy_results):
    """Test that depth observations are self-consistent and valid (GoPro config).

    Note: This test checks depth validity and self-consistency, NOT exact matching
    to saved test data, because depth rendering varies across platforms/GPUs/drivers.
    """
    initial_obs_dict = gopro_policy_results["initial_obs_dict"]

    # Find depth sensors - should be exactly one (wrist_camera_depth)
    depth_sensors = [key for key in initial_obs_dict if key.endswith("_depth")]
    assert len(depth_sensors) == 1, "Should have exactly one depth sensor (wrist_camera_depth)"
    assert depth_sensors[0] == "wrist_camera_depth", "Depth sensor should be wrist_camera_depth"

    for depth_sensor in depth_sensors:
        depth_obs = initial_obs_dict[depth_sensor]

        # Verify depth format
        assert depth_obs.ndim == 2, f"{depth_sensor} should be 2D (H, W)"
        assert depth_obs.dtype == np.float32, f"{depth_sensor} should be float32"

        # Verify depth is valid (not NaN, not inf, not all zeros)
        assert not np.any(np.isnan(depth_obs)), f"{depth_sensor} contains NaN values"
        assert not np.any(np.isinf(depth_obs)), f"{depth_sensor} contains infinite values"
        assert np.any(depth_obs > 0), f"{depth_sensor} is all zeros (no valid depth)"

        # Verify depth is in reasonable range for wrist camera
        # Wrist camera should see objects within ~5cm to 2m
        depth_min = np.min(depth_obs)
        depth_max = np.max(depth_obs)
        depth_median = np.median(depth_obs)

        assert depth_min >= 0.0, f"{depth_sensor} has negative depth: {depth_min}m"
        assert depth_max < 5.0, (
            f"{depth_sensor} max depth {depth_max}m seems unreasonably large (>5m)"
        )
        assert depth_median < 2.0, f"{depth_sensor} median depth {depth_median}m seems too large"

        # Verify depth has variation (not a constant image)
        depth_std = np.std(depth_obs)
        assert depth_std > 0.01, (
            f"{depth_sensor} has no variation (std={depth_std}m), likely frozen"
        )

        print(f"\n{depth_sensor} sanity checks passed:")
        print(f"  Range: [{depth_min:.3f}m, {depth_max:.3f}m]")
        print(f"  Median: {depth_median:.3f}m, Std: {depth_std:.3f}m")
        print(f"  Shape: {depth_obs.shape}, dtype: {depth_obs.dtype}")


def test_gopro_depth_observations_after_steps(gopro_task, gopro_policy_results):
    """Test that depth observations change appropriately after running policy steps (GoPro config).

    Note: This test checks depth changes are valid and consistent with motion, NOT exact
    matching to saved test data, because depth rendering varies across platforms/GPUs/drivers.
    """
    initial_obs_dict = gopro_policy_results["initial_obs_dict"]
    final_obs_dict = gopro_policy_results["final_obs_dict"]

    # Find depth sensors - should be exactly one (wrist_camera_depth)
    depth_sensors = [key for key in initial_obs_dict if key.endswith("_depth")]
    assert len(depth_sensors) == 1, "Should have exactly one depth sensor (wrist_camera_depth)"
    assert depth_sensors[0] == "wrist_camera_depth", "Depth sensor should be wrist_camera_depth"

    # Verify depth observation changed
    for depth_sensor in depth_sensors:
        initial_obs = initial_obs_dict[depth_sensor]
        final_obs = final_obs_dict[depth_sensor]

        # Print stats for both
        print_depth_stats(initial_obs, f"{depth_sensor} (initial)")
        print_depth_stats(final_obs, f"{depth_sensor} (after steps)")

        # Visualize both observations
        visualize_depth_image(
            initial_obs,
            f"{depth_sensor} - After Steps (Initial)",
            save_path=DEBUG_IMAGES_DIR / f"gopro_{depth_sensor}_after_steps_initial.png",
        )
        visualize_depth_image(
            final_obs,
            f"{depth_sensor} - After Steps (Final)",
            save_path=DEBUG_IMAGES_DIR / f"gopro_{depth_sensor}_after_steps_final.png",
        )

        # Verify both observations are valid
        assert not np.any(np.isnan(initial_obs)), f"{depth_sensor} initial has NaN values"
        assert not np.any(np.isnan(final_obs)), f"{depth_sensor} final has NaN values"
        assert not np.any(np.isinf(initial_obs)), f"{depth_sensor} initial has infinite values"
        assert not np.any(np.isinf(final_obs)), f"{depth_sensor} final has infinite values"

        # Calculate depth difference (both are float32)
        depth_diff = np.abs(final_obs - initial_obs)
        num_different_pixels = np.sum(depth_diff > 0.001)  # Changed more than 1mm
        total_pixels = depth_diff.size
        percent_changed = 100 * num_different_pixels / total_pixels

        mean_depth_change = np.mean(depth_diff)
        max_depth_change = np.max(depth_diff)

        print(f"\n{depth_sensor} after policy execution:")
        print(f"  Pixels changed >1mm: {percent_changed:.2f}%")
        print(f"  Mean depth change: {mean_depth_change * 1000:.3f} mm")
        print(f"  Max depth change: {max_depth_change * 1000:.3f} mm")

        # Depth observations should have changed (robot moved)
        # At least some pixels should show depth change > 1mm
        assert num_different_pixels > 0, (
            f"{depth_sensor}: No significant depth changes detected after policy execution"
        )

        # Verify depth changes are reasonable (not entire scene changed dramatically)
        # Most pixels should change by less than 50cm (if entire scene changed by >50cm, something's wrong)
        large_changes = np.sum(depth_diff > 0.5)
        large_change_fraction = large_changes / total_pixels
        assert large_change_fraction < 0.8, (
            f"{depth_sensor}: Too many pixels changed drastically (>{large_change_fraction * 100:.1f}% changed >50cm), "
            "suggesting depth rendering issue rather than robot motion"
        )

        print(
            f"  Depth change validation passed: {num_different_pixels:,} pixels changed, "
            f"{large_changes:,} changed >50cm"
        )
