"""
Integration test for JSON benchmark evaluation pipeline.

This test runs the full run_evaluation() pipeline and verifies the results.
The integration test output is used as a fixture for downstream validation tests.

The test verifies:
1. run_evaluation() completes successfully with the test benchmark
2. EvaluationResults contains expected structure and data
3. Episode results are collected and contain required fields
4. Simulation state matches benchmark specification (robot pose, objects)

USAGE:
    python -m pytest mlspaces_tests/data_generation/test_json_benchmark_integration.py -v -s
"""

import json
import logging
from pathlib import Path

import h5py

# Enable logging to see progress during evaluation
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

print("[TEST MODULE] Starting test_json_benchmark_integration.py imports...", flush=True)

import numpy as np
import pytest

from molmo_spaces.configs.policy_configs import DummyPolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec, load_all_episodes
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.evaluation.eval_main import EvaluationResults, run_evaluation
from molmo_spaces.policy.dummy_policy import DummyPolicy

print("[TEST MODULE] All imports complete.", flush=True)

# Path to the test benchmark in the test directory
TEST_BENCHMARK_DIR = Path(__file__).parent / "test_benchmark"
EVAL_TASK_HORIZON_STEPS = 5
EVAL_POLICY_DT_MS = 200.0


def _check_assets_available() -> bool:
    """Check if required scene assets are available for the test benchmark."""
    try:
        from molmo_spaces.molmo_spaces_constants import get_scenes_root

        holodeck_val_dir = get_scenes_root() / "holodeck-objaverse-val"
        return holodeck_val_dir.exists()
    except Exception:
        return False


# Skip all tests in this module if assets aren't available
pytestmark = pytest.mark.skipif(
    not _check_assets_available(),
    reason="holodeck-objaverse-val scene assets not installed",
)


class _TestBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """
    Test config that inherits from JsonBenchmarkEvalConfig.

    This tests the recommended pattern from evaluation/README.md:
    external repos should inherit from JsonBenchmarkEvalConfig and provide
    their robot_config and policy_config. The benchmark JSON provides all
    episode-specific data (cameras, poses, task params).

    Note: Prefixed with underscore to avoid pytest collection warning since
    this inherits from a class with __init__.
    """

    # Timing - short horizon for testing
    task_horizon: int = 10
    seed: int = 42
    policy_dt_ms: float = EVAL_POLICY_DT_MS

    # Robot config - standard Franka
    robot_config: FrankaRobotConfig = FrankaRobotConfig()

    # Policy config - DummyPolicy returns empty dict (no-op)
    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    @property
    def tag(self) -> str:
        return "test_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Disable action noise for deterministic testing
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


@pytest.fixture(scope="module")
def benchmark_episodes() -> list[EpisodeSpec]:
    """Load all episodes from the test benchmark."""
    assert TEST_BENCHMARK_DIR.exists(), f"Test benchmark not found at {TEST_BENCHMARK_DIR}"
    episodes = load_all_episodes(TEST_BENCHMARK_DIR)
    assert len(episodes) > 0, "No episodes found in test benchmark"
    return episodes


@pytest.fixture(scope="module")
def single_episode_benchmark(tmp_path_factory) -> Path:
    """
    Create a single-episode benchmark for fast testing.

    Loads the first episode from the full test benchmark and writes it
    to a temporary benchmark.json file.
    """
    print("[FIXTURE] Creating single-episode benchmark...", flush=True)

    # Load full benchmark and take just the first episode
    all_episodes = load_all_episodes(TEST_BENCHMARK_DIR)
    first_episode = all_episodes[0]

    # Create temp benchmark directory with just 1 episode
    benchmark_dir = tmp_path_factory.mktemp("single_episode_benchmark")
    benchmark_file = benchmark_dir / "benchmark.json"

    with open(benchmark_file, "w") as f:
        json.dump([first_episode.model_dump()], f, indent=2)

    print(f"[FIXTURE] Single-episode benchmark created at {benchmark_dir}", flush=True)
    return benchmark_dir


@pytest.fixture(scope="module")
def evaluation_results(tmp_path_factory, single_episode_benchmark) -> EvaluationResults:
    """
    Run the full evaluation pipeline and return results.

    This is a module-scoped fixture that runs once and provides results
    for all downstream tests. The evaluation uses DummyPolicy which returns
    empty actions, causing the robot to stay stationary.

    Uses single_episode_benchmark for faster testing (1 episode instead of 10).
    """
    import sys

    print("\n[FIXTURE] Starting evaluation_results fixture...", flush=True)
    sys.stdout.flush()

    # Create a preloaded dummy policy to pass to run_evaluation.
    # This bypasses the policy instantiation which expects (config, task_type),
    # allowing us to use DummyPolicy which expects (config, task).
    print("[FIXTURE] Creating DummyPolicy...", flush=True)
    dummy_policy = DummyPolicy(config=_TestBenchmarkEvalConfig())

    output_dir = tmp_path_factory.mktemp("eval_output")
    print(f"[FIXTURE] Output dir: {output_dir}", flush=True)
    print(f"[FIXTURE] Benchmark dir: {single_episode_benchmark}", flush=True)
    print("[FIXTURE] Calling run_evaluation() with 1 episode...", flush=True)
    sys.stdout.flush()

    results = run_evaluation(
        eval_config_cls=_TestBenchmarkEvalConfig,
        benchmark_dir=single_episode_benchmark,
        checkpoint_path=None,  # DummyPolicy doesn't need a checkpoint
        task_horizon_sec=EVAL_TASK_HORIZON_STEPS * EVAL_POLICY_DT_MS / 1000.0,
        output_dir=output_dir,
        num_workers=1,
        use_wandb=False,
        preloaded_policy=dummy_policy,
    )

    print(
        f"[FIXTURE] run_evaluation() complete. {results.total_count} episodes evaluated.",
        flush=True,
    )
    return results


@pytest.fixture(scope="module")
def raw_benchmark_data() -> list[dict]:
    """Load raw JSON data from benchmark.json for comparison with parsed/simulation values."""
    benchmark_file = TEST_BENCHMARK_DIR / "benchmark.json"
    with open(benchmark_file) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def first_episode_raw(raw_benchmark_data) -> dict:
    """First episode's raw JSON dict for end-to-end verification."""
    return raw_benchmark_data[0]


@pytest.fixture(scope="module")
def trajectory_data(evaluation_results) -> dict:
    """
    Extract data from the first trajectory in HDF5 output.

    This provides access to the actual simulation state for verifying
    that the benchmark was correctly applied.
    """
    h5_files = list(evaluation_results.output_dir.rglob("*.h5"))
    assert h5_files, "No HDF5 output files found"

    with h5py.File(h5_files[0], "r") as f:
        traj = f["traj_0"]

        # qpos is stored as 2D array of uint8: shape (T, max_len)
        # Each row is a JSON string encoded as bytes, padded with zeros
        qpos_array = traj["obs"]["agent"]["qpos"][()]
        # Take first timestep (initial state), convert to bytes, strip null padding
        initial_qpos_bytes = bytes(qpos_array[0]).rstrip(b"\x00")
        qpos_data = json.loads(initial_qpos_bytes.decode("utf-8"))

        # obs_scene is byte-encoded JSON with task info
        obs_scene_bytes = traj["obs_scene"][()]
        if isinstance(obs_scene_bytes, bytes):
            obs_scene_bytes = obs_scene_bytes.decode("utf-8")
        obs_scene = json.loads(obs_scene_bytes)

        return {
            "initial_qpos": qpos_data,  # dict with move group keys like {"arm": [...], "gripper": [...]}
            "obs_scene": obs_scene,
            "episode_length": len(qpos_array),
        }


class TestJsonBenchmarkIntegration:
    """Integration tests for JSON benchmark evaluation pipeline."""

    def test_evaluation_completes_successfully(self, evaluation_results: EvaluationResults):
        """
        Verify run_evaluation() completes and returns valid EvaluationResults.

        This is the core integration test - if this passes, the full pipeline works.
        """
        assert isinstance(evaluation_results, EvaluationResults)
        assert evaluation_results.total_count > 0, "No episodes were evaluated"
        assert evaluation_results.output_dir.exists(), "Output directory not created"

        print(f"Evaluated {evaluation_results.total_count} episodes")
        print(f"Success rate: {evaluation_results.success_rate:.1%}")
        print(f"Output dir: {evaluation_results.output_dir}")

    def test_episode_results_collected(self, evaluation_results: EvaluationResults):
        """
        Verify episode results are collected with expected fields.

        Each episode should have result data including success status,
        step counts, and house information.
        """
        assert len(evaluation_results.episode_results) == evaluation_results.total_count, (
            f"Episode results count mismatch: "
            f"{len(evaluation_results.episode_results)} != {evaluation_results.total_count}"
        )

        # Verify each episode result has required fields
        for result in evaluation_results.episode_results:
            assert result.house_id is not None, "Episode missing house_id"
            assert isinstance(result.success, bool), "Episode success must be boolean"
            assert result.num_steps >= 0, "Episode must have non-negative step count"

        print(f"Collected {len(evaluation_results.episode_results)} episode results")

    def test_output_files_created(self, evaluation_results: EvaluationResults):
        """
        Verify HDF5 trajectory files are created for evaluation output.
        """
        output_files = list(evaluation_results.output_dir.rglob("*.h5"))
        assert len(output_files) > 0, "No HDF5 trajectory files created"

        print(f"Created {len(output_files)} HDF5 trajectory files")

    def test_success_count_consistent(self, evaluation_results: EvaluationResults):
        """
        Verify success_count matches the sum of successful episodes.
        """
        episode_success_count = sum(1 for r in evaluation_results.episode_results if r.success)
        assert evaluation_results.success_count == episode_success_count, (
            f"Success count mismatch: {evaluation_results.success_count} != {episode_success_count}"
        )

    def test_dummy_policy_causes_task_failure(self, evaluation_results: EvaluationResults):
        """
        With DummyPolicy (no-op), pick tasks should fail since robot doesn't move.

        This validates that the task success detection is working - a stationary
        robot cannot complete a pick task.
        """
        # With a very short horizon (5 steps) and no-op actions, success should be 0
        # With only 1 episode, we expect 0 successes
        assert evaluation_results.success_count == 0, (
            f"Unexpected success ({evaluation_results.success_count}) with DummyPolicy - "
            f"robot shouldn't be able to complete pick tasks without moving"
        )


class TestSimulationMatchesBenchmark:
    """
    End-to-end tests verifying the simulation state matches the benchmark specification.

    These tests verify that values from benchmark.json are correctly applied to the
    actual MuJoCo simulation, not just parsed into Python objects.
    """

    def test_robot_initialized_at_benchmark_pose(self, trajectory_data, first_episode_raw):
        """
        Verify the robot in MuJoCo simulation starts at the pose specified in benchmark.

        This is a critical end-to-end test: it confirms that init_qpos values from the
        benchmark JSON are correctly parsed, passed through JsonEvalTaskSampler, and
        applied to the robot joints in MuJoCo.
        """
        expected_init_qpos = first_episode_raw["robot"]["init_qpos"]
        actual_qpos = trajectory_data["initial_qpos"]

        # Compare each move group
        for group_name, expected_values in expected_init_qpos.items():
            if not expected_values:  # Skip empty groups like "base": []
                continue
            assert group_name in actual_qpos, (
                f"Move group '{group_name}' from benchmark not found in simulation output"
            )
            np.testing.assert_allclose(
                actual_qpos[group_name],
                expected_values,
                rtol=1e-5,
                err_msg=f"Robot {group_name} NOT initialized at benchmark pose",
            )

    def test_correct_object_used_in_task(self, trajectory_data, first_episode_raw):
        """
        Verify the task was set up with the correct pickup object from benchmark.

        This confirms that pickup_obj_name from the benchmark JSON is correctly passed
        through to the task configuration and recorded in obs_scene.
        """
        expected_object = first_episode_raw["task"]["pickup_obj_name"]
        actual_object = trajectory_data["obs_scene"].get("object_name")

        assert actual_object == expected_object, (
            f"Task used wrong object: expected '{expected_object}', got '{actual_object}'"
        )

    def test_episode_length_consistent(self, trajectory_data):
        # pipeline runs for 1 extra step
        expected = EVAL_TASK_HORIZON_STEPS + 1
        assert trajectory_data["episode_length"] == expected, (
            f"Episode length mismatch: {trajectory_data['episode_length']} != {expected}"
        )


class TestBenchmarkLoading:
    """Tests for benchmark loading and parsing."""

    def test_benchmark_loads_all_episodes(self, benchmark_episodes):
        """Verify the test benchmark loads expected number of episodes."""
        print(f"Loaded {len(benchmark_episodes)} episodes from {TEST_BENCHMARK_DIR}")
        assert len(benchmark_episodes) == 10, (
            f"Expected 10 episodes in test benchmark, got {len(benchmark_episodes)}"
        )

    def test_episode_has_required_fields(self, benchmark_episodes):
        """Verify each episode has required fields."""
        first_episode = benchmark_episodes[0]

        # Verify core fields
        assert first_episode.house_index is not None
        assert first_episode.scene_dataset is not None
        assert first_episode.data_split is not None
        assert first_episode.robot is not None
        assert first_episode.task is not None

    def test_first_episode_values(self, benchmark_episodes):
        """Verify the first episode has expected values from the test benchmark."""
        first_episode = benchmark_episodes[0]

        assert first_episode.house_index == 1194
        assert first_episode.scene_dataset == "holodeck-objaverse"
        assert first_episode.data_split == "val"
        assert first_episode.robot.robot_name == "franka_droid"

    def test_task_class_parseable(self, benchmark_episodes):
        """Verify task class can be retrieved from episode spec."""
        first_episode = benchmark_episodes[0]
        task_cls = first_episode.get_task_cls()
        assert task_cls == "molmo_spaces.tasks.pick_task.PickTask"

    def test_task_has_pickup_obj_name(self, benchmark_episodes):
        """Verify pick task has required pickup_obj_name field."""
        first_episode = benchmark_episodes[0]
        pickup_obj_name = first_episode.task.get("pickup_obj_name")
        assert pickup_obj_name is not None
        assert pickup_obj_name == "cup_b31afea732a344b3a47a266ce9bb531f_1_0_0"

    def test_cameras_present(self, benchmark_episodes):
        """Verify benchmark has camera specifications."""
        first_episode = benchmark_episodes[0]
        assert len(first_episode.cameras) >= 1

        camera_names = [c["name"] if isinstance(c, dict) else c.name for c in first_episode.cameras]
        assert "wrist_camera" in camera_names

    def test_init_qpos_values(self, benchmark_episodes):
        """Verify expected init_qpos values from the benchmark."""
        first_episode = benchmark_episodes[0]

        expected_arm_qpos = [
            0.08781547099351883,
            -0.6426845788955688,
            -0.15489166975021362,
            -2.498445510864258,
            -0.13134074211120605,
            1.8058809041976929,
            0.25663477182388306,
        ]
        expected_gripper_qpos = [0.002959999954327941, 0.002959999954327941]

        actual_arm_qpos = first_episode.robot.init_qpos.get("arm", [])
        actual_gripper_qpos = first_episode.robot.init_qpos.get("gripper", [])

        np.testing.assert_allclose(
            actual_arm_qpos,
            expected_arm_qpos,
            rtol=1e-7,
            err_msg="Arm qpos from benchmark doesn't match expected values",
        )
        np.testing.assert_allclose(
            actual_gripper_qpos,
            expected_gripper_qpos,
            rtol=1e-7,
            err_msg="Gripper qpos from benchmark doesn't match expected values",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
