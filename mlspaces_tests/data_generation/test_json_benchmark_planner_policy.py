"""
High-level evaluation test: run_evaluation() with a real planner policy.

test_json_benchmark_integration.py already exercises the full run_evaluation()
pipeline end to end, but only with DummyPolicy (a no-op), which is used there
specifically to prove that a *stationary* robot fails the task. That leaves
the actual planner-policy code path through the evaluation harness untested:
PickPlannerPolicy is instantiated per-episode by
`exp_config.policy_config.policy_factory(exp_config, task)` inside
`data_generation.pipeline`, and none of the JSON-benchmark eval tests ever
build one.

This test runs the same one-episode benchmark through `run_evaluation()` with
`PickPlannerPolicyConfig` (a real object-manipulation planner) instead, for a
handful of steps, and pins the resulting qpos to an in-repo golden fixture --
a "single sample, whole module" characterization test of the evaluation
harness + planner-policy stack together, complementing the lower-level
Franka-pick rollout in test_franka_pick.py (which exercises PickPlannerPolicy
directly against a task sampled by PickTaskSampler, not through the JSON
benchmark / run_evaluation() replay path).

Note: `filter_colliding_grasps=False` is required here. `PickPlannerPolicy`'s
default grasp-collision check (grasp_sample.get_noncolliding_grasp_mask) adds
"grasp_collision_N" probe bodies to the *scene*, which the normal data
generation task samplers (PickTaskSampler et al.) add during scene
construction but the JSON-benchmark replay path does not -- benchmark
episodes describe robot/object/camera state, not scene-construction-time
probe geometry. Disabling the check here doesn't skip any of the planning
or motion execution being tested; it just means grasp candidates aren't
collision-filtered before being ranked, which is fine for a coverage/
regression test that isn't trying to validate grasp quality.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.evaluation.eval_main import EvaluationResults, run_evaluation

TEST_BENCHMARK_DIR = Path(__file__).parent / "test_benchmark"
GOLDEN_QPOS_PATH = (
    Path(__file__).resolve().parent
    / "test_fixtures"
    / "json_benchmark_planner_policy"
    / "final_qpos.npy"
)
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


pytestmark = pytest.mark.skipif(
    not _check_assets_available(),
    reason="holodeck-objaverse-val scene assets not installed",
)


class _PlannerPolicyBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """Same shape as test_json_benchmark_integration.py's dummy-policy config,
    but with a real planner (see module docstring for filter_colliding_grasps)."""

    task_horizon: int = 10
    seed: int = 42
    policy_dt_ms: float = EVAL_POLICY_DT_MS

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig(filter_colliding_grasps=False)

    @property
    def tag(self) -> str:
        return "test_json_benchmark_planner_policy"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


@pytest.fixture(scope="module")
def single_episode_benchmark(tmp_path_factory) -> Path:
    """Same single-episode benchmark test_json_benchmark_integration.py builds,
    reconstructed here so this file has no import-order dependency on it."""
    all_episodes = load_all_episodes(TEST_BENCHMARK_DIR)
    first_episode = all_episodes[0]

    benchmark_dir = tmp_path_factory.mktemp("single_episode_benchmark_planner")
    benchmark_file = benchmark_dir / "benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump([first_episode.model_dump()], f, indent=2)

    return benchmark_dir


@pytest.fixture(scope="module")
def planner_evaluation_results(tmp_path_factory, single_episode_benchmark) -> EvaluationResults:
    """Run the full evaluation pipeline with a real planner policy.

    Unlike the DummyPolicy fixture in test_json_benchmark_integration.py, no
    `preloaded_policy` is passed: `PickPlannerPolicy` needs the sampled task at
    construction time (it reads object/gripper poses off it immediately), so it
    must go through the normal per-episode
    `policy_config.policy_factory(exp_config, task)` path inside
    data_generation.pipeline, not be built upfront.
    """
    output_dir = tmp_path_factory.mktemp("eval_output_planner")
    results = run_evaluation(
        eval_config_cls=_PlannerPolicyBenchmarkEvalConfig,
        benchmark_dir=single_episode_benchmark,
        checkpoint_path=None,  # PickPlannerPolicy isn't a checkpointed/learned policy
        task_horizon_sec=EVAL_TASK_HORIZON_STEPS * EVAL_POLICY_DT_MS / 1000.0,
        output_dir=output_dir,
        num_workers=1,
        use_wandb=False,
    )
    return results


def _load_qpos_endpoints(evaluation_results: EvaluationResults) -> tuple[np.ndarray, np.ndarray]:
    h5_files = list(evaluation_results.output_dir.rglob("*.h5"))
    assert h5_files, "No HDF5 output files found"
    with h5py.File(h5_files[0], "r") as f:
        qpos_array = f["traj_0"]["obs"]["agent"]["qpos"][()]
        initial_qpos_dict = json.loads(bytes(qpos_array[0]).rstrip(b"\x00").decode("utf-8"))
        final_qpos_dict = json.loads(bytes(qpos_array[-1]).rstrip(b"\x00").decode("utf-8"))
    # Concatenate move groups in a stable (sorted) order so the flattened
    # vector's layout doesn't depend on dict insertion order.
    initial_qpos = np.concatenate(
        [np.asarray(initial_qpos_dict[k]) for k in sorted(initial_qpos_dict)]
    )
    final_qpos = np.concatenate([np.asarray(final_qpos_dict[k]) for k in sorted(final_qpos_dict)])
    return initial_qpos, final_qpos


def test_planner_evaluation_completes_successfully(
    planner_evaluation_results: EvaluationResults,
):
    """The planner-policy path through run_evaluation() should complete and produce output."""
    assert isinstance(planner_evaluation_results, EvaluationResults)
    assert planner_evaluation_results.total_count == 1
    assert planner_evaluation_results.output_dir.exists()

    h5_files = list(planner_evaluation_results.output_dir.rglob("*.h5"))
    assert len(h5_files) > 0, "No HDF5 trajectory files created"


def test_planner_moves_the_robot(planner_evaluation_results: EvaluationResults):
    """Contrast with test_dummy_policy_causes_task_failure: unlike DummyPolicy,
    a real planner should actually move the robot within a handful of steps."""
    initial_qpos, final_qpos = _load_qpos_endpoints(planner_evaluation_results)
    assert initial_qpos.shape == final_qpos.shape
    assert np.any(np.abs(final_qpos - initial_qpos) > 1e-4), (
        "Robot qpos should change when driven by a real planner policy"
    )


def test_planner_rollout_matches_golden(planner_evaluation_results: EvaluationResults, request):
    """Pin the planner rollout's final qpos through the full eval harness."""
    _, final_qpos = _load_qpos_endpoints(planner_evaluation_results)

    if request.config.getoption("--regen-golden"):
        GOLDEN_QPOS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(GOLDEN_QPOS_PATH, final_qpos)
        pytest.skip(f"Regenerated golden fixture at {GOLDEN_QPOS_PATH}")

    expected_final_qpos = np.load(GOLDEN_QPOS_PATH)
    assert final_qpos.shape == expected_final_qpos.shape
    np.testing.assert_allclose(
        final_qpos,
        expected_final_qpos,
        atol=1e-4,
        err_msg=(
            "Final qpos from the planner-policy evaluation rollout no longer "
            f"matches the golden fixture at {GOLDEN_QPOS_PATH}. If this is an "
            "intentional change, regenerate with:\n"
            "  python -m pytest "
            "mlspaces_tests/data_generation/test_json_benchmark_planner_policy.py "
            "--regen-golden"
        ),
    )
