"""
Rollout test for RBY1 nav-to-object with the A* planner policy.

This is a "single sample, whole module" characterization test: build the
NavToObjTaskSampler + NavToObjTask + AStarPlannerPolicy stack once at a fixed
seed, step it for a handful of policy steps, and assert the resulting joint
positions stay pinned to a small in-repo fixture (regenerable with
`--regen-golden`, no resource-manager/R2 upload needed, unlike the
Franka/RUM tests' golden data).

Before this test, `policy/solvers/navigation/astar_planner_policy.py` and
`tasks/nav_task_sampler.py` had no test coverage at all -- everything else in
mlspaces_tests/data_generation exercises object-manipulation tasks (pick,
pick-and-place, open/close), never navigation.
"""

from pathlib import Path

import numpy as np
import pytest

from mlspaces_tests.data_generation.config import NavToObjTestConfig
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler
from molmo_spaces.utils.test_utils import run_task_for_steps_with_observations

FIXTURE_DIR = Path(__file__).resolve().parent / "test_fixtures" / "nav_to_obj"
GOLDEN_QPOS_PATH = FIXTURE_DIR / "rollout_final_qpos.npy"
NUM_STEPS = 5

# `--regen-golden` is registered in this directory's conftest.py.


@pytest.fixture(scope="module")
def nav_config():
    return NavToObjTestConfig()


@pytest.fixture(scope="module")
def nav_task_sampler(nav_config):
    task_sampler_class = nav_config.task_sampler_config.task_sampler_class
    assert task_sampler_class is NavToObjTaskSampler
    sampler = task_sampler_class(nav_config)
    # Anchor determinism at the moment sampling starts, matching the
    # Franka/RUM tests' convention (see their fixtures for why).
    sampler.seed_task_sampling(sampler.current_seed)
    sampler.reset()
    yield sampler
    sampler.env.close()


@pytest.fixture(scope="module")
def nav_task(nav_task_sampler):
    task = nav_task_sampler.sample_task()
    assert isinstance(task, NavToObjTask)
    return task


@pytest.fixture(scope="module")
def nav_rollout(nav_config, nav_task):
    nav_task.reset()

    policy_config = nav_config.policy_config
    policy = policy_config.policy_factory(nav_config, nav_task)
    policy.reset()

    initial_qpos, final_qpos, initial_obs, final_obs = run_task_for_steps_with_observations(
        nav_task, policy, num_steps=NUM_STEPS
    )
    return {
        "initial_qpos": initial_qpos,
        "final_qpos": final_qpos,
    }


def test_nav_imports():
    from molmo_spaces.data_generation.config.nav_to_obj_configs import NavToObjDataGenConfig
    from molmo_spaces.policy.solvers.navigation.astar_planner_policy import AStarPlannerPolicy

    assert NavToObjTestConfig is not None
    assert NavToObjDataGenConfig is not None
    assert NavToObjTaskSampler is not None
    assert NavToObjTask is not None
    assert AStarPlannerPolicy is not None


def test_nav_task_sampler_produces_task(nav_config, nav_task):
    policy_config = nav_config.policy_config
    assert policy_config.policy_cls is not None
    assert nav_task.env is not None


def test_nav_rollout_moves_the_robot(nav_rollout):
    """Running the planner policy for a few steps should change the robot's qpos."""
    initial_qpos = nav_rollout["initial_qpos"]
    final_qpos = nav_rollout["final_qpos"]

    assert initial_qpos.shape == final_qpos.shape
    qpos_diff = np.abs(final_qpos - initial_qpos)
    assert np.any(qpos_diff > 1e-4), "qpos should change after stepping the nav policy"


def test_nav_rollout_matches_golden(nav_rollout, request):
    """Pin the rollout's final qpos so a planner/task regression shows up here."""
    final_qpos = nav_rollout["final_qpos"]

    if request.config.getoption("--regen-golden"):
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(GOLDEN_QPOS_PATH, final_qpos)
        pytest.skip(f"Regenerated golden fixture at {GOLDEN_QPOS_PATH}")

    expected_final_qpos = np.load(GOLDEN_QPOS_PATH)
    assert final_qpos.shape == expected_final_qpos.shape
    np.testing.assert_allclose(
        final_qpos,
        expected_final_qpos,
        atol=1e-4,
        err_msg=(
            "Final qpos after a 5-step nav rollout no longer matches the golden "
            f"fixture at {GOLDEN_QPOS_PATH}. If this is an intentional change to "
            "the planner/task sampler, regenerate with:\n"
            "  python -m pytest mlspaces_tests/data_generation/test_nav_to_obj_rollout.py "
            "--regen-golden"
        ),
    )
