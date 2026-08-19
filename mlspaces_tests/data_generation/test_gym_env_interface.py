"""Tests for the gymnasium interface on tasks.

A task can build itself -- creating its own sampler, having it prepare a scene
and sample an episode -- instead of being built by a sampler. These tests cover
that path and pin the parts of the gymnasium contract that do hold; see
``docs/gym_compatibility.md`` for the parts that do not.
"""

import gymnasium as gym
import pytest

from mlspaces_tests.data_generation.config import FrankaPickAndPlaceDroidTestConfig
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler_errors import EpisodesExhausted


@pytest.fixture(scope="module")
def gym_config():
    config = FrankaPickAndPlaceDroidTestConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    return config


@pytest.fixture(scope="module")
def self_configured_task(gym_config):
    """A task that sampled its own episode (the gymnasium entry point)."""
    task = gym_config.task_config.task_cls(exp_config=gym_config, configure=True)
    yield task
    task.close()


def test_task_is_a_gym_env(self_configured_task):
    assert isinstance(self_configured_task, gym.Env)
    # gym.Env's own initialization must have happened, not just ours.
    assert self_configured_task.unwrapped is self_configured_task
    assert self_configured_task.np_random is not None


def test_self_configuration_produced_an_episode(self_configured_task):
    task = self_configured_task
    assert task._owns_sampler, "task should have created its own sampler"
    assert task._env is task._sampler.env, "task must bind the sampler's current sim env"
    assert task._env.n_batch == 1
    assert task.get_task_description().strip()
    # Step ratios are derived from the bound env's model, so they must be sane.
    assert task._n_sim_steps_per_ctrl >= 1
    assert task._n_ctrl_steps_per_policy >= 1


def test_reset_returns_observation_and_info(self_configured_task):
    observation, info = self_configured_task.reset()
    assert len(observation) == 1
    assert observation[0], "expected a non-empty observation dict"
    assert len(info) == 1


def test_sampler_path_and_gym_path_agree_on_observation_keys(gym_config, self_configured_task):
    """Both construction paths must yield the same observation contract."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        sampler_task = sampler.sample_task()
        assert isinstance(sampler_task, BaseMujocoTask)
        sampler_obs, _ = sampler_task.reset()
        gym_obs, _ = self_configured_task.reset()
        assert set(sampler_obs[0]) == set(gym_obs[0])
        sampler_task.close()
    finally:
        sampler.close()


def test_batched_config_is_rejected_on_the_gym_path(gym_config, monkeypatch):
    """The gymnasium interface is single-env only, and says so up front."""
    task = None
    try:
        task = gym_config.task_config.task_cls(exp_config=gym_config, configure=True)
        monkeypatch.setattr(type(task._env), "n_batch", property(lambda self: 2))
        with pytest.raises(ValueError, match="single-environment only"):
            gym_config.task_config.task_cls(
                exp_config=gym_config, task_sampler=task._sampler, configure=True
            )
    finally:
        if task is not None:
            task.close()


def test_exhausted_sampler_raises_rather_than_returning_none(gym_config):
    """sample_task() returns None when out of tasks; the gym path must not."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        sampler._current_tasks_left = 0
        assert sampler.sample_task() is None
        with pytest.raises(EpisodesExhausted):
            gym_config.task_config.task_cls(
                exp_config=gym_config, task_sampler=sampler, configure=True
            )
    finally:
        sampler.close()
