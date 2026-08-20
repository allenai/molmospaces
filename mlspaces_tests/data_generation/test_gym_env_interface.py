"""Tests for the gymnasium interface on tasks.

``gym_registration.make_env`` builds a task sampler, samples one episode and
returns the resulting task as a ``gymnasium.Env``. These tests pin the parts of
the gymnasium contract that hold; see ``docs/gym_compatibility.md`` for the
parts that do not, chief among them that the env holds a single episode.
"""

import gymnasium as gym
import numpy as np
import pytest

from mlspaces_tests.data_generation.config import FrankaPickAndPlaceDroidTestConfig
from molmo_spaces.tasks import gym_registration
from molmo_spaces.tasks.task import BaseMujocoTask


@pytest.fixture(scope="module")
def gym_config():
    config = FrankaPickAndPlaceDroidTestConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    return config


@pytest.fixture(scope="module")
def gym_env(gym_config):
    """An env from the gymnasium entry point, with its own sampler."""
    env = gym_registration.make_env(exp_config=gym_config)
    yield env
    env.close()


def test_task_is_a_gym_env(gym_env):
    assert isinstance(gym_env, gym.Env)
    assert isinstance(gym_env, BaseMujocoTask)
    # gym.Env's own initialization must have happened, not just ours.
    assert gym_env.unwrapped is gym_env
    assert gym_env.np_random is not None


def test_make_env_sampled_an_episode(gym_env):
    assert gym_env._gym_sampler is not None, "make_env should own the sampler it built"
    assert gym_env._env is gym_env._gym_sampler.env
    assert gym_env._env.n_batch == 1
    assert gym_env.get_task_description().strip()


def test_reset_returns_observation_and_info(gym_env):
    observation, info = gym_env.reset()
    assert len(observation) == 1
    assert observation[0], "expected a non-empty observation dict"
    assert len(info) == 1


def test_second_reset_is_not_supported(gym_config):
    """One episode per env: the second reset says so rather than silently repeating."""
    env = gym_registration.make_env(exp_config=gym_config)
    try:
        assert env.gym_single_episode
        env.reset()
        with pytest.raises(NotImplementedError, match="single episode"):
            env.reset()
    finally:
        env.close()


def test_sampled_tasks_may_still_be_reset_more_than_once(gym_config):
    """The single-reset rule is the gym path's, not the sampler path's.

    Data generation resets once before register_policy and once after, so the
    check must stay off for tasks the caller sampled itself.
    """
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        task = sampler.sample_task()
        assert not task.gym_single_episode
        task.reset()
        task.reset()
        task.close()
    finally:
        sampler.close()


def test_reset_rejects_seed_and_options(gym_config):
    """Episode selection belongs to the sampler, so reset() cannot honour these."""
    env = gym_registration.make_env(exp_config=gym_config)
    try:
        with pytest.raises(NotImplementedError, match="not supported"):
            env.reset(seed=0)
        with pytest.raises(NotImplementedError, match="not supported"):
            env.reset(options={"house_index": 0})
        # A rejected reset must not count against the single allowed one.
        env.reset()
    finally:
        env.close()


def test_gym_path_and_sampler_path_agree_on_observation_keys(gym_config, gym_env):
    """The gym env is a sampled task, so its observation contract is the same one."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        sampler_task = sampler.sample_task()
        assert isinstance(sampler_task, BaseMujocoTask)
        sampler_obs, _ = sampler_task.reset()
        gym_env_task = gym_registration.make_env(exp_config=gym_config, task_sampler=sampler)
        gym_obs, _ = gym_env_task.reset()
        assert set(sampler_obs[0]) == set(gym_obs[0])
        gym_env_task.close()
        sampler_task.close()
    finally:
        sampler.close()


def test_make_env_does_not_close_a_caller_supplied_sampler(gym_config):
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        env = gym_registration.make_env(exp_config=gym_config, task_sampler=sampler)
        assert env.gym_single_episode, "still a gym env, whoever built the sampler"
        assert env._gym_sampler is None, "a caller's sampler stays the caller's to close"
        env.close()
        # Still usable, so close() left it alone.
        assert sampler.sample_task() is not None
    finally:
        sampler.close()


def test_batched_config_is_rejected_on_the_gym_path(gym_config, monkeypatch):
    """The gymnasium interface is single-env only, and says so up front."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        task = sampler.sample_task()
        monkeypatch.setattr(type(task._env), "n_batch", property(lambda self: 2))
        with pytest.raises(ValueError, match="single-environment only"):
            gym_registration.make_env(exp_config=gym_config, task_sampler=sampler)
        task.close()
    finally:
        sampler.close()


def test_exhausted_sampler_raises_rather_than_returning_none(gym_config):
    """sample_task() returns None when out of tasks; the gym path must not."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        sampler._current_tasks_left = 0
        assert sampler.sample_task() is None
        with pytest.raises(RuntimeError, match="no task"):
            gym_registration.make_env(exp_config=gym_config, task_sampler=sampler)
    finally:
        sampler.close()


def test_render_returns_an_rgb_frame(gym_env):
    frame = gym_env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert "rgb_array" in type(gym_env).metadata["render_modes"]


def test_render_rejects_unsupported_mode(gym_env, monkeypatch):
    monkeypatch.setattr(gym_env, "render_mode", "human")
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        gym_env.render()


def test_spaces_are_deliberately_absent(gym_env):
    """Pinned so the omission is a decision, not an accident. See docs/gym_compatibility.md."""
    assert getattr(gym_env, "action_space", None) is None
    assert getattr(gym_env, "observation_space", None) is None


def test_make_env_requires_exactly_one_source(gym_config):
    with pytest.raises(ValueError, match="exactly one"):
        gym_registration.make_env()
    with pytest.raises(ValueError, match="exactly one"):
        gym_registration.make_env(config_name="X", exp_config=gym_config)


def test_register_configs_is_idempotent():
    import molmo_spaces.data_generation.config.object_manipulation_datagen_configs  # noqa: F401

    first = gym_registration.register_configs()
    assert first, "expected at least one config to register"
    assert all(env_id in gym.registry for env_id in first)
    # A second call must not raise or re-register.
    assert gym_registration.register_configs() == []
