"""Tests for the gymnasium interface on tasks.

A task can build itself -- creating its own sampler, having it prepare a scene
and sample an episode -- instead of being built by a sampler. These tests cover
that path and pin the parts of the gymnasium contract that do hold; see
``docs/gym_compatibility.md`` for the parts that do not.
"""

import gymnasium as gym
import numpy as np
import pytest

from mlspaces_tests.data_generation.config import FrankaPickAndPlaceDroidTestConfig
from molmo_spaces.tasks import gym_registration
from molmo_spaces.tasks.task import BaseMujocoTask, EpisodeSource
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
    task = gym_config.task_config.task_cls(exp_config=gym_config)
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
        task = gym_config.task_config.task_cls(exp_config=gym_config)
        monkeypatch.setattr(type(task._env), "n_batch", property(lambda self: 2))
        with pytest.raises(ValueError, match="single-environment only"):
            gym_config.task_config.task_cls(exp_config=gym_config, task_sampler=task._sampler)
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
            gym_config.task_config.task_cls(exp_config=gym_config, task_sampler=sampler)
    finally:
        sampler.close()


def test_first_reset_keeps_the_episode_built_at_construction(gym_config):
    """__init__ already sampled an episode; the first reset must not throw it away.

    Not just for speed: samplers consume their per-house candidate pool, so a
    wasted episode brings the next HouseInvalidForTask closer.
    """
    task = gym_config.task_config.task_cls(exp_config=gym_config)
    try:
        description = task.get_task_description()
        task.reset()
        assert task.get_task_description() == description
    finally:
        task.close()


def test_resets_keep_the_task_bound_to_the_live_env(gym_config):
    """A self-configured task advances episodes on reset, as gym callers expect."""
    task = gym_config.task_config.task_cls(exp_config=gym_config)
    try:
        assert task.episode_source is EpisodeSource.SELF, "derived from omitting env"
        task.reset()
        first = task.get_task_description()
        obs, info = task.reset()
        assert obs[0], "expected observations for the new episode"
        assert task.episode_step_count == 0
        assert not task.observation_cache[1:], "caches must be cleared for the new episode"
        # The sampler advanced, so task state was rebuilt; the description may or
        # may not differ (same house can resample the same objects), but the env
        # binding must always be the sampler's current one.
        assert task._env is task._sampler.env
        assert isinstance(first, str)
    finally:
        task.close()


def test_sampler_built_task_reset_does_not_resample(gym_config):
    """The data generation path keeps its reset semantics: clear state, same episode."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        task = sampler.sample_task()
        description = task.get_task_description()
        task.reset()
        task.reset()
        assert task.get_task_description() == description
        assert task.episode_source is EpisodeSource.SAMPLER
        task.close()
    finally:
        sampler.close()


def test_reset_rejects_unknown_options(self_configured_task):
    """Unknown options are prepare_episode's TypeError, not a hand-rolled check."""
    with pytest.raises(TypeError, match="not_a_real_option"):
        self_configured_task.reset(options={"not_a_real_option": 1})


def test_render_returns_an_rgb_frame(self_configured_task):
    frame = self_configured_task.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert "rgb_array" in type(self_configured_task).metadata["render_modes"]


def test_render_rejects_unsupported_mode(self_configured_task, monkeypatch):
    monkeypatch.setattr(self_configured_task, "render_mode", "human")
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        self_configured_task.render()


def test_spaces_are_deliberately_absent(self_configured_task):
    """Pinned so the omission is a decision, not an accident. See docs/gym_compatibility.md."""
    assert getattr(self_configured_task, "action_space", None) is None
    assert getattr(self_configured_task, "observation_space", None) is None


def test_make_env_builds_a_task_from_a_config_object(gym_config):
    env = gym_registration.make_env(exp_config=gym_config)
    try:
        assert isinstance(env, gym.Env)
        assert env.episode_source is EpisodeSource.SELF
        assert env.get_task_description().strip()
    finally:
        env.close()


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
