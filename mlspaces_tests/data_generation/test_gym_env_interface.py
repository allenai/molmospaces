"""Tests for the gymnasium interface on tasks.

``GymEnv`` wraps a task sampler: ``reset()`` samples an episode and returns its
first observation, ``step()`` drives the task it sampled. These tests pin the
parts of the gymnasium contract that hold; see ``docs/gym_compatibility.md`` for
the parts that do not.
"""

import gymnasium as gym
import numpy as np
import pytest

from mlspaces_tests.data_generation.config import FrankaPickAndPlaceDroidTestConfig
from molmo_spaces.data_generation.config_registry import (
    _MJT_CONFIG_REGISTRY,
    register_config,
)
from molmo_spaces.tasks import gym_env as gym_env_module
from molmo_spaces.tasks.gym_env import GymEnv
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
    """A reset env with its own sampler, shared by the read-only tests."""
    env = GymEnv(exp_config=gym_config)
    env.reset()
    yield env
    env.close()


def test_env_is_a_gym_env(gym_env):
    assert isinstance(gym_env, gym.Env)
    # gym.Env's own initialization must have happened, not just ours.
    assert gym_env.unwrapped is gym_env
    assert gym_env.np_random is not None


def test_reset_sampled_an_episode(gym_env):
    assert isinstance(gym_env.task, BaseMujocoTask)
    assert gym_env.task.env is gym_env.task_sampler.env
    assert gym_env.task.env.n_batch == 1
    assert gym_env.task.get_task_description().strip()


def test_reset_returns_observation_and_info(gym_env):
    observation, info = gym_env.reset()
    assert len(observation) == 1
    assert observation[0], "expected a non-empty observation dict"
    assert len(info) == 1


def test_reset_samples_a_new_episode(gym_config):
    """The point of the wrapper: a gym loop can reset per episode."""
    env = GymEnv(exp_config=gym_config)
    try:
        env.reset()
        first = env.task
        env.reset()
        assert env.task is not first, "reset() must sample a new task"
        assert first._env is None, "the replaced task must be closed"
    finally:
        env.close()


def test_step_and_render_before_reset_raise(gym_config):
    env = GymEnv(exp_config=gym_config)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.step({})
        with pytest.raises(RuntimeError, match="reset"):
            env.render()
    finally:
        env.close()


def test_reset_options_reach_sample_task(gym_config):
    """``options`` carries the sample_task arguments an env id cannot."""
    env = GymEnv(exp_config=gym_config)
    try:
        house_index = env.task_sampler._house_inds[0]
        env.reset(options={"house_index": house_index})
        assert env.task_sampler.current_house_index == house_index
        with pytest.raises(ValueError, match="Unsupported reset options"):
            env.reset(options={"nonsense": 1})
    finally:
        env.close()


def test_gym_path_and_sampler_path_agree_on_observation_keys(gym_config, gym_env):
    """The gym env drives a sampled task, so the observation contract is the same."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        sampler_task = sampler.sample_task()
        assert isinstance(sampler_task, BaseMujocoTask)
        # Read the first task's observation before sampling again: a second episode
        # can load a scene, which replaces the sim env the first task is bound to.
        sampler_obs, _ = sampler_task.reset()
        sampler_task.close()

        env = GymEnv(task_sampler=sampler)
        try:
            gym_obs, _ = env.reset()
            assert set(sampler_obs[0]) == set(gym_obs[0])
        finally:
            env.close()
    finally:
        sampler.close()


def test_close_leaves_a_caller_supplied_sampler_open(gym_config):
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        env = GymEnv(task_sampler=sampler)
        env.reset()
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
        monkeypatch.setattr(type(task.env), "n_batch", property(lambda self: 2))
        env = GymEnv(task_sampler=sampler)
        with pytest.raises(ValueError, match="single-environment only"):
            env.reset()
        task.close()
    finally:
        sampler.close()


def test_exhausted_sampler_raises_rather_than_returning_none(gym_config):
    """sample_task() returns None when out of tasks; the gym path must not."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        sampler._current_tasks_left = 0
        assert sampler.sample_task() is None
        env = GymEnv(task_sampler=sampler)
        with pytest.raises(RuntimeError, match="no task"):
            env.reset()
    finally:
        sampler.close()


def test_render_returns_an_rgb_frame(gym_env):
    frame = gym_env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert "rgb_array" in GymEnv.metadata["render_modes"]


def test_render_rejects_unsupported_mode(gym_env, monkeypatch):
    monkeypatch.setattr(gym_env, "render_mode", "human")
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        gym_env.render()


def test_spaces_are_deliberately_absent(gym_env):
    """Pinned so the omission is a decision, not an accident. See docs/gym_compatibility.md."""
    assert getattr(gym_env, "action_space", None) is None
    assert getattr(gym_env, "observation_space", None) is None


def test_requires_exactly_one_config_source(gym_config):
    with pytest.raises(ValueError, match="exactly one"):
        GymEnv()
    with pytest.raises(ValueError, match="exactly one"):
        GymEnv(config_name="X", exp_config=gym_config)


def test_a_task_sampler_may_not_be_combined_with_a_config(gym_config):
    """A sampler already has a config; two would leave it ambiguous which wins."""
    sampler = gym_config.task_sampler_config.task_sampler_class(gym_config)
    try:
        with pytest.raises(ValueError, match="brings its own config"):
            GymEnv(exp_config=gym_config, task_sampler=sampler)
    finally:
        sampler.close()


def test_gymnasium_make_returns_the_env_itself(gym_config):
    """``gym.make`` must hand back the GymEnv, not a wrapper.

    Gymnasium wrappers stopped proxying attribute access in 1.0, so an
    ``OrderEnforcing`` wrapper would hide ``task`` and ``task_sampler`` behind
    ``.unwrapped``. Hence ``order_enforce=False`` at registration.
    """
    # Env ids come from the datagen config registry, so the test config has to be
    # in it. Registered and removed here rather than left behind for other tests.
    config_name = "GymMakeTestConfig"
    env_id = gym_env_module.env_id_for_config(config_name)
    register_config(config_name, strict=False)(type(gym_config))
    try:
        gym_env_module.register_configs()
        env = gym.make(
            env_id,
            config_overrides={"use_passive_viewer": False, "use_wandb": False, "profile": False},
        )
        try:
            assert isinstance(env, GymEnv), "a wrapper would hide the env API"
            assert env.unwrapped is env
            env.reset()
            assert env.task.get_task_description().strip()
            assert env.render().ndim == 3
        finally:
            env.close()
    finally:
        _MJT_CONFIG_REGISTRY.pop(config_name, None)
        gym.registry.pop(env_id, None)


def test_registered_envs_skip_the_checker_and_the_order_wrapper():
    import molmo_spaces.data_generation.config.object_manipulation_datagen_configs  # noqa: F401

    gym_env_module.register_configs()
    spec = gym.registry[gym_env_module.env_id_for_config("FrankaPickDroidDataGenConfig")]
    assert spec.disable_env_checker, "no spaces are declared, so the checker cannot pass"
    assert not spec.order_enforce, "the wrapper would hide the env API"
    assert spec.max_episode_steps is None, "the task enforces its own horizon"


def test_register_configs_is_idempotent():
    import molmo_spaces.data_generation.config.object_manipulation_datagen_configs  # noqa: F401

    gym_env_module.register_configs()
    assert [
        env_id for env_id in gym.registry if env_id.startswith(f"{gym_env_module.NAMESPACE}/")
    ], "expected at least one config to be registered"
    # A second call must not raise or re-register.
    assert gym_env_module.register_configs() == []
