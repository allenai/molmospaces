"""Gymnasium registration for molmospaces tasks.

Tasks are gymnasium envs (see :mod:`molmo_spaces.tasks.task`), so they can be
built through ``gymnasium.make`` once registered:

    import molmo_spaces.tasks.gym_registration as reg
    reg.register_configs()
    env = gymnasium.make("MolmoSpaces/FrankaPickAndPlace-v0")

That builds the config's task sampler, samples **one** episode from it and hands
back the resulting task. Nothing about the sampler path changes -- this is the
same ``sample_task()`` data generation calls, just wrapped so gymnasium can
construct it. The env therefore holds a single episode: ``reset()`` works once
and raises ``NotImplementedError`` after that.

Env ids come from the data generation config registry, so anything registered
with ``@register_config`` is available here under the same name.

Read ``docs/gym_compatibility.md`` before using these envs with third-party RL
code: they declare neither ``action_space`` nor ``observation_space``, so
``gymnasium.utils.env_checker.check_env`` and most wrappers will not work.
"""

import logging
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.data_generation.config_registry import (
    get_config_class,
    list_available_configs,
)
from molmo_spaces.tasks.task import BaseMujocoTask

if TYPE_CHECKING:
    from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler

log = logging.getLogger(__name__)

NAMESPACE = "MolmoSpaces"


def env_id_for_config(config_name: str, version: int = 0) -> str:
    """The gymnasium env id for a registered config name."""
    return f"{NAMESPACE}/{config_name}-v{version}"


def make_env(
    config_name: str | None = None,
    exp_config: MlSpacesExpConfig | None = None,
    *,
    config_overrides: dict[str, Any] | None = None,
    task_sampler: "BaseMujocoTaskSampler | None" = None,
    seed: int | None = None,
    render_camera: str | None = None,
    **sample_task_kwargs: Any,
) -> BaseMujocoTask:
    """Build a task sampler, sample one episode from it, and return that task.

    Args:
        config_name: Name of a config in the data generation config registry.
        exp_config: An already-built config, instead of ``config_name``.
        config_overrides: Attributes to set on the config before sampling.
        task_sampler: Sample from this sampler rather than building one. The
            caller keeps ownership: it is not closed with the task.
        seed: Seeds episode sampling via ``seed_task_sampling``. NOTE that is
            process-global, see ``docs/gym_compatibility.md``.
        render_camera: Camera ``render()`` should use; defaults to the first
            camera in the config.
        **sample_task_kwargs: Forwarded to ``sample_task`` (``house_index``,
            ``force_advance_scene``).

    Returns:
        A task holding one sampled episode.
    """
    if (config_name is None) == (exp_config is None):
        raise ValueError("Pass exactly one of config_name or exp_config")

    if exp_config is None:
        exp_config = get_config_class(config_name)()

    for key, value in (config_overrides or {}).items():
        if not hasattr(exp_config, key):
            raise ValueError(f"Config {type(exp_config).__name__} has no attribute {key!r}")
        setattr(exp_config, key, value)

    sampler = task_sampler
    if sampler is None:
        sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)

    try:
        if seed is not None:
            sampler.seed_task_sampling(seed)

        task = sampler.sample_task(**sample_task_kwargs)
        if task is None:
            raise RuntimeError(
                f"{type(sampler).__name__} returned no task (max_tasks reached); "
                f"call task_sampler.reset() to start over."
            )
        if task._env.n_batch != 1:
            raise ValueError(
                f"The gymnasium interface is single-environment only, got "
                f"n_batch={task._env.n_batch}. Use the task sampler directly for batches."
            )
    except BaseException:
        # We built the sampler, so nothing else will close it.
        if task_sampler is None:
            sampler.close()
        raise

    task.render_camera = render_camera
    # A gym caller resets per episode, and this env only has the one.
    task.gym_single_episode = True
    if task_sampler is None:
        # Hand the sampler to the task so task.close() closes it. A caller-supplied
        # sampler stays the caller's to close.
        task._gym_sampler = sampler
    return task


def register_configs(version: int = 0) -> list[str]:
    """Register every config in the config registry as a gymnasium env.

    Idempotent: ids already present in the gymnasium registry are skipped, so
    calling this more than once (or after a partial registration) is safe.

    Returns:
        The env ids registered by this call.
    """
    registered = []
    for config_name in list_available_configs():
        env_id = env_id_for_config(config_name, version=version)
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="molmo_spaces.tasks.gym_registration:make_env",
            kwargs={"config_name": config_name},
            # Task horizon is enforced by the task itself (is_timed_out), and
            # letting gym also wrap it would truncate twice.
            max_episode_steps=None,
            # These envs cannot pass gym's checker (no spaces declared).
            disable_env_checker=True,
        )
        registered.append(env_id)

    log.info(f"Registered {len(registered)} molmospaces envs with gymnasium")
    return registered
