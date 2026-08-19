"""Gymnasium registration for molmospaces tasks.

Tasks are gymnasium envs (see :mod:`molmo_spaces.tasks.task`), so they can be
built through ``gymnasium.make`` once registered:

    import molmo_spaces.tasks.gym_registration as reg
    reg.register_configs()
    env = gymnasium.make("MolmoSpaces/FrankaPickAndPlace-v0")

Env ids come from the data generation config registry, so anything registered
with ``@register_config`` is available here under the same name.

Read ``docs/gym_compatibility.md`` before using these envs with third-party RL
code: they declare neither ``action_space`` nor ``observation_space``, so
``gymnasium.utils.env_checker.check_env`` and most wrappers will not work.
"""

import logging
from typing import Any

import gymnasium as gym

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.data_generation.config_registry import (
    get_config_class,
    list_available_configs,
)
from molmo_spaces.tasks.task import BaseMujocoTask

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
    **task_kwargs: Any,
) -> BaseMujocoTask:
    """Build a task as a gymnasium env, sampling its own episode.

    Args:
        config_name: Name of a config in the data generation config registry.
        exp_config: An already-built config, instead of ``config_name``.
        config_overrides: Attributes to set on the config before sampling.
        **task_kwargs: Forwarded to the task (e.g. ``render_mode``,
            ``task_sampler``, ``episode_options``).

    Returns:
        A task bound to a freshly sampled episode.
    """
    if (config_name is None) == (exp_config is None):
        raise ValueError("Pass exactly one of config_name or exp_config")

    if exp_config is None:
        exp_config = get_config_class(config_name)()

    for key, value in (config_overrides or {}).items():
        if not hasattr(exp_config, key):
            raise ValueError(f"Config {type(exp_config).__name__} has no attribute {key!r}")
        setattr(exp_config, key, value)

    task_cls = exp_config.task_config.task_cls
    if task_cls is None:
        raise ValueError(
            f"Config {type(exp_config).__name__} has no task_config.task_cls, "
            f"so there is no task class to build."
        )

    return task_cls(exp_config=exp_config, configure=True, **task_kwargs)


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
