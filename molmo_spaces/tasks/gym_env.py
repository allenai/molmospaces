"""A gymnasium env for molmospaces tasks.

:class:`GymEnv` wraps a task sampler, so it behaves the way gym callers expect:
``reset()`` samples a fresh episode, ``step()`` drives the task it sampled.

    import molmo_spaces.tasks.gym_env as gym_env
    gym_env.register_configs()
    env = gymnasium.make("MolmoSpaces/FrankaPickAndPlace-v0")
    observation, info = env.reset()   # samples an episode
    observation, info = env.reset()   # samples another one

Nothing about the sampler path changes: a task still holds exactly one episode
and is still built by ``sample_task()``. The env just owns the sampler and does
that call for you, keeping the current task on ``env.task``.

Env ids come from the data generation config registry, so anything registered
with ``@register_config`` is available here under the same name.

Read ``docs/gym_compatibility.md`` before using these envs with third-party RL
code: they declare neither ``action_space`` nor ``observation_space``, so
``gymnasium.utils.env_checker.check_env`` and most wrappers will not work.
"""

import logging
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

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

# The only sample_task() arguments reset(options=...) may set.
RESET_OPTIONS = ("house_index", "force_advance_scene")


class GymEnv(gym.Env):
    """A gymnasium env that samples its episodes from a molmospaces task sampler.

    The env owns the sampler; the task is the episode currently in the scene.
    ``reset()`` samples the next one and returns its first observation, so a
    normal gym loop over episodes works. There is no attribute proxying: the
    task's own API is reached through ``env.task`` (``register_policy``,
    ``get_task_description``, ``env``, ...).

    See ``docs/gym_compatibility.md`` for which parts of the gymnasium contract
    hold. Notably neither ``action_space`` nor ``observation_space`` is declared,
    and only ``n_batch == 1`` is supported.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Deliberately not declared, see docs/gym_compatibility.md:
    #   action_space      -- actions are dicts keyed by move group, with no space
    #   observation_space -- sensors have per-sensor spaces, but observations are
    #                        list[dict] (one per batch element), so no single
    #                        space describes what reset()/step() return

    def __init__(
        self,
        config_name: str | None = None,
        exp_config: MlSpacesExpConfig | None = None,
        *,
        config_overrides: dict[str, Any] | None = None,
        task_sampler: "BaseMujocoTaskSampler | None" = None,
        render_mode: str | None = None,
        render_camera: str | None = None,
        **sample_task_kwargs: Any,
    ) -> None:
        """Give either a config to build a sampler from, or a ready sampler.

        Args:
            config_name: Name of a config in the data generation config registry.
            exp_config: An already-built config, instead of ``config_name``.
            config_overrides: Attributes to set on the config before sampling.
            task_sampler: Sample from this sampler instead of building one from a
                config. It carries its own config, so pass no config with it. The
                caller keeps ownership: ``close()`` leaves it open.
            render_mode: ``None`` or ``"rgb_array"``; ``render()`` returns a frame
                either way.
            render_camera: Camera ``render()`` should use; defaults to the first
                camera in the config.
            **sample_task_kwargs: Defaults forwarded to every ``sample_task()``
                call (``house_index``, ``force_advance_scene``). Per-episode
                values go through ``reset(options=...)``.
        """
        if task_sampler is not None:
            if config_name is not None or exp_config is not None or config_overrides:
                raise ValueError(
                    "task_sampler brings its own config; pass no config_name, "
                    "exp_config or config_overrides with it."
                )
            self._owns_task_sampler = False
        else:
            if (config_name is None) == (exp_config is None):
                raise ValueError("Pass exactly one of config_name, exp_config or task_sampler")

            if exp_config is None:
                exp_config = get_config_class(config_name)()

            for key, value in (config_overrides or {}).items():
                if not hasattr(exp_config, key):
                    raise ValueError(f"Config {type(exp_config).__name__} has no attribute {key!r}")
                setattr(exp_config, key, value)

            task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
            self._owns_task_sampler = True

        self.task_sampler: BaseMujocoTaskSampler = task_sampler
        self.render_mode = render_mode
        self.render_camera = render_camera
        self._sample_task_kwargs = sample_task_kwargs

        # The episode currently in the scene; None until the first reset().
        self.task: BaseMujocoTask | None = None

    def _require_task(self) -> BaseMujocoTask:
        """The current task, or a clear error if ``reset()`` has not run yet."""
        if self.task is None:
            raise RuntimeError(f"Call {type(self).__name__}.reset() before step()/render().")
        return self.task

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Sample a new episode and return its first observation.

        Args:
            seed: Seeds episode sampling via ``seed_task_sampling``. NOTE that is
                process-global, see ``docs/gym_compatibility.md``.
            options: ``sample_task()`` arguments for this episode only
                (``house_index``, ``force_advance_scene``).
        """
        super().reset(seed=seed)
        if seed is not None:
            self.task_sampler.seed_task_sampling(seed)

        unknown = set(options or ()) - set(RESET_OPTIONS)
        if unknown:
            raise ValueError(
                f"Unsupported reset options {sorted(unknown)}; supported: {list(RESET_OPTIONS)}."
            )

        # Close the old task first: sampling can load a scene, which replaces the
        # sim env the old task is bound to.
        if self.task is not None:
            self.task.close()
            self.task = None

        task = self.task_sampler.sample_task(**{**self._sample_task_kwargs, **(options or {})})
        if task is None:
            raise RuntimeError(
                f"{type(self.task_sampler).__name__} returned no task (max_tasks reached); "
                f"call env.task_sampler.reset() to start over."
            )
        n_batch = task.env.n_batch
        if n_batch != 1:
            task.close()
            raise ValueError(
                f"The gymnasium interface is single-environment only, got "
                f"n_batch={n_batch}. Use the task sampler directly for batches."
            )

        self.task = task
        return task.reset()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], Any, Any, Any, list[dict[str, Any]]]:
        return self._require_task().step(action)

    def render(self) -> np.ndarray:
        """Render the current episode as an RGB array.

        Uses ``render_camera`` if set, else the first camera in the camera config.
        Unlike ``gym.Env.render`` this does not require ``render_mode`` to have
        been set -- returning a frame beats returning None for an unset mode.
        """
        if self.render_mode not in (None, "rgb_array"):
            raise ValueError(
                f"Unsupported render_mode {self.render_mode!r}; supported: 'rgb_array'."
            )
        return self._require_task().render(self.render_camera)

    def close(self) -> None:
        """Close the current task, and the sampler if this env built it."""
        if self.task is not None:
            self.task.close()
            self.task = None
        if self._owns_task_sampler:
            self.task_sampler.close()


def env_id_for_config(config_name: str, version: int = 0) -> str:
    """The gymnasium env id for a registered config name."""
    return f"{NAMESPACE}/{config_name}-v{version}"


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
            entry_point="molmo_spaces.tasks.gym_env:GymEnv",
            kwargs={"config_name": config_name},
            # Task horizon is enforced by the task itself (is_timed_out), and
            # letting gym also wrap it would truncate twice.
            max_episode_steps=None,
            # These envs cannot pass gym's checker (no spaces declared).
            disable_env_checker=True,
            # Without this gym.make returns an OrderEnforcing wrapper, and
            # gymnasium wrappers no longer proxy attribute access -- so this env's
            # own API (task, task_sampler) would be unreachable except through
            # .unwrapped. The order it enforces (reset before step) is already
            # enforced here, with a message naming the env.
            order_enforce=False,
        )
        registered.append(env_id)

    log.info(f"Registered {len(registered)} molmospaces envs with gymnasium")
    return registered
