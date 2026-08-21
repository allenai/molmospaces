# Gym Compatibility

MolmoSpaces and Gymanisum are designed for different purposes and have different
abstractions. MolmoSpaces scenes are a bit larger and can have several tasks
associated with them. To make this more efficinent task construction is split into
a task sampler, which gives you a new task, and tasks, which you can call step on.
See [concepts.md](concepts.md) for detail.

As a usability feature we provide wrappers to our classes that provide gym-style
APIs. These are only partially implemented and are probably not suitable for
scaled use in datagen and traning.
`GymEnv` (`molmo_spaces.tasks.gym_env`) is a `gymnasium.Env` wrapper around a
molmospaces task sampler, and every data generation config is registered as a
gym env id.

## How a gym env is built

```python
# Data generation / evaluation, unchanged: the sampler builds the task.
task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
task = task_sampler.sample_task()

# Gymnasium: the env owns a sampler and makes that call in reset().
import molmo_spaces.tasks.gym_env as gym_env
gym_env.register_configs()
env = gymnasium.make("MolmoSpaces/FrankaPickDroidDataGenConfig-v0")

observation, info = env.reset()   # sample_task() + task.reset()
observation, info = env.reset()   # a new episode
```

The env owns a task sampler and holds the current episode's task on `env.task`.
`reset()` closes the previous task, calls `sample_task()` for the next episode
and returns `task.reset()`; `step()` and `render()` delegate to the current task.
Tasks, samplers and the datagen pipeline are untouched by this -- a task still
holds exactly one episode and is still built only by a sampler.

There is *no attribute proxying. The task's own API is reached explicitly
through `env.task` (`register_policy`, `get_task_description`, `env`, ...) and
the sampler through `env.task_sampler`.

Registration passes `order_enforce=False`, so `gymnasium.make` returns the
`GymEnv` itself rather than wrapping it -- gymnasium wrappers stopped proxying
attribute access in 1.0, so `OrderEnforcing` would put `task` and `task_sampler`
behind `.unwrapped`. `GymEnv` enforces reset-before-step itself, with a message
that names the env. Wrap it yourself if you want gym's version back.

Env ids come from the data generation config registry, so any config registered
with `@register_config` is available as `MolmoSpaces/<ConfigName>-v0`.

`GymEnv.__init__` takes the arguments the env id cannot carry: `exp_config`
instead of `config_name`, `config_overrides`, `render_mode`, `render_camera`, and
`sample_task()` defaults (`house_index`, `force_advance_scene`). Per-episode
`sample_task()` arguments go through `reset(options=...)`.

Instead of a config you can hand it an existing `task_sampler` to sample from --
that sampler carries its own config, so it takes no `config_name`, `exp_config`
or `config_overrides` alongside it, and `close()` leaves it open.

## What does hold

- `isinstance(env, gymnasium.Env)`, `env.unwrapped`, `env.np_random`.
- `reset()` returning `(observation, info)` for a freshly sampled episode, as
  many times as the sampler has episodes (it raises once `max_tasks` is
  exhausted; call `env.task_sampler.reset()` to start over).
- `reset(seed=...)` and `reset(options={"house_index": ..., ...})`.
- `step(action)` returning `(observation, reward, terminated, truncated, info)`.
- `render()` returning an RGB `uint8` array, from `render_camera` if set or the
  first configured camera otherwise. `metadata["render_modes"] == ["rgb_array"]`.
  Unlike `gym.Env.render`, it does not require `render_mode` to be set.
- `close()`, which closes the current task and the sampler -- but only when the
  env built the sampler; a caller-supplied one is never closed underneath the
  caller.
- `gymnasium.make` returning the `GymEnv` itself, so no `.unwrapped` hop is
  needed to reach `env.task`.


## What doesn't hold

- No `action_space`** — actions are dicts keyed by move group (`arm`,
  `gripper`, etc.), so `check_env` and most RL libraries (SB3, CleanRL, RLlib)
  choke on construction; registered envs set `disable_env_checker=True`. Build
  your own adapter if you need a `Box`/`Dict` space.
- No `observation_space`** — observations are `list[dict]` (one per batch
  element), and the key set isn't fixed per config: robot/policy sensors and
  episode-dependent sizing mean it can change between resets.
- `reset()` is expensive and kills the old task** — it samples an episode,
  possibly loading a scene and compiling a MuJoCo model (seconds). The
  previous task is closed in the process, so grab what you need from
  `env.task` before calling `reset()` again. `reset(seed=...)` also reseeds
  `random`, `np.random`, and `torch` globally.
- Single environment only — `reset()` requires `n_batch == 1`; batched
  state exists in `step()` but termination/success are index-0 only, so it's
  not usable end-to-end. No `VectorEnv`; use the task sampler directly for
  batching.
- *One live episode per sampler — a sampler owns one sim env, so sharing a
  `task_sampler` across `GymEnv`s clobbers scenes. Each env makes its own
  sampler by default (share one only for sequential use), so N parallel envs
  each pay their own scene-load cost.
