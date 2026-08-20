# Gym Compatibility

`GymEnv` (`molmo_spaces.tasks.gym_env`) is a `gymnasium.Env` wrapper around a
molmospaces task sampler, and every data generation config is registered as a
gym env id. The compatibility is **deliberately partial**: nothing about the data
generation path changed to accommodate it. Read this before pointing third-party
RL code at a molmospaces env.

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

There is **no attribute proxying**. The task's own API is reached explicitly
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

## What does not hold

### No `action_space`

Actions are dicts keyed by move group (`{"arm": ..., "gripper": ...}`), and no
space describes them. Consequences:

- `gymnasium.utils.env_checker.check_env` fails.
- Most wrappers and RL libraries (Stable-Baselines3, CleanRL, RLlib) raise on
  construction, since they read `action_space` unconditionally. Registered envs
  therefore set `disable_env_checker=True`.

If you need one, write an adapter that declares a space for your robot and
translates arrays to the action dict; it is not provided because the useful
shape (flat `Box` vs nested `Dict`) depends on the consumer.

### No `observation_space`

Individual sensors carry `gymnasium` spaces (`sensor.observation_space`, and
`sensor_suite.observation_spaces`), but observations returned by `reset()` and
`step()` are `list[dict]` -- one dict per batch element -- so no single space
describes them. Rather than declare a space that
`observation_space.contains(obs)` would reject, none is declared.

The observation contract is also **not fixed for a config**: robot sensors are
added at construction, `register_policy()` adds policy sensors, and some sensors
size themselves from the sampled episode's objects. A new episode can therefore
change the observation keys, and each `reset()` builds a new task.

### `reset()` is expensive, and invalidates the previous task

Each `reset()` samples an episode, which may load a scene and compile a MuJoCo
model -- seconds, not milliseconds. It also closes the task it replaces: a task
is bound to the sampler's single sim env, which scene loading replaces, so a
handle kept from before a `reset()` is dead. Read what you need from
`env.task` before resetting again.

`reset(seed=...)` reseeds the sampler via `seed_task_sampling`. Note that
seeding is **process-global** -- it calls `random.seed`, `np.random.seed` and
`torch.manual_seed`, so it also reseeds the caller's RNG.

### Single environment only

`reset()` requires `n_batch == 1` and raises otherwise. Batched task state exists
(`step()` returns arrays over the batch), but termination and success are index-0
only, so batching is not usable end-to-end. There is no
`gymnasium.vector.VectorEnv` implementation; use the task sampler directly for
batches.

### One live episode per sampler

A sampler holds exactly one sim env, so two `GymEnv`s sharing a `task_sampler`
would clobber each other's scene. Each env builds its own sampler by default;
pass an existing one only when the envs are used sequentially. N parallel envs
therefore each pay their own scene load.

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

## Tests

`mlspaces_tests/data_generation/test_gym_env_interface.py` covers the gym path:
observation parity with a directly sampled task, resampling on reset, reset
options, reset-before-step, sampler ownership, the batched rejection, sampler
exhaustion, render, `gymnasium.make` returning an unwrapped env, the registration
flags, and the deliberate absence of both spaces.
