# Gym Compatibility

Tasks (`BaseMujocoTask`) subclass `gymnasium.Env`, and can be built through
`gymnasium.make`. The compatibility is **deliberately partial**: nothing about
the data generation path changed to accommodate it. Read this before pointing
third-party RL code at a molmospaces env.

## How a gym env is built

```python
# Data generation / evaluation, unchanged: the sampler builds the task.
task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
task = task_sampler.sample_task()

# Gymnasium: exactly the same two lines, wrapped so gymnasium can call them.
import molmo_spaces.tasks.gym_registration as gym_registration
gym_registration.register_configs()
env = gymnasium.make("MolmoSpaces/FrankaPickDroidDataGenConfig-v0")
```

`gymnasium.make` builds the config's task sampler, calls `sample_task()` **once**
and hands back the task, which is already a `gymnasium.Env`. There is only one
way to build a task -- a sampler builds it -- so tasks, samplers and the datagen
pipeline are untouched by this.

Registration passes `order_enforce=False`, so `gymnasium.make` returns the task
itself rather than wrapping it. Gymnasium wrappers stopped proxying attribute
access in 1.0, so a wrapper would put the task's own API (`register_policy`,
`env`, `get_task_description`) behind `.unwrapped` and make `render()` raise
before the first `reset()`. The order it enforces -- reset before step -- is
already implied by an env holding one episode. Wrap it yourself if you want it
back.

Env ids come from the data generation config registry, so any config registered
with `@register_config` is available as `MolmoSpaces/<ConfigName>-v0`.

`gym_registration.make_env` takes the arguments the env id cannot carry:
`exp_config` instead of `config_name`, `config_overrides`, an existing
`task_sampler` to sample from, a `seed`, a `render_camera`, and anything
`sample_task()` accepts (`house_index`, `force_advance_scene`).

## What does not hold

### One episode per env

An env is one sampled episode. `reset()` clears task state for that episode and
returns its first observation; a **second** `reset()` raises
`NotImplementedError`. A new episode means another `sample_task()`, or another
`gymnasium.make`.

The check is the `gym_single_episode` flag, which `make_env` sets and nothing
else does. A task you sampled yourself can still be reset repeatedly, as the
datagen path does (once before `register_policy`, once after) -- there it is
understood that reset replays the same episode, which is exactly the assumption
a gym caller does not make.

This is the main thing standard RL code will trip over -- training loops reset
every episode. It is deliberate: making `reset()` re-sample means the task has to
survive its scene being replaced and be re-configured in place, which would push
episode-advancing machinery into every task and every sampler. Loop over
`sample_task()` instead.

`reset(seed=...)` and `reset(options=...)` raise `NotImplementedError` for the
same reason: by the time the task exists, its episode is already chosen. Seed
`BaseMujocoTaskSampler.seed_task_sampling` (or pass `seed=` to `make_env`) before
sampling. Note that seeding is **process-global** -- it calls `random.seed`,
`np.random.seed` and `torch.manual_seed`, so it also reseeds the caller's RNG.

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
size themselves from the sampled episode's objects.

### Single environment only

`make_env` requires `n_batch == 1` and raises otherwise. Batched task state
exists (`step()` returns arrays over the batch), but termination and success are
index-0 only, so batching is not usable end-to-end. There is no
`gymnasium.vector.VectorEnv` implementation; use the task sampler directly for
batches.

### Construction is expensive

`gymnasium.make` builds a sampler, loads a scene and compiles a MuJoCo model, so
it takes seconds. Each env owns its own sampler and therefore its own sim env, so
N parallel envs each pay their own scene load. Pass an existing `task_sampler` to
`make_env` to reuse one -- but only sequentially: a sampler holds exactly one sim
env, so two live envs from one sampler would share and clobber a scene.

## What does hold

- `isinstance(task, gymnasium.Env)`, `task.unwrapped`, `task.np_random`.
- `reset()` returning `(observation, info)`, once.
- `step(action)` returning `(observation, reward, terminated, truncated, info)`.
- `render()` returning an RGB `uint8` array, from `render_camera` if set or the
  first configured camera otherwise. `metadata["render_modes"] == ["rgb_array"]`.
  Unlike `gym.Env.render`, it does not require `render_mode` to be set.
- `close()`, which closes the sampler only when `make_env` created it -- a
  caller-supplied sampler is never closed underneath the caller.
- `gymnasium.make` returning the task itself, so `isinstance(env, BaseMujocoTask)`
  holds and no `.unwrapped` hop is needed.

## Tests

`mlspaces_tests/data_generation/test_gym_env_interface.py` covers the gym path:
observation parity with a directly sampled task, the single-reset rule and its
absence on the sampler path, rejected `seed`/`options`, sampler ownership, the
batched rejection, sampler exhaustion, render, `gymnasium.make` returning an
unwrapped task, the registration flags, and the deliberate absence of both
spaces.
