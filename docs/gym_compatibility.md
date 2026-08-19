# Gym Compatibility

Tasks (`BaseMujocoTask`) subclass `gymnasium.Env`, and can be built through
`gymnasium.make`. The compatibility is **deliberately partial**: the parts that
would need to change the data generation path were left alone. Read this before
pointing third-party RL code at a molmospaces env.

## Two ways to build a task

```python
# 1. Data generation / evaluation: the sampler builds the task (unchanged).
task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
task = task_sampler.sample_task()

# 2. Gymnasium: omit the env and the task builds itself, creating its own
#    sampler. NOTE this samples an episode, so it can load a scene.
task = exp_config.task_config.task_cls(exp_config=exp_config)

# 3. Gymnasium, via registration.
import molmo_spaces.tasks.gym_registration as gym_registration
gym_registration.register_configs()
env = gymnasium.make("MolmoSpaces/FrankaPickDroidDataGenConfig-v0")
```

Env ids come from the data generation config registry, so any config registered
with `@register_config` is available as `MolmoSpaces/<ConfigName>-v0`.

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
size themselves from the sampled episode's objects.

### Single environment only

The gymnasium path requires `n_batch == 1` and raises otherwise. Batched task
state exists (`step()` returns arrays over the batch), but termination and
success are index-0 only, so batching is not usable end-to-end. There is no
`gymnasium.vector.VectorEnv` implementation; use the task sampler directly for
batches.

### `reset(seed=...)` seeds process-global RNG

`seed` is forwarded to `BaseMujocoTaskSampler.seed_task_sampling`, which calls
`random.seed`, `np.random.seed` and `torch.manual_seed`. **It reseeds the whole
process**, including the RNG of the code that called `reset` -- so seeding an env
per episode will also reset your policy's exploration noise.

This is because the samplers draw from the global `random`/`np.random` modules
in ~70 places; localizing that would change every sampled episode and redefine
what `config.seed` reproduces. `reset()` without a seed touches no RNG.

### `reset()` semantics depend on how the task was built

| Built via | `reset()` does |
| --- | --- |
| `EpisodeSource.SELF` (no env passed; gym) | Samples a **new** episode, except the first call, which uses the episode built during construction. Pass `seed` or `options` to force a re-sample. |
| `EpisodeSource.SAMPLER` (env passed, via `sample_task()`; datagen) | Clears task state only; the scene and episode are untouched. A new episode means another `sample_task()`. |

`options` is forwarded to `BaseMujocoTaskSampler.prepare_episode`, so
`house_index` and `force_advance_scene`; anything else is that function's
`TypeError`.

`episode_source` is derived, not passed: an `env` argument means a sampler built
the episode, no `env` means the task samples its own.

Re-sampling is not unbounded. Samplers consume their per-house candidate pool, so
enough resets on one house eventually raise `HouseInvalidForTask` -- the same
signal data generation handles by advancing houses. Gym callers should be ready
to catch it, or pass `options={"force_advance_scene": True}`.

### Reset can be expensive

A reset that crosses a house boundary loads a scene and recompiles the MuJoCo
model, which takes seconds. Callers that assume a cheap reset will see spikes.
Note also that loading a scene **replaces** the underlying `CPUMujocoEnv`, so
never cache `task.env` across a reset.

### No cross-env scene reuse

Each gym env owns its own sampler and therefore its own sim env, so N parallel
envs each pay their own scene loads. Samplers cannot be shared between live envs:
a sampler holds exactly one sim env, and one env's reset would mutate the other's
scene.

## What does hold

- `isinstance(task, gymnasium.Env)`, `task.unwrapped`, `task.np_random`.
- `reset(*, seed=None, options=None)` returning `(observation, info)`.
- `step(action)` returning `(observation, reward, terminated, truncated, info)`.
- `render()` returning an RGB `uint8` array, from `render_camera` if set or the
  first configured camera otherwise. `metadata["render_modes"] == ["rgb_array"]`.
  Unlike `gym.Env.render`, it does not require `render_mode` to be set.
- `close()`, which closes the sampler only when the task created it -- a
  caller-supplied sampler is never closed underneath the caller.
- Both construction paths produce the **same observation keys**, which is covered
  by a test.

## Tests

`mlspaces_tests/data_generation/test_gym_env_interface.py` covers the gym path,
cross-path observation parity, reset semantics for both paths, the batched
rejection, sampler exhaustion (`EpisodesExhausted` rather than `None`), render,
registration, and the deliberate absence of both spaces.
