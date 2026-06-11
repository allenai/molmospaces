# JSON Evaluation Pipeline Lifecycle

This document describes the end-to-end lifecycle of a JSON benchmark evaluation
in `molmo_spaces`, what happens at each stage, and which knobs are available
to customize behavior. For a getting-started/setup guide, see
[`evaluation_guide.md`](./evaluation_guide.md). For schema details on a single
episode, see the "Sample Episode Spec" section there or
`molmo_spaces/evaluation/benchmark_schema.py`.

## High-level flow

```text
CLI args / programmatic call
        │
        ▼
run_evaluation()  (molmo_spaces/evaluation/eval_main.py)
  ├─ resolve eval config class (registry name or "module:Class")
  ├─ load benchmark episodes from JSON
  ├─ resolve task_horizon (CLI override > benchmark task_horizon_sec)
  ├─ create_eval_config(): instantiate eval config, force eval-mode flags
  │     (no action noise, no datagen profiler, seed=42, output_dir, ...)
  ├─ apply CLI overrides (camera_config, camera_names, light intensity)
  ├─ JsonEvalRunner.patch_config(): attach EvalRuntimeParams
  └─ JsonEvalRunner.adjust_robot(): wire robot_eval_override (if any)
        │
        ▼
JsonEvalRunner.__init__()
  ├─ load_all_episodes(benchmark_dir)
  ├─ truncate to max_episodes / single episode_idx
  ├─ derive task_sampler_config.house_inds + samples_per_house
  └─ super().__init__()  -> ParallelRolloutRunner sets up workers
        │
        ▼
JsonEvalRunner.run() -> ParallelRolloutRunner.run()
  └─ for each (house_id, batch) work item, dispatch to workers
        │
        ▼
ParallelRolloutRunner.process_single_house()  (per worker)
  ├─ load_episodes_for_house()       # JsonEvalRunner override
  ├─ for each EpisodeSpec:
  │     prepare_episode_config()
  │     get_episode_task_sampler()    -> JsonEvalTaskSampler(exp_config, ep)
  │     sample_task_from_spec()       -> task = sampler.sample_task(...)
  │     run_single_rollout()          -> step env at policy_dt_ms
  │     should_close_episode_task_sampler()  (True for JSON eval)
  └─ write trajectories.h5 + per-episode artifacts under output_dir
        │
        ▼
collect_episode_results() + (optional) wandb logging
        │
        ▼
EvaluationResults (success_count, total_count, output_dir, episode_results, exp_config)
```

## Stage-by-stage detail

### 1. Entrypoints

There are two equivalent entrypoints, both implemented in
`molmo_spaces/evaluation/eval_main.py`:

- CLI: `python molmo_spaces/evaluation/eval_main.py <eval_config_cls> --benchmark_dir ...`
- Programmatic: `run_evaluation(eval_config_cls=..., benchmark_dir=..., ...)`

`eval_config_cls` may be:

- a Python class object (subclass of `MlSpacesExpConfig`),
- a registry name (looked up via `get_config_class`), or
- a fully qualified `"module.path:ClassName"` string.

A small data-version sanity check (`_assert_data_versions_match`) runs at import
time to ensure the assets pinned in `_EXPECTED_DATA_VERSIONS` match
`DATA_TYPE_TO_SOURCE_TO_VERSION`. Mismatches raise immediately so benchmarks are
not silently evaluated against the wrong assets.

### 2. Loading the benchmark

`load_all_episodes(benchmark_dir)` reads either a single `benchmark.json` or
the legacy `house_*/episode_*.json` layout and returns a flat list of
`EpisodeSpec` objects (defined in
`molmo_spaces/evaluation/benchmark_schema.py`). Each `EpisodeSpec` is
**fully self-contained**:

- `source`: provenance (h5 file, traj key, dates)
- `house_index`, `scene_dataset`, `data_split`, `seed`
- `robot`: `robot_name` and `init_qpos` per move group
- `img_resolution` and `cameras` (robot-mounted or exocentric specs)
- `scene_modifications`: `added_objects`, `removed_objects`, `object_poses`
- `task`: `task_cls`, `robot_base_pose`, and task-specific fields
- `task_relevant_objects`: bodies used for camera visibility checks
- `language`: `task_description`, referral expressions, and priorities

Notably, **timing parameters (`policy_dt_ms`, `ctrl_dt_ms`, `sim_dt_ms`) and
`task_horizon` are NOT stored on the episode** — they come from the eval
config or CLI so the same benchmark can be replayed at different control rates.

### 3. Resolving `task_horizon`

`determine_task_horizon()` picks the horizon (in policy steps) using this
priority:

1. Explicit CLI override (`--task_horizon_steps` or `--task_horizon_sec`,
   mutually exclusive). With `--task_horizon_sec` the value is converted to
   steps using `eval_config.policy_dt_ms`.
2. Per-episode `task.task_horizon_sec` from the benchmark JSON, converted
   using `policy_dt_ms`. All episodes must agree on the value, otherwise
   evaluation fails loudly.

If neither is available, evaluation aborts rather than silently picking a
default.

### 4. Building the experiment config

`create_eval_config()` instantiates the user's eval config class
(typically a subclass of `JsonBenchmarkEvalConfig`) and applies eval-mode
defaults:

- `policy_config.checkpoint_path` overridden if `--checkpoint_path` was given.
- `output_dir` set to `eval_output/<config>/<timestamp>` (or
  `<output_dir>/<config>/<timestamp>` if `--output_dir` is provided).
- `num_workers` from the CLI / argument.
- `robot_config.action_noise_config = ActionNoiseConfig(enabled=False)` —
  evaluation is always deterministic w.r.t. the policy's actions.
- `datagen_profiler = False`, `profile = False` for clean output.
- `filter_for_successful_trajectories = False` — every rollout is saved.
- `seed = 42`.
- `task_horizon` set to the resolved value from the previous step.
- `camera_config` replaced with the override if provided (only honored for
  `FrankaEvalCameraSystem`).
- `eval_runtime_params` populated with an `EvalRuntimeParams` object.

`run_evaluation` then applies a few late overrides:

- `environment_light_intensity` (filament renderer only) from CLI.
- `policy_config.camera_names` from `--camera_names`.

### 5. Patching runtime parameters

`JsonEvalRunner.patch_config(...)` stores an `EvalRuntimeParams` dataclass on
the experiment config:

- `episode_idx`: only evaluate this single episode (`--idx`).
- `max_episodes`: cap the number of episodes pulled from the benchmark.
- `add_custom_object`, `custom_object_path`, `custom_object_name`: replace the
  target object of `pick`/`pick_and_place`-style episodes with a user-provided
  XML asset (`--add_custom_object --custom_object_path ...`).

These parameters are read by both the runner (when shaping work items) and the
worker (when loading episodes for its house).

### 6. Robot eval override (per-robot evaluation hooks)

Some robots need small environment tweaks at eval time that are not part of
the benchmark JSON (e.g. different intrinsics on the wrist camera, depth
recording, a slightly different robot base pose). These are implemented as
**robot eval overrides** in `molmo_spaces/evaluation/robot_eval_overrides.py`.

```python
def cap_robot_eval_override(
    episode_spec: EpisodeSpec,
    exp_config: MlSpacesExpConfig,
) -> None:
    camera_config = exp_config.camera_config

    camera_config.cameras[0] = MjcfCameraConfig(
        name="wrist_camera",
        mjcf_name="wrist_camera",
        robot_namespace="robot_0/gripper/",
        fov=53.0,
        fov_noise_degrees=(0.0, 0.0),
        pos_noise_range=(0.0, 0.0),
        orientation_noise_degrees=0.0,
        record_depth=True,
    )

    camera_config.cameras[1].record_depth = True
    camera_config.cameras[1].fov = 71

    rot_base = R.from_quat(episode_spec.task["robot_base_pose"][3:7], scalar_first=True).as_matrix()
    episode_spec.task["robot_base_pose"][:3] += 0.05 * rot_base[0:3, 0]
    episode_spec.task["robot_base_pose"][2] -= 0.2

    camera_config.img_resolution = (960, 720)

    episode_spec.robot.init_qpos = {
        "base": [],
        "arm": [[0, -1.5, 0.116, -2.45, 0, 0.842, 0.965]],
        "gripper": [0.00296, 0.00296],
    }


ROBOT_OVERRIDE_REGISTRY: dict[type[BaseRobotConfig], OverrideFn] = {
    FrankaCAPRobotConfig: cap_robot_eval_override,
}
```

How it gets wired in:

1. `JsonEvalRunner.adjust_robot(exp_config)` looks up an override for
   `exp_config.robot_config`'s class in `ROBOT_OVERRIDE_REGISTRY`. The lookup
   walks the class MRO, so subclasses inherit their parent's override.
2. The resolved function (or `None`) is stored on the config as
   `exp_config.eval_runtime_params.robot_override_fn`.
3. Inside `JsonEvalTaskSampler.__init__`, after the recorded camera config and
   task type have been wired up but before `super().__init__`, the override is
   invoked: `robot_override_fn(episode_spec, exp_config)`. Because it receives
   the live `episode_spec` and the full `MlSpacesExpConfig`, it can mutate the
   camera config, robot config, or any other part of the experiment config for
   that episode.

To add an override for a new robot class:

1. Implement an `OverrideFn` (`(EpisodeSpec, MlSpacesExpConfig) -> None`).
2. Call `register_robot_override(MyRobotConfig, my_override_fn)` from an importing
   module with the robot config class and an override function.
3. The override is only applied during JSON evaluation (it's wired through
   `JsonEvalRunner.adjust_robot`), so datagen behavior is unaffected.

> [!WARNING]
> Use robot eval overrides sparingly: the JSON benchmark is meant to be
> authoritative, and overrides defeat that contract.



> [!TIP]
> If all you need is depth recording on every camera, prefer
> `policy_config.force_enable_depth = True` over a robot eval override. When
> set, `JsonEvalTaskSampler` flips `record_depth = True` on every camera in
> `exp_config.camera_config` before the sampler is initialized (see
> `molmo_spaces/tasks/json_eval_task_sampler.py`), so you don't need to mutate
> the camera config from a robot-specific hook.

### 7. Runner construction (`JsonEvalRunner.__init__`)

The runner re-loads the benchmark, then:

- Truncates to the first `max_episodes` if requested.
- Groups episodes by `house_index` (so each house can be processed
  independently across workers).
- If `episode_idx` is set, narrows `task_sampler_config.house_inds` to the
  single house that owns that episode and sets `samples_per_house = 1`.
- Otherwise sets `house_inds` to all houses present in the benchmark and
  `samples_per_house` to the maximum number of episodes any single house has.
- Stores `benchmark_path` on the config so worker processes can re-load
  episodes inside their own process.

Then `super().__init__(exp_config)` runs `ParallelRolloutRunner.__init__`,
which builds the work-item list `(house_id, batch_samples, batch_num,
total_batches)` and prepares multiprocessing primitives.

### 8. Per-episode execution

Inside each worker, `ParallelRolloutRunner.process_single_house()` iterates
the episodes for a single house. The `JsonEvalRunner` overrides hook out the
JSON-specific behavior:

- `load_episodes_for_house()` re-reads the benchmark from
  `exp_config.benchmark_path` and applies `EvalRuntimeParams` filters
  (`max_episodes`, `episode_idx`, custom-object replacement via
  `replace_target_object_with_custom`).
- `get_max_episode_attempts()` returns `len(episode_specs)` — every benchmark
  episode is run exactly once (no retries, unlike datagen).
- `should_stop_early()` returns `True` once a single-episode run has
  collected its one rollout.
- `prepare_episode_config()` deep-copies `exp_config` and sets
  `scene_dataset` / `data_split` from the spec. `task_horizon` is kept from
  `exp_config`.
- `get_episode_task_sampler()` constructs a fresh `JsonEvalTaskSampler` per
  episode (mixed task types within one benchmark are supported).
- `sample_task_from_spec()` calls `sampler.sample_task(house_index=house_id)`,
  which sets robot base pose, applies the per-task config, applies any
  `_randomize_colors`, calls `setup_cameras`, and instantiates the right
  task class (`task.task_cls` is imported dynamically).
- `get_episode_seed()` uses `episode_spec.seed` if set, else the index.
- `should_close_episode_task_sampler()` returns `True` so the sampler is torn
  down each episode (avoids leaking state between mixed-task episodes).

`JsonEvalTaskSampler.randomize_scene` does **not** randomize — it deterministically
sets:

- object poses from `scene_modifications.object_poses`,
- robot per-move-group joint positions from `robot.init_qpos`,
- pickup-object joint state when applicable,
- per-object colors from `task.object_colors` (e.g. for
  `PickAndPlaceColorTask`).

Cameras come either from the JSON-recorded specs (default) or from a
`FrankaEvalCameraSystem` that performs deterministic, per-episode
visibility-aware spherical perturbation when `--use_eval_cameras` is passed.

### 9. Result collection

After `runner.run(...)` returns `(success_count, total_count)`,
`collect_episode_results(output_dir)` walks the output tree and produces a
list of `EpisodeResult` records (`house_id`, `episode_idx`, `success`, paths
to per-episode artifacts, etc.). These are returned in the
`EvaluationResults` dataclass, alongside `success_rate`,  `output_dir`, and
the resolved `exp_config`.

If `use_wandb=True`:

- A run is initialized with run name `"<ckpt_name>_<timestamp>"` and config
  including the resolved checkpoint path, benchmark dir, horizon (steps and
  seconds), config class, and episode/house counts.
- Composed videos per episode are built via `compose_episode_videos` (when
  `policy_config.camera_names` is non-empty) and uploaded with
  `log_eval_results_to_wandb`.

After everything finishes, the user typically runs
`scripts/benchmarks/eval_to_csv.py <output_dir> ...` to aggregate per-episode
results into a CSV (see `evaluation_guide.md`).

## What users can configure

The table below summarizes each user-facing knob, where it lives, and what it
affects. For full details on a flag, see `get_args()` in
`eval_main.py` or the docstring on `run_evaluation()`.

### Required selection

| CLI flag | `run_evaluation` arg | Purpose |
| --- | --- | --- |
| positional `exp_config_cls` | `eval_config_cls` | The eval config class (object, registry name, or `"module:Class"`). Provides `policy_config`, `robot_config`, and timing. |
| `--benchmark_dir` | `benchmark_dir` | Path to the benchmark (`benchmark.json` or legacy directory layout). |

### Policy / checkpoint

| CLI flag | `run_evaluation` arg | Effect |
| --- | --- | --- |
| `--checkpoint_path` | `checkpoint_path` | Override `policy_config.checkpoint_path`. |
| (none) | `preloaded_policy` | Pass an already-instantiated `BasePolicy`. Single-worker only — multi-worker requires creating connections inside each worker. |
| `--camera_names` | `camera_names_override` | Replace `policy_config.camera_names` (e.g. `--camera_names randomized_zed2_analogue_1 wrist_camera`). |

### Episode horizon and selection

| CLI flag | `run_evaluation` arg | Effect |
| --- | --- | --- |
| `--task_horizon_steps` | `task_horizon_steps` | Hard-cap policy steps per episode. Mutually exclusive with `--task_horizon_sec`. |
| `--task_horizon_sec` | `task_horizon_sec` | Same, but expressed in seconds; converted via `policy_dt_ms`. |
| `--max_episodes` | `max_episodes` | Evaluate only the first N episodes (filtered before house grouping). |
| `--idx` | `episode_idx` | Run a single episode by its index in the benchmark. |

### Cameras

| CLI flag | `run_evaluation` arg | Effect |
| --- | --- | --- |
| `--use_eval_cameras` | (via `camera_config_override`) | Enable `FrankaEvalCameraSystem` — replaces JSON-recorded cameras with a deterministic, visibility-checked spherical perturbation. |
| `--camera_rand_level` | (via `camera_config_override`) | 0–100 randomization scale for the eval camera system. Only relevant with `--use_eval_cameras`. |
| `--use-filament` | (build separately) | Switch to the filament renderer (requires the custom wheel). |
| `--environment-light-intensity` | `environment_light_intensity` | Override default ambient intensity (filament only). |

### Custom objects (asset-swap experiments)

| CLI flag | `run_evaluation` arg | Effect |
| --- | --- | --- |
| `--add_custom_object` | `add_custom_object` | Replace the target object in `pick` / `pick_and_place`-type episodes with a custom XML asset. |
| `--custom_object_path` | `custom_object_path` | Path to the replacement object XML (required when `add_custom_object` is set). |
| `--custom_object_name` | `custom_object_name` | Natural-language name used in instructions (defaults to the file stem if omitted). |

These are typically combined with `--idx` to do focused per-episode swaps.

### Output and logging

| CLI flag | `run_evaluation` arg | Effect |
| --- | --- | --- |
| `--output_dir` | `output_dir` | Root output directory. Final path is `<output_dir>/<config>/<timestamp>`. |
| `--num_workers` | `num_workers` | Multiprocessing workers (one process per worker). |
| `--no_wandb` | `use_wandb=False` | Disable wandb logging. |
| `--wandb_project` | `wandb_project` | wandb project name (default: `mlspaces-json-eval`). |

### Eval config (your own subclass of `JsonBenchmarkEvalConfig`)

Your eval config controls everything that is not part of the per-episode JSON:

- `policy_config`: model class/factory, checkpoint, `camera_names`,
  `action_move_group_names`, `action_spec`. Set
  `policy_config.force_enable_depth = True` to require depth from every camera
  — at JSON eval time `JsonEvalTaskSampler` will set `record_depth = True` on
  all cameras automatically, which is the lightweight alternative to a robot
  eval override when depth is the only thing you need to flip.
- `robot_config`: which `*RobotConfig` subclass — also drives whether a
  `robot_eval_override` from the registry is applied.
- `policy_dt_ms`, `ctrl_dt_ms`, `sim_dt_ms`: timing rates. The same benchmark
  can be replayed at different rates by changing these.
- `end_on_success`: stop the rollout as soon as the task is judged successful
  (e.g. `PiPolicyEvalConfig`).
- Anything else inherited from `MlSpacesExpConfig` (renderer, profiling
  knobs, etc.).

See `molmo_spaces/evaluation/configs/evaluation_configs.py` for examples
(`PiPolicyEvalConfig`, `CAPPolicyEvalConfig`, `TeleopPolicyEvalConfig`).

### Robot eval overrides (advanced)

If you need to tweak environment setup specifically at evaluation time for a
particular robot, register an `OverrideFn` in
`molmo_spaces/evaluation/robot_eval_overrides.py`:

```python
def my_robot_eval_override(episode_spec, exp_config):
    exp_config.camera_config.cameras[0].record_depth = True
    episode_spec.task["robot_base_pose"][2] -= 0.05

ROBOT_OVERRIDE_REGISTRY = {
    MyRobotConfig: my_robot_eval_override,
}
# or, from another module:
# register_robot_override(MyRobotConfig, my_robot_eval_override)
```

The override is selected by the **robot config class** (with MRO traversal so
subclasses inherit), applied per episode by `JsonEvalTaskSampler`, and only
active during JSON evaluation.

## Where to look in code

- `molmo_spaces/evaluation/eval_main.py` — entrypoints, config building,
  result collection.
- `molmo_spaces/evaluation/json_eval_runner.py` — `JsonEvalRunner` and the
  hook overrides on top of `ParallelRolloutRunner`.
- `molmo_spaces/evaluation/robot_eval_overrides.py` — robot-specific eval
  overrides registry.
- `molmo_spaces/tasks/json_eval_task_sampler.py` — per-episode
  `JsonEvalTaskSampler` (scene modifications, cameras, task class loading).
- `molmo_spaces/evaluation/benchmark_schema.py` — `EpisodeSpec` and all
  related Pydantic schemas.
- `molmo_spaces/data_generation/pipeline.py` — base
  `ParallelRolloutRunner.process_single_house` loop the JSON runner plugs
  into.
- `molmo_spaces/utils/eval_camera_randomization_utils.py` — `--use_eval_cameras`
  / `--camera_rand_level` CLI handling and the spherical perturbation logic.
- `molmo_spaces/evaluation/configs/evaluation_configs.py` — example eval
  configs (`JsonBenchmarkEvalConfig`, `PiPolicyEvalConfig`, etc.).
