# MolmoSpaces CommonSense

The CommonSense benchmark's tasks, task samplers, configs and analysis scripts.
Like [`projects/fetchman`](../fetchman/README.md), **this is a leaf**: nothing
under `molmo_spaces/` imports it. It is imported as the top-level package
`commonsense`, with `projects/` on `sys.path`.

## Layout

```
projects/commonsense/
├── configs/
│   ├── commonsense_configs.py              datagen configs (registered by name),
│   │                                       CommonSenseFrankaRobotConfig
│   ├── commonsense_eval_configs.py         PackingPlannerEvalConfig (registered)
│   ├── commonsense_policy_configs.py       BlockStacking / PackingPlanner policy configs
│   ├── commonsense_task_configs.py         task configs
│   ├── commonsense_task_sampler_configs.py task sampler configs
│   └── semantic_pick_prompts.py            semantic_grasp_pick prompt_level ablation
├── policies/
│   ├── block_stacking_planner_policy.py
│   └── packing_planner_policy.py           packs every clutter object into the box
├── tasks/
│   ├── __init__.py                         registers tasks, specs and prompt builders
│   ├── specs.py                            BlockSupportTaskSpec, MugBallPickTaskSpec
│   ├── block_support_task.py               + block_support_task_sampler.py
│   ├── mug_ball_pick_task.py               + mug_ball_pick_task_sampler.py
│   ├── semantic_grasp_pick_task.py         + semantic_grasp_pick_task_sampler.py
│   └── packing_task_sampler.py             CommonSense packing datagen (core PackingTask)
├── object_permanence_pointing.py           prompts/schema shared by the pointing eval
└── scripts/                                entrypoints, VLM rollout evals, paper plots
```

The general-purpose pieces these tasks rely on stay in `molmo_spaces/`, because
core code or other tasks use them too:
- `utils/primitive_object_utils.py` (generic primitive replay in JSON eval);
- `PackingTask`'s multi-object success (the published `bench_pack_*` JSONs name the core task);
- the clutter options on `PickTaskSampler`;
- the semantic-grasp filter in the pick planner;
- `prompt_level` on the learned policies;
- the env/object-manager and eval-runner additions.

The covering/occlusion/containment pick benchmarks use the core `PickTask`, with
clutter stored in each episode, so they need nothing from this package at eval time.

## How molmo_spaces finds it

- **JSON benchmark eval.** `JsonEvalTaskSampler` imports the module named by each
  episode's `task_cls` before it looks up the task's config/spec class and
  `task_type`. Importing `commonsense.tasks` calls
  `register_json_eval_task(...)` for all three tasks, so no core mapping names
  them.
- **Old benchmark files.** Benchmarks generated before the move store
  `task_cls: "molmo_spaces.tasks.commonsense_tasks.<module>.<Class>"`.
  `EpisodeSpec.get_task_cls()` rewrites that prefix to `commonsense.tasks.`,
  so the published `molmospaces-commonsense-bench` keeps loading unchanged.
- **Prompts.** `commonsense.tasks` also registers the `semantic_grasp_pick` builder in
  `PromptSamplerLearnedPolicy.TASK_PROMPT_BUILDERS`, which `prompt_level` selects from.
- **Datagen / eval configs.** These are `@register_config`'d, so core `run_pipeline.py` and
  `eval_main` pick them up by name once the wrappers below have imported them.

## Running it

From the repo root:

```bash
# datagen -- same flags as scripts/datagen/run_pipeline.py
python projects/commonsense/scripts/run_pipeline.py --config BlockSupportConfig
python projects/commonsense/scripts/run_pipeline.py --config MugBallPickConfig
python projects/commonsense/scripts/run_pipeline.py --config SemanticGraspPickConfig
python projects/commonsense/scripts/run_pipeline.py --config FrankaPackingDroidDataGenConfig --clutter

# eval -- same arguments as molmo_spaces.evaluation.eval_main
python projects/commonsense/scripts/eval.py \
    molmo_spaces.configs.policy_configs_baselines:<EvalConfig> \
    --benchmark_dir <molmospaces-commonsense-bench>/<task>
python projects/commonsense/scripts/eval.py PackingPlannerEvalConfig \
    --benchmark_dir <molmospaces-commonsense-bench>/procthor-10k/bench_pack_covering/<run>
```

Every entrypoint imports [`scripts/_bootstrap.py`](scripts/_bootstrap.py) first,
which puts this checkout's root and `projects/` at the front of `sys.path`. No
`PYTHONPATH` is needed. To use `commonsense` from your own code, put `projects/`
on the path first.

## Tooling

- **Not shipped.** `[tool.setuptools.packages.find]` includes only
  `molmo_spaces*`, so this directory is not in the wheel.
- **`scripts/` is not linted.** `[tool.ruff] exclude` lists
  `projects/commonsense/scripts/`, alongside the repo's top-level `scripts/`.
