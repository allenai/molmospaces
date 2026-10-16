# MolmoSpaces CommonSense

The CommonSense benchmark's tasks, task samplers, configs and analysis scripts.
Like [`projects/fetchman`](../fetchman/README.md), **this is a leaf**: nothing
under `molmo_spaces/` imports it. It is imported as the top-level package
`commonsense`, with `projects/` on `sys.path`.

## Layout

```
projects/commonsense/
├── configs/
│   ├── commonsense_configs.py              datagen configs (registered by name)
│   ├── commonsense_task_configs.py         task configs
│   └── commonsense_task_sampler_configs.py task sampler configs
├── tasks/
│   ├── __init__.py                         registers the tasks with JSON benchmark eval
│   ├── block_support_task.py               + block_support_task_sampler.py
│   ├── mug_ball_pick_task.py               + mug_ball_pick_task_sampler.py
│   └── semantic_grasp_pick_task.py         + semantic_grasp_pick_task_sampler.py
├── object_permanence_pointing.py           prompts/schema shared by the pointing eval
└── scripts/                                entrypoints, VLM rollout evals, paper plots
```

The general-purpose pieces these tasks rely on stay in `molmo_spaces/`, because
other tasks can use them too: the packing and block-stacking planner policies,
`utils/primitive_object_utils.py`, `configs/semantic_pick_prompts.py` (used by
`PromptSamplerLearnedPolicy`'s `prompt_level`), the `BlockSupportTaskSpec` /
`MugBallPickTaskSpec` benchmark schema entries, and the env/object-manager and
eval-runner additions.

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
- **Datagen.** The three datagen configs are `@register_config`'d, so the core
  `run_pipeline.py` picks them up by name once they are imported.

## Running it

From the repo root:

```bash
# datagen -- same flags as scripts/datagen/run_pipeline.py
python projects/commonsense/scripts/run_pipeline.py --config BlockSupportConfig
python projects/commonsense/scripts/run_pipeline.py --config MugBallPickConfig
python projects/commonsense/scripts/run_pipeline.py --config SemanticGraspPickConfig

# eval -- same arguments as molmo_spaces.evaluation.eval_main
python projects/commonsense/scripts/eval.py \
    molmo_spaces.configs.policy_configs_baselines:<EvalConfig> \
    --benchmark_dir <molmospaces-commonsense-bench>/<task>
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
