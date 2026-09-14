# FetchMan reference stack

⚠️ **This is the FetchMan paper's implementation, and it is a leaf in this
repository.** Nothing under `molmo_spaces/` imports it. Its defining constraint
is that the rollout must reproduce `g1_molmo`'s own printed trace **byte for
byte**, so the sampling is kept exactly as gold wrote it; only the interfaces
around it were adapted to molmo_spaces' abstractions.

## Layout

```
fetchman/
├── scene_g1ms.py                Scene — MJCF compile, occupancy maps
├── env_g1ms.py                  G1CPUMujocoEnv, a CPUMujocoEnv subclass
├── configs/
│   ├── g1_port_configs.py       G1ExpConfig, G1TaskSamplerConfig
│   └── bowl_fetchman.py         the bowl config (objects="bowl")
├── tasks/
│   ├── task_g1ms.py             G1Task     — a molmo_spaces BaseMujocoTask
│   ├── pick_g1ms.py             PickTask
│   ├── open_g1ms.py             OpenTask
│   └── pick_task_sampler_g1ms.py  G1TaskSampler — a BaseMujocoTaskSampler
└── scripts/                     the gates and the collection harness
```

Two things live in molmo_spaces because both stacks use them:

| | |
|---|---|
| `molmo_spaces/env/arena/scene_spec_ops.py` | gold's pre-compile MjSpec edits, shared with the native scene build |
| `.../object_manipulation/g1_pick_policy.py` | `G1Controller` + the native `G1PickPlannerPolicy` adapter |

## Native abstractions, gold's sampling

`G1Task` is a `BaseMujocoTask` and `G1TaskSampler` a `BaseMujocoTaskSampler`, so
molmo_spaces' pipeline can drive them with no wrapper —
`cfg.task_sampler_config.task_sampler_class(cfg)` then
`sample_task(house_index=N)`, which indexes this sampler's own scene glob.

Neither runs its base `__init__`, and each says why in its docstring: the task
is built *before* the first `load_scene` (the load needs the task, for
`set_objects`) so it cannot read control rates off a model that does not exist
yet; the sampler resolves scenes from its own glob rather than a dataset house
index, and the base would seed global RNG that nothing here reads. Both set the
base attributes their inherited methods need.

What has **not** changed is the sampling. `G1TaskSampler` draws from
`np_random` **28,156 times per 5-episode rollout** across 13 call sites, and
retry loops make the per-episode counts data-dependent. The episode the robot
succeeds on, the object it targets and the pose it spawns at are all functions
of that exact order, so the reference algorithm is kept exactly as gold wrote
it — only signatures and interface members were added.

## The gates

Both live in [`scripts/`](scripts/) and both must stay green.

**Gold parity** — does this stack still behave like `g1_molmo`'s reference?

```bash
cd ~/code/g1_molmo && conda run -n g1_molmo python \
    molmospaces/scripts/g1_molmo_comparison/generate_gold_rollout.py --seed 0 > /tmp/gold.txt 2>&1
conda run -n mlspaces python fetchman/scripts/generate_ported_rollout.py --seed 0 > /tmp/ours.txt 2>&1
conda run -n mlspaces python fetchman/scripts/check_gold_parity.py /tmp/gold.txt /tmp/ours.txt --strict
```

Expect `PASS (strict): 299 trace lines byte-identical`. `--strict` is only
meaningful when both conda envs are on the same MuJoCo (currently 3.11.0 in
both); otherwise drop it and compare discrete invariants only. Run the
ported-vs-ported form of this around **every** refactor.

**Trajectory equivalence** — do the native modules and the `fetchman` shim
produce the same rollouts?

```bash
conda run -n mlspaces python fetchman/scripts/collect_trajectories.py --stack native --episodes 10 --out /tmp/n.json
conda run -n mlspaces python fetchman/scripts/collect_trajectories.py --stack port   --episodes 10 --out /tmp/p.json
conda run -n mlspaces python fetchman/scripts/collect_trajectories.py --compare /tmp/n.json /tmp/p.json
```

Expect `PASS: 10/10 trajectories identical`. Compares target, spawn pose, step
count, sim time, success/terminated/truncated and a SHA over final qpos/qvel.

## Running it

From the repo root, in the `mlspaces` conda env:

```bash
bash fetchman/scripts/collect_single.sh          # watch a rollout
conda run -n mlspaces python fetchman/scripts/nav_demo.py
```

The parity scripts additionally need a `g1_molmo` checkout with the
`procthor-10k-val` scenes downloaded. Assets resolve through
`molmo_spaces_constants.ASSETS_DIR` (override with `MLSPACES_ASSETS_DIR`);
grasps through `MOLMOSPACES_GRASPS_DIR`; scene textures come from the curated
pack at `$ASSETS_DIR/textures/fetchman/`, which is not fetched automatically.

Every entrypoint imports [`scripts/_bootstrap.py`](scripts/_bootstrap.py) first,
which puts *this* checkout's root at the front of `sys.path` — that also stops
an editable `molmo_spaces` install from a different clone shadowing the one
these scripts live beside. No `PYTHONPATH` needed.

## Tooling

- **Not shipped.** `[tool.setuptools.packages.find]` lists `include =
  ["molmo_spaces*"]`, so this directory is not in the wheel.
- **`scripts/` is not linted.** `[tool.ruff] exclude` lists `fetchman/scripts/`
  alongside the repo's top-level `scripts/`, so these stay close to their
  upstream g1_molmo originals.
