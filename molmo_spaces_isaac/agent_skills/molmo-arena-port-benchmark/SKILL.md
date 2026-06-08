---
name: molmo-arena-port-benchmark
description: Convert a MolmoSpaces/MuJoCo benchmark family into Isaac Lab Arena episode specs and environments. Use when adding a new benchmark, extending ArenaEpisodeSpec conversion, resolving scene/object assets, or validating converted episodes before policy evaluation.
---

# Molmo Arena Port Benchmark

## Purpose

Use this workflow to port another MolmoSpaces benchmark family into Isaac Lab
Arena using the same low-touch pattern as the iTHOR pick migration.

## Inputs To Establish

- benchmark directory and JSON schema
- task class and success condition
- scene dataset and house/scene identifiers
- manipulated object identifiers
- robot embodiment, reset qpos, base pose, and cameras
- USD scene/object asset root layout
- policy observation/action contract, if policy evaluation is in scope

## Workflow

1. Inspect representative benchmark JSON episodes and map source fields into
   `ArenaEpisodeSpec`. Preserve source identifiers so results can be traced back
   to the original MolmoSpaces/MuJoCo benchmark.

2. Extend conversion/build logic only where needed:

- `molmo_spaces_isaac/src/molmo_spaces_isaac/arena/episode_to_arena.py`
- `molmo_spaces_isaac/src/molmo_spaces_isaac/arena/build_from_spec.py`

3. Resolve assets conservatively:

- use MolmoSpaces scene USDs as Arena backgrounds when available
- prefer scene-embedded object references for objects already present in the
  scene
- keep added-object spawning available for benchmarks that introduce objects
- avoid per-episode pose offsets unless asset or frame mismatch has been
  confirmed

4. Run offline conversion and preflight before launching Isaac Sim:

```bash
python3 molmo_spaces_isaac/scripts/export_arena_episode_specs.py \
  --benchmark_dir /path/to/benchmark_dir \
  --assets_root /path/to/usd_assets \
  --out /tmp/molmo_arena_specs.json

python3 molmo_spaces_isaac/scripts/preflight_arena_benchmark.py \
  --benchmark_dir /path/to/benchmark_dir \
  --assets_root /path/to/usd_assets \
  --json_out /tmp/molmo_arena_preflight.json
```

5. Add focused tests for offline conversion behavior. Avoid requiring Isaac Sim
   for tests that only need benchmark JSON/spec conversion.

## Acceptance Checks

- Representative episodes convert and preflight successfully.
- The full target benchmark reports actionable ready/failed counts.
- Scene-embedded and added-object cases are handled intentionally.
- Robot reset pose, task text, success threshold, and object identity match the
  source benchmark.
- Policy evaluation commands are documented in the customer handoff doc.

## Useful References

- `molmo_spaces_isaac/docs/isaaclab_arena_customer_handoff.md`
- `molmo_spaces_isaac/scripts/export_arena_episode_specs.py`
- `molmo_spaces_isaac/scripts/preflight_arena_benchmark.py`
- `molmo_spaces_isaac/scripts/run_arena_benchmark_batch.py`
