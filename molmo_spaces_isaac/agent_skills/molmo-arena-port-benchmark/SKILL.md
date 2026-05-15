---
name: molmo-arena-port-benchmark
description: Convert a MolmoSpaces/MuJoCo benchmark family into Isaac Lab Arena episode specs and environments. Use when adding a new benchmark, implementing ArenaEpisodeSpec conversion, scene USD/object resolution, asset handling, preflight scripts, or offline conversion tests.
---

# Molmo Arena Port Benchmark

## Purpose

Use this skill to port a new MolmoSpaces benchmark family into Isaac Lab Arena
with the same low-touch pattern used for the iTHOR pick PoC.

## Inputs To Establish

- benchmark directory and JSON schema
- task class and success condition
- scene dataset and house/scene identifiers
- pickup/manipulated object identifiers
- robot embodiment, reset qpos, base pose, and cameras
- asset root layout for USD scenes and object assets
- policy observation/action contract if policy evaluation is in scope

## Workflow

1. Inspect benchmark JSON examples and map fields into `ArenaEpisodeSpec`.
Keep original benchmark identifiers so artifacts can be traced back to MuJoCo.

2. Add or extend conversion in:

- `molmo_spaces_isaac/src/molmo_spaces_isaac/arena/episode_to_arena.py`
- `molmo_spaces_isaac/src/molmo_spaces_isaac/arena/build_from_spec.py`

3. Resolve assets conservatively:

- use the MolmoSpaces scene USD as the Arena background
- prefer existing scene prims for scene-embedded pickups
- keep added-object spawning available for benchmarks that introduce objects
- do not add per-episode Z offsets until scene-object support is ruled out

4. Build a preflight path before running policies:

- validate every episode can resolve scene assets
- validate pickup references resolve
- validate robot reset data and task text survive conversion
- write a JSON summary with ready/failed counts and failure reasons

Useful scripts from the current iTHOR work:

- `molmo_spaces_isaac/scripts/export_arena_episode_specs.py`
- `molmo_spaces_isaac/scripts/preflight_arena_benchmark.py`
- `molmo_spaces_isaac/scripts/diagnose_arena_episode.py`

5. Add focused tests for offline conversion behavior. Avoid requiring Isaac Sim
for tests that only need JSON/spec conversion.

## Acceptance Checks

- A representative episode builds in Arena.
- The full target benchmark preflights with actionable failure reasons.
- Scene-embedded and added-object cases are handled intentionally.
- Robot reset pose, task text, success threshold, and object identity match the
  source benchmark.
- The progress/subgoal trackers record what is supported and what is deferred.

## Watchouts From iTHOR

- Some pickup objects already exist inside scene USDs.
- Static scene geometry must stay stable while pickup rigid bodies remain dynamic.
- Imported mass/inertia values may be invalid or sentinel-like.
- Asset layouts vary; support common versioned iTHOR scene paths without hard
  coding one local machine layout.
