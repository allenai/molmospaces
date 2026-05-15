---
name: molmo-arena-orient
description: Resume or plan MolmoSpaces-to-Isaac-Lab-Arena migration work. Use when Codex needs to understand the current benchmark target, progress docs, diagnostics, branches, blockers, or choose the next step for converting MolmoSpaces/MuJoCo tasks to Isaac Lab Arena.
---

# Molmo Arena Orient

## Purpose

Use this skill first when entering an existing MolmoSpaces to Isaac Lab Arena
migration thread. The goal is to rebuild the project state from repo artifacts,
not from chat memory.

## Workflow

1. Check the local repo and branch state:

```bash
git status --short --branch
git branch -vv
git remote -v
```

2. Read the trackers in this order:

- `molmo_spaces_isaac/docs/isaaclab_arena_migration_goals.md`
- `molmo_spaces_isaac/docs/isaaclab_arena_migration_subgoals.md`
- `molmo_spaces_isaac/docs/isaaclab_arena_migration_progress.md`

3. Identify the active migration target:

- benchmark family and episode range
- MuJoCo baseline success rate and settings
- current Arena status
- current blocker
- diagnostic artifacts already produced
- exact next runbook or report target

4. Inspect recent diagnostics only as needed. Prefer summaries and reports over
large raw traces:

```bash
find /home/horde/molmo-proj/diagnostics -maxdepth 4 -type f \
  | rg -i 'summary|report|montage|preflight|policy|trace|mujoco|arena'
```

5. End with a short state update:

- completed steps
- current blocker
- next concrete command or artifact to produce
- any uncertainty that must be resolved before running expensive simulations

## Current iTHOR Anchor

For the current PoC, the target is real iTHOR `FrankaPickHardBench`.
The local MuJoCo baseline is `14 / 69 = 20.29%` with `pi05_droid_jointpos`.
Arena conversion preflights all `69` episodes, but policy parity is still open.

The current sharp debugging target is original idx `14`, where MuJoCo is
`7/10` and Arena is `0/10` in the paired repeated-rollout slice.

## Update Rule

When the migration state changes, update the tracker docs first. Then update
this skill only if the repeatable orientation process itself changed.
