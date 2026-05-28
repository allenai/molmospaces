---
name: molmospaces-migration-validate
description: Build and interpret MolmoSpaces to IsaacLab Arena migration validation reports. Use when comparing converted Arena episode specs against MuJoCo/MolmoSpaces source episodes, creating dashboard-style per-episode and aggregate metrics, checking robot/object/camera/reset parity, or deciding whether policy eval differences are migration issues versus transfer-performance issues.
---

# MolmoSpaces Migration Validate

## Core Rule

Treat policy evaluation as an optional downstream overlay, not the migration score. Validate migration in layers:

1. Static/spec parity: episode identity, house/scene mapping, target object, target pose, robot base pose, initial robot qpos, camera metadata, and success threshold.
2. Reset-state parity: launch Arena diagnostics and compare actual reset object pose, robot joints, EEF pose, and camera poses against the converted spec.
3. Replay parity: replay MuJoCo HDF5 trajectories in Arena and compare trajectories/contact events; do not collapse this to success/fail alone.
4. Policy transfer: report OpenPI or other policy success rates separately and label them as policy transfer outcomes.

## Quick Start

From the MolmoSpaces repo root, generate the static/reset dashboard with:

```bash
python3 molmo_spaces_isaac/scripts/build_migration_validation_dashboard.py \
  --arena_spec_manifest /path/to/arena_episode_specs.json \
  --scenes_root /path/to/converted/usd/scenes \
  --task_description_map /path/to/task_description_from_mujoco_h5.json \
  --arena_results_json /path/to/optional_arena_policy_results.json \
  --arena_reset_summary_glob '/path/to/diagnostics/**/summary.json' \
  --out_dir /path/to/diagnostics/migration_validation_dashboard
```

Open `index.html` in the output directory. Also inspect `migration_validation_episodes.csv` for per-episode sorting/filtering and `migration_validation_summary.json` for programmatic aggregate metrics.

## Workflow

1. Locate inputs:
   - Arena spec manifest from `export_arena_episode_specs.py` or pre-existing diagnostics.
   - Optional task-description map derived from MuJoCo HDF5 metadata.
   - Optional converted USD scenes root, for example `/home/horde/.molmospaces/usd/scenes`, when the manifest was written with paths from another machine.
   - Optional Arena reset summaries from `diagnose_arena_episode.py`.
   - Optional Arena policy result JSON only as an overlay.
2. Run `build_migration_validation_dashboard.py`.
3. Interpret hard structural failures first: wrong house, wrong task type, missing robot init, target pose transform mismatch, or reset object/joint mismatch.
4. Treat warnings as coverage gaps or workspace issues: missing local USD path, missing reset summary, missing HDF5 replay file.
5. For representative or failing episodes, run HDF5 replay commands from the CSV and compare:
   - end-effector trajectory error,
   - gripper command/state timing,
   - target object trajectory,
   - first unsupported/lift instant,
   - robot-target and target-scene contact events.

## Metrics To Prefer

Use these before policy success:

- Target pose in robot frame and its recomputed transform error.
- Robot base pose and yaw distribution.
- Robot initial joint qpos presence and reset joint max error.
- Target distance from robot and target Z in robot frame.
- Scene/house ID consistency.
- Reset object position/quaternion error from Arena diagnostic summaries.
- Camera pose/image availability from reset summaries.
- MuJoCo HDF5 replay artifact availability.

Read `references/metrics.md` for metric meanings, thresholds, and caveats when writing summaries.

## Reporting Tone

Be explicit about what the dashboard proves. A clean static/reset report says the automated conversion is internally consistent and launches with the intended initial state. It does not prove policy transfer or physics equivalence. Policy eval belongs in a separate paragraph or column labeled "policy overlay" or "transfer outcome."
