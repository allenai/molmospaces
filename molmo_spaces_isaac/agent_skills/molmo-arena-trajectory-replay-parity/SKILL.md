---
name: molmo-arena-trajectory-replay-parity
description: Replay a successful MolmoSpaces/MuJoCo HDF5 trajectory in Isaac Lab Arena and compare MuJoCo versus Arena external/wrist videos side by side. Use when validating whether a converted Arena scene follows the same open-loop robot trajectory as the original MuJoCo episode.
---

# Molmo Arena Trajectory Replay Parity

## Purpose

Use this check when policy success rates alone are ambiguous. It answers:

- Does the same saved MuJoCo joint-position trajectory move the Arena robot plausibly?
- Do the target object, gripper, support surface, and camera views look comparable?
- Does Arena diverge because of scene/physics/contact differences rather than policy inputs?

## Current Helper

Use:

`molmo_spaces_isaac/scripts/run_mujoco_arena_replay_parity.py`

It wraps `run_arena_benchmark_episode.py --policy_type h5_replay`, then creates:

- `manifest.json`
- `report.md`
- `arena_replay.log`
- `arena_replay_result.json`
- `arena_videos/arena_epXXXX_external_camera_rgb.mp4`
- `arena_videos/arena_epXXXX_wrist_camera_rgb.mp4`
- `mujoco_arena_replay_side_by_side.mp4`

## Default Example

This picks the first successful MuJoCo trajectory for episode idx 14 from the
configured MuJoCo summary and replays it in Arena:

```bash
cd /path/to/molmospaces
python3 molmo_spaces_isaac/scripts/run_mujoco_arena_replay_parity.py \
  --episode_idx 14 \
  --arena_spec_manifest /tmp/arena_episode_specs_real_ithor_pick_hard.json
```

Outputs go under:

`$MOLMO_PROJ_ROOT/diagnostics/trajectory_replay_parity/ep0014_traj_*/`

If `MOLMO_PROJ_ROOT` is unset, the helper treats the parent directory of the
`molmospaces` checkout as the workspace root.

## Recommended Settings

- Use the Lab 2.3 / Isaac Sim 5.1 Arena checkout via `--arena_root` when it is
  not under `$MOLMO_PROJ_ROOT/IsaacLab-Arena-working-lab2.3`.
- Use `--replay_action_repeat 3` for roughly 60 ms per MuJoCo policy action.
- Use `--record_video_stride 3` so the Arena replay video has one frame per
  MuJoCo action step at the default 15 fps.
- By default, the helper starts from the first non-dummy HDF5 action row. Use
  `--replay_init_from_h5_qpos` only when intentionally replaying from a later
  trajectory segment.

## Interpreting Results

Treat this as an open-loop parity check, not a policy-performance metric. A
failed Arena replay can still be valuable if the side-by-side video shows where
the robot/object contact behavior first diverges. If the replay fails from reset
but looks reasonable after `--replay_init_from_h5_qpos` at a later row, suspect
reset pose, timing, or early contact mismatch.
