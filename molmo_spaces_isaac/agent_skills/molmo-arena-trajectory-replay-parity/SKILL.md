---
name: molmo-arena-trajectory-replay-parity
description: Replay a MolmoSpaces/MuJoCo HDF5 trajectory in Isaac Lab Arena and compare MuJoCo versus Arena external/wrist videos side by side. Use when validating whether a converted Arena scene follows the same open-loop robot trajectory as the original MuJoCo episode.
---

# Molmo Arena Trajectory Replay Parity

## Purpose

Use this check when policy success rates alone are ambiguous. It answers:

- Does the same saved MuJoCo joint-position trajectory move the Arena robot
  plausibly?
- Do the target object, gripper, support surface, and camera views look
  comparable?
- Does Arena diverge because of scene/physics/contact differences rather than
  policy inputs?

## Helper

Use:

`molmo_spaces_isaac/scripts/run_mujoco_arena_replay_parity.py`

It wraps `run_arena_benchmark_episode.py --policy_type h5_replay`, then writes:

- `manifest.json`
- `report.md`
- `arena_replay.log`
- `arena_replay_result.json`
- `arena_videos/arena_epXXXX_external_camera_rgb.mp4`
- `arena_videos/arena_epXXXX_wrist_camera_rgb.mp4`
- `mujoco_arena_replay_side_by_side.mp4`

## Example

```bash
cd /path/to/molmospaces
python3 molmo_spaces_isaac/scripts/run_mujoco_arena_replay_parity.py \
  --episode_idx 14 \
  --arena_spec_manifest /tmp/arena_episode_specs_real_ithor_pick_hard.json \
  --mujoco_h5 /path/to/successful_mujoco_eval/house_XX/trajectories_batch_1_of_1.h5 \
  --assets_root /path/to/usd_assets \
  --scenes_root /path/to/usd_assets/scenes
```

If MuJoCo camera videos are not in the same directory as the HDF5 file with
standard MolmoSpaces eval names, pass `--mujoco_external_video` and
`--mujoco_wrist_video`.

## Recommended Settings

- Use `--replay_action_repeat 3` for roughly 60 ms per MuJoCo policy action
  when the Arena env step is 0.02 s.
- Use `--record_video_stride 3` so Arena replay video records roughly one frame
  per MuJoCo action step at the default 15 fps.
- Use `--arena_root` when Isaac Lab Arena is not under the parent directory of
  the `molmospaces` checkout.
- Use `--replay_init_from_h5_qpos` only when intentionally replaying from a
  later trajectory segment.

## Interpreting Results

Treat this as an open-loop parity check, not a policy-performance metric. A
failed Arena replay can still be useful if the side-by-side video shows where
robot/object contact behavior first diverges.
