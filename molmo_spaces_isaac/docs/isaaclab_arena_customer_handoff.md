# MolmoSpaces to Isaac Lab Arena Handoff

## Current status

The current migration target is the real iTHOR `FrankaPickHardBench` pick
benchmark. The migration supports MolmoSpaces pick episodes in Isaac Lab Arena
through an offline conversion layer:

```text
MolmoSpaces benchmark JSON
  -> ArenaEpisodeSpec manifest
  -> Isaac Lab Arena environment
  -> policy eval or HDF5 trajectory replay
```

The latest local checkpoint converted and preflighted all 69 iTHOR pick
episodes:

- Arena spec export: `69 ready / 0 failed`
- Arena preflight: `69 ready`
- MuJoCo baseline: `14 / 69 = 20.3%`
- Arena online OpenPI run: `18 / 69 = 26.1%`

The Arena policy number should be treated as a policy-transfer/debugging result,
not as final simulator parity. The conversion, asset resolution, reset state,
camera plumbing, success check, post-success recording, and replay tooling are
now in place for customer review.

## What changed

- Convert MolmoSpaces pick benchmark episodes into `ArenaEpisodeSpec`.
- Resolve iTHOR scene USDs and THOR object USDs from `ms-download` asset roots.
- Build Arena environments from converted specs, including robot base pose,
  initial qpos, DROID camera setup, and scene pickup materialization.
- Match MolmoSpaces pick success more closely: the target object must lift above
  its start height and must not still be supported by non-robot scene geometry.
- Add post-success video recording with `--post_success_record_steps`.
- Add HDF5 replay support and a helper for MuJoCo-vs-Arena side-by-side replay
  parity videos.

## Assets and converted artifacts

Do not commit generated converted episode manifests or videos to the repo. The
manifest contains machine-local absolute paths such as the USD asset root and
benchmark directory, and the videos are large diagnostic artifacts.

Install the preconverted USD assets/scenes with:

```bash
ms-download --type usd --install-dir /home/$USER/.molmospaces/usd \
  --assets thor --scenes ithor
```

## Convert episodes

Regenerate the Arena spec manifest after installing assets:

```bash
python3 molmo_spaces_isaac/scripts/export_arena_episode_specs.py \
  --benchmark_dir /path/to/FrankaPickHardBench_20260206_json_benchmark \
  --assets_root /home/$USER/.molmospaces/usd \
  --out /tmp/arena_episode_specs_real_ithor_pick_hard.json
```

Expected for the current iTHOR target: `69 ready / 0 failed`.

## Preflight

Run the offline preflight before launching Isaac Sim:

```bash
python3 molmo_spaces_isaac/scripts/preflight_arena_benchmark.py \
  --benchmark_dir /path/to/FrankaPickHardBench_20260206_json_benchmark \
  --assets_root /home/$USER/.molmospaces/usd \
  --json_out /tmp/preflight_real_ithor_pick_hard.json
```

Expected for the current iTHOR target: `69 ready`.

## Arena zero-agent smoke eval

Use Isaac Lab Arena with Isaac Lab 2.3 / Isaac Sim 5.1 for the current working
stack. A zero-agent run is useful as a quick launch/assets/camera smoke test;
it is not expected to solve the pick task.

```bash
cd /path/to/molmospaces
python3 molmo_spaces_isaac/scripts/run_arena_benchmark_batch.py \
  --isaac_python "/path/to/IsaacLab-Arena/submodules/IsaacLab/isaaclab.sh -p" \
  --work_dir /path/to/molmospaces \
  --arena_spec_manifest /tmp/arena_episode_specs_real_ithor_pick_hard.json \
  --episode_indices 0 \
  --results_json /tmp/molmo_arena_zero_smoke_result.json \
  -- \
  --assets_root /home/$USER/.molmospaces/usd \
  --scenes_root /home/$USER/.molmospaces/usd/scenes \
  --policy_type zero \
  --with_cameras \
  --steps 200 \
  --record_video_dir /tmp/molmo_arena_zero_smoke_videos \
  --headless
```

The zero-agent command can return a non-zero eval status because the zero policy
usually times out; use the per-episode JSON/log/video to confirm launch health.

## Arena policy eval

Start the OpenPI policy server separately, then run the converted Arena episodes:

```bash
cd /path/to/molmospaces
python3 molmo_spaces_isaac/scripts/run_arena_benchmark_batch.py \
  --isaac_python "/path/to/IsaacLab-Arena/submodules/IsaacLab/isaaclab.sh -p" \
  --work_dir /path/to/molmospaces \
  --arena_spec_manifest /tmp/arena_episode_specs_real_ithor_pick_hard.json \
  --episode_indices 0-68 \
  --results_json /tmp/molmo_arena_69_results.json \
  -- \
  --assets_root /home/$USER/.molmospaces/usd \
  --scenes_root /home/$USER/.molmospaces/usd/scenes \
  --policy_type pi_remote \
  --joint_pos_policy \
  --steps 1500 \
  --pi_action_repeat 3 \
  --post_success_record_steps 150 \
  --record_video_dir /tmp/molmo_arena_videos \
  --headless
```

`--post_success_record_steps 150` records roughly 3 seconds after the first
success frame because the Arena env step is 0.02 s.

## Replay parity

To compare a successful MuJoCo trajectory against Arena open-loop replay:

```bash
python3 molmo_spaces_isaac/scripts/run_mujoco_arena_replay_parity.py \
  --episode_idx 14 \
  --arena_spec_manifest /tmp/arena_episode_specs_real_ithor_pick_hard.json \
  --mujoco_h5 /path/to/successful_mujoco_eval/house_XX/trajectories_batch_1_of_1.h5 \
  --assets_root /home/$USER/.molmospaces/usd \
  --scenes_root /home/$USER/.molmospaces/usd/scenes
```

This writes a report, Arena replay videos, and a MuJoCo/Arena side-by-side video
under the chosen `--out_dir`. The helper infers MuJoCo camera videos from the
HDF5 directory when they follow MolmoSpaces eval naming; otherwise pass
`--mujoco_external_video` and `--mujoco_wrist_video`.

## Open questions

- Should acceptance be based on spec/preflight parity, replay parity, policy
  success rate, or a combination?
- Is the 69-episode iTHOR pick PoC sufficient for review, or should the next
  target be the public MS-Pick / MolmoSpaces benchmark family?
- What exact edge cases should count as pick success when an object tips,
  slides, or briefly loses contact with the support surface?
- Are post-success videos enough for visual review, or should we add a formal
  dashboard/report?
