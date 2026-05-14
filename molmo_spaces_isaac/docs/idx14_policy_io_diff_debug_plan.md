# idx 14 MuJoCo vs Arena Policy I/O Diff Plan

This document defines the next debugging exercise for the iTHOR pick PoC. The
goal is to let a human inspect exactly what pi0.5 receives and returns in
MolmoSpaces/MuJoCo versus Isaac Lab Arena for the same episode, instead of
relying on aggregate success rates or terminal logs.

## Debug target

- Benchmark: iTHOR `FrankaPickHardBench`
- Original benchmark index: `14`
- House: `20`
- Task: `Pick up the smooth gray bowl`
- Current paired result: MuJoCo `7 / 10`, Arena `0 / 10`
- Slice benchmark:
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/benchmark_3eps_x10`
- Summary:
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/mujoco_vs_arena_3eps_x10_summary.json`

This is the right debugging episode because the policy can solve it in MuJoCo
often enough to provide useful reference behavior, while Arena currently never
succeeds.

## Core questions

1. Are the policy inputs equivalent at reset?
2. If the inputs are equivalent, does OpenPI return equivalent first action
   chunks?
3. If first-frame policy I/O is equivalent, where does the closed-loop rollout
   diverge?
4. If the rollout diverges, is the cause camera observations, proprioception,
   action decoding, control timing, robot physics/contact, or scene/object
   physics?

## Required output

Produce a self-contained viewable report at:

`/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/report.html`

Also save the source data and plots beside it:

- `manifest.json`: run metadata, commands, commit, checkpoint, episode id.
- `mujoco_trace.npz` or `mujoco_trace.h5`: captured MuJoCo policy trace.
- `arena_trace.npz` or `arena_trace.h5`: captured Arena policy trace.
- `plots/`: PNG plots used by the report.
- `frames/`: policy camera inputs used by the report.
- `videos/`: short MuJoCo/Arena rollout videos if available.
- `report.md`: Markdown version of the same report for code review.

The report must be understandable without reading code. It should show side by
side images and overlaid plots with captions that say what is being compared.

## Capture layers

### Layer 1: reset-frame policy I/O diff

This isolates scene conversion, camera rendering, observation encoding, prompt
formatting, and first action decoding.

Capture exactly one policy input from MuJoCo and Arena immediately before the
first OpenPI inference call:

- `prompt`
- raw shoulder/exterior RGB image before resize/pad
- raw wrist RGB image before resize/pad
- resized OpenPI shoulder image, `224x224`
- resized OpenPI wrist image, `224x224`
- arm joint position vector, 7 values, in policy order
- gripper position scalar after the policy adapter's normalization
- full qpos available to the simulator, for debugging convention mismatches
- robot base pose
- end-effector pose
- target object pose
- camera intrinsics and extrinsics if available

Then run the same policy input through OpenPI and save:

- raw returned action chunk, shape `chunk_size x 8`
- decoded arm joint-position commands, 7 values per action
- decoded gripper command before and after thresholding
- action repeat / timing settings used by the simulator

Important: deterministic OpenPI serving may be used for this layer only, because
the goal is to remove policy sampling as a confounder. Deterministic output
should not be used as a benchmark success-rate claim.

### Layer 2: closed-loop trace diff

This shows how the two simulators diverge once actions are applied.

Run one representative MuJoCo rollout and one Arena rollout for idx `14`, using
the normal pi0.5 server path. Prefer one MuJoCo-successful rollout when possible.
Capture at every policy inference step, not every low-level physics step:

- timestep / policy step index
- success flag and terminal reason
- policy input images
- arm joint positions
- gripper position
- end-effector pose
- target object pose
- target object height relative to reset
- finger-to-object distances if available
- raw action chunk returned at that inference
- selected action from the chunk
- decoded arm action applied to the sim
- decoded gripper action applied to the sim
- contact/lift diagnostics used by Arena success

If MuJoCo and Arena cannot be meaningfully aligned after closed-loop divergence,
compare the first `N=10` policy inference calls by step index and clearly mark
that later differences may be downstream effects, not root causes.

## Report layout

### 1. Run summary

Show one table:

| Field | MuJoCo | Arena |
|---|---|---|
| episode original idx | `14` | `14` |
| house | `20` | `20` |
| prompt | exact string | exact string |
| checkpoint | path/name | path/name |
| chunk size | value | value |
| gripper threshold | value | value |
| policy dt / action repeat | value | value |
| rollout result | success/fail | success/fail |

### 2. Reset camera inputs

Show a 2x2 tile:

- MuJoCo shoulder/exterior OpenPI input
- Arena shoulder/exterior OpenPI input
- MuJoCo wrist OpenPI input
- Arena wrist OpenPI input

Also show image-difference diagnostics:

- absolute pixel difference heatmap for shoulder
- absolute pixel difference heatmap for wrist
- simple image metrics: mean absolute error, max error, SSIM if available

### 3. Reset proprioception

Plot MuJoCo and Arena on the same graph:

- 7 arm joint positions, grouped by joint name
- gripper policy scalar
- full gripper raw joint values, if simulator conventions differ

Use one bar chart for reset values and one delta chart:

- `mujoco_value`
- `arena_value`
- `arena_minus_mujoco`

### 4. First action chunk

Plot MuJoCo and Arena first model outputs on the same graphs:

- raw action chunk, joint dimensions `0..6`
- raw gripper action dimension `7`
- decoded joint-position commands
- decoded binary gripper commands

This catches prompt/camera/qpos differences that do not look obvious by eye but
change OpenPI's sampled action.

### 5. Rollout proprioception trace

For each policy inference step, overlay:

- observed arm qpos for MuJoCo and Arena
- observed gripper scalar for MuJoCo and Arena
- end-effector xyz for MuJoCo and Arena
- target object xyz for MuJoCo and Arena
- target object lift from reset

These plots answer whether Arena starts correctly but drifts because of control
or contact.

### 6. Rollout action trace

For each policy inference step, overlay:

- selected raw action values
- decoded arm joint-position commands
- decoded gripper command
- gripper open/close threshold decision

These plots answer whether the policy is asking both robots to do the same
thing and whether the Arena adapter applies the same thing.

### 7. Scene and contact diagnostics

Show a short table or plots for:

- target object starting pose and support surface height
- bowl/counter/cabinet relationship at reset
- robot base pose and root height
- target object mass/inertia values in Arena after patching
- finger-to-object minimum distance over time
- robot-only contact force versus non-robot contact force
- success-condition terms: contact, lift, timeout

This catches the case where policy I/O matches, but the object or gripper
physics does not.

## Pass/fail criteria

Do not move on from idx `14` until one of these is true:

- Policy input mismatch is found and categorized as camera, qpos, prompt, or
  timing/action adapter.
- Policy output mismatch is found after controlling for policy randomness.
- Policy I/O is close enough, but robot/object physics divergence is found.
- No meaningful mismatch is found in reset-frame I/O, first action, rollout
  traces, or contact diagnostics; then the next step is a broader stochastic
  benchmark or policy fine-tuning hypothesis.

Suggested numerical tolerances for "close enough":

- prompt: exact string match after the same lowercase transformation used by
  MolmoSpaces `PI_Policy`
- arm qpos: max absolute difference under `0.03 rad` at reset
- gripper policy scalar: absolute difference under `0.05`
- camera resized input: visually similar, with mean absolute pixel difference
  tracked rather than hard-gated initially
- first deterministic raw action chunk: same qualitative direction; exact
  numeric equality is not expected unless policy serving is deterministic and
  inputs are byte-identical
- decoded action timing: same chunk size, same action repeat, same gripper
  threshold, same joint order

## Implementation checklist

1. Add MuJoCo trace capture for idx `14`.
   - Instrument the existing MolmoSpaces `PI_Policy.obs_to_model_input`,
     `inference_model`, and `model_output_to_action` path without changing
     policy behavior.
   - Save policy-step traces and images.
2. Add Arena trace capture for idx `14`.
   - Instrument the Arena pi remote adapter at the matching boundaries:
     policy input dict, raw OpenPI action chunk, decoded action, applied action.
   - Save the same field names as MuJoCo wherever possible.
3. Add a report builder.
   - Load MuJoCo and Arena traces.
   - Generate side-by-side camera tiles.
   - Generate overlaid proprioception/action plots.
   - Generate `report.html` and `report.md`.
4. Run reset-frame deterministic diff.
5. Run normal closed-loop trace diff.
6. Inspect the report and classify the mismatch.

## Expected next decision

After this report exists, the next fix should be chosen by evidence:

- If cameras differ: adjust camera frame/FOV/render preprocessing.
- If qpos differs: fix joint ordering, root transform, or gripper normalization.
- If first action differs despite matching inputs: isolate OpenPI server
  stochasticity and compare deterministic/offline inference.
- If actions match but motion diverges: debug Arena robot control, action
  repeat, gripper actuator, contact, mass, inertia, or object collision.
- If idx `14` is fixed, rerun the same `3 x 10` slice before returning to the
  full 69-episode iTHOR baseline.
