---
name: molmo-arena-policy-io-diff
description: Capture and report MuJoCo versus Isaac Lab Arena policy input/output differences for MolmoSpaces episodes. Use when a policy succeeds in MuJoCo but fails in Arena and Codex needs to instrument camera inputs, prompts, qpos, raw OpenPI chunks, decoded actions, rollout traces, contacts, or generate report.html.
---

# Molmo Arena Policy I/O Diff

## Purpose

Use this skill when aggregate success rates point to a specific MuJoCo-vs-Arena
policy mismatch. The output should be a human-readable report that explains
where the two systems first diverge.

## Required Report Shape

Create a self-contained report directory with:

- `manifest.json`
- MuJoCo trace data
- Arena trace data
- `frames/`
- `plots/`
- optional `videos/`
- `report.md`
- `report.html`

For the current iTHOR idx `14` target, use:

`/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff_promptfix_thresh001_rerun7/report.html`

The first light-scale ablation report is:

`/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff_promptfix_thresh001_light079/report.html`

## Capture Layers

1. Reset-frame policy I/O:

- prompt
- raw shoulder/exterior RGB before resize/pad
- raw wrist RGB before resize/pad
- OpenPI `224x224` shoulder input
- OpenPI `224x224` wrist input
- arm qpos in policy order
- gripper policy scalar and raw gripper qpos
- full simulator qpos
- robot base, end-effector, target object poses
- camera intrinsics/extrinsics where available
- raw OpenPI action chunk
- decoded arm commands
- decoded gripper command and threshold decision
- action repeat/timing settings

Prompt comparison must be exact, including terminal punctuation. Do not
normalize by auto-adding punctuation before declaring parity.

2. Closed-loop trace at policy inference steps:

- policy input images
- qpos/proprioception
- selected raw and decoded actions
- end-effector and target object pose
- target lift from reset
- finger/object distances if available
- contact and success-condition diagnostics

## Workflow

1. Read the episode-specific runbook if present. For iTHOR idx `14`:

`molmo_spaces_isaac/docs/idx14_policy_io_diff_debug_plan.md`

2. Instrument both sides at the same semantic boundaries. If exact simulator
step alignment breaks after divergence, compare the first several policy
inference calls by policy step index and mark later differences as downstream.

3. Prefer deterministic OpenPI serving for reset-frame first-action comparison
when policy sampling would obscure the signal. Use normal stochastic serving
for benchmark-style success claims.

4. Build plots that answer one question each:

- Are reset camera inputs equivalent?
- Are reset qpos/proprioception values equivalent?
- Is the first raw action chunk equivalent?
- Are decoded arm/gripper commands equivalent?
- Does object/contact behavior diverge after similar commands?

5. When gripper motion is missing, always separate three cases:

- raw model gripper output never crosses the binary close threshold
- raw output crosses the threshold but decoded command stays open
- decoded close is sent but the Arena gripper actuator does not move

For threshold-gated failures, report the max raw gripper score and counts above
candidate thresholds such as `0.5`, `0.05`, and `0.01`.
For current pi0.5 DROID joint-position runs, use `pi_grasping_threshold=0.01`
unless deliberately reproducing an old trace.

6. End the report with a categorized finding:

- camera/prompt/input mismatch
- proprioception/qpos mismatch
- policy output mismatch after randomness control
- action decode/timing mismatch
- robot/object physics or contact mismatch
- inconclusive, with the next missing capture

## Current Helper

The current report builder draft is:

`molmo_spaces_isaac/scripts/build_idx14_policy_io_report.py`

The current lighting report helper is:

`molmo_spaces_isaac/scripts/compare_idx14_lighting.py`

Keep this skill updated as the report format stabilizes.
