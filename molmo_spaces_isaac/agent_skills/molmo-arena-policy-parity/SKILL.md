---
name: molmo-arena-policy-parity
description: Run and interpret pi0/OpenPI policy parity between MolmoSpaces/MuJoCo and Isaac Lab Arena. Use when establishing baselines, running stochastic or deterministic rollouts, replaying HDF5 actions, tuning gripper/action adapter settings, or comparing benchmark success rates.
---

# Molmo Arena Policy Parity

## Purpose

Use this skill after basic conversion and visual/state parity are credible. The
goal is to compare policy behavior fairly, not to tune Arena until one rollout
happens to pass.

## Baseline Rules

- Compare the same benchmark family, checkpoint, horizon, prompt formatting,
  camera convention, action decode settings, and success definition.
- Preserve benchmark prompt punctuation exactly after strip/lower; do not add a
  period in Arena unless MuJoCo used one for the paired run.
- Keep ProcTHOR/MS-Pick numbers separate from iTHOR numbers.
- Treat single stochastic rollouts as weak evidence. Prefer repeated slices.
- Do not claim deterministic-server results as benchmark success rates; use them
  for paired debugging only.

## Workflow

1. Establish or locate the MuJoCo baseline:

- official or clearly identified MolmoSpaces commit
- policy checkpoint path/name
- task horizon and policy interval
- success scoring mode
- output summaries and videos

2. Run Arena with matched settings using:

- `molmo_spaces_isaac/scripts/run_arena_benchmark_episode.py`
- `molmo_spaces_isaac/scripts/run_arena_benchmark_batch.py`
- `molmo_spaces_isaac/scripts/test_pi_remote.sh`

3. Use repeated-rollout slices before expensive full runs. Pick episodes that
separate issues:

- success in both: smoke regression
- failure in both: policy-bad sample
- MuJoCo success and Arena failure: parity target

4. If Arena fails but conversion looks good, try:

- known MuJoCo-success HDF5 action replay in Arena
- Arena initialized at successful replay qpos
- deterministic OpenPI server for seed/action pairing
- gripper threshold/action decode inspection
- forced-close or known-close Arena smoke to distinguish missing close commands
  from gripper actuator failures

5. Record results in the progress tracker with enough settings to reproduce.

## iTHOR Current Reference

The current iTHOR baseline is `14 / 69 = 20.29%` for
`pi05_droid_jointpos` on official MolmoSpaces main with `task_horizon_steps=500`.

The paired slice result:

- idx `8`: MuJoCo `5/10`, Arena `4/10`
- idx `14`: MuJoCo `7/10`, Arena `0/10`
- idx `17`: MuJoCo `0/10`, Arena `0/10`

Use idx `14` as the main mismatch target.

From the first idx `14` report: Arena did not close because raw OpenPI gripper
scores stayed below the traced `0.5` binary threshold. The current
prompt-fixed rerun uses `pi_grasping_threshold=0.01`; prompt text and reset arm
qpos now match exactly, close commands are emitted, and the task still fails
after `1500` steps. Treat the remaining idx `14` gap as approach/action trace,
wrist/camera input, gripper contact, or policy stochasticity evidence until
finger/object diagnostics narrow it further.

The first report also showed Arena was sending `pick up the smooth gray bowl.`
while MuJoCo sent `pick up the smooth gray bowl`. Arena prompt normalization now
preserves the benchmark prompt exactly instead of auto-adding punctuation; keep
that exact prompt parity in future reruns.

## Output

Report:

- exact command/settings
- success counts
- artifact paths
- whether failure looks like policy stochasticity, policy I/O mismatch,
  action decode mismatch, or physics/contact mismatch
- next diagnostic skill to use
