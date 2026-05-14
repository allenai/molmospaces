# MolmoSpaces to Isaac Lab Arena Migration Goals

## Overarching goal

Build a low-touch proof of concept that converts one MolmoSpaces task into an
Isaac Lab Arena task and runs it end to end.

The first target is the iTHOR pick benchmark. A successful onboarding should show
that an episode from MolmoSpaces can be represented as an Arena task with matching
scene geometry, robot pose, object pose, camera observations, and policy/action
interfaces.

## Success definition

This migration is considered successful when the pi0-family policy success rate
on the migrated Isaac Lab Arena benchmark is close enough to, or matches, the
same MolmoSpaces benchmark running in MuJoCo.

Before measuring policy success, visual and state parity must be checked:

- The same episode renders similarly in MolmoSpaces/MuJoCo and Isaac Lab Arena.
- The pickup object is in the same relative location and support relationship.
- The robot base, mounted height, joint pose, gripper pose, and camera views are
  aligned closely enough for a fair policy comparison.
- The Arena observation and action interfaces match the policy checkpoint being
  evaluated.

## Required integration steps

Following Zoe's integration outline:

0. Convert MolmoSpaces episode JSON into an `ArenaEpisodeSpec`.
1. Build an Arena environment from that spec.
2. Bring MolmoSpaces USD assets into Isaac Lab and wrap them in Arena scene configs.
3. Create an Arena task for the MolmoSpaces pick task.
4. Test with a pi0-family policy.

## Engineering constraints

- Keep the implementation as low-touch as possible because this lives in the
  MolmoSpaces repo and needs to pass review there.
- Follow Isaac Lab Arena structure and best practices where possible.
- Add only the custom scaffolding needed to bridge MolmoSpaces episodes and
  assets into Arena.
- Prefer reusable converters, scene wrappers, and diagnostics over one-off fixes.
- Avoid changing Isaac Lab Arena internals unless there is no clean extension
  point in MolmoSpaces.
- Preserve benchmark semantics rather than tuning the scene until a single
  episode happens to work.

## Current proof-of-concept target

- Benchmark: real iTHOR `FrankaPickHardBench`.
- Current parity baseline: official `allenai/molmospaces` main at commit
  `286594e`, run locally in MuJoCo with `pi05_droid_jointpos`, scored
  `14 / 69 = 20.29%` on the iTHOR `FrankaPickHardBench` JSON benchmark with
  `task_horizon_steps=500`.
- Initial visual/debug validation episode: episode index 8, house 17.
- Pickup: existing scene bowl
  `bowl_9087df6907f975021e5d8ac01d4c2557_1_0_0`.
- Embodiment: Arena DROID with the MolmoSpaces stand removed and the robot root
  height calibrated to match MuJoCo TCP height for the same Franka qpos.
- Cameras: DROID shoulder/external and wrist camera observations rendered at
  MolmoSpaces' raw `624x352` resolution and adapted to OpenPI's `224x224`
  padded image inputs.
- Policy target: forked OpenPI `pi05_droid_jointpos` checkpoint, served through
  `openpi_client` and consumed by Arena through the `pi_remote` adapter.

Episode 8 remains useful for scene and robot parity because the bowl/cabinet and
robot-height issues are visible there. It is now also the first policy-parity
target: with the forked `pi05_droid_jointpos` checkpoint decoded using
`chunk_size=15` and `grasping_threshold=0.01`, MolmoSpaces/MuJoCo succeeds on
this episode. Arena executes one earlier MuJoCo-successful HDF5 replay
successfully for this episode, which validates the converted scene, dynamic
pickup, robot reset, gripper/contact path, and action timing for a known-good
trajectory. A later stochastic MuJoCo-successful HDF5 replay did not port
open-loop from reset, but initializing Arena at that trajectory's successful
HDF5 qpos succeeds in six steps. That narrows the fresh failure to path/contact
divergence during the approach rather than a missing scene/object conversion.
Arena also succeeds online on this episode with OpenPI when the Arena adapter
uses `chunk_size=15` and `grasping_threshold=0.05`, but that online result is
stochastic because the OpenPI server advances its JAX sampling RNG on every
inference. A representative six-episode online Arena batch currently reports
`0/6` successes, including a failed episode 8 when run after another episode
advanced the server RNG. The remaining policy-parity blocker is
controlled/repeated pi0 evaluation: decide whether to compare stochastic success
rates directly, restart/control the policy sampling RNG for paired debugging, or
improve observation/camera parity enough that Arena matches MuJoCo under the
same stochastic evaluation protocol.

The controlled-serving path now exists. A deterministic OpenPI wrapper can reset
the policy RNG per websocket connection and optionally step seeds across
connections. Seed 0 fails in both MuJoCo and Arena for episode 8; seeds 1-10
also failed the MuJoCo side at the shorter scan horizons tested so far. That
means low deterministic seeds are currently policy-bad samples rather than
Arena-only failures. The next useful paired test is to find or replay a
MuJoCo-successful deterministic/random sample, then evaluate that same action
stream or seed in Arena.

The current PoC scene conversion target is met for the full 69-episode real
iTHOR pick benchmark at preflight level. The single-episode PoC is complete for
Arena episode construction, open-loop policy-action replay, and one online
pi0.5/OpenPI success under the current Arena gripper-threshold calibration. The
next acceptance risk is scaling from one proven episode to a benchmark-level
success-rate comparison against the measured `20.29%` MolmoSpaces/MuJoCo iTHOR
baseline. ProcTHOR/MS-Pick remains out of scope until iTHOR parity is proven.

## Tracking docs

- Subgoals: `docs/isaaclab_arena_migration_subgoals.md`
- Progress: `docs/isaaclab_arena_migration_progress.md`

Update those docs as the path changes. The goals document should stay stable
unless the project definition or success criteria changes.
