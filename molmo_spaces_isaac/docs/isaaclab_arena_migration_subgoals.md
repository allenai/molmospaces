# MolmoSpaces to Isaac Lab Arena Migration Subgoals

This document is the working map. Update it when the path changes.

## 1. Episode conversion

Goal: Convert MolmoSpaces benchmark JSON into an `ArenaEpisodeSpec` without
requiring per-episode manual edits.

Status: working for the real iTHOR pick PoC. The full `FrankaPickHardBench`
conversion currently preflights at `69 ready / 0 failed` with the Isaac USD
asset root.

Remaining subgoals:

- Keep pickup objects already present in the scene USD on the main path.
- Keep added THOR pickup object support for episodes that introduce objects.
- Keep Objaverse support optional until it is needed for the PoC.
- Preserve robot base pose, robot init qpos, scene dataset, house index, pickup
  object identity, success threshold, and task description.
- Keep exporting full benchmark manifests so conversion can be reviewed offline.

## 2. Arena environment construction

Goal: Build an Isaac Lab Arena environment from `ArenaEpisodeSpec` using Arena
patterns instead of replacing Arena's runtime.

Status: working for one real iTHOR episode and the exported benchmark specs.

Remaining subgoals:

- Keep the resolved MolmoSpaces scene USD as the Arena background.
- Keep existing scene pickup prims wrapped as Arena-compatible rigid objects.
- Keep static scene geometry kinematic while preserving the pickup as
  dynamic.
- Keep applying episode robot init qpos deterministically at reset.
- Preserve DROID camera and action interfaces needed by pi0-family policies.
- Avoid relying on synthetic benchmark-only Z offsets for real scene pickups.
- Batch-step a representative subset after each scene/robot calibration change.

## 3. MolmoSpaces USD asset handling

Goal: Use MolmoSpaces USD assets in Isaac Lab with minimum custom translation.

Status: working for iTHOR scenes and scene-embedded pickups.

Remaining subgoals:

- Keep resolving iTHOR scene USD paths from benchmark `scene_dataset` and
  `house_index`.
- Keep the resolver tolerant of common Isaac asset layouts, including
  `assets/usd/scenes/...` and versioned iTHOR scene directories.
- Keep using existing scene prims for scene-embedded pickups.
- Keep duplicated THOR asset spawning available for older smoke tests.
- Validate scene object physics and collision behavior across more than one
  house.
- Document any required asset download or symlink layout assumptions.

## 4. MolmoSpaces pick task in Arena

Goal: Represent MolmoSpaces pick as an Arena task with comparable success logic.

Status: working for the PoC with lift-based success.

Remaining subgoals:

- Confirm the pick success threshold matches MolmoSpaces intent.
- Confirm episode timeout and step rate are fair for policy evaluation.
- Make task construction reusable across the benchmark.
- Add diagnostics for pickup Z, lift, robot motion, and camera observations.

## 5. Visual and state parity

Goal: Compare MuJoCo and Arena for the same episode before running policy metrics.

Status: scene/object/robot parity is working for episode 8; benchmark scaling
remains. The original bowl-in-cabinet issue and extra robot stand issue are
fixed for the real PoC episode; robot root height has also been calibrated
against MuJoCo TCP height. Raw camera render size/FOV now matches the
MolmoSpaces DROID eval path before OpenPI resizing. Full HDF5 replay succeeds
for episode 8 after fixing dynamic pickup physics. Online OpenPI/pi0.5 now
succeeds for episode 8 with `--pi_grasping_threshold 0.05`, while strict
`0.01` MuJoCo decode parity still fails in Arena. A representative online Arena
batch over indices `0,8,17,34,51,68` builds all selected scenes and pickups, but
currently has `0/6` policy successes. A diagnostic DROID root xyz offset
improved FK parity for one fresh MuJoCo-success row, but broke the known
full-HDF5 replay when promoted to the default; keep xyz mount offsets as
diagnostic overrides instead of defaulting them until validated across
trajectories.

Remaining subgoals:

- Keep the fresh paired MuJoCo/Arena montage after calibrated root height,
  stand removal, and 624x352 camera rendering.
- Verify bowl/cabinet/support relationship, robot pose, gripper pose, and
  camera view for a small representative set.
- Compare robot body poses and camera extrinsics where available.
- Treat lighting as a policy-input parity variable: inventory MuJoCo lights and
  USD `UsdLux` prims, then compare resized policy image luminance statistics
  before relying on qualitative brightness judgments.
- Use the idx `14` lighting comparison report and runtime light scale hook for
  ablations instead of editing downloaded USD scene assets in place. First
  tested image-space scale target is approximately `0.79`, but brightness alone
  does not explain the full image mismatch. The runtime
  `MOLMO_ARENA_SCENE_LIGHT_SCALE=0.79` rerun also failed and left reset exterior
  and wrist mean-absolute image gaps essentially unchanged.
- Resolve or explicitly document the wrist-camera parent-frame mismatch between
  the MuJoCo `gripper/wrist_camera` MJCF frame and Arena's flattened DROID USD
  `Robotiq_2F_85/base_link` frame.
- Keep comparing full OpenPI inputs for the same policy step, including resized
  shoulder image, resized wrist image, arm qpos, gripper qpos, and first action
  chunk. Prompt text and reset arm qpos now match exactly for idx `14`; the old
  MuJoCo HDF5 still lacks raw OpenPI chunks, so strict raw-output comparison
  requires a new MuJoCo trace.
- Keep scene-embedded pickup rigid bodies dynamic by matching every rigid body
  under the pickup scene-object path, not only the root prim name.
- Keep patched pickup mass/inertia handling for scene USDs that import invalid
  or sentinel mass attributes.
- Batch-check a small representative set of houses before full benchmark runs.
- Keep visual artifacts available for reviewer/debug use.
- Track problematic scene-render warnings separately from pickup conversion
  errors; episode 68 currently shows large-transform renderer warnings for
  unrelated bathroom showerhead geometry while still reaching the policy loop.

## 6. pi0-family policy evaluation

Goal: Run pi0-family policy on the migrated Arena task and compare against
MolmoSpaces MuJoCo benchmark success.

Status: open-loop one-episode replay works; online one-episode policy PoC works
with an Arena-specific gripper threshold; benchmark-level parity is not
complete, and online policy success is stochastic. A deterministic OpenPI
serving path now exists for paired debugging, but low seeds tested so far are
MuJoCo failures too. The active parity target is now the 69-episode iTHOR
`FrankaPickHardBench`. A clean official-main MuJoCo run with
`pi05_droid_jointpos` and `task_horizon_steps=500` established the local
baseline at `14 / 69 = 20.29%`. ProcTHOR/MS-Pick remains deferred until this
iTHOR MuJoCo-vs-Arena comparison is proven. A paired 3-episode repeated-rollout
slice now gives a sharper debugging target: idx `8` is roughly comparable
(`5/10` MuJoCo vs `4/10` Arena), idx `17` fails in both (`0/10` vs `0/10`), and
idx `14` is the clear simulator mismatch (`7/10` MuJoCo vs `0/10` Arena).

Current finding: the correct forked `pi05_droid_jointpos` policy runs from an
OpenPI server and Arena can consume it. Episode 8 succeeds in MolmoSpaces/MuJoCo
when the policy is decoded with `chunk_size=15` and `grasping_threshold=0.01`.
After the dynamic-pickup fix, one successful HDF5 action sequence succeeds in
Arena. A fresh MuJoCo-success HDF5 sampled later does not replay successfully
from reset in Arena, but succeeds when Arena is initialized at its successful
HDF5 qpos. Online Arena OpenPI succeeds on episode 8 with prompt parity and
`--pi_grasping_threshold 0.05`, but fails with the stricter `0.01` threshold
used for the successful MuJoCo decode. A post-batch trace shows episode 8 can
also fail with `0.05` when the OpenPI server RNG has advanced: reset qpos and
prompt match, camera inputs are nearly identical, but the first sampled action
chunk differs substantially. Treat the remaining gap as stochastic policy
evaluation, approach-path/contact parity, policy-input parity, and
gripper-decode parity work, not a scene-conversion blocker. A deterministic
server reset with seed 0 fails in both MuJoCo and Arena for episode 8; seeds
1-10 also fail MuJoCo under the scan horizons used so far. This makes the next
policy-parity target either a known successful HDF5/open-loop action sequence or
a deterministic seed/action sample that first succeeds in MuJoCo.

The idx `14` policy I/O reruns changed the blocker shape. Prompt text and
reset arm qpos now match exactly, and the intended `0.01` threshold sends close
commands in Arena (`47/60` traced calls in the unscaled rerun, `57/60` in the
first `0.79` light-scale rerun, and `60/60` in the latest `0.79` rerun). These
runs still fail after `1500` steps. The latest rerun kept the end-effector far
from the bowl (`~0.43m` to `~0.49m` near the sampled debug points), so the
current idx `14` gap is downstream of prompt parity, reset qpos parity, and
basic close-command emission. The old MuJoCo HDF5 stores decoded/applied
action fields but not raw OpenPI chunks, so a strict raw model-output diff
needs a new MuJoCo trace with matching hooks.

Remaining subgoals:

- Keep the reproducible OpenPI server/client setup documented.
- Use the official-main iTHOR MuJoCo baseline `14 / 69 = 20.29%` as the current
  Arena success-rate target.
- Use repeated-rollout slices before full benchmark reruns. The current slice is
  stored at
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/benchmark_3eps_x10`.
- Focus the next policy-parity investigation on original iTHOR idx `14`
  ("Pick up the smooth gray bowl"), where MuJoCo is `7/10` and Arena is `0/10`.
- Use the idx `14` policy I/O diff plan as the next debugging runbook:
  `/home/horde/molmo-proj/molmospaces/molmo_spaces_isaac/docs/idx14_policy_io_diff_debug_plan.md`.
- For idx `14`, separate "Arena received close commands" from "Arena can close
  and lift this bowl on this approach path" by running a forced-close or
  known-close replay smoke test with finger/object/contact diagnostics.
  Latest evidence says the arm does not approach closely enough yet, so prefer
  TCP/finger/object trajectory comparison before contact-only debugging.
- Keep Arena OpenPI prompt strings exact to the benchmark/MuJoCo prompt after
  strip/lower; do not auto-add terminal punctuation.
- Keep episode 8 as the first MuJoCo-successful policy-parity target and smoke
  regression.
- Run a small representative Arena batch with online OpenPI and the current
  successful Arena adapter calibration (`--pi_grasping_threshold 0.05`) to catch
  conversion problems beyond the first episode. Current result: `0/6` successes,
  but all six episodes reached the policy loop and wrote per-episode artifacts.
- Use repeated indices in the subprocess batch runner, for example
  `--episode_indices 8,8,8`, to estimate stochastic online success for a single
  converted episode.
- Use deterministic OpenPI serving mode for paired MuJoCo/Arena debugging when
  testing exact seeds or seed sweeps. The stock server remains useful for
  stochastic success-rate measurement.
- Compare observation image shapes, camera names, joint positions, and gripper
  action scaling against MolmoSpaces pi0 eval.
- Continue gripper/contact diagnostics under the MuJoCo-aligned `0.01`
  threshold. The old no-close symptom at `0.5` is resolved, but idx `14` still
  fails after close commands are emitted.
- Add/keep diagnostics for arm command tracking, all Robotiq gripper joints, and
  finger-to-object distances.
- Keep `MOLMO_ARENA_DROID_MOUNT_X/Y/Z` as diagnostic overrides for root-frame
  calibration experiments. Do not promote an xyz offset without preserving the
  known successful full-HDF5 replay and visual parity.
- Run enough episodes to estimate success rate.
- Investigate any gap between Arena and MuJoCo success rates.
- Keep ProcTHOR/MS-Pick out of the immediate parity report. Do not mix those
  benchmark numbers with iTHOR results.

## 7. Review readiness

Goal: Keep the final patch understandable and acceptable for MolmoSpaces review.

Status: not started.

Remaining subgoals:

- Minimize custom scripts or clearly mark them as diagnostics.
- Keep migration-facing docs current with known limitations.
- Separate customer-review artifacts from local diagnostics and generated media.
- Before review, make the target branch buildable without local-only state beyond
  documented asset/cache paths.
