# MolmoSpaces to Isaac Lab Arena Migration Progress

This tracker is append-only at the high level. Add dated entries for completed
work, failed attempts, decisions, and open risks.

## 2026-05-14

### Completed

- Reviewed the generated idx `14` MuJoCo-vs-Arena policy I/O report.
- Confirmed reset arm qpos is aligned at the policy boundary, so idx `14` is no
  longer a basic reset-qpos mismatch.
- Confirmed the policy camera inputs are qualitatively similar but materially
  different: Arena is brighter/washed compared with MuJoCo, and the wrist view
  differs enough to remain a likely policy-input contributor.
- Inspected the converted FloorPlan20 USD lighting. The Arena scene currently
  has a default `DistantLight` (`intensity=1000`, `rotateX=-10`) plus an iTHOR
  `DomeLight` (`intensity=500`, warm color), while the MuJoCo base scene uses a
  renderer headlight (`ambient=0.35`, `diffuse=0.4`). This lighting was not
  imported as a matched rig, so lighting/render-parity is now a first-class
  idx `14` hypothesis.
- Added and ran a focused idx `14` lighting comparison report. It measures
  MuJoCo/Arena policy-input luminance and tests simple image-space exposure
  corrections. The report shows Arena is somewhat brighter, but dimming alone
  only reduces a small part of the image error, so lighting is a contributor
  rather than a complete explanation for the idx `14` failure.
- Added runtime Arena scene-light ablation controls so future runs can scale or
  disable the imported USD `DistantLight` and `DomeLight` without editing the
  downloaded scene assets in place.
- Fixed Arena OpenPI prompt normalization to preserve the benchmark prompt text
  exactly after strip/lower instead of auto-adding terminal punctuation. This
  removes the idx `14` mismatch where MuJoCo sent
  `pick up the smooth gray bowl` but Arena sent `pick up the smooth gray bowl.`.
- Aligned the idx `14` report builder's default Arena gripper threshold with
  the current Arena CLI default (`0.01`) so future reports do not accidentally
  re-use the old `0.5` no-close plotting threshold.
- Confirmed prompt strings differ only by terminal punctuation in the traced
  report: MuJoCo sends `pick up the smooth gray bowl`, while Arena sends
  `pick up the smooth gray bowl.`.
- In the initial traced report, confirmed the Arena gripper did not close
  because the Arena raw OpenPI gripper score never exceeded the traced binary
  close threshold `0.5`. The maximum observed Arena raw gripper score in that
  report was about `0.268`, so the adapter decoded every gripper action as
  open/no-close.

### Evidence

- Report:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/report.html`
- Summary:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/summary.json`
- Gripper plot:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/plots/arena_raw_gripper_score.png`
- Reset camera comparison:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/frames/reset_policy_input_cameras.png`
- Lighting comparison:
  `/home/horde/molmo-proj/diagnostics/idx14_lighting_compare/report.html`

### Rerun result

- Restored a working local Isaac/Arena runtime by launching through the project
  venv (`VIRTUAL_ENV=/home/horde/molmo-proj/.venv`) and using
  `/home/horde/.molmospaces/usd/scenes` as the scene root.
- Added the required local USD cache symlink from
  `/home/horde/.molmospaces/usd/scenes/objects/thor` to
  `/home/horde/.molmospaces/usd/objects/thor/20260128`. The exported iTHOR
  scene USDs reference `scenes/objects/thor/...`, while the downloaded object
  assets are materialized under `usd/objects/thor/20260128/...`.
- Added an Arena-side OpenPI websocket client wrapper that disables websocket
  ping keepalive. The first JAX inference can exceed the default 20s ping
  timeout; `PiRemotePolicy` still owns the application-level inference timeout.
- Reran idx `14` with the prompt fix, `pi_chunk_size=15`, and
  `pi_grasping_threshold=0.01`:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff_promptfix_thresh001_rerun7/report.html`.
  The rollout completed `1500` steps and still failed (`0/1` for this rerun).
- The rerun report now shows prompt exact match and reset arm qpos exact match.
  The gripper no-close symptom from the old `0.5` threshold is gone: with the
  `0.01` threshold, decoded close commands are sent (`47/60` traced calls close;
  first close at call `0`, raw gripper score `0.01085`).
- Ran the first runtime lighting ablation with
  `MOLMO_ARENA_SCENE_LIGHT_SCALE=0.79`:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff_promptfix_thresh001_light079/report.html`.
  It also completed `1500` steps and failed. The log confirms `2` scene light
  intensity attributes were scaled.
- The `0.79` lighting ablation did not materially change the reset image gap:
  unscaled exterior/wrist mean abs was `36.98`/`55.34`; scaled was
  `36.98`/`55.31`. This supports the earlier finding that simple scalar light
  dimming is not the complete idx `14` mismatch.

### Current finding

- The old missing-gripper-motion symptom was threshold-induced in the first
  report. With the intended `0.01` threshold, Arena does send close commands,
  yet idx `14` still fails. The remaining failure is now downstream of prompt
  parity, reset arm qpos parity, and basic close-command emission.
- The next likely causes to isolate are action/approach trajectory divergence,
  wrist/camera policy-input mismatch, gripper contact/object interaction, and
  policy-output stochasticity. Simple scalar light dimming did not recover the
  visual gap or the task.
- The old MuJoCo HDF5 still lacks raw OpenPI chunks, so the current report can
  compare Arena raw chunks/actions against saved MuJoCo decoded/applied action
  fields, but not perform a strict raw model-output diff.

### Next planned work

- Run a forced-close or known-close replay smoke in Arena to separately verify
  that the gripper actuator/contact path closes on the bowl when commanded.
- Add or run time-series diagnostics for TCP pose, finger-to-object distance,
  bowl pose/lift, and commanded/actual gripper joints around the first close
  calls in the new idx `14` traces.
- If a strict raw policy-output comparison is needed, rerun MuJoCo with matching
  OpenPI trace hooks; the old successful HDF5 is not enough for raw-chunk parity.
- Continue camera/wrist parity work, but do not expect scalar scene-light
  dimming alone to close the gap.

## 2026-05-07

### Completed

- Created a focused idx `14` policy I/O diff plan to compare MuJoCo and Arena
  at the boundaries that matter to pi0.5: policy camera inputs, prompt,
  proprioception, raw OpenPI output chunks, decoded joint actions, gripper
  commands, rollout traces, and contact/object diagnostics.

### Evidence

- Debug plan:
  `/home/horde/molmo-proj/molmospaces/molmo_spaces_isaac/docs/idx14_policy_io_diff_debug_plan.md`

### Next planned work

- Implement the trace capture and report builder described in the idx `14`
  policy I/O diff plan.
- Generate a self-contained report at:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/report.html`

## 2026-05-06

### Completed

- Stopped the full sequential Arena baseline before completion and replaced it
  with a smaller paired debugging benchmark: 10 rollouts each for original
  iTHOR benchmark indices `8`, `14`, and `17`.
- Built a temporary 30-entry benchmark slice with repeated copies of those
  three episodes. This keeps both simulators on the normal benchmark/eval path
  while giving per-episode stochastic success rates.
- Ran the slice in MolmoSpaces/MuJoCo with official-main eval settings and
  `pi05_droid_jointpos`.
- Ran the same slice in Isaac Lab Arena with the current MolmoSpaces Arena
  conversion and strict contact-aware pick success.
- Measured per-episode success rates:
  - idx `8`, "Pick up the bowl": MuJoCo `5 / 10`, Arena `4 / 10`.
  - idx `14`, "Pick up the smooth gray bowl": MuJoCo `7 / 10`, Arena `0 / 10`.
  - idx `17`, "Pick up the spoon": MuJoCo `0 / 10`, Arena `0 / 10`.

### Evidence

- Benchmark slice:
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/benchmark_3eps_x10`
- Combined MuJoCo-vs-Arena summary:
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/mujoco_vs_arena_3eps_x10_summary.json`
- MuJoCo slice output:
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/mujoco_slice/PiRemotePolicyEvalConfig/20260505_231442`
- Arena slice per-episode outputs:
  `/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/arena_slice/arena_3eps_x10_episodes`

### Current finding

- Online pi0.5 is stochastic even for repeated identical MuJoCo episode JSON.
  Single-rollout success/failure is not enough to judge parity.
- Episode idx `8` is close enough for coarse parity (`5/10` MuJoCo vs `4/10`
  Arena) and is less useful as the main blocker.
- Episode idx `17` fails in both simulators (`0/10` vs `0/10`), so it is not a
  good policy-success debugging target.
- Episode idx `14` is the clear mismatch: MuJoCo succeeds reliably enough
  (`7/10`) while Arena never succeeds (`0/10`). Use idx `14` as the next
  focused scene/action/observation parity target.

## 2026-05-05

### Completed

- Adopted the current project focus: defer ProcTHOR/MS-Pick and use iTHOR
  `FrankaPickHardBench` as the immediate MuJoCo-vs-Arena policy parity target.
- Ran the full 69-episode iTHOR `FrankaPickHardBench` on a clean official
  `allenai/molmospaces` main checkout at commit `286594e`, with only an
  external compatibility shim for the current `PI_Policy` constructor mismatch.
- Confirmed the policy checkpoint in use is forked OpenPI
  `pi05_droid_jointpos`, served from
  `/home/horde/molmo-proj/openpi_cache/openpi-assets/checkpoints/pi05_droid_jointpos`.
- Established the local MuJoCo iTHOR baseline: `14 / 69 = 20.29%` success with
  `task_horizon_steps=500`.
- Generated MolmoSpaces benchmark CSV summaries for the iTHOR baseline. `oracle`
  and `both` scoring both report `20.29%`; for `both`, at-end and oracle agree.
- Stopped the in-progress modified-branch MuJoCo benchmark run because it was
  using local Isaac/Arena branch policy config changes and the 69-episode
  iTHOR `FrankaPickHardBench`, not the official MolmoSpaces leaderboard pick
  benchmark.
- Pulled the official MolmoSpaces website leaderboard data for MS-Pick. The
  published pi0.5 DROID result is `364 / 1000 = 36.4%` overall, with a 95%
  confidence interval of `33.48% - 39.43%`.
- Identified the official MS-Pick command/benchmark from the website:
  `FrankaPickDroidMiniBench_json_benchmark_20251231` under
  `procthor-10k`, with `--task_horizon_steps 450`.
- Created a clean official `allenai/molmospaces` main worktree at
  `/home/horde/molmo-proj/molmospaces-main-baseline`.
- Tried the current official `main` commit `286594e`; it fails before rollout
  because `PI_Policy.__init__(exp_config)` no longer matches
  `pipeline.setup_policy`, which still calls `policy_cls(exp_config, task)`.
- Created an older official-release worktree at
  `/home/horde/molmo-proj/molmospaces-official-molmobot-release` from commit
  `cd23bec` and ran the official MS-Pick benchmark path with only an external
  remote-OpenPI eval config.
- Ran five MS-Pick pi0.5 smoke episodes from that official release path:
  indices `0..4` produced `2 / 5` successes. This confirms the stock
  MolmoSpaces/MuJoCo pi0.5 path can succeed locally and qualitatively matches
  the nonzero official `36.4%` leaderboard result.

### Evidence

- Official website CSV source:
  `https://molmospaces.allen.ai/benchmark/data/ms_pick/pi0.5.csv`
- Official website task index:
  `https://molmospaces.allen.ai/benchmark/data/index.json`
- External remote-server eval config used for the older release smoke:
  `/home/horde/molmo-proj/diagnostics/pi_remote_eval_config.py`
- Smoke run artifacts:
  `/home/horde/molmo-proj/diagnostics/mujoco_official_release_ms_pick_pi05_smoke`
- Official-main iTHOR MuJoCo baseline output:
  `/home/horde/molmo-proj/diagnostics/mujoco_official_main_ithor_pick_pi05_full/PiRemotePolicyEvalConfig/20260505_180404`
- Official-main iTHOR MuJoCo CSV summaries:
  `/home/horde/molmo-proj/diagnostics/mujoco_official_main_ithor_pick_pi05_full/pi05_droid_jointpos_ithor_pick_official_main_oracle.csv`
  and
  `/home/horde/molmo-proj/diagnostics/mujoco_official_main_ithor_pick_pi05_full/pi05_droid_jointpos_ithor_pick_official_main_both.csv`
- Current-main external compatibility shim:
  `/home/horde/molmo-proj/diagnostics/official_main_pi_remote_eval_config.py`

### Failed attempts and lessons

- The modified-branch 69-episode iTHOR run is useful for Arena conversion and
  scene debugging, but it is not the official MS-Pick baseline target.
- Current official `main` appears to have a policy-constructor mismatch for
  Pi eval. Use either a fixed current-main checkout or the older official
  release commit for baseline reproduction.
- The official MS-Pick benchmark is `1000` fixed ProcTHOR episodes across
  hundreds of houses, not one scene repeated.
- A full local 1000-episode rerun would be expensive; the website leaderboard
  number should be treated as the target baseline unless a full reproduction run
  is explicitly needed.

### Current open risks

- Arena conversion work currently targets iTHOR `FrankaPickHardBench`. This is
  now the active PoC parity target; ProcTHOR `FrankaPickDroidMiniBench` and the
  website MS-Pick leaderboard are deferred until iTHOR MuJoCo-vs-Arena parity is
  demonstrated.
- The same iTHOR benchmark must now be run in Arena with the same checkpoint,
  prompt formatting, camera convention, action decode settings, and horizon
  before claiming policy parity.
- Any Arena success-rate comparison must use the same benchmark family, policy
  checkpoint, policy decode settings, horizon, and camera convention as the
  selected MuJoCo baseline.

## 2026-05-01

### Completed

- Installed and used Isaac Lab Arena from source with the MolmoSpaces fork in
  the shared workspace.
- Installed the dependencies needed to run MolmoSpaces benchmark loading,
  MuJoCo rendering, Arena environment construction, and OpenPI client smoke
  tests.
- Added conversion support from MolmoSpaces episode JSON to `ArenaEpisodeSpec`
  for iTHOR pick episodes.
- Added support for real benchmark pickups that already exist inside the scene
  USD, instead of assuming every pickup is an added THOR object.
- Added an Arena wrapper for existing scene pickup prims.
- Built a real Arena environment from iTHOR `FrankaPickHardBench` episode 8.
- Patched scene physics so the referenced iTHOR scene is stable while the pickup
  object remains dynamic.
- Applied MolmoSpaces robot init qpos deterministically in Arena reset.
- Aligned Arena DROID camera offsets with MolmoSpaces DROID eval camera pose.
- Removed Arena's extra DROID `Robot_Stand` prop for MolmoSpaces runs while
  keeping the robot root at the MolmoSpaces `fr3_link0` mount height.
- Rendered the same real episode in MolmoSpaces/MuJoCo and Isaac Lab Arena for
  visual parity comparison.
- Verified the bowl is on the counter/cabinet in Arena, not inside the cabinet.
- Verified the real iTHOR pick benchmark preflight reports `69 ready / 0 failed`.
- Exported all 69 real iTHOR pick episodes to an Arena spec manifest.
- Ran a short Arena zero-policy smoke test on the real episode.
- Ran a one-step pi0.5/OpenPI remote smoke test that reached the OpenPI server
  and returned an 8D joint-position action.

### Evidence

- MuJoCo vs Arena visual parity with stand removed:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/mujoco_vs_arena_scene_pickup_no_stand_ep8.png`
- Arena no-stand diagnosis summary:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/arena_scene_pickup_no_stand/FrankaPickHardBench_20260206_json_benchmark_ep8/summary.json`
- Full real iTHOR preflight:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/preflight_real_ithor_pick_hard_all.json`
- Full real iTHOR exported Arena specs:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/arena_episode_specs_real_ithor_pick_hard.json`

### Failed attempts and lessons

- The bundled synthetic `benchmark_ithor_pick_hard_simple` was misleading for
  visual parity because an added bowl and an existing scene bowl could both be
  present. Real benchmark episodes exposed the actual issue more clearly.
- The first real benchmark preflight failed because scene-embedded pickups were
  not treated as Arena objects. Fix: wrap existing scene pickup prims.
- Initial pi remote smoke could not import `openpi_client` from the Isaac Lab
  virtualenv. Fix: installed the local OpenPI client package into that env.
- Arena DROID originally showed a separate stand that MolmoSpaces/MuJoCo did not.
  Fix: remove Arena's `Robot_Stand` for MolmoSpaces DROID runs and keep the
  robot root mount height.

### Current open risks

- Only one real episode has been visually checked in detail.
- The pi0.5 policy has only been smoke-tested for one step, not evaluated for
  task success.
- Camera visual differences remain due to renderer, robot model, and lighting
  differences; need to confirm they are within acceptable policy tolerance.
- Success-rate parity with MolmoSpaces/MuJoCo has not been measured yet.
- Some diagnostics are custom scripts; before merge, decide which are production
  helpers and which should remain developer-only.

### Next planned work

- Batch-render a small set of real iTHOR pick episodes in MuJoCo and Arena.
- Compare robot/object/camera parity for those episodes.
- Run longer pi0.5 rollouts on the fixed Arena setup.
- Compare Arena pi0-family success rate against MolmoSpaces/MuJoCo success rate.
- Add focused offline tests for episode conversion, especially scene pickups.

## 2026-05-02

### Completed

- Installed the forked OpenPI checkout needed by MolmoSpaces policy evals:
  `/home/horde/molmo-proj/openpi-omarrayyann`.
- Downloaded and served the documented `pi05_droid_jointpos` checkpoint from
  `/home/horde/molmo-proj/openpi_cache/openpi-assets/checkpoints/pi05_droid_jointpos`.
- Added Arena support for the forked joint-position policy path:
  absolute Franka joint-position actions, DROID camera observations, qpos
  observation conversion, and policy action repeat matching MolmoSpaces'
  500 ms policy interval.
- Calibrated Arena DROID root height to `z=0.445` after removing Arena's
  visual stand. This matches the MuJoCo TCP reset height for the same episode
  qpos much better than the previous `z=0.58` assumption.
- Preserved the external camera world height by pairing the calibrated root
  height with a DROID external camera local `z=0.795`.
- Restored the wrist camera to the MolmoSpaces gripper-relative transform after
  a compensating root-height attempt made the wrist image worse.
- Ran longer Arena `pi05_droid_jointpos` rollouts on episode 8 before and after
  root-height calibration. The calibrated run follows the MuJoCo-style motion
  more closely but still fails to lift/close.
- Ran MolmoSpaces/MuJoCo `pi05_droid_jointpos` on episode 8 at h36, h100, and
  h500; all failed. This shows episode 8 is useful for scene debugging but is
  not a reliable policy-success PoC target.
- Ran a representative MuJoCo h100 sweep over indices 0, 17, 34, 51, and 68;
  all failed with the same policy/checkpoint.
- Re-ran the full real iTHOR pick Arena preflight after the USD resolver and
  scene-pickup guard changes: `69 ready / 0 failed`.
- Matched raw DROID camera rendering to MolmoSpaces/OpenPI input shape:
  `624x352` render, then resize-with-padding to `224x224` before OpenPI.
- Added diagnostics for actual Isaac sensor camera poses and fixed diagnostic
  settling so joint-position embodiments hold the reset qpos instead of stepping
  to zero.
- Tested wrist-camera calibration candidates against the same MuJoCo comparison
  image. Strictly translating the MuJoCo MJCF local camera pose or the rendered
  `cam2world` pose into Arena made the wrist view worse because the MuJoCo
  gripper camera parent frame and Arena flattened DROID USD parent frame do not
  line up. The current default remains the best visual match among tested
  candidates, with env-var overrides left available for further calibration.
- Ran focused offline tests for conversion and policy adapter behavior:
  `5 passed`.
- Byte-compiled the Arena integration scripts and modified Arena bridge modules.
- Found the MolmoSpaces/MuJoCo policy decoding mismatch: the documented
  `pi05_droid_jointpos` checkpoint emits useful close signals below the old
  `0.5` binary threshold. With `chunk_size=15` and
  `grasping_threshold=0.01`, episode 8 succeeds in MolmoSpaces/MuJoCo.
- Updated the MolmoSpaces pi policy defaults and Arena pi remote adapter defaults
  to match the working `pi05_droid_jointpos` decode path.
- Added Arena HDF5 replay mode so a MuJoCo-successful trajectory can be replayed
  open-loop in Arena without OpenPI camera/observation differences.
- Replayed the successful MuJoCo episode 8 HDF5 in Arena and confirmed it still
  failed to lift before the gripper-actuator fix. That narrows the current
  blocker to Arena-side robot/action/contact parity.
- Added Arena debug output for arm-command tracking, gripper joint state, and
  finger-to-object distance.
- Patched the MolmoSpaces DROID Arena wrapper to use the fuller Robotiq actuator
  set used by Isaac Lab's graspable Franka+Robotiq config, so the driver joint
  and inner finger chain participate in the close motion.
- Found and fixed the scene-pickup dynamics bug: the iTHOR scene object root is
  named like `bowl_...`, but the actual rigid body child is named `Bowl_17`.
  The previous runtime physics patch compared only prim names, so it made the
  pickup rigid body kinematic. The patch now keeps any rigid body under the
  scene pickup path dynamic and applies sane mass/inertia attributes.
- Added an Arena FK probe against MuJoCo HDF5 states. It showed that Arena FK is
  close to MuJoCo at the successful late grasp state (`idx=462`: MuJoCo TCP
  distance `0.035 m`, Arena tool-center distance `0.027 m`), proving the static
  robot/object conversion is close enough for this episode.
- After the dynamic-pickup fix, a late-segment HDF5 replay initialized from
  MuJoCo row 440 succeeds in Arena, confirming the pickup can now move/lift.
- Replayed the full MuJoCo-successful episode 8 HDF5 trajectory in Arena from
  reset after the dynamic-pickup fix: success at Arena step 3220.
- Ran an online OpenPI `pi05_droid_jointpos` policy against the converted Arena
  episode 8 after the dynamics fix: success at Arena step 589.
- Added episode camera-spec propagation into `ArenaEpisodeSpec` and DROID camera
  patching so the Arena shoulder/external camera uses the benchmark's per-episode
  offset, quaternion, vertical FOV, and raw `624x352` render size.
- Confirmed the improved Arena shoulder image is visually close to the
  MolmoSpaces/MuJoCo shoulder image: bowl, counter/cabinet, gripper, and robot
  base all appear in the corresponding layout.
- Re-ran the full MuJoCo-successful episode 8 HDF5 trajectory in Arena after the
  camera-spec changes: success at Arena step 3220.
- Re-ran online OpenPI `pi05_droid_jointpos` after the camera-spec changes:
  failed after 5000 Arena steps. The object did not lift, and the gripper stayed
  far from the bowl for most of the rollout. Arm command tracking still worked,
  so the latest blocker is live-policy observation parity rather than basic
  Arena physics/action execution.
- Ran an isolation rollout that fed Arena's shoulder/external camera image as
  both the OpenPI shoulder and wrist image. It still failed after 3000 Arena
  steps, so the remaining live-policy gap is not explained by the wrist image
  alone.
- Matched the Arena remote OpenPI prompt formatting to MolmoSpaces `PI_Policy`
  by lowercasing and adding terminal punctuation when the episode task text does
  not already include it.
- Re-ran online Arena OpenPI after prompt formatting parity with the same
  `0.01` gripper threshold: failed after 5000 Arena steps, but moved closer to
  the bowl than the previous camera-spec rollout.
- Added a paired OpenPI-input diagnostic that compares MuJoCo video/HDF5-derived
  model inputs against Arena trace chunks, writes a montage, and optionally
  queries the same OpenPI server for first-action differences.
- Used the paired diagnostic to verify qpos parity at the policy boundary:
  episode 8 MuJoCo and Arena arm joint positions are exactly equal at reset.
  Remaining model-input differences are camera images and the gripper scalar.
- Tested three wrist-camera hypotheses:
  current episode-spec wrist, a world-position fit from MuJoCo `cam2world_gl`,
  and a direct `cam2world_gl` orientation interpretation. The direct orientation
  gives a wrist image that is visually closer to MuJoCo's cabinet-facing view,
  but it did not recover online policy success.
- Added an opt-out switch for Arena DROID camera patching
  (`MOLMO_ARENA_DROID_SKIP_CAMERA_PATCH=1`) so camera-parity changes can be
  compared against Arena's native DROID camera config without editing code.
- Re-ran online Arena OpenPI with Arena's native DROID camera defaults: failed
  after 5000 Arena steps.
- Re-ran online Arena OpenPI with prompt parity, current episode cameras, and
  `--pi_grasping_threshold 0.05`: success at Arena step 1683. This gives the
  current one-episode online pi0.5 PoC, but the threshold is an Arena adapter
  calibration and must be handled explicitly in benchmark parity reporting.
- Added a subprocess benchmark runner,
  `molmo_spaces_isaac/scripts/run_arena_benchmark_batch.py`, that launches one
  fresh Isaac process per episode and aggregates per-episode logs/results. This
  avoids the same-process multi-episode Isaac/Kit teardown issue observed when
  episode 0 completed and episode 8 hung during the next environment build.
- Verified the subprocess batch runner on the known PoC episode 8: success at
  step 1006 with `--pi_grasping_threshold 0.05`.
- Ran a representative isolated Arena online batch for indices
  `0,8,17,34,51,68`: `0/6` successes. All selected episodes reached the policy
  loop and wrote result JSON/logs, so this is a policy/parity blocker rather
  than a conversion/preflight blocker.
- Found the online policy stochasticity issue: OpenPI's served JAX policy
  advances an internal sampling RNG on every inference. Episode 8 can succeed or
  fail under the same Arena scene/reset depending on the server RNG state.
- Captured a post-batch failing trace for episode 8. Reset qpos and prompt match
  the successful trace, and first-frame camera inputs are nearly identical, but
  the first action chunk differs substantially (`mean_abs_diff ~= 0.111`),
  confirming action-noise sampling as a major variable.
- Added duplicate-index support to the subprocess batch runner so repeated
  commands like `--episode_indices 8,8,8` can estimate online policy
  stochasticity for a single converted episode without overwriting artifacts.
- Re-ran MolmoSpaces/MuJoCo episode 8 after the representative Arena batch; it
  succeeded in 32 steps with the current OpenPI server RNG state. Replaying that
  fresh successful HDF5 open-loop in Arena failed from reset even with a longer
  5000-step horizon, while initializing Arena at the successful HDF5 row 31
  qpos succeeded in 6 steps. This narrows that trajectory's Arena gap to
  approach-path/contact/controller divergence, not a missing scene conversion.
- Added configurable DROID mount xyz offsets through
  `MOLMO_ARENA_DROID_MOUNT_X/Y/Z` plus `MOLMO_ARENA_DROID_DISABLE_MOUNT_POSE`.
  A candidate offset `(0.026, -0.051, 0.462)` improved static FK parity for the
  fresh successful row 31, but broke the previously successful full-HDF5 replay
  when used as the default. The production default remains `(0.0, 0.0, 0.445)`.
- Added a deterministic OpenPI websocket wrapper,
  `molmo_spaces_isaac/scripts/serve_openpi_deterministic.py`, for paired
  MuJoCo/Arena debugging. It resets the OpenPI JAX RNG per websocket connection
  and can step the seed per connection for seed sweeps without reloading the
  checkpoint.
- Added `MOLMO_PI_SERVER_HOST` / `MOLMO_PI_SERVER_PORT` overrides to
  `PiPolicyConfig` so MolmoSpaces MuJoCo evals can point at deterministic or
  alternate OpenPI servers without adding a new eval config class.
- Verified the deterministic server health path, no-ping forked OpenPI client
  path, and Arena episode construction with the correct Isaac-ready USD asset
  root.
- Ran deterministic seed 0 on the converted Arena episode 8 with the fixed
  server/client setup: it completed the full 5000-step Arena horizon and failed
  cleanly, producing reusable OpenPI request traces.
- Ran matching deterministic seed 0 on MolmoSpaces/MuJoCo episode 8 for a
  200-policy-step horizon: it also failed. This makes seed 0 a policy-bad sample
  in both simulators, not an Arena-specific regression.
- Scanned deterministic seeds 1-3 on MuJoCo with a 120-step horizon and seeds
  4-10 with an 80-step horizon. None succeeded, so no Arena time was spent on
  those seeds as parity targets.
- Re-ran the full real iTHOR pick Arena preflight after the deterministic
  policy/debug changes: `69 ready / 0 failed`.

### Evidence

- Calibrated Arena root-height diagnosis:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/arena_mount_calibrated_default/FrankaPickHardBench_20260206_json_benchmark_ep8/summary.json`
- Current calibrated Arena diagnosis:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/arena_mount_calibrated_current/FrankaPickHardBench_20260206_json_benchmark_ep8/summary.json`
- Fresh MuJoCo vs Arena calibrated montage:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/mujoco_vs_arena_calibrated_current_ep8.png`
- Wrist calibration comparison montage:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/wrist_calibration_compare4_ep8.png`
- Current full real iTHOR preflight:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/preflight_real_ithor_pick_hard_all_current.json`
- Current full real iTHOR preflight after deterministic-server work:
  `/home/horde/molmo-proj/diagnostics/real_ithor_pick_ep8/preflight_real_ithor_pick_hard_all_after_deterministic_server.json`
- Arena calibrated policy rollout log:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_droid_jointpos_ep8_mount_calibrated_h100.log`
- MuJoCo episode 8 h500 policy run:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_h500/molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig/20260502_003302`
- MuJoCo policy-frame montage for episode 8:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_frames/mujoco_policy_frames_montage.png`
- Representative MuJoCo h100 sweep:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_representative_h100`
- MuJoCo episode 8 successful pi0.5 run after policy decode fix:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_chunk15_thresh001/PiPolicyEvalConfigChunk15Threshold001/20260502_014008`
- Successful MuJoCo HDF5 trajectory used for Arena replay:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_chunk15_thresh001/PiPolicyEvalConfigChunk15Threshold001/20260502_014008/house_17/trajectories_batch_1_of_1.h5`
- Arena FK probe summary:
  `/home/horde/molmo-proj/diagnostics/arena_fk_probe_ep8/FrankaPickHardBench_20260206_json_benchmark_ep8/summary.json`
- Latest successful Arena HDF5 replay log:
  `/tmp/isaaclab/logs/isaaclab_2026-05-02_03-10-00.log`
- Latest successful online Arena OpenPI rollout log:
  `/tmp/isaaclab/logs/isaaclab_2026-05-02_03-15-01.log`
- Latest successful Arena HDF5 replay after camera-spec changes:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_h5_replay_ep8_after_camera_results.json`
- Latest failed online Arena OpenPI rollout after camera-spec changes:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_after_camera_online_results.json`
- Latest Arena online OpenPI request traces:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_after_camera_online`
- Prompt-parity online Arena OpenPI result with `0.01` threshold:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_prompt_period_online_results.json`
- Arena default-camera online diagnostic result:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_arena_default_cameras_online_results.json`
- Wrist `cam2world_gl` direct-pose online diagnostic result:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_wrist_h5_direct_online_results.json`
- Successful online Arena OpenPI PoC with `--pi_grasping_threshold 0.05`:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_thresh005_online_results.json`
- Successful online Arena OpenPI traces:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_thresh005_online`
- Subprocess batch runner episode 8 success:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_batch_ep8_thresh005_results.json`
- Representative isolated Arena online batch:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_representative_isolated_thresh005_results.json`
- Representative batch per-episode artifacts:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_representative_isolated_thresh005_results_episodes`
- Post-batch failing episode 8 trace:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_post_batch_thresh005_trace`
- Post-batch failing episode 8 rollout:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_post_batch_thresh005_results.json`
- Post-batch MuJoCo episode 8 success:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_post_batch_rng/molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig/20260502_065200`
- Fresh MuJoCo-success HDF5 that failed full Arena replay from reset:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_post_batch_rng/molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig/20260502_065200/house_17/trajectories_batch_1_of_1.h5`
- Failed fresh-HDF5 Arena replay from reset:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_h5_replay_ep8_post_batch_mujoco_success_h5000_results.json`
- Fresh-HDF5 Arena FK probe with default mount:
  `/home/horde/molmo-proj/diagnostics/arena_fk_probe_ep8_fresh_success/FrankaPickHardBench_20260206_json_benchmark_ep8/summary.json`
- Fresh-HDF5 Arena FK probe with diagnostic mount xyz candidate:
  `/home/horde/molmo-proj/diagnostics/arena_fk_probe_ep8_fresh_success_mount_xyz_candidate/FrankaPickHardBench_20260206_json_benchmark_ep8/summary.json`
- Successful Arena replay initialized at fresh-HDF5 row 31 qpos:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_h5_replay_ep8_fresh_row31_init_mount_xyz_results.json`
- Failed online Arena OpenPI run with diagnostic mount xyz candidate:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_mount_xyz_thresh005_results.json`
- Failed known-old-success HDF5 replay after trying the diagnostic mount xyz
  candidate as the default:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_h5_replay_ep8_old_success_mount_xyz_default_results.json`
- Deterministic OpenPI seed-0 Arena rollout:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_deterministic_seed0_results.json`
- Deterministic OpenPI seed-0 Arena traces:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_deterministic_seed0`
- Deterministic OpenPI seed-0 MuJoCo h200 run:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_deterministic_seed0_h200`
- Deterministic OpenPI seed 1-3 MuJoCo h120 scans:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_deterministic_seed1_h120`,
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_deterministic_seed2_h120`,
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_deterministic_seed3_h120`
- Deterministic OpenPI seed 4-10 MuJoCo h80 scans:
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_deterministic_seed4_h80`
  through
  `/home/horde/molmo-proj/diagnostics/mujoco_pi_eval_ep8_deterministic_seed10_h80`
- Deterministic OpenPI server logs:
  `/home/horde/molmo-proj/diagnostics/openpi_deterministic_seed1.log`,
  `/home/horde/molmo-proj/diagnostics/openpi_deterministic_seed2.log`,
  `/home/horde/molmo-proj/diagnostics/openpi_deterministic_seed3.log`,
  `/home/horde/molmo-proj/diagnostics/openpi_deterministic_seed4_step1.log`
- Paired MuJoCo/Arena OpenPI-input comparison for the successful Arena run:
  `/home/horde/molmo-proj/diagnostics/pi_input_compare/ep8_thresh005_online/summary.json`
- Paired input montage for the successful Arena run:
  `/home/horde/molmo-proj/diagnostics/pi_input_compare/ep8_thresh005_online/policy_input_montage.png`
- Shoulder-as-wrist isolation result:
  `/home/horde/molmo-proj/diagnostics/pi_rollouts/arena_pi05_ep8_shoulder_as_wrist_results.json`
- Shoulder-as-wrist isolation traces:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_shoulder_as_wrist`
- Improved Arena shoulder camera image:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_after_camera_online/chunk_0000/exterior_image_1_left.png`
- Still-mismatched Arena wrist camera image:
  `/home/horde/molmo-proj/diagnostics/pi_traces/arena_ep8_after_camera_online/chunk_0000/wrist_image_left.png`

### Failed attempts and lessons

- Treating MuJoCo's `fr3_link0` mount height as the Arena DROID root height
  placed the Arena TCP too high. The correct Arena root value is a calibrated
  value for Arena's flattened DROID USD, not the MuJoCo model's internal mount
  offset.
- A wrist-camera transform that compensated for the root-height change reduced
  visual parity. The better current choice is to keep the MolmoSpaces
  gripper-relative wrist camera transform and continue debugging with paired
  wrist/external images.
- Direct use of the MuJoCo MJCF wrist camera pose
  (`pos="0.031 0.074 0.022"`, `euler="0.339710886 3.15302530717 0"`) is not
  correct in Arena because the parent frame is different.
- Fitting the Arena wrist camera to MuJoCo's rendered `cam2world` position and
  orientation also produced the wrong visual view. This suggests the remaining
  wrist issue should be solved by explicitly mapping the MuJoCo gripper base
  frame to the Arena USD `base_link` frame, not by copying camera numbers alone.
- Applying the shoulder/root-height Z compensation directly to the wrist
  camera's local Z axis was wrong: in the Arena Robotiq parent frame that moved
  the camera laterally rather than improving the view. The production default
  keeps the stable gripper-relative wrist transform with env-var overrides for
  further calibration.
- Replacing the wrist policy input with a duplicate of the shoulder policy input
  did not recover online policy success. Keep wrist calibration on the list, but
  also compare the full OpenPI input/action stream against MuJoCo.
- Exact MuJoCo reset `cam2world_gl` wrist pose does not directly solve Arena
  wrist parity because the flattened Arena Robotiq geometry occludes the camera
  differently from MolmoSpaces/MuJoCo. The remaining wrist work needs to account
  for robot mesh/frame mismatch, not only the camera matrix.
- Arena online success is sensitive to the binary gripper threshold. With the
  current Isaac-rendered observations, `0.01` closes too early for episode 8;
  `0.05` delays closing and succeeds. Treat this as a policy-adapter calibration
  to validate across more episodes, not as proof of benchmark-level parity.
- Same-process multi-episode Arena evaluation is not reliable enough for this
  migration. A run over `0,8,17,34,51,68` completed episode 0, then hung while
  constructing episode 8 in the same Isaac process. Use the subprocess batch
  runner for benchmark-scale Arena evaluation.
- The one-episode online OpenPI success is not deterministic. The stock OpenPI
  websocket server does not expose an episode reset for its JAX sampling RNG;
  it advances the RNG on each inference. Treat online pi0.5 results as
  stochastic unless using open-loop HDF5 replay or a deterministic serving mode.
- Episode 68 reached the policy loop but emitted RTX large-transform warnings
  for unrelated showerhead geometry in the bathroom scene. This is a scene USD
  quality/rendering risk to track separately from target pickup conversion.
- Policy failure on episode 8 is not enough to diagnose an Arena policy bug
  because the same policy also fails the same episode in MolmoSpaces/MuJoCo.
- That previous episode-8 conclusion was superseded after fixing the pi policy
  decode settings. Episode 8 is now MuJoCo-successful and is the active Arena
  policy/action parity target.
- Replaying a MuJoCo-successful HDF5 trajectory in Arena is a stronger diagnostic
  than an online OpenPI rollout: when it fails, the bug is below policy image
  observations and inside robot kinematics, action timing, gripper contact, or
  scene physics.
- For scene-embedded pickups, matching by scene object root name is not enough.
  Runtime physics fixes must match every rigid body under the scene object root
  path, otherwise the pickup wrapper can observe the object while the actual
  child body remains kinematic.
- A DROID root xyz offset can improve a single static FK row while hurting a
  full successful trajectory. Keep root-pose calibration evidence-based across
  reset, mid-approach, grasp, full HDF5 replay, and visual parity before
  changing defaults.
- A MuJoCo-successful stochastic trajectory is not automatically an Arena
  open-loop success, even when the terminal qpos can pick the object in Arena.
  Use row-init probes to separate terminal grasp/contact feasibility from
  approach-path/controller divergence.
- A deterministic policy seed is only a useful Arena parity target after it
  succeeds in MuJoCo. Seeds 0-10 tested so far do not satisfy that condition for
  episode 8, so the next paired test should start from a MuJoCo-success action
  sequence or continue searching deterministic policy samples with MuJoCo as the
  cheap filter.
- The Arena Pi remote path must import the forked `openpi_client` with disabled
  websocket ping keepalive for slow first inferences. In this workspace, set
  `MOLMO_OPENPI_VENV_SITE=/home/horde/molmo-proj/openpi-omarrayyann/.venv/lib/python3.11/site-packages`.

### Current open risks

- The one-episode PoC is complete for conversion and HDF5 replay, but not for
  online pi0-family policy parity. The current representative Arena online batch
  is `0/6`, and episode 8 success is stochastic under the stock OpenPI server.
- Controlled/repeated policy evaluation is needed because the stock OpenPI
  server's stochastic action sampling can flip episode 8 between success and
  failure without any scene conversion change.
- Deterministic low-seed scans have not yet found a MuJoCo-successful sample
  for episode 8. Until one is found, deterministic failures should not be treated
  as Arena regressions.
- The `0.05` Arena gripper threshold may be an acceptable adapter calibration
  for some sampled action chunks, but it is not enough to stabilize
  representative online success.
- The next observation-side risk is still wrist-camera parent-frame mapping if a
  deterministic or averaged policy comparison underperforms MolmoSpaces/MuJoCo.
- The shoulder camera now looks close to MuJoCo, but the wrist camera is not yet
  a proven extrinsic match. Keep wrist parity on the list, but do not attribute
  every online failure to wrist mismatch now that policy sampling stochasticity
  has been isolated.
- The object root Z reported by MuJoCo and Arena differs for episode 8, but the
  visual bowl/cabinet support relationship looks correct. Do not add a blind
  object Z offset without visual validation.
- The Arena remote OpenPI client now disables websocket ping keepalive to avoid
  first-inference timeout with the current server/client combination. Keep the
  application-level inference timeout in `PiRemotePolicy` as the safety bound.
- The Arena DROID robot model uses a flattened USD with Robotiq frames/joints
  that do not line up exactly with MolmoSpaces/MuJoCo `franka_droid`. The current
  fix path is to keep adapting through MolmoSpaces' Arena wrapper rather than
  editing Arena internals.
- Full-HDF5 replay is now a regression test for known-good trajectories, not a
  guarantee that every stochastic MuJoCo success will replay open-loop in Arena.
  Online policy parity still needs repeated or deterministic evaluation.

### 2026-05-07 idx14 policy I/O report

- Added a viewable idx14 MuJoCo-vs-Arena policy I/O report:
  `/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/report.html`.
- The report compares MuJoCo `traj_1` for original idx `14`
  (`7/10` MuJoCo slice success, selected rollout succeeds at row `88`) against
  the traced Arena subset idx `10` rollout (`0/10` Arena slice success, traced
  rollout failed after `1500` steps).
- Reset arm qpos matches exactly at the policy boundary. Reset gripper scalar
  is close but not identical (`abs diff ~= 0.00098`).
- Prompt formatting is not exact at the traced boundary: MuJoCo sends
  `pick up the smooth gray bowl`, while Arena sends
  `pick up the smooth gray bowl.`. This only differs by terminal punctuation,
  but it should be fixed before treating action differences as policy evidence.
- Reset camera inputs remain visually similar but not pixel-identical. The wrist
  camera is the larger image gap in this report (`mean abs ~= 55.3`, exterior
  `mean abs ~= 37.0`).
- MuJoCo raw OpenPI action chunks were not stored in the old HDF5, so the report
  compares saved MuJoCo decoded/applied actions against Arena decoded actions.
  A strict raw action-chunk diff still requires rerunning MuJoCo with matching
  trace hooks.

### Next planned work

- Keep episode 8 as a smoke regression: preflight and HDF5 replay should remain
  green while online policy parity is fixed. Treat online OpenPI as stochastic
  until the OpenPI RNG is controlled or success is measured over repeated runs.
- Use the subprocess batch runner with repeated indices, for example
  `--episode_indices 8,8,8`, to estimate stochastic online success for the same
  converted episode.
- Continue deterministic OpenPI seed/action search with MuJoCo as the filter,
  then run Arena only for MuJoCo-successful deterministic samples.
- Build a small trajectory-parity table for episode 8 that includes reset,
  mid-approach, and grasp rows from both the old successful HDF5 and the fresh
  stochastic-success HDF5. Track static FK distance, dynamic replay result, and
  whether the row-init probe succeeds.
- Run or schedule a benchmark-level MolmoSpaces/MuJoCo vs Arena success-rate
  comparison once the representative batch is stable.
- Add a small frame-mapping diagnostic for MuJoCo gripper base to Arena DROID
  `Robotiq_2F_85/base_link` so wrist-camera calibration is based on a parent
  frame transform instead of image-only tuning.
- Use the paired OpenPI-input diagnostic to compare successful and failed Arena
  traces, then decide whether the `0.05` threshold should remain an Arena-only
  adapter calibration or be replaced by tighter camera/robot visual parity.
- Keep production changes localized and move any temporary diagnostics into
  clearly documented helper scripts before review.
