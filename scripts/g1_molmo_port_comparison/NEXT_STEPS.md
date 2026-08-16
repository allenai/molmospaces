# G1 merge — where it stands and what to do next

The goal: make `InteractiveShell.pick()` (and the native G1 pick pipeline)
run the **reference** grasp logic — the one verified equivalent to
`~/code/g1_molmo`'s own stack — instead of the independent rewrite that
diverged from it and never contacts the object.

**Status: done for the grasp.** `run_house_sweep.py 0` now reports
`fingers_in_contact=True`, `lift_height=0.120`, `pick success=True`. What
remains is navigation (§3) and the walking gap (§4).

---

## 1. How to check gold vs ported

### Run the two rollouts

```bash
# 1. Gold — the actual g1_molmo env.py + agents/policy.py stack
cd /Users/maxa/code/g1_molmo
conda run -n g1_molmo python molmospaces/scripts/g1_molmo_comparison/generate_gold_rollout.py \
    --seed 0 > /tmp/gold.txt 2>&1

# 2. Ported — molmospaces' g1_molmo_port/env_g1ms.py + the reference policy
cd /Users/maxa/code/molmospaces
conda run -n mlspaces python scripts/g1_molmo_port_comparison/generate_ported_rollout.py \
    --seed 0 > /tmp/ported.txt 2>&1
```

### Compare them

```bash
conda run -n mlspaces python scripts/g1_molmo_port_comparison/check_gold_parity.py \
    /tmp/gold.txt /tmp/ported.txt
```

**Expected: `PASS: 9/9 discrete invariants identical`**, ending in
`SUCCESS on episode 4`, with ~13/58 lines showing continuous-state drift.

### Why gold vs ported is NOT bit-identical (and never will be)

The two conda envs ship different physics engines:

| package | `g1_molmo` env | `mlspaces` env |
| --- | --- | --- |
| **mujoco** | **3.11.0** | **3.5.0** (what `pyproject.toml` pins) |
| numpy | 2.2.6 | 2.4.3 |

That diverges continuous state (`tcp_pos`, arm joint angles) at ~1e-3 over
~2300 steps. It is **not** a port defect. Don't chase it, and don't "fix" it by
loosening anything — if you want true bit-parity, the lever is aligning the
MuJoCo versions across the two envs, which is a real dependency decision
(`molmo-spaces` pins `mujoco~=3.5.0`).

What must match, and does: episode selection, target object, spawn pose
(bit-identical), per-episode `steps` / `sim_time` / `success` — including the
intermediate failures at 750 and 288 steps — and `SUCCESS on episode 4`
(`steps=2338 sim_time=12.04s`).

### The actual regression gate: ported vs ported

Because the ported stack **is** bit-reproducible within its own env, the strict
gate for any refactor is comparing the ported rollout against a baseline of
itself:

```bash
# record a baseline BEFORE touching anything
conda run -n mlspaces python scripts/g1_molmo_port_comparison/generate_ported_rollout.py \
    --seed 0 > /tmp/baseline.txt 2>&1

# ... make a change ...

conda run -n mlspaces python scripts/g1_molmo_port_comparison/generate_ported_rollout.py \
    --seed 0 > /tmp/after.txt 2>&1
conda run -n mlspaces python scripts/g1_molmo_port_comparison/check_gold_parity.py \
    /tmp/baseline.txt /tmp/after.txt --strict
```

**Expected: `PASS (strict): 58 trace lines byte-identical`.**

Run this after **every** step below. If it fails, stop and fix before
continuing — that is the whole reason the merge has stayed correct so far.

### Native pick (the thing actually being fixed)

The gate above only covers the reference path. The native pipeline — what
`InteractiveShell.pick()` runs — is measured by:

```bash
cd /Users/maxa/code/g1_molmo/molmospaces/scripts/g1_molmo_comparison
conda run -n mlspaces python run_house_sweep.py 0 1 2 5
```

Look for `lift_height` and `fingers_in_contact` per house. Current status:
**house 0 succeeds** (`lift_height=0.120`, `fingers_in_contact=True`); houses
1/2/5 end in `PHASE_DONE` because their targets are out of reach without
navigation — see §3.

> This script wraps task sampling in `try/except` and prints only the exception
> message, swallowing the traceback. When you hit an error, re-run the failing
> case with that handler bypassed (or in a REPL) to get the call site — this
> costs one minute and saves a lot of guessing.

### Scene textures / renders

Neither gate above looks at pixels, and the two stacks used to sample
*different* textures (gold: its repo-local `assets/textures/<Category>/`
pack; ported: pools reconstructed from THOR's `material-database.json`).
The pack now lives in the ResourceManager cache as
`$ASSETS_DIR/textures/fetchman/<Category>/*.png` and is mandatory —
`build_thor_texture_pools()` raises a `JORDI-TODO` error if it is missing,
rather than silently falling back. It is not yet a registered
`molmospaces_resources` source; until it is, unzip `textures.zip` into
`$ASSETS_DIR`.

```bash
cd /Users/maxa/code/g1_molmo && conda run -n g1_molmo python \
    /Users/maxa/code/molmospaces/scripts/g1_molmo_port_comparison/check_texture_parity.py \
    --stack gold --out /tmp/tex_gold
cd /Users/maxa/code/molmospaces && conda run -n mlspaces python \
    scripts/g1_molmo_port_comparison/check_texture_parity.py --stack ported --out /tmp/tex_ported
conda run -n mlspaces python scripts/g1_molmo_port_comparison/check_texture_parity.py \
    --compare /tmp/tex_gold /tmp/tex_ported
```

**Expected:** `PASS pool_basenames` (258 files / 5 categories), `PASS
applied_basenames` (25 files), `wrist_image` byte-identical, `head_image`
`mean|diff|≈0.09, max 25` — faint edge-antialiasing noise from the same
MuJoCo 3.11-vs-3.5 split as above, not a texture difference.

---

## 2. What is already done

Both stacks now live in `molmo_spaces` proper, each move verified with the
strict gate:

| commit | change |
| --- | --- |
| `474ad30`,`f379a3b`,`165dc55` | reference G1Robot → `molmo_spaces/robots/g1.py`, config-driven, real `Robot` subclass |
| `847b6ef` | waypoint→velocity nav bridge + `nav_demo.py` |
| `ea32230` | promoted to `robots/g1.py`; old rewrite → `robots/g1_old_reference.py` |
| `a248348` | **real bug fix** in the rewrite: persist pelvis height smoothing |
| `fc47c75` | reference policy → `policy/solvers/object_manipulation/g1_pick_policy.py` |
| `902abc7` | robot: `from_mj_data()` + `update_control()` |
| `fa1ef5e` | `G1PickPlannerPolicy` — native planner-policy wrapper |
| `7bd04e3` | `robot_model_root_name`/`reset`, dual `robot_view`, Controller ABC members |
| `ea4bb4c` | `check_gold_parity.py` + these notes |
| `f627598` | **the flip: native pick switched to the reference stack, and now succeeds** |

Key naming: **`robots/g1.py` is the reference-derived implementation**;
**`robots/g1_old_reference.py` is the superseded independent rewrite**, which
nothing constructs any more (see §3). They are NOT interchangeable —
`(mj_data, exp_config)` vs `(model, data, ...)`.

---

## 3. The flip: DONE — the native pick works

`G1Config` and `FetchmanPickPlannerPolicyConfig` now build the
reference-derived `robots/g1.py` + `G1PickPlannerPolicy` (commit `f627598`).
`InteractiveShell.pick()` picks these up automatically.

```
run_house_sweep.py 0
  start_z=0.760  final_z=0.880  lift_height=0.120
  fingers_in_contact=True  ->  pick success=True
  approach 0.226 -> descend 0.185 -> open_hold 0.149 -> close 0.140
  -> post_close 0.123 -> lift 0.122 m
```

Six gaps were closed, each from a real traceback:

1. `LegsWaistController.set_target` accepts both the flat 7-vector and the
   reference's `(cmd3, height, waist3)`. **This corrects the guess an earlier
   version of this file recorded**: the culprit was NOT `init_qpos`, it was
   `PickTaskSampler._randomize_robot_standing_height` calling the controller
   *directly*, bypassing `Robot.update_control`.
2. `G1Robot` applies `robot_config.physics_timestep` in `__init__` (the
   reference sets it in `set_env`, which the native env never calls).
3. `G1Robot.compute_control` mirrors `execute_action`'s dispatch — controller
   order is load-bearing.
4. `target_poses["grasp"]` published as a live 4x4 for `GraspPoseSensor`.
5. `goal_xy` from the controller's own `_xy()`, so `reset()` takes the
   already-at-goal branch instead of stalling in the walking phase.
6. The env view must **not** expose `grasp_frame_pose` — the reference
   branches on `hasattr`, and its absence is what selects the pick path.

### Remaining work

- **Navigation for pick.** Houses 1/2/5 end in `PHASE_DONE`: their targets are
  2.5–6.8 m away and `_build_info` always reports already-at-goal, so the
  planner correctly rejects them as out of reach. In the shell that is fine
  (`nav_to` is a separate command); for the batch datagen pipeline the info
  dict needs a real `goal_xy`/`goal_yaw` and a `nav_occupancy_map` so
  `_plan_path` can run.
- **Retire the rewrite.** Nothing constructs `FetchmanPickPlannerPolicy` now;
  rename it to `*_old_reference.py` alongside `robots/g1_old_reference.py`.
  Then `robots/g1_old_reference.py` itself, once nothing imports it.
- **Delete `g1_molmo_port/`** once its shims have no consumers.

---

## 4. Known open issues

- **G1 walking does not translate — and it is now localized.** Commanding
  `[0.5, 0, 0]` via `nav_demo.py` (reference robot + `controller_g1ms`) moves
  the robot ~0.115 m in 4 s. It stays upright and the command reaches the walk
  policy (`norm(cmd)=0.5 > 0.05`, so `_walk_sess` runs — verified).

  **But `mlspaces_tests/component_tests/test_g1_robot.py::
  test_walks_forward_on_command` passes**, asserting ~2.3 m in 4.5 s at the
  same `vx=0.5` — on `g1_old_reference` driven by the native
  `controllers/g1_walk.py::G1WalkController`.

  Same ONNX weights, same PD gains, same 4-tick decimation, opposite outcome.
  So the fault is in the **`controller_g1ms` path**, not the robot, the command
  plumbing, or the policy. That is a much smaller search space than before.
  Prime suspect: gait-clock ownership — native `G1WalkController` increments
  `_step_counter` itself inside `compute_ctrl_inputs` (`g1_walk.py:232`),
  whereas `controller_g1ms` takes it as an argument and relies on the *policy*
  to advance it (`_step_counter += 1` in `sample_actions`). Anything driving the
  reference controller outside `sample_actions` must call
  `G1Robot.advance_control_clock()` at exactly the right cadence, and a
  mismatch there would desynchronize the gait without breaking balance —
  precisely the observed symptom.

  Fastest next experiment: port `test_walks_forward_on_command` to the new
  robot (it needs `G1Robot.from_mj_data`) and bisect the clock cadence.

- **`robots/g1_old_reference.py` / `fetchman_pick_planner_policy.py`** still
  carry the three documented divergences (arm `actfrcrange`/`dof_damping`,
  action noise on the IK command, and — fixed in `a248348` — the non-persisted
  height smoothing). They are now dead code paths; retiring them is preferable
  to fixing them further.
