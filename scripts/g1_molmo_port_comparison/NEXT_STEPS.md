# G1 merge — where it stands and what to do next

The goal: make `InteractiveShell.pick()` (and the native G1 pick pipeline)
run the **reference** grasp logic — the one verified equivalent to
`~/code/g1_molmo`'s own stack — instead of the independent rewrite that
diverged from it and never contacts the object.

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
converging but **not** grasping (`fingers_in_contact=False`).

> This script wraps task sampling in `try/except` and prints only the exception
> message, swallowing the traceback. When you hit an error, re-run the failing
> case with that handler bypassed (or in a REPL) to get the call site — this
> costs one minute and saves a lot of guessing.

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

Key naming: **`robots/g1.py` is the reference-derived implementation**;
**`robots/g1_old_reference.py` is the superseded independent rewrite**, which
is still what `G1Config` constructs. They are NOT interchangeable —
`(mj_data, exp_config)` vs `(model, data, ...)`.

---

## 3. The next step: finish the flip

The flip is two lines, both currently reverted:

```python
# molmo_spaces/configs/robot_configs.py  — G1Config
from molmo_spaces.robots.g1 import G1Robot          # was: g1_old_reference
robot_factory: ... = G1Robot.from_mj_data           # was: G1Robot

# molmo_spaces/configs/policy_configs.py — FetchmanPickPlannerPolicyConfig
from ...g1_pick_policy import G1PickPlannerPolicy   # was: FetchmanPickPlannerPolicy
self.policy_cls = self.policy_factory = G1PickPlannerPolicy
```

Four blockers have been cleared already; each attempt got further:

1. ~~`NotImplementedError` — scene wouldn't compile~~ → added
   `robot_model_root_name`/`reset`
2. ~~`'G1RobotView' has no attribute 'get_move_group'`~~ → dual `robot_view`
3. ~~`'LeftArmController' has no attribute 'reset'`~~ → Controller ABC members
4. **CURRENT:** `ValueError: too many values to unpack (expected 3)` during
   task sampling

### Diagnosing blocker 4

`LegsWaistController.set_target` (`g1_molmo_port/components/controller_g1ms.py`)
unpacks `cmd, height_cmd, waist = target` — a 3-tuple. Something is handing it
a longer sequence. `G1Robot.update_control` is **not** the culprit; it builds a
correct 3-tuple from the native 7-vector.

Prime suspect: **`G1Config.init_qpos["legs_waist"]` is a 15-element joint-position
array**, while the WBC's `legs_waist` target is a 7-element *command*
`[vx, vy, yaw_rate, height, waist(3)]`. The same move-group name means two
different things to the two stacks, and reset/init paths push `init_qpos`
through it.

That is a semantic decision, not a patch: decide whether

- (a) the reference robot's `reset()`/init path should bypass `init_qpos`
  entirely (it already uses `DEFAULT_QPOS` via `set_defaults()`), and whatever
  still forwards `init_qpos` into `set_target` should be taught not to; or
- (b) `LegsWaistController.set_target` should accept both arities explicitly.

(a) is cleaner — `init_qpos` is a *pose*, and a velocity-commanded move group
has no meaningful "initial joint position". Get the traceback first (see the
note in §1) to confirm which call site is doing it.

### After the flip works

Run **both** checks — the strict ported gate (must stay byte-identical) **and**
`run_house_sweep.py 0 1 2 5`. Then:

- `InteractiveShell.pick()` needs no change: it already defaults to
  `FetchmanPickPlannerPolicyConfig` for G1 in WBC mode, which the flip
  repoints at the reference policy.
- Retire the rewrite: rename `fetchman_pick_planner_policy.py` →
  `*_old_reference.py` once nothing imports it (only `policy_configs.py` does).
- Delete `g1_molmo_port/` once its shims have no consumers.

---

## 4. Known open issues (not blockers for the above)

- **G1 walking does not translate.** Commanding `[0.5, 0, 0]` moves the robot
  ~0.115 m in 4 s of sim; it stays upright and the command reaches the walk
  policy (`norm(cmd)=0.5 > 0.05`, so `_walk_sess` runs — verified). Reproduce
  with `nav_demo.py`. **Pre-existing** — `controller_g1ms.py` is untouched by
  this whole merge, and the pick rollout never exercises walking because
  `place_robot_near` spawns the robot 0.2–0.5 m from the target. Next
  diagnostic: command pure yaw; if it turns but won't translate, the policy is
  live and forward locomotion specifically is broken.

- **The rewrite's grasp still misses**, even after `a248348`. TCP now converges
  (descend 0.647→0.161 m, close 0.569→0.097 m, vs gold 0.131 m) and the bowl is
  no longer knocked aside, but `fingers_in_contact=False`. The TCP settles ~5 cm
  above the object centre where the reference settles ~1.6 cm. Remaining known
  divergence: this file composes **world-frame** grasps and doubles them via
  `flip_grasps`, where the reference uses **object-local** transforms composed
  as `Tw @ go` (`G1PickPlannerPolicy._build_info` already does it the reference
  way). If the flip lands, this becomes moot — the reference policy replaces
  this code path entirely.
