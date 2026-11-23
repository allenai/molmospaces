# Sim-to-real system identification

Fits a molmospaces robot's joint dynamics and position-actuator gains so that replaying the
commands of a real-robot recording in sim reproduces the real joint trajectory. The robot
XMLs keep their nominal values; the fitted values live in a JSON that molmo_spaces applies
when the robot is added to a scene.

Currently: **`franka_droid`** (FR3 + Robotiq 2F85) →
[`molmo_spaces/resources/sysid/franka_droid.json`](../../molmo_spaces/resources/sysid/franka_droid.json),
wired up via `FrankaRobotConfig.sysid_params_file`.

## How it is used at load time

`Robot.apply_control_overrides` calls `molmo_spaces.robots.sysid.apply_sysid_params` when
`robot_config.sysid_params_file` is set. Order: XML nominal → sysid JSON →
`K_stiffness` / `K_damping` / `force_limit` (explicit config overrides still win).
To get the nominal XML robot: `FrankaRobotConfig(sysid_params_file=None)`.

## What is fitted

Per arm joint: actuator `kp`, `kv`, joint `frictionloss`, `armature` (28 parameters, all
joints jointly, `scipy.optimize.least_squares` on the stacked per-sample error, log-space for
positive scales). Joint `damping` is **not** fitted: for a position actuator it is the same
torque term as `kv` (−c·q̇) and cannot be separated; it stays at the XML value.

The fit has local minima (wrist joints can trade a stiff `kp` against near-saturating
friction), so `calibrate.py` runs several starts (nominal XML, the real controller's gains,
the committed JSON) and keeps the best.

Replay is closed loop: the recorded commanded targets `q_des` are held for one control
period (25 Hz) on the model built exactly as the envs build it (`base_scene.xml` physics
options, `add_robot_to_scene`, gravity compensation), starting at rest at the first measured
`q`. `real q[i+1]` is compared to the sim after command `i` (`--obs-offset 1`, the rcs
recording stores the observation before the action).

## Data (not in git)

Unpack the hand-off bundle `franka_fr3_sysid.zip` and copy the recording here:

```
projects/sysid/data/fr3_2026-09-02/real_joints.csv          # used by default
projects/sysid/data/fr3_2026-09-02/2026-09-02_part-0.parquet  # full rcs recording, optional (--data <dir>)
```

The recording: real FR3, 40 s multi-sine excitation around the DROID home pose, 25 Hz
joint-position targets through the libfranka impedance controller with
`kp=[100,100,100,100,75,150,50]`, `kd=[20,20,20,20,7.5,15,5]`, gripper open and fixed.
The bundle's README documents how it was recorded and how to record more.

## Re-running (e.g. after a MuJoCo upgrade)

```bash
python projects/sysid/scripts/validate.py            # no refit; exits 1 if mean RMS regressed > 0.05 deg
python projects/sysid/scripts/calibrate.py           # refit, overwrites the committed JSON (~10 min)
python projects/sysid/scripts/validate.py --plot q.png
```

The JSON records the MuJoCo version, git SHA, timestep/integrator, nominal params and
nominal/fitted RMS, so diffs of it show what an upgrade changed. `--out` writes elsewhere,
`--starts current` does only a warm-started refit.

Adding a robot: add an entry to `ROBOTS` in [`common.py`](common.py) and set
`sysid_params_file` on its config.

## Current result (MuJoCo 4.0.0, 2026-11-23)

| joint | kp | kv | frictionloss | armature | RMS nominal → fitted [deg] |
|---|---|---|---|---|---|
| fr3_joint1 | 96.1 | 21.7 | 0.81 | 0.62 | 1.83 → 0.09 |
| fr3_joint2 | 90.9 | 20.6 | 0.47 | 0.46 | 1.80 → 0.65 |
| fr3_joint3 | 97.2 | 20.7 | 0.65 | 0.60 | 1.95 → 0.09 |
| fr3_joint4 | 114.2 | 24.5 | 1.25 | 0.75 | 1.74 → 0.23 |
| fr3_joint5 | 56.1 | 6.0 | 0.43 | 0.15 | 0.80 → 0.18 |
| fr3_joint6 | 141.3 | 16.4 | 0.46 | 0.35 | 0.56 → 0.06 |
| fr3_joint7 | 316.1 | 38.0 | 2.46 | 1.03 | 0.95 → 0.06 |

Mean 1.37° → 0.19°. The fitted gains land close to the real controller's
(100/20, 75/7.5, 150/15) without being told, which is a good sign the model is right.

## Caveats

- **In-sample only**: one 40 s trajectory at one centre pose. No held-out run exists yet;
  record one with a different seed / centre and add it as a `--data` validation.
- **joint 7** is weakly identified (the "real-gains" start reaches the same RMS with
  kp≈59), and **joint 2** keeps 0.65° residual, likely a gravity/payload mismatch the joint
  parameters can't express (link masses are not fitted).
- The fitted gains are much softer than the nominal kp=4500, so the sim arm now lags
  joint targets like the real one does. Policies and planners tuned on the stiff arm may
  track worse; `FrankaRobotConfig(sysid_params_file=None)` restores the old behaviour.
