# Sensors

This page documents the **sensor system**: the abstraction MolmoSpaces uses to
turn raw simulator state into the observation dicts that policies consume and
that data generation writes to disk.

If you're reading this because you're writing a new task or a new robot and
need to decide which sensors to attach, the short version is:

!!! tip "New users: start from `get_core_sensors()`"
    For almost any new task you should compose your sensor suite by starting
    from [`get_core_sensors(exp_config)`][molmo_spaces.env.sensors.get_core_sensors]
    and then extending it with the few task-specific sensors you actually
    need. **Do not** copy or extend
    [`get_rby1_door_opening_sensors`][molmo_spaces.env.rby1_sensors.get_rby1_door_opening_sensors],
    [`get_nav_task_sensors`][molmo_spaces.env.sensors.get_nav_task_sensors],
    or any other ad-hoc bundle — they exist for backward compatibility and
    bake in legacy assumptions (e.g. robot name checks) that you almost
    certainly do not want.

The rest of this page explains how the system fits together.

## How sensors work

A **sensor** is a small object that, given the env and the current task, returns
one piece of an observation. Every sensor inherits from
[`Sensor`][molmo_spaces.env.abstract_sensors.Sensor] (in
`molmo_spaces/env/abstract_sensors.py`):

```python
class Sensor(ABC):
    uuid: str                       # unique identifier, used as the obs dict key
    observation_space: gym.Space    # gymnasium space describing the output
    is_dict: bool = False           # if True, output is a dict that will be JSON-encoded
    str_max_len: int = 2000         # padding length for the JSON byte buffer

    @abstractmethod
    def get_observation(self, env, task, batch_index: int = 0, ...): ...

    def reset(self) -> None: ...    # optional, override if the sensor has state
```

### The `uuid` becomes the observation key

A sensor's `uuid` is the string key it occupies in the observation dict that
`task.step()` returns and in the HDF5 file that data generation produces.
Two sensors in the same suite are not allowed to share a `uuid` — the
[`SensorSuite`][molmo_spaces.env.abstract_sensors.SensorSuite] constructor and
`add()` method both assert uniqueness.

### `is_dict` and `str_max_len`

There are two flavors of sensor output:

- **Plain array sensors** (`is_dict = False`) return a `np.ndarray` whose shape
  and dtype match `observation_space`. Camera RGB, depth, TCP pose, etc. are
  all in this category.
- **Dict sensors** (`is_dict = True`) return a (possibly nested) Python `dict`
  of JSON-serializable values. At save time this dict is `json.dumps`'d,
  UTF-8 encoded, and packed into a fixed-length `np.uint8` buffer of length
  `str_max_len` (right-padded with `\x00`). The corresponding
  `observation_space` is always a `Box(0, 255, (str_max_len,), uint8)`.

The encoding round-trip happens in
[`save_utils.dict_to_byte_array`][molmo_spaces.utils.save_utils.dict_to_byte_array]
/ `byte_array_to_string`; see [Saving](#how-sensors-relate-to-saving) below.
A warning is logged if your JSON exceeds `str_max_len` and gets truncated, so
if you're storing variable-length data (e.g. per-object dicts) pick
`str_max_len` generously.

### `reset()`

Sensors that maintain state across calls — for example
[`LastCommandedRelativeJointPosSensor`][molmo_spaces.env.sensors.LastCommandedRelativeJointPosSensor]
(needs the previous joint pos),
[`ObjectStartPoseSensor`][molmo_spaces.env.sensors.ObjectStartPoseSensor]
(caches the first observed pose), and
[`GraspStateSensor`][molmo_spaces.env.sensors.GraspStateSensor] (caches
gripper/object geom ids) — override `reset()`. The base
`BaseMujocoTask.reset()` calls `reset()` on every sensor in the suite, so
state never leaks between episodes.

### `SensorSuite`

A [`SensorSuite`][molmo_spaces.env.abstract_sensors.SensorSuite] is just an
`OrderedDict[str, Sensor]` with a couple of helpers:

- `get_observations(env, task, batch_index=...)` calls every sensor's
  `get_observation(...)` and returns the resulting dict.
- `add(sensor)` / `extend(sensors)` append to the suite while enforcing the
  `uuid` uniqueness invariant.

The task's `get_observations()` invokes the suite once per env in the batch
(`n_batch` is typically 1; see the warning in
[Key Concepts](concepts.md#env)).

## Where sensors are registered

Sensor registration is split across the three layers that own the data
(this was the result of a recent refactor — see commit history for details).
Each layer contributes the sensors it knows about; the task composes them
into the final suite at construction and policy-registration time.

### 1. Task — task-specific sensors

Each `BaseMujocoTask` subclass implements
[`_create_sensor_suite_from_config(exp_config)`][molmo_spaces.tasks.task.BaseMujocoTask._create_sensor_suite_from_config]
(which is `@abstractmethod`). This is where the **core sensor bundle** plus any
**task-specific sensors** get instantiated. The recommended pattern is:

```python
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors import GraspStateSensor, ObjectStartPoseSensor, get_core_sensors

class MyTask(BaseMujocoTask):
    def _create_sensor_suite_from_config(self, config):
        sensors = get_core_sensors(config)
        sensors.extend([
            ObjectStartPoseSensor(object_name=config.task_config.pickup_obj_name,
                                  uuid="obj_start"),
            GraspStateSensor(object_name=config.task_config.pickup_obj_name,
                             uuid="grasp_state_pickup_obj"),
        ])
        return SensorSuite(sensors)
```

[`PickTask`][molmo_spaces.tasks.pick_task.PickTask] and
[`PickAndPlaceTask`][molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask]
are good reference implementations.

### 2. Robot — robot-specific sensors

Each `Robot` subclass overrides
[`create_robot_sensors()`][molmo_spaces.robots.abstract.Robot.create_robot_sensors],
which returns a list of sensors that only make sense for that robot. Currently:

| Robot | Adds |
|---|---|
| `FrankaRobot`, `MobileFrankaRobot`, `I2rtYamRobot`, `FloatingRUMRobot` | `TCPPoseSensor(uuid="tcp_pose")` |
| `BimanualYamRobot` | `TCPPoseSensor(uuid="tcp_pose_left")`, `TCPPoseSensor(uuid="tcp_pose_right")` |
| `RBY1` | `RBY1GraspStateSensor(uuid="rby1_left_grasp_state", ...)`, `RBY1GraspStateSensor(uuid="rby1_right_grasp_state", ...)` |

`BaseMujocoTask.__init__` calls `current_robot.create_robot_sensors()` and
`extend`s the suite with them, so tasks never have to think about
robot-specific sensors directly.

### 3. Policy — policy-specific sensors

Each `BasePolicy` subclass overrides
[`create_policy_sensors()`][molmo_spaces.policy.base_policy.BasePolicy.create_policy_sensors].
The defaults are:

| Policy class | Adds |
|---|---|
| `BasePolicy` | nothing |
| `PlannerPolicy` | `PolicyPhaseSensor(uuid="policy_phase")`, `PolicyNumRetriesSensor(uuid="policy_num_retries")` |
| `BaseObjectManipulationPlannerPolicy` | `GraspPoseSensor(uuid="grasp_pose")` |

These are attached when the policy is bound to the task via
[`task.register_policy(policy)`][molmo_spaces.tasks.task.BaseMujocoTask.register_policy].
A task may only have one policy registered over its lifetime — re-registration
raises `ValueError`.

### Composition order

The final suite for a typical episode is built up in this order:

```
BaseMujocoTask.__init__:
    suite = task._create_sensor_suite_from_config(exp_config)   # task sensors (incl. core)
    suite.extend(robot.create_robot_sensors())                  # robot sensors

task.register_policy(policy):
    suite.extend(policy.create_policy_sensors())                # policy sensors
```

Putting `register_policy` after construction is important because, e.g.,
`PolicyPhaseSensor` and `GraspPoseSensor` reference the bound policy via
`task._registered_policy` and would fail otherwise.

### Opting out

If `exp_config.task_config.use_sensors = False`, `BaseMujocoTask` skips
creation of the suite entirely (`self._sensor_suite = None`) and
`get_observations()` returns an empty dict per env. `register_policy` then
becomes a no-op for sensor extension. This codepath exists for cases like
unit tests or pure physics-only rollouts.

## The core sensor suite

[`get_core_sensors(exp_config)`][molmo_spaces.env.sensors.get_core_sensors] is
the recommended starting point for any new task. It is *task-, policy-, and
robot-agnostic* — anything that would be specific to one of those should be
contributed by the corresponding `create_*_sensors` hook, not by extending or
copying this function.

It builds:

| Category | Sensors |
|---|---|
| **Cameras** (per `camera_spec` in `exp_config.camera_config.cameras`) | [`CameraParameterSensor`][molmo_spaces.env.sensors_cameras.CameraParameterSensor] (`sensor_param_{name}`), [`CameraSensor`][molmo_spaces.env.sensors_cameras.CameraSensor] (`{name}`), and conditionally [`DepthSensor`][molmo_spaces.env.sensors_cameras.DepthSensor] (`{name}_depth`) when `camera_spec.record_depth` is true |
| **Robot proprioception** | [`RobotJointPositionSensor`][molmo_spaces.env.sensors.RobotJointPositionSensor] (`qpos`), [`RobotJointVelocitySensor`][molmo_spaces.env.sensors.RobotJointVelocitySensor] (`qvel`), [`RobotBasePoseSensor`][molmo_spaces.env.sensors.RobotBasePoseSensor] (`robot_base_pose`) |
| **Environment state** | [`EnvStateSensor`][molmo_spaces.env.sensors.EnvStateSensor] (`env_states`), [`TaskInfoSensor`][molmo_spaces.env.sensors.TaskInfoSensor] (`task_info`) |
| **Actions** | [`LastActionSensor`][molmo_spaces.env.sensors.LastActionSensor] (`actions/commanded_action`), [`LastCommandedJointPosSensor`][molmo_spaces.env.sensors.LastCommandedJointPosSensor] (`actions/joint_pos`), [`LastCommandedRelativeJointPosSensor`][molmo_spaces.env.sensors.LastCommandedRelativeJointPosSensor] (`actions/joint_pos_rel`), [`LastCommandedEETwistSensor`][molmo_spaces.env.sensors.LastCommandedEETwistSensor] (`actions/ee_twist`), [`LastCommandedEEPoseSensor`][molmo_spaces.env.sensors.LastCommandedEEPoseSensor] (`actions/ee_pose`) |
| **Object tracking** | [`ObjectImagePointsSensor`][molmo_spaces.env.sensors.ObjectImagePointsSensor] (`object_image_points`) — samples in-mask pixel coordinates per camera for the task objects returned by `task.get_task_objects()` |

A few notes:

- The action sensors all return `{}` when `task.is_terminal()`. This is used
  as the **sentinel `done` action** at the tail of every trajectory — see
  [Data Format](data_format.md#data-postprocessing) for what consumers do
  with this.
- `LastCommandedRelativeJointPosSensor` and `LastCommandedEETwistSensor` need
  the previous joint pos / leaf pose, so they ship a dummy zero observation
  on the first step.
- Move groups that aren't position-commanded (e.g. velocity-controlled
  grippers with mismatched action/state dims) are silently dropped from the
  relative-action sensor output.
- `ObjectImagePointsSensor` falls back to the legacy
  `task_config.pickup_obj_name` / `place_receptacle_name` lookup if a task
  doesn't override `get_task_objects()`. For new tasks, prefer overriding
  `get_task_objects()` (see
  [`BaseMujocoTask.get_task_objects`][molmo_spaces.tasks.task.BaseMujocoTask.get_task_objects]).

## Task-specific sensors

These are added by individual tasks on top of the core suite. Notable examples:

| Task | Adds (on top of `get_core_sensors`) |
|---|---|
| [`PickTask`][molmo_spaces.tasks.pick_task.PickTask] | `ObjectStartPoseSensor(uuid="obj_start")`, `GraspStateSensor(uuid="grasp_state_pickup_obj")`, `PickupObjGoalPoseSensor(uuid="obj_end")` |
| [`PickAndPlaceTask`][molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask] | `ObjectStartPoseSensor`, `GraspStateSensor` for both the pickup object **and** the place receptacle |
| [`OpeningTask`][molmo_spaces.tasks.opening_tasks.OpeningTask] | Inherits the `PickTask` suite (since opening an articulated object is structurally similar to picking) |
| [`DoorOpeningTask`][molmo_spaces.tasks.opening_tasks.DoorOpeningTask] | Uses `get_rby1_door_opening_sensors` — see warning below |
| [`NavToObjTask`][molmo_spaces.tasks.nav_task.NavToObjTask] | Uses `get_nav_task_sensors` — see warning below |
| [`MultiTask`][molmo_spaces.tasks.multi_task.MultiTask] | Shares its child task's suite; its own `_create_sensor_suite_from_config` is a stub that returns `SensorSuite(get_core_sensors(...))` |

Some commonly-reused task-side sensors:

- [`ObjectPoseSensor`][molmo_spaces.env.sensors.ObjectPoseSensor] — pose of one
  or more named objects, expressed relative to the robot base. Used by
  navigation and door opening, output is a dict (`is_dict = True`).
- [`ObjectStartPoseSensor`][molmo_spaces.env.sensors.ObjectStartPoseSensor] —
  caches the object's pose at the start of the episode. For pick / open /
  close tasks it short-circuits to `task.config.task_config.pickup_obj_start_pose`.
- [`GraspStateSensor`][molmo_spaces.env.sensors.GraspStateSensor] — per-gripper
  `{"touching": bool, "held": bool}` heuristic based on MuJoCo contact pairs.
- [`DoorStateSensor`][molmo_spaces.env.sensors.DoorStateSensor] — joint angle,
  opening percentage, handle position/extents (door-task only).
- [`PickupObjGoalPoseSensor`][molmo_spaces.tasks.pick_task.PickupObjGoalPoseSensor]
  (lives next to `PickTask`) — the target end pose for the picked object.

## Policy-specific sensors

These are attached when a policy is registered with the task:

- [`PolicyPhaseSensor`][molmo_spaces.env.sensors.PolicyPhaseSensor] (`policy_phase`)
  — integer index of the current phase in
  `policy.get_all_phases()`. Added by every `PlannerPolicy`.
- [`PolicyNumRetriesSensor`][molmo_spaces.env.sensors.PolicyNumRetriesSensor]
  (`policy_num_retries`) — `policy.retry_count`. Added by every
  `PlannerPolicy`.
- [`GraspPoseSensor`][molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy.GraspPoseSensor]
  (`grasp_pose`) — the planned grasp pose in 7D. Added by
  `BaseObjectManipulationPlannerPolicy` and lives in the same module as the
  policy.

For learned / inference policies that don't carry phase or grasp metadata,
`create_policy_sensors()` returns `[]` and nothing is appended.

## Robot-specific sensors

Robot-specific sensors are added in
[`create_robot_sensors()`][molmo_spaces.robots.abstract.Robot.create_robot_sensors]
on each `Robot` subclass. Today these are mostly TCP poses for arms; see
[Where sensors are registered](#2-robot-robot-specific-sensors) above for the
full table.

If you're adding a new robot, register here any sensor that is meaningless on
other robots (e.g. a specific gripper's contact state). Don't add it to the
core sensor suite.

## How sensors relate to saving

The full path from "sensor returns a numpy array" to "value on disk in HDF5"
runs through three stages.

### 1. Per-step accumulation

`BaseMujocoTask.step()` calls `task.get_observations()`, which in turn calls
`SensorSuite.get_observations(env, task, batch_index)`. The resulting dict
(keyed by sensor `uuid`) is appended to `task.observation_cache` along with
rewards, terminals, etc. After the episode ends, `task.get_history()` packs
all of these into a single dict.

### 2. `prepare_episode_for_saving`

The data-generation pipeline (`molmo_spaces/data_generation/pipeline.py`,
function `save_house_trajectories`) takes the per-episode history dict and
hands it to
[`prepare_episode_for_saving`][molmo_spaces.utils.save_utils.prepare_episode_for_saving].
This step:

1. Flattens the per-timestep list of dicts (one entry per env in the batch;
   we only support `n_batch=1` for saving).
2. **Saves all camera videos before batching**, via
   [`save_videos_from_raw_observations`][molmo_spaces.utils.save_utils.save_videos_from_raw_observations].
   Each camera (RGB and depth) becomes its own MP4 file in the house output
   directory. This is a memory optimization: RGB/depth frames are ~80% of an
   episode's memory, and saving them out before the `torch.stack` step in
   batching avoids the giant transient tensor copies.
3. **Removes the camera sensor keys** from the in-memory observation dicts
   so they're not also batched as tensors. Camera sensors are identified by
   [`is_camera_sensor`][molmo_spaces.utils.save_utils.is_camera_sensor],
   which checks the suite for `CameraSensor` / `DepthSensor` instances (with
   a name-based fallback).
4. Calls [`batch_observations`][molmo_spaces.utils.save_utils.batch_observations]
   to transpose `List[dict]` to `Dict[uuid, Tensor(T, ...)]`. As part of this,
   `convert_to_arr` looks up `is_dict` and `str_max_len` on each sensor and
   JSON-encodes dict sensors into fixed-length `uint8` buffers
   (right-padded with `\x00`).
5. Appends `rewards`, `terminals`, `truncateds`, `successes`, and
   `obs_scene` (JSON string).

The fact that all of this works correctly — for both plain-tensor and
dict-encoded sensors — depends on the `SensorSuite` being available so that
`is_dict` and `str_max_len` can be looked up by uuid. This is why
`prepare_episode_for_saving` takes a `sensor_suite` argument.

### 3. `save_trajectories`

[`save_trajectories`][molmo_spaces.utils.save_utils.save_trajectories] writes
the batched episode dict to an HDF5 file with the layout documented in
[Data Format](data_format.md). The relevant routing is:

| Episode key | Goes to HDF5 path | Notes |
|---|---|---|
| `qpos`, `qvel` | `traj_{i}/obs/agent/{name}` | from `RobotJointPositionSensor` / `RobotJointVelocitySensor` |
| `actions/*` | `traj_{i}/actions/{name}` | strips the `actions/` prefix |
| `sensor_param_{cam}` | `traj_{i}/obs/sensor_param/{cam}/{intrinsic_cv,extrinsic_cv,cam2world_gl}` | one group per camera |
| Camera sensors (`{cam}`, `{cam}_depth`) | `traj_{i}/obs/sensor_data/{cam}` | dataset value is the **byte-encoded MP4 filename**, not the frames themselves; the file lives next to the HDF5 |
| `env_states` | `traj_{i}/env_states/{actors,articulations}` | JSON byte buffer is decoded and re-bucketed |
| Anything in the `extra_sensor_mapping` allowlist | `traj_{i}/obs/extra/{target_name}` | This is an **explicit allowlist** in `_save_extra_data_from_batched` — see warning below |
| `rewards`, `terminated`, `truncated`, `success`, `fail`, `obs_scene` | `traj_{i}/{name}` | rest is metadata |

!!! warning "The `extra_sensor_mapping` allowlist"
    `_save_extra_data_from_batched` has a hard-coded dict that maps sensor
    uuids to HDF5 dataset names. If you register a brand-new sensor with a
    uuid that isn't in this dict and that isn't a camera / camera-param /
    action / qpos / qvel / env_states / metadata sensor, **your sensor's data
    will not be written to disk** — it'll happily flow through the in-memory
    observation pipeline but be silently dropped at save time. If you add a
    new task-side sensor that should be persisted, add its uuid to
    `extra_sensor_mapping` in `molmo_spaces/utils/save_utils.py`.

### Quick mental model

```
sensor.get_observation()                         # numpy / dict / nested dict
  └── task.get_observations()                    # dict[uuid -> obs]
        └── observation_cache (per-step list)
              └── prepare_episode_for_saving():
                    ├── save_videos_from_raw_observations()  # MP4 to disk
                    ├── strip camera keys
                    └── batch_observations()                  # → Dict[uuid, Tensor(T, ...)]
                          └── save_trajectories()             # HDF5 on disk
```

## Adding a new sensor

The minimal recipe:

1. **Subclass `Sensor`** and place where appropriate.
2. **Pick a uuid** that doesn't collide with anything in the core suite.
3. **Set `is_dict` and `str_max_len`** if your output is a (nested) dict.
   Otherwise set `observation_space` to a `gym.spaces.Box` that matches the
   array you return.
4. **Implement `get_observation(env, task, batch_index=0, ...)`**. Don't
   raise on missing state silently — either raise loudly, or return a
   well-defined sentinel that downstream consumers can detect.
5. **Implement `reset()`** if you cache anything across calls.
6. **Register it** in the right place:
    - Task-specific → extend the list returned by your task's
      `_create_sensor_suite_from_config`.
    - Robot-specific → extend the list returned by your robot's
      `create_robot_sensors`.
    - Policy-specific → extend the list returned by your policy's
      `create_policy_sensors`.
7. **Add the uuid to `extra_sensor_mapping`** in
   `molmo_spaces/utils/save_utils.py` if you want it persisted to the
   `obs/extra/...` group of the HDF5 file. (Camera / action / qpos sensors
   are routed automatically.)

When in doubt, look at how [`PickTask`][molmo_spaces.tasks.pick_task.PickTask]
composes its suite — it's the smallest end-to-end example that uses the
recommended `get_core_sensors()` + task-side extensions pattern.
