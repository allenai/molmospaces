# Key Concepts

This page explains the core abstractions in MolmoSpaces and how they compose.

## Robot System

The robot abstraction is a three-layer hierarchy: **Move Groups** are assembled into a **Robot View**, which is held by a **Robot**.

### Move Group

A **move group** is the atomic unit of robot control: a named collection of MuJoCo joints and actuators that move together.
Each move group knows its joint/actuator IDs, its slice of `qpos`/`qvel`/`ctrl`, and can compute its own frame transforms and Jacobian.

Crucially, move groups **abstract away the underlying MuJoCo actuators and joint names**.
The number of joints and actuators in a group may differ — for example, a gripper may have 2 joints but only 1 actuator (mirrored/coupled), or a free joint has 7 DoF in `qpos` but 6 in `qvel`.
Some groups have passive (unactuated) joints.
In extreme cases, actuators can be entirely "faked" — `FloatingRUMBaseGroup` reports 7 actuators and a working `ctrl` property, but has no corresponding MuJoCo actuators at all; it reads and writes a mocap body pose instead.
The rest of the system doesn't need to know any of this: it interacts with move groups through `joint_pos`, `ctrl`, and `noop_ctrl` regardless of what's happening underneath.

**Base class:** [`MoveGroup`][molmo_spaces.robots.robot_views.abstract.MoveGroup]

**Specializations:**

| Class | Purpose |
|-------|---------|
| [`SimplyActuatedMoveGroup`][molmo_spaces.robots.robot_views.abstract.SimplyActuatedMoveGroup] | 1:1 mapping between joints, actuators, and position/velocity addresses |
| [`GripperGroup`][molmo_spaces.robots.robot_views.abstract.GripperGroup] | Adds gripper-specific controls (`set_gripper_ctrl_open`, `is_open`, `inter_finger_dist`) |
| [`RobotBaseGroup`][molmo_spaces.robots.robot_views.abstract.RobotBaseGroup] | Represents the robot's pose in the world |
| [`MocapRobotBaseGroup`][molmo_spaces.robots.robot_views.abstract.MocapRobotBaseGroup] | Fixed teleportable base (e.g. tabletop Franka) |
| [`FreeJointRobotBaseGroup`][molmo_spaces.robots.robot_views.abstract.FreeJointRobotBaseGroup] | Full 6-DoF free joint base |
| [`HoloJointsRobotBaseGroup`][molmo_spaces.robots.robot_views.abstract.HoloJointsRobotBaseGroup] | Virtual x, y, theta (holonomic) |

**Mixin:**

| Class | Purpose |
|-------|---------|
| [`MJCFFrameMixin`][molmo_spaces.robots.robot_views.abstract.MJCFFrameMixin] | A move group whose leaf frame is represented by a body or site in the MJCF model|

#### SimplyActuatedMoveGroup

The base `MoveGroup` makes no assumptions about the relationship between joints and actuators — a group can have more joints than actuators (e.g. a mirrored gripper), or joints whose `qpos` dimension differs from their `qvel` dimension (free and ball joints). `SimplyActuatedMoveGroup` narrows this: every joint is a simple 1-DoF hinge or slide, and there is exactly one actuator per joint. This means `n_joints == pos_dim == vel_dim == n_actuators`, and the internal ID/address lists can be safely exposed as public properties (`joint_ids`, `actuator_ids`, `joint_posadr`, `joint_veladr`). Groups like the RBY1 torso or Franka FR3 arm extend `SimplyActuatedMoveGroup` directly.

#### MJCFFrameMixin

Most move groups define their leaf frame as a specific element in the MJCF model — either a MuJoCo **site** or **body**. `MJCFFrameMixin` captures this pattern: subclasses implement `leaf_frame_id` (the integer ID) and `leaf_frame_type` (`"site"` or `"body"`), and the mixin provides a default `get_jacobian()` that dispatches to `mj_jacSite` or `mj_jacBody` accordingly. All arm groups, gripper groups, and the RBY1 torso/head use this mixin. The base groups (`FreeJointRobotBaseGroup`, `HoloJointsRobotBaseGroup`) do **not** use it, since their leaf frame is derived from joint state rather than a fixed MJCF element.

**Key interface:**

- **State:** `joint_pos`, `joint_vel`, `ctrl` (get/set numpy arrays)
- **Limits:** `joint_pos_limits`, `ctrl_limits`
- **Frames:** `leaf_frame_to_world`, `root_frame_to_world`, `leaf_frame_to_root`
- **Control:** `noop_ctrl`, `get_jacobian()`

#### Frames

Each move group represents a kinematic chain between two frames: a **root frame** and a **leaf frame**.
For an arm, the root frame is typically the shoulder/base of the arm and the leaf frame is the end-effector.
For a gripper or a fixed base, the root and leaf frames may be the same.

The move group provides transforms between these frames and the world:

- `leaf_frame_to_world` — the leaf frame's 4×4 pose in world coordinates (e.g. end-effector pose)
- `root_frame_to_world` — the root frame's 4×4 pose in world coordinates (e.g. arm base mount)
- `leaf_frame_to_root` — the leaf frame relative to the root (computed from the above two)
- `leaf_frame_to_robot` / `root_frame_to_robot` — relative to the robot's base frame (uses `robot_base_group` if available)

The Jacobian returned by `get_jacobian()` maps joint velocities to spatial velocity of the **leaf frame**.

### Robot View

A **robot view** assembles a set of named move groups into a single coherent robot interface.
It provides bulk state queries (`get_qpos_dict`, `get_ctrl_dict`), Jacobian column masking across groups, and gripper lookups.

**Base class:** [`RobotView`][molmo_spaces.robots.robot_views.abstract.RobotView]

A `RobotView` is constructed from an `MjData` handle and a `dict[str, MoveGroup]`.
The string keys (e.g. `"arm"`, `"gripper"`, `"base"`) are the **move group IDs** used throughout the codebase — in configs, action dicts, and policy outputs.

**Key interface:**

- `move_group_ids()` — list of all group names
- `get_move_group(mg_id)` — look up a single group
- `get_qpos_dict(mg_ids)` / `set_qpos_dict(qpos_dict)` — bulk joint position access
- `get_ctrl_dict(mg_ids)` / `get_noop_ctrl_dict()` — bulk control access
- `get_jacobian(move_group_id, input_move_group_ids)` — Jacobian for one group's frame, with columns restricted to the listed input groups

### Robot

The **robot** is the top-level abstraction that composes a `RobotView` with controllers and kinematics.
It handles the control loop: receiving action commands keyed by move group ID, dispatching them to controllers, and writing MuJoCo `ctrl`.

**Base class:** [`Robot`][molmo_spaces.robots.abstract.Robot]

**What it holds:**

- `robot_view` — the assembled `RobotView`
- `controllers` — `dict[str, Controller]`, typically one per commanded move group
- `kinematics` / `parallel_kinematics` — FK/IK solvers

**Key interface:**

- `update_control(action_command_dict)` — feed per-group action arrays to controllers
- `compute_control()` — run controllers and write `ctrl` to MuJoCo
- `set_world_pose(pose)` — set the robot base pose (e.g. via mocap)
- `reset()` — reset controllers and internal state

**Action format:** Actions throughout the codebase are `dict[str, np.ndarray]` mapping move group IDs to command arrays, e.g. `{"arm": np.array([...]), "gripper": np.array([...])}`.

#### Configuration

Robot configs (`BaseRobotConfig`) reference move group IDs as dictionary keys:

```python
class BaseRobotConfig:
    init_qpos: dict[str, list[float]]      # e.g. {"arm": [...], "gripper": [...]}
    command_mode: dict[str, str]            # e.g. {"arm": "joint_position", "gripper": "joint_position"}
```

These keys must match the names of the robot's move groups.

The **command mode** determines what the action arrays for each move group mean and which controller is used:

- `"joint_position"` — action values are target joint positions (absolute)
- `"joint_rel_position"` — action values are deltas added to the current joint positions

Each command mode maps to a different `Controller` subclass that translates the action into MuJoCo `ctrl` signals.
Controllers run at the **control timestep** (`ctrl_dt`), which is typically much slower than the MuJoCo **simulation timestep** (`sim_dt`).
On each control step, the controller updates `ctrl` once and MuJoCo simulates multiple sub-steps at `sim_dt` before the next control update.
This separation is handled by the task layer (see [Timing](#timing) below).

### How they compose

For example:

```
Robot
├── robot_view: RobotView
│   └── move_groups: dict[str, MoveGroup]
│       ├── "arm"     → MoveGroup (7 DoF)
│       ├── "gripper" → GripperGroup (1 DoF)
│       └── "base"    → ImmobileRobotBaseGroup (0 DoF)
├── controllers: dict[str, Controller]
│   ├── "arm"     → JointPosController
│   └── "gripper" → JointPosController
└── kinematics: MlSpacesKinematics
```

## Environment, Tasks, and Task Samplers

The simulation lifecycle is a three-layer stack: **Env** runs physics, **Task** wraps it for episodic interaction, and **Task Sampler** generates randomized task instances.

### Env

The **environment** is the MuJoCo-backed physics and rendering substrate.
It owns the compiled model, batched simulation data, robots, cameras, and object managers.

**Base class:** [`BaseMujocoEnv`][molmo_spaces.env.env.BaseMujocoEnv] / [`CPUMujocoEnv`][molmo_spaces.env.env.CPUMujocoEnv]

**What it manages:**

- `MjModel` and one `MjData` per batch slot
- Robot instances (created via factory from config)
- Rendering (Filament or OpenGL)
- `CameraManager` and `ObjectManager` per batch row
- Collision checks, segmentation, visibility queries

**Key interface:**

- `reset(idxs)` — `mj_resetData` + `mj_forward` for selected batch indices
- `step(n_steps)` — `mj_step` across all batch data

!!! warning "Batched environments"
    The env API is nominally batched (multiple `MjData` slots, per-index reset, etc.), but in practice batch sizes greater than 1 are not well tested and have sharp edges throughout the stack. Assume `n_batch=1` for now; broader batching support may be improved in the future.

### Task

A **task** wraps (but does not own!) an env for Gymnasium-style episodic interaction.
It defines timing (control dt vs sim dt vs policy dt), aggregates sensors into observations, implements reward/success semantics, and manages the step counter.
Note that the lifecycle of an env is generally longer than that of a task.

**Base class:** [`BaseMujocoTask`][molmo_spaces.tasks.task.BaseMujocoTask]

**Key interface:**

- `reset()` → `(observation, info)` — clears episode state, resets sensors and policy, returns first observation
- `step(action)` → `(obs, reward, terminated, truncated, info)` — applies action, runs nested physics steps, polls sensors
- `is_done()` — `is_terminal() or is_timed_out()`
- `judge_success()` — abstract, implemented by subclasses
- `get_task_description()` — natural language instruction for the episode

#### Timing

The task manages three nested timestep rates:

- **Simulation dt** (`sim_dt`) — the MuJoCo physics timestep (e.g. 2ms). This is set in the MuJoCo model and determines numerical integration accuracy.
- **Control dt** (`ctrl_dt_ms`) — how often robot controllers update `ctrl` (e.g. 20ms). Each control step runs `ctrl_dt / sim_dt` simulation sub-steps.
- **Policy dt** (`policy_dt_ms`) — how often the policy is queried for a new action (e.g. 200ms). Each policy step runs `policy_dt / ctrl_dt` control steps.

A single call to `task.step(action)` corresponds to **one policy step**: it sets the action on the controllers, then loops over `n_ctrl_steps_per_policy` control ticks. On each control tick, the controllers write `ctrl` and the env advances `n_sim_steps_per_ctrl` simulation sub-steps. This means the physics is simulated at high frequency for stability while the policy and controllers operate at their own (slower) rates.

**Important:** `task.reset()` does **not** call `env.reset()`.
The task assumes the environment is already in the desired physical state (set up by the sampler).
It only resets its own bookkeeping: step counter, caches, sensors, and registered policy.

**Concrete example:** [`PickTask`][molmo_spaces.tasks.pick_task.PickTask] adds lift-based rewards, success checking via object height, and task-specific sensor configuration.

### Task Sampler

A **task sampler** owns the environment lifecycle and generates randomized task instances.
It loads scenes (houses), places robots and objects, configures cameras, and constructs a concrete `Task`.

**Base class:** [`BaseMujocoTaskSampler`][molmo_spaces.tasks.task_sampler.BaseMujocoTaskSampler]

**What it does that a task doesn't:**

| | Task Sampler | Task |
|---|---|---|
| Owns the env | Yes (creates and closes it) | Holds a reference |
| Loads/compiles scenes | Yes (MjSpec, assets, houses) | No |
| Randomizes placement | Yes (robot pose, objects, lighting) | No |
| Implements `reset`/`step` | No | Yes |
| Defines reward/success | No | Yes |

**Key interface:**

- `sample_task()` → `BaseMujocoTask | None` — the main entry point; loads or reuses a scene (an env), randomizes it, and returns a ready-to-use task
- `randomize_scene(env, robot_view)` — abstract; subclass randomizes lighting, textures, dynamics, joint noise
- `_sample_task(env)` — abstract; subclass selects objects, places the robot, configures the task, and returns a `Task` instance

**Concrete example:** `PickTaskSampler` selects a graspable object from candidates, places the robot within reach, generates referral expressions, and returns a `PickTask`.

### Episode lifecycle

A typical episode flows through these layers:

1. **Construct sampler** — `PickTaskSampler(config)` seeds RNG; env is `None` until the first scene loads.

2. **`task = sampler.sample_task()`** — Loads or reuses a house scene (an env), randomizes object/robot placement, and constructs a `PickTask`. The env is now in a specific physical state.

3. **`obs, info = task.reset()`** — Clears episode bookkeeping (step counter, caches). Resets sensors and the registered policy. Returns the first observation. The MuJoCo state is **not** reset here.

4. **`obs, reward, terminated, truncated, info = task.step(action)`** — The action dict (keyed by move group ID) is dispatched to robot controllers. The env steps MuJoCo forward. Sensors produce the next observation.

5. **Termination** — `task.is_done()` returns `True` when the task succeeds, a "done" action is sent, or the horizon is reached.

6. **Next episode** — Call `sampler.sample_task()` again. The sampler may reuse the same compiled scene or load a new house.

**Ownership:** The sampler owns and closes the env. Closing a task only clears its env reference without shutting down the simulator.
