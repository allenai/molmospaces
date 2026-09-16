"""G1Task: the task layer of the port (reset/step/render, observations,
reward, terminal and success rules), a plain task like BaseMujocoTask, driven
through molmo_spaces' GymEnv wrapper. Episode choice is G1TaskSampler's, the
substrate G1CPUMujocoEnv's; PickTask/OpenTask fill in the abstract hooks.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext

import mujoco
import numpy as np

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors_g1 import TARGET_POINT_IN_HEAD_SENSOR, task_sensors
from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_OPEN_HOLD,
    PHASE_POST_CLOSE,
    PHASE_REALIGN,
)
from molmo_spaces.tasks.task import BaseMujocoTask


class G1Task(BaseMujocoTask, ABC):
    """One G1 rollout over a ``G1CPUMujocoEnv``. Constructed once and
    re-configured by every ``sample_task()``: a fresh task per episode would
    consume RNG draws gold never makes and drop the grasp-file cache.

    A real ``BaseMujocoTask``, but one that cannot run the base ``__init__``:
    that reads ``self._env.mj_model.opt.timestep`` to derive the control rates,
    and this task is constructed *before* the first ``load_scene`` (the load
    needs the task, for ``set_objects``). The rates therefore come from the
    config below -- ``G1Config.physics_timestep`` is what puts ``sim_dt_ms`` on
    the model, so the two agree by construction -- and the handful of base
    attributes the inherited methods read are set here instead.
    """

    def __init__(self, env, exp_config):
        """Signature mirrors molmo_spaces' ``BaseMujocoTask(env, exp_config)``.
        This class reads ``exp_config.task_config``; the sampler reads
        ``exp_config.task_sampler_config``."""
        # `env` is a read-only property on BaseMujocoTask, backed by `_env`.
        self._env = env
        self.config = exp_config
        tc = exp_config.task_config

        # Control-loop rates, as in BaseMujocoTask.__init__ -- but read from
        # the config rather than the loaded model, because the sampler builds
        # this task *before* the first load_scene (load_scene needs the task,
        # for set_objects). G1Config.physics_timestep is what puts sim_dt_ms on
        # the model, so the two agree by construction.
        if exp_config.ctrl_dt_ms % exp_config.sim_dt_ms != 0:
            raise ValueError(
                f"Control dt {exp_config.ctrl_dt_ms}ms is not divisible by "
                f"sim dt {exp_config.sim_dt_ms}ms"
            )
        self._n_sim_steps_per_ctrl = int(exp_config.ctrl_dt_ms // exp_config.sim_dt_ms)
        self._n_ctrl_steps_per_policy = int(exp_config.policy_dt_ms // exp_config.ctrl_dt_ms)
        # What BaseMujocoTask.__init__ would have set, for the methods this
        # class inherits rather than overrides (is_timed_out, is_done, ...).
        self._ctrl_dt_ms = exp_config.ctrl_dt_ms
        self._task_horizon = (
            exp_config.task_horizon if exp_config.task_horizon is not None else np.inf
        )
        self._cumulative_reward = np.zeros(1)
        self._num_steps_taken = np.zeros(1, dtype=int)
        self.episode_step_count = 0
        self.viewer = None
        self.frozen_config = None
        self._datagen_profiler = None
        self._done_action_received = False
        self._sensor_suite = None

        self._action_noise_offset = np.zeros(10, dtype=np.float64)
        self._action_noise_std = tc.action_noise_std
        self._action_noise_step = 0
        self._action_noise_stride = max(1, tc.action_noise_stride)
        self._prev_grasp_phase = None
        self._terminate_before_grasp_collision = tc.terminate_before_grasp_collision
        self._terminate_grasp_if_not_visible = tc.terminate_grasp_if_not_visible
        self._terminate_on_grasp_collision = tc.terminate_on_grasp_collision
        # The acting policy, attached via register_policy().
        self.agent = None
        # Overlaid on recorded video by the env; the concrete tasks set it when
        # they sample the episode's language prompt.
        self._prompt = ""
        # Set by the sampler at the end of each sample_task(); returned by reset().
        self._episode_info = {}

    # ---- episode lifecycle ----

    def reset(self):
        """First observation of the episode the sampler has already set up.
        Does NOT sample a new episode -- ``G1TaskSampler.sample_task()`` does,
        same division as molmo_spaces' BaseMujocoTask.reset."""
        return self._build_obs(), dict(self._episode_info)

    def render(self, camera_name: str | None = None) -> np.ndarray:
        """RGB array, same signature as BaseMujocoTask.render (which is how the
        GymEnv wrapper calls it).

        Args:
            camera_name: Camera to render from; defaults to the first camera
                this env exposes.
        """
        return self.env.render_rgb_frame(camera_name or next(iter(self.env.cameras)))

    def close(self):
        """Drop this episode's state. Keeps the env (the sampler owns it) and
        `agent`, since this one task is reused across episodes."""
        self._sensor_suite = None
        self._episode_info = {}

    def register_policy(self, policy):
        """Attach the acting policy, mirroring BaseMujocoTask.register_policy.
        The policy holds the task, not the env: the task survives scene
        reloads, so this is a one-time wiring and only ``policy.setup()`` has
        to be re-run against a rebuilt scene (the sampler does that).
        """
        self.agent = policy
        policy.set_task(self)
        # The reference controller advances the WBC gait clock itself (inside
        # sample_actions), so compute_control must not advance it again --
        # same handoff G1PickPlannerPolicy performs.
        self.env.robot._external_gait_clock = True

    # ---- rollout ----

    def sync_viewer(self):
        if not self.env._viewer:
            return
        with self.env._viewer.lock():
            if getattr(self.env, "debug", False):
                self._draw_debug_markers()
            self.env._viewer.sync()

    def _draw_debug_markers(self):
        scn = self.env._viewer.user_scn
        scn.ngeom = 0
        if self.agent is None:
            return
        wps = getattr(self.agent, "_waypoints", None)
        if wps:
            cur_idx = int(getattr(self.agent, "_wp_idx", 0))
            eye = np.eye(3).flatten()
            size = np.array([0.20, 0.003, 0.0], dtype=np.float32)
            z = 0.003
            for i, wp in enumerate(wps):
                if scn.ngeom >= len(scn.geoms):
                    break
                done = i < cur_idx
                rgba = (
                    np.array([0.0, 1.0, 0.0, 0.7], dtype=np.float32)
                    if done
                    else np.array([1.0, 0.1, 0.1, 0.7], dtype=np.float32)
                )
                pos = np.array([float(wp[0]), float(wp[1]), z], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    size=size,
                    pos=pos,
                    mat=eye,
                    rgba=rgba,
                )
                scn.ngeom += 1
        # Probe overlay is purely visual; the real probe stays parked at z=10.
        planner = getattr(self.agent, "_grasp_planner", None)
        if planner is None:
            return
        grasp_pos = getattr(planner, "_grasp_pos", None)
        grasp_rot = getattr(planner, "_grasp_rot", None)
        if grasp_pos is None or grasp_rot is None:
            return
        if not hasattr(self.env, "_probe_local_geoms"):
            self.env._probe_local_geoms = self.env._cache_probe_local_geoms()
        grasp_xpos = np.asarray(grasp_pos, dtype=np.float64)
        grasp_xmat = np.asarray(grasp_rot, dtype=np.float64)
        rgba = np.array([0.7, 0.9, 1.0, 0.45], dtype=np.float32)
        for type_int, size, local_pos, local_mat in self.env._probe_local_geoms:
            if scn.ngeom >= len(scn.geoms):
                break
            world_pos = grasp_xpos + grasp_xmat @ local_pos
            world_mat = grasp_xmat @ local_mat
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=type_int,
                size=np.asarray(size, dtype=np.float32),
                pos=world_pos.astype(np.float32),
                mat=world_mat.flatten().astype(np.float32),
                rgba=rgba,
            )
            scn.ngeom += 1
        # Articulate end pose (slide/hinge target) — orange overlay so we can see
        # where the gripper is supposed to end up at the end of the pull/swing.
        end_pos = getattr(self.agent, "_articulate_end_pos", None)
        end_rot = getattr(self.agent, "_articulate_end_rot", None)
        if end_pos is not None and end_rot is not None:
            end_xpos = np.asarray(end_pos, dtype=np.float64)
            end_xmat = np.asarray(end_rot, dtype=np.float64)
            end_rgba = np.array([1.0, 0.55, 0.1, 0.45], dtype=np.float32)
            for type_int, size, local_pos, local_mat in self.env._probe_local_geoms:
                if scn.ngeom >= len(scn.geoms):
                    break
                world_pos = end_xpos + end_xmat @ local_pos
                world_mat = end_xmat @ local_mat
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=type_int,
                    size=np.asarray(size, dtype=np.float32),
                    pos=world_pos.astype(np.float32),
                    mat=world_mat.flatten().astype(np.float32),
                    rgba=end_rgba,
                )
                scn.ngeom += 1
            # Joint pivot (purple sphere) + joint axis (cyan line) + intermediate
            # waypoints (small white spheres) — exposes the geometry that drives
            # the articulate trajectory so wrong axis/pivot is visible.
            kind = getattr(self.agent, "_articulate_kind", None)
            pivot = getattr(self.agent, "_articulate_pivot", None)
            axis = getattr(self.agent, "_articulate_axis", None)
            if pivot is not None and scn.ngeom < len(scn.geoms):
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
                    size=np.array([0.03, 0.0, 0.0], dtype=np.float32),
                    pos=np.asarray(pivot, dtype=np.float32),
                    mat=np.eye(3, dtype=np.float32).flatten(),
                    rgba=np.array([0.6, 0.0, 0.8, 0.9], dtype=np.float32),
                )
                scn.ngeom += 1
            if axis is not None and scn.ngeom < len(scn.geoms):
                anchor = np.asarray(pivot if pivot is not None else grasp_xpos, dtype=np.float64)
                a = np.asarray(axis, dtype=np.float64)
                a_norm = float(np.linalg.norm(a))
                if a_norm > 1e-6:
                    a = a / a_norm
                    half = 0.25
                    p0, p1 = anchor - a * half, anchor + a * half
                    mid = (p0 + p1) * 0.5
                    z_axis = a
                    tmp = (
                        np.array([1.0, 0.0, 0.0])
                        if abs(z_axis[0]) < 0.9
                        else np.array([0.0, 1.0, 0.0])
                    )
                    x_axis = np.cross(tmp, z_axis)
                    x_axis /= np.linalg.norm(x_axis) + 1e-9
                    y_axis = np.cross(z_axis, x_axis)
                    mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                        size=np.array([0.006, half, 0.0], dtype=np.float32),
                        pos=mid.astype(np.float32),
                        mat=mat.astype(np.float32),
                        rgba=np.array([0.1, 0.9, 1.0, 0.9], dtype=np.float32),
                    )
                    scn.ngeom += 1
            # Trajectory waypoints (10 small white spheres) along the arc/line.
            from scipy.spatial.transform import Rotation as _R

            n_wp = 10
            disp = getattr(self.agent, "_articulate_disp", None)
            ang = getattr(self.agent, "_articulate_angle", None)
            for i in range(1, n_wp + 1):
                if scn.ngeom >= len(scn.geoms):
                    break
                t = i / n_wp
                if kind == "slide" and disp is not None:
                    wp = grasp_xpos + t * np.asarray(disp, dtype=np.float64)
                elif kind == "hinge" and ang is not None and pivot is not None and axis is not None:
                    R_t = _R.from_rotvec(
                        np.asarray(axis, dtype=np.float64) * (t * float(ang))
                    ).as_matrix()
                    wp = np.asarray(pivot, dtype=np.float64) + R_t @ (
                        grasp_xpos - np.asarray(pivot, dtype=np.float64)
                    )
                else:
                    continue
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
                    size=np.array([0.008, 0.0, 0.0], dtype=np.float32),
                    pos=wp.astype(np.float32),
                    mat=np.eye(3, dtype=np.float32).flatten(),
                    rgba=np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32),
                )
                scn.ngeom += 1

    def _target_visible_in_head(self):
        """True if the target object projects to a pixel inside the head fisheye
        frame (not behind the camera and within [0,W]x[0,H]). Matches the
        target_point obs: not visible == the (-1,-1)/out-of-frame sentinel."""
        pt = TARGET_POINT_IN_HEAD_SENSOR.get_observation(self.env, self)
        if pt is None:
            return False
        H, W = self.env.camera_size
        u, v = pt
        return 0.0 <= u <= W and 0.0 <= v <= H

    @property
    def sensor_suite(self) -> SensorSuite:
        """Built on first use rather than in __init__, because the sampler has
        to construct this task *before* the first load_scene (load_scene's
        candidate selection needs the task), so `env.current_robot` does not
        exist yet. molmo_spaces' own BaseMujocoTask builds it eagerly for the
        opposite reason: its sampler loads the scene first."""
        if self._sensor_suite is None:
            self._sensor_suite = self._create_sensor_suite_from_config(self.config)
            self._sensor_suite.extend(self.env.current_robot.create_robot_sensors())
        return self._sensor_suite

    def _create_sensor_suite_from_config(self, exp_config) -> SensorSuite:
        """Task-specific sensors, as in BaseMujocoTask. Robot-specific sensors
        are appended separately from `robot.create_robot_sensors()`."""
        return SensorSuite(task_sensors())

    def _build_obs(self):
        return self.sensor_suite.get_observations(self.env, self)

    def _robot_touches_world(self):
        """Returns True if any robot geom is in contact with a non-robot, non-floor,
        non-target geom. Pure read of MuJoCo's already-computed contact array —
        microseconds. The target object is excluded so policies that brush against
        the bowl during PHASE_REALIGN don't get pre-crash terminated."""
        m, d = self.env.scene.model, self.env.scene.data
        rset = self.env._robot_body_set
        if not rset:
            return False
        # Target body set (for compound objects). May be absent on bare/open tasks.
        tset = getattr(self, "_target_body_set", None) or set()
        PLANE = mujoco.mjtGeom.mjGEOM_PLANE
        for i in range(d.ncon):
            c = d.contact[i]
            bid1 = int(m.geom_bodyid[c.geom1])
            bid2 = int(m.geom_bodyid[c.geom2])
            b1 = bid1 in rset
            b2 = bid2 in rset
            if b1 == b2:  # both robot (self) OR both non-robot (world-world)
                continue
            # Floor / world plane: don't count.
            if m.geom_type[c.geom1] == PLANE or m.geom_type[c.geom2] == PLANE:
                continue
            # Target object (bowl/etc): don't count — contact with the goal is fine.
            other = bid2 if b1 else bid1
            if other in tset:
                continue
            return True
        return False

    def is_terminal(self, reward=None):
        """Terminal check, matching molmo_spaces' BaseMujocoTask.is_terminal().
        `reward` is accepted and unused, so both the reference call sites (which
        pass one) and molmo_spaces' (which do not) work.
        `reward` is passed for signature compatibility with the pre-split
        version, which used it for a reward-threshold fallback when the task
        object defined no is_terminated(); every G1Task subclass now defines
        one (it is abstract here), so the fallback is gone and this is a
        plain delegation."""
        return bool(self.is_terminated(self.env.scene))

    def judge_success(self, reward=None):
        """Success check, matching molmo_spaces' BaseMujocoTask.judge_success().
        `reward` is accepted and unused -- see is_terminal."""
        return bool(self.is_success(self.env.scene))

    def step(self, action):
        """`action` is a move-group dict, molmo_spaces' own action encoding
        (`{"legs_waist": [vx, vy, yaw_rate, height, waist(3)], "right_arm": (7),
        "right_gripper": (1)}`) -- see g1_wbc.flat15_to_move_groups for the
        translation from gold's flat-15, which the reference controller still
        emits. A "done" key is consumed as the policy's terminal signal, as in
        BaseMujocoTask._apply_action.
        """
        action = dict(action)
        if action.pop("done", False):
            self._done_action_received = True
        legs_waist = np.asarray(action["legs_waist"], dtype=np.float64)
        # Record pre-noise base velocity command for next-step observation.
        self.env._last_base_vel_cmd = legs_waist[0:3].astype(np.float32).copy()
        if self._action_noise_std > 0:
            in_precision = (
                self.agent is not None
                and hasattr(self.agent, "in_precision_phase")
                and self.agent.in_precision_phase()
            )
            if not in_precision:
                # Resample noise once per record-stride env.steps so saved frames stay consistent.
                if self._action_noise_step % self._action_noise_stride == 0:
                    self._action_noise_offset = self.env.np_random.normal(
                        0.0, self._action_noise_std, size=10
                    )
                self._action_noise_step += 1
                # gold injects onto flat[4:14] (g1_wbc.ACT_WAIST + ACT_ARM):
                # the waist, legs_waist[4:7] here, plus the whole right arm.
                right_arm = np.asarray(action["right_arm"], dtype=np.float64).copy()
                legs_waist = legs_waist.copy()
                legs_waist[4:7] += self._action_noise_offset[0:3]
                right_arm += self._action_noise_offset[3:10]
                action["legs_waist"] = legs_waist
                action["right_arm"] = right_arm
            else:
                self._action_noise_offset.fill(0.0)
                self._action_noise_step = 0
        # Waist envelope is enforced upstream by the WBC IK joint limits (policy.py).
        terminated, sim_error = False, None
        lock = self.env._viewer.lock() if self.env._viewer else nullcontext()
        try:
            with lock:
                # molmo_spaces' own control dispatch (BaseMujocoTask._apply_action):
                # set the targets once, then run the control loop.
                self.env.robot.update_control(action)
                for _ in range(self._n_ctrl_steps_per_policy):
                    self.env.robot.compute_control()
                    self.env.step(self._n_sim_steps_per_ctrl)
            if not self.env.robot.state_is_finite():
                terminated, sim_error = True, "non-finite state"
            elif self.env.robot.pelvis_height() < 0.15:
                terminated, sim_error = True, f"fell (z={self.env.robot.pelvis_height():.3f})"
            else:
                # Pre-grasp gripper-collision check: active while walking/realigning,
                # off once controller starts reaching for the object (APPROACH onward).
                # Set terminate_before_grasp_collision=False in env cfg to skip this entirely
                # (e.g. during RL fine-tuning where collisions are part of exploration).
                if self._terminate_before_grasp_collision:
                    phase = (
                        getattr(self.agent, "_grasp_phase", None)
                        if self.agent is not None
                        else None
                    )
                    pre_grasp = phase in (None, PHASE_IDLE, PHASE_REALIGN)
                    if pre_grasp and self._robot_touches_world():
                        self.env._gripper_precrash = True
                        terminated, sim_error = True, "robot hit world before grasp"
                # During-grasp collision check: terminate when the robot
                # contacts any non-floor, non-target geom while in the actual
                # grasping phases (arm reaching / closing / lifting). The
                # target body set is already excluded inside _robot_touches_world,
                # so contact with the bowl during CLOSE/LIFT doesn't count.
                if not terminated and self._terminate_on_grasp_collision:
                    phase = (
                        getattr(self.agent, "_grasp_phase", None)
                        if self.agent is not None
                        else None
                    )
                    in_grasp = phase in (
                        PHASE_APPROACH,
                        PHASE_DESCEND,
                        PHASE_OPEN_HOLD,
                        PHASE_CLOSE,
                        PHASE_POST_CLOSE,
                        PHASE_LIFT,
                    )
                    if in_grasp and self._robot_touches_world():
                        terminated, sim_error = True, "robot hit world during grasp"
                # Visibility check: at the single step the controller transitions
                # into reaching (enters APPROACH), the object must be visible in
                # the head camera. Otherwise the demo would teach a grasp of
                # something the policy can't see, so terminate (failure -> not
                # saved). Checked once at the transition, not every step.
                if not terminated and self._terminate_grasp_if_not_visible:
                    phase = (
                        getattr(self.agent, "_grasp_phase", None)
                        if self.agent is not None
                        else None
                    )
                    starts_reaching = (
                        phase == PHASE_APPROACH and self._prev_grasp_phase != PHASE_APPROACH
                    )
                    if starts_reaching and not self._target_visible_in_head():
                        terminated, sim_error = True, "target not visible at grasp"
                    self._prev_grasp_phase = phase
        except mujoco.FatalError as e:
            terminated, sim_error = True, str(e)

        dt = self.env.robot.n_substeps * self.env.scene.model.opt.timestep
        self.env._sim_time += dt
        if self.env._viewer:
            self.sync_viewer()

        obs = self._build_obs()
        tgt = self.target
        dist = float(
            np.linalg.norm(self.env.robot.get_xy() - tgt.position(self.env.scene.data)[:2])
        )
        reward = self.compute_reward(self.env.scene)
        if self.is_terminal(reward):
            terminated = True
        success = self.judge_success(reward)
        info = self.step_info()
        info.update(
            target_object_position=tgt.position(self.env.scene.data),
            target_object_pose=np.concatenate(
                [tgt.position(self.env.scene.data), tgt.quat(self.env.scene.data)]
            ),
            distance=dist,
            success=success,
            sim_error=sim_error,
        )
        self.attach_grasps(info)
        return obs, reward, terminated, False, info

    def get_reward(self):
        """``BaseMujocoTask.get_reward``: the batched reward for the current
        state. The reference reward is per-scene and scalar, so this is
        ``compute_reward`` boxed into the single-env shape."""
        return np.asarray([self.compute_reward(self.env.scene)], dtype=np.float64)

    def get_task_description(self) -> str:
        """``BaseMujocoTask.get_task_description``: the episode's language
        prompt, which the concrete tasks set when they sample it."""
        return self._prompt

    def step_chunk(self, action_chunk, stop_on_success: bool = False):
        """Run `action_chunk` open-loop, reporting only the final observation --
        molmo_spaces' BaseMujocoTask.step_chunk contract, which the datagen
        pipeline steps episodes through.

        The base implementation reaches for `_apply_action`/`_observe_and_cache`
        to skip sensor polling mid-chunk; this task has no such split, so each
        action is a full step and the chunk simply stops early on termination.
        Returns the batched shapes the single-env reference stack implies.
        """
        if not action_chunk:
            raise ValueError("step_chunk requires at least one action")
        obs = reward = info = None
        terminated = truncated = False
        for action in action_chunk:
            obs, reward, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                break
            if stop_on_success and bool(info.get("success")):
                break
        return (
            [obs],
            np.asarray([reward], dtype=np.float64),
            np.asarray([terminated], dtype=bool),
            np.asarray([truncated], dtype=bool),
            [info],
        )

    def set_datagen_profiler(self, profiler) -> None:
        """Accepted for interface compatibility; this task has no sub-timing
        hooks to attach a profiler to."""
        self._datagen_profiler = profiler

    def consume_skip_episode(self):
        s = self.env._skip_episode
        self.env._skip_episode = False
        return s

    # ---- hooks the concrete tasks (PickTask / OpenTask) supply ----

    @abstractmethod
    def set_objects(self, scene):
        """Collect the candidate target objects from a freshly loaded scene.
        Raises ValueError when the scene holds none -- callers treat that as
        'skip this scene'."""

    @abstractmethod
    def perturb_objects(self, scene, rng):
        """Per-episode jitter of object placement, before the goal is sampled."""

    @abstractmethod
    def select_target(self, rng, randomize=None):
        """Pick (or keep) this episode's target object and return it."""

    @abstractmethod
    def init_target_tracking(self, scene):
        """Snapshot whatever the reward needs about the target's start state."""

    @abstractmethod
    def get_obs(self, scene):
        """Task-specific observation entries."""

    @abstractmethod
    def compute_reward(self, scene):
        """Scalar reward for the current sim state."""

    @abstractmethod
    def is_terminated(self, scene) -> bool:
        """Whether the episode should end now."""

    @abstractmethod
    def is_success(self, scene) -> bool:
        """Whether the episode counts as a success."""

    @abstractmethod
    def make_info(self, scene, rng):
        """Build the episode info dict returned by reset()."""

    @abstractmethod
    def step_info(self):
        """Build the per-step info dict returned by step()."""

    @abstractmethod
    def attach_grasps(self, info):
        """Add the target's grasp set to an info dict."""
