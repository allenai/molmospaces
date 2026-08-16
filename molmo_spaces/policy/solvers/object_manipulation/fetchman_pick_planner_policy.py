from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import PickPlannerPolicy
from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask
from molmo_spaces.utils.grasps import get_pickup_grasps

log = logging.getLogger(__name__)

# g1_molmo's own robot.py PELVIS_FORWARD_OFFSET (~/code/g1_molmo/
# molmospaces/components/robot.py) -- kept as a literal local constant (not
# imported -- that module doesn't exist here). The 30-joint flat order that
# module also defines (JOINT_NAMES) isn't needed here at all: this file's
# own action output is a move-group-keyed dict (see get_action_chunk), not
# g1_molmo's own flat, joint-order-indexed array.
_PELVIS_FWD = 0.05

# g1_molmo's own module-level _astar/_coarsen_and_dist/_simplify_path
# (occupancy-grid A* over its own env's occupancy map representation), and
# the flat-joint-array bookkeeping (_IDX/_LEGS/ACTION_DIM) g1_molmo's own
# GraspPolicy.__call__/FetchmanPickPlannerPolicy.get_action_chunk used to
# build a raw 15-element action array, are gone entirely -- see _plan_path
# (walks straight at a fixed standoff point via self.robot_view only, per
# explicit request -- no path planning at all) and get_action_chunk (builds
# a move-group-keyed action dict via self.robot_view.get_ctrl_dict()
# instead).

PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_REALIGN = "realign"
PHASE_DESCEND = "descend"
PHASE_OPEN_HOLD = "open_hold"
PHASE_CLOSE = "close"
PHASE_POST_CLOSE = "post_close"
PHASE_LIFT = "lift"
PHASE_DONE = "done"
_PHASE_ORDER = [
    PHASE_APPROACH,
    PHASE_DESCEND,
    PHASE_OPEN_HOLD,
    PHASE_CLOSE,
    PHASE_POST_CLOSE,
    PHASE_LIFT,
    PHASE_DONE,
]


class GraspPolicy:
    """Candidate-selection helper for FetchmanPickPlannerPolicy. Ported from
    g1_molmo's own GraspPolicy, but its internal IK/collision-probe machinery
    (_solve_ik, _build_collision_limit) is dead code in g1_molmo's own usage
    too: FetchmanPickPlannerPolicy.setup() always sets
    self._grasp_planner._ik_error_fn = self._wbc_ik_error, so _ik_error()
    always takes that branch and never falls through to this class's own
    _solve_ik -- that's why those two methods (and _steps/_target/_advance/
    __call__, g1_molmo's own alternate standalone-usage interface, never
    invoked via FetchmanPickPlannerPolicy either) aren't ported here at all.

    setup() takes the owning FetchmanPickPlannerPolicy directly (not
    g1_molmo's own (model, data, prefix), which this class no longer needs
    now that its real IK work is delegated) for env/task/robot_view access.
    """

    PREGRASP_OFFSET = (0.05, 0.125)
    LIFT = 0.15

    def __init__(self):
        self._phase = PHASE_IDLE
        self._hand = "right"
        self._ik_error_fn = None

    def setup(self, policy: FetchmanPickPlannerPolicy) -> None:
        self._policy = policy

    @property
    def done(self):
        return self._phase == PHASE_DONE

    def _path_is_clear(self, grasp_T: np.ndarray) -> bool:
        """g1_molmo's own _path_is_clear sweeps a dedicated "gripper_probe"
        collision-probe body (a g1_molmo-specific scene asset) along
        pregrasp->mid->grasp checkpoints. That asset doesn't exist in our
        scenes, so this instead reuses get_noncolliding_grasp_mask (the same
        utility FetchmanPickPlannerPolicy's own _path_clear already used pre-
        rewrite) over the same 3 checkpoints -- same sweep, our own
        already-working collision-check mechanism instead of g1_molmo's probe
        body.
        """
        offset = np.random.uniform(0.10, 0.15)
        pregrasp_pos = grasp_T[:3, 3] - offset * grasp_T[:3, 2]
        pregrasp_checkpoint = grasp_T.copy()
        pregrasp_checkpoint[:3, 3] = pregrasp_pos
        mid_checkpoint = grasp_T.copy()
        mid_checkpoint[:3, 3] = 0.5 * pregrasp_pos + 0.5 * grasp_T[:3, 3]
        checkpoints = np.stack([pregrasp_checkpoint, mid_checkpoint, grasp_T])
        noncolliding = get_noncolliding_grasp_mask(
            self._policy.task.env.current_model,
            self._policy.task.env.current_data,
            checkpoints,
            batch_size=3,
        )
        return bool(noncolliding.all())

    def _ik_error(self, T: np.ndarray, hand: str) -> float:
        # _ik_error_fn is always set by FetchmanPickPlannerPolicy.setup (see
        # class docstring) -- no fallback branch needed.
        return self._ik_error_fn(T, hand)

    def plan(self, info: dict) -> None:
        """Direct port of g1_molmo's GraspPolicy.plan(), with two structural
        adaptations: `info` here is populated by
        FetchmanPickPlannerPolicy._refresh_info_target_pose (computed
        directly from molmo_spaces state, not carried in an incoming env obs/
        info dict the way g1_molmo's own env provides it -- our own task's
        info dict has no such keys at all), and get_pickup_grasps' candidates
        arrive already in *world* frame (not object-local like g1_molmo's own
        valid_grasps, which need `Tw @ go`) -- see the loop below.
        """
        self._phase = PHASE_IDLE
        self._hand = "right"
        tp = info.get("target_object_pose")
        vg = info.get("valid_grasps")
        log.info(
            f"[G1_MOLMO_TRACE] plan() entry: tp_is_none={tp is None} "
            f"vg_len={0 if vg is None else len(vg)}"
        )
        if tp is None:
            self._phase = PHASE_DONE
            return

        Tw = np.asarray(tp, dtype=np.float64)
        prefix = self._policy._prefix
        model = self._policy.task.env.current_model
        data = self._policy.task.env.current_data
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + "right_grasp")

        grasp_T = None
        if vg is not None and len(vg) > 0:
            vg = np.asarray(vg, dtype=np.float64)
            flip = R.from_euler("z", np.pi)
            cands = []
            for hand in ("right",):
                hr = R.from_matrix(data.site_xmat[site_id].reshape(3, 3))
                for go in vg:
                    # get_pickup_grasps already returns world-frame candidates
                    # (unlike g1_molmo's own object-local valid_grasps, which
                    # need `Tw @ go` here) -- see this method's own docstring.
                    c = go.copy()
                    ro = R.from_matrix(c[:3, :3])
                    rf = ro * flip
                    # Pick the yaw=0 or yaw=pi flip closer to current hand -- keeps IK well-conditioned.
                    if (hr.inv() * rf).magnitude() < (hr.inv() * ro).magnitude():
                        c = c.copy()
                        c[:3, :3] = rf.as_matrix()
                    cands.append((c, hand))
            np.random.shuffle(cands)
            # Geometric reach pre-filter: skip candidates whose target is beyond
            # arm reach from the shoulder. Kills 50-80% of cands BEFORE any IK.
            shoulder_world = {}
            for hand in ("right",):
                sb = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{hand}_shoulder_pitch_link"
                )
                if sb >= 0:
                    shoulder_world[hand] = data.xpos[sb].copy()
            MAX_REACH = 0.85  # G1 arm fully extended, generous
            MIN_REACH = 0.10
            if shoulder_world:
                cands = [
                    (c, h)
                    for (c, h) in cands
                    if h not in shoulder_world
                    or MIN_REACH <= float(np.linalg.norm(c[:3, 3] - shoulder_world[h])) <= MAX_REACH
                ]
            ik_results = []
            # Cap candidates aggressively. cands is shuffled so 30 is representative.
            for c, h in cands[:30]:
                e = self._ik_error(c, h)
                ik_results.append((e, c, h))
            if ik_results:
                ik_results.sort(key=lambda x: x[0])
                PATH_CHECK_K = 5
                log.info(
                    f"[G1_MOLMO_TRACE] plan(): {len(cands)} candidates after reach "
                    f"pre-filter, top errors={[round(e, 4) for e, _, _ in ik_results[:5]]}"
                )
                for e, c, h in ik_results[:PATH_CHECK_K]:
                    if e >= 0.1:
                        break  # even the best is too far -- give up
                    if self._path_is_clear(c):
                        grasp_T = c
                        self._hand = h
                        break
            else:
                log.info("[G1_MOLMO_TRACE] plan(): 0 candidates survived reach pre-filter")
            if grasp_T is None:
                log.info(
                    "[G1_MOLMO_TRACE] plan(): no grasp_T found (all candidates too far "
                    "or path blocked) -> PHASE_DONE"
                )
                self._phase = PHASE_DONE
                return
        else:
            self._hand = "right"
            cr = data.site_xmat[site_id].reshape(3, 3)
            fd = cr[:, 0].copy()
            fd[2] = 0
            if np.linalg.norm(fd) < 1e-6:
                fd = np.array([1.0, 0.0, 0.0])
            fd /= np.linalg.norm(fd)
            up = np.array([0.0, 0.0, 1.0])
            rt = np.cross(up, fd)
            rt /= np.linalg.norm(rt) + 1e-8
            fd = np.cross(rt, up)
            grasp_T = np.eye(4)
            grasp_T[:3, :3] = np.column_stack([fd, rt, up])
            grasp_T[:3, 3] = Tw[:3, 3]

        off = np.random.uniform(*self.PREGRASP_OFFSET)
        pre = grasp_T.copy()
        pre[:3, 3] -= off * grasp_T[:3, 2]
        self._pregrasp = pre[:3, 3].copy()
        self._grasp_pos = grasp_T[:3, 3].copy()
        self._grasp_rot = grasp_T[:3, :3].copy()
        self._lift_pos = grasp_T[:3, 3].copy()
        self._lift_pos[2] += self.LIFT
        self._grasp_local = np.linalg.inv(Tw) @ grasp_T
        log.info(
            f"[G1_MOLMO_TRACE] plan(): hand={self._hand} grasp_pos={self._grasp_pos.tolist()} "
            f"grasp_rot_euler_xyz={R.from_matrix(self._grasp_rot).as_euler('xyz').tolist()} "
            f"pregrasp_pos={self._pregrasp.tolist()} pregrasp_offset={off:.4f}"
        )
        # g1_molmo's own _skip_pregrasp_ik=True path (set in
        # FetchmanPickPlannerPolicy.setup): the WBC re-solves IK per step and
        # never reads pregrasp_joints, so that (slow, full-scene) solve is
        # skipped entirely -- ported as unconditional here since we always
        # run that way.
        self._pregrasp_joints = None

        gripper_mg_id = self._policy.robot_view.get_gripper_movegroup_ids()[0]
        start_pose = self._policy.robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        self._start_pos = start_pose[:3, 3].copy()
        self._start_rot = start_pose[:3, :3].copy()
        self._rot_slerp = Slerp(
            [0, 1], R.concatenate([R.from_matrix(self._start_rot), R.from_matrix(self._grasp_rot)])
        )

        self._phase = PHASE_APPROACH

    def refresh_grasp_for_current_object(self) -> bool:
        if getattr(self, "_grasp_local", None) is None:
            return False
        pickup_obj = self._policy._get_pickup_obj()
        Tw = pickup_obj.pose
        grasp_T = Tw @ self._grasp_local
        self._grasp_pos = grasp_T[:3, 3].copy()
        self._grasp_rot = grasp_T[:3, :3].copy()
        self._lift_pos = self._grasp_pos.copy()
        self._lift_pos[2] += self.LIFT
        return True


class FetchmanPickPlannerPolicy(PickPlannerPolicy):
    WAYPOINT_REACH = 0.10
    FINAL_REACH = 0.05
    SPEED = 0.3
    TURN_KP = 2.0
    MAX_TURN = 1.0
    FACE_TURN = 1.2
    FACE_TOL = 0.1
    FACE_WP_TOL = 0.25

    NEAR_GOAL_DIST = 0.05
    REALIGN_MAX = 0.30
    REALIGN_REACH = 0.020
    REALIGN_YAW_REACH = 0.04
    REALIGN_SPEED_X = 0.25
    REALIGN_SPEED_X_MIN = 0.07
    REALIGN_RAMP_STEPS = 12
    REALIGN_RAMP_DOWN_DIST = 0.08
    REALIGN_INITIAL_HOLD = 20
    REALIGN_SPEED_Y = 0.045
    REALIGN_YAW_SPEED = 0.40
    REALIGN_DRIVE_STEPS = 150
    REALIGN_SETTLE = 80

    # g1_molmo's own IK_DT/EXECUTION_IK_MAX_ITERS_*/IK_POS_TOLERANCE etc. --
    # see G1Robot.kinematics_wbc/_wbc_ik_error for how these are used; kept
    # as a class constant (g1_molmo's own reference value) rather than the
    # previous port's separately-named EXECUTION_IK_MAX_ITERS_APPROACH/_PRECISION.
    IK_DT = 1e-2

    def __init__(self, config, task) -> None:
        """Combines g1_molmo's own __init__/create/set_env/setup (four
        separate calls driven by its own env's construction sequence) into
        molmo_spaces' single Policy constructor -- self.robot_view/self.task/
        self.config below come from PickPlannerPolicy.__init__, replacing
        g1_molmo's own self._env/self._data/self._model.
        """
        super().__init__(config, task)
        robot_config = task.env.current_robot.exp_config.robot_config
        assert self.robot_view.name == "g1" and not getattr(robot_config, "use_holo_base", False), (
            "FetchmanPickPlannerPolicy requires G1Config's default WBC-walking mode "
            "(use_holo_base=False) -- it commands legs_waist via "
            "G1WalkController.set_target's [vx,vy,yaw_rate,height,waist] interface, "
            "which only that mode's G1WalkController implements."
        )
        self._prefix = "robot_0/"
        self._walk_timeout_s = 20.0
        self._walk_fail_dist = 0.2
        self._face_yaw_offset_max = 0.0
        # g1_molmo's own get_config() defaults this True; the closer-waypoint
        # retry it drives (_install_closer_waypoint) hasn't been converted to
        # molmo_spaces' own nav-goal sampling -- disabled rather than left
        # half-working.
        self._grasp_retry_closer = False
        self._speed = self.SPEED
        self._min_speed = 0.3
        self._grasp_speed_scale = 1.25
        self._nav_noise_scale = 0.0
        self._upper_cmd = None
        # g1_molmo's own G1WalkController._DEFAULT_HEIGHT_CMD (see
        # controllers/g1_walk.py) -- its _BaseController._reset_wbc_state
        # initializes this; ported directly as a literal since that base
        # class isn't part of molmo_spaces.
        self._height_cmd = 0.74

        # Whole-body grasp IK (mink, standalone ~35-DOF model, not the live
        # scene model -- ~2600x faster per iteration, needed since plan()
        # solves up to 30 candidates per call) now lives on G1Robot itself
        # (kinematics_wbc()/_ensure_wbc_ik_setup()), built lazily on first
        # use there instead of duplicated per-policy here.
        self._active_hand = "right"

        self._waypoints = []
        self._wp_idx = 0
        self._arrived = self._facing = False
        self._target_xy = self._object_xy = None
        self._has_path = False

        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._grasp_pos = None
        self._grasp_rot = None
        self._hand = "right"
        self._grasp_retries_used = 0
        self._step_counter = 0
        # g1_molmo's own _BaseController.__init__/_reset_wbc_state initializes
        # this ([vx, vy, yaw_rate] velocity command) -- that base class isn't
        # part of molmo_spaces, so it's set directly here instead.
        self._cmd = np.zeros(3, dtype=np.float32)

        self._grasp_planner = GraspPolicy()
        self._grasp_planner.setup(self)
        self._grasp_planner._ik_error_fn = self._wbc_ik_error

    @property
    def has_path(self):
        return self._has_path

    def in_precision_phase(self):
        return self._grasp_phase in (
            PHASE_REALIGN,
            PHASE_DESCEND,
            PHASE_OPEN_HOLD,
            PHASE_CLOSE,
            PHASE_POST_CLOSE,
            PHASE_LIFT,
        )

    @property
    def done(self):
        return self._grasp_phase == PHASE_DONE

    # g1_molmo's own precheck_grasp (a fast, state-restoring "is this
    # reachable" check meant to be called before committing to a spawn) isn't
    # ported -- it's not on our own live call path (molmo_spaces' task
    # sampler has its own, separate _precheck_grasp_reachable implementation,
    # see molmo_spaces/tasks/pick_task_sampler.py, which doesn't call
    # anything on the policy at all) and it referenced removed g1_molmo-only
    # internals (self._env, self._set_groot_defaults) that no longer exist
    # here.

    def _get_pickup_obj(self):
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        return om.get_object_by_name(task_config.pickup_obj_name)

    def _xy(self) -> np.ndarray:
        base_pose = self.robot_view.base.pose
        yaw = self._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([base_pose[0, 3] + c * _PELVIS_FWD, base_pose[1, 3] + s * _PELVIS_FWD])

    def _yaw(self) -> float:
        return float(R.from_matrix(self.robot_view.base.pose[:3, :3]).as_euler("xyz")[2])

    # Reasonable arm-reach standoff distance from the object's own position --
    # see this method's own note below for why this replaces g1_molmo's own
    # occupancy-grid A* (_astar/_simplify_path over its own env's occupancy
    # map representation) entirely, rather than adapting that grid-search to
    # molmo_spaces' own (differently-shaped) occupancy map.
    STANDOFF_DIST = 0.4

    def _plan_path(self, info=None) -> None:
        """Per explicit request, this does not reuse molmo_spaces' own
        AStarPlanner/NavGoalSampler (grid-based path planning with wall
        clearance, visibility checks, etc.) -- only self.robot_view. Walks
        straight at a fixed standoff point on the line from the robot's
        current position to the object, no path planning at all. Simpler and
        less robust than both g1_molmo's own grid A* and molmo_spaces'
        AStarPlanner/NavGoalSampler (neither avoids obstacles), but PickG1's
        own narrow spawn radius (base_pose_sampling_radius_range=(0.2, 0.5))
        means the robot is almost always already this close to the object
        anyway -- see the near-zero-distance branch below.
        """
        pickup_obj = self._get_pickup_obj()
        self._object_xy = np.asarray(pickup_obj.position[:2], dtype=np.float64)
        xy = self._xy()
        delta = self._object_xy - xy
        dist = float(np.linalg.norm(delta))
        if dist <= self.STANDOFF_DIST:
            target_xy = xy.copy()
        else:
            target_xy = self._object_xy - self.STANDOFF_DIST * (delta / dist)
        self._target_xy = target_xy
        self._waypoints = [target_xy]
        self._wp_idx = 0
        self._has_path = True

    def _update_nav_command(self):
        if self._arrived:
            self._cmd[:] = 0
            return
        if not self._waypoints:
            self._cmd[:] = 0
            return
        xy, yaw = self._xy(), self._yaw()
        if self._facing:
            face = self._object_xy if self._object_xy is not None else self._target_xy
            if face is not None:
                desired = np.arctan2(face[1] - xy[1], face[0] - xy[0]) + getattr(
                    self, "_face_yaw_offset", 0.0
                )
                ye = (desired - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(ye) > self.FACE_TOL:
                    self._cmd[:] = [
                        0,
                        0,
                        np.clip(self.TURN_KP * ye, -self.FACE_TURN, self.FACE_TURN),
                    ]
                    return
            self._arrived = True
            self._cmd[:] = 0
            return
        wp = self._waypoints[self._wp_idx]
        if (
            np.linalg.norm(xy - wp) < self.WAYPOINT_REACH
            and self._wp_idx < len(self._waypoints) - 1
        ):
            self._wp_idx += 1
        wp = self._waypoints[self._wp_idx]
        delta = wp - xy
        dist = np.linalg.norm(delta)
        final = self._wp_idx >= len(self._waypoints) - 1
        # Smoothstep brake — zero derivative at both ends so we "roll" into a stop
        # instead of stepping. Longer runway (0.70 m) gives ~2.3s of decel at
        # 0.3 m/s — feels natural in real-life walking.
        BRAKE_DIST = 0.70
        STOP_PAD = 0.04  # extra margin to absorb walking inertia
        stop_dist = self.FINAL_REACH + STOP_PAD
        # Arrive at stop_dist (where the brake zeros speed), else robot hangs in the
        # dead zone short of FINAL_REACH.
        if final and dist <= stop_dist:
            self._facing = True
            self._cmd[:] = 0
            return
        ye = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(ye) > self.FACE_WP_TOL:
            self._cmd[:] = [0, 0, np.clip(self.TURN_KP * ye, -self.MAX_TURN, self.MAX_TURN)]
            return
        if final:
            if dist <= stop_dist:
                spd = 0.0
            elif dist >= BRAKE_DIST:
                spd = self._speed
            else:
                t = (dist - stop_dist) / (BRAKE_DIST - stop_dist)
                spd = self._speed * (3 * t * t - 2 * t * t * t)
        else:
            spd = np.clip(dist, self._min_speed, self._speed)
        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
        ln = max(np.sqrt(lx**2 + ly**2), 1e-6)
        ang = np.sign(self.TURN_KP * ye) * np.clip(abs(self.TURN_KP * ye), 0.05, self.MAX_TURN)
        self._cmd[:] = [spd * lx / ln, np.clip(spd * ly / ln, -0.5, 0.5), ang]

    # g1_molmo's own perturb_action_for_rollout (nav-command noise injection
    # for training-data diversity) isn't called anywhere in molmo_spaces
    # either, and operated on g1_molmo's own flat ACTION_DIM=15 array
    # (out[0:3]/out[2]) -- not the move-group-keyed dict get_action_chunk
    # returns now. self._nav_noise_scale is also always 0.0 here (see
    # reset()), so it would have been a no-op regardless.

    def _in_precision_grasp_phase(self) -> bool:
        # Tighter IK during precision phases — wrist has to actually land on the grasp/close target.
        return self._grasp_phase in (PHASE_DESCEND, PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT)

    def _wbc_ik_error(self, T, hand):
        _, _, _, err = self.task.env.current_robot.kinematics_wbc(
            T[:3, 3], T[:3, :3], hand=hand, precision=self._in_precision_grasp_phase()
        )
        return err

    def _start_grasp(self, info=None) -> None:
        """g1_molmo's own version also has a "_start_at_pregrasp cached
        candidate -> jump straight to PHASE_REALIGN" fast path -- dropped
        here: that path exists for its own env's episode-init modes
        (init_arm_at_pregrasp/start_at_pregrasp) we don't use at all (our own
        info dict never carries those keys), so it's dead code in our
        configuration. Likewise _install_closer_waypoint's retry (gated on
        self._grasp_retry_closer, hardcoded False in __init__ -- see that
        field's own docstring) never fires.
        """
        info = self._refresh_info_target_pose(info)
        self._grasp_planner.plan(info)
        if self._grasp_planner._phase == PHASE_DONE:
            self._grasp_phase = PHASE_DONE
            return
        self._grasp_pos = self._grasp_planner._grasp_pos.copy()
        self._grasp_rot = self._grasp_planner._grasp_rot.copy()
        self._pregrasp = self._grasp_planner._pregrasp.copy()
        self._pregrasp_rot = None
        self._lift_pos = self._grasp_pos.copy()
        self._lift_pos[2] += 0.15
        self._hand = self._grasp_planner._hand
        self._active_hand = self._hand
        self._grasp_phase = PHASE_APPROACH
        self._grasp_step = 0
        # Real candidate now selected -- replace reset()'s placeholder (see
        # its own comment) so GraspPoseSensor reports the actual grasp target.
        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = self._grasp_rot
        grasp_pose[:3, 3] = self._grasp_pos
        self.target_poses["grasp"] = grasp_pose

    def _refresh_info_target_pose(self, info: dict | None = None) -> dict:
        """g1_molmo's own version refreshes an existing info dict's
        target_object_pose/valid_grasps keys (in case the object moved during
        nav) by reading its own env.target/env.scene state. Adapted here to
        *build* that dict directly from molmo_spaces state instead of
        reading it out of an incoming obs/info dict -- our own task's info
        dict carries none of g1_molmo's own keys at all, so there's nothing
        to refresh; only to construct fresh from pickup_obj.pose and
        get_pickup_grasps each time this is called (matching g1_molmo's own
        intent: always reflect the object's *current* pose, not a stale one
        from an earlier tick).
        """
        pickup_obj = self._get_pickup_obj()
        info = dict(info or {})
        info["target_object_pose"] = pickup_obj.pose
        info["valid_grasps"] = get_pickup_grasps(
            self.task.env, pickup_obj, grasp_libraries=self.policy_config.grasp_libraries
        )
        return info

    def _current_tcp_pose(self) -> np.ndarray:
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world

    def _grasp_target(self):
        t = min(self._grasp_step / max(1, self._grasp_steps()), 1.0)
        p = self._grasp_phase
        if p == PHASE_APPROACH:
            if self._grasp_step == 0:
                current_pose = self._current_tcp_pose()
                self._approach_start = current_pose[:3, 3].copy()
                target_rot = (
                    self._pregrasp_rot if self._pregrasp_rot is not None else self._grasp_rot
                )
                self._approach_slerp = Slerp(
                    [0, 1],
                    R.concatenate([R.from_matrix(current_pose[:3, :3]), R.from_matrix(target_rot)]),
                )
            pos = self._approach_start * (1 - t) + self._pregrasp * t
            rot = self._approach_slerp(t).as_matrix()
            return pos, rot
        if p == PHASE_DESCEND:
            if self._grasp_step == 0:
                current_pose = self._current_tcp_pose()
                self._descend_start = current_pose[:3, 3].copy()
                self._descend_slerp = Slerp(
                    [0, 1],
                    R.concatenate(
                        [R.from_matrix(current_pose[:3, :3]), R.from_matrix(self._grasp_rot)]
                    ),
                )
            rot = self._descend_slerp(t).as_matrix()
            return self._descend_start * (1 - t) + self._grasp_pos * t, rot
        if p == PHASE_REALIGN:
            return (
                self._pregrasp,
                self._pregrasp_rot if self._pregrasp_rot is not None else self._grasp_rot,
            )
        if p in (PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE):
            return self._grasp_pos, self._grasp_rot
        if p == PHASE_LIFT:
            return self._grasp_pos * (1 - t) + self._lift_pos * t, self._grasp_rot
        return None, None

    def _grasp_steps(self):
        steps = {
            "approach": 700,
            "realign": 500,
            "descend": 600,
            "open_hold": 160,
            "close": 800,
            "post_close": 400,
            "lift": 600,
        }.get(self._grasp_phase, 1)
        if self._grasp_phase in (PHASE_APPROACH, PHASE_DESCEND, PHASE_LIFT):
            steps = int(round(steps / self._grasp_speed_scale))
        return max(1, steps)

    def _grasp_min_steps(self):
        if self._grasp_phase == PHASE_APPROACH and getattr(self, "_quick_approach", False):
            return 5
        return {
            "approach": 40,
            "realign": 20,
            "descend": 40,
            "open_hold": 10,
            "close": 240,
            "post_close": 80,
            "lift": 40,
        }.get(self._grasp_phase, 1)

    def _grasp_phase_goal(self):
        p = self._grasp_phase
        if p in (PHASE_APPROACH, PHASE_REALIGN):
            return self._pregrasp, self._grasp_rot
        if p in (PHASE_DESCEND, PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE):
            return self._grasp_pos, self._grasp_rot
        if p == PHASE_LIFT:
            return self._lift_pos, self._grasp_rot
        return None, None

    def _grasp_phase_done(self):
        if self._grasp_step < self._grasp_min_steps():
            return False
        if self._grasp_phase == PHASE_CLOSE:
            gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
            gripper = self.robot_view.get_gripper(gripper_mg_id)
            # g1_molmo's own check reads its own GRIPPER_CLOSED joint-position
            # constant directly (qpos[right_grip_qa] > GRIPPER_CLOSED - 0.004).
            # inter_finger_dist_range is (closed_dist, open_dist) regardless
            # of this gripper's own open/closed sign convention (see
            # GripperGroup.inter_finger_dist's docstring) -- convention-
            # agnostic equivalent of the same "closed enough" check.
            closed_dist, _open_dist = gripper.inter_finger_dist_range
            return gripper.inter_finger_dist < closed_dist + 0.004
        if self._grasp_phase in (PHASE_OPEN_HOLD, PHASE_POST_CLOSE):
            return True
        if self._grasp_phase == PHASE_REALIGN:
            # Dead in our configuration -- PHASE_REALIGN is only ever entered
            # via g1_molmo's own start_at_pregrasp fast path, which we never
            # set (see _start_grasp's docstring). Left in place (rather than
            # removed) only so a stray self._grasp_phase == PHASE_REALIGN
            # wouldn't hard-crash if something unexpected ever set it.
            if self._start_at_pregrasp:
                if self._realign_arrived_at is not None:
                    return (self._grasp_step - self._realign_arrived_at) >= self.REALIGN_SETTLE
                return False
            close = float(np.linalg.norm(self._xy() - self._realign_target_xy)) < self.REALIGN_REACH
            if not close:
                self._realign_arrived_at = None
                return False
            if self._realign_arrived_at is None:
                self._realign_arrived_at = self._grasp_step
            return (self._grasp_step - self._realign_arrived_at) >= self.REALIGN_SETTLE
        goal_pos, goal_rot = self._grasp_phase_goal()
        if goal_pos is None:
            return False
        current_pose = self._current_tcp_pose()
        pos_err = float(np.linalg.norm(current_pose[:3, 3] - goal_pos))
        # angle(R1^T @ R2) = acos((trace(R1^T @ R2) - 1) / 2) — no scipy.Rotation overhead.
        R1 = current_pose[:3, :3]
        rel_trace = float(np.dot(R1.ravel(), goal_rot.ravel()))
        cos = max(-1.0, min(1.0, (rel_trace - 1.0) * 0.5))
        rot_err = float(np.arccos(cos))
        return pos_err < 0.035 and rot_err < 0.45

    def _log_trace(self, new_phase):
        arm = getattr(self, "_last_arm_joints", None)
        waist = getattr(self, "_last_waist_joints", None)
        t = self.robot_view.mj_data.time
        log.info(
            f"[G1_MOLMO_TRACE] phase->{new_phase} t={t} xy={self._xy().tolist()} "
            f"yaw={self._yaw():.4f} height_cmd={self._height_cmd:.4f} "
            f"arm={None if arm is None else np.round(arm, 4).tolist()} "
            f"waist={None if waist is None else np.round(waist, 4).tolist()}"
        )

    def _advance_grasp(self):
        prev = self._grasp_phase
        if prev == PHASE_REALIGN:
            self._grasp_phase = PHASE_DESCEND
            self._grasp_step = 0
            self._ik_cache = None
            self._upper_cmd = None
            self._realign_done = True
            self._realign_arrived_at = None
            self._quick_approach = True
            if self._grasp_planner.refresh_grasp_for_current_object():
                self._grasp_pos = self._grasp_planner._grasp_pos.copy()
                self._grasp_rot = self._grasp_planner._grasp_rot.copy()
                self._lift_pos = self._grasp_planner._lift_pos.copy()
            self._log_trace(self._grasp_phase)
            return
        try:
            idx = _PHASE_ORDER.index(self._grasp_phase)
            self._grasp_phase = _PHASE_ORDER[idx + 1]
        except (ValueError, IndexError):
            self._grasp_phase = PHASE_DONE
        self._grasp_step = 0
        self._ik_cache = None
        self._upper_cmd = None
        self._quick_approach = False
        if prev == PHASE_APPROACH and self._grasp_phase == PHASE_DESCEND:
            if self._grasp_planner.refresh_grasp_for_current_object():
                self._grasp_pos = self._grasp_planner._grasp_pos.copy()
                self._grasp_rot = self._grasp_planner._grasp_rot.copy()
                self._lift_pos = self._grasp_planner._lift_pos.copy()
        self._log_trace(self._grasp_phase)

    # g1_molmo's own _start_realign (the only call site that would ever
    # enter PHASE_REALIGN through the *normal*, non-start_at_pregrasp path)
    # is never actually called anywhere in g1_molmo's own file either -- see
    # _grasp_phase_done's own note on PHASE_REALIGN being unreachable here.
    # Not ported: it referenced removed g1_molmo-only internals (self._sites,
    # self._data) for a branch nothing reaches.

    def _realign_cmd(self):
        if self._start_at_pregrasp and self._realign_axis == "x":
            if self._grasp_step < self.REALIGN_INITIAL_HOLD:
                return np.zeros(3, dtype=np.float32)
            delta_world = self._realign_target_xy - self._xy()
            yaw = self._yaw()
            lx = np.cos(yaw) * delta_world[0] + np.sin(yaw) * delta_world[1]
            direction = -float(np.sign(self._realign_offset))
            signed_progress_remaining = direction * lx
            if signed_progress_remaining <= self.REALIGN_REACH:
                self._realign_arrived_at = self._realign_arrived_at or self._grasp_step
                return np.zeros(3, dtype=np.float32)
            spd_up = self.REALIGN_SPEED_X_MIN + (
                self.REALIGN_SPEED_X - self.REALIGN_SPEED_X_MIN
            ) * min(1.0, self._grasp_step / max(1.0, self.REALIGN_RAMP_STEPS))
            decel_t = min(1.0, signed_progress_remaining / max(1e-6, self.REALIGN_RAMP_DOWN_DIST))
            spd_down = (
                self.REALIGN_SPEED_X_MIN
                + (self.REALIGN_SPEED_X - self.REALIGN_SPEED_X_MIN) * decel_t
            )
            speed = min(spd_up, spd_down)
            return np.array([direction * speed, 0.0, 0.0], dtype=np.float32)
        if self._start_at_pregrasp and self._realign_axis == "yaw":
            yaw_err = (self._goal_yaw_target - self._yaw() + np.pi) % (2 * np.pi) - np.pi
            direction = -float(np.sign(self._realign_offset))
            signed_progress_remaining = direction * (-yaw_err)
            if abs(yaw_err) <= self.REALIGN_YAW_REACH or signed_progress_remaining <= 0:
                self._realign_arrived_at = self._realign_arrived_at or self._grasp_step
                return np.zeros(3, dtype=np.float32)
            return np.array([0.0, 0.0, direction * self.REALIGN_YAW_SPEED], dtype=np.float32)
        if self._start_at_pregrasp and self._realign_axis == "y":
            vy = -float(np.sign(self._realign_offset)) * self.REALIGN_SPEED_Y
            return np.array([0.0, vy, 0.0], dtype=np.float32)
        delta_world = self._realign_target_xy - self._xy()
        yaw = self._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        lx = c * delta_world[0] + s * delta_world[1]
        ly = -s * delta_world[0] + c * delta_world[1]
        dist = float(np.linalg.norm(delta_world))
        if dist < self.REALIGN_REACH:
            return np.zeros(3, dtype=np.float32)
        ln = max(np.sqrt(lx * lx + ly * ly), 1e-6)
        speed = min(self.REALIGN_SPEED_X, dist * 2.0)
        return np.array([speed * lx / ln, speed * ly / ln, 0.0], dtype=np.float32)

    def reset(self, reset_retries: bool = True) -> None:
        """g1_molmo's own reset(info) reads a large number of episode-init
        keys from its own env's info dict (init_arm_at_pregrasp,
        start_at_pregrasp, realign_axis/offset, pregrasp_joints, goal_xy/
        goal_yaw, occupancy_map, init_height, init_upper_pose) -- our own
        task never produces any of these (no info dict is passed in at all;
        molmo_spaces' own Policy.reset() takes no info argument), so every
        one of them is just its own "not set" default below. The trailing
        pregrasp-joints/init-upper-pose application g1_molmo's own reset()
        does at the end is dropped entirely since both are always None here.
        """
        self._arrived = self._facing = False
        self._waypoints = []
        self._wp_idx = 0
        self._has_path = False
        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._step_counter = 0
        self._ik_cache = None
        if reset_retries:
            self._grasp_retries_used = 0
        self._upper_cmd = None
        self._realign_done = False
        self._realign_target_xy = None
        self._realign_arrived_at = None
        self._quick_approach = False
        self._last_arm_joints = None
        self._last_waist_joints = None
        self._cmd_smoothed = None
        self._stall_min_dist = float("inf")
        self._stall_last_progress_t = 0.0
        self._init_arm_at_pregrasp = False
        self._start_at_pregrasp = False
        self._realign_axis = None
        self._realign_offset = 0.0
        self._pregrasp_joints = None
        self._goal_xy_target = None
        self._goal_yaw_target = None
        self._init_upper = None
        self._speed = float(np.random.uniform(0.3, 0.5))
        self._min_speed = 0.05
        self._face_yaw_offset = 0.0
        self._grasp_speed_scale = float(np.random.uniform(0.85, 1.15))
        self._nav_noise_scale = 0.0
        self._height_cmd = 0.74
        self._info = None

        # Placeholder "grasp" entry keeps GraspPoseSensor (which reads
        # target_poses["grasp"] unconditionally every tick, molmo_spaces'
        # own sensor -- not part of g1_molmo -- see abstract_sensors) from
        # KeyError-ing while still walking; _start_grasp overwrites this with
        # the real candidate pose once one is selected.
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        self.target_poses = {
            "grasp": self.robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        }

        self._plan_path()

    def set_step_info(self, info):
        self._info = info

    def _sample_action(self, obs):
        """g1_molmo's own sample_actions(obs) -- one action per call, matching
        its own env's one-action-per-env.step() semantics (see
        get_action_chunk below for how this becomes a real multi-tick chunk).
        Output is now a move-group-keyed dict
        (self.robot_view.get_ctrl_dict()'s own format -- "legs_waist"/
        "right_arm"/the gripper move-group id), not g1_molmo's flat
        ACTION_DIM=15 array. Two of g1_molmo's own branches are dropped
        entirely as dead code in our configuration (see reset()'s docstring
        for why _init_arm_at_pregrasp/_start_at_pregrasp/_init_upper are
        always their "unset" defaults here): the init_arm_at_pregrasp stall-
        recovery block, and the elif branch driving a raw _init_upper pose
        from obs["base_height"]/obs["joint_pos"] (this policy's own
        decisions never read `obs` at all, matching get_action_chunk's other
        callers -- see FetchmanPickPlannerPolicy's earlier get_action_chunk
        docstring on why that's safe to replay stale/ignore).
        """
        self._step_counter += 1
        if (
            not self._arrived
            and self._grasp_phase == PHASE_IDLE
            and self.robot_view.mj_data.time > self._walk_timeout_s
            and self._target_xy is not None
            and float(np.linalg.norm(self._xy() - self._target_xy)) > self._walk_fail_dist
        ):
            self._grasp_phase = PHASE_DONE
        self._update_nav_command()
        if self._grasp_phase == PHASE_REALIGN:
            self._cmd[:] = self._realign_cmd()
        else:
            target = self._cmd.copy()
            prev = getattr(self, "_cmd_smoothed", None)
            if prev is None:
                prev = np.zeros(3, dtype=np.float32)
            if self._arrived or self._facing:
                self._cmd_smoothed = target.astype(np.float32)
            else:
                alpha = 0.15
                self._cmd_smoothed = (1.0 - alpha) * prev + alpha * target
                self._cmd_smoothed[2] = float(target[2])
            self._cmd[:] = self._cmd_smoothed
        if self._arrived and self._grasp_phase == PHASE_IDLE:
            self._start_grasp()

        pos, rot, grip = None, None, 0.0
        if self._grasp_phase not in (PHASE_IDLE, PHASE_DONE):
            if self._grasp_phase_done():
                self._advance_grasp()
            pos, rot = self._grasp_target()
            if self._grasp_phase in (PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT):
                grip = 1.0
            if grip > 0 and self._grasp_phase == PHASE_CLOSE:
                grip = min(self._grasp_step / max(1, self._grasp_steps()), 1.0)
            self._grasp_step += 1
            if self._grasp_step >= self._grasp_steps():
                self._advance_grasp()

        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        gripper = self.robot_view.get_gripper(gripper_mg_id)
        # gripper.ctrl == a raw joint-position target (see
        # G1GripperGroup.set_gripper_ctrl_open); derive the open/closed ctrl
        # values from inter_finger_dist_range (closed_dist, open_dist)
        # instead of hardcoding g1_molmo's own GRIPPER_OPEN/GRIPPER_CLOSED
        # constants, which may not match this gripper's own calibration --
        # convention-agnostic equivalent of g1_molmo's own
        # g_val = GRIPPER_OPEN*(1-grip) + GRIPPER_CLOSED*grip.
        closed_dist, open_dist = gripper.inter_finger_dist_range
        gripper_open_ctrl, gripper_closed_ctrl = -open_dist, -closed_dist
        g_val = gripper_open_ctrl * (1 - grip) + gripper_closed_ctrl * grip

        arm_joints = np.zeros(7, dtype=np.float32)
        waist_joints = np.zeros(3, dtype=np.float32)
        height_cmd = self._height_cmd
        if pos is not None:
            IK_DECIM = 3
            cache = getattr(self, "_ik_cache", None)
            if cache is None or (self._step_counter % IK_DECIM) == 0:
                # Self-collision avoidance on only during grasp execution.
                arm_joints, waist_joints, ik_h, _ = self.task.env.current_robot.kinematics_wbc(
                    pos,
                    rot,
                    hand=self._active_hand,
                    avoid_self_collision=True,
                    precision=self._in_precision_grasp_phase(),
                )
                self._ik_cache = (arm_joints, waist_joints, ik_h)
            else:
                arm_joints, waist_joints, ik_h = self._ik_cache
            # Exponential smoothing toward the IK's solved pelvis height. This
            # only converges if the smoothed value is carried forward. The
            # reference stack gets that for free because its `_height_cmd`
            # lives on the shared low-level controller, which writes it back
            # every tick from the action it just executed (see
            # g1_molmo_port/components/controller_g1ms.py's
            # `o._height_cmd = float(height_cmd)`). Here `_height_cmd` is this
            # policy's own attribute and was otherwise written only in
            # __init__/reset, so every tick recomputed 0.74 + 0.1*(ik_h - 0.74)
            # from the same frozen 0.74: the pelvis stopped 10% of the way down
            # and never crouched to the object. The arm then tried to make up
            # the remaining reach on its own, saturating the waist envelope.
            self._height_cmd = self._height_cmd + 0.1 * (ik_h - self._height_cmd)
            height_cmd = self._height_cmd
            last_arm = getattr(self, "_last_arm_joints", None)
            if last_arm is not None and self._quick_approach and self._grasp_step < 30:
                alpha = float(np.clip(self._grasp_step / 30.0, 0.0, 1.0))
                arm_joints = (1.0 - alpha) * last_arm + alpha * arm_joints
                last_waist = getattr(self, "_last_waist_joints", None)
                if last_waist is not None:
                    waist_joints = (1.0 - alpha) * last_waist + alpha * waist_joints
            self._last_arm_joints = arm_joints.copy()
            self._last_waist_joints = waist_joints.copy()

        action = self.robot_view.get_ctrl_dict()
        action["legs_waist"] = np.array(
            [self._cmd[0], self._cmd[1], self._cmd[2], height_cmd, *waist_joints], dtype=np.float32
        )
        action[gripper_mg_id] = np.array([gripper_open_ctrl], dtype=np.float32)
        if pos is not None:
            action["right_arm"] = arm_joints.astype(np.float32)
            action[gripper_mg_id] = np.array([g_val], dtype=np.float32)
        if self._grasp_phase == PHASE_DONE:
            action["done"] = True
        return action

    # g1_molmo's own render_cameras() (~/code/g1_molmo/molmospaces/env.py) is
    # never called from inside env.step() at all -- env.step()'s own
    # per-tick obs (_build_obs) is pure analytic math, no pixels. The real
    # image render is decimated by the *external* harness driving the env:
    # train/parallel_rollout.py's action_repeat=20 steps physics 20 times
    # (5ms each) per rendered frame -- a render every 100ms (10Hz), fully
    # independent of IK_DECIM's own (3-tick, 15ms) WBC-target-refresh
    # cadence, which is a different concern entirely (see IK_DECIM's own
    # docstring). RENDER_DECIM below is molmo_spaces' analogous knob:
    # BaseMujocoTask.step_chunk (molmo_spaces/tasks/task.py) only polls
    # sensors -- including rendering the head/wrist camera RGB images, by
    # far the most expensive part of a tick once use_sensors=True -- once
    # per chunk, so sizing the chunk to RENDER_DECIM ticks reproduces
    # g1_molmo's own 10Hz render cadence instead of (re-)solving IK's own
    # 65Hz cadence for something it was never meant to gate.
    #
    # This is specific to FetchmanPickPlannerPolicy -- BasePolicy.
    # get_action_chunk's own default (return None) is untouched, so every
    # other policy in molmo_spaces keeps observing every tick (chunk size 1,
    # no decimation) exactly as before.
    RENDER_DECIM = 20

    def get_action_chunk(self, obs) -> list[dict[str, Any]] | None:
        """Bundle RENDER_DECIM consecutive _sample_action() calls into one
        chunk -- see RENDER_DECIM's own docstring for why 20 (not IK_DECIM's
        3) is the right number here.

        This policy's own decisions never actually read `obs` at all --
        every _sample_action() call works from live ground-truth state
        (self.robot_view accessors), not the sensor suite -- so replaying
        the same (stale) observation argument across a chunk introduces no
        approximation on that front. Restricted to the grasp-execution phase
        (self._arrived) -- not the walk phase, whose nav command
        (_update_nav_command) does want fresh position feedback every tick
        to steer correctly.
        """
        if not self._arrived:
            return None
        chunk = []
        for _ in range(self.RENDER_DECIM):
            action = self._sample_action(obs)
            chunk.append(action)
            if action.get("done"):
                break
        return chunk

    def get_action(self, observation):
        """BasePolicy.get_action is abstract -- must exist for this class to
        be instantiable at all -- but during the grasp-execution phase
        get_action_chunk above always returns a real chunk, never None, so
        the framework's own `policy.get_action_chunk(obs) or
        [policy.get_action(obs)]` fallback only actually reaches this while
        still walking (get_action_chunk returns None there).
        """
        return self._sample_action(observation)

    def get_phase(self) -> str:
        if not self._arrived:
            return "facing" if self._facing else "walking"
        return self._grasp_phase

    def get_all_phases(self) -> dict[str, int]:
        # Overrides BaseObjectManipulationPlannerPolicy.get_all_phases's own
        # generic pregrasp/grasp/gripper-close/lift/preplace/place/retreat/
        # go_home enum (a TCPMoveSegment-era vocabulary this policy no longer
        # uses at all -- see this module's own docstring) with g1_molmo's own
        # phase names directly: walking/facing (this policy's own pre-arrival
        # states, not in _PHASE_ORDER) plus PHASE_IDLE and every entry of
        # _PHASE_ORDER. Consumed by PolicyPhaseSensor (molmo_spaces/env/
        # sensors.py), which otherwise logs "Unknown phase" for every one of
        # get_phase()'s real return values.
        names = ["unknown", "walking", "facing", PHASE_IDLE, *_PHASE_ORDER]
        return {name: i for i, name in enumerate(names)}
