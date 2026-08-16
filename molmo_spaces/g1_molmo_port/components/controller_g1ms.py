from __future__ import annotations

import collections
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from molmo_spaces.g1_molmo_port import ASSETS_DIR
from molmo_spaces.robots.g1 import DEFAULT_QPOS as _ROBOT_DEFAULT_QPOS
from molmo_spaces.robots.g1 import JOINT_NAMES as _JOINTS

# Named slices of the flat 15-element action array execute_action() parses --
# molmo_spaces' own FetchmanPickPlannerPolicy builds a move-group-keyed dict
# instead of a flat array (self.robot_view.get_ctrl_dict(), no such
# abstraction here), but until that split happens these give the same
# move-group boundaries symbolic names instead of magic-number slices, so
# execute_action() and agents/policy_g1ms.py's sample_actions() (the two
# ends of this array contract) stay in sync by construction.
# Flat legs_waist target width molmo_spaces policies emit; mirrors
# controllers/g1_walk.py NUM_TARGET_DIMS. See LegsWaistController.set_target.
NUM_LEGS_WAIST_TARGET_DIMS = 7

ACTION_DIM = 15  # cmd(3) + height(1) + waist(3) + right_arm(7) + right_grip(1)
ACT_CMD = slice(0, 3)
ACT_HEIGHT = 3
ACT_WAIST = slice(4, 7)
ACT_ARM = slice(7, 14)
ACT_GRIP = 14
ACT_UPPER = slice(
    3, 14
)  # height+waist+arm together -- agents/policy_g1ms.py's own upper-body smoothing range

_WBC_NUM_ACTIONS = 15
_WBC_DEFAULT = np.array(
    [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], dtype=np.float32
)
_WBC_CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
_WBC_ACTION_SCALE = 0.25
_WBC_ANG_VEL_SCALE = 0.5
_WBC_DOF_POS_SCALE = 1.0
_WBC_DOF_VEL_SCALE = 0.05
_WBC_HEIGHT_CMD = 0.74
_WBC_OBS_HISTORY = 6
_WBC_CONTROL_DEC = 4
_WBC_OBS_DIM = 86
_WBC_BODY = _JOINTS[:29]  # legs(12) + waist(3) + arms(14); gripper tendon-controlled.
_WBC_N = 29

_WBC_KP = np.array(
    [
        150,
        150,
        150,
        200,
        40,
        40,
        150,
        150,
        150,
        200,
        40,
        40,
        250,
        250,
        250,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.float32,
)
_WBC_KD = np.array(
    [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.float32,
)


def _gravity_orientation(quat):
    w, x, y, z = quat
    gv = np.array([0.0, 0.0, -1.0])
    c = np.array([w, -x, -y, -z])
    return np.array(
        [
            gv[0] * (c[0] ** 2 + c[1] ** 2 - c[2] ** 2 - c[3] ** 2)
            + gv[1] * 2 * (c[1] * c[2] - c[0] * c[3])
            + gv[2] * 2 * (c[1] * c[3] + c[0] * c[2]),
            gv[0] * 2 * (c[1] * c[2] + c[0] * c[3])
            + gv[1] * (c[0] ** 2 - c[1] ** 2 + c[2] ** 2 - c[3] ** 2)
            + gv[2] * 2 * (c[2] * c[3] - c[0] * c[1]),
            gv[0] * 2 * (c[1] * c[3] - c[0] * c[2])
            + gv[1] * 2 * (c[2] * c[3] + c[0] * c[1])
            + gv[2] * (c[0] ** 2 - c[1] ** 2 - c[2] ** 2 + c[3] ** 2),
        ],
        dtype=np.float32,
    )


# Resolved through molmo_spaces' own asset manager (same location the real
# G1Robot/G1WalkController -- robots/g1.py's
# `robot_config.get_robot_dir() / "policies"` -- already loads these same
# ONNX weights from) rather than a locally-copied agents/models/ dir, so
# this port carries no binary weight files of its own at all.
_DEFAULT_MODELS_DIR = ASSETS_DIR / "robots" / "g1" / "policies"


class MoveGroup:
    """Minimal MoveGroup abstraction mirroring molmo_spaces/robots/
    robot_views/abstract.py's shape: owns raw qpos/qvel/ctrl I/O for a named
    subset of G1's actuated DOFs. Index arrays (`jqpos`/`jdof`/`act_ids`) are
    populated once in G1Controller.setup() exactly as before (same
    `_WBC_BODY`-indexed lookups) -- this only groups and names them, doesn't
    change how they're computed.
    """

    def __init__(self, name, jqpos, jdof, act_ids):
        self.name = name
        self.jqpos = jqpos
        self.jdof = jdof
        self.act_ids = act_ids

    def joint_pos(self, data):
        return data.qpos[self.jqpos].astype(np.float32)

    def joint_vel(self, data):
        return data.qvel[self.jdof].astype(np.float32)

    def set_ctrl(self, data, values):
        for i, aid in enumerate(self.act_ids):
            if aid >= 0:
                data.ctrl[aid] = values[i]


class Controller:
    """Minimal Controller abstraction mirroring molmo_spaces/controllers/
    abstract.py's shape: owns the control law for one MoveGroup
    (`set_target` + `compute_ctrl_inputs`). G1Controller.execute_action is
    the dispatcher: set_target on each, then compute_ctrl_inputs + set_ctrl
    for each, in the same order gold's single execute_action body used.
    """

    def __init__(self, move_group):
        self.move_group = move_group
        self.target = None

    def set_target(self, target):
        self.target = target

    def compute_ctrl_inputs(self, data, step_counter):
        raise NotImplementedError

    # -- molmo_spaces/controllers/abstract.py's Controller surface --
    #
    # The reference stack never calls these: it drives every controller through
    # G1Controller.execute_action, which set_targets each one every tick. They
    # exist so molmo_spaces' own task/robot stack (Robot.reset,
    # Robot.set_stationary) can drive this robot too. Clearing `target` is what
    # "stationary" means for these controllers -- G1Controller.execute_action
    # overwrites it on the very next tick regardless, so this cannot change the
    # reference stack's behaviour.

    @property
    def stationary(self) -> bool:
        return self.target is None

    def set_to_stationary(self) -> None:
        self.target = None

    def reset(self) -> None:
        self.target = None


class LegsWaistController(Controller):
    """Combined 15-dof legs+waist PD law + ONNX WBC residual. Legs and waist
    form ONE move group on G1 (matches molmo_spaces' G1LegsWaistGroup) since
    the WBC computes one coupled torque law across both -- waist is a direct
    PD target, legs come from the ONNX-network's `_target_lower` output.

    Deliberately NOT self-contained state: `_cmd`/`_height_cmd`/
    `_waist_target`/`_target_lower`/`_wbc_action`/`_obs_history`/
    `_obs_buffer`/`_step_counter` all live on the *owning* G1Controller
    instance (`owner`), not here, because agents/policy_g1ms.py's subclass
    reads and writes several of these directly on `self` (e.g. `_cmd` from
    its own nav logic, `_height_cmd` for its own height-smoothing, both
    assuming a single shared instance -- see reset()'s `self._height_cmd =
    float(init_height)` override and sample_actions()'s `self._height_cmd`
    reads). Giving this controller its own private copies would silently
    fork that shared state and break the ping-pong between policy and
    controller. This class only owns the *math* (the control law itself,
    relocated verbatim); state ownership stays exactly where gold's
    `execute_action` always kept it.
    """

    def __init__(self, move_group, owner):
        super().__init__(move_group)
        self._owner = owner

    def reset(self):
        o = self._owner
        o._cmd[:] = 0
        o._wbc_action[:] = 0
        o._target_lower[:] = _WBC_DEFAULT
        o._height_cmd = _WBC_HEIGHT_CMD
        o._waist_target[:] = 0
        o._obs_history.clear()
        for _ in range(_WBC_OBS_HISTORY):
            o._obs_history.append(np.zeros(_WBC_OBS_DIM, dtype=np.float32))

    def set_target(self, target):
        # Two calling conventions, distinguished by shape (not by guessing):
        #
        #   (cmd3, height, waist3)                 -- the reference stack's, via
        #                                             G1Controller.execute_action
        #   [vx, vy, yaw_rate, height, waist(3)]   -- molmo_spaces', the flat
        #                                             7-vector G1WalkController.
        #                                             set_target defines
        #
        # Native callers reach this controller directly, not only through
        # Robot.update_control -- e.g. PickTaskSampler._randomize_robot_standing_
        # height -- so accepting the native form here is what makes this
        # controller drop-in for molmo_spaces' own task/policy stack. A 3-tuple
        # of (array3, scalar, array3) and a flat 7-vector are unambiguously
        # different lengths, so this dispatch cannot misfire.
        if len(target) == NUM_LEGS_WAIST_TARGET_DIMS:
            target = np.asarray(target, dtype=np.float32)
            cmd, height_cmd, waist = target[0:3], float(target[3]), target[4:7]
        elif len(target) == 3:
            cmd, height_cmd, waist = target
        else:
            raise ValueError(
                "legs_waist target must be either (cmd3, height, waist3) or a flat "
                f"{NUM_LEGS_WAIST_TARGET_DIMS}-vector [vx, vy, yaw_rate, height, "
                f"waist(3)]; got length {len(target)}"
            )
        o = self._owner
        o._cmd[:] = cmd
        o._height_cmd = float(height_cmd)
        o._waist_target[:] = waist

    def compute_ctrl_inputs(self, data, step_counter):
        o = self._owner
        target_q15 = np.empty(15, dtype=np.float32)
        target_q15[:12] = o._target_lower[:12]
        target_q15[12:15] = o._waist_target

        q = self.move_group.joint_pos(data)
        dq = self.move_group.joint_vel(data)
        tau = (target_q15 - q) * _WBC_KP[:15] - dq * _WBC_KD[:15]
        for i in range(12, 15):
            tau[i] += data.qfrc_bias[self.move_group.jdof[i]]

        if step_counter % _WBC_CONTROL_DEC == 0:
            qj = data.qpos[o._jqpos[:29]].astype(np.float32)
            dqj = data.qvel[o._jdof[:29]].astype(np.float32)
            quat = data.qpos[o._fj_qa + 3 : o._fj_qa + 7].astype(np.float32)
            omega = data.qvel[o._fj_da + 3 : o._fj_da + 6].astype(np.float32)
            padded = np.zeros(29, dtype=np.float32)
            padded[:15] = _WBC_DEFAULT
            single_obs = np.zeros(_WBC_OBS_DIM, dtype=np.float32)
            single_obs[0:3] = o._cmd * _WBC_CMD_SCALE
            single_obs[3] = o._height_cmd
            single_obs[4:7] = [o._waist_target[1], o._waist_target[2], o._waist_target[0]]
            single_obs[7:10] = omega * _WBC_ANG_VEL_SCALE
            single_obs[10:13] = _gravity_orientation(quat)
            single_obs[13:42] = (qj - padded) * _WBC_DOF_POS_SCALE
            single_obs[42:71] = dqj * _WBC_DOF_VEL_SCALE
            single_obs[71:86] = o._wbc_action
            o._obs_history.append(single_obs)
            while len(o._obs_history) < _WBC_OBS_HISTORY:
                o._obs_history.appendleft(np.zeros_like(single_obs))
            for i, h in enumerate(o._obs_history):
                o._obs_buffer[i * _WBC_OBS_DIM : (i + 1) * _WBC_OBS_DIM] = h
            obs_tensor = o._obs_buffer[None, :].astype(np.float32)
            if np.linalg.norm(o._cmd) < 0.05:
                o._wbc_action = o._stand_sess.run(
                    None, {o._stand_sess.get_inputs()[0].name: obs_tensor}
                )[0][0].astype(np.float32)
            else:
                o._wbc_action = o._walk_sess.run(
                    None, {o._walk_sess.get_inputs()[0].name: obs_tensor}
                )[0][0].astype(np.float32)
            o._target_lower = o._wbc_action * _WBC_ACTION_SCALE + _WBC_DEFAULT

        return tau


class LeftArmController(Controller):
    """Left arm: zero-gain actuators (falls under gravity, per
    G1Controller.setup's actuator_gainprm[aid,0]=0.0). Target is always 0 --
    execute_action never sets a left-arm target, matching gold exactly."""

    def compute_ctrl_inputs(self, data, step_counter):
        return np.zeros(len(self.move_group.act_ids), dtype=np.float32)


class RightArmController(Controller):
    """Right arm: plain position-actuator passthrough (PD gains baked into
    the MJCF actuator, kp=2000/kd=60 -- see G1Controller.setup)."""

    def compute_ctrl_inputs(self, data, step_counter):
        return np.asarray(self.target, dtype=np.float32)


class RightGripperController(Controller):
    """Right gripper: single tendon position-actuator passthrough."""

    def compute_ctrl_inputs(self, data, step_counter):
        return np.asarray([self.target], dtype=np.float32)


class G1Controller:
    def __init__(self, models_dir: Path | str | None = None):
        d = Path(models_dir) if models_dir is not None else _DEFAULT_MODELS_DIR
        self._stand_sess = ort.InferenceSession(
            str(d / "groot_balance.onnx"), providers=["CPUExecutionProvider"]
        )
        self._walk_sess = ort.InferenceSession(
            str(d / "groot_walk.onnx"), providers=["CPUExecutionProvider"]
        )

    def set_env(self, env):
        m = env.scene.model
        m.opt.timestep = 0.005
        m.opt.noslip_iterations = 5
        m.opt.impratio = 1.0
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        m.opt.gravity[:] = [0, 0, -9.81]
        m.opt.enableflags = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        env.robot.n_substeps = 1
        for gid in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE or "floor" in gname.lower():
                m.geom_friction[gid] = [1.0, 0.005, 0.0001]
                m.geom_solref[gid] = [0.02, 1.0]
        prefix = self._prefix if hasattr(self, "_prefix") else "robot_0/"
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if (
                bname.startswith(prefix)
                and "ankle_roll" in bname
                and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
            ):
                m.geom_friction[gid] = [1.0, 0.005, 0.0001]

    def setup(self, model, data, prefix="robot_0/"):
        self._model, self._data, self._prefix = model, data, prefix

        self._jqpos = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{n}")
                ]
                for n in _WBC_BODY
            ],
            dtype=np.int32,
        )
        self._jdof = np.array(
            [
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{n}")
                ]
                for n in _WBC_BODY
            ],
            dtype=np.int32,
        )
        self._fj_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}floating_base_joint"
        )
        self._fj_qa = model.jnt_qposadr[self._fj_id]
        self._fj_da = model.jnt_dofadr[self._fj_id]
        self._pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}pelvis")

        self._act_ids = []
        for i, jname in enumerate(_WBC_BODY):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}walk_{jname}")
            if aid < 0:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{jname}")
            self._act_ids.append(aid)
            if aid >= 0:
                if i < 15:
                    model.actuator_gainprm[aid, 0] = 1.0
                    model.actuator_biasprm[aid, :] = 0.0
                    model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
                    model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
                elif i < 22:
                    # Left arm: zero gain so it falls under gravity.
                    model.actuator_gainprm[aid, 0] = 0.0
                    model.actuator_biasprm[aid, :] = 0.0
                    model.actuator_forcerange[aid] = [-400, 400]
                else:
                    kp, kd = 2000.0, 60.0
                    model.actuator_gainprm[aid, 0] = kp
                    model.actuator_biasprm[aid, 0] = 0.0
                    model.actuator_biasprm[aid, 1] = -kp
                    model.actuator_biasprm[aid, 2] = -kd
                    model.actuator_forcerange[aid] = [-400, 400]
        self._act_ids = np.array(self._act_ids)

        for i in range(15, _WBC_N):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{_WBC_BODY[i]}")
            if jid >= 0:
                model.jnt_actfrcrange[jid] = [-400, 400]
                model.dof_damping[model.jnt_dofadr[jid]] = 3.0

        self._rgripper = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}right_grip"
        )

        # Shared WBC state -- read/written directly by agents/policy_g1ms.py's
        # subclass too (nav command, height smoothing), so these stay owned
        # by this G1Controller instance, not by LegsWaistController (see its
        # docstring for why).
        self._obs_history = collections.deque(maxlen=_WBC_OBS_HISTORY)
        self._obs_buffer = np.zeros(_WBC_OBS_DIM * _WBC_OBS_HISTORY, dtype=np.float32)
        self._wbc_action = np.zeros(_WBC_NUM_ACTIONS, dtype=np.float32)
        self._target_lower = _WBC_DEFAULT.copy()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._height_cmd = _WBC_HEIGHT_CMD
        self._step_counter = 0

        self._target_q = np.zeros(_WBC_N, dtype=np.float32)
        self._target_q[:15] = _WBC_DEFAULT
        self._waist_target = np.zeros(3, dtype=np.float32)

        # MoveGroups: named slices of the 29-length _WBC_BODY index arrays
        # (+ the separately-resolved right-gripper actuator), matching G1's
        # real actuated groups (see molmo_spaces/robots/robot_views/g1_view.py
        # for the equivalent G1LegsWaistGroup/G1ArmGroup/G1GripperGroup).
        legs_waist_mg = MoveGroup(
            "legs_waist", self._jqpos[:15], self._jdof[:15], self._act_ids[:15]
        )
        left_arm_mg = MoveGroup(
            "left_arm", self._jqpos[15:22], self._jdof[15:22], self._act_ids[15:22]
        )
        right_arm_mg = MoveGroup(
            "right_arm", self._jqpos[22:29], self._jdof[22:29], self._act_ids[22:29]
        )
        right_gripper_mg = MoveGroup(
            "right_gripper",
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([self._rgripper]),
        )

        self._legs_waist_ctrl = LegsWaistController(legs_waist_mg, owner=self)
        self._left_arm_ctrl = LeftArmController(left_arm_mg)
        self._right_arm_ctrl = RightArmController(right_arm_mg)
        self._right_gripper_ctrl = RightGripperController(right_gripper_mg)
        self._controllers = [
            self._legs_waist_ctrl,
            self._left_arm_ctrl,
            self._right_arm_ctrl,
            self._right_gripper_ctrl,
        ]

    def _set_groot_defaults(self):
        d = self._data
        for i in range(15):
            d.qpos[self._jqpos[i]] = _WBC_DEFAULT[i]
        # Left arm keeps DEFAULT_QPOS (unactuated, hangs); right arm zeroed for the policy to set.
        for i in range(15, 22):
            d.qpos[self._jqpos[i]] = _ROBOT_DEFAULT_QPOS[i]
        for i in range(22, _WBC_N):
            d.qpos[self._jqpos[i]] = 0.0
        d.qpos[self._fj_qa + 2] = 0.793

    def _reset_wbc_state(self):
        # Same reset LegsWaistController.reset() would perform (it operates
        # on these same owner attributes) -- kept as the original direct
        # inline reset rather than routed through the controller, so this
        # method's behavior is trivially identical to gold's.
        self._cmd[:] = 0
        self._wbc_action[:] = 0
        self._target_lower[:] = _WBC_DEFAULT
        self._height_cmd = _WBC_HEIGHT_CMD
        self._waist_target[:] = 0
        self._step_counter = 0
        self._obs_history.clear()
        for _ in range(_WBC_OBS_HISTORY):
            self._obs_history.append(np.zeros(_WBC_OBS_DIM, dtype=np.float32))

    def _write_default_pose(self):
        d = self._data
        for i in range(15):
            d.qpos[self._jqpos[i]] = _WBC_DEFAULT[i]
        for i in range(15, 22):
            d.qpos[self._jqpos[i]] = _ROBOT_DEFAULT_QPOS[i]
        for i in range(22, _WBC_N):
            d.qpos[self._jqpos[i]] = 0.0
        d.qpos[self._fj_qa + 2] = 0.793
        d.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, d)

        self._target_q[:15] = _WBC_DEFAULT
        self._target_q[15:22] = _ROBOT_DEFAULT_QPOS[15:22]
        self._target_q[22:] = 0.0
        q = d.qpos[self._jqpos].astype(np.float32)
        dq = d.qvel[self._jdof].astype(np.float32)
        tau = (self._target_q[:15] - q[:15]) * _WBC_KP[:15] - dq[:15] * _WBC_KD[:15]
        for i in range(12, 15):
            tau[i] += d.qfrc_bias[self._jdof[i]]
        for i in range(15):
            if self._act_ids[i] >= 0:
                d.ctrl[self._act_ids[i]] = tau[i]
        for i in range(15, _WBC_N):
            if self._act_ids[i] >= 0:
                d.ctrl[self._act_ids[i]] = self._target_q[i]

    def execute_action(self, action):
        """Dispatcher: parse the flat 15-dim action into per-move-group
        targets, set_target on each Controller, then compute_ctrl_inputs +
        set_ctrl for each -- in the SAME order gold's single execute_action
        body used (legs_waist tau computed from the *previous* call's
        _target_lower, written to ctrl, THEN [if due] the WBC ONNX inference
        runs and updates _target_lower for the *next* call -- this ordering
        is load-bearing and must not change).
        """
        cmd, height_cmd, waist = action[ACT_CMD], action[ACT_HEIGHT], action[ACT_WAIST]
        r_arm, r_grip = action[ACT_ARM], action[ACT_GRIP]

        self._legs_waist_ctrl.set_target((cmd, height_cmd, waist))
        self._right_arm_ctrl.set_target(r_arm)
        self._right_gripper_ctrl.set_target(r_grip)

        # Legacy _target_q bookkeeping, kept verbatim even though nothing
        # currently reads _target_q[:15]/[22:29] afterward (only [15:] is
        # read elsewhere, at reset time, by agents/policy_g1ms.py) -- cheap
        # to preserve exactly, so left in the one place gold itself had it.
        self._target_q[:15] = self._target_lower
        self._target_q[12:15] = waist
        self._target_q[15:22] = 0.0
        self._target_q[22:29] = r_arm

        d = self._data
        for controller in self._controllers:
            values = controller.compute_ctrl_inputs(d, self._step_counter)
            controller.move_group.set_ctrl(d, values)
