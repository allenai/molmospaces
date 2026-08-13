"""Whole-body walking controller for the G1 humanoid (Phase 3).

Ports the WBC piece of the reference `g1_molmo` stack's `G1Controller`
(`components/controller.py`) into a single `Controller` implementing
molmospaces' `Controller` ABC, bound to the combined 15-DOF `legs_waist`
MoveGroup (see `G1LegsWaistGroup`).

Not ported from the reference: the surrounding nav/grasp policy
(`agents/policy.py`, ~1450 lines) that decides *what* velocity/height/waist
command to send each tick against g1_molmo's own occupancy-grid/task
abstractions. That's a separate integration (bridging molmospaces' own
task samplers/policies to this controller's `set_target()` interface), not
part of this port.

Known fidelity gap vs. the reference: the reference zeroes the left arm's
actuator gain so it hangs passively under gravity (matching what the ONNX
policy was trained against). Here the left arm keeps Phase 2's active
JointPosController holding its init pose instead, to avoid a second
actuator-reconfiguration edge case up front. If walking stability suffers,
this is the first place to revisit.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import onnxruntime as ort

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.robots.robot_views.abstract import MoveGroup

# Action layout for set_target(): [cmd_vx, cmd_vy, cmd_yaw_rate, height, waist(3)].
# (Unlike the reference's 15-dim external action, right_arm/right_gripper are NOT
# part of this controller's target -- those stay on their own JointPosControllers.)
NUM_TARGET_DIMS = 7

_NUM_ACTIONS = 15  # ONNX model output: legs(12) + waist(3) offsets
_DEFAULT_POSE = np.array(
    [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], dtype=np.float32
)
_CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
_ACTION_SCALE = 0.25
_ANG_VEL_SCALE = 0.5
_DOF_VEL_SCALE = 0.05
_DEFAULT_HEIGHT_CMD = 0.74
_OBS_HISTORY = 6
# g1_molmo's reference (components/controller.py) uses this exact decimation
# constant (4) at its own 0.005s physics timestep, invoking the WBC ONNX
# policies once every 4*5ms=20ms (~50Hz) -- what they were actually
# trained/tuned at. G1Config.physics_timestep now applies that same 0.005s
# timestep to our own model too (see its docstring and G1Robot.__init__),
# instead of leaving G1 at the scene's own 0.002s default -- so this can be
# g1_molmo's own raw "4" again, invoking the WBC at the matching 4*5ms=20ms.
#
# Earlier history, for context if physics_timestep ever gets removed/unset:
# at the scene's 0.002s default, the same raw "4" would invoke the WBC at
# 4*2ms=8ms (~125Hz) instead -- 2.5x too fast -- and this constant was scaled
# up to 10 (10*2ms=20ms) to preserve the correct ~50Hz rate without touching
# the physics timestep. Confirmed empirically at the time: at ~125Hz the
# WBC's real walking gait degrades badly under combined turn+translate
# commands (near-zero net displacement despite a sustained,
# correctly-computed nonzero command) in a way it does not at the intended
# ~50Hz -- this was the dominant cause of FetchmanPickPlannerPolicy's walk
# phase stalling indefinitely partway through a multi-waypoint path. If
# G1Config.physics_timestep is ever removed for some G1 variant, this
# constant must go back to a value that reproduces ~50Hz at whatever the
# real timestep is then (10 at 0.002s, 4 at 0.005s, etc.) -- it is NOT
# self-adjusting.
_CONTROL_DECIMATION = 4
_OBS_DIM = 86
_N = 29  # legs(12) + waist(3) + left_arm(7) + right_arm(7); gripper excluded

_KP = np.array(
    [150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250], dtype=np.float32
)
_KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5], dtype=np.float32)


def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Project the world +z gravity direction into the body frame given a wxyz quaternion."""
    w, x, y, z = quat
    c = np.array([w, -x, -y, -z])
    return np.array(
        [
            2 * (c[1] * c[3] + c[0] * c[2]) * -1,
            2 * (c[2] * c[3] - c[0] * c[1]) * -1,
            -(c[0] ** 2 - c[1] ** 2 - c[2] ** 2 + c[3] ** 2),
        ],
        dtype=np.float32,
    )


class G1WalkController(Controller):
    """Whole-body PD-torque + ONNX walking policy for G1's combined legs+waist group.

    Deliberately a plain `Controller` (not `AbstractPositionController`): the target
    here (cmd velocity/height/waist) isn't a joint position, and `Robot._apply_action_noise_
    and_save_unnoised_cmd_jp` only special-cases `AbstractPositionController` instances
    (for TCP-noise and `target_pos` bookkeeping) -- neither applies to this controller.
    """

    def __init__(
        self,
        robot_move_group: MoveGroup,
        base_move_group: MoveGroup,
        left_arm_move_group: MoveGroup,
        right_arm_move_group: MoveGroup,
        models_dir: Path | str,
    ) -> None:
        super().__init__(robot_move_group)
        self._base = base_move_group
        self._left_arm = left_arm_move_group
        self._right_arm = right_arm_move_group

        d = Path(models_dir)
        # Pin to a single thread: onnxruntime's default multi-threaded CPU
        # execution reduces floating-point ops in a non-deterministic order,
        # so the *same* seed can produce a different WBC output (and thus a
        # different physics trajectory) from one run to the next -- confirmed
        # by running the same episode twice back to back and getting a
        # different pick outcome each time until this was pinned. These are
        # small (~1MB) MLP-sized policies, so the throughput cost is minor.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        self._stand_sess = ort.InferenceSession(
            str(d / "groot_balance.onnx"),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._walk_sess = ort.InferenceSession(
            str(d / "groot_walk.onnx"), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

        self._stationary = True
        self._cmd = np.zeros(3, dtype=np.float32)
        self._height_cmd = _DEFAULT_HEIGHT_CMD
        self._waist_target = np.zeros(3, dtype=np.float32)
        self._target = np.concatenate([self._cmd, [self._height_cmd], self._waist_target])

        self._wbc_action = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        self._target_lower = _DEFAULT_POSE.copy()
        self._obs_history: collections.deque = collections.deque(maxlen=_OBS_HISTORY)
        self._obs_buffer = np.zeros(_OBS_DIM * _OBS_HISTORY, dtype=np.float32)
        self._step_counter = 0

        self.reset()

    @property
    def target(self):
        return self._target.copy()

    @property
    def stationary(self):
        return self._stationary

    def set_target(self, target) -> None:
        """Args: target: [cmd_vx, cmd_vy, cmd_yaw_rate, height, waist_yaw, waist_roll, waist_pitch]."""
        target = np.asarray(target, dtype=np.float32)
        assert target.shape == (NUM_TARGET_DIMS,), (
            f"Expected shape ({NUM_TARGET_DIMS},), got {target.shape}"
        )
        self._stationary = False
        self._target = target.copy()
        self._cmd = target[0:3]
        self._height_cmd = float(target[3])
        self._waist_target = target[4:7]

    def set_to_stationary(self) -> None:
        # "Stationary" for a bipedal humanoid still means actively balancing in place
        # (zero velocity command), not holding a fixed joint position like other
        # controllers' stationary mode -- compute_ctrl_inputs() keeps running either way.
        self._stationary = True
        self._cmd[:] = 0.0

    def _proprioception_29(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint positions/velocities for legs(12)+waist(3)+left_arm(7)+right_arm(7)."""
        qj = np.concatenate(
            [self.robot_move_group.joint_pos, self._left_arm.joint_pos, self._right_arm.joint_pos]
        ).astype(np.float32)
        dqj = np.concatenate(
            [self.robot_move_group.joint_vel, self._left_arm.joint_vel, self._right_arm.joint_vel]
        ).astype(np.float32)
        assert qj.shape == (_N,) and dqj.shape == (_N,)
        return qj, dqj

    def compute_ctrl_inputs(self):
        legs_waist_q = self.robot_move_group.joint_pos.astype(np.float32)
        legs_waist_dq = self.robot_move_group.joint_vel.astype(np.float32)

        target_q = self._target_lower.copy()
        target_q[12:15] = self._waist_target

        tau = (target_q - legs_waist_q) * _KP - legs_waist_dq * _KD
        # Gravity/coriolis compensation for the waist DOFs only (matches reference).
        waist_dofadr = self.robot_move_group.joint_veladr[12:15]
        tau[12:15] += self.robot_move_group.mj_data.qfrc_bias[waist_dofadr]

        if self._step_counter % _CONTROL_DECIMATION == 0:
            qj, dqj = self._proprioception_29()
            base_pos = self._base.joint_pos.astype(np.float32)
            base_vel = self._base.joint_vel.astype(np.float32)
            quat = base_pos[3:7]
            omega = base_vel[3:6]

            padded = np.zeros(_N, dtype=np.float32)
            padded[:15] = _DEFAULT_POSE

            single_obs = np.zeros(_OBS_DIM, dtype=np.float32)
            single_obs[0:3] = self._cmd * _CMD_SCALE
            single_obs[3] = self._height_cmd
            single_obs[4:7] = [self._waist_target[1], self._waist_target[2], self._waist_target[0]]
            single_obs[7:10] = omega * _ANG_VEL_SCALE
            single_obs[10:13] = _gravity_orientation(quat)
            single_obs[13:42] = qj - padded
            single_obs[42:71] = dqj * _DOF_VEL_SCALE
            single_obs[71:86] = self._wbc_action

            self._obs_history.append(single_obs)
            for i, h in enumerate(self._obs_history):
                self._obs_buffer[i * _OBS_DIM : (i + 1) * _OBS_DIM] = h
            obs_tensor = self._obs_buffer[None, :].astype(np.float32)

            sess = self._stand_sess if np.linalg.norm(self._cmd) < 0.05 else self._walk_sess
            self._wbc_action = sess.run(None, {sess.get_inputs()[0].name: obs_tensor})[0][0].astype(
                np.float32
            )
            self._target_lower = self._wbc_action * _ACTION_SCALE + _DEFAULT_POSE

        self._step_counter += 1
        return tau

    def reset(self) -> None:
        self._stationary = True
        self._cmd[:] = 0.0
        self._height_cmd = _DEFAULT_HEIGHT_CMD
        self._waist_target[:] = 0.0
        self._target = np.concatenate([self._cmd, [self._height_cmd], self._waist_target])
        self._wbc_action[:] = 0.0
        self._target_lower = _DEFAULT_POSE.copy()
        self._step_counter = 0
        self._obs_history.clear()
        for _ in range(_OBS_HISTORY):
            self._obs_history.append(np.zeros(_OBS_DIM, dtype=np.float32))
