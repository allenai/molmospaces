"""Sensor classes for the `_g1ms` port -- relocated out of env_g1ms.py to its
own module, matching molmo_spaces' own layout (molmo_spaces/env/sensors.py,
abstract_sensors.py, sensors_cameras.py, rby1_sensors.py all live separately
from env.py). Only consumer is G1TaskSampler (tasks/pick_task_sampler_g1ms.py,
via OBS_SENSORS/TARGET_POINT_IN_HEAD_SENSOR).

Merge step (see molmo_spaces/g1_molmo_port/__init__.py's own docstring on
the "wholesale port, then iteratively merge" plan): the 14 real obs-dict
sensors below now subclass molmo_spaces' own Sensor ABC (molmo_spaces/env/
abstract_sensors.py) and are collected into a real SensorSuite, instead of
g1_molmo's own ad hoc `uuid` + `get_observation(env)` shape. Every sensor's
own math is untouched (relocated verbatim from the original inline
_build_obs, same as before this merge step) -- only the wrapping changed, so
results stay bit-identical to gold.

PelvisFrameSensor and TargetPointInHeadSensor are NOT part of OBS_SENSORS/
the SensorSuite -- they return a closure and a raw (u, v)-or-None tuple
respectively, not `gym.Space`-shaped array data, so wrapping them in the
real Sensor ABC (which requires a real observation_space) would misrepresent
their contract. They stay plain helper classes, called directly by the
other sensors below and by G1TaskSampler's own _target_visible_in_head.
"""

import gymnasium.spaces as spaces
import mujoco
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor


def _mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


def _base_rpy(env):
    pm = env.scene.data.xmat[env._obs_pelvis_bid]
    pitch = np.arcsin(-pm[6])
    roll = np.arctan2(pm[7], pm[8])
    yaw = np.arctan2(pm[3], pm[0])
    return np.array([roll, pitch, yaw], dtype=np.float32)


class PelvisFrameSensor:
    """NOT an obs-dict entry -- see this module's own docstring for why this
    isn't a real Sensor subclass. A shared helper other sensors call for
    world-to-pelvis-local-frame math. Returns a `to_local(pos_w, quat_w=None)`
    closure, not an array -- callers use the module-level
    PELVIS_FRAME_SENSOR instance below rather than constructing their own.
    """

    uuid = "_pelvis_frame"

    def get_observation(self, env):
        d = env.scene.data
        pos = d.xpos[env._obs_pelvis_bid].copy()
        mat = d.xmat[env._obs_pelvis_bid].reshape(3, 3).copy()
        quat_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_inv, _mat_to_quat(mat))

        def to_local(pos_w, quat_w=None):
            rel_pos = mat.T @ (pos_w - pos)
            if quat_w is None:
                return rel_pos
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, quat_inv, quat_w)
            if q[0] < 0:
                q = -q
            return np.concatenate([rel_pos, q])

        return to_local


PELVIS_FRAME_SENSOR = PelvisFrameSensor()


class TargetPointInHeadSensor:
    """NOT an obs-dict entry -- see this module's own docstring for why this
    isn't a real Sensor subclass. TargetPointSensor wraps this to produce the
    actual `target_point` obs key (normalized, with an out-of-frame
    sentinel); G1TaskSampler's _target_visible_in_head (tasks/
    pick_task_sampler_g1ms.py) also calls it directly for grasp-visibility
    termination. Returns a raw (u, v) pixel tuple, or None if behind the
    camera / unavailable, not an array.
    """

    uuid = "_target_point_in_head"

    def get_observation(self, env):
        head_id = env.camera_manager.ids.get("head_image", -1)
        tgt = env.target
        if head_id < 0 or tgt is None:
            return None
        if env.camera_manager.fisheye is None:
            env._ensure_fisheye("head_pov", *env.camera_manager.size, 512)
        d = env.scene.data
        cam_pos = d.cam_xpos[head_id]
        cam_mat = d.cam_xmat[head_id].reshape(3, 3)
        p_cam = cam_mat.T @ (tgt.position(d) - cam_pos)
        return env.camera_manager.fisheye.project_camera_point(p_cam)


TARGET_POINT_IN_HEAD_SENSOR = TargetPointInHeadSensor()


class BasePositionSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_position", observation_space=spaces.Box(-np.inf, np.inf, shape=(2,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return env.robot.get_xy().astype(np.float32)


class BaseYawSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_yaw", observation_space=spaces.Box(-np.pi, np.pi, shape=(1,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return np.array([_base_rpy(env)[2]], dtype=np.float32)


class BaseRPYSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_rpy", observation_space=spaces.Box(-np.pi, np.pi, shape=(3,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return _base_rpy(env)


class BaseRPSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_rp", observation_space=spaces.Box(-np.pi, np.pi, shape=(2,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return _base_rpy(env)[:2].astype(np.float32)


class LastBaseVelCmdSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="last_base_vel_cmd", observation_space=spaces.Box(-np.inf, np.inf, shape=(3,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return env._last_base_vel_cmd.copy()


class BaseHeightSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_height", observation_space=spaces.Box(-np.inf, np.inf, shape=(1,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return np.array([env.robot.pelvis_height()], dtype=np.float32)


class BaseVelocitySensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_velocity", observation_space=spaces.Box(-np.inf, np.inf, shape=(3,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.scene.data
        return d.qvel[env._obs_fj_dadr : env._obs_fj_dadr + 3].astype(np.float32)


class BaseAngularVelocitySensor(Sensor):
    def __init__(self):
        super().__init__(uuid="base_angular_velocity", observation_space=spaces.Box(-np.inf, np.inf, shape=(3,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.scene.data
        return d.qvel[env._obs_fj_dadr + 3 : env._obs_fj_dadr + 6].astype(np.float32)


class JointPosSensor(Sensor):
    def __init__(self, n_joints):
        super().__init__(uuid="joint_pos", observation_space=spaces.Box(-np.inf, np.inf, shape=(n_joints,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return env.scene.data.qpos[env._obs_qpos_ids].astype(np.float32)


class UpperJointPosSensor(Sensor):
    def __init__(self, n_upper_joints):
        super().__init__(
            uuid="upper_joint_pos", observation_space=spaces.Box(-np.inf, np.inf, shape=(n_upper_joints,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        joint_pos = env.scene.data.qpos[env._obs_qpos_ids].astype(np.float32)
        return np.concatenate([joint_pos[12:15], joint_pos[22:30]])


class RightHandPoseSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="right_hand_pose", observation_space=spaces.Box(-np.inf, np.inf, shape=(7,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.scene.data
        to_local = PELVIS_FRAME_SENSOR.get_observation(env)
        r_quat = _mat_to_quat(d.site_xmat[env._obs_r_sid])
        return to_local(d.site_xpos[env._obs_r_sid], r_quat).astype(np.float32)


class RightGripperPosSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="right_gripper_pos", observation_space=spaces.Box(-0.0222, 0.0245, shape=(1,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        return np.array([env.scene.data.qpos[env._obs_r_grip_qa]], dtype=np.float32)


class TargetObjectPoseSensor(Sensor):
    def __init__(self):
        super().__init__(uuid="target_object_pose", observation_space=spaces.Box(-np.inf, np.inf, shape=(7,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.scene.data
        tgt = env.task.target
        to_local = PELVIS_FRAME_SENSOR.get_observation(env)
        return to_local(tgt.position(d), tgt.quat(d)).astype(np.float32)


class TargetPointSensor(Sensor):
    """(u, v) pixel of the target object's position in the rendered head
    fisheye image, normalized to [0,1], or (-1,-1) if behind the camera /
    unavailable. See TargetPointInHeadSensor above."""

    def __init__(self):
        super().__init__(uuid="target_point", observation_space=spaces.Box(-np.inf, np.inf, shape=(2,)))

    def get_observation(self, env, task=None, *args, **kwargs):
        pt = TARGET_POINT_IN_HEAD_SENSOR.get_observation(env)
        if pt is None:
            return np.array([-1.0, -1.0], dtype=np.float32)
        H, W = env.camera_manager.size
        return np.array([pt[0] / W, pt[1] / H], dtype=np.float32)


# joint_pos layout (30): legs[0:12] waist[12:15] left_arm[15:22] right_arm[22:29] right_grip[29].
_N_JOINTS = 30
_N_UPPER_JOINTS = 11

OBS_SENSORS = [
    BasePositionSensor(),
    BaseYawSensor(),
    BaseRPYSensor(),
    BaseRPSensor(),
    LastBaseVelCmdSensor(),
    BaseHeightSensor(),
    BaseVelocitySensor(),
    BaseAngularVelocitySensor(),
    JointPosSensor(_N_JOINTS),
    UpperJointPosSensor(_N_UPPER_JOINTS),
    RightHandPoseSensor(),
    RightGripperPosSensor(),
    TargetObjectPoseSensor(),
    TargetPointSensor(),
]
