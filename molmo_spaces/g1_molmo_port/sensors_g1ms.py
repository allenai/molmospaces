"""Sensor classes for the `_g1ms` port -- relocated out of env_g1ms.py to its
own module, matching molmo_spaces' own layout (molmo_spaces/env/sensors.py,
abstract_sensors.py, sensors_cameras.py, rby1_sensors.py all live separately
from env.py). Only consumer is G1TaskSampler (tasks/pick_task_sampler_g1ms.py,
via OBS_SENSORS/TARGET_POINT_IN_HEAD_SENSOR) -- G1Env itself never reads
these names, so env_g1ms.py doesn't need to import this module at all.
"""

import mujoco
import numpy as np


def _mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


class Sensor:
    """Minimal Sensor abstraction mirroring molmo_spaces/env/abstract_sensors.py's
    contract (a `uuid` obs-dict key + `get_observation(env)`) -- self-contained
    here since env_g1ms.py can't import across repos/conda envs, same shape
    only. One instance per _build_obs() dict key; every sensor's math below is
    relocated verbatim from the original inline _build_obs (see env.py), not
    rewritten, so results stay bit-identical to gold.
    """

    uuid: str

    def get_observation(self, env):
        raise NotImplementedError


def _base_rpy(env):
    pm = env.scene.data.xmat[env._obs_pelvis_bid]
    pitch = np.arcsin(-pm[6])
    roll = np.arctan2(pm[7], pm[8])
    yaw = np.arctan2(pm[3], pm[0])
    return np.array([roll, pitch, yaw], dtype=np.float32)


class BasePositionSensor(Sensor):
    uuid = "base_position"

    def get_observation(self, env):
        return env.robot.get_xy().astype(np.float32)


class BaseYawSensor(Sensor):
    uuid = "base_yaw"

    def get_observation(self, env):
        return np.array([_base_rpy(env)[2]], dtype=np.float32)


class BaseRPYSensor(Sensor):
    uuid = "base_rpy"

    def get_observation(self, env):
        return _base_rpy(env)


class BaseRPSensor(Sensor):
    uuid = "base_rp"

    def get_observation(self, env):
        return _base_rpy(env)[:2].astype(np.float32)


class LastBaseVelCmdSensor(Sensor):
    uuid = "last_base_vel_cmd"

    def get_observation(self, env):
        return env._last_base_vel_cmd.copy()


class BaseHeightSensor(Sensor):
    uuid = "base_height"

    def get_observation(self, env):
        return np.array([env.robot.pelvis_height()], dtype=np.float32)


class BaseVelocitySensor(Sensor):
    uuid = "base_velocity"

    def get_observation(self, env):
        d = env.scene.data
        return d.qvel[env._obs_fj_dadr : env._obs_fj_dadr + 3].astype(np.float32)


class BaseAngularVelocitySensor(Sensor):
    uuid = "base_angular_velocity"

    def get_observation(self, env):
        d = env.scene.data
        return d.qvel[env._obs_fj_dadr + 3 : env._obs_fj_dadr + 6].astype(np.float32)


class JointPosSensor(Sensor):
    uuid = "joint_pos"

    def get_observation(self, env):
        return env.scene.data.qpos[env._obs_qpos_ids].astype(np.float32)


class UpperJointPosSensor(Sensor):
    uuid = "upper_joint_pos"

    def get_observation(self, env):
        joint_pos = env.scene.data.qpos[env._obs_qpos_ids].astype(np.float32)
        return np.concatenate([joint_pos[12:15], joint_pos[22:30]])


class PelvisFrameSensor(Sensor):
    """NOT an obs-dict entry -- not registered in OBS_SENSORS, never appears
    in _build_obs()'s output. A shared helper other sensors call for
    world-to-pelvis-local-frame math, expressed as a Sensor-shaped class
    (get_observation(env)) for consistency with everything else that reads
    env/robot state, rather than being a bare G1Env method. Returns a
    `to_local(pos_w, quat_w=None)` closure, not an array -- callers use the
    module-level `PELVIS_FRAME_SENSOR` instance below rather than
    constructing their own.
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


class TargetPointInHeadSensor(Sensor):
    """NOT an obs-dict entry -- not registered in OBS_SENSORS.
    TargetPointSensor wraps this to produce the actual `target_point` obs
    key (normalized, with an out-of-frame sentinel); G1TaskSampler's
    _target_visible_in_head (tasks/pick_task_sampler_g1ms.py) also calls it
    directly for grasp-visibility termination. Returns a raw (u, v) pixel
    tuple, or None if behind the camera / unavailable, not an array.
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


class RightHandPoseSensor(Sensor):
    uuid = "right_hand_pose"

    def get_observation(self, env):
        d = env.scene.data
        to_local = PELVIS_FRAME_SENSOR.get_observation(env)
        r_quat = _mat_to_quat(d.site_xmat[env._obs_r_sid])
        return to_local(d.site_xpos[env._obs_r_sid], r_quat).astype(np.float32)


class RightGripperPosSensor(Sensor):
    uuid = "right_gripper_pos"

    def get_observation(self, env):
        return np.array([env.scene.data.qpos[env._obs_r_grip_qa]], dtype=np.float32)


class TargetObjectPoseSensor(Sensor):
    uuid = "target_object_pose"

    def get_observation(self, env):
        d = env.scene.data
        tgt = env.task.target
        to_local = PELVIS_FRAME_SENSOR.get_observation(env)
        return to_local(tgt.position(d), tgt.quat(d)).astype(np.float32)


class TargetPointSensor(Sensor):
    """(u, v) pixel of the target object's position in the rendered head
    fisheye image, normalized to [0,1], or (-1,-1) if behind the camera /
    unavailable. See TargetPointInHeadSensor above."""

    uuid = "target_point"

    def get_observation(self, env):
        pt = TARGET_POINT_IN_HEAD_SENSOR.get_observation(env)
        if pt is None:
            return np.array([-1.0, -1.0], dtype=np.float32)
        H, W = env.camera_manager.size
        return np.array([pt[0] / W, pt[1] / H], dtype=np.float32)


OBS_SENSORS = [
    BasePositionSensor(),
    BaseYawSensor(),
    BaseRPYSensor(),
    BaseRPSensor(),
    LastBaseVelCmdSensor(),
    BaseHeightSensor(),
    BaseVelocitySensor(),
    BaseAngularVelocitySensor(),
    JointPosSensor(),
    UpperJointPosSensor(),
    RightHandPoseSensor(),
    RightGripperPosSensor(),
    TargetObjectPoseSensor(),
    TargetPointSensor(),
]
