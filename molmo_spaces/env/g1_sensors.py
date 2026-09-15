"""G1 observation sensors (the 14 keys of g1_molmo's obs dict) on molmo_spaces'
Sensor ABC; the math is g1_molmo's, bit-identical to gold.

They read `env.current_data` and ignore `batch_index`, unlike the rest of
molmo_spaces' sensors; changing that can change numbers, so run the strict
gate in fetchman/scripts/check_gold_parity.py around it.
PelvisFrameSensor and TargetPointInHeadSensor are helpers, not sensors: they
return a closure and a raw (u, v)-or-None tuple, not gym-shaped arrays.
"""

import gymnasium.spaces as spaces
import mujoco
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.robots.g1 import JOINT_NAMES, PREFIX

_IDS_ATTR = "_g1_sensor_ids"


class _ObsIds:
    """The MJCF ids these sensors index with, for one model."""

    __slots__ = ("model", "pelvis_bid", "fj_dadr", "qpos_ids", "r_sid", "r_grip_qa")


def _obs_ids(env) -> _ObsIds:
    """Joint/site/body ids from `env.current_model`, cached until the model
    changes, so these sensors run on any env with a `current_model`."""
    model = env.current_model
    ids = getattr(env, _IDS_ATTR, None)
    if ids is not None and ids.model is model:
        return ids

    def _jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{name}")

    ids = _ObsIds()
    ids.model = model
    ids.qpos_ids = np.array([model.jnt_qposadr[_jid(n)] for n in JOINT_NAMES])
    ids.fj_dadr = model.jnt_dofadr[_jid("floating_base_joint")]
    ids.r_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{PREFIX}right_grasp")
    ids.r_grip_qa = model.jnt_qposadr[_jid("right_Joint1_1")]
    ids.pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX}pelvis")
    setattr(env, _IDS_ATTR, ids)
    return ids


def _mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


def _base_rpy(env):
    pm = env.current_data.xmat[_obs_ids(env).pelvis_bid]
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

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.current_data
        pos = d.xpos[_obs_ids(env).pelvis_bid].copy()
        mat = d.xmat[_obs_ids(env).pelvis_bid].reshape(3, 3).copy()
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

    def get_observation(self, env, task=None, *args, **kwargs):
        tgt = task.target
        # The head camera's MJCF id and its cubemap FisheyeRenderer both come
        # from the env's CameraManager (camera_manager.MjcfCameraInfo, filled
        # in for a FisheyeMjcfCameraConfig); G1CPUMujocoEnv exposes them by
        # observation key. An env without them has no projection to report,
        # which is exactly this sensor's documented "unavailable" case
        # (TargetPointSensor turns None into its (-1, -1) out-of-frame sentinel).
        head_id = env.mjcf_camera_id("head_image") if hasattr(env, "mjcf_camera_id") else -1
        if head_id < 0 or tgt is None or not hasattr(env, "ensure_fisheye"):
            return None
        fisheye = env.ensure_fisheye()
        d = env.current_data
        cam_pos = d.cam_xpos[head_id]
        cam_mat = d.cam_xmat[head_id].reshape(3, 3)
        p_cam = cam_mat.T @ (tgt.position(d) - cam_pos)
        return fisheye.project_camera_point(p_cam)


TARGET_POINT_IN_HEAD_SENSOR = TargetPointInHeadSensor()


class BasePositionSensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="base_position", observation_space=spaces.Box(-np.inf, np.inf, shape=(2,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        return env.current_robot.get_xy().astype(np.float32)


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
        super().__init__(
            uuid="last_base_vel_cmd", observation_space=spaces.Box(-np.inf, np.inf, shape=(3,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        cmd = getattr(env, "_last_base_vel_cmd", None)
        if cmd is not None:
            return np.asarray(cmd, dtype=np.float32).copy()
        # Envs that do not record the command (CPUMujocoEnv): read it off the
        # legs_waist WBC controller's `_cmd`. Differs from the port's record
        # only when action noise is on (the port records pre-noise).
        controller = env.current_robot.controllers.get("legs_waist")
        cmd = getattr(controller, "_cmd", None) if controller is not None else None
        if cmd is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(cmd[:3], dtype=np.float32).copy()


class BaseHeightSensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="base_height", observation_space=spaces.Box(-np.inf, np.inf, shape=(1,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        return np.array([env.current_robot.pelvis_height()], dtype=np.float32)


class BaseVelocitySensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="base_velocity", observation_space=spaces.Box(-np.inf, np.inf, shape=(3,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.current_data
        return d.qvel[_obs_ids(env).fj_dadr : _obs_ids(env).fj_dadr + 3].astype(np.float32)


class BaseAngularVelocitySensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="base_angular_velocity", observation_space=spaces.Box(-np.inf, np.inf, shape=(3,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.current_data
        return d.qvel[_obs_ids(env).fj_dadr + 3 : _obs_ids(env).fj_dadr + 6].astype(np.float32)


class JointPosSensor(Sensor):
    def __init__(self, n_joints):
        super().__init__(
            uuid="joint_pos", observation_space=spaces.Box(-np.inf, np.inf, shape=(n_joints,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        return env.current_data.qpos[_obs_ids(env).qpos_ids].astype(np.float32)


class UpperJointPosSensor(Sensor):
    def __init__(self, n_upper_joints):
        super().__init__(
            uuid="upper_joint_pos",
            observation_space=spaces.Box(-np.inf, np.inf, shape=(n_upper_joints,)),
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        joint_pos = env.current_data.qpos[_obs_ids(env).qpos_ids].astype(np.float32)
        return np.concatenate([joint_pos[12:15], joint_pos[22:30]])


class RightHandPoseSensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="right_hand_pose", observation_space=spaces.Box(-np.inf, np.inf, shape=(7,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.current_data
        to_local = PELVIS_FRAME_SENSOR.get_observation(env)
        r_quat = _mat_to_quat(d.site_xmat[_obs_ids(env).r_sid])
        return to_local(d.site_xpos[_obs_ids(env).r_sid], r_quat).astype(np.float32)


class RightGripperPosSensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="right_gripper_pos", observation_space=spaces.Box(-0.0222, 0.0245, shape=(1,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        return np.array([env.current_data.qpos[_obs_ids(env).r_grip_qa]], dtype=np.float32)


class TargetObjectPoseSensor(Sensor):
    def __init__(self):
        super().__init__(
            uuid="target_object_pose", observation_space=spaces.Box(-np.inf, np.inf, shape=(7,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        d = env.current_data
        tgt = task.target
        to_local = PELVIS_FRAME_SENSOR.get_observation(env)
        return to_local(tgt.position(d), tgt.quat(d)).astype(np.float32)


class TargetPointSensor(Sensor):
    """(u, v) pixel of the target object's position in the rendered head
    fisheye image, normalized to [0,1], or (-1,-1) if behind the camera /
    unavailable. See TargetPointInHeadSensor above."""

    def __init__(self):
        super().__init__(
            uuid="target_point", observation_space=spaces.Box(-np.inf, np.inf, shape=(2,))
        )

    def get_observation(self, env, task=None, *args, **kwargs):
        pt = TARGET_POINT_IN_HEAD_SENSOR.get_observation(env, task)
        if pt is None:
            return np.array([-1.0, -1.0], dtype=np.float32)
        H, W = env.camera_size
        return np.array([pt[0] / W, pt[1] / H], dtype=np.float32)


# joint_pos layout (30): legs[0:12] waist[12:15] left_arm[15:22] right_arm[22:29] right_grip[29].
_N_JOINTS = 30
_N_UPPER_JOINTS = 11


def robot_sensors() -> list[Sensor]:
    """The robot-state half of the observation, built fresh per call --
    what G1Robot.create_robot_sensors returns, matching how every other robot
    supplies its own sensors (see robots/franka.py, robots/rby1.py)."""
    return [
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
    ]


def task_sensors() -> list[Sensor]:
    """The task half: everything about the episode's target object. Built by
    G1Task._create_sensor_suite_from_config, as in BaseMujocoTask."""
    return [
        TargetObjectPoseSensor(),
        TargetPointSensor(),
    ]
