"""Headless closed-loop replay of recorded joint targets in the molmospaces robot model.

The model is built exactly as the envs build it (base_scene.xml physics options, the
robot config's `add_robot_to_scene` + `apply_control_overrides`), so the fit reflects
what molmospaces actually simulates, and a MuJoCo upgrade is picked up automatically.
Commanded targets are held for one control period (zero-order hold), like the
envs' `policy_dt` loop.
"""

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

import molmo_spaces
from molmo_spaces.configs.robot_configs import BaseRobotConfig

BASE_SCENE = Path(molmo_spaces.__file__).parent / "resources" / "base_scene.xml"
PARAM_NAMES = ("kp", "kv", "frictionloss", "armature")


def build_model(robot_config: BaseRobotConfig) -> mujoco.MjModel:
    assert robot_config.robot_cls is not None
    spec = mujoco.MjSpec.from_file(str(BASE_SCENE))
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=[0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    robot_config.robot_cls.apply_control_overrides(spec, robot_config)
    return spec.compile()


@dataclass
class ArmReplay:
    """Replays q_des through the arm's position actuators and returns the simulated q."""

    model: mujoco.MjModel
    namespace: str
    joint_names: list[str]  # un-namespaced, e.g. fr3_joint1..7
    actuator_names: list[str]
    control_dt: float
    gripper_ctrl: float = 0.0  # all non-arm actuators are held at this (Robotiq: 0 = open)
    data: mujoco.MjData = field(init=False)

    def __post_init__(self) -> None:
        m = self.model
        self.data = mujoco.MjData(m)
        jids = [m.joint(self.namespace + n).id for n in self.joint_names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dofadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.actid = np.array([m.actuator(self.namespace + n).id for n in self.actuator_names])
        self.n_substeps = int(round(self.control_dt / m.opt.timestep))
        assert abs(self.n_substeps * m.opt.timestep - self.control_dt) < 1e-9, (
            f"control_dt {self.control_dt} is not a multiple of timestep {m.opt.timestep}"
        )

    # --- parameters, read from / written to the compiled model ---
    def get_params(self) -> dict[str, np.ndarray]:
        m = self.model
        return {
            "kp": m.actuator_gainprm[self.actid, 0].copy(),
            "kv": -m.actuator_biasprm[self.actid, 2].copy(),
            "frictionloss": m.dof_frictionloss[self.dofadr].copy(),
            "armature": m.dof_armature[self.dofadr].copy(),
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        m = self.model
        if "kp" in params:
            m.actuator_gainprm[self.actid, 0] = params["kp"]
            m.actuator_biasprm[self.actid, 1] = -np.asarray(params["kp"])
        if "kv" in params:
            m.actuator_biasprm[self.actid, 2] = -np.asarray(params["kv"])
        if "frictionloss" in params:
            m.dof_frictionloss[self.dofadr] = params["frictionloss"]
        if "armature" in params:
            m.dof_armature[self.dofadr] = params["armature"]

    # --- rollout ---
    def rollout(self, q_des: np.ndarray, q0: np.ndarray) -> np.ndarray:
        """sim_q[i] = arm qpos after holding q_des[i] for one control period, starting at rest at q0."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = q0
        d.ctrl[:] = self.gripper_ctrl
        mujoco.mj_forward(m, d)
        out = np.empty((len(q_des), len(self.qadr)))
        for i, target in enumerate(q_des):
            d.ctrl[self.actid] = target
            mujoco.mj_step(m, d, self.n_substeps)
            out[i] = d.qpos[self.qadr]
        return out


def tracking_error(
    replay: ArmReplay, q_des: np.ndarray, q_real: np.ndarray, obs_offset: int
) -> np.ndarray:
    """(T - obs_offset, n_joints) sim - real error [rad].

    `obs_offset`: real q[i + obs_offset] is the measurement that follows command q_des[i].
    In the rcs/LeRobot recordings obs[i] is captured *before* action[i] is sent, hence 1.
    """
    T = len(q_des) - obs_offset
    sim_q = replay.rollout(q_des[:T], q0=q_real[0])
    return sim_q - q_real[obs_offset : obs_offset + T]


def rms_deg(err: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.sqrt((err**2).mean(0)))
