"""The franka_droid sysid JSON is applied over the XML values when the robot is loaded."""

import mujoco
import numpy as np
import pytest

import molmo_spaces
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.robots.sysid import load_sysid_params


def _compile(config: FrankaRobotConfig) -> mujoco.MjModel:
    assert config.robot_cls is not None
    spec = mujoco.MjSpec.from_file(
        str(molmo_spaces.__file__.rsplit("/", 1)[0] + "/resources/base_scene.xml")
    )
    config.robot_cls.add_robot_to_scene(
        config, spec, prefix=config.robot_namespace, pos=[0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    config.robot_cls.apply_control_overrides(spec, config)
    return spec.compile()


def test_franka_droid_sysid_applied():
    config = FrankaRobotConfig()
    assert config.sysid_params_file is not None and config.sysid_params_file.exists()
    data = load_sysid_params(config.sysid_params_file)
    m = _compile(config)
    ns = config.robot_namespace
    for i, (jn, an) in enumerate(zip(data["joints"], data["actuators"], strict=True)):
        dof = m.jnt_dofadr[m.joint(ns + jn).id]
        act = m.actuator(ns + an).id
        p = data["params"]
        assert m.dof_frictionloss[dof] == pytest.approx(p["frictionloss"][i])
        assert m.dof_armature[dof] == pytest.approx(p["armature"][i])
        assert m.actuator_gainprm[act, 0] == pytest.approx(p["kp"][i])
        assert m.actuator_biasprm[act, 1] == pytest.approx(-p["kp"][i])
        assert m.actuator_biasprm[act, 2] == pytest.approx(-p["kv"][i])


def test_sysid_disabled_keeps_xml_values():
    m = _compile(FrankaRobotConfig(sysid_params_file=None))
    act = m.actuator("robot_0/fr3_joint1").id
    assert m.actuator_gainprm[act, 0] == 4500.0  # franka_droid/model.xml nominal
    assert np.all(m.dof_frictionloss[:7] == 0.0)


def test_explicit_gains_override_sysid():
    m = _compile(FrankaRobotConfig(K_stiffness=[1234.0] * 7))
    assert m.actuator_gainprm[m.actuator("robot_0/fr3_joint1").id, 0] == 1234.0
