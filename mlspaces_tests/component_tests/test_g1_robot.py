"""Smoke test for the G1 humanoid robot (Phase 3: whole-body walking).

Exercises G1WalkController (PD-torque + ONNX stand/walk policy) on the
combined legs_waist move group, plus JointPosController for both arms and
the right gripper.

Requires the G1 MJCF/mesh assets and the groot_balance.onnx/groot_walk.onnx
policy weights to be present locally under `get_robot_path("g1")` (not yet
registered in the shared resource manifest -- see the plan); skips if
unavailable.
"""

import mujoco
import numpy as np
import pytest

from molmo_spaces.configs.robot_configs import ActionNoiseConfig, G1Config
from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR
from molmo_spaces.robots.g1 import G1Robot

_G1_ASSETS_DIR = ROBOTS_DIR / "g1"
_G1_POLICIES_DIR = _G1_ASSETS_DIR / "policies"


class _FakeExpConfig:
    """Minimal stand-in for MlSpacesExpConfig -- G1Robot only reads .robot_config."""

    def __init__(self, robot_config: G1Config) -> None:
        self.robot_config = robot_config


def _g1_config() -> G1Config:
    if not _G1_ASSETS_DIR.exists():
        pytest.skip(f"G1 assets not found at {_G1_ASSETS_DIR} (local-only checkout)")
    if not (_G1_POLICIES_DIR / "groot_balance.onnx").exists():
        pytest.skip(f"G1 walking policy weights not found at {_G1_POLICIES_DIR}")
    return G1Config(
        action_noise_config=ActionNoiseConfig(enabled=False),
    )


def _build_standing_scene(config: G1Config):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    G1Robot.add_robot_to_scene(
        config, spec, prefix="robot_0/", pos=[0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    G1Robot.apply_control_overrides(spec, config)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


@pytest.fixture
def g1_robot():
    config = _g1_config()
    model, data = _build_standing_scene(config)
    robot = G1Robot(data, _FakeExpConfig(config))
    robot.reset()
    mujoco.mj_forward(model, data)
    return model, data, robot


class TestG1Config:
    def test_move_groups(self):
        config = _g1_config()
        assert config.robot_view_factory is not None
        _, data = _build_standing_scene(config)
        view = config.robot_view_factory(data, config.robot_namespace)
        assert set(view.move_group_ids()) == {
            "base",
            "legs_waist",
            "left_arm",
            "right_arm",
            "right_gripper",
        }
        assert not view.base.is_mobile, "G1's pelvis has no base actuators"


class TestG1WholeBodyControl:
    def test_stands_indefinitely(self, g1_robot):
        """With G1WalkController active, an empty action (-> stationary, cmd=0)
        engages the ONNX standing policy every tick rather than holding an
        open-loop pose. Measured empirically: unlike Phase 2's plain PD (which
        tipped over by t=2s), the WBC holds pelvis height ~0.74-0.75m and
        upright ~0.999 indefinitely -- tested here for 3s.
        """
        model, data, robot = g1_robot

        robot.update_control({})  # no commanded action -> stationary (cmd=0, still balancing)
        n_steps = int(3.0 / model.opt.timestep)
        for _ in range(n_steps):
            robot.compute_control()
            mujoco.mj_step(model, data)

        assert np.isfinite(data.qpos).all(), "qpos went non-finite (NaN/Inf)"
        assert np.isfinite(data.qvel).all(), "qvel went non-finite (NaN/Inf)"
        assert np.isfinite(data.ctrl).all(), "ctrl went non-finite (NaN/Inf)"

        pelvis_pose = robot.robot_view.base.pose
        pelvis_height = pelvis_pose[2, 3]
        assert pelvis_height > 0.6, f"pelvis collapsed to height {pelvis_height:.3f}m"

        upright = pelvis_pose[2, 2]  # cosine of tilt between pelvis local-z and world-z
        assert upright > 0.95, f"pelvis tipped over (upright cosine {upright:.3f})"

    def test_walks_forward_on_command(self, g1_robot):
        """Commanding a positive forward velocity on legs_waist should move the
        pelvis forward roughly at the commanded speed while staying upright.
        Measured empirically: commanding vx=0.5 m/s covers ~2.3m in 4.5s
        (~0.5 m/s average) while upright stays > 0.99.
        """
        model, data, robot = g1_robot

        robot.update_control({})
        n_settle = int(1.0 / model.opt.timestep)
        for _ in range(n_settle):
            robot.compute_control()
            mujoco.mj_step(model, data)

        start_x = robot.robot_view.base.pose[0, 3]

        robot.update_control({"legs_waist": np.array([0.5, 0.0, 0.0, 0.74, 0.0, 0.0, 0.0])})
        n_walk = int(4.5 / model.opt.timestep)
        for _ in range(n_walk):
            robot.compute_control()
            mujoco.mj_step(model, data)

        assert np.isfinite(data.qpos).all(), "qpos went non-finite (NaN/Inf)"
        assert np.isfinite(data.qvel).all(), "qvel went non-finite (NaN/Inf)"

        pelvis_pose = robot.robot_view.base.pose
        distance = pelvis_pose[0, 3] - start_x
        assert distance > 1.5, f"expected forward progress > 1.5m, got {distance:.3f}m"

        upright = pelvis_pose[2, 2]
        assert upright > 0.95, f"pelvis tipped over while walking (upright cosine {upright:.3f})"
