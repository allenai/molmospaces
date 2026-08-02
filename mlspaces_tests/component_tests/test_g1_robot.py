"""Smoke test for the G1 humanoid robot (Phase 2 of the G1 upstreaming plan):
loads into a bare scene with a floor and holds a static standing pose under
gravity. No walking controller exists yet -- this only exercises
JointPosController plus the MJCF's own tuned PD actuator gains.

Requires the G1 MJCF/mesh assets to be present locally (not yet registered in
the shared resource manifest -- see the plan); skips if unavailable.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from molmo_spaces.configs.robot_configs import ActionNoiseConfig, G1Config
from molmo_spaces.robots.g1 import G1Robot

_G1_ASSETS_DIR = Path("~/code/g1_molmo/molmospaces/assets/robots/g1").expanduser()


class _FakeExpConfig:
    """Minimal stand-in for MlSpacesExpConfig -- G1Robot only reads .robot_config."""

    def __init__(self, robot_config: G1Config) -> None:
        self.robot_config = robot_config


def _g1_config() -> G1Config:
    if not _G1_ASSETS_DIR.exists():
        pytest.skip(f"G1 assets not found at {_G1_ASSETS_DIR} (local-only checkout)")
    return G1Config(
        robot_dir=_G1_ASSETS_DIR,
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
            "legs",
            "waist",
            "left_arm",
            "right_arm",
            "right_gripper",
        }
        assert not view.base.is_mobile, "G1's pelvis has no base actuators"


class TestG1StaticStand:
    def test_loads_and_holds_stand(self, g1_robot):
        """A plain JointPosController holding a constant commanded pose has no
        active balance correction (no CoM/ZMP feedback) -- that's the walking
        controller's job (Phase 3, not yet built). Measured empirically: G1
        holds its standing pose (pelvis ~0.69-0.73m, upright) through about
        t=1s, then genuinely tips over by t=2s under pure open-loop PD. This
        test's job is to validate Phase 2's wiring (joints/actuators/scene
        assembly are all correct) over a short window, not indefinite
        unaided balance -- that's an explicit Phase 3 dependency, not a bug.
        """
        model, data, robot = g1_robot

        robot.update_control({})  # no commanded action -> hold reset pose (stationary)
        n_steps = int(1.0 / model.opt.timestep)  # ~1 simulated second
        for _ in range(n_steps):
            robot.compute_control()
            mujoco.mj_step(model, data)

        assert np.isfinite(data.qpos).all(), "qpos went non-finite (NaN/Inf)"
        assert np.isfinite(data.qvel).all(), "qvel went non-finite (NaN/Inf)"
        assert np.isfinite(data.ctrl).all(), "ctrl went non-finite (NaN/Inf)"

        pelvis_pose = robot.robot_view.base.pose
        pelvis_height = pelvis_pose[2, 3]
        assert pelvis_height > 0.55, f"pelvis collapsed to height {pelvis_height:.3f}m"

        upright = pelvis_pose[2, 2]  # cosine of tilt between pelvis local-z and world-z
        assert upright > 0.7, f"pelvis tipped over (upright cosine {upright:.3f})"
