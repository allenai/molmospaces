"""Smoke test for the Rocketbox avatars driven as robots (molmo_spaces/robots/human_rb.py).

Covers what `robotize_human_rb_spec` has to get right for an avatar to be a robot
at all: the ball-joint skeleton comes out as actuated hinges the robot view can
find, the position actuators actually hold a commanded pose against gravity,
`lower_arms` brings the authored A-pose arms down to the sides, and both
skeleton builds work while the static one is rejected for saying so.

Requires one of the humanoid libraries under ROBOTS_DIR (see HumanRBVariant, and
scripts/assets/convert_human_rb.py); skips if absent.
"""

import mujoco
import numpy as np
import pytest

from molmo_spaces.configs.robot_configs import HumanRBRobotConfig
from molmo_spaces.configs.task_sampler_configs import HumanRBVariant
from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR
from molmo_spaces.robots.human_rb import HumanRBRobot, arms_down_shoulder_angles
from molmo_spaces.robots.robot_views.human_rb_view import SIDE_PREFIX

_UID = "Female_Adult_16"


class _FakeExpConfig:
    """Minimal stand-in for MlSpacesExpConfig -- HumanRBRobot only reads .robot_config."""

    def __init__(self, robot_config: HumanRBRobotConfig) -> None:
        self.robot_config = robot_config


def _skip_unless_installed(config) -> None:
    """Skip rather than install.

    config.get_robot_xml_path() resolves through get_robot_path, which would pull
    the library down -- or fail trying, for a build not published yet -- and a
    unit test should not install ~150MB of assets as a side effect of deciding
    whether to run. So look only at what is already on disk.
    """
    library_dir = ROBOTS_DIR / str(config.avatar_variant)
    if not (library_dir / config.uid / f"{config.uid}.xml").exists():
        pytest.skip(f"{config.avatar_variant} not installed at {library_dir}")


def _human_rb_config(**kwargs) -> HumanRBRobotConfig:
    config = HumanRBRobotConfig(uid=_UID, **kwargs)
    _skip_unless_installed(config)
    return config


def _build_standing_scene(config: HumanRBRobotConfig):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    HumanRBRobot.add_robot_to_scene(
        config, spec, prefix=config.robot_namespace, pos=[0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    HumanRBRobot.apply_control_overrides(spec, config)
    model = spec.compile()
    data = mujoco.MjData(model)
    robot = HumanRBRobot(data, _FakeExpConfig(config))
    robot.reset()
    mujoco.mj_forward(model, data)
    return model, data, robot


def _step(model, data, robot, n_steps: int) -> None:
    """Step with every controller holding its reset targets, failing loudly if
    the sim diverges (MuJoCo only warns, then silently resets the state)."""
    robot.update_control({})
    for i in range(n_steps):
        robot.compute_control()
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qacc).all() and np.abs(data.qacc).max() < 1e6, (
            f"simulation diverged at step {i}"
        )


@pytest.fixture
def human_rb_robot():
    return _build_standing_scene(_human_rb_config())


class TestHumanRBRobotization:
    def test_move_groups(self, human_rb_robot):
        _, _, robot = human_rb_robot
        assert set(robot.robot_view.move_group_ids()) == {
            "base",
            "torso",
            "head",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
        }
        assert not robot.robot_view.base.is_mobile, "the avatar's pelvis has no base actuators"

    def test_every_bone_is_actuated(self, human_rb_robot):
        """19 ball joints -> 57 hinges, one position actuator each, all of them
        reachable through the move groups."""
        model, _, robot = human_rb_robot
        assert model.nu == 57
        chains = [
            robot.robot_view.get_move_group(mg_id)
            for mg_id in robot.robot_view.move_group_ids()
            if mg_id != "base"
        ]
        assert sum(mg.n_joints for mg in chains) == 57
        assert sum(mg.n_actuators for mg in chains) == 57

    def test_mass_is_redistributed(self, human_rb_robot):
        """The converter leaves ~240kg in the pelvis capsule and 1e-08 in every
        limb, which no position actuator can drive."""
        model, _, robot = human_rb_robot
        assert model.body_mass.sum() == pytest.approx(70.0)
        hand = model.body(f"{robot.namespace}L_Hand").id
        assert 0.1 < model.body_mass[hand] < 2.0

    def test_skin_survives_robotization(self, human_rb_robot):
        """The renames (uid prefix stripped, then the robot namespace added at
        attach) have to be followed through into the skin's bone references."""
        model, _, robot = human_rb_robot
        assert model.nskin == 1
        bone_bodies = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b))
            for b in model.skin_bonebodyid
        }
        assert f"{robot.namespace}L_UpperArm" in bone_bodies


class TestAvatarPoseHolding:
    def test_holds_reset_pose(self, human_rb_robot):
        """Welded pelvis plus position-held bones: 5s of physics should leave
        the avatar standing where it was put, in the pose it was reset to."""
        model, data, robot = human_rb_robot
        targets = {
            mg_id: qpos.copy()
            for mg_id, qpos in robot.robot_view.get_qpos_dict().items()
            if mg_id != "base"
        }
        pelvis = model.body(f"{robot.namespace}Pelvis").id
        pelvis_start = data.xpos[pelvis].copy()

        _step(model, data, robot, 2500)

        for mg_id, target in targets.items():
            drift = np.abs(robot.robot_view.get_move_group(mg_id).joint_pos - target).max()
            assert drift < 0.05, f"{mg_id} sagged {drift:.3f} rad off its held pose"
        assert np.linalg.norm(data.xpos[pelvis] - pelvis_start) < 0.01

    def test_planar_placement_keeps_it_standing(self, human_rb_robot):
        """`(x, y, yaw)` means "put it down over there" -- but the avatar's root
        body is the pelvis, not a base at floor level, so the generic
        implementation's z=0 would bury it to the waist. The weld target has to
        follow too, or physics drags it straight back."""
        model, data, robot = human_rb_robot
        pelvis = model.body(f"{robot.namespace}Pelvis").id
        standing_height = data.xpos[pelvis][2]

        robot.set_world_pose([3.0, -2.0, 1.2])
        mujoco.mj_forward(model, data)
        assert data.xpos[pelvis] == pytest.approx([3.0, -2.0, standing_height], abs=1e-6)

        _step(model, data, robot, 1500)
        assert data.xpos[pelvis] == pytest.approx([3.0, -2.0, standing_height], abs=0.01)

    def test_unwelded_base_falls_over(self):
        """The weld isn't decoration: the avatar's only collision geom is a
        single body-length capsule, so a free pelvis topples."""
        model, data, robot = _build_standing_scene(_human_rb_config(weld_base=False))
        pelvis = model.body(f"{robot.namespace}Pelvis").id
        start_height = data.xpos[pelvis][2]
        _step(model, data, robot, 2500)
        assert data.xpos[pelvis][2] < start_height - 0.2


class TestHumanRBVariants:
    """Which of the three converter builds can be driven as a robot.

    Robotization only touches ball joints, names and inertials, so the two
    skeleton builds come out identical -- same joint count, same actuators.
    """

    @pytest.mark.parametrize("variant", [HumanRBVariant.SKINNED, HumanRBVariant.ARTICULATED])
    def test_skeleton_variants_stand(self, variant):
        config = _human_rb_config(avatar_variant=variant)
        model, data, robot = _build_standing_scene(config)
        assert model.nu == 57

        pelvis = model.body(f"{robot.namespace}Pelvis").id
        standing_height = data.xpos[pelvis][2]
        _step(model, data, robot, 1500)
        assert data.xpos[pelvis][2] == pytest.approx(standing_height, abs=0.01)

    def test_static_variant_is_rejected(self):
        """The static build is one free-jointed mannequin with no bones. It has
        to fail saying so, not with a bare missing-body error."""
        config = _human_rb_config(avatar_variant=HumanRBVariant.STATIC)
        with pytest.raises(ValueError, match="no skeleton to actuate"):
            _build_standing_scene(config)


class TestAvatarArmsDown:
    def test_lower_arms_hangs_the_arms(self):
        """Both hands should end up tucked in against the body and lower than
        they start in the authored A-pose."""
        _, a_pose_data, a_pose = _build_standing_scene(_human_rb_config(lower_arms=False))
        model, data, robot = _build_standing_scene(_human_rb_config(lower_arms=True))

        for prefix in SIDE_PREFIX.values():
            hand = model.body(f"{robot.namespace}{prefix}Hand").id
            lowered, authored = data.xpos[hand], a_pose_data.xpos[hand]
            assert lowered[2] < authored[2] - 0.1, f"{prefix}Hand is no lower than the A-pose"
            assert abs(lowered[0]) < 0.5 * abs(authored[0]), f"{prefix}Hand is still held out"
        assert a_pose.exp_config.robot_config.lower_arms is False

    def test_shoulder_angles_are_measured_per_avatar(self, human_rb_robot):
        """The angle comes from the avatar's own bone geometry and is mirrored
        between the sides, rather than being one hardcoded number."""
        model, _, robot = human_rb_robot
        angles = arms_down_shoulder_angles(model, robot.namespace)
        assert angles["left"] > 0.3
        assert angles["left"] == pytest.approx(-angles["right"], abs=1e-3)
