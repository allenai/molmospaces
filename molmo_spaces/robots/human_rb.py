"""Rocketbox humanoid avatars as molmospaces robots.

The avatar assets (scripts/assets/convert_human_rb.py, `skin` command)
are ragdolls, not robots: a ~20-bone chain of *ball* joints, no actuators, all
the mass in one 240kg pelvis capsule and every limb massless. That is enough to
scatter them around a scene as soft obstacles (see
molmo_spaces/tasks/pick_with_human_rb_task_sampler.py) but not to pose or
control one.

`robotize_human_rb_spec` closes that gap at load time -- it rewrites the avatar
MjSpec into something the rest of molmospaces already knows how to drive:

  - each ball joint becomes three hinges about the bone's local x/y/z, each
    with a position actuator, so every bone is commandable and *holds* its
    commanded angle (a ball joint has no position actuator in MuJoCo, which is
    why they can't simply be actuated in place);
  - bodies get anthropometric masses summing to `total_mass` instead of the
    converter's placeholder 1e-08 limbs, so the actuators act on something
    physical;
  - body and joint names lose their per-avatar uid prefix, so one uid-agnostic
    robot view (molmo_spaces/robots/robot_views/human_rb_view.py) serves all 116
    avatars.

Nothing else about the asset changes: the free-jointed pelvis, its body-length
collision capsule, the two bbox marker geoms `place_object_near` needs, and the
skin (with its bone references followed through the renames) all survive.

The avatars are authored standing in a wide A-pose with the arms held out and
down at roughly 45 degrees. `HumanRBRobotConfig.lower_arms` (default) brings
them to rest at the sides on reset, computed per avatar from its own bone
geometry rather than hardcoded -- see `arms_down_shoulder_angles`.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtEq, mjtObj
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.human_rb_view import (
    ARM_BONES,
    HEAD_BONES,
    HINGE_AXES,
    LEG_BONES,
    SIDE_PREFIX,
    TORSO_BONES,
    hinge_joint_names,
)
from molmo_spaces.utils.mujoco_scene_utils import namespace_skins

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig, HumanRBRobotConfig
    from molmo_spaces.configs.task_sampler_configs import HumanRBVariant

log = logging.getLogger(__name__)


# The avatar's root body, once robotize_human_rb_spec has stripped the uid prefix
# off `{uid}_Pelvis`.
ROOT_BODY_NAME = "Pelvis"

# Mocap body added next to the pelvis when HumanRBRobotConfig.weld_base is set,
# and weld-constrained to it. Same trick FloatingRUMRobot uses for its base.
WELD_TARGET_BODY_NAME = "human_rb_base_target"

# Suffix the converter gives every ball joint (`{uid}_{bone}_jnt`).
_BALL_JOINT_SUFFIX = "_jnt"

# Fraction of total body mass per bone, from Winter's segment tables, with the
# trunk split evenly over the three spine bones and the clavicles carved out of
# the shoulder mass. Normalized before use, so these only need to be right
# relative to each other.
_MASS_FRACTIONS = {
    "Pelvis": 0.142,
    "Spine": 0.100,
    "Spine1": 0.100,
    "Spine2": 0.100,
    "Neck": 0.020,
    "Head": 0.070,
    "Clavicle": 0.020,
    "UpperArm": 0.028,
    "Forearm": 0.016,
    "Hand": 0.006,
    "Thigh": 0.100,
    "Calf": 0.047,
    "Foot": 0.015,
}

# Per-bone hinge range (+/- radians, same on all three axes). A ball joint has
# no per-axis limits to carry over, so these are deliberately generous
# anatomical bounds rather than a faithful joint model -- they exist to keep an
# IK solve or a stray command from folding a limb through the body, not to
# reproduce human range of motion.
_JOINT_RANGES = {
    "Spine": 0.5,
    "Spine1": 0.5,
    "Spine2": 0.5,
    "Neck": 0.6,
    "Head": 0.6,
    "Clavicle": 0.3,
    "UpperArm": 2.6,
    "Forearm": 2.4,
    "Hand": 1.0,
    "Thigh": 1.5,
    "Calf": 2.2,
    "Foot": 0.8,
}

_DEFAULT_RANGE = 1.0
_DEFAULT_MASS_FRACTION = 0.02

# Bone length used for the inertia estimate of a leaf bone (hands, feet, head),
# which has no child body to measure against.
_LEAF_BONE_LENGTH = 0.08


def human_rb_library_dir(variant: "HumanRBVariant") -> Path:
    """Directory holding one build of the humanoids, installing it if published.

    The published builds (HumanRBVariant.is_downloadable) come down through the
    same resource manifest as every other robot model, so a fresh checkout needs
    no Rocketbox clone and no conversion. The rest have to have been generated
    locally by scripts/assets/convert_human_rb.py, and get a pointer to where
    that would have put them -- callers report the absence, since this is also
    the path used to check whether they are there at all.
    """
    if variant.is_downloadable:
        from molmo_spaces.molmo_spaces_constants import get_robot_path

        return get_robot_path(str(variant))
    return ROBOTS_DIR / str(variant)


def _bone_of(body_name: str) -> str:
    """The lookup key for the per-bone tables above: the bone name with any
    L_/R_ side prefix dropped, so `L_UpperArm` and `R_UpperArm` share an entry.
    """
    for prefix in SIDE_PREFIX.values():
        if body_name.startswith(prefix):
            return body_name[len(prefix) :]
    return body_name


def _strip_uid_prefix(spec: MjSpec, uid: str) -> None:
    """Rename every body and joint from `{uid}_{name}` to `{name}`.

    The avatar converter namespaces everything by uid so several avatars can be
    staged in one scene as plain objects. A robot gets its own `robot_0/`
    namespace at attach time instead, and the uid prefix would otherwise force
    the robot view to know which of the 116 avatars it is looking at.

    Skins reference their bones *by body name*, and MuJoCo does not follow a
    rename, so the bone lists are rewritten here too -- the same gap
    utils.mujoco_scene_utils.namespace_skins works around for attach.
    """
    prefix = f"{uid}_"
    renamed: dict[str, str] = {}
    for body in spec.bodies:
        if body.name.startswith(prefix):
            renamed[body.name] = body.name[len(prefix) :]
            body.name = renamed[body.name]
    for joint in spec.joints:
        if joint.name.startswith(prefix):
            joint.name = joint.name[len(prefix) :]
    for skin in spec.skins:
        skin.bodyname = [renamed.get(n, n) for n in skin.bodyname]


def _bone_length(body: mujoco.MjsBody) -> float:
    """Distance from a bone's origin to its children, averaged; a nominal
    length for leaf bones. Only used to size the inertia estimate.
    """
    child_dists = [float(np.linalg.norm(child.pos)) for child in body.bodies]
    if not child_dists:
        return _LEAF_BONE_LENGTH
    return float(np.mean(child_dists))


def _set_masses(root: mujoco.MjsBody, total_mass: float) -> None:
    """Give every bone an explicit inertial, replacing the converter's 1e-08
    placeholders (and the pelvis capsule's ~240kg of default-density solid).

    Each bone is modelled as a uniform sphere of its own length, which is crude
    but keeps the mass distribution and the inertia scale consistent with each
    other -- good enough for a position-controlled figure that is welded in
    place or standing still.
    """
    bodies = [root] + root.find_all("body")
    weights = np.array(
        [_MASS_FRACTIONS.get(_bone_of(b.name), _DEFAULT_MASS_FRACTION) for b in bodies]
    )
    masses = total_mass * weights / weights.sum()
    for body, mass in zip(bodies, masses, strict=True):
        radius = 0.5 * _bone_length(body)
        inertia = 0.4 * mass * radius**2
        body.mass = float(mass)
        body.inertia = [inertia, inertia, inertia]
        body.ipos = [0.0, 0.0, 0.0]
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        body.explicitinertial = True


def _ball_joints_to_actuated_hinges(
    spec: MjSpec, root: mujoco.MjsBody, kp: float, kd: float, damping: float, armature: float
) -> int:
    """Replace every ball joint under `root` with three actuated hinges.

    Returns the number of hinges (and hence actuators) added. The hinges are
    added in x, y, z order, which is the order their qpos entries end up in and
    what human_rb_view's move groups assume.
    """
    n_hinges = 0
    for body in root.find_all("body"):
        for joint in [j for j in body.joints if j.type == mujoco.mjtJoint.mjJNT_BALL]:
            bone = joint.name.removesuffix(_BALL_JOINT_SUFFIX)
            limit = _JOINT_RANGES.get(_bone_of(bone), _DEFAULT_RANGE)
            spec.delete(joint)
            for i, axis in enumerate(HINGE_AXES):
                name = f"{bone}_{axis}"
                body.add_joint(
                    name=name,
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=np.eye(3)[i],
                    limited=True,
                    range=[-limit, limit],
                    damping=damping,
                    armature=armature,
                )
                # A plain MuJoCo `position` actuator: gain kp, bias (0, -kp, -kd).
                actuator = spec.add_actuator(
                    name=f"{name}_act",
                    target=name,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                    gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                    biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                    ctrllimited=True,
                    ctrlrange=[-limit, limit],
                )
                actuator.gainprm[0] = kp
                actuator.biasprm[1] = -kp
                actuator.biasprm[2] = -kd
                n_hinges += 1
    return n_hinges


def robotize_human_rb_spec(spec: MjSpec, config: "HumanRBRobotConfig") -> MjSpec:
    """Turn a converted Rocketbox avatar MjSpec into an actuated robot in place.

    Works on either skeleton build -- SKINNED and ARTICULATED share the
    converter's body chain, so they robotize identically (the difference is
    only whether the mesh follows the bones as a skin or as rigid per-bone
    chunks that visibly tear at the seams once posed). STATIC has no skeleton
    at all and is rejected below.

    See this module's docstring for what changes and what deliberately doesn't.
    """
    _strip_uid_prefix(spec, config.uid)
    root = spec.body(ROOT_BODY_NAME)
    if root is None:
        raise ValueError(
            f"Avatar {config.uid} ({config.avatar_variant}) has no {ROOT_BODY_NAME!r} body, so "
            "there is no skeleton to actuate. The STATIC build is a single free-jointed mannequin "
            "by design -- use HumanRBVariant.SKINNED (or ARTICULATED) to drive an avatar as a robot."
        )

    n_hinges = _ball_joints_to_actuated_hinges(
        spec,
        root,
        kp=config.joint_kp,
        kd=config.joint_kd,
        damping=config.joint_damping,
        armature=config.joint_armature,
    )
    _set_masses(root, config.total_mass)
    log.debug(f"Robotized avatar {config.uid}: {n_hinges} actuated hinges")
    return spec


def arms_down_shoulder_angles(mj_model: mujoco.MjModel, namespace: str = "") -> dict[str, float]:
    """Per side, the shoulder hinge angle that brings the upper arm to vertical.

    The avatars stand in an A-pose: the upper arm points out and down at
    roughly 45 degrees, and the exact angle differs per avatar (adults,
    children and the costumed professions are not the same build). Every bone
    frame in the converter's output is world-axis-aligned, so the upper arm's
    direction is just its child (forearm) offset, and rotating the shoulder's
    y hinge by `atan2(dx, -dz)` sends that direction to straight down -- for
    both sides at once, since a left arm has dx > 0 and a right arm dx < 0.
    """
    angles = {}
    for side, prefix in SIDE_PREFIX.items():
        forearm_offset = mj_model.body(f"{namespace}{prefix}Forearm").pos
        angles[side] = float(np.arctan2(forearm_offset[0], -forearm_offset[2]))
    return angles


class HumanRBRobot(Robot):
    """A Rocketbox humanoid avatar, driven as a 57-DOF position-controlled robot.

    One JointPosController per bone chain (torso, head, both arms, both legs);
    the pelvis is unactuated and either weld-held or free-falling, see
    HumanRBRobotConfig.weld_base.
    """

    def __init__(self, mj_data: MjData, config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, config)
        robot_config = config.robot_config
        self._robot_view = robot_config.robot_view_factory(mj_data, robot_config.robot_namespace)
        self._kinematics: MlSpacesKinematics | None = None
        self._parallel_kinematics: DummyParallelKinematics | None = None
        self._controllers = {
            mg_id: JointPosController(self._robot_view.get_move_group(mg_id))
            for mg_id in self._robot_view.move_group_ids()
            if mg_id != "base"
        }

    @property
    def namespace(self) -> str:
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        """Built lazily -- it compiles a second copy of the whole avatar (skin
        included), and scene population never asks for IK.
        """
        if self._kinematics is None:
            self._kinematics = MlSpacesKinematics(self.exp_config.robot_config)
        return self._kinematics

    @property
    def parallel_kinematics(self):
        if self._parallel_kinematics is None:
            self._parallel_kinematics = DummyParallelKinematics(
                self.exp_config.robot_config, self.kinematics
            )
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def get_arm_move_group_ids(self) -> list[str]:
        """No TCP-bounded action noise: the avatar's arms end in a single rigid
        hand bone with no gripper or TCP site, so there is no end-effector for
        the Jacobian-based noise model in Robot._apply_tcp_noise_to_move_group
        to be bounded against.
        """
        return []

    def set_world_pose(self, robot_world_pose: np.ndarray | list[float]) -> None:
        """Move the pelvis, keeping the avatar standing and the weld target with it.

        Two things the generic implementation gets wrong for an avatar:

        - A planar `(x, y, yaw)` placement means "put the robot down there",
          and Robot.set_world_pose reads that as z=0 for the robot's root body.
          Every other robot's root sits at its own base, but an avatar's is the
          pelvis, ~0.9m up (and a different ~0.9m per avatar, so this can't be
          a `fixed_base_height` constant either) -- so a planar placement would
          bury it to the waist. The current pelvis height is carried over
          instead, which is the standing height set at insertion.
        - Writing the free joint alone would leave a welded avatar being
          dragged back to wherever it was inserted. The mocap body is placed
          coincident with the pelvis at insertion (see add_robot_to_scene), so
          it just mirrors whatever pose the pelvis ends up at.
        """
        robot_world_pose = np.asarray(robot_world_pose)
        if robot_world_pose.shape == (3,):
            x, y, yaw = robot_world_pose
            standing_pose = np.eye(4)
            standing_pose[:3, 3] = [x, y, self._robot_view.base.pose[2, 3]]
            standing_pose[:3, :3] = R.from_euler("Z", yaw).as_matrix()
            robot_world_pose = standing_pose

        super().set_world_pose(robot_world_pose)
        if self._weld_target is not None:
            pose = self._robot_view.base.pose
            self.mj_data.mocap_pos[self._weld_target] = pose[:3, 3]
            self.mj_data.mocap_quat[self._weld_target] = R.from_matrix(pose[:3, :3]).as_quat(
                scalar_first=True
            )

    @property
    def _weld_target(self) -> int | None:
        """Mocap id of the weld target body, or None if the base isn't welded."""
        if not self.exp_config.robot_config.weld_base:
            return None
        body_id = self.mj_model.body(f"{self.namespace}{WELD_TARGET_BODY_NAME}").id
        return int(self.mj_model.body_mocapid[body_id])

    def reset(self) -> None:
        robot_config = self.exp_config.robot_config
        for mg_id, default_pos in robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

        if robot_config.lower_arms:
            for side, angle in arms_down_shoulder_angles(self.mj_model, self.namespace).items():
                mg = self._robot_view.get_move_group(f"{side}_arm")
                joint_pos = mg.joint_pos.copy()
                joint_pos[hinge_joint_names(mg.bones).index(f"{SIDE_PREFIX[side]}UpperArm_y")] = (
                    angle
                )
                mg.joint_pos = joint_pos

        # Hold whatever pose we just set, rather than the pre-reset one.
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        return ROOT_BODY_NAME

    @classmethod
    def _load_robot_spec(
        cls, robot_config: "BaseRobotConfig", strip_meshes: bool = False
    ) -> MjSpec:
        return robotize_human_rb_spec(
            super()._load_robot_spec(robot_config, strip_meshes), robot_config
        )

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BaseRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        pos = pos + [0.0] if len(pos) == 2 else pos
        # Attached by hand rather than through Robot.add_robot_to_scene: the
        # avatar's skin has to be namespaced in the window between loading the
        # spec and attaching it, which the base implementation does in one step.
        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        namespace_skins(robot_spec, prefix)
        root = robot_spec.body(ROOT_BODY_NAME)
        if root is None:
            raise ValueError(f"Avatar root body {ROOT_BODY_NAME!r} not found in {robot_spec}")
        spec.worldbody.add_frame(pos=pos, quat=quat).attach_body(root, prefix, "")

        if not robot_config.weld_base:
            return

        # Put the mocap body exactly where the pelvis lands (the avatar MJCF
        # holds the pelvis ~0.93m above its own origin) so the weld's captured
        # relative pose is the identity and set_world_pose can mirror one onto
        # the other.
        pelvis_offset = spec.body(f"{prefix}{ROOT_BODY_NAME}").pos
        target_pos = np.asarray(pos) + R.from_quat(quat, scalar_first=True).apply(pelvis_offset)
        target_body_name = f"{prefix}{WELD_TARGET_BODY_NAME}"
        spec.worldbody.add_body(name=target_body_name, pos=target_pos, quat=quat, mocap=True)

        eq = spec.add_equality()
        eq.name1 = target_body_name
        eq.name2 = f"{prefix}{ROOT_BODY_NAME}"
        eq.objtype = mjtObj.mjOBJ_BODY
        eq.type = mjtEq.mjEQ_WELD
        eq.solref = np.array([0.02, 1])
        eq.solimp = np.array([0.9, 0.95, 0.0, 1, 2])


def human_rb_move_group_dofs() -> dict[str, int]:
    """Hinge count per move group -- what a zeroed `init_qpos` has to match."""
    return {
        "torso": 3 * len(TORSO_BONES),
        "head": 3 * len(HEAD_BONES),
        "left_arm": 3 * len(ARM_BONES),
        "right_arm": 3 * len(ARM_BONES),
        "left_leg": 3 * len(LEG_BONES),
        "right_leg": 3 * len(LEG_BONES),
    }
