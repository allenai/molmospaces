"""Move groups and robot view for the Rocketbox humanoid avatars.

The avatars ship as a ~20-bone skeleton (see `CURATED_BONES` in
scripts/assets/convert_human_rb.py) whose mesh rides on the bones as a
MuJoCo skin. `molmo_spaces.robots.human_rb.robotize_human_rb_spec` turns that
skeleton into an actuated robot: every ball joint becomes three hinges (local
x, y, z) with a position actuator each, named `{bone}_{axis}` /
`{bone}_{axis}_act` with the per-avatar uid prefix stripped, so everything
below is uid-agnostic and only needs the robot namespace.

The bones are grouped the way a person is: a free-floating pelvis base, a
three-link torso, a two-link head, two four-link arms and two three-link legs.
There is no gripper group -- the hands are single rigid bones with no fingers,
so there is nothing to grasp with.
"""

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

# Name of the free joint on the root (pelvis) body, added by the avatar
# converter as `{uid}_Pelvis_jntfree` and renamed by robotize_human_rb_spec.
BASE_JOINT_NAME = "Pelvis_jntfree"

# The three hinge axes each original ball joint is decomposed into, in the
# order robotize_human_rb_spec adds them to the body -- which is also the order
# their qpos entries appear in, so move group joint vectors read
# [bone0_x, bone0_y, bone0_z, bone1_x, ...].
HINGE_AXES = ("x", "y", "z")

# Bones per move group, in kinematic order (root first). The leaf frame of each
# group is the last bone listed.
TORSO_BONES = ("Spine", "Spine1", "Spine2")
HEAD_BONES = ("Neck", "Head")
ARM_BONES = ("Clavicle", "UpperArm", "Forearm", "Hand")
LEG_BONES = ("Thigh", "Calf", "Foot")

# Side prefixes as the Rocketbox rig names them (Bip01 L/R ... -> L_/R_).
SIDE_PREFIX = {"left": "L_", "right": "R_"}


def hinge_joint_names(bones: tuple[str, ...]) -> list[str]:
    """The 3 * len(bones) hinge joint names for a chain of bones, in qpos order."""
    return [f"{bone}_{axis}" for bone in bones for axis in HINGE_AXES]


class HumanRBBaseGroup(FreeJointRobotBaseGroup):
    """The avatar's free-floating pelvis.

    Unactuated: with `HumanRBRobotConfig.weld_base` (the default) the pelvis is
    instead held by a mocap-weld equality constraint added at insertion time,
    exactly like FloatingRUMRobot's base -- the avatar stays standing where it
    was placed rather than toppling over on its single body-length collision
    capsule. `set_world_pose` still writes the free joint directly, so
    teleporting the avatar works the same either way.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        base_joint_id = mj_data.model.joint(f"{namespace}{BASE_JOINT_NAME}").id
        super().__init__(mj_data, base_joint_id, [], [])

    @property
    def noop_ctrl(self) -> np.ndarray:
        return np.array([])


class HumanRBBoneChainGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """A chain of avatar bones, three actuated hinges each.

    Root frame is the first bone's body, leaf frame the last one's -- e.g. for
    an arm, shoulder-side clavicle in and hand out.
    """

    def __init__(
        self,
        mj_data: MjData,
        bones: tuple[str, ...],
        base: RobotBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self.bones = bones
        joint_names = hinge_joint_names(bones)
        joint_ids = [model.joint(f"{namespace}{n}").id for n in joint_names]
        act_ids = [model.actuator(f"{namespace}{n}_act").id for n in joint_names]
        self._root_id = model.body(f"{namespace}{bones[0]}").id
        self._leaf_id = model.body(f"{namespace}{bones[-1]}").id
        super().__init__(mj_data, joint_ids, act_ids, self._root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)


class HumanRBRobotView(RobotView):
    """A Rocketbox avatar as a molmospaces robot: 6 actuated bone chains (57
    hinges) hanging off a free-floating pelvis.

    No gripper move group -- see this module's docstring.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = HumanRBBaseGroup(mj_data, namespace=namespace)

        def chain(bones: tuple[str, ...]) -> HumanRBBoneChainGroup:
            return HumanRBBoneChainGroup(mj_data, bones, base, namespace=namespace)

        move_groups = {
            "base": base,
            "torso": chain(TORSO_BONES),
            "head": chain(HEAD_BONES),
        }
        for side, prefix in SIDE_PREFIX.items():
            move_groups[f"{side}_arm"] = chain(tuple(prefix + b for b in ARM_BONES))
            move_groups[f"{side}_leg"] = chain(tuple(prefix + b for b in LEG_BONES))
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "avatar"

    @property
    def base(self) -> HumanRBBaseGroup:
        return self._move_groups["base"]

    def get_ik_excluded_movegroup_ids(self) -> list[str]:
        """Keep the legs and torso out of generic arm-reach IK solves.

        Same reasoning as G1RobotView's: they sit kinematically upstream of the
        arms, so leaving them unlocked lets a single 6-DOF pose target fold a
        standing human into arbitrary poses to shave millimetres off the
        residual.
        """
        return ["torso", "head", "left_leg", "right_leg"]
