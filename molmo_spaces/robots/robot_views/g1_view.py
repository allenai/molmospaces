"""
Implementation of the Unitree G1 robot model.

The G1 (as configured here) is a bipedal humanoid with:
- A free-floating pelvis (real 6-DOF dynamics, not a kinematic/wheeled base)
- 12 leg joints (not independently controlled by any Controller yet -- Phase 2
  only needs them to hold a static standing pose via JointPosController)
- A 3-DOF waist/torso
- Two 7-DOF arms (left arm has no gripper and is commanded to its natural
  hanging pose; right arm has a dexterous gripper)
- A single right-hand gripper, tendon-actuated, with two mechanically coupled
  finger joints (joint2 = joint1, enforced by an MJCF <equality> constraint)

Each component is implemented as a MoveGroup, with the overall robot structure
managed by the G1RobotView class. There is no independent head MoveGroup
(the head/cameras are rigidly mounted to the torso) and no left-hand gripper
(no hardware exists for it).
"""

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

# Dex gripper: positive qpos closes the fingers, negative opens (matches the
# actuator's ctrlrange and the MJCF <equality joint1="right_Joint1_1"
# joint2="right_Joint2_1" polycoef="0 1 0 0 0"/>, i.e. joint2 = joint1).
GRIPPER_OPEN = -0.0222
GRIPPER_CLOSED = 0.0245

_LEG_JOINT_SUFFIXES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_WAIST_JOINT_SUFFIXES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)


class G1BaseGroup(FreeJointRobotBaseGroup):
    """The G1's free-floating pelvis. No wheels, no holonomic joints -- true
    6-DOF rigid-body dynamics, integrated directly by MuJoCo's physics."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}floating_base_joint").id
        super().__init__(mj_data, base_joint_id, [], [])

    @property
    def noop_ctrl(self) -> np.ndarray:
        return np.array([])


class G1LegsGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """The G1's 12 leg joints. Not driven by any Controller yet in Phase 2
    beyond a constant JointPosController target -- walking (Phase 3) will add
    a dedicated WBC Controller for this move group."""

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [model.joint(f"{namespace}{n}").id for n in _LEG_JOINT_SUFFIXES]
        act_ids = [model.actuator(f"{namespace}walk_{n}").id for n in _LEG_JOINT_SUFFIXES]
        self._pelvis_id = model.body(f"{namespace}pelvis").id
        super().__init__(mj_data, joint_ids, act_ids, self._pelvis_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._pelvis_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._pelvis_id)


class G1WaistGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """The G1's 3-DOF waist (yaw, roll, pitch)."""

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [model.joint(f"{namespace}{n}").id for n in _WAIST_JOINT_SUFFIXES]
        act_ids = [model.actuator(f"{namespace}walk_{n}").id for n in _WAIST_JOINT_SUFFIXES]
        self._waist_root_id = model.body(f"{namespace}waist_yaw_link").id
        self._waist_leaf_id = model.body(f"{namespace}torso_link").id
        super().__init__(mj_data, joint_ids, act_ids, self._waist_root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._waist_leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._waist_root_id)


class G1ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """One of the G1's 7-DOF arms, excluding the gripper.

    The left arm has no gripper hardware and is normally commanded to its
    natural hanging pose; the right arm carries the dexterous gripper.
    """

    def __init__(
        self, mj_data: MjData, side: str, base: RobotBaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self.side = side
        joint_ids = [model.joint(f"{namespace}{side}_{n}").id for n in _ARM_JOINT_SUFFIXES]
        act_ids = [model.actuator(f"{namespace}walk_{side}_{n}").id for n in _ARM_JOINT_SUFFIXES]
        self._ee_site_id = model.site(f"{namespace}{side}_grasp").id
        self._arm_root_id = model.body(f"{namespace}{side}_shoulder_pitch_link").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class G1GripperGroup(MJCFFrameMixin, GripperGroup):
    """The G1's right-hand dexterous gripper.

    Two finger joints (right_Joint1_1, right_Joint2_1) are mechanically
    coupled by an MJCF <equality> constraint enforcing joint2 = joint1 (unlike
    RBY1's gripper, which couples finger2 = -finger1). One tendon-driven
    position actuator (right_grip) drives both.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [
            model.joint(f"{namespace}right_Joint1_1").id,
            model.joint(f"{namespace}right_Joint2_1").id,
        ]
        act_ids = [model.actuator(f"{namespace}right_grip").id]
        self._ee_site_id = model.site(f"{namespace}right_grasp").id
        root_body_id = model.body(f"{namespace}right_dex_base").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = np.array([GRIPPER_OPEN if open else GRIPPER_CLOSED])

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        # Distance is defined as -joint_pos (more negative qpos == more open).
        return -GRIPPER_CLOSED, -GRIPPER_OPEN

    @property
    def inter_finger_dist(self) -> float:
        return float(-self.joint_pos[0])

    @property
    def joint_pos(self) -> np.ndarray:
        return self.mj_data.qpos[self._joint_posadr]

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set joint positions, applying the joint2 = joint1 coupling.

        Args:
            joint_pos: Either a single value (applied to both coupled joints)
                or two values (used directly).
        """
        if len(joint_pos) == 1:
            coupled_pos = np.array([joint_pos[0], joint_pos[0]])
        else:
            coupled_pos = joint_pos
        self.mj_data.qpos[self._joint_posadr] = coupled_pos

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class G1RobotView(RobotView):
    """Implementation of the complete G1 robot (Phase 2 shape: standing only).

    No `head` move group (head/cameras are rigidly mounted to the torso) and
    no `left_gripper` move group (no hardware exists for it).
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = G1BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "legs": G1LegsGroup(mj_data, base, namespace=namespace),
            "waist": G1WaistGroup(mj_data, base, namespace=namespace),
            "left_arm": G1ArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": G1ArmGroup(mj_data, "right", base, namespace=namespace),
            "right_gripper": G1GripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "g1"

    @property
    def base(self):
        return self.get_move_group("base")
