"""Robot view for the Unitree G1 Dex1.1 humanoid."""

from typing import Literal

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
from molmo_spaces.utils.mj_model_and_data_utils import body_pose, site_pose

UNITREE_G1_LEG_JOINTS = [
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
]

UNITREE_G1_WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

UNITREE_G1_ARM_JOINTS = [
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
]

UNITREE_G1_HAND_JOINTS = [
    "dex1_finger_joint_1",
    "dex1_finger_joint_2",
]


class UnitreeG1BaseGroup(FreeJointRobotBaseGroup):
    """Floating pelvis base for Unitree G1."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}floating_base_joint").id
        super().__init__(mj_data, base_joint_id, [], [], floating=True)

    @property
    def noop_ctrl(self) -> np.ndarray:
        return np.array([])


class UnitreeG1BodyMoveGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """A simple joint-position-controlled G1 move group."""

    def __init__(
        self,
        mj_data: MjData,
        joint_names: list[str],
        root_body_name: str,
        leaf_frame_name: str,
        base: RobotBaseGroup,
        namespace: str = "",
        leaf_frame_type: Literal["body", "site"] = "body",
    ) -> None:
        model = mj_data.model
        joint_ids = [model.joint(f"{namespace}{joint_name}").id for joint_name in joint_names]
        act_ids = [model.actuator(f"{namespace}{joint_name}").id for joint_name in joint_names]
        self._root_body_id = model.body(f"{namespace}{root_body_name}").id
        self._leaf_frame_type = leaf_frame_type
        if leaf_frame_type == "body":
            self._leaf_frame_id = model.body(f"{namespace}{leaf_frame_name}").id
        elif leaf_frame_type == "site":
            self._leaf_frame_id = model.site(f"{namespace}{leaf_frame_name}").id
        else:
            raise ValueError(f"Unsupported leaf_frame_type: {leaf_frame_type}")
        super().__init__(mj_data, joint_ids, act_ids, self._root_body_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_frame_id

    @property
    def leaf_frame_type(self):
        return self._leaf_frame_type

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        if self.leaf_frame_type == "body":
            return body_pose(self.mj_data, self._leaf_frame_id)
        return site_pose(self.mj_data, self._leaf_frame_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)


class UnitreeG1SideMoveGroup(UnitreeG1BodyMoveGroup):
    """A side-specific G1 move group with conventional joint-name prefixes."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        joint_suffixes: list[str],
        root_body_name: str,
        leaf_frame_name: str,
        base: RobotBaseGroup,
        namespace: str = "",
        leaf_frame_type: Literal["body", "site"] = "body",
    ) -> None:
        joint_names = [f"{side}_{joint_suffix}" for joint_suffix in joint_suffixes]
        super().__init__(
            mj_data,
            joint_names,
            root_body_name,
            leaf_frame_name,
            base,
            namespace,
            leaf_frame_type,
        )


class UnitreeG1DexGripperGroup(MJCFFrameMixin, GripperGroup):
    """Dex1.1 two-finger hand exposed through the MolmoSpaces gripper API."""

    OPEN_CTRL = np.array([0.0245, 0.0245])
    CLOSED_CTRL = np.array([-0.02, -0.02])

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base: RobotBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        joint_names = [f"{side}_{joint_suffix}" for joint_suffix in UNITREE_G1_HAND_JOINTS]
        joint_ids = [model.joint(f"{namespace}{joint_name}").id for joint_name in joint_names]
        act_ids = [model.actuator(f"{namespace}{joint_name}").id for joint_name in joint_names]
        root_body_id = model.body(f"{namespace}{side}_wrist_yaw_link").id
        self._ee_site_id = model.site(f"{namespace}{side}_grasp_site").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = self.OPEN_CTRL if open else self.CLOSED_CTRL

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, float(np.sum(self.OPEN_CTRL - self.CLOSED_CTRL))

    @property
    def inter_finger_dist(self) -> float:
        return float(np.sum(np.clip(self.joint_pos - self.CLOSED_CTRL, 0.0, None)))

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return site_pose(self.mj_data, self._ee_site_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class UnitreeG1RobotView(RobotView):
    """Move-group view for the Unitree G1 Dex1.1 model."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = UnitreeG1BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "left_leg": UnitreeG1SideMoveGroup(
                mj_data,
                "left",
                UNITREE_G1_LEG_JOINTS,
                "pelvis",
                "left_ankle_roll_link",
                base,
                namespace,
            ),
            "right_leg": UnitreeG1SideMoveGroup(
                mj_data,
                "right",
                UNITREE_G1_LEG_JOINTS,
                "pelvis",
                "right_ankle_roll_link",
                base,
                namespace,
            ),
            "waist": UnitreeG1BodyMoveGroup(
                mj_data,
                UNITREE_G1_WAIST_JOINTS,
                "pelvis",
                "torso_link",
                base,
                namespace,
            ),
            "left_arm": UnitreeG1SideMoveGroup(
                mj_data,
                "left",
                UNITREE_G1_ARM_JOINTS,
                "torso_link",
                "left_grasp_site",
                base,
                namespace,
                "site",
            ),
            "right_arm": UnitreeG1SideMoveGroup(
                mj_data,
                "right",
                UNITREE_G1_ARM_JOINTS,
                "torso_link",
                "right_grasp_site",
                base,
                namespace,
                "site",
            ),
            "left_hand": UnitreeG1SideMoveGroup(
                mj_data,
                "left",
                UNITREE_G1_HAND_JOINTS,
                "left_wrist_yaw_link",
                "left_dex1_finger_link_1",
                base,
                namespace,
            ),
            "right_hand": UnitreeG1SideMoveGroup(
                mj_data,
                "right",
                UNITREE_G1_HAND_JOINTS,
                "right_wrist_yaw_link",
                "right_dex1_finger_link_1",
                base,
                namespace,
            ),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}unitree_g1_29dof_dex1_1"

    @property
    def base(self) -> UnitreeG1BaseGroup:
        return self._move_groups["base"]


class UnitreeG1RightArmPickRobotView(RobotView):
    """Policy-facing G1 view for right-arm-only pick attempts."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = UnitreeG1BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "right_arm": UnitreeG1SideMoveGroup(
                mj_data,
                "right",
                UNITREE_G1_ARM_JOINTS,
                "torso_link",
                "right_grasp_site",
                base,
                namespace,
                "site",
            ),
            "gripper": UnitreeG1DexGripperGroup(mj_data, "right", base, namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}unitree_g1_29dof_dex1_1_right_arm_pick"

    @property
    def base(self) -> UnitreeG1BaseGroup:
        return self._move_groups["base"]
