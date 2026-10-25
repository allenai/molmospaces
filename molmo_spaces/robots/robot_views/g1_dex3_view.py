"""RobotView for the dual-Dex3 G1 (`g1_dex3`): the G1's base / legs_waist / arm groups, with a
7-joint `left_hand` and `right_hand` move group in place of the Dex1 `right_gripper`.
"""

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    MJCFFrameMixin,
    MoveGroup,
    RobotBaseGroup,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.robots.robot_views.g1_view import (
    G1ArmGroup,
    G1BaseGroup,
    G1LegsWaistGroup,
    G1RobotView,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

HAND_JOINT_SUFFIXES = (
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
    "index_0_joint",
    "index_1_joint",
    "middle_0_joint",
    "middle_1_joint",
)


class G1Dex3HandGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """One Unitree Dex3-1 hand: thumb (3 joints), index (2), middle (2), each position-actuated.
    Joint values are flexion angles (0 = fingers extended). The leaf frame is the
    `{side}_grasp` site at the extended index fingertip, rooted at the wrist-yaw link."""

    def __init__(
        self, mj_data: MjData, side: str, base: RobotBaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self.side = side
        names = [f"{namespace}{side}_hand_{n}" for n in HAND_JOINT_SUFFIXES]
        joint_ids = [model.joint(n).id for n in names]
        act_ids = [model.actuator(n).id for n in names]
        self._ee_site_id = model.site(f"{namespace}{side}_grasp").id
        self._wrist_id = model.body(f"{namespace}{side}_wrist_yaw_link").id
        super().__init__(mj_data, joint_ids, act_ids, self._wrist_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._wrist_id)


class G1Dex3RobotView(G1RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = G1BaseGroup(mj_data, namespace=namespace)
        move_groups: dict[str, MoveGroup] = {
            "base": base,
            "legs_waist": G1LegsWaistGroup(mj_data, base, namespace=namespace),
            "left_arm": G1ArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": G1ArmGroup(mj_data, "right", base, namespace=namespace),
            "left_hand": G1Dex3HandGroup(mj_data, "left", base, namespace=namespace),
            "right_hand": G1Dex3HandGroup(mj_data, "right", base, namespace=namespace),
        }
        # RobotView.__init__, skipping G1RobotView's (which would build the Dex1 groups).
        super(G1RobotView, self).__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "g1_dex3"
