import numpy as np
import mujoco
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    MocapRobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose, site_pose


class XArm7BaseGroup(MocapRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        body_id: int = mj_data.model.body(f"{namespace}base").id
        super().__init__(mj_data, body_id)


class XArm7ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(
        self,
        mj_data: MjData,
        base_group: XArm7BaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [model.joint(f"{namespace}joint{i + 1}").id for i in range(7)]
        act_ids = [model.actuator(f"{namespace}act{i + 1}").id for i in range(7)]
        self._arm_root_id = model.body(f"{namespace}link_base").id
        self._ee_site_id = model.site(f"{namespace}link_tcp").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base_group)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return site_pose(self.mj_data, self._ee_site_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class XArm7GripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(self, mj_data: MjData, base_group: XArm7BaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._namespace = namespace
        joint_ids = [
            model.joint(f"{namespace}left_driver_joint").id,
            model.joint(f"{namespace}right_driver_joint").id,
        ]
        act_ids = [model.actuator(f"{namespace}gripper").id]
        root_body_id = model.body(f"{namespace}xarm_gripper_base_link").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{namespace}link_tcp").id
        self._finger_1_geom_id = model.geom(f"{namespace}left_finger_pad_2").id
        self._finger_2_geom_id = model.geom(f"{namespace}right_finger_pad_2").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [0 if open else 255]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.004, 0.089

    @property
    def inter_finger_dist(self) -> float:
        dist = mujoco.mj_geomDistance(
            self.mj_model, self.mj_data, self._finger_1_geom_id, self._finger_2_geom_id, 0.1, None
        )
        return max(0.0, dist)

    @property
    def leaf_frame_to_world(self) -> np.ndarray:
        return site_pose(self.mj_data, self._ee_site_id)

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class XArm7RobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = XArm7BaseGroup(mj_data, namespace)
        move_groups = {
            "base": base,
            "arm": XArm7ArmGroup(mj_data, base, namespace),
            "gripper": XArm7GripperGroup(mj_data, base, namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "xarm7"

    @property
    def base(self) -> XArm7BaseGroup:
        return self._move_groups["base"]
