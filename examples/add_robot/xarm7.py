import logging
from typing import TYPE_CHECKING, Any, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from xarm7_config import XArm7RobotConfig


log = logging.getLogger(__name__)


class XArm7Robot(Robot):
    def __init__(
        self,
        mj_data: MjData,
        config: "MlSpacesExpConfig",
    ) -> None:
        super().__init__(mj_data, config)
        self._robot_view = config.robot_config.robot_view_factory(
            mj_data, config.robot_config.robot_namespace
        )
        self._kinematics = MlSpacesKinematics(config.robot_config)
        self._parallel_kinematics = SimpleWarpKinematics(config.robot_config)

        arm_controller_cls = (
            JointPosController
            if config.robot_config.command_mode == {}
            or config.robot_config.command_mode["arm"] == "joint_position"
            else JointRelPosController
        )
        self._controllers = {
            "arm": arm_controller_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self):
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def state_dim(self) -> int:
        return 7

    def action_dim(self, move_group_ids: list[str]):
        return sum(self._robot_view.get_move_group(mg_id).n_actuators for mg_id in move_group_ids)

    def get_arm_move_group_ids(self) -> list[str]:
        return ["arm"]

    def update_control(self, action_command_dict: dict[str, Any]) -> None:
        action_command_dict = self._apply_action_noise_and_save_unnoised_cmd_jp(action_command_dict)

        for mg_id, controller in self.controllers.items():
            if mg_id in action_command_dict and action_command_dict[mg_id] is not None:
                controller.set_target(action_command_dict[mg_id])
            elif not controller.stationary:
                controller.set_to_stationary()

    def compute_control(self) -> None:
        for controller in self.controllers.values():
            ctrl_inputs = controller.compute_ctrl_inputs()
            controller.robot_move_group.ctrl = ctrl_inputs

    def set_joint_pos(self, robot_joint_pos_dict) -> None:
        for mg_id, joint_pos in robot_joint_pos_dict.items():
            self._robot_view.get_move_group(mg_id).joint_pos = joint_pos

    def set_world_pose(self, robot_world_pose) -> None:
        self._robot_view.base.pose = robot_world_pose

    def reset(self) -> None:
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

    @staticmethod
    def robot_model_root_name() -> str:
        return "link_base"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "XArm7RobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("XArm7RobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = pos + [0.0] if len(pos) == 2 else pos

        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos,
            quat=quat,
            mocap=True,
        )
        if add_base:
            base_height = robot_config.base_size[2]

            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                rgba=[0.4, 0.4, 0.4, 1.0],
                group=0,  # Visual group
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")


if __name__ == "__main__":
    import mujoco
    from mujoco.viewer import launch

    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

    from xarm7_config import XArm7RobotConfig

    robot_config = XArm7RobotConfig()
    spec = MjSpec()

    XArm7Robot.add_robot_to_scene(
        robot_config,
        spec,
        prefix="robot_0/",
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    model = spec.compile()
    data = MjData(model)
    mujoco.mj_forward(model, data)

    launch(model, data)
