"""Bimanual YAM robot implementation for the framework."""

from typing import TYPE_CHECKING, cast

from mujoco import MjData, MjSpec, mjtGeom

from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.kinematics.bimanual_yam_kinematics import BimanualYamKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import (
    DummyParallelKinematics,
)
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BimanualYamRobotConfig


class BimanualYamRobot(Robot):
    """Bimanual YAM robot implementation (two 6-DOF arms with parallel grippers)."""

    def __init__(
        self,
        mj_data: MjData,
        config: "MlSpacesExpConfig",
    ) -> None:
        super().__init__(mj_data, config)
        self._robot_view = config.robot_config.robot_view_factory(
            mj_data, config.robot_config.robot_namespace
        )
        self._kinematics = BimanualYamKinematics(
            self.mj_model,
            namespace=config.robot_config.robot_namespace,
            robot_view_factory=config.robot_config.robot_view_factory,
        )

        # Use DummyParallelKinematics for batch IK (wraps the MlSpacesKinematics)
        # Default to left arm for parallel kinematics
        self._parallel_kinematics = DummyParallelKinematics(
            config.robot_config,
            self._kinematics,
            mg_id="left_arm",
            unlocked_mg_ids=["left_arm", "right_arm"],
        )

        # Determine controller classes based on command mode
        arm_command_mode = config.robot_config.command_mode.get("arm", "joint_position")
        if arm_command_mode == "joint_rel_position":
            arm_controller_cls = JointRelPosController
        else:
            arm_controller_cls = JointPosController

        gripper_command_mode = config.robot_config.command_mode.get("gripper", "joint_position")
        if gripper_command_mode == "joint_rel_position":
            gripper_controller_cls = JointRelPosController
        else:
            gripper_controller_cls = JointPosController

        self._controllers = {
            "left_arm": arm_controller_cls(self._robot_view.get_move_group("left_arm")),
            "right_arm": arm_controller_cls(self._robot_view.get_move_group("right_arm")),
            "left_gripper": gripper_controller_cls(self._robot_view.get_move_group("left_gripper")),
            "right_gripper": gripper_controller_cls(
                self._robot_view.get_move_group("right_gripper")
            ),
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
    def controllers(self):
        return self._controllers

    @property
    def state_dim(self) -> int:
        return 12  # Two 6-DOF arms

    def action_dim(self, move_group_ids: list[str]):
        return sum(self._robot_view.get_move_group(mg_id).n_actuators for mg_id in move_group_ids)

    def get_arm_move_group_ids(self) -> list[str]:
        """Bimanual YAM has two independent arms."""
        return ["left_arm", "right_arm"]

    def update_control(self, action_command_dict) -> None:
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
        """The root body name in the bimanual_yam.xml."""
        return "bimanual_base"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BimanualYamRobotConfig",
        spec: MjSpec,
        robot_spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
    ) -> None:
        robot_config = cast("BimanualYamRobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = pos + [0.0] if len(pos) == 2 else pos

        # Create a mocap body to control the robot base pose
        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos,
            quat=quat,
            mocap=True,
        )

        if add_base:
            base_height = robot_config.base_size[2]

            # Create a base material (plain dark wood color)
            material_name = f"{prefix}robot_base_material"
            spec.add_material(name=material_name, rgba=[0.3, 0.2, 0.1, 1.0])

            # Add base geometry (platform)
            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                material=material_name,
                group=0,  # Visual group
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        # Attach the bimanual robot model at the bimanual_base body
        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")
