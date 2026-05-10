"""Unitree G1 Dex1.1 robot implementation."""

from typing import TYPE_CHECKING, Any

import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig


class UnitreeG1Robot(Robot):
    """Unitree G1 Dex1.1 with joint-position control for all non-base move groups."""

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
        self._parallel_kinematics = DummyParallelKinematics(
            config.robot_config,
            self._kinematics,
        )
        self._pinned_base_pose: np.ndarray | None = None
        self._controllers = {
            move_group_id: JointPosController(self._robot_view.get_move_group(move_group_id))
            for move_group_id in self._robot_view.move_group_ids()
            if move_group_id != "base"
            and config.robot_config.command_mode.get(move_group_id) == "joint_position"
        }
        self.reset()

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

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config,
        spec: MjSpec,
        robot_spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
    ) -> None:
        super().add_robot_to_scene(
            robot_config,
            spec,
            robot_spec,
            prefix,
            pos,
            quat,
            randomize_textures,
        )
        root_body = spec.body(prefix + cls.robot_model_root_name())
        if root_body is None:
            raise ValueError(f"Robot root body {prefix}{cls.robot_model_root_name()} not found")
        root_body.pos = robot_config.default_world_pose[:3]
        root_body.quat = robot_config.default_world_pose[3:7]

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config) -> None:
        super().apply_control_overrides(spec, robot_config)
        if robot_config.gravcomp:
            root_body = spec.body(robot_config.robot_namespace + cls.robot_model_root_name())
            if root_body is not None:
                root_body.gravcomp = 1.0

    @property
    def state_dim(self) -> int:
        return sum(
            self._robot_view.get_move_group(move_group_id).pos_dim
            for move_group_id in self._robot_view.move_group_ids()
        )

    def action_dim(self, move_group_ids: list[str]) -> int:
        return sum(self._robot_view.get_move_group(mg_id).n_actuators for mg_id in move_group_ids)

    def get_arm_move_group_ids(self) -> list[str]:
        return ["left_arm", "right_arm"]

    def update_control(self, action_command_dict: dict[str, Any]) -> None:
        action_command_dict = self._apply_action_noise_and_save_unnoised_cmd_jp(action_command_dict)

        for mg_id, controller in self.controllers.items():
            if mg_id in action_command_dict and action_command_dict[mg_id] is not None:
                controller.set_target(action_command_dict[mg_id])
            elif not controller.stationary:
                controller.set_to_stationary()

    def compute_control(self) -> None:
        self._pin_base_if_configured()
        for controller in self.controllers.values():
            ctrl_inputs = controller.compute_ctrl_inputs()
            controller.robot_move_group.ctrl = ctrl_inputs
        self._pin_base_if_configured()

    def set_joint_pos(self, robot_joint_pos_dict) -> None:
        for mg_id, joint_pos in robot_joint_pos_dict.items():
            self._robot_view.get_move_group(mg_id).joint_pos = joint_pos
            if mg_id == "base":
                self._set_pinned_base_pose_if_configured()

    def set_world_pose(self, robot_world_pose) -> None:
        self._robot_view.base.pose = robot_world_pose
        self.sync_pinned_base_pose()

    def sync_pinned_base_pose(self) -> None:
        """Latch the current base pose as the fixed pose for pinned-base workflows."""
        self._set_pinned_base_pose_if_configured()
        self._zero_base_velocity_if_configured()

    def reset(self) -> None:
        self.mj_data.qpos[:] = self.mj_model.qpos0
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                move_group = self._robot_view.get_move_group(mg_id)
                move_group.joint_pos = np.asarray(default_pos)
                move_group.joint_vel = np.zeros(move_group.vel_dim)
        if self._should_pin_base():
            self._pinned_base_pose = None
            self._zero_base_velocity_if_configured()

    def _should_pin_base(self) -> bool:
        return bool(getattr(self.exp_config.robot_config, "pin_base_in_place", False))

    def _set_pinned_base_pose_if_configured(self) -> None:
        if self._should_pin_base():
            self._pinned_base_pose = self._robot_view.base.pose.copy()

    def _zero_base_velocity_if_configured(self) -> None:
        if self._should_pin_base():
            self._robot_view.base.joint_vel = np.zeros(self._robot_view.base.vel_dim)

    def _pin_base_if_configured(self) -> None:
        if not self._should_pin_base():
            return
        if self._pinned_base_pose is None:
            self._pinned_base_pose = self._robot_view.base.pose.copy()
        self._robot_view.base.pose = self._pinned_base_pose
        self._robot_view.base.joint_vel = np.zeros(self._robot_view.base.vel_dim)

    @staticmethod
    def robot_model_root_name() -> str:
        return "pelvis"
