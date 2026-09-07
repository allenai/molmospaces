from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco as mj
import numpy as np

from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.floating_rum_kinematics import FloatingRUMKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


class FloatingRUMRobot(Robot):
    def __init__(self, mj_data: mj.MjData, config: BaseRobotConfig):
        super().__init__(mj_data, config)

        assert config.robot_view_factory, (
            "Something went wrong, 'robot_view_factory' shouldn't be None"
        )
        self._robot_view = config.robot_view_factory(mj_data, config.robot_namespace)
        self._kinematics = FloatingRUMKinematics(config)
        self._parallel_kinematics = DummyParallelKinematics(config, self._kinematics)
        self._last_cmd_action: dict[str, np.ndarray] | None = None

    @property
    def namespace(self):
        return self.config.robot_namespace

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
        return {}

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [
            TCPPoseSensor(uuid="tcp_pose"),
        ]

    def update_control(self, action_command_dict: dict[str, Any]):
        action_command_dict = self._apply_action_noise_and_save_unnoised_cmd_jp(action_command_dict)
        self._last_cmd_action = action_command_dict

    def compute_control(self) -> None:
        assert self._last_cmd_action is not None
        for mg_id, ctrl in self._last_cmd_action.items():
            if ctrl is not None:
                self._robot_view.get_move_group(mg_id).ctrl = ctrl

    def reset(self):
        self._last_cmd_action = None
        for mg_id, default_pos in self.config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = np.array(default_pos)

    @staticmethod
    def robot_model_root_name() -> str:
        return "base"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: BaseRobotConfig,
        spec: mj.MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        pos = pos + [0.0] if len(pos) == 2 else pos
        super().add_robot_to_scene(
            robot_config=robot_config,
            spec=spec,
            prefix=prefix,
            pos=pos,
            quat=quat,
            randomize_textures=randomize_textures,
            strip_meshes=strip_meshes,
        )

        target_body_name = f"{prefix}target_ee_pose"
        spec.worldbody.add_body(name=target_body_name, pos=pos, quat=quat, mocap=True)
        spec.add_equality(
            type=mj.mjtEq.mjEQ_WELD,
            name1=target_body_name,
            name2=f"{prefix}{cls.robot_model_root_name()}",
            solref=[0.02, 1],
            solimp=[0.9, 0.95, 0.0, 1, 2],
            objtype=mj.mjtObj.mjOBJ_BODY,
        )
