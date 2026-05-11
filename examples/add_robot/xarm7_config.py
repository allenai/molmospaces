from pathlib import Path
from typing import Callable, Any

from mujoco import MjData

from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory

from xarm7 import XArm7Robot
from xarm7_view import XArm7RobotView


class XArm7RobotConfig(BaseRobotConfig):
    robot_cls: type[XArm7Robot] | None = XArm7Robot
    robot_factory: Callable[[MjData, Any], Robot] | None = XArm7Robot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = XArm7RobotView
    default_world_pose: list[float] = [0, 0, 0, 1, 0, 0, 0]  # (xyz + quat)
    name: str = "xarm7"
    robot_xml_path: Path = Path("xarm7.xml")
    robot_dir: Path = Path("assets/ufactory_xarm7").resolve()
    base_size: list[float] | None = [0.25, 0.25, 0.25]
    init_qpos: dict[str, list[float]] = {
        "base": [],
        "arm": [0.0] * 7,
        "gripper": [0.0, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]
