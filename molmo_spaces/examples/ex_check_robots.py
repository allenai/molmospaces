from dataclasses import dataclass
from typing import Literal

import mujoco as mj
import mujoco.viewer as mjviewer  # ty: ignore
import tyro

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR
from molmo_spaces.configs.robot_configs import (
    BaseRobotConfig,
    BimanualYamRobotConfig,
    FloatingRobotiq2f85RobotConfig,
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    I2rtYamRobotConfig,
    MobileFrankaRobotConfig,
    RBY1Config,
)

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"


@dataclass
class RobotInfo:
    config: BaseRobotConfig
    init_pos: tuple[float, float, float]
    init_quat: tuple[float, float, float, float]


ROBOTS_INFO: dict[str, RobotInfo] = {
    "franka-droid": RobotInfo(
        config=FrankaRobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "i2rt-yam": RobotInfo(
        config=I2rtYamRobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "bimanual-yam": RobotInfo(
        config=BimanualYamRobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "floating-rum": RobotInfo(
        config=FloatingRUMRobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "floating-robotiq": RobotInfo(
        config=FloatingRobotiq2f85RobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "rby1": RobotInfo(
        config=RBY1Config(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "mobile-franka": RobotInfo(
        config=MobileFrankaRobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
}


@dataclass
class Args:
    robot_id: Literal[
        "franka-droid",
        "i2rt-yam",
        "bimanual-yam",
        "floating-rum",
        "floating-robotiq",
        "rby1",
        "mobile-franka",
    ]


def main() -> int:
    args = tyro.cli(Args)

    if args.robot_id not in ROBOTS_INFO:
        return 1

    robot_info = ROBOTS_INFO[args.robot_id]
    robot_config = robot_info.config
    robot_cls = robot_config.robot_cls
    assert robot_cls, f"Robot class from config of robot '{args.robot_id}' not defined"

    spec = mj.MjSpec.from_file(SCENE_EMPTY_XML.as_posix())

    robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=list(robot_info.init_pos),
        quat=list(robot_info.init_quat),
    )

    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    _ = robot_cls(data, robot_config)

    with mjviewer.launch_passive(
        model, data, key_callback=None, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            t_start = data.time
            while data.time - t_start < 1.0 / 60.0:
                mj.mj_step(model, data)

            viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
