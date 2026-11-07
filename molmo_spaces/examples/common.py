from dataclasses import dataclass
from typing import Literal

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR
from molmo_spaces.configs.robot_configs import (
    BaseRobotConfig,
    BimanualYamRobotConfig,
    FloatingRobotiq2f85RobotConfig,
    FloatingRUMRobotConfig,
    FrankaRobotConfig,
    G1Config,
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


ROBOTS_SINGLE_ARMS_INFO: dict[str, RobotInfo] = {
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
}

ROBOTS_BIMANUAL_ARMS_INFO: dict[str, RobotInfo] = {
    "bimanual-yam": RobotInfo(
        config=BimanualYamRobotConfig(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
}

ROBOTS_FLOATING_GRIPPERS_INFO: dict[str, RobotInfo] = {
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
}

ROBOTS_MOBILE_MANIPULATORS_INFO: dict[str, RobotInfo] = {
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

ROBOTS_HUMANOIDS_INFO: dict[str, RobotInfo] = {
    "g1": RobotInfo(
        config=G1Config(),
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
}

ALL_ROBOTS_INFO = (
    ROBOTS_SINGLE_ARMS_INFO
    | ROBOTS_BIMANUAL_ARMS_INFO
    | ROBOTS_FLOATING_GRIPPERS_INFO
    | ROBOTS_MOBILE_MANIPULATORS_INFO
    | ROBOTS_HUMANOIDS_INFO
)


@dataclass
class ExampleBaseArgs:
    robot_id: Literal[
        "franka-droid",
        "i2rt-yam",
        "bimanual-yam",
        "floating-rum",
        "floating-robotiq",
        "rby1",
        "mobile-franka",
    ]
