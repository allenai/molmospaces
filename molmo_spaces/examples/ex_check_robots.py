from dataclasses import dataclass
from typing import Literal

import mujoco as mj
import mujoco.viewer as mjviewer
import tyro

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR
from molmo_spaces.molmo_spaces_constants import get_robot_path

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"


@dataclass
class RobotInfo:
    xml_file: str
    init_pos: tuple[float, float, float]
    init_quat: tuple[float, float, float, float]


ROBOTS_INFO: dict[str, RobotInfo] = {
    "franka_droid": RobotInfo(
        xml_file="model.xml",
        init_pos=(0, 0, 0),
        init_quat=(1, 0, 0, 0),
    ),
    "g1": RobotInfo(
        xml_file="g1_dex.xml",
        init_pos=(0, 0, 2),
        init_quat=(1, 0, 0, 0),
    ),
}


@dataclass
class Args:
    robot_id: Literal[
        "franka_droid",
        "g1",
    ]


def main() -> int:
    args = tyro.cli(Args)

    if args.robot_id not in ROBOTS_INFO:
        return 1

    robot_info = ROBOTS_INFO[args.robot_id]

    spec = mj.MjSpec.from_file(SCENE_EMPTY_XML.as_posix())

    robot_path = get_robot_path(args.robot_id) / robot_info.xml_file
    robot_spec = mj.MjSpec.from_file(robot_path.as_posix())

    robot_frame = spec.worldbody.add_frame(pos=robot_info.init_pos, quat=robot_info.init_quat)
    robot_frame.attach_body(robot_spec.worldbody.first_body(), prefix="robot/")

    model = spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)

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
