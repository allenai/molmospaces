from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer as mjviewer  # ty: ignore
import tyro

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR

from .common import ROBOTS_INFO, ExampleBaseArgs

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"


@dataclass
class Args(ExampleBaseArgs):
    pass


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
