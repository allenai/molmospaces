from dataclasses import dataclass, field

import mujoco as mj
import numpy as np
import tyro
from PIL import Image

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR
from molmo_spaces.examples.common import ALL_ROBOTS_INFO, ExampleBaseArgs
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"


DEFAULT_CAMERA_SETUP = {
    "franka-droid": dict(
        pos=[1.472, -1.552, 1.948],
        xyaxes=[0.732, 0.681, -0.000, -0.408, 0.438, 0.801],
    ),
    "i2rt-yam": dict(
        pos=[1.009, -1.355, 1.726],
        xyaxes=[0.814, 0.581, -0.000, -0.369, 0.518, 0.772],
    ),
    "bimanual-yam": dict(
        pos=[0.999, -1.083, 1.659],
        xyaxes=[0.739, 0.673, 0.000, -0.455, 0.500, 0.737],
    ),
    "floating-rum": dict(
        pos=[1.151, -1.397, 0.965],
        xyaxes=[0.764, 0.646, 0.000, -0.339, 0.401, 0.851],
    ),
    "rby1": dict(
        pos=[2.181, -2.161, 2.746],
        xyaxes=[0.727, 0.686, 0.000, -0.431, 0.457, 0.778],
    ),
    "g1": dict(
        pos=[1.285, -1.996, 1.638],
        xyaxes=[0.849, 0.528, -0.000, -0.223, 0.358, 0.907],
    ),
}


@dataclass
class Args(ExampleBaseArgs):
    resolution: tuple[int, int] = field(default_factory=lambda: (800, 600))
    fov: float = 45.0
    pos: list[float] | None = None
    xyaxes: list[float] | None = None


def main() -> int:
    args = tyro.cli(Args)

    robot_info = ALL_ROBOTS_INFO[args.robot_id]
    robot_config = robot_info.config
    robot_cls = robot_config.robot_cls
    robot_view_factory = robot_config.robot_view_factory
    assert robot_cls, f"Robot class from config of robot '{args.robot_id}' not defined"
    assert robot_view_factory, (
        f"Robot view factory from config of robot '{args.robot_id}' not defined"
    )

    spec = mj.MjSpec.from_file(SCENE_EMPTY_XML.as_posix())
    spec.worldbody.add_camera(
        name="main",
        resolution=args.resolution,
        fovy=args.fov,
        pos=args.pos or DEFAULT_CAMERA_SETUP[args.robot_id]["pos"],
        xyaxes=args.xyaxes or DEFAULT_CAMERA_SETUP[args.robot_id]["xyaxes"],
    )

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

    renderer = MjOpenGLRenderer(model, width=args.resolution[0], height=args.resolution[1])

    _ = robot_cls(data, robot_config)
    robot_view = robot_view_factory(data, robot_config.robot_namespace)

    robot_init_qpos = {key: np.array(val) for key, val in robot_config.init_qpos.items()}
    robot_view.set_qpos_dict(robot_init_qpos)

    mj.mj_forward(model, data)

    renderer.update(data, "main")
    pixels = renderer.render()

    pil_img = Image.fromarray(pixels)
    pil_img.save(f"img_{args.robot_id.replace('-', '_')}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
