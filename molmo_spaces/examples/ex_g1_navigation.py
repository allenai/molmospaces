from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer as mjviewer  # ty: ignore
import tyro

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR
from molmo_spaces.configs.robot_configs import G1Config

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"


@dataclass
class Args:
    pass


@dataclass
class SimOptions:
    running: bool = True


OPTIONS = SimOptions()


def key_callback(keycode: int) -> None:
    global OPTIONS

    # print(f"{keycode=}")
    match keycode:
        case 32:  # PAUSE
            OPTIONS.running = not OPTIONS.running


def main() -> int:
    global OPTIONS

    _ = tyro.cli(Args)

    robot_cfg = G1Config()
    robot_cls = robot_cfg.robot_cls
    assert robot_cls, "Robot class for G1 is not defined for some reason LOL"

    spec = mj.MjSpec.from_file(SCENE_EMPTY_XML.as_posix())
    spec.option.timestep = 0.002

    robot_cls.add_robot_to_scene(
        robot_cfg,
        spec,
        prefix=robot_cfg.robot_namespace,
        pos=[0, 0, 0],
        quat=[1, 0, 0, 0],
    )

    # NOTE(wilbert): the head visual geoms of the g1 are by default on group 5 on the resources LOL
    mjs_torso_body = spec.body(f"{robot_cfg.robot_namespace}torso_link")
    for mjs_geom in mjs_torso_body.geoms:
        assert isinstance(mjs_geom, mj.MjsGeom)
        if mjs_geom.classname.name == f"{robot_cfg.robot_namespace}visual":
            mjs_geom.group = 2

    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    _ = robot_cls(data, robot_cfg)

    viewer_args = dict(
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )
    with mjviewer.launch_passive(model, data, **viewer_args) as viewer:
        while viewer.is_running():
            if OPTIONS.running:
                t_start = data.time
                while data.time - t_start < 1.0 / 60.0:
                    mj.mj_step(model, data)
            else:
                mj.mj_forward(model, data)

            viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
