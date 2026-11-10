from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer as mjviewer  # ty: ignore
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR
from molmo_spaces.configs.robot_configs import RBY1Config
from molmo_spaces.robots.robot_views.rby1_view import RBY1HoloBaseGroup, RBY1RobotView

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"

MOCAP_WAYPOINT_NAME = "waypoint"
WAYPOINT_START_XY = (2.0, 0.0)


@dataclass
class Args:
    paused: bool = False


@dataclass
class SimOptions:
    running: bool = True
    reset: bool = False


OPTIONS = SimOptions()


def key_callback(keycode: int) -> None:
    global OPTIONS

    # print(f"{keycode=}")
    match keycode:
        case 32:  # SPACE == PAUSE
            OPTIONS.running = not OPTIONS.running
        case 82:  # R == Reset
            OPTIONS.reset = True


def main() -> int:
    global OPTIONS

    args = tyro.cli(Args)

    robot_cfg = RBY1Config()
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

    # Add a mocap body for the waypoint used for naviation
    body_waypoint = spec.worldbody.add_body(
        name=MOCAP_WAYPOINT_NAME,
        pos=(*WAYPOINT_START_XY, 0.03),
        mocap=True,
    )
    body_waypoint.add_geom(
        type=mj.mjtGeom.mjGEOM_BOX,
        size=(0.06, 0.06, 0.06),
        contype=0,
        conaffinity=0,
        rgba=(0.6, 0.3, 0.3, 0.5),
    )
    body_waypoint.add_site(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=(0.01,),
        rgba=(0.0, 0.0, 1.0, 1.0),
    )

    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    robot = robot_cls(data, robot_cfg)
    robot_view = robot.robot_view

    assert isinstance(robot_view, RBY1RobotView)

    base_group = robot_view.base
    assert isinstance(base_group, RBY1HoloBaseGroup)

    mocap_waypoint_id = model.body(MOCAP_WAYPOINT_NAME).mocapid[0].item()

    viewer_args = dict(
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )

    OPTIONS.running = not args.paused
    with mjviewer.launch_passive(model, data, **viewer_args) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            if OPTIONS.reset:
                robot.reset()
                OPTIONS.reset = False

            if OPTIONS.running:
                waypoint_xy = data.body(MOCAP_WAYPOINT_NAME).xpos[:2].tolist()
                # waypoint_euler = R.from_quat(
                #     data.body(MOCAP_WAYPOINT_NAME).xquat, scalar_first=True
                # ).as_euler("xyz", degrees=False)

                ctrls = {"base": np.array([*waypoint_xy, 0.0])}
                robot.update_control(ctrls)

                t_start = data.time
                while data.time - t_start < 1.0 / 60.0:
                    robot.compute_control()
                    mj.mj_step(model, data)
            else:
                mj.mj_forward(model, data)

            data.mocap_pos[mocap_waypoint_id][2] = 0.03
            waypoint_yaw = (
                R.from_quat(data.mocap_quat[mocap_waypoint_id], scalar_first=True)
                .as_euler("xyz", degrees=False)[2]
                .item()
            )

            data.mocap_quat[mocap_waypoint_id] = R.from_euler(
                "xyz", (0.0, 0.0, waypoint_yaw), degrees=False
            ).as_quat(scalar_first=True)

            viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
