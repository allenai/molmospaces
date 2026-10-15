from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer as mjviewer  # ty: ignore
import numpy as np
import tyro

from molmo_spaces import MOLMO_SPACES_PACKAGED_ASSETS_DIR

from .common import ROBOTS_INFO, ExampleBaseArgs

SCENE_EMPTY_XML = MOLMO_SPACES_PACKAGED_ASSETS_DIR / "scene_empty.xml"

MOCAP_TARGET_NAME = "target"


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
    robot_view_factory = robot_config.robot_view_factory
    assert robot_cls, f"Robot class from config of robot '{args.robot_id}' not defined"
    assert robot_view_factory, (
        f"Robot view factory from config of robot '{args.robot_id}' not defined"
    )

    spec = mj.MjSpec.from_file(SCENE_EMPTY_XML.as_posix())

    body_target = spec.worldbody.add_body(
        name=MOCAP_TARGET_NAME,
        pos=(0.5, 0.0, 0.6),
        quat=(0.0, 1.0, 0.0, 0.0),
        mocap=True,
    )
    body_target.add_geom(
        type=mj.mjtGeom.mjGEOM_BOX,
        size=(0.05, 0.05, 0.05),
        contype=0,
        conaffinity=0,
        rgba=(0.6, 0.3, 0.3, 0.5),
    )
    body_target.add_site(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=(0.01,),
        rgba=(0.0, 0.0, 1.0, 1.0),
        group=1,
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

    robot = robot_cls(data, robot_config)
    robot_view = robot_view_factory(data, robot_config.robot_namespace)

    robot_init_qpos = {key: np.array(val) for key, val in robot_config.init_qpos.items()}
    robot_view.set_qpos_dict(robot_init_qpos)

    with mjviewer.launch_passive(
        model, data, key_callback=None, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            t_start = data.time

            if movegroup_ids := robot_view.get_gripper_movegroup_ids():
                # TODO(wilbert): handle multiple arms|grippers (bimanual setups)

                gripper_movegroup_id = movegroup_ids[0]

                target_pose = np.eye(4)
                target_pose[:3, 3] = data.body(MOCAP_TARGET_NAME).xpos
                target_pose[:3, :3] = data.body(MOCAP_TARGET_NAME).xmat.reshape(3, 3).T

                ik_res = robot.kinematics.ik(
                    gripper_movegroup_id,
                    target_pose,
                    unlocked_move_group_ids=None,
                    q0=robot_view.get_qpos_dict(),
                    base_pose=robot_view.base.pose,
                    rel_to_base=False,
                )

                if ik_res:
                    qpos_dict = {key: val.tolist() for key, val in ik_res.items()}
                    robot_view.set_qpos_dict(qpos_dict)

            while data.time - t_start < 1.0 / 60.0:
                mj.mj_step(model, data)

            viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
