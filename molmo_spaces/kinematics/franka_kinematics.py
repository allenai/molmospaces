from mujoco import MjData, MjModel

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.franka_fr3_view import FrankaFR3RobotView


class FrankaKinematics(MlSpacesKinematics):
    def __init__(
        self,
        model: MjModel,
        data: MjData | None = None,
        namespace: str = "",
        robot_view_factory: RobotViewFactory = FrankaFR3RobotView,
    ) -> None:
        if data is None:
            data = MjData(model)
        robot_view = robot_view_factory(data, namespace)
        super().__init__(data, robot_view)


if __name__ == "__main__":

    def main() -> None:
        import mujoco
        import numpy as np
        from mujoco.viewer import launch_passive

        np.set_printoptions(linewidth=np.inf)
        import time

        model_xml = """
        <mujoco>
            <asset>
                <model name="franka" file="assets/robots/franka_fr3/fr3.xml"/>
            </asset>
            <worldbody>
                <body name="base" mocap="true">
                    <attach model="franka" body="fr3_link0" prefix="" />
                </body>
            </worldbody>
        </mujoco>
        """

        model = MjModel.from_xml_string(model_xml)
        robot_view_factory = FrankaFR3RobotView

        data = MjData(model)
        mujoco.mj_forward(model, data)

        ns = ""
        robot_view = robot_view_factory(data, ns)
        arm_group = robot_view.get_move_group("arm")
        kinematics = FrankaKinematics(model, namespace=ns, robot_view_factory=robot_view_factory)

        qp = np.mean(arm_group.joint_pos_limits, axis=1)
        arm_group.joint_pos = qp
        print(qp)
        mujoco.mj_forward(model, data)

        pose0 = robot_view.base.pose @ arm_group.leaf_frame_to_robot
        pose1 = pose0.copy()
        pose0[2, 3] += 0.1
        pose1[2, 3] -= 0.1

        groups = [
            "arm",
        ]

        with launch_passive(model, data) as viewer:
            viewer.sync()
            i = 0
            while viewer.is_running():
                if i % 2 == 0:
                    ret = kinematics.ik(
                        "arm", pose1, groups, robot_view.get_qpos_dict(), robot_view.base.pose
                    )
                else:
                    ret = kinematics.ik(
                        "arm", pose0, groups, robot_view.get_qpos_dict(), robot_view.base.pose
                    )
                print(i % 2, ret)
                i += 1
                if ret is not None:
                    robot_view.set_qpos_dict(ret)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(3)

    main()
