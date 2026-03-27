"""Kinematics solver for the i2rt YAM robot."""

from mujoco import MjData, MjModel

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.i2rt_yam_view import I2rtYamRobotView


class I2rtYamKinematics(MlSpacesKinematics):
    """Kinematics solver for the i2rt YAM 6-DOF arm."""

    def __init__(
        self,
        model: MjModel,
        data: MjData | None = None,
        namespace: str = "",
        robot_view_factory: RobotViewFactory = I2rtYamRobotView,
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

        # Load YAM robot with a mocap base (similar to how add_robot_to_scene works)
        model_xml = """
        <mujoco>
            <asset>
                <model name="yam" file="assets/robots/i2rt_yam/yam.xml"/>
            </asset>
            <worldbody>
                <body name="base" mocap="true">
                    <attach model="yam" body="arm" prefix="" />
                </body>
            </worldbody>
        </mujoco>
        """

        model = MjModel.from_xml_string(model_xml)
        robot_view_factory = I2rtYamRobotView

        data = MjData(model)
        mujoco.mj_forward(model, data)

        ns = ""
        robot_view = robot_view_factory(data, ns)
        arm_group = robot_view.get_move_group("arm")
        kinematics = I2rtYamKinematics(model, namespace=ns, robot_view_factory=robot_view_factory)

        # Set arm to middle of joint limits
        qp = np.mean(arm_group.joint_pos_limits, axis=1)
        arm_group.joint_pos = qp
        print(f"Initial joint positions: {qp}")
        mujoco.mj_forward(model, data)

        # Get current EE pose and create two target poses (up and down)
        pose0 = robot_view.base.pose @ arm_group.leaf_frame_to_robot
        pose1 = pose0.copy()
        pose0[2, 3] += 0.05  # Move up 5cm
        pose1[2, 3] -= 0.05  # Move down 5cm

        groups = ["arm"]

        with launch_passive(model, data) as viewer:
            viewer.sync()
            i = 0
            while viewer.is_running():
                # Alternate between two target poses
                target_pose = pose1 if i % 2 == 0 else pose0
                ret = kinematics.ik(
                    "arm", target_pose, groups, robot_view.get_qpos_dict(), robot_view.base.pose
                )
                print(f"IK iteration {i}: {'Success' if ret is not None else 'Failed'}")
                i += 1
                if ret is not None:
                    robot_view.set_qpos_dict(ret)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(2)

    main()
