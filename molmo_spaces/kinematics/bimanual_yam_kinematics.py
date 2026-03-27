"""Kinematics solver for the bimanual YAM robot."""

from mujoco import MjData, MjModel

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.bimanual_yam_view import BimanualYamRobotView


class BimanualYamKinematics(MlSpacesKinematics):
    """Kinematics solver for the bimanual YAM robot (two 6-DOF arms)."""

    def __init__(
        self,
        model: MjModel,
        data: MjData | None = None,
        namespace: str = "",
        robot_view_factory: RobotViewFactory = BimanualYamRobotView,
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

        # Load bimanual YAM robot with a mocap base
        model_xml = """
        <mujoco>
            <asset>
                <model name="bimanual_yam" file="assets/robots/i2rt_yam/bimanual_yam.xml"/>
            </asset>
            <worldbody>
                <body name="base" mocap="true">
                    <attach model="bimanual_yam" prefix="" />
                </body>
            </worldbody>
        </mujoco>
        """

        model = MjModel.from_xml_string(model_xml)
        robot_view_factory = BimanualYamRobotView

        data = MjData(model)
        mujoco.mj_forward(model, data)

        ns = ""
        robot_view = robot_view_factory(data, ns)
        left_arm_group = robot_view.get_move_group("left_arm")
        right_arm_group = robot_view.get_move_group("right_arm")
        kinematics = BimanualYamKinematics(
            model, namespace=ns, robot_view_factory=robot_view_factory
        )

        # Set both arms to middle of joint limits
        left_qp = np.mean(left_arm_group.joint_pos_limits, axis=1)
        right_qp = np.mean(right_arm_group.joint_pos_limits, axis=1)
        left_arm_group.joint_pos = left_qp
        right_arm_group.joint_pos = right_qp
        print(f"Initial left arm joint positions: {left_qp}")
        print(f"Initial right arm joint positions: {right_qp}")
        mujoco.mj_forward(model, data)

        # Get current EE poses and create target poses for IK test
        # Left arm: move up/down
        left_pose0 = robot_view.base.pose @ left_arm_group.leaf_frame_to_robot
        left_pose1 = left_pose0.copy()
        left_pose0[2, 3] += 0.05  # Move up 5cm
        left_pose1[2, 3] -= 0.05  # Move down 5cm

        # Right arm: move forward/backward
        right_pose0 = robot_view.base.pose @ right_arm_group.leaf_frame_to_robot
        right_pose1 = right_pose0.copy()
        right_pose0[0, 3] += 0.05  # Move forward 5cm
        right_pose1[0, 3] -= 0.05  # Move backward 5cm

        with launch_passive(model, data) as viewer:
            viewer.sync()
            i = 0
            while viewer.is_running():
                # Alternate between two target poses for each arm
                left_target = left_pose1 if i % 2 == 0 else left_pose0
                right_target = right_pose1 if i % 2 == 0 else right_pose0

                # Run IK for left arm
                qpos_dict = robot_view.get_qpos_dict()
                left_ret = kinematics.ik(
                    "left_arm", left_target, ["left_arm"], qpos_dict, robot_view.base.pose
                )
                if left_ret is not None:
                    robot_view.set_qpos_dict(left_ret)
                    qpos_dict = robot_view.get_qpos_dict()

                # Run IK for right arm
                right_ret = kinematics.ik(
                    "right_arm", right_target, ["right_arm"], qpos_dict, robot_view.base.pose
                )
                if right_ret is not None:
                    robot_view.set_qpos_dict(right_ret)

                left_status = "Success" if left_ret is not None else "Failed"
                right_status = "Success" if right_ret is not None else "Failed"
                print(f"IK iteration {i}: left={left_status}, right={right_status}")

                i += 1
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(2)

    main()
