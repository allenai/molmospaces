import os
import time

import mujoco
import numpy as np
from mujoco import MjData
from scipy.spatial.transform import Rotation as R

from molmo_spaces.data_generation.recorder import RGBRecorder
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.utils.devices.keyboard import Keyboard
from molmo_spaces.utils.devices.spacemouse import SpaceMouse

GRIPPER_OPEN_POS = -0.3  # #0.04 # -0.3  # 0.04 #-0.3
GRIPPER_CLOSE_POS = -2.0  # 255 #-0.5 #-2.0  # 0.0 #-1.2
N_GRIPPER_JOINTS = 1  # 2 # 1


def measure_contact_force(model, data, robot_name="robot_0/"):
    # get the contact force between the robot finger and the object
    robot_id = model.body(robot_name).id

    total_force = np.zeros(3)
    total_torque = np.zeros(3)

    contacts = data.contact  # mjContact
    for i in range(data.ncon):
        c = contacts[i]
        geom1 = c.geom1
        geom2 = c.geom2

        root1 = model.body(model.geom(geom1).bodyid).rootid
        root2 = model.body(model.geom(geom2).bodyid).rootid

        if root1 == robot_id or root2 == robot_id:
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, force)

            R = np.reshape(c.frame, (3, 3))
            f_world = R.T @ force[:3]
            t_world = R.T @ force[3:]

            total_force += f_world[:3]
            total_torque += t_world[:3]

    # return abs max of force and torque but keep the sign
    # max_force = total_force[np.argmax(np.abs(total_force))]
    # max_torque = total_torque[np.argmax(np.abs(total_torque))]

    mag_force = np.linalg.norm(total_force)
    mag_torque = np.linalg.norm(total_torque)

    return mag_force, mag_torque


class GripperTeleopController:
    def __init__(
        self,
        scene_model,
        use_keyboard: bool = True,
        use_spacemouse: bool = False,
        nstep: int = 1,
        replay_trajectory: list[float] = None,
        replay_trajectory_path: str = None,
        use_viewer: bool = True,
        handle_id: int = 0,
    ) -> None:
        self.use_viewer = use_viewer
        self.model = scene_model
        self.data = MjData(self.model)
        self.use_keyboard = use_keyboard
        self.use_spacemouse = use_spacemouse
        self.handle_id = handle_id
        self.handle_joint_qpos = scene_model.joint(handle_id).qposadr[0]

        if use_keyboard:
            self.keyboard = Keyboard()
        if use_spacemouse:
            self.spacemouse = SpaceMouse(product_id=50746)  # 50741)
            self.start_teleop_devices()

        self.ee_pos_scale = 0.001
        self.ee_rot_scale = 0.01
        self._pos_delta = np.zeros(3)
        self._ori_delta = np.zeros(3)

        ## record mocap position and quaternion
        self.nstep = nstep
        self.mocap_pos = []
        self.mocap_quat = []
        self.renderer = MjOpenGLRenderer(model=self.model, device_id=None)  # device_id)
        self.recorders = [
            RGBRecorder(
                period_ms=100,
                camera_name="robot_0/follower",  # "robot_0/egocentric", # rum specific
                renderer=self.renderer,
                save_video=True,
                save_images=False,
            ),
            RGBRecorder(
                period_ms=100,
                camera_name="robot_0/egocentric",  # "robot_0/egocentric", # rum specific
                renderer=self.renderer,
                save_video=True,
                save_images=False,
            ),
        ]

        # replay trajectory
        self.force_applied = []
        self.torque_applied = []
        self.replay_trajectory_index = 0
        self.do_replay = False
        if replay_trajectory_path is not None:
            self.replay_trajectory_path = replay_trajectory_path
            self.replay_trajectory = np.load(self.replay_trajectory_path)
            self.do_replay = True
        elif replay_trajectory is not None:
            self.replay_trajectory = replay_trajectory
            self.do_replay = True

    def start_teleop_devices(self) -> None:
        if self.use_spacemouse:
            self.spacemouse.start_control()

    def save_mocap_data(self, save_path) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, nstep=self.nstep, mocap_pos=self.mocap_pos, mocap_quat=self.mocap_quat)

    def save(self, save_dir, xml_path, handle_name=None, success_info=None) -> None:
        new_time_date = time.strftime("%Y%m%d_%H%M%S")
        new_save_dir = os.path.join(save_dir, handle_name, new_time_date)
        os.makedirs(new_save_dir, exist_ok=True)
        # save video
        if self.recorders is not None:
            for recorder in self.recorders:
                recorder.save(new_save_dir)
        # save trajectory
        new_save_path = os.path.join(new_save_dir, "mocap_data.npz")
        # save mocap data
        # self.save_mocap_data(new_save_path)

        # save metadata
        with open(new_save_path.replace(".npz", ".txt"), "w") as f:
            f.write(f"xml_path: {xml_path}\n")
            f.write(f"nstep: {self.nstep}\n")
            if handle_name is not None:
                f.write(f"handle_name: {handle_name}\n")
            if success_info is not None:
                f.write(f"success: {success_info['success']}\n")
                f.write(f"object_type: {success_info.get('object_type', 'unknown')}\n")
                f.write(
                    f"handle_joint_pos_change: {success_info.get('handle_joint_pos_change', 0.0):.6f}\n"
                )
                f.write(
                    f"post_trajectory_handle_qpos_change: {success_info.get('post_trajectory_handle_qpos_change', 0.0):.6f}\n"
                )
                f.write(f"min_force_applied: {success_info.get('min_force_applied', 0.0):.6f}\n")
        print(f"Saved to {new_save_dir}")

    def update_target_pose_visualization(self) -> None:
        self.data.mocap_pos[0] += self._pos_delta

        curr_ori = R.from_quat(self.data.mocap_quat[0], scalar_first=True).as_euler(
            "xyz", degrees=True
        )
        quat = R.from_euler("xyz", curr_ori + self._ori_delta, degrees=True).as_quat(
            scalar_first=True
        )
        self.data.mocap_quat[0] = quat
        self.mocap_pos.append(self.data.mocap_pos[0].copy())
        self.mocap_quat.append(self.data.mocap_quat[0].copy())

    def replay(self, viewer=None) -> None:
        # Before replay
        # get handle joint position
        approach_path = self.replay_trajectory.get("approach_path", None)
        pull_path = self.replay_trajectory.get("pull_path", None)

        n_approach_path = len(approach_path["mocap_pos"])
        n_pull_path = len(pull_path["mocap_pos"])
        n_total_path = n_approach_path + n_pull_path

        force_trajectory = []
        torque_trajectory = []
        joint_pos_trajectory = []
        ##### START replay
        ## APPROACH HANDLE
        while self.replay_trajectory_index < n_approach_path:
            self.data.mocap_pos[0] = approach_path["mocap_pos"][self.replay_trajectory_index]
            self.data.mocap_quat[0] = approach_path["mocap_quat"][self.replay_trajectory_index]
            self.replay_trajectory_index += 1

            # Close some amount so it's not fully open
            self.data.ctrl[0:N_GRIPPER_JOINTS] = GRIPPER_OPEN_POS
            mujoco.mj_step(self.model, self.data, nstep=self.nstep)
            if viewer is not None:
                viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # close gripper
        for _i in range(10):
            self.data.ctrl[0:N_GRIPPER_JOINTS] = GRIPPER_CLOSE_POS
            mujoco.mj_step(self.model, self.data, nstep=self.nstep)
            force, torque = measure_contact_force(self.model, self.data)
            force_trajectory.append(force)
            torque_trajectory.append(torque)
            joint_pos_trajectory.append(self.data.qpos[self.handle_joint_qpos])

            if viewer is not None:
                viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)
                    # move backward

        # move in a circular or linear path to open
        while self.replay_trajectory_index < n_total_path:
            index = self.replay_trajectory_index - n_approach_path
            self.data.mocap_pos[0] = pull_path["mocap_pos"][index]
            self.data.mocap_quat[0] = pull_path["mocap_quat"][index]
            self.replay_trajectory_index += 1

            self.data.ctrl[0:N_GRIPPER_JOINTS] = GRIPPER_CLOSE_POS
            mujoco.mj_step(self.model, self.data, nstep=self.nstep)
            force, torque = measure_contact_force(self.model, self.data)
            force_trajectory.append(force)
            torque_trajectory.append(torque)
            joint_pos_trajectory.append(self.data.qpos[self.handle_joint_qpos])

            if viewer is not None:
                viewer.sync()

            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)
        """
        else:
            # move back to the start through a linear path to open
            while self.replay_trajectory_index > 0:
                self.replay_trajectory_index -= 1

                self.data.mocap_pos[0] = linear_path["mocap_pos"][self.replay_trajectory_index]
                self.data.mocap_quat[0] = linear_path["mocap_quat"][self.replay_trajectory_index]
                self.data.ctrl[0] = GRIPPER_CLOSE_POS
                mujoco.mj_step(self.model, self.data, nstep=self.nstep)
                if viewer is not None:
                    viewer.sync()
                if self.recorders is not None:
                    for recorder in self.recorders:
                        recorder(self.data)
        """
        # give it more time to reach the handle
        for _i in range(10):
            self.data.ctrl[0:N_GRIPPER_JOINTS] = GRIPPER_CLOSE_POS
            mujoco.mj_step(self.model, self.data, nstep=self.nstep)
            force, torque = measure_contact_force(self.model, self.data)
            force_trajectory.append(force)
            torque_trajectory.append(torque)
            joint_pos_trajectory.append(self.data.qpos[self.handle_joint_qpos])

            if viewer is not None:
                viewer.sync()
            if self.recorders is not None:
                for recorder in self.recorders:
                    recorder(self.data)

        # open gripper
        self.data.ctrl[0:N_GRIPPER_JOINTS] = GRIPPER_OPEN_POS
        mujoco.mj_step(self.model, self.data, nstep=self.nstep)
        force, torque = measure_contact_force(self.model, self.data)
        force_trajectory.append(force)
        torque_trajectory.append(torque)
        joint_pos_trajectory.append(self.data.qpos[self.handle_joint_qpos])

        if viewer is not None:
            viewer.sync()
        if self.recorders is not None:
            for recorder in self.recorders:
                recorder(self.data)

        # STOP REPLAY
        self.do_replay = False
        if viewer is not None:
            viewer.close()
        # get metadata -> handle joint position . did it open????
        self.replay_trajectory_index = 0
        self.replay_trajectory = None

        # update
        self.force_applied = force_trajectory
        self.torque_applied = torque_trajectory
        self.joint_pos_trajectory = joint_pos_trajectory

    def run(self):
        viewer = mujoco.viewer.launch_passive(self.model, self.data) if self.use_viewer else None

        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

        # Initialize mocap data
        self.mocap_pos = []
        self.mocap_quat = []
        self.update_target_pose_visualization()

        if viewer is None and self.do_replay:
            self.replay()
            return self.data

        while viewer.is_running():
            if self.do_replay:
                self.replay(viewer)
            else:
                if self.use_keyboard:
                    self.keyboard.update_velocity()
                    # TODO: change for gripper control
                    v_lin_x, v_lin_y, v_lin_z, v_yaw, v_arm_rot = self.keyboard.get_velocity()
                    ## up down/ left right/ 1 and 2 / 5 and 6
                    print(v_lin_x, v_lin_y, v_lin_z, v_yaw, v_arm_rot)

                    self._pos_delta = [
                        v_lin_x * self.ee_pos_scale,
                        v_lin_y * self.ee_pos_scale,
                        v_lin_z * self.ee_pos_scale,
                    ]
                    self._ori_delta = [0, 0, 0]

                if self.use_spacemouse:
                    joint_control = self.spacemouse.control
                    gripper_control = self.spacemouse.gripper
                    # record_state = self.spacemouse.reset_button_state
                    self.data.ctrl[0] = gripper_control

                    dx, dy, dz = joint_control[:3] * self.ee_pos_scale
                    droll, dpitch, dyaw = joint_control[3:] * self.ee_rot_scale

                    self._pos_delta = [dx, dy, dz]
                    self._ori_delta = [dpitch, -droll, -dyaw]
                self.update_target_pose_visualization()
                mujoco.mj_step(self.model, self.data, nstep=self.nstep)
                viewer.sync()
        return self.data
