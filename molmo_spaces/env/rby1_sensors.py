import json

import gymnasium.spaces as gyms
import mujoco
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.env.sensors import (
    DoorStateSensor,
    EnvStateSensor,
    LastActionSensor,
    LastCommandedEEPoseSensor,
    LastCommandedEETwistSensor,
    LastCommandedJointPosSensor,
    LastCommandedRelativeJointPosSensor,
    ObjectImagePointsSensor,
    ObjectPoseSensor,
    PolicyPhaseSensor,
    RobotBasePoseSensor,
    RobotJointPositionSensor,
    RobotJointVelocitySensor,
    TaskInfoSensor,
)
from molmo_spaces.env.sensors_cameras import (
    CameraParameterSensor,
    CameraSensor,
    DepthSensor,
)
from molmo_spaces.utils.pose import pose_mat_to_7d


class RBY1RobotStateSensor(Sensor):
    """Sensor for RBY1 robot joint positions, velocities, and dual end-effector poses."""

    def __init__(self, uuid: str = "rby1_robot_state", str_max_len: int = 4000) -> None:
        self.str_max_len = str_max_len
        # Use bytes array for HDF5 compatibility
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get RBY1 robot state observation for a specific environment in the batch."""
        robot = env.robots[batch_index]
        robot_view = robot.robot_view

        # Get joint positions and velocities
        qpos_dict = robot_view.get_qpos_dict()
        # qvel_dict = robot_view.get_qvel_dict()

        # Get dual end-effector poses for RBY1
        left_ee_pose = robot_view.get_move_group("left_arm").leaf_frame_to_robot
        right_ee_pose = robot_view.get_move_group("right_arm").leaf_frame_to_robot

        # Define expected joint groups and their max lengths for RBY1
        expected_joint_groups = {
            "left_arm": 7,  # Left arm (7 joints)
            "right_arm": 7,  # Right arm (7 joints)
            "left_gripper": 2,  # Left gripper (2 joints)
            "right_gripper": 2,  # Right gripper (2 joints)
            "base": 3,  # Base (3 DOF)
            "torso": 6,  # Torso (6 DOF)
            "head": 2,  # Head (2 DOF)
        }

        # Pad qpos to consistent lengths
        padded_qpos = {}
        for group_name, max_length in expected_joint_groups.items():
            if group_name in qpos_dict and qpos_dict[group_name].size > 0:
                actual_data = qpos_dict[group_name]
                padded_array = np.zeros(max_length)
                padded_array[: len(actual_data)] = actual_data[:max_length]  # Truncate if too long
                padded_qpos[group_name] = padded_array.tolist()
            else:
                # Fill with zeros if group doesn't exist
                padded_qpos[group_name] = [0.0] * max_length

        # Create data dict with consistent structure and ordering
        data_dict = {
            "joint_positions": padded_qpos,
            "left_ee_pose": pose_mat_to_7d(left_ee_pose).tolist(),
            "right_ee_pose": pose_mat_to_7d(right_ee_pose).tolist(),
            "timestamp": float(env.mj_datas[batch_index].time),
        }

        # Convert to JSON string, then to bytes
        data_str = json.dumps(data_dict, separators=(",", ":"))
        data_bytes = data_str.encode("utf-8")

        # Pad or truncate to fixed length
        if len(data_bytes) > self.str_max_len:
            data_bytes = data_bytes[: self.str_max_len]
        else:
            data_bytes = data_bytes + b"\x00" * (self.str_max_len - len(data_bytes))

        return np.frombuffer(data_bytes, dtype=np.uint8)


class RBY1TCPPoseSensor(Sensor):
    """Sensor for RBY1 TCP (Tool Center Point) poses in 7D format for both arms."""

    def __init__(self, uuid: str = "rby1_tcp_pose", arm_side: str = "left") -> None:
        """
        Args:
            uuid: Unique identifier for this sensor
            arm_side: Which arm to track ("left" or "right")
        """
        self.arm_side = arm_side
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get TCP pose in world coordinates for the specified arm."""

        try:
            robot_view = env.robots[batch_index].robot_view
            # Get TCP pose relative to robot base
            gripper_mg_id = f"{self.arm_side}_gripper"
            tcp_pose_matrix = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_robot
            return pose_mat_to_7d(tcp_pose_matrix).astype(np.float32)
        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get TCP pose: {e}")
            return np.zeros(7, dtype=np.float32)


class RBY1GraspStateSensor(Sensor):
    def __init__(self, uuid: str = "rby1_grasp_state", arm_side: str = "left") -> None:
        self.arm_side = arm_side
        self.finger1_name = f"robot_0/ee_finger_{self.arm_side[0]}1"  # e.g., robot_0/ee_finger_l1
        self.finger2_name = f"robot_0/ee_finger_{self.arm_side[0]}2"  # e.g., robot_0/ee_finger_l2
        self.obj_name = None  # TODO: if pick, pick and place, or open task then object name is self.get_task_objects()["pickup_obj"] if it is door opening it is self.get_task_objects()["door_handle"]
        observation_space = gyms.Box(low=0, high=1, shape=(1,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        # Lazy init
        if self.obj_name is None and task is not None:
            task_objects = task.get_task_objects(batch_index)
            if "door_handle" in task_objects:
                self.obj_name = task_objects["door_handle"]
            elif "pickup_obj" in task_objects:
                self.obj_name = task_objects["pickup_obj"]

        mj_model = env.mj_model
        mj_data = env.mj_datas[batch_index]

        finger1_contact = False
        finger2_contact = False

        # Check all contacts
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            body1_id = mj_model.geom_bodyid[contact.geom1]
            body2_id = mj_model.geom_bodyid[contact.geom2]
            body1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Check if finger1 is in contact with object
            if (body1_name == self.finger1_name and body2_name == self.obj_name) or (
                body2_name == self.finger1_name and body1_name == self.obj_name
            ):
                finger1_contact = True

            # Check if finger2 is in contact with object
            if (body1_name == self.finger2_name and body2_name == self.obj_name) or (
                body2_name == self.finger2_name and body1_name == self.obj_name
            ):
                finger2_contact = True

        grasping_object = finger1_contact and finger2_contact
        return np.array([int(grasping_object)], dtype=np.uint8)


class RBY1GraspPoseSensor(Sensor):
    """Sensor for RBY1 grasp pose in 7D format (can be current TCP or planned grasp pose)."""

    def __init__(self, uuid: str = "rby1_grasp_pose", arm_side: str = "left") -> None:
        """
        Args:
            uuid: Unique identifier for this sensor
            arm_side: Which arm to track ("left" or "right")
        """
        self.arm_side = arm_side
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get grasp pose (using current TCP pose as proxy) for the specified arm."""
        try:
            robot_view = env.robots[batch_index].robot_view

            # Get TCP pose as grasp pose proxy
            arm_group_name = f"{self.arm_side}_arm"
            tcp_pose_matrix = robot_view.get_move_group(arm_group_name).leaf_frame_to_robot
            tcp_pose_world = robot_view.base.pose @ tcp_pose_matrix

            return pose_mat_to_7d(tcp_pose_world).astype(np.float32)

        except Exception as e:
            print(f"Warning: Could not get RBY1 grasp pose for {self.arm_side} arm: {e}")
            return np.zeros(7, dtype=np.float32)


def get_rby1_door_opening_sensors(exp_config):
    """Get core sensors for RBY1 door opening data generation.

    Args:
        exp_config: Experiment configuration object with sensor parameters
    Returns:
        List of initialized sensors
    """
    sensors = []
    for camera_spec in exp_config.camera_config.cameras:
        camera_name = camera_spec.name
        camera_name = camera_name.split("/")[-1]
        cam_params = CameraParameterSensor(
            camera_name=camera_name, uuid=f"sensor_param_{camera_name}"
        )
        sensors.append(cam_params)

        # RGB sensor
        cam_rgb = CameraSensor(
            camera_name=camera_name,
            img_resolution=exp_config.camera_config.img_resolution,
            uuid=camera_name,
        )
        sensors.append(cam_rgb)
        if camera_spec.record_depth:
            cam_depth = DepthSensor(
                camera_name=camera_name,
                img_resolution=exp_config.camera_config.img_resolution,
                uuid=f"{camera_name}_depth",
            )
            sensors.append(cam_depth)

    # Agent data sensors - RBY1 has more joints (dual arm + base + torso + head + grippers)
    sensors.append(RobotJointVelocitySensor(uuid="qvel", max_joints=25))
    sensors.append(RobotJointPositionSensor(uuid="qpos", max_joints=25))
    # Action sensors - RBY1 specific action spec
    sensors.append(
        LastActionSensor(
            dtype=exp_config.task_config.action_dtype,
        )
    )
    # Position action sensors
    sensors.append(LastCommandedJointPosSensor())
    sensors.append(LastCommandedRelativeJointPosSensor())
    sensors.append(LastCommandedEETwistSensor())
    sensors.append(LastCommandedEEPoseSensor())
    # Door opening specific sensors
    sensors.append(DoorStateSensor(uuid="door_state"))

    # Door opening policy phase sensor
    sensors.append(PolicyPhaseSensor(uuid="policy_phase"))

    # TCP poses for both arms (RBY1 is dual-arm) - use separate sensors for each arm
    sensors.append(RBY1TCPPoseSensor(uuid="left_tcp_pose", arm_side="left"))
    sensors.append(RBY1TCPPoseSensor(uuid="right_tcp_pose", arm_side="right"))
    sensors.append(RobotBasePoseSensor(uuid="robot_base_pose"))

    # Grasp state for both arms
    sensors.append(RBY1GraspStateSensor(uuid="rby1_left_grasp_state", arm_side="left"))
    sensors.append(RBY1GraspStateSensor(uuid="rby1_right_grasp_state", arm_side="right"))

    # Environment state sensors
    sensors.append(EnvStateSensor(uuid="env_states"))
    sensors.append(TaskInfoSensor(uuid="task_info"))

    # sensors.append(RBY1RobotStateSensor(uuid="robot_state"))
    # TODO: just make sure the door and handle are in the tracked objects and have this take care of it
    sensors.append(
        ObjectPoseSensor(
            object_names=exp_config.task_config.tracked_object_names or [],
            uuid="object_poses",
        )
    )

    # Object tracking sensor for image points (door handle, etc.)
    sensors.append(ObjectImagePointsSensor(exp_config=exp_config))

    return sensors
