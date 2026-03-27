import logging

import gymnasium.spaces as gyms
import numpy as np

from molmo_spaces.configs.robot_configs import RBY1Config
from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.env.sensors_cameras import (
    CameraParameterSensor,
    CameraSensor,
    DepthSensor,
)
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.camera_utils import erode_segmentation_mask, normalize_points
from molmo_spaces.utils.linalg_utils import transform_to_twist
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


def _cmd_joint_pos(robot: Robot):
    # if the unnoised cmd joint pos is available, use it
    unnoised_cmd_jp = robot.last_unnoised_cmd_joint_pos()
    if unnoised_cmd_jp is not None:
        return unnoised_cmd_jp

    # robot.update_control() hasn't been called yet, recover the command from the controllers
    cmd_jp: dict[str, np.ndarray] = {}
    for name, controller in robot.controllers.items():
        if isinstance(controller, AbstractPositionController):
            cmd_jp[name] = controller.target_pos
        else:
            raise NotImplementedError(
                f"Controller '{name}' is not a position controller, saving is unsupported!"
            )
    return cmd_jp


class RobotStateSensor(Sensor):
    """Sensor for robot joint positions, velocities, and end-effector pose."""

    def __init__(self, uuid: str = "robot_state", str_max_len: int = 2000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        # Use bytes array for HDF5 compatibility
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get robot state observation for a specific environment in the batch."""
        robot = env.robots[batch_index]
        robot_view = robot.robot_view

        # Get joint positions and velocities
        qpos_dict = robot_view.get_qpos_dict()
        # qvel_dict = robot_view.get_qvel_dict()

        # Get end-effector pose
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_robot

        # Ensure consistent structure with fixed-length arrays
        # Define expected joint groups and their max lengths for padding
        expected_joint_groups = {
            "arm": 7,  # Typical arm has 7 joints
            "gripper": 2,  # Typical gripper has 2 joints
            "base": 3,  # Base might have 3 DOF
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
        data = {
            "qpos": padded_qpos,
            # "qvel": padded_qvel,  # Add when needed
            "ee_pose": ee_pose.tolist(),
        }

        # Convert to JSON string and then to bytes with consistent formatting
        return data


class TCPPoseSensor(Sensor):
    """Sensor for TCP (Tool Center Point / End Effector) pose in 7D format."""

    def __init__(self, uuid: str = "tcp_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get TCP pose in robot frame."""
        try:
            robot_view = env.robots[batch_index].robot_view
            # Get TCP pose relative to robot base
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            tcp_pose_matrix = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_robot

            return pose_mat_to_7d(tcp_pose_matrix).astype(np.float32)

        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get TCP pose: {e}")
            return np.zeros(7, dtype=np.float32)


class RobotBasePoseSensor(Sensor):
    """Sensor for robot base pose in 7D format."""

    def __init__(self, uuid: str = "robot_base_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get robot base pose."""
        try:
            robot_view = env.robots[batch_index].robot_view
            base_pose = robot_view.base.pose

            return pose_mat_to_7d(base_pose).astype(np.float32)

        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get robot base pose: {e}")
            return np.zeros(7, dtype=np.float32)


class ObjectPoseSensor(Sensor):
    """Sensor for object poses relative to robot base."""

    def __init__(
        self,
        object_names: list[str],
        uuid: str = "object_poses",
        str_max_len: int = 2000,
    ) -> None:
        self.object_names = object_names
        self.str_max_len = str_max_len
        self.is_dict = True
        # Use bytes array for HDF5 compatibility
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get object poses relative to robot base for a specific environment."""
        data = env.mj_datas[batch_index]
        robot_view = env.robots[batch_index].robot_view

        # Ensure consistent structure - always include all tracked objects
        object_poses = {}
        valid_objects = [x for x in self.object_names if x is not None]
        for obj_name in sorted(valid_objects):  # Sort for consistent ordering
            # try:
            obj_body = create_mlspaces_body(data, obj_name)
            # Get pose relative to robot base
            obj_pose_rel = np.linalg.inv(robot_view.base.pose) @ obj_body.pose
            object_poses[obj_name] = obj_pose_rel.tolist()
            # except (AttributeError, KeyError) as e:
            #    # Always use identity matrix for missing objects (consistent structure)
            #    object_poses[obj_name] = np.eye(4).tolist()

        # Convert to JSON string with consistent formatting
        return object_poses


class LastCommandedJointPosSensor(Sensor):
    def __init__(self, uuid: str = "actions/joint_pos") -> None:
        super().__init__(uuid=uuid, observation_space=gyms.Box(0, 255, (1,), dtype=np.uint8))
        self.is_dict = True

    def get_observation(
        self, env: BaseMujocoEnv, task: BaseMujocoTask, batch_index: int = 0, *args, **kwargs
    ):
        # sentinel done action when the task is terminal
        if task.is_terminal():
            return {}
        robot = env.robots[batch_index]
        return {k: v.tolist() for k, v in _cmd_joint_pos(robot).items()}


class LastCommandedRelativeJointPosSensor(Sensor):
    def __init__(self, uuid: str = "actions/joint_pos_rel") -> None:
        super().__init__(uuid=uuid, observation_space=gyms.Box(0, 255, (1,), dtype=np.uint8))
        self.is_dict = True
        self._prev_jp = None

    def get_observation(
        self, env: BaseMujocoEnv, task: BaseMujocoTask, batch_index: int = 0, *args, **kwargs
    ):
        # sentinel done action when the task is terminal
        if task.is_terminal():
            return {}

        robot = env.robots[batch_index]
        # on the first step, return dummy action
        if self._prev_jp is None:
            self._prev_jp = robot.robot_view.get_qpos_dict()
            return {name: np.zeros_like(jp).tolist() for name, jp in self._prev_jp.items()}

        prev_cmd_jp = _cmd_joint_pos(robot)
        cmd_rel_jp: dict[str, list[float]] = {}
        for name, jp in prev_cmd_jp.items():
            # some move groups (e.g. grippers) have different action and state dimensions,
            # it doesn't make much sense to compute relative actions in these cases.
            if jp.shape == self._prev_jp[name].shape:
                cmd_rel_jp[name] = (jp - self._prev_jp[name]).tolist()
        self._prev_jp = robot.robot_view.get_qpos_dict()
        return cmd_rel_jp

    def reset(self) -> None:
        self._prev_jp = None


class LastCommandedEEPoseSensor(Sensor):
    def __init__(self, uuid: str = "actions/ee_pose") -> None:
        super().__init__(uuid=uuid, observation_space=gyms.Box(0, 255, (1,), dtype=np.uint8))
        self.is_dict = True

    def get_observation(
        self, env: BaseMujocoEnv, task: BaseMujocoTask, batch_index: int = 0, *args, **kwargs
    ):
        # sentinel done action when the task is terminal
        if task.is_terminal():
            return {}

        robot = env.robots[batch_index]
        prev_cmd_jp = _cmd_joint_pos(robot)
        prev_cmd_poses = robot.kinematics.fk(prev_cmd_jp, np.eye(4), rel_to_base=True)
        prev_cmd_posquats = {
            name: pose_mat_to_7d(pose).tolist() for name, pose in prev_cmd_poses.items()
        }
        return prev_cmd_posquats


class LastCommandedEETwistSensor(Sensor):
    def __init__(self, uuid: str = "actions/ee_twist") -> None:
        super().__init__(uuid=uuid, observation_space=gyms.Box(0, 255, (1,), dtype=np.uint8))
        self.is_dict = True
        self._prev_poses = None
        self._tracked_keys: set[str] | None = None  # move group ids that are position-commanded

    def get_observation(
        self, env: BaseMujocoEnv, task: BaseMujocoTask, batch_index: int = 0, *args, **kwargs
    ):
        # sentinel done action when the task is terminal
        if task.is_terminal():
            return {}

        robot = env.robots[batch_index]
        prev_cmd_jp = _cmd_joint_pos(robot)
        if self._tracked_keys is None:
            self._tracked_keys = set(prev_cmd_jp.keys())
        curr_poses = {
            name: robot.robot_view.get_move_group(name).leaf_frame_to_robot
            for name in self._tracked_keys
        }

        # on the first step, return dummy action
        if self._prev_poses is None:
            self._prev_poses = curr_poses
            return {name: np.zeros(6).tolist() for name in self._tracked_keys}

        prev_cmd_poses = robot.kinematics.fk(prev_cmd_jp, np.eye(4), rel_to_base=True)
        prev_cmd_twists: dict[str, list[float]] = {}
        for name, prev_pose in self._prev_poses.items():
            prev_cmd_pose = prev_cmd_poses[name]
            cmd_twist = np.concatenate(transform_to_twist(np.linalg.inv(prev_pose) @ prev_cmd_pose))
            prev_cmd_twists[name] = cmd_twist.tolist()
        self._prev_poses = curr_poses
        return prev_cmd_twists

    def reset(self) -> None:
        self._prev_poses = None


class LastActionSensor(Sensor):
    """Sensor for robot actions (absolute values only)."""

    def __init__(
        self,
        dtype: str = "float32",
        uuid: str = "actions/commanded_action",
        str_max_len: int = 2000,
    ) -> None:
        """
        Args:
            dtype: Target dtype for all action components
            uuid: Sensor UUID
        """
        self.dtype = getattr(np, dtype)
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict[str, list]:
        """Get action dictionary with absolute values."""
        current_action = getattr(task, "last_action", None)

        if current_action is None:
            # Return dummy if no action has been set yet (padding value)
            action_dict = {}
        else:
            action_dict = {}
            for component_name, data in current_action.items():
                if not isinstance(data, np.ndarray):
                    data = np.array(data, dtype=self.dtype)
                else:
                    data = data.astype(self.dtype)
                if data.ndim == 0:
                    data = data.reshape(1)
                action_dict[component_name] = data.tolist()

        return action_dict


class RobotJointPositionSensor(Sensor):
    """Sensor for robot joint positions as numerical array."""

    def __init__(self, uuid: str = "qpos", max_joints: int = 9, str_max_len: int = 2000) -> None:
        self.max_joints = max_joints
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env, task, batch_index: int = 0, *args, **kwargs
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Get robot joint positions as numerical array."""
        robot = env.robots[batch_index]
        robot_view = robot.robot_view

        # Get joint positions
        qpos_dict = robot_view.get_qpos_dict()

        qpos_data = {k: v.astype(np.float32).tolist() for k, v in qpos_dict.items()}
        return qpos_data


class RobotJointVelocitySensor(Sensor):
    """Sensor for robot joint velocities as numerical array."""

    def __init__(self, uuid: str = "qvel", max_joints: int = 9, str_max_len: int = 2000) -> None:
        self.max_joints = max_joints
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env, task, batch_index: int = 0, *args, **kwargs
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Get robot joint velocities as numerical array."""
        robot = env.robots[batch_index]
        robot_view = robot.robot_view

        # Get joint velocities
        qvel_dict = robot_view.get_qvel_dict()

        qvel_data = {k: v.astype(np.float32).tolist() for k, v in qvel_dict.items()}
        return qvel_data


class EnvStateSensor(Sensor):
    # TODO(rose) BUSTED - ag
    """Sensor for complete MuJoCo environment state."""

    def __init__(self, uuid: str = "env_states", str_max_len: int = 50000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        # Use bytes array for HDF5 compatibility
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get complete environment state from MuJoCo."""
        try:
            import mujoco

            from molmo_spaces.env.data_views import create_mlspaces_body

            model = env.mj_model
            data = env.mj_datas[batch_index]

            # Collect all body states
            actors = {}
            articulations = {}

            # Get robot state first (for articulations)
            robot_view = env.robots[batch_index].robot_view
            all_qpos = []
            all_qvel = []
            for mg_name in robot_view.move_group_ids():
                mg = robot_view.get_move_group(mg_name)
                all_qpos.extend(mg.joint_pos)
                all_qvel.extend(mg.joint_vel)

            # Pad robot state to 31 dimensions
            robot_state = np.zeros(31)
            combined_state = np.concatenate([all_qpos, all_qvel])
            actual_length = min(len(combined_state), 31)
            robot_state[:actual_length] = combined_state[:actual_length]
            articulations["panda"] = robot_state.tolist()

            # Get ALL bodies for actors (not hardcoded)
            for body_id in range(model.nbody):
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None:
                    continue

                # Skip robot bodies (they go in articulations)
                if any(
                    robot_name in body_name.lower() for robot_name in ["panda", "franka", "robot"]
                ):
                    continue

                try:
                    body = create_mlspaces_body(data, body_name)

                    # Create 13D state: position(3) + quaternion(4) + velocity(6)
                    body_state = np.concatenate(
                        [
                            body.position,  # 3D position
                            body.quaternion,  # 4D quaternion [w,x,y,z]
                            body.velocities[:6],  # 6D velocity (3 linear + 3 angular)
                        ]
                    )
                    actors[body_name] = body_state.tolist()

                except Exception:
                    # Skip bodies that can't be processed
                    continue

            # Create complete environment state
            env_state_data = {"actors": actors, "articulations": articulations}

            # Convert to JSON string with consistent formatting
            return env_state_data

        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get environment state: {e}")
            # Return empty structure
            empty_data = {
                "actors": {},
                "articulations": {"panda": np.zeros(31).tolist()},
            }
            return empty_data


class ActorStateSensor(Sensor):
    """Sensor for actor (object) states in numerical format."""

    def __init__(self, actor_names: list[str], uuid: str = "actor_states") -> None:
        self.actor_names = actor_names
        # Each actor has 13D state: pos(3) + quat(4) + vel(6)
        observation_space = gyms.Box(
            low=-np.inf, high=np.inf, shape=(len(actor_names), 13), dtype=np.float32
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get actor states as numerical array."""
        try:
            from molmo_spaces.env.data_views import create_mlspaces_body

            data = env.mj_datas[batch_index]
            actor_states = []

            for actor_name in self.actor_names:
                try:
                    body = create_mlspaces_body(data, actor_name)

                    # Create 13D state: position(3) + quaternion(4) + velocity(6)
                    actor_state = np.concatenate(
                        [
                            body.position,  # 3D position
                            body.quaternion,  # 4D quaternion [w,x,y,z]
                            body.velocities[:6],  # 6D velocity (3 linear + 3 angular)
                        ]
                    )
                    actor_states.append(actor_state)

                except Exception:
                    # Use zeros for missing/invalid actors
                    actor_states.append(np.zeros(13))

            return np.array(actor_states, dtype=np.float32)

        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get actor states: {e}")
            return np.zeros((len(self.actor_names), 13), dtype=np.float32)


class ArticulationStateSensor(Sensor):
    """Sensor for articulation (robot) state in numerical format."""

    def __init__(self, state_dim: int = 31, uuid: str = "articulation_states") -> None:
        self.state_dim = state_dim
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get articulation state as numerical array."""
        try:
            robot_view = env.robots[batch_index].robot_view

            # Collect all joint positions and velocities
            all_qpos = []
            all_qvel = []
            for mg_name in robot_view.move_group_ids():
                mg = robot_view.get_move_group(mg_name)
                all_qpos.extend(mg.joint_pos)
                all_qvel.extend(mg.joint_vel)

            # Combine and pad to target dimension
            combined_state = np.concatenate([all_qpos, all_qvel])
            padded_state = np.zeros(self.state_dim, dtype=np.float32)
            actual_length = min(len(combined_state), self.state_dim)
            padded_state[:actual_length] = combined_state[:actual_length]

            return padded_state

        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get articulation state: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)


#
# Task based sensors
#
class GraspStateSensor(Sensor):
    """
    Sensor for grasp state. For each gripper, track whether it is touching (and possibly holding) the object.
    The held state is calculated with a heuristic, and only tracks whether the object is only touching the gripper,
    not whether the object is actually stably supported by the gripper. More sophisticated heuristics may be added in the future.
    """

    def __init__(self, object_name: str, uuid: str | None = None, str_max_len: int = 2000) -> None:
        self.object_name = object_name
        self.str_max_len = str_max_len
        self.is_dict = True
        if uuid is None:
            uuid = f"grasp_state_{object_name}"

        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

        self._object_geoms: set[int] | None = None
        self._gripper_geoms: dict[str, set[int]] | None = None

    def reset(self) -> None:
        self._object_geoms = None
        self._gripper_geoms = None

    def get_observation(
        self, env: BaseMujocoEnv, task: BaseMujocoTask, batch_index: int = 0, *args, **kwargs
    ) -> dict:
        model = env.mj_model
        robot_view = env.robots[batch_index].robot_view

        if self._gripper_geoms is None:
            self._gripper_geoms = {}
            for mg_id in robot_view.get_gripper_movegroup_ids():
                mg = robot_view.get_move_group(mg_id)
                self._gripper_geoms[mg_id] = descendant_geoms(
                    env.mj_model, mg.root_body_id, visual_only=False
                )

        if self._object_geoms is None:
            object_body = create_mlspaces_body(env.mj_datas[batch_index], self.object_name)
            self._object_geoms = set(
                descendant_geoms(model, object_body.body_id, visual_only=False)
            )

        held = True
        gripper_touching = {k: False for k in self._gripper_geoms}

        for cid in range(env.mj_datas[batch_index].ncon):
            c = env.mj_datas[batch_index].contact[cid]

            # skip contacts between the object and itself and contacts not involving the object
            if (c.geom1 in self._object_geoms) == (c.geom2 in self._object_geoms):
                continue

            other_geom = c.geom2 if c.geom1 in self._object_geoms else c.geom1
            for gripper_id, gripper_geoms in self._gripper_geoms.items():
                if other_geom in gripper_geoms:
                    gripper_touching[gripper_id] = True
                    break
            else:
                # object is in contact with a non-gripper geom, so it is not held
                held = False

        grasp_state = {}
        for gripper_id, touching in gripper_touching.items():
            grasp_state[gripper_id] = {
                "touching": touching,
                "held": held and touching,
            }

        return grasp_state


class TaskInfoSensor(Sensor):
    """Sensor for task information."""

    def __init__(self, uuid: str = "task_info", str_max_len: int = 4000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def _sanitize(self, info: dict) -> dict:
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                info[k] = v.tolist() if v.ndim > 0 else v.item()
            elif isinstance(v, np.floating | np.integer | np.bool_):
                info[k] = v.item()
            elif isinstance(v, dict):
                self._sanitize(v)

    def get_observation(
        self, env: BaseMujocoEnv, task: BaseMujocoTask, batch_index: int = 0, *args, **kwargs
    ) -> dict:
        info = task.get_info()[batch_index]
        self._sanitize(info)
        return info


class ObjectStartPoseSensor(Sensor):
    """Sensor for initial object pose in 7D format (x, y, z, qw, qx, qy, qz)."""

    def __init__(self, object_name: str, uuid: str | None = None) -> None:
        self.object_name = object_name
        if uuid is None:
            uuid = f"obj_start_{object_name}"

        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

        # Store initial pose when first called
        self._initial_pose = None

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get initial object pose."""
        if task.config.task_type in ["pick", "open", "close"]:
            return np.array(task.config.task_config.pickup_obj_start_pose, dtype=np.float32)
        else:
            try:
                data = env.mj_datas[batch_index]
                # Store initial pose on first call (typically during reset)
                if self._initial_pose is None:
                    obj_body = create_mlspaces_body(data, self.object_name)
                    self._initial_pose = pose_mat_to_7d(obj_body.pose)
                return self._initial_pose.astype(np.float32)
            except (AttributeError, KeyError) as e:
                raise ValueError(f"Could not get initial pose for object {self.object_name}") from e

    def reset(self) -> None:
        """Reset stored initial pose."""
        self._initial_pose = None


class ObjectEndPoseSensor(Sensor):
    """Sensor for target/end object pose in 7D format (x, y, z, qw, qx, qy, qz)."""

    def __init__(self, object_name: str, uuid: str | None = None) -> None:
        self.object_name = object_name
        if uuid is None:
            uuid = f"obj_end_{object_name}"

        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get target object pose."""
        if task.config.task_type in ["pick", "open", "close"]:
            goal_pose = np.array(task.config.task_config.pickup_obj_goal_pose, dtype=np.float32)
            return goal_pose
        else:
            # TODO(max): fix this
            goal_pose = np.zeros(7, dtype=np.float32)
            return goal_pose
            # raise ValueError(f"Invalid action type {task.config.task_type}")


class DoorStateSensor(Sensor):
    """Sensor for door state including joint angle and opening percentage."""

    def __init__(
        self,
        uuid: str = "door_state",
        str_max_len: int = 1000,
    ) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        # Use bytes array for HDF5 compatibility
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env, task, batch_index: int = 0, *args, **kwargs
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Get door state as encoded JSON."""
        try:
            # Get door state from task
            # TODO: Add handle bbox dimensions
            door_state_data = {
                "joint_angle": [0.0],
                "opening_percentage": [0.0],
                "handle_position": [0.0, 0.0, 0.0],
                "handle_extents": [0.0, 0.0, 0.0],
                "door_position": [0.0, 0.0, 0.0],
                "is_open": [False],
            }

            if hasattr(task, "door_object") and task.door_object is not None:
                # Get door joint angle
                if hasattr(task, "current_door_joint_state"):
                    door_state_data["joint_angle"] = [float(task.current_door_joint_state.item())]

                # Calculate opening percentage
                if hasattr(task, "exp_config") and hasattr(
                    task.config.task_config, "articulated_joint_range"
                ):
                    joint_range = task.config.task_config.articulated_joint_range
                    current_angle = door_state_data["joint_angle"]
                    opening_percentage = (current_angle - joint_range[0]) / (
                        joint_range[1] - joint_range[0]
                    )
                    if isinstance(opening_percentage, np.ndarray):
                        opening_percentage = opening_percentage.item()
                    door_state_data["opening_percentage"] = [
                        float(np.clip(opening_percentage, 0.0, 1.0))
                    ]

                # Get door handle position
                if hasattr(task, "get_door_handle_position"):
                    try:
                        handle_pos = task.get_door_handle_position()
                        door_state_data["handle_position"] = handle_pos.tolist()
                    except AttributeError:
                        pass

                # Get door handle bbox extents
                if hasattr(task, "get_door_handle_extents"):
                    try:
                        handle_extents = task.get_door_handle_extents()
                        door_state_data["handle_extents"] = handle_extents.tolist()
                    except AttributeError:
                        pass

                # Get door position
                if hasattr(task, "get_door_joint_position"):
                    try:
                        door_pos = task.get_door_joint_position()
                        door_state_data["door_position"] = door_pos.tolist()
                    except AttributeError:
                        pass

                # Check if door is considered open
                if hasattr(task, "door_opened"):
                    door_state_data["is_open"] = [bool(task.door_opened)]

            return door_state_data

        except (AttributeError, KeyError) as e:
            print(f"Warning: Could not get door state: {e}")
            # Return empty structure
            empty_data = {
                "joint_angle": 0.0,
                "opening_percentage": 0.0,
                "handle_position": [0.0, 0.0, 0.0],
                "handle_extents": [0.0, 0.0, 0.0],
                "door_position": [0.0, 0.0, 0.0],
                "is_open": False,
            }
            return empty_data


class ObjectImagePointsSensor(Sensor):
    """Sensor for tracking object pixel coordinates across multiple cameras.

    Detects task objects in camera views and returns sampled pixel coordinates
    normalized to 0-1 range. Returns HDF5-native numpy arrays (not JSON).

    Output structure (nested dict of numpy arrays):
    {
        "<object_key>": {
            "<camera_name>": {
                "points": np.ndarray (max_points, 2) - normalized (x, y) coords, NaN-padded
                "num_points": np.ndarray (1,) - number of valid points
            },
            ...
        },
        ...
    }
    """

    def __init__(
        self,
        exp_config,
        camera_names: list[str] | None = None,
        uuid: str = "object_image_points",
        max_points: int = 10,
        erosion_iterations: int = 2,
        exclude_follower_cameras: bool = True,
    ) -> None:
        """
        Args:
            exp_config: Experiment configuration with camera_config
            camera_names: Optional list of camera names to track. If None, uses all cameras from config.
            uuid: Unique sensor identifier
            max_points: Maximum number of points to sample per camera
            erosion_iterations: Number of erosion iterations for segmentation mask
            exclude_follower_cameras: If True, exclude cameras with 'follower' in their name
        """
        # Build camera spec lookup from config
        all_camera_specs = {
            camera_spec.name: camera_spec for camera_spec in exp_config.camera_config.cameras
        }

        # Filter to requested cameras or use all
        if camera_names is not None:
            self.camera_specs = {}
            for cam_name in camera_names:
                if cam_name not in all_camera_specs:
                    raise ValueError(
                        f"Camera '{cam_name}' not found in camera config. Available cameras: {list(all_camera_specs.keys())}"
                    )
                self.camera_specs[cam_name] = all_camera_specs[cam_name]
        else:
            self.camera_specs = all_camera_specs

        # Exclude follower cameras (they follow robot/workspace and aren't useful for object tracking)
        if exclude_follower_cameras:
            self.camera_specs = {
                name: spec
                for name, spec in self.camera_specs.items()
                if "follower" not in name.lower()
            }

        self.camera_names = list(self.camera_specs.keys())
        self.img_width, self.img_height = exp_config.camera_config.img_resolution
        self.max_points = max_points
        self.erosion_iterations = erosion_iterations

        # Use nested dict structure - NOT is_dict (which triggers JSON serialization)
        # The nested dict will be handled natively by batch_observations and save_utils
        self.is_dict = False

        # Observation space is a nested dict structure (for reference, not enforced)
        # Each camera produces (max_points, 2) points array + (1,) num_points array
        observation_space = gyms.Dict({})  # Dynamic based on task objects
        super().__init__(uuid=uuid, observation_space=observation_space)

    def _create_empty_points_data(self) -> dict[str, np.ndarray]:
        """Create empty points data structure for a single camera."""
        return {
            "points": np.full((self.max_points, 2), np.nan, dtype=np.float32),
            "num_points": np.array([0], dtype=np.int32),
        }

    def get_observation(
        self,
        env,
        task,
        batch_index: int = 0,
        *args,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Get pixel coordinates of task objects in each camera view.

        All points are normalized to 0-1 range based on image resolution.
        Returns HDF5-native numpy arrays (not JSON strings).

        Returns:
            Dictionary with structure:
            {
                "<object_key>": {
                    "<camera_name>": {
                        "points": np.ndarray (max_points, 2) - normalized (x, y), NaN-padded
                        "num_points": np.ndarray (1,) - count of valid points
                    },
                    ...
                },
                ...
            }
            Where object_key is determined by task.get_task_objects() (e.g., 'pickup_obj',
            'place_receptacle', 'door_handle').
        """
        # Get object names from task's get_task_objects() method (preferred)
        object_names = {}
        if hasattr(task, "get_task_objects"):
            object_names = task.get_task_objects(batch_index)

        # Fall back to legacy config-based lookup for backward compatibility
        if not object_names:
            if hasattr(task, "config") and hasattr(task.config, "task_config"):
                task_config = task.config.task_config
                if hasattr(task_config, "pickup_obj_name") and task_config.pickup_obj_name:
                    object_names["pickup_obj"] = task_config.pickup_obj_name
                if (
                    hasattr(task_config, "place_receptacle_name")
                    and task_config.place_receptacle_name
                ):
                    object_names["place_receptacle"] = task_config.place_receptacle_name

        # Initialize result dict with empty numpy arrays for all object/camera combinations
        result = {
            obj_key: {camera: self._create_empty_points_data() for camera in self.camera_names}
            for obj_key in object_names
        }

        # If no objects found, return empty results
        if not object_names:
            log.warning("No task objects found from get_task_objects() or task config")
            return result

        # Check if environment supports segmentation masks
        if not hasattr(env, "get_segmentation_mask_of_object"):
            log.warning("Environment does not support segmentation masks")
            return result

        # Process each object
        for obj_key, obj_name in object_names.items():
            for camera_name in self.camera_names:
                try:
                    # Get segmentation mask for the target object
                    segmentation_mask = env.get_segmentation_mask_of_object(
                        obj_name, camera_name=camera_name, batch_index=batch_index
                    )

                    if segmentation_mask is None or not np.any(segmentation_mask > 0):
                        # Keep empty result (already initialized with NaN/0)
                        continue

                    # Skip erosion for thin objects (door handles, grippers) or cameras configured to skip
                    camera_spec = self.camera_specs.get(camera_name)
                    skip_erosion = "handle" in obj_key or (camera_spec and camera_spec.skip_erosion)
                    if skip_erosion:
                        mask_to_use = segmentation_mask
                    else:
                        # Erode mask to get more stable interior points
                        eroded_mask = erode_segmentation_mask(
                            segmentation_mask, iterations=self.erosion_iterations
                        )
                        # Fall back to original mask if erosion removed all points
                        if eroded_mask is not None and np.any(eroded_mask > 0):
                            mask_to_use = eroded_mask
                        else:
                            mask_to_use = segmentation_mask

                    # Find the points where the object is visible
                    if mask_to_use is not None and np.any(mask_to_use > 0):
                        points = np.argwhere(mask_to_use > 0)

                        # Sample random subset up to max_points
                        num_points = min(len(points), self.max_points)
                        if len(points) > self.max_points:
                            indices = np.random.choice(len(points), self.max_points, replace=False)
                            points = points[indices]

                        # Switch from (row, col) to (x, y) format
                        switched_points = points[:, [1, 0]].astype(np.float32)

                        # Check if this specific camera is warped
                        is_warped = camera_spec.is_warped if camera_spec else False

                        # Get distortion map if camera is warped
                        distortion_map = None
                        if is_warped:
                            raise NotImplementedError(
                                "Distortion map not implemented - what are you doing here?"
                            )

                        # Normalize points (handles both distortion correction and 0-1 normalization)
                        normalized_points = normalize_points(
                            switched_points,
                            self.img_width,
                            self.img_height,
                            distortion_map=distortion_map,
                        )

                        # Round to 4 decimal places
                        rounded_points = np.round(normalized_points, 4)

                        # Create padded array (NaN for unused slots)
                        points_array = np.full((self.max_points, 2), np.nan, dtype=np.float32)
                        points_array[:num_points] = rounded_points

                        result[obj_key][camera_name] = {
                            "points": points_array,
                            "num_points": np.array([num_points], dtype=np.int32),
                        }

                except NotImplementedError:
                    log.warning(
                        f"Segmentation mask retrieval not yet implemented for {camera_name}"
                    )
                    # Keep empty result
                except Exception as e:
                    log.exception(
                        f"Error processing camera {camera_name} for object {obj_name}: {e}"
                    )
                    # Keep empty result

        return result


#
# Policy based sensors
#
class PolicyPhaseSensor(Sensor):
    """Sensor for tracking the current phase of a planner policy."""

    def __init__(self, uuid: str = "policy_phase") -> None:
        observation_space = gyms.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> int:
        """Return the current phase of the policy as a string encoded as uint8 array."""
        if task._registered_policy is not None:
            phase_name = task._registered_policy.get_phase()
            all_phases = task._registered_policy.get_all_phases()
            try:
                phase_num = all_phases[phase_name]
            except KeyError:
                log.warning(f"Unknown phase {phase_name}, options are {all_phases.keys()}")
                phase_num = -1
        else:
            log.warning("No registered policy, cannot get policy phase. Using default phase -1.")
            phase_num = -1

        return int(phase_num)


class PolicyNumRetriesSensor(Sensor):
    """Sensor for tracking the number of retries of the object manipulation policy."""

    def __init__(self, uuid: str = "policy_num_retries") -> None:
        observation_space = gyms.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)
        self._logged_warning = False

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> int:
        """Return the number of retries of the object manipulation policy."""
        policy = task._registered_policy
        if policy is None:
            if not self._logged_warning:
                log.warning("No registered policy, cannot track retries.")
                self._logged_warning = True
            return 0
        elif hasattr(policy, "retry_count"):
            return policy.retry_count
        else:
            if not self._logged_warning:
                log.warning(f"Policy {type(policy)} does not support tracking retries.")
                self._logged_warning = True
            return 0

    def reset(self) -> None:
        super().reset()
        self._logged_warning = False


class GraspPoseSensor(Sensor):
    """Sensor for the planned grasp pose in 7D format."""

    def __init__(self, uuid: str = "grasp_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get grasp pose (using current TCP pose as proxy)."""
        if task._registered_policy is None:
            log.warning("No registered policy, cannot get grasp pose.")
            return np.zeros(7, dtype=np.float32)
        else:
            return np.array(
                pose_mat_to_7d(task._registered_policy.target_poses["grasp"]),
                dtype=np.float32,
            )


def get_core_sensors(exp_config):
    """Get core sensors for Franka pick-place data generation.

    Args:
        exp_config: Experiment configuration object with sensor parameters

    Returns:
        List of initialized sensors
    """
    sensors = []

    # Get camera names dynamically from camera config instead of hardcoded list
    for camera_spec in exp_config.camera_config.cameras:
        camera_name = camera_spec.name

        # Camera parameter sensor
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

        # Depth sensor (conditional based on camera config)
        if camera_spec.record_depth:
            cam_depth = DepthSensor(
                camera_name=camera_name,
                img_resolution=exp_config.camera_config.img_resolution,
                uuid=f"{camera_name}_depth",
            )
            sensors.append(cam_depth)

        # # Segmentation sensor
        # cam_seg = SegmentationSensor(camera_name=camera_name, img_resolution=config.img_resolution, uuid=f"{camera_name}_seg")
        # sensors.append(cam_seg)

    # Robot State sensors
    sensors.append(RobotJointPositionSensor(uuid="qpos", max_joints=9))
    sensors.append(RobotJointVelocitySensor(uuid="qvel", max_joints=9))
    sensors.append(TCPPoseSensor(uuid="tcp_pose"))
    sensors.append(RobotBasePoseSensor(uuid="robot_base_pose"))

    # Environment state sensors
    sensors.append(EnvStateSensor(uuid="env_states"))

    # Task pose sensors
    sensors.append(
        ObjectStartPoseSensor(
            object_name=exp_config.task_config.pickup_obj_name, uuid="obj_start_pose"
        )
    )
    sensors.append(
        ObjectEndPoseSensor(
            object_name=exp_config.task_config.place_target_name, uuid="obj_end_pose"
        )
    )
    sensors.append(
        GraspStateSensor(
            object_name=exp_config.task_config.pickup_obj_name,
            uuid="grasp_state_pickup_obj",
        )
    )
    # TODO: this kind of hacky hardcoded conditionals should be refactored.
    # Tasks should register their own task-specific sensors.
    if (
        hasattr(exp_config.task_config, "place_receptacle_name")
        and exp_config.task_config.place_receptacle_name
    ):
        sensors.append(
            GraspStateSensor(
                object_name=exp_config.task_config.place_receptacle_name,
                uuid="grasp_state_place_receptacle",
            )
        )
    if isinstance(exp_config.robot_config, RBY1Config):
        from molmo_spaces.env.rby1_sensors import RBY1GraspStateSensor

        sensors.append(RBY1GraspStateSensor(uuid="rby1_left_grasp_state", arm_side="left"))
        sensors.append(RBY1GraspStateSensor(uuid="rby1_right_grasp_state", arm_side="right"))

    sensors.append(TaskInfoSensor(uuid="task_info"))

    sensors.append(GraspPoseSensor(uuid="grasp_pose"))

    # Policy sensors
    sensors.append(PolicyPhaseSensor(uuid="policy_phase"))
    sensors.append(PolicyNumRetriesSensor(uuid="policy_num_retries"))

    # Action sensors
    sensors.append(
        LastActionSensor(
            dtype=exp_config.task_config.action_dtype,
        )
    )
    sensors.append(LastCommandedJointPosSensor())
    sensors.append(LastCommandedRelativeJointPosSensor())
    sensors.append(LastCommandedEETwistSensor())
    sensors.append(LastCommandedEEPoseSensor())

    # Object tracking sensors
    sensors.append(ObjectImagePointsSensor(exp_config=exp_config))

    # Legacy sensors for debugging
    sensors.append(RobotStateSensor(uuid="robot_state"))
    sensors.append(
        ObjectPoseSensor(
            object_names=exp_config.task_config.tracked_object_names
            or [exp_config.task_config.pickup_obj_name, exp_config.task_config.place_target_name],
            uuid="object_poses",
        )
    )

    return sensors


def get_nav_task_sensors(exp_config):
    """Get sensors for navigation to object task.

    Args:
        exp_config: Experiment configuration object with sensor parameters

    Returns:
        List of initialized sensors
    """
    sensors = []

    # Get camera names dynamically from camera config instead of hardcoded list
    for camera_spec in exp_config.camera_config.cameras:
        camera_name = camera_spec.name

        # Camera parameter sensor
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
    # Robot State sensors
    sensors.append(RobotBasePoseSensor(uuid="robot_base_pose"))
    sensors.append(RobotJointPositionSensor(uuid="qpos", max_joints=25))

    # Environment state sensors
    sensors.append(EnvStateSensor(uuid="env_states"))

    # Task pose sensors - determine pickup object names
    if (
        hasattr(exp_config.task_config, "pickup_obj_candidates")
        and exp_config.task_config.pickup_obj_candidates is not None
        and len(exp_config.task_config.pickup_obj_candidates) > 0
    ):
        pickup_obj_names = exp_config.task_config.pickup_obj_candidates
    elif exp_config.task_config.pickup_obj_name:
        pickup_obj_names = [exp_config.task_config.pickup_obj_name]
    else:
        pickup_obj_names = []

    if pickup_obj_names:  # Only add sensor if there are objects to track
        sensors.append(ObjectPoseSensor(object_names=pickup_obj_names, uuid="pickup_obj_pose"))

    # Action sensors
    sensors.append(
        LastActionSensor(
            dtype=exp_config.task_config.action_dtype,
        )
    )

    return sensors
