from functools import cached_property
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.controllers.base_pose import DiffDriveBasePoseController
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.controllers.torso_height import TorsoHeightJointPosController
from molmo_spaces.kinematics.rby1_kinematics import RBY1Kinematics
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView


class RBY1(Robot):
    """RBY1 Robot class for the RBY1 robot.
    This class extends the Robot class and provides specific implementations for the RBY1 robot
    including the robot view, controllers, and a kinematic solver.

    The mode in which the robot arm, gripper, or base is commanded (Eg. ee position command or joint
    position command) can be set using the `arm_command_mode`, `gripper_command_mode`, and `base_command_mode`
    parameters in the robot configuration.

    The move_group i.e., the set of robot parts that are to be moved can be dynamically changed,
    e.g., move base for navigation, then move both base and arm for opening doors.

    """

    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        """
        Args:
            exp_config: Experiment configuration params
            mj_data: MuJoCo data structure containing the current simulation state
        """
        super().__init__(mj_data, exp_config)

        self._namespace = self.exp_config.robot_config.robot_namespace
        self._use_holo_base = self.exp_config.robot_config.use_holo_base

        # Create the robot view:
        self._robot_view = RBY1RobotView(mj_data, self.namespace, holo_base=self._use_holo_base)

        # Create kinematic solver:
        self._kinematics = RBY1Kinematics(
            self.mj_model, namespace=self.namespace, holo_base=self._use_holo_base
        )

        # Create controllers:

        # All the joint actuators are position controlled.
        # However, they can be commanded in other modes like velocity, ee position, etc. and the controller will
        # perform the necessary conversions to joint position.
        self.arm_command_modes = [
            "joint_position",
            "joint_rel_position",
            "joint_velocity",
            "ee_position",
            "ee_velocity",
        ]
        if self.exp_config.robot_config.command_mode["arm"] is not None:
            assert self.exp_config.robot_config.command_mode["arm"] in self.arm_command_modes, (
                f"Arm command mode {self.exp_config.robot_config.command_mode['arm']} not in {self.arm_command_modes}"
            )
        self.arm_command_mode = (
            "joint_position"
            if not self.exp_config.robot_config.command_mode["arm"]
            else self.exp_config.robot_config.command_mode["arm"]
        )
        if self.arm_command_mode == "joint_rel_position":
            left_arm_controller = JointRelPosController(self.robot_view.get_move_group("left_arm"))
            right_arm_controller = JointRelPosController(
                self.robot_view.get_move_group("right_arm")
            )
        elif self.arm_command_mode == "joint_position":
            left_arm_controller = JointPosController(self.robot_view.get_move_group("left_arm"))
            right_arm_controller = JointPosController(self.robot_view.get_move_group("right_arm"))
        else:
            raise NotImplementedError(
                f"Arm command mode {self.arm_command_mode} not implemented yet."
            )

        # Gripper command modes - separate from arm command modes
        self.gripper_command_modes = [
            "joint_position",
            "joint_rel_position",
            "joint_velocity",
        ]
        if self.exp_config.robot_config.command_mode["gripper"] is not None:
            assert (
                self.exp_config.robot_config.command_mode["gripper"] in self.gripper_command_modes
            ), (
                f"Gripper command mode {self.exp_config.robot_config.command_mode['gripper']} not in {self.gripper_command_modes}"
            )
        self.gripper_command_mode = (
            "joint_position"
            if not self.exp_config.robot_config.command_mode["gripper"]
            else self.exp_config.robot_config.command_mode["gripper"]
        )
        if self.gripper_command_mode == "joint_rel_position":
            left_gripper_controller = JointRelPosController(
                self.robot_view.get_move_group("left_gripper")
            )
            right_gripper_controller = JointRelPosController(
                self.robot_view.get_move_group("right_gripper")
            )
        elif self.gripper_command_mode == "joint_position":
            left_gripper_controller = JointPosController(
                self.robot_view.get_move_group("left_gripper")
            )
            right_gripper_controller = JointPosController(
                self.robot_view.get_move_group("right_gripper")
            )
        else:
            raise NotImplementedError(
                f"Gripper command mode {self.gripper_command_mode} not implemented yet."
            )

        # The base actuators (wheels) are usually velocity controlled. If using virtual holonomic joints, they are position controlled.
        # Base actuators can still be commanded in other modes like planar position, planar velocity, etc.
        # and the controller should perform the necessary conversions to wheel velocity or holo joint position.
        self.base_command_modes = [
            "planar_position",
            "planar_velocity",
            "wheel_velocity",
            "holo_joint_planar_position",
            "holo_joint_rel_planar_position",
        ]
        if self.exp_config.robot_config.command_mode["base"] is not None:
            assert self.exp_config.robot_config.command_mode["base"] in self.base_command_modes, (
                f"Base command mode {self.exp_config.robot_config.command_mode['base']} not in {self.base_command_modes}"
            )
        self.base_command_mode = (
            "planar_position"
            if not self.exp_config.robot_config.command_mode["base"]
            else self.exp_config.robot_config.command_mode["base"]
        )
        if self.base_command_mode == "planar_position":
            base_controller = DiffDriveBasePoseController(
                self.exp_config.robot_config, self.robot_view.get_move_group("base")
            )
        elif self.base_command_mode == "holo_joint_rel_planar_position":
            base_controller = JointRelPosController(self.robot_view.get_move_group("base"))
        elif self.base_command_mode == "holo_joint_planar_position":
            base_controller = JointPosController(self.robot_view.get_move_group("base"))
        else:
            raise NotImplementedError(
                f"Base command mode {self.base_command_mode} not implemented yet."
            )

        # Head is fixed - no head actions are supported
        self.head_command_mode = self.exp_config.robot_config.command_mode.get("head")
        assert self.head_command_mode is None, (
            "RBY1 head actuation is disabled. The head is fixed at init_qpos['head'] with optional "
            "randomization via init_qpos_noise_range['head']. "
            "Do not set command_mode['head'] to a non-None value."
        )

        # Torso command modes
        self.torso_command_modes = ["joint_position", "height"]
        self.torso_command_mode = (
            self.exp_config.robot_config.command_mode.get("torso", "joint_position")
            or "joint_position"
        )
        assert self.torso_command_mode in self.torso_command_modes, (
            f"Torso command mode {self.torso_command_mode} not in {self.torso_command_modes}"
        )
        if self.torso_command_mode == "height":
            torso_controller = TorsoHeightJointPosController(
                self.robot_view.get_move_group("torso")
            )
        else:
            torso_controller = JointPosController(self.robot_view.get_move_group("torso"))

        self._controllers = {
            "base": base_controller,
            "torso": torso_controller,
            "left_arm": left_arm_controller,
            "right_arm": right_arm_controller,
            "left_gripper": left_gripper_controller,
            "right_gripper": right_gripper_controller,
        }
        assert set(self._controllers.keys()).issubset(set(self._robot_view.move_group_ids())), (
            "All controller keys must be move group IDs"
        )

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("Parallel kinematics not implemented for RBY1")

    @cached_property
    def state_dim(self) -> int:
        # return sum of all the joints of interest
        self._state_dim = 0
        for move_group in self.robot_view.move_group_ids():
            if move_group == "base":
                self._state_dim += 3  # Using only 2D base position (x, y, theta)
            else:
                self._state_dim += self.robot_view.get_move_group(move_group).n_joints
        return self._state_dim

    def action_dim(self, move_group_ids: list) -> int:
        # return sum of the commanded joints based on the move group ids
        action_dim = 0
        if "base" in move_group_ids:
            if "planar" in self.base_command_mode:
                action_dim += 3  # actions provided are 2D planar positions / velocities
            else:
                action_dim += self._robot_view.get_move_group(
                    "base"
                ).n_actuators  # wheel velocities
        if "torso" in move_group_ids:
            if self.torso_command_mode == "height":
                action_dim += 1
            else:
                action_dim += self._robot_view.get_move_group("torso").n_actuators
        if "left_arm" in move_group_ids:
            if "joint" in self.arm_command_mode:
                action_dim += self._robot_view.get_move_group("left_arm").n_actuators
            elif "ee" in self.arm_command_mode:
                action_dim += 7  # ee pos + orientation (quaternion)
        if "right_arm" in move_group_ids:
            if "joint" in self.arm_command_mode:
                action_dim += self._robot_view.get_move_group("right_arm").n_actuators
            elif "ee" in self.arm_command_mode:
                action_dim += 7  # ee pos + orientation (quaternion)
        if "left_gripper" in move_group_ids:
            action_dim += self._robot_view.get_move_group("left_gripper").n_actuators
        if "right_gripper" in move_group_ids:
            action_dim += self._robot_view.get_move_group("right_gripper").n_actuators
        # Note: head is not included - RBY1 head actuation is disabled
        return action_dim

    def get_arm_move_group_ids(self) -> list[str]:
        """RBY1 has two independent arms - each gets independent noise."""
        return ["left_arm", "right_arm"]

    def _apply_base_noise(
        self,
        commanded_base_pos: np.ndarray,
    ) -> np.ndarray:
        """Apply planar noise to base commands (x, y, theta).

        The noise model:
        1. Computes displacement from current base pose
        2. Scales noise proportionally to displacement magnitude
        3. Samples noise in planar space (bounded by config)

        Args:
            commanded_base_pos: The commanded base position [x, y, theta]

        Returns:
            Noisy base position [x, y, theta]
        """
        noise_config = self.exp_config.robot_config.action_noise_config

        # Get current base position (x, y, theta)
        base_mg = self.robot_view.get_move_group("base")
        current_base_pos = base_mg.joint_pos  # [x, y, theta] for holo base

        # Compute displacement
        delta = commanded_base_pos - current_base_pos
        position_delta = delta[:2]  # x, y
        rotation_delta = delta[2] if len(delta) > 2 else 0.0  # theta

        position_delta_norm = np.linalg.norm(position_delta)
        rotation_delta_abs = abs(rotation_delta)

        # Compute noise scale proportional to action magnitude
        # When delta is zero, noise is zero
        scale_factor = noise_config.base_action_scale_factor

        position_noise_std = scale_factor * position_delta_norm
        rotation_noise_std = scale_factor * rotation_delta_abs

        # Sample noise
        position_noise = np.random.randn(2) * position_noise_std
        rotation_noise = np.random.randn() * rotation_noise_std

        # Clip to maximum bounds
        position_noise = np.clip(
            position_noise,
            -noise_config.max_base_position_noise,
            noise_config.max_base_position_noise,
        )
        rotation_noise = np.clip(
            rotation_noise,
            -noise_config.max_base_rotation_noise,
            noise_config.max_base_rotation_noise,
        )

        # Apply noise
        noisy_base_pos = commanded_base_pos.copy()
        noisy_base_pos[:2] += position_noise
        if len(noisy_base_pos) > 2:
            noisy_base_pos[2] += rotation_noise

        return noisy_base_pos

    def apply_action_noise(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply action noise to the commanded action.

        Extends the base class implementation to also apply base noise
        for RBY1's holonomic base commands.

        Args:
            action: Action dict with move_group_id -> joint positions

        Returns:
            Modified action dict with noise added
        """
        noise_config = self.exp_config.robot_config.action_noise_config
        if not noise_config.enabled:
            return action

        # Apply arm noise via parent class
        noisy_action = super().apply_action_noise(action)

        # Apply base noise if base command is present and using holonomic planar mode
        if "base" in action and action["base"] is not None:
            if "holo_joint" in self.base_command_mode and "planar" in self.base_command_mode:
                commanded_base_pos = np.asarray(action["base"])
                noisy_base_pos = self._apply_base_noise(commanded_base_pos)
                noisy_action["base"] = noisy_base_pos

        return noisy_action

    def update_control(self, action_command_dict: dict[str, Any]) -> None:
        """Update the control inputs to the robot based on the provided action commands.
        Args:
        action_command_dict: Dictionary containing action commands for the robot
                             based on the move groups ids to be used.
        """
        # Loops through all the controllers and updates their control inputs
        # NOTE: All joints are always controlled. If no action_command is provided, we assume the
        # controller will maintain the current state of the joint.

        action_command_dict = self._apply_action_noise_and_save_unnoised_cmd_jp(action_command_dict)

        for move_group_id, controller in self.controllers.items():
            if move_group_id in action_command_dict:
                action_command = action_command_dict[move_group_id]
                # send command target for the controller
                controller.set_target(action_command)
            else:
                # If no action command is provided, controller should switch to
                # stationary mode (if not already set to stationary previously)
                if not controller.stationary:
                    controller.set_to_stationary()

    def compute_control(self) -> None:
        for mg_id, controller in self.controllers.items():
            ctrl_inputs = controller.compute_ctrl_inputs()
            self.robot_view.get_move_group(mg_id).ctrl = ctrl_inputs

    def set_joint_pos(self, robot_joint_pos_dict) -> None:
        """Set all the robot's joint positions to the specified values.
        Args:
            robot_joint_pos_dict: Dictionary or SimpleNamespace containing joint positions for the robot
            based on the move groups ids.
        """
        # Handle both dict and SimpleNamespace objects
        if hasattr(robot_joint_pos_dict, "__dict__"):
            # SimpleNamespace object
            items = robot_joint_pos_dict.__dict__.items()
        else:
            # Dictionary object
            items = robot_joint_pos_dict.items()

        for move_group_id, joint_pos in items:
            if move_group_id in self.robot_view.move_group_ids():
                move_group = self.robot_view.get_move_group(move_group_id)
                # set the joint positions
                move_group.joint_pos = joint_pos
            else:
                raise ValueError(f"Move group {move_group_id} not found in robot view.")

    def get_world_pose_tf_mat(self):
        """Get the robot's world pose transformation matrix.
        Returns:
            np.ndarray: 4x4 transformation matrix for the robot base pose in world frame
        """
        return self.robot_view.get_move_group("base").pose

    def set_world_pose(self, robot_world_pose) -> None:
        """Set the robot's world pose to the specified location in the world."""

        pose_tf = np.eye(4, dtype=np.float64)
        if len(robot_world_pose) == 3:
            # (x, y, theta)
            x, y, theta = robot_world_pose
            pose_tf[:2, :2] = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )
            pose_tf[:3, 3] = np.array([x, y, 0.0])
        elif len(robot_world_pose) == 7:
            # (x, y, z, w, x, y, z) (pos + quat, scalar_first)
            pos = np.array(robot_world_pose[:3], dtype=np.float64)
            quat = np.array(robot_world_pose[3:], dtype=np.float64)
            rot_mat = R.from_quat(quat, scalar_first=True).as_matrix()
            pose_tf[:3, :3] = rot_mat
            pose_tf[:3, 3] = pos
        else:
            raise ValueError(
                "robot_world_pose must be either (x, y, theta) or (x, y, z, w, x, y, z) (pos + quat, scalar_first)."
            )

        # The robot_view expects a transformation matrix in the form of a 4x4 numpy array
        self.robot_view.get_move_group("base").pose = pose_tf

    def reset(self, robot_joint_pos_dict=None, robot_world_pose=None) -> None:
        """Reset the robot to its initial position or a provided set of positions and world pose."""

        # Load default robot configuration and world pose
        init_qpos_dict = self.exp_config.robot_config.init_qpos
        default_world_pose = self.exp_config.robot_config.default_world_pose
        if robot_joint_pos_dict is not None:
            for move_group_id, joint_pos in robot_joint_pos_dict.items():
                init_qpos_dict[move_group_id] = joint_pos
        if robot_world_pose is not None:
            default_world_pose = robot_world_pose
        # Set the joint positions
        self.set_joint_pos(init_qpos_dict)
        # Set the world pose
        self.set_world_pose(default_world_pose)

        # reset controllers
        for _, controller in self._controllers.items():
            controller.reset()

        # Set the head ctrl to match its qpos to prevent jerking at trajectory start.
        # The head has actuators but no controller, so we manually set ctrl = noop_ctrl.
        head_mg = self.robot_view.get_move_group("head")
        head_mg.ctrl = head_mg.noop_ctrl

    def get_joint_position(self, move_group_ids: list[str]) -> np.ndarray:
        """Get the current joint positions of the move groups"""
        return np.concatenate(
            [
                self._robot_view.get_move_group(move_group_id).joint_pos.copy()
                for move_group_id in move_group_ids
            ]
        )

    def get_joint_ranges(self, move_group_ids: list[str]):
        """Get the joint ranges of the move groups"""
        joint_ranges = {}
        count = 0
        for move_group_id in move_group_ids:
            joint_ranges[move_group_id] = (
                count,
                count + self._robot_view.get_move_group(move_group_id).n_joints,
            )
            count += self._robot_view.get_move_group(move_group_id).n_joints
        return joint_ranges

    def _read_from_sensor(self, sensor_name: str, data: MjData) -> None:
        # TODO
        pass
        # s_adr = self.model.sensor(self.namespace + sensor_name).adr.item()
        # s_dim = self.model.sensor(self.namespace + sensor_name).dim.item()
        # return data.sensordata[s_adr : s_adr + s_dim].copy()

    def check_collision(self, model: MjModel, data: MjData, grasped_objs: set[int] = None) -> bool:
        # TODO
        pass
        # # TODO: ignore finger collision with grasped objects
        # # check if the agent is in collision with any geoms in the scene
        # agent_id = self.root_id
        # contacts = data.contact
        # for c in contacts:
        #     if c.exclude != 0:
        #         continue
        #     body1 = self.model.geom_bodyid[c.geom1]
        #     body2 = self.model.geom_bodyid[c.geom2]

        #     rootbody1 = self.model.body_rootid[body1]
        #     rootbody2 = self.model.body_rootid[body2]

        #     if rootbody1 == agent_id or rootbody2 == agent_id:
        #         # Ignore Collision with gripper tips and target object
        #         if (
        #             self.model.body(body1).name == self.left_gripper_geom_name
        #             or self.model.body(body1).name == self.right_gripper_geom_name
        #         ) or (
        #             self.model.body(body2).name == self.left_gripper_geom_name
        #             or self.model.body(body2).name == self.right_gripper_geom_name
        #         ):
        #             if grasped_objs is not None and grasped_objs:
        #                 if (
        #                     self.model.body(body1).id in grasped_objs
        #                     or self.model.body(body2).id in grasped_objs
        #                 ):
        #                     continue
        #             else:
        #                 continue

        #             # Ignore Collision with itself
        #             if self.model.body(rootbody1).name == self.model.body(rootbody2).name:
        #                 continue

        #         # Collision with the floor
        #         if (
        #             self.model.body(rootbody2).name == "floor"
        #             or self.model.body(rootbody1).name == "floor"
        #         ):
        #             other_body = body1 if self.model.body(rootbody2).name == "floor" else body2
        #             if self.model.body(other_body).name in [
        #                 self.namespace + "fl_wheel_link",
        #                 self.namespace + "fr_wheel_link",
        #                 self.namespace + "rl_wheel_link",
        #                 self.namespace + "rr_wheel_link",
        #             ]:
        #                 # print("Collision with floor detected", model.body(body1).name, model.body(body2).name)
        #                 continue
        #             # print(
        #             #    "Collision with floor detected",
        #             #    self.model.body(body1).name,
        #             #    self.model.body(body2).name,
        #             # )
        #             # return False

        #         print(
        #             "Collision detected",
        #             self.model.body(body1).name,
        #             self.model.body(body2).name,
        #         )
        #         return True
        # return False

    def get_grasped_objs(self, model: MjModel, data: MjData) -> set[int]:
        """Return a set of grasped object body IDs."""
        # TODO
        pass
        # root_id = model.body_rootid[next(iter(self.finger_body_ids))]
        # grasped_objs = set()
        # for c in data.contact:
        #     if c.exclude != 0:
        #         continue
        #     b1 = model.geom_bodyid[c.geom1]
        #     b2 = model.geom_bodyid[c.geom2]

        #     # ignore all collisions not involving only one finger
        #     if not ((b1 in self.finger_body_ids) ^ (b2 in self.finger_body_ids)):
        #         continue
        #     # ignore self-collisions
        #     if model.body_rootid[b1] == root_id and model.body_rootid[b2] == root_id:
        #         continue

        #     if b1 in self.finger_body_ids:
        #         grasped_objs.add(b2)
        #     else:
        #         grasped_objs.add(b1)
        # grasped_objs = {model.body_rootid[obj_id] for obj_id in grasped_objs}

        # return grasped_objs

    @staticmethod
    def robot_model_root_name() -> str:
        return "base"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "MlSpacesExpConfig.RobotConfig",
        spec: MjSpec,
        robot_spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
    ) -> None:
        super().add_robot_to_scene(
            robot_config, spec, robot_spec, prefix, pos, quat, randomize_textures
        )

        def add_slider_act(
            name: str, ctrlrange: float, gainprm: float, biasprm: list[float], gear_idx: int
        ):
            act = spec.add_actuator()
            act.name = f"{prefix}{name}"
            act.target = f"{prefix}base_site"
            act.refsite = f"{prefix}world"
            act.ctrlrange = np.array([-ctrlrange, ctrlrange])
            act.gainprm[0] = gainprm
            act.biasprm[: len(biasprm)] = biasprm
            act.trntype = mujoco.mjtTrn.mjTRN_SITE
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gear = [0] * 6
            gear[gear_idx] = 1
            act.gear = gear
            return act

        if robot_config.use_holo_base:
            spec.worldbody.add_site(name=f"{prefix}world", pos=[0, 0, 0.005], quat=[1, 0, 0, 0])
            add_slider_act("base_x_act", 25, 25000, [0, -25000, 0.5], 0)
            add_slider_act("base_y_act", 25, 25000, [0, -25000, 0.5], 1)
            add_slider_act("base_theta_act", np.pi, 5000, [0, -5000, 0.5], 5)

        # TODO(snehal): don't use bodies in the MJCF, just use visual geoms to render these
        # add target ee pose bodies
        ee_viz_right = spec.worldbody.add_body(
            name="target_ee_pose_right", pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap=True
        )
        ee_viz_right.add_site(
            name="target_ee_pose_right",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.05, 0.05],
            rgba=[0, 0, 1, 0.3],
            group=1,
        )

        ee_viz_left = spec.worldbody.add_body(
            name="target_ee_pose_left", pos=[0, 0, 0], quat=[1, 0, 0, 0], mocap=True
        )
        ee_viz_left.add_site(
            name="target_ee_pose_left",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.05, 0.05],
            rgba=[1, 0, 0, 0.3],
            group=1,
        )
