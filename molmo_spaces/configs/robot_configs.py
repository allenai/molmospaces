"""Robot configuration classes for MolmoSpaces experiments.

This module contains:
- ActionNoiseConfig: TCP-bounded noise configuration for arm actions
- BaseRobotConfig: Base configuration for all robots
- Robot-specific configs: FrankaRobotConfig, RBY1Config, FloatingRUMRobotConfig
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.task_sampler_configs import HumanRBVariant
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.bimanual_yam import BimanualYamRobot
from molmo_spaces.robots.floating_robotiq import FloatingRobotiqRobot
from molmo_spaces.robots.floating_rum import FloatingRUMRobot
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.g1 import G1Robot
from molmo_spaces.robots.g1_dex3 import G1Dex3Robot
from molmo_spaces.robots.human_rb import (
    HumanRBRobot,
    human_rb_library_dir,
    human_rb_move_group_dofs,
)
from molmo_spaces.robots.i2rt_yam import I2rtYamRobot
from molmo_spaces.robots.mobile_franka import MobileFrankaRobot
from molmo_spaces.robots.rby1 import RBY1
from molmo_spaces.robots.robot_views.abstract import RobotViewFactory
from molmo_spaces.robots.robot_views.bimanual_yam_view import BimanualYamRobotView
from molmo_spaces.robots.robot_views.franka_cap_view import (
    FrankaCAPRobotView,
)
from molmo_spaces.robots.robot_views.franka_droid_view import (
    FloatingRobotiq2f85RobotView,
    FrankaDroidRobotView,
)
from molmo_spaces.robots.robot_views.g1_dex3_view import G1Dex3RobotView
from molmo_spaces.robots.robot_views.g1_view import G1RobotView
from molmo_spaces.robots.robot_views.human_rb_view import HumanRBRobotView
from molmo_spaces.robots.robot_views.i2rt_yam_view import I2rtYamRobotView
from molmo_spaces.robots.robot_views.mobile_franka_droid_view import MobileFrankaDroidRobotView
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView
from molmo_spaces.robots.robot_views.rum_gripper_view import FloatingRUMRobotView


class ActionNoiseConfig(Config):
    """Configuration for action noise injection.

    This noise model supports:
    - Arm noise: TCP-bounded noise that maps through Jacobian to joint space
    - Base noise: Planar noise applied directly to (x, y, theta) commands

    Noise is proportional to the commanded action magnitude:
        noise_std = action_scale_factor * ||delta||

    When the commanded delta is zero, no noise is applied.
    """

    enabled: bool = True  # Whether to apply action noise

    # === Arm noise configuration (TCP-bounded) ===

    # Scale factor for arm noise proportional to TCP delta magnitude
    # noise_std = action_scale_factor * ||tcp_delta||
    # e.g., action_scale_factor=0.1 means noise std is 10% of commanded TCP delta
    action_scale_factor: float = 0.1

    # Rotation noise scale relative to position noise
    rotation_noise_scale: float = 0.1

    # Maximum noise magnitude in TCP space (clipped to this bound)
    max_tcp_position_noise: float = 0.02  # 2cm max position noise
    max_tcp_rotation_noise: float = 0.1  # ~5.7 degrees max rotation noise

    # === Base noise configuration (planar) ===

    # Scale factor for base noise proportional to commanded displacement magnitude
    # position_noise_std = base_action_scale_factor * ||position_delta||
    # rotation_noise_std = base_action_scale_factor * |rotation_delta|
    base_action_scale_factor: float = 0.1

    # Maximum base noise magnitude (clipped to this bound)
    max_base_position_noise: float = 0.02  # 2cm max
    max_base_rotation_noise: float = 0.05  # ~2.8 degrees max


class BaseRobotConfig(Config):
    """Base configuration for robot setup."""

    robot_cls: type[Robot] | None

    robot_factory: Callable[[mj.MjData, Any], Robot] | None
    # (MjData, MlSpacesExpConfig) -> Robot. here (and subclasses) we use Any to avoid annotation dependency on MlSpacesExpConfig

    robot_view_factory: RobotViewFactory | None

    robot_namespace: str
    """Namespace used to differentiate between one or multiple robots and the environment"""

    command_mode: dict[str, str | None]
    """move_group to command_mode e.g., 'joint', 'cartesian', 'velocity'"""

    init_qpos: dict[str, list[float]]
    init_qpos_noise_range: dict[str, list[float]] | None
    name: str | None
    robot_xml_path: Path  # path to the robot XML file within the robot directory
    robot_dir: Path | None = (
        None  # path to the robot directory, if not using a prepackaged MlSpaces robot
    )

    # configurable control parameters for low-level mujoco controllers
    gravcomp: bool = False  # apply gravity compensation to every body in the robot
    K_stiffness: list[float] | None = None  # if None use values from model
    K_damping: list[float] | None = None  # if None use values from model
    force_limit: list[float] | None = (
        None  # Limit actuator-applied generalized force magnitude, if None use values from model
    )

    # Action noise configuration - applied per-robot in Robot.apply_action_noise()
    action_noise_config: ActionNoiseConfig | None = None

    # If set, task samplers should place the robot's base at this world-frame z
    # height rather than deriving spawn height from the target object's height
    # (target_z + robot_object_z_offset +/- noise, see PickTaskSampler). That
    # target-relative placement assumes an adjustable base height (RBY1's torso
    # lift, FloatingRUM's freely-positioned floating base) and produces an
    # unnatural spawn height for a robot whose base height is fixed by its own
    # controller (e.g. LegsWaistController's WBC holds a constant standing height
    # regardless of where the robot is placed) -- planning code that reads the
    # robot's pose at task reset (before physics has run) would then see a
    # spawn height physics is about to correct away, silently invalidating any
    # grasp/reach poses computed from it.
    fixed_base_height: float | None = None

    # If set, overrides the scene's compiled model.opt.timestep (normally
    # 0.002s, set in molmo_spaces/resources/base_scene.xml) once the robot is
    # constructed -- see Robot subclasses' __init__ (e.g. G1Robot's, which
    # sets this for G1). None (default) leaves the scene's own timestep
    # untouched. Only override this for a robot whose controller was
    # trained/tuned at a specific physics rate that differs from our own
    # scene default (see G1Config's own physics_timestep for why G1 needs
    # this) -- mutating model.opt.timestep for a robot whose controllers
    # don't care is pure risk with no benefit.
    physics_timestep: float | None = None

    def model_post_init(self, _context):
        """Ensure action_noise_config is always initialized, even when loading from old configs."""
        if self.action_noise_config is None:
            object.__setattr__(self, "action_noise_config", ActionNoiseConfig())

    def get_robot_dir(self) -> Path:
        """
        Get the path to the robot directory, which may or may not be a prepackaged MlSpaces robot.
        """
        if self.robot_dir is not None:
            return self.robot_dir
        return get_robot_path(self.name)

    def get_robot_xml_path(self) -> Path:
        """
        Get the full path to the robot XML file.
        """
        return self.get_robot_dir() / self.robot_xml_path


# Concrete robot configurations


class FrankaRobotConfig(BaseRobotConfig):
    """Configuration for Franka FR3 robot."""

    robot_cls: type[Robot] | None = FrankaRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = FrankaRobot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = FrankaDroidRobotView
    name: str | None = "franka_droid"
    robot_xml_path: Path = Path("model.xml")
    base_size: list[float] | None = [0.5, 0.5, 0.58]
    init_qpos: dict[str, list[float]] = {
        "arm": [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        # selected to allow for more displacement in later joints and keep TCP displacement <=10cm
        # joint_weights = [1, ..., 7] (allow more movement in later joints)
        # J_p is 3x7 Jacobian of TCP position wrt arm joints
        # dq = joint_weights * 0.1 / ||J_p @ joint_weights||
        "arm": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
    }
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",  # e.g., "joint_position", "joint_velocity", "ee_position", "ee_velocity"
        "gripper": "joint_position",
    }
    gravcomp: bool = True
    # texture randomization parameters, ignored if texture randomization is disabled
    perturb_texture_probability: float = 0.7

    def model_post_init(self, _context: Any):
        super().model_post_init(_context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class MobileFrankaRobotConfig(BaseRobotConfig):
    robot_cls: type[Robot] | None = MobileFrankaRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = MobileFrankaRobot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = MobileFrankaDroidRobotView
    name: str | None = "franka_droid"
    robot_xml_path: Path = Path("model.xml")
    base_size: list[float] = [0.5, 0.5, 0.58]
    init_qpos: dict[str, list[float]] = {
        "base": [0, 0, 0],
        "arm": [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.0],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        "arm": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
    }
    command_mode: dict[str, str | None] = {
        "base": "holo_joint_planar_position",
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    base_control_params: dict[str, dict[str, float]] = {
        "base_x_act": {
            "kp": 25000,
            "damping_ratio": 1.0,
            "ctrlrange": 25,
        },
        "base_y_act": {
            "kp": 25000,
            "damping_ratio": 1.0,
            "ctrlrange": 25,
        },
        "base_theta_act": {
            "kp": 5000,
            "damping_ratio": 1.0,
        },
    }


class FrankaCAPRobotConfig(BaseRobotConfig):
    """Configuration for Franka FR3 robot."""

    robot_cls: type[Robot] | None = FrankaRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = FrankaRobot
    robot_namespace: str = "robot_0/"
    robot_view_factory: RobotViewFactory | None = FrankaCAPRobotView
    name: str | None = "franka_cap"
    robot_xml_path: Path = Path("model.xml")
    base_size: list[float] | None = [0.5, 0.5, 0.58]
    init_qpos: dict[str, list[float]] = {
        "arm": [0, -1.5, 0.116, -2.45, 0, 0.842, 0.965],
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {
        # selected to allow for more displacement in later joints and keep TCP displacement <=10cm
        # joint_weights = [1, ..., 7] (allow more movement in later joints)
        # J_p is 3x7 Jacobian of TCP position wrt arm joints
        # dq = joint_weights * 0.1 / ||J_p @ joint_weights||
        "arm": [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
    }
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",  # e.g., "joint_position", "joint_velocity", "ee_position", "ee_velocity"
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, _context: Any):
        super().model_post_init(_context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class RBY1Config(BaseRobotConfig):
    """Configuration for RBY1 robot."""

    robot_cls: type[Robot] | None = RBY1
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = RBY1
    robot_view_factory: RobotViewFactory | None = None  # set in model_post_init
    robot_namespace: str = "robot_0/"
    init_qpos: dict[str, list[float]] = {
        "base": [0.0, 0.0, 0.0],  # x, y, theta
        "head": [
            0.0,
            0.6,
        ],  # (pan, tilt) - 0 pan = forward, ~0.4 rad tilt = looking down ~34 degrees
        "left_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
        "left_gripper": [-0.05],  # Open position - coupling handled in RBY1GripperGroup
        "right_arm": [0.5, 0.0, 0.0, -2.3, 0.0, -0.5, 0.0],
        "right_gripper": [-0.05],  # Open position - coupling handled in RBY1GripperGroup
        "torso": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    # TODO: Add noise ranges for arms etc
    init_qpos_noise_range: dict[str, list[float]] | None = {
        "base": [0.0, 0.0, 0.0],
        # "head": [0.15, 0.1],  # (pan, tilt) noise in radians (~8.5 deg, ~5.7 deg)
        "head": [0.2, 0.2],  # (pan, tilt) noise in radians (~11.4 deg, ~11.4 deg)
        "left_arm": [
            0.05,
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.175,
        ],  # Graduated noise: more distal = more variation
        "left_gripper": [0.01],
        "right_arm": [
            0.05,
            0.05,
            0.075,
            0.1,
            0.125,
            0.15,
            0.175,
        ],  # Graduated noise: more distal = more variation
        "right_gripper": [0.01],
        "torso": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    use_holo_base: bool = True  # Whether to use virtual holonomic base joints or not
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",  # e.g., "joint_position", "joint_velocity", "ee_position", "ee_velocity"
        "gripper": "joint_position",
        "base": "holo_joint_planar_position",  # e.g., "planar_position", "planar_velocity", "wheel_velocity"
        "head": None,  # Must be None - RBY1 head actuation is disabled
    }
    name: str | None = "rby1"
    robot_xml_path: Path = Path("rby1_site_control.xml")
    gravcomp: bool = True

    def model_post_init(self, _context):
        super().model_post_init(_context)
        self.robot_view_factory = partial(RBY1RobotView, holo_base=self.use_holo_base)


class RBY1MConfig(RBY1Config):
    """Configuration for RBY1M i.e. mecanum wheel robot."""

    use_holo_base: bool = True  # Whether to use virtual holonomic base joints or not
    name: str | None = "rby1m"
    robot_xml_path: Path = Path("rby1_v1.2_site_control.xml")
    # NOTE: No wheel control for now so we can re-use this config for both the robot types


class RBY1MOpenCloseConfig(RBY1MConfig):
    """RBY1M config for open/close tasks.

    Uses single-scalar torso height control (torso_1 = torso_3 = h, torso_2 = -2*h)
    instead of commanding all 6 torso joints independently.
    """

    command_mode: dict[str, str | None] = {
        "arm": "joint_rel_position",
        "gripper": "joint_position",
        "base": "holo_joint_rel_planar_position",
        "head": None,
        "torso": "height",
    }


class G1Config(BaseRobotConfig):
    """Configuration for the Unitree G1 humanoid robot.

    One base control mode:
    - Whole-body walking: the combined `legs_waist` move group is
      driven by the WBC (a PD-torque law plus an ONNX walking policy --
      see `molmo_spaces.controllers.g1_wbc`), commanded via
      `set_target([vx, vy, yaw_rate, height, waist_yaw, waist_roll,
      waist_pitch])`. The base has no actuators (free-floating pelvis
      integrated directly by MuJoCo's physics).

    Both arms and the right gripper always stay on plain JointPosControllers,
    relying on the MJCF's own tuned PD actuator gains.
    """

    robot_cls: type[G1Robot] | None = G1Robot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = G1Robot.from_mj_data
    robot_view_factory: RobotViewFactory | None = G1RobotView
    robot_namespace: str = "robot_0/"
    name: str = "g1"
    robot_xml_path: Path = Path("g1_dex.xml")
    # Default standing pose, taken from the source G1 stack's validated
    # gravity-settled/nominal joint values. Note this is the *reset* pose, a
    # different (more upright) pose than LegsWaistController's own internal
    # `_DEFAULT_POSE` action-space reference offset -- both exist in the
    # source stack too, for the same reason (env reset vs. policy reference).
    init_qpos: dict[str, np.ndarray] = {
        "legs_waist": np.array(
            [
                -0.312,
                0.0,
                0.0,
                0.669,
                -0.363,
                0.0,
                -0.312,
                0.0,
                0.0,
                0.669,
                -0.363,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        "left_arm": np.array([0.212, -0.017, 0.062, 1.216, 0.005, 0.258, 0.006]),
        "right_arm": np.array([0.2, -0.2, 0.0, -0.2, 0.0, 0.0, 0.0]),
        "right_gripper": np.array([-0.0222]),
    }
    init_qpos_noise_range: dict[str, np.ndarray] | None = None
    command_mode: dict[str, str | None] = {
        "legs_waist": "joint_position",
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = False

    # Matches g1_molmo's components/controller.py set_env(), which sets
    # `m.opt.timestep = 0.005` unconditionally for G1: the WBC's ONNX policy
    # (controllers/g1_wbc.py) was trained at that rate, not our scene default
    # of 0.002s. Applied in G1Robot.__init__. g1_wbc.py's _WBC_CONTROL_DEC (4)
    # is set to match -- change both together or the WBC leaves its ~50Hz.
    physics_timestep: float = 0.005

    # The WBC (or, in holo-base mode, G1HoloBaseGroup's mocap weld) holds the
    # pelvis at a constant height wherever the robot is placed, so unlike RBY1
    # or FloatingRUM the G1 cannot stand at a target-relative height. See
    # BaseRobotConfig.fixed_base_height.
    #
    # 0.793m is the pelvis height measured with this init_qpos right after
    # reset()/mj_forward, before the WBC takes over -- NOT the WBC's own
    # _WBC_HEIGHT_CMD (0.74), which is a crouched action-space reference. At
    # 0.74 these leg angles put the ankles through the floor, failing 10/10
    # placements. Confirmed stable for 3s+ in the standing smoke test.
    fixed_base_height: float | None = 0.793

    # Override for the right gripper pad geoms' (tangential, torsional, rolling)
    # friction triple -- None keeps the MJCF's own default (5, 0.05, 0.01).
    # Applied in G1Robot._apply_solver_overrides by matching geoms against that
    # default's tangential value, since the gripper_pad geoms are unnamed in
    # g1_dex.xml (only grouped by an MJCF `class`, which isn't queryable on the
    # compiled model). See molmo_spaces/robots/g1.py for the matching logic.
    gripper_friction: tuple[float, float, float] | None = None


class G1Dex3Config(G1Config):
    """The Unitree G1 with a Dex3-1 three-finger hand on each wrist (`g1_dex3`).

    Identical to `G1Config` (walking WBC, arms, head fisheye / wrist cameras) except the right
    Dex1 gripper is replaced by two 7-joint hands, move groups `left_hand` and `right_hand`
    (order: thumb_0..2, index_0..1, middle_0..1; 0 = fingers extended, positive/negative
    flexion per hand), and the left arm is actively controlled instead of hanging limp.
    Build the robot asset with scripts/data/format_conversion/build_g1_dex3.py.
    """

    robot_cls: type[G1Robot] | None = G1Dex3Robot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = G1Dex3Robot.from_mj_data
    robot_view_factory: RobotViewFactory | None = G1Dex3RobotView
    name: str = "g1_dex3"
    robot_xml_path: Path = Path("g1_dex3.xml")
    init_qpos: dict[str, np.ndarray] = {
        "legs_waist": G1Config.model_fields["init_qpos"].default["legs_waist"],
        "left_arm": np.array([0.212, -0.017, 0.062, 1.216, 0.005, 0.258, 0.006]),
        "right_arm": np.array([0.2, -0.2, 0.0, -0.2, 0.0, 0.0, 0.0]),
        "left_hand": np.zeros(7),
        "right_hand": np.zeros(7),
    }
    command_mode: dict[str, str | None] = {
        "legs_waist": "joint_position",
        "arm": "joint_position",
        "hand": "joint_position",
    }


class FloatingRUMRobotConfig(BaseRobotConfig):
    robot_cls: type[Robot] | None = FloatingRUMRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = FloatingRUMRobot
    robot_view_factory: RobotViewFactory | None = FloatingRUMRobotView
    robot_namespace: str = "robot_0/"
    ctrl_dt_ms: float = 50.0
    command_mode: dict = {}
    name: str | None = "floating_rum"
    robot_xml_path: Path = Path("model.xml")
    init_qpos: dict[str, list] = {
        "gripper": [0.0, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {}


class FloatingRobotiq2f85RobotConfig(BaseRobotConfig):
    robot_cls: type[Robot] | None = FloatingRobotiqRobot
    robot_factory: Callable[[mj.MjData, BaseRobotConfig], Robot] | None = FloatingRobotiqRobot
    robot_view_factory: RobotViewFactory | None = FloatingRobotiq2f85RobotView
    robot_namespace: str = "robot_0/"
    ctrl_dt_ms: float = 50.0
    command_mode: dict = {}
    action_spec: dict[str, int] = {"base": 7, "gripper": 2}  # Max lengths for action components
    name: str | None = "floating_robotiq"
    robot_xml_path: Path = Path("model.xml")
    init_qpos: dict[str, list] = {
        "gripper": [0.00296, 0.00296],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = {}


class I2rtYamRobotConfig(BaseRobotConfig):
    """Configuration for i2rt YAM 6-DOF robot."""

    robot_cls: type[Robot] | None = I2rtYamRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = I2rtYamRobot
    robot_view_factory: RobotViewFactory | None = I2rtYamRobotView
    robot_namespace: str = "robot_0/"
    name: str | None = "i2rt_yam"
    robot_xml_path: Path = Path("yam.xml")
    # Base platform size [width, depth, height] - raises robot above ground
    base_size: list[float] | None = [0.3, 0.3, 0.7]
    # Initial joint positions - modified from XML keyframe "home" to avoid wrist singularity
    # Original: "0 1.047 1.047 0 0 0" but joints 4,5,6 at 0 causes wrist singularity
    # Adding small offsets to wrist joints (4,5) to move away from singular configuration
    init_qpos: dict[str, list[float]] = {
        "arm": [0.0, 1.047, 1.047, 0.1, -0.1, 0.0],  # Offset joints 4,5 to avoid singularity
        "gripper": [0.0, 0.0],  # left_finger, right_finger (coupled)
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, _context: Any):
        super().model_post_init(_context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class BimanualYamRobotConfig(BaseRobotConfig):
    """Configuration for bimanual YAM robot (two 6-DOF arms with parallel grippers).

    The bimanual YAM consists of two YAM arms positioned 44cm apart,
    both facing forward.
    """

    robot_cls: type[Robot] | None = BimanualYamRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = BimanualYamRobot
    robot_view_factory: RobotViewFactory | None = BimanualYamRobotView
    robot_namespace: str = "robot_0/"
    name: str | None = "i2rt_yam"  # Use same directory as single-arm YAM
    robot_xml_path: Path = Path("bimanual_yam.xml")
    # Base platform size [x, y, z] - raises robot above ground
    # Wider in Y to accommodate both arms (44cm apart along Y axis)
    base_size: list[float] | None = [0.3, 0.8, 0.7]
    # Initial joint positions for both arms
    # These initializations are taken from observation values that I saw in the dataset
    init_qpos: dict[str, list[float]] = {
        "left_arm": [0.0624, 0.0109, 0.1707, -0.5938, 0.411, 0.3401],
        "right_arm": [0.0006, 0.0147, 0.1669, -0.6407, 0.0746, 0.1516],
        "left_gripper": [0.03914, 0.0],
        "right_gripper": [0.04068, 0.0],
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str | None] = {
        "arm": "joint_position",
        "gripper": "joint_position",
    }
    gravcomp: bool = True

    def model_post_init(self, _context: Any):
        super().model_post_init(_context)
        if "gripper" in self.command_mode:
            assert self.command_mode["gripper"] == "joint_position"
        if "arm" in self.command_mode:
            assert self.command_mode["arm"] in ["joint_position", "joint_rel_position"]


class HumanRBRobotConfig(BaseRobotConfig):
    """Configuration for a Rocketbox humanoid avatar driven as a robot.

    Unlike every other robot here the model is not one robot but one *of* a
    library: `avatar_variant` picks the build (see HumanRBVariant) and `uid` one
    of its characters, so `robot_dir` / `robot_xml_path` are derived from the
    two rather than set by hand.

    The articulated and skinned builds are published as robot asset sources and
    install on demand like any other robot model. The static build is not, and
    has to be generated locally by scripts/assets/convert_human_rb.py, which is
    also where the setup instructions live.

    `HumanRBRobot._load_robot_spec` rewrites that asset into an actuated robot on
    the way in -- see molmo_spaces/robots/human_rb.py for what that involves and
    which of the knobs below feed it.
    """

    robot_cls: type[HumanRBRobot] | None = HumanRBRobot
    robot_factory: Callable[[mj.MjData, Any], Robot] | None = HumanRBRobot
    robot_view_factory: RobotViewFactory | None = HumanRBRobotView
    robot_namespace: str = "robot_0/"
    name: str = "avatar"

    # Which character, and which build of it. Only the skinned build makes a
    # usable robot: the articulated one tears open at the seams once its joints
    # leave the rest pose, and the static one has no skeleton at all.
    uid: str = "Female_Adult_16"
    avatar_variant: HumanRBVariant = HumanRBVariant.SKINNED

    # Set from uid/avatar_variant in model_post_init; present here because
    # BaseRobotConfig declares them required.
    robot_xml_path: Path = Path("unset.xml")
    robot_dir: Path | None = None

    # Total body mass, distributed over the bones anthropometrically (see
    # human_rb.py's _MASS_FRACTIONS). The converter leaves every limb at 1e-08 kg
    # and the pelvis capsule at ~240kg, which no position actuator can drive.
    total_mass: float = 70.0

    # Position-actuator gains and passive joint properties for the 57 hinges
    # human_rb.py substitutes for the rig's ball joints. One set for all of them:
    # the mass distribution already scales what each joint has to hold, and the
    # avatar is a figure to be posed, not a machine with per-joint hardware to
    # model. These gains hold the reset pose to within ~0.02 rad against gravity
    # (measured over 5s on adult, child and costumed avatars alike).
    #
    # `joint_armature` is a stability floor, not a tuning knob: a hand bone
    # weighs 0.4kg and has an inertia around 1e-4, so a stiff position servo
    # driving it directly diverges within ~30 steps at the default 2ms
    # timestep. 0.05 is the first value that holds across the gain range;
    # 0.03 still blows up. Lower it only alongside a smaller timestep.
    joint_kp: float = 600.0
    joint_kd: float = 50.0
    joint_damping: float = 1.0
    joint_armature: float = 0.05

    # Hold the pelvis with a mocap weld (see HumanRBRobot.add_robot_to_scene).
    # Without it the avatar is a free-floating ragdoll balancing on the rounded
    # end of a single body-length collision capsule, i.e. it falls over.
    weld_base: bool = True

    # Bring the arms from the authored A-pose down to rest at the sides on
    # reset, per-avatar from its own bone geometry -- see
    # human_rb.arms_down_shoulder_angles. Applied on top of init_qpos.
    lower_arms: bool = True

    # The rig's own rest pose, i.e. standing. `lower_arms` above is what moves
    # the arms off it.
    init_qpos: dict[str, list[float]] = {
        mg_id: [0.0] * n_dofs for mg_id, n_dofs in human_rb_move_group_dofs().items()
    }
    init_qpos_noise_range: dict[str, list[float]] | None = None
    command_mode: dict[str, str] = dict.fromkeys(human_rb_move_group_dofs(), "joint_position")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        object.__setattr__(self, "robot_xml_path", Path(f"{self.uid}.xml"))

    def get_robot_dir(self) -> Path:
        """This character's directory inside its library.

        Resolved here rather than baked into `robot_dir` at construction so that
        merely building a config does not install a ~150MB asset library --
        get_robot_path downloads on first miss, and configs get constructed to be
        inspected, serialized and copied (see
        BaseMujocoTaskSampler._robot_configs) far more often than to be loaded.
        """
        if self.robot_dir is not None:
            return self.robot_dir
        return human_rb_library_dir(self.avatar_variant) / self.uid
