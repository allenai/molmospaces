"""Policy configuration classes for MolmoSpaces experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.planner.astar_planner import AStarPlannerConfig
from molmo_spaces.policy.base_policy import BasePolicy

# Import CuroboPlannerConfig if available (requires GPU), otherwise create a stub
try:
    from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
except (ImportError, RuntimeError):
    # Create a stub class when CuRobo isn't available (e.g., on non-GPU nodes)
    # This allows Pydantic to resolve forward references during config validation
    if TYPE_CHECKING:
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
    else:

        class CuroboPlannerConfig(Config):  # type: ignore
            """Stub for CuroboPlannerConfig when CuRobo is not available."""

            pass


class BasePolicyConfig(Config):
    """Base configuration for policies."""

    policy_cls: type[
        BasePolicy
    ]  # unless pre-instantiated before eval/datagen, should take (config, task) in constructor
    policy_type: str  # Type of the policy, e.g., "planner", "teleop", "learned", etc.


class ObjectManipulationPlannerPolicyConfig(BasePolicyConfig):
    """Configuration for Franka pick planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_type: str = "planner"

    # Pick-and-place pose offsets
    pregrasp_z_offset: float = 0.04  # Height above object for pregrasp
    postgrasp_z_offset: float = 0.05  # Height above object for postgrasp
    grasp_z_offset: float = 0.03  # Lower distance from pregrasp to grasp
    place_z_offset: float = 0.07  # Lower distance from preplace to place
    end_z_offset: float = 0.05  # Height above place target for final pose

    # Speed settings
    speed_slow: float = 0.08  # m/s for precise movements
    speed_fast: float = 0.20  # m/s for transport movements
    move_settle_time: float = 0.1  # seconds

    # Gripper timing
    gripper_close_duration: float = 0.5  # Time to close gripper
    gripper_open_duration: float = 0.25  # Time to open gripper

    # Randomization parameters
    randomize_grasp: bool = False  # Enable grasp pose randomization
    grasp_xy_noise: float = 0.02  # Max XY offset from object center (meters)
    grasp_yaw_noise: float = 0.5  # Max rotation around Z-axis (radians)
    pregrasp_height_noise: float = 0.03  # Additional height variation for pregrasp
    postgrasp_height_noise: float = 0.02  # Height variation for lift phase

    # Retry behavior parameters
    max_retries: int = 3  # Maximum number of retry attempts
    gripper_empty_threshold: float = 0.002  # Gripper separation to detect empty gripper (meters)
    phase_timeout: float = 10.0  # Maximum time to spend in any phase (seconds)
    max_sequential_ik_failures: int = 8  # Maximum number of IK failures
    tcp_pos_err_threshold: float = 0.1  # Retry if position error is greater than this
    tcp_rot_err_threshold: float = np.radians(30.0)  # Retry if rotation error is greater than this

    # grasp sampling configuration (collision checking)
    filter_colliding_grasps: bool = True
    grasp_collision_batch_size: int = 128
    grasp_collision_max_grasps: int = 512
    grasp_width: float = 0.08
    grasp_length: float = 0.05
    grasp_height: float = 0.01
    grasp_base_pos: list[float] = [0.0, 0.0, -0.04]  # position of grasp base in tcp frame
    # grasp sampling configuration (cost weighting)
    grasp_pos_cost_weight: float = 1.0
    grasp_rot_cost_weight: float = 0.01
    grasp_vertical_cost_weight: float = 2.0
    grasp_com_dist_cost_weight: float = 8.0
    # grasp sampling configuration (feasibility checking)
    filter_feasible_grasps: bool = True
    grasp_feasibility_batch_size: int = 256
    grasp_feasibility_max_grasps: int = 256

    # Debugging
    debug_poses: bool = False  # Enable debug printing for poses
    verbose: bool = True  # Enable verbose output for debugging


class OpenClosePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    # For opening tasks: horizontal orientation is strongly preferred over position
    # grasp_horizontal_cost_weight is multiplied by 10x for opening tasks to strongly penalize vertical orientations
    # The cost uses squared term: (abs(z-axis z-component))^2, so vertical orientations get heavily penalized
    grasp_pos_cost_weight: float = 1.0
    grasp_rot_cost_weight: float = 0.05
    grasp_vertical_cost_weight: float = 0.0
    grasp_horizontal_cost_weight: float = (
        10.0  # Base weight, multiplied by 10x for opening tasks (effective: 20.0)
    )
    grasp_com_dist_cost_weight: float = 0.0
    pregrasp_z_offset: float = 0.04  # Height above object for postgrasp

    # Speed settings
    speed_slow: float = 0.04  # m/s for precise movements
    speed_fast: float = 0.08  # m/s for transport movements
    move_settle_time: float = 0.2  # seconds

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.open_close_planner_policy import (
                OpenClosePlannerPolicy,
            )

            self.policy_cls = OpenClosePlannerPolicy


class PickPlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    postgrasp_z_offset: float = 0.08  # Height above object for postgrasp

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
                PickPlannerPolicy,
            )

            self.policy_cls = PickPlannerPolicy


class UnitreeG1RightArmPickPlannerPolicyConfig(PickPlannerPolicyConfig):
    """Pick planner config for Unitree G1 right-arm-only smoke datagen."""

    policy_cls: type = None
    filter_colliding_grasps: bool = False
    filter_feasible_grasps: bool = False
    grasp_feasibility_batch_size: int = 32
    grasp_feasibility_max_grasps: int = 32
    grasp_collision_batch_size: int = 32
    grasp_collision_max_grasps: int = 64
    max_retries: int = 0
    max_sequential_ik_failures: int = 4
    phase_timeout: float = 20.0

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
            UnitreeG1RightArmPickPlannerPolicy,
        )

        self.policy_cls = UnitreeG1RightArmPickPlannerPolicy


class PickAndPlacePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    move_settle_time: float = 0.5

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
                PickAndPlacePlannerPolicy,
            )

            self.policy_cls = PickAndPlacePlannerPolicy


class UnitreeG1RightArmPickAndPlacePlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    """Pick-and-place planner config for fixed-base G1 right-arm tabletop datagen."""

    policy_cls: type = None
    pregrasp_z_offset: float = 0.06
    gripper_close_duration: float = 1.0
    gripper_empty_threshold: float = 0.0005
    grasp_rot_cost_weight: float = 0.02
    grasp_vertical_cost_weight: float = 0.5
    filter_colliding_grasps: bool = False
    filter_feasible_grasps: bool = False
    grasp_feasibility_batch_size: int = 32
    grasp_feasibility_max_grasps: int = 128
    grasp_collision_batch_size: int = 32
    grasp_collision_max_grasps: int = 64
    max_retries: int = 0
    max_sequential_ik_failures: int = 4
    phase_timeout: float = 24.0
    pregrasp_tcp_pos_err_threshold: float = 0.1
    pregrasp_tcp_rot_err_threshold: float = float("inf")
    enable_failure_diagnostics: bool = True
    diagnostic_large_object_max_extent_m: float = 0.18
    diagnostic_large_object_volume_m3: float = 0.004
    diagnostic_failure_hold_duration_s: float = 2.0
    record_unfiltered_attempt_on_no_feasible_grasp: bool = True
    g1_ik_debug: bool = False
    g1_ik_debug_higher_z_offset: float = 0.05
    g1_ik_debug_top_k_grasps: int = 5
    g1_online_grasp_selector: bool = True
    g1_grasp_candidate_limit: int = 256
    g1_grasp_ik_eval_limit: int = 256
    g1_grasp_require_all_pick_place_phases: bool = True
    g1_grasp_joint_margin_weight: float = 0.5
    g1_grasp_joint_motion_weight: float = 0.25
    g1_grasp_topdown_weight: float = 1.0
    g1_grasp_max_tcp_rot_deg: float = 120.0
    g1_ignore_flipped_grasps: bool = True
    g1_grasp_inward_xy_offset: float = 0.006
    g1_grasp_table_clearance: float = 0.065
    g1_center_grasp_lateral: bool = True
    g1_grasp_lateral_centering_scale: float = 1.0
    g1_grasp_lateral_centering_max_offset: float = 0.02
    g1_level_grasp_orientation: bool = False
    g1_grasp_level_max_tilt_deg: float = 35.0
    g1_require_fingertip_pad_grasp_contact: bool = False
    g1_reject_non_fingertip_grasp_object_contact: bool = True
    g1_reject_grasp_table_contact: bool = True
    g1_reject_open_grasp_object_contact: bool = True
    g1_grasp_single_pad_contact_penalty: float = 0.5
    g1_closed_grasp_quality_enabled: bool = True
    g1_closed_grasp_settle_steps: int = 25
    g1_closed_grasp_min_pad_geom_count: int = 2
    g1_closed_grasp_max_object_shift_m: float = 0.05
    g1_closed_grasp_penalty_per_missing_pad: float = 1.0
    g1_pregrasp_min_vertical_lift: float = 0.10
    g1_pregrasp_object_clearance: float = 0.12
    g1_place_travel_object_clearance: float = 0.13
    g1_held_object_speed: float = 0.04
    g1_postgrasp_hold_duration: float = 0.4
    g1_record_partial_attempt_on_no_full_grasp_candidate: bool = True
    g1_pick_lift_only: bool = False

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
            UnitreeG1RightArmPickAndPlacePlannerPolicy,
        )

        self.policy_cls = UnitreeG1RightArmPickAndPlacePlannerPolicy


class CuroboOpenClosePlannerPolicyConfig(OpenClosePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    left_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    right_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    enable_collision_avoidance: bool = True
    batch_size: int = 4
    max_grasping_timesteps: int = 5
    max_opening_timesteps: int = 5
    max_steps_per_waypoint: int = 10
    max_batch_plan_attempts: int = 4
    pregrasp_z_offset: float = 0.02
    max_planning_reattempts: int = 2
    gripper_closed_pos: float = 0.0
    gripper_closed_tolerance: float = 0.005
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "head": 0.5,
        "right_arm": 0.5,
        "left_arm": 0.5,
    }
    grasp_vertical_cost_weight: float = 2.0
    attach_obj: bool = False
    max_settle_steps: int = 5
    max_height_adjustment_steps: int = 10
    server_timeout: float | None = (
        300.0  # gRPC deadline for motion planning calls (seconds), None = no deadline
    )
    server_urls: list[str] = [
        "jupiter-cs-aus-107.reviz.ai2.in:10002",
    ]


class CuroboPickAndPlacePlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    left_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    right_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    enable_collision_avoidance: bool = True
    batch_size: int = 4
    max_grasping_timesteps: int = 5
    max_opening_timesteps: int = 5
    max_steps_per_waypoint: int = 10
    max_batch_plan_attempts: int = 4
    pregrasp_z_offset: float = 0.02  # [m]
    max_planning_reattempts: int = 5
    gripper_closed_pos: float = 0.0  # [m]
    gripper_closed_tolerance: float = 0.005  # [m]
    velocity_constraints: dict[str, float] = {
        "base": 0.5,  # [m / policy_dt_ms]
        "head": 0.5,  # [rad / policy_dt_ms]
        "right_arm": 0.5,  # [rad / policy_dt_ms]
        "left_arm": 0.5,  # [rad / policy_dt_ms]
    }
    grasp_vertical_cost_weight: float = 0.5
    attach_obj: bool = False
    max_settle_steps: int = 5
    server_timeout: float | None = (
        300.0  # gRPC deadline for motion planning calls (seconds), None = no deadline
    )
    server_urls: list[str] = [
        "jupiter-cs-aus-107.reviz.ai2.in:10002",
    ]


class PickAndPlaceNextToPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_next_to_planner_policy import (
            PickAndPlaceNextToPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceNextToPlannerPolicy


class PickAndPlaceColorPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_color_planner_policy import (
            PickAndPlaceColorPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceColorPlannerPolicy


class DoorOpeningPolicyConfig(BasePolicyConfig):
    """Configuration for RBY1 door opening planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_type: str = "planner"

    # RBY1-specific policy parameters
    # Motion planning parameters
    left_curobo_planner_config: CuroboPlannerConfig | None = (
        None  # will be set in __init_policy_config
    )
    right_curobo_planner_config: CuroboPlannerConfig | None = (
        None  # will be set in __init_policy_config
    )

    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "head": 0.5,
        "right_arm": 0.5,
        "left_arm": 0.5,
    }
    enable_collision_avoidance: bool = True  # Whether to enable collision avoidance
    relevant_collision_objects_radius: float = (
        3.0  # Radius in meters from the door handle around which collision objects are considered
    )
    plan_in_robot_frame: bool = (
        True  # Whether to plan in robot frame or world frame (True keeps base stable)
    )
    max_planning_failures: int = 15

    # Trajectory execution parameters
    max_steps_per_waypoint: int = 10
    joint_position_tolerance: float = 0.0275

    # Gripper control parameters
    gripper_closed_pos: float = 0.0
    left_gripper_close_command: dict = {"left_gripper": 100.0}
    left_gripper_open_command: dict = {"left_gripper": -100.0}
    right_gripper_close_command: dict = {"right_gripper": 100.0}
    right_gripper_open_command: dict = {"right_gripper": -100.0}
    gripper_closed_tolerance: float = 0.005  # [m]
    max_grasping_timesteps: int = 5

    # Door opening parameters
    pre_grasp_distance: float = -0.18  # distance from door handle before grasping it
    articulation_deltas: list[float] = [
        (np.pi / 180.0) * 13.0
    ]  # delta radians to articulate door joint(s)
    first_pushing_articulation_deltas: list[float] = [
        (np.pi / 180.0) * 30.0
    ]  # special first delta articulation when pushing door

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = False  # Enable verbose output for debugging


class NavToObjPlannerPolicyConfig(BasePolicyConfig):
    """Base configuration for navigation to object planner policies."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_type: str = "planner"

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = True  # Enable verbose output for debugging


class AStarNavToObjPolicyConfig(NavToObjPlannerPolicyConfig):
    """Configuration for A* navigation policy (discrete grid-based planner)."""

    policy_cls: type = None

    # A* planner configuration
    planner_config: AStarPlannerConfig = AStarPlannerConfig()

    # A* planner parameters (for backward compatibility)
    map_path: str | None = None  # Path to occupancy map
    downscale: int = 5  # Downscaling factor for grid

    # Policy-related parameters
    path_interpolation_density: int = (
        1  # Num points to add between planner waypoint pairs (regardless of distance)
    )
    path_max_inter_waypoint_dist: float = 0.25  # Max distance between consecutive waypoints
    path_max_inter_waypoint_angle: float = float(
        np.deg2rad(10)
    )  # Max arc length between consecutive waypoints
    path_min_dist_to_target_center: float = (
        0.8  # Skip approaching target center below this distance
    )
    plan_max_retries: int = 3  # Allowed number of planning retries in episode

    # TODO the replanning criterion is weak, as it does not rely on actual collision,
    #  but a loose estimate based on rate decrease of spatial-angular distance to next waypoint.
    #  It needs further work to be usable, so you may want to keep a large value to prevent it for now.
    plan_fail_after_waypoint_steps: int = (
        10  # Number of steps within current waypoint to check for need to replan
    )

    plan_fail_max_dist_delta: float = 0.01  # Max difference between dists to waypoint to consider need to replan after plan_fail_after_waypoint_steps
    plan_stick_to_original_target: bool = (
        False  # Allows replanning to other possible valid targets when False
    )

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.navigation.astar_planner_policy import (
                AStarSmoothPlannerPolicy,
            )

            self.policy_cls = AStarSmoothPlannerPolicy


class DummyPolicyConfig(BasePolicyConfig):
    """Policy config that uses DummyPolicy for testing."""

    policy_type: str = "dummy"
    policy_cls: type = None  # Set in model_post_init

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import DummyPolicy

            object.__setattr__(self, "policy_cls", DummyPolicy)


class BrownianMotionPolicyConfig(BasePolicyConfig):
    """Policy that applies Gaussian noise increments over noop control, resulting in Brownian motion."""

    policy_cls: type = None
    policy_type: str = "dummy"
    std: float = 0.1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy

            self.policy_cls = BrownianMotionPolicy
