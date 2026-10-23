"""Policy configuration classes for MolmoSpaces experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.env.scene import spec_ops
from molmo_spaces.planner.astar_planner import AStarPlannerConfig
from molmo_spaces.planner.curobo_planner_config import CuroboPlannerConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient


class BasePolicyConfig(Config):
    """Base configuration for policies."""

    policy_cls: type[BasePolicy] | None

    policy_factory: PolicyFactory | None
    """Factory function to create the policy instance from a config and task, can be same as ``policy_cls``"""

    policy_type: str
    """Type of the policy, e.g., "planner", "teleop", "learned", etc."""

    force_enable_depth: bool = False
    """Whether or not to require all cameras to record depth"""


class ObjectManipulationPlannerPolicyConfig(BasePolicyConfig):
    """Configuration for Franka pick planner policy."""

    policy_cls: type[BasePolicy] | None = None
    policy_factory: PolicyFactory | None = None
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
    # Grasp-collision probes added to the scene (spec_ops.
    # add_grasp_probe_bodies). Each one tests a candidate grasp, so a batch of N
    # checks N grasps per mj_kinematics+mj_collision pass -- but every probe is a
    # freejointed body that costs DOF on EVERY sim step thereafter.
    #
    # Measured on procthor-10k-val/val_0 with the G1 (4k geoms, MuJoCo 3.11):
    #     N=1    mj_step 0.322 ms   512-grasp check 165.7 ms
    #     N=128  mj_step 0.474 ms   512-grasp check  36.9 ms
    # so a batch costs +0.152 ms/step and saves 128.8 ms/check -- break-even at
    # ~850 sim steps per grasp check. Batching wins by only ~4.5x, not 128x,
    # because each pass re-collides the whole scene.
    #
    # Default 1: rollout workloads (datagen episodes, the interactive shell, the
    # FetchMan parity run) simulate thousands of steps per grasp check and sit
    # far past that break-even, and one probe is also what gold's scene has.
    # Raise it for reset-heavy sweeps that check many grasps and barely
    # simulate -- grasp-library filtering, placement validation, house sweeps.
    grasp_collision_batch_size: int = 1
    grasp_collision_max_grasps: int = 512
    # Which probe shapes the scene build puts in the model for this policy
    # (spec_ops.PROBE_SHAPE_*). Only the shapes a policy actually drives are
    # loaded: every probe is a freejointed body that costs DOF on every sim
    # step. The parametric jaw is what `get_noncolliding_grasp_mask` scatters
    # over candidates, so it is what a policy needs by default; a robot's own
    # gripper model (PROBE_SHAPE_GRIPPER_XML) is added only by the policies
    # that drive it -- see FetchmanPickPlannerPolicyConfig.
    grasp_probe_shapes: tuple[str, ...] = (spec_ops.PROBE_SHAPE_JAW,)
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

    # which grasp libraries to use, in descending priority (will be filtered by availability for each asset)
    # if None, all available libraries for the object will be used
    grasp_libraries: list[str] | None = None

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

    grasp_libraries: list[str] | None = ["droid"]  # only thor provides articulated grasps

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.open_close_planner_policy import (
                OpenClosePlannerPolicy,
            )

            self.policy_cls = OpenClosePlannerPolicy
            self.policy_factory = OpenClosePlannerPolicy


class PickPlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type[BasePolicy] | None = None
    postgrasp_z_offset: float = 0.08  # Height above object for postgrasp

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
                PickPlannerPolicy,
            )

            self.policy_cls = PickPlannerPolicy
            self.policy_factory = PickPlannerPolicy


class PickAndPlacePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type[BasePolicy] | None = None
    move_settle_time: float = 0.5

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
                PickAndPlacePlannerPolicy,
            )

            self.policy_cls = PickAndPlacePlannerPolicy
            self.policy_factory = PickAndPlacePlannerPolicy


class CuroboOpenClosePlannerPolicyConfig(OpenClosePlannerPolicyConfig):
    policy_cls: type[BasePolicy] | None = None
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
    policy_cls: type[BasePolicy] | None = None
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
    policy_cls: type[BasePolicy] | None = None

    def model_post_init(self, _context: Any) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_next_to_planner_policy import (
            PickAndPlaceNextToPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceNextToPlannerPolicy
        self.policy_factory = PickAndPlaceNextToPlannerPolicy


class PickAndPlaceColorPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type[BasePolicy] | None = None

    def model_post_init(self, _context: Any) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_color_planner_policy import (
            PickAndPlaceColorPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceColorPlannerPolicy
        self.policy_factory = PickAndPlaceColorPlannerPolicy


class DoorOpeningPolicyConfig(BasePolicyConfig):
    """Configuration for RBY1 door opening planner policy."""

    policy_cls: type[BasePolicy] | None = None
    policy_factory: PolicyFactory | None = None
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

    policy_cls: type[BasePolicy] | None = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = True  # Enable verbose output for debugging


class AStarNavToObjPolicyConfig(NavToObjPlannerPolicyConfig):
    """Configuration for A* navigation policy (discrete grid-based planner)."""

    policy_cls: type[BasePolicy] | None = None

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
            self.policy_factory = AStarSmoothPlannerPolicy


class FetchManBasePlannerPolicyConfig(NavToObjPlannerPolicyConfig):
    """Configuration for FetchManBasePlannerPolicy -- a port of g1_molmo's
    navigation policy (molmospaces/agents/policy.py in the g1_molmo reference
    repo). Unlike AStarPlannerPolicy, which pre-bakes an explicit
    rotate-then-drive waypoint schedule at plan time, this recomputes a
    [vx, vy, yaw_rate] base velocity command from the robot's live pose every
    step (see FetchManBasePlannerPolicy._update_nav_command)."""

    policy_cls: type = None

    # Grid A* (ported from g1_molmo's _astar/_coarsen_and_dist)
    planner_config: AStarPlannerConfig = AStarPlannerConfig()
    downscale: int = 4  # Coarsening factor for the A* search grid
    wall_radius: int = 10  # Distance (in coarse cells) at which the wall-clearance cost reaches 0
    wall_gain: float = 6.0
    wall_exp: float = 2.0
    simplify_clearance: int = 6  # Px clearance required for a line-of-sight path shortcut

    # Live waypoint-following control law (ported from g1_molmo's _update_nav_command)
    waypoint_reach: float = 0.10  # Distance to advance to the next non-final waypoint
    # See FetchmanPickPlannerPolicyConfig's final_reach comment: raised from
    # g1_molmo's 0.05 to clear the effective minimum turning radius imposed by
    # G1Robot's velocity deadband/floor (min_speed/drive_max_turn) -- 0.05
    # left the robot orbiting the goal forever instead of ever arriving.
    final_reach: float = 0.3  # Distance to consider the final waypoint reached
    turn_kp: float = 2.0  # Proportional gain, heading error -> yaw rate
    max_turn: float = 1.0  # Max yaw rate (rad/s) while driving
    face_turn: float = 1.2  # Max yaw rate (rad/s) during the terminal face-the-target turn
    # Loosened from g1_molmo's 0.1/0.25 rad, which sit below the WBC's ~15deg
    # yaw-tracking ceiling: the heading error never settled and the turn/drive
    # branches hunted indefinitely ~0.3m short of the goal.
    face_tol: float = 0.35  # ~20deg -- heading error tolerance to end the terminal facing turn
    face_wp_tol: float = 0.524  # 30deg -- heading error above which translation is suppressed
    speed: float = 0.4  # Cruise linear speed (m/s)
    min_speed: float = 0.15  # Minimum linear speed while still short of a non-final waypoint
    brake_dist: float = 0.70  # Distance from the goal at which the smoothstep brake engages
    stop_pad: float = 0.04  # Extra margin added to final_reach to absorb walking inertia
    # Tighter yaw-rate cap while translating (max_turn/face_turn are for
    # in-place turns): the WBC's gait stalls to a crawl under a forward speed
    # plus a yaw_rate near max_turn.
    drive_max_turn: float = 0.3

    plan_max_retries: int = 3  # Number of alternate target candidates to try if planning fails

    # FetchManBasePlannerPolicyPort only: walk to a grasping standoff on this
    # annulus around the object (pick_planner_policy_g1.sample_standoff_pose) instead
    # of NavGoalSampler's goal, so a following `pick` grasps without walking.
    standoff_radius_range: tuple[float, float] | None = None
    standoff_map_extra_inflation: float = 0.125  # planning-map inflation, as the pick's

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.navigation.fetchman_base_planner_policy import (
                FetchManBasePlannerPolicy,
            )

            self.policy_cls = FetchManBasePlannerPolicy
            self.policy_factory = FetchManBasePlannerPolicy


class DummyPolicyConfig(BasePolicyConfig):
    """Policy config that uses DummyPolicy for testing."""

    policy_type: str = "dummy"
    policy_cls: type[BasePolicy] | None = None
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import DummyPolicy

            self.policy_cls = DummyPolicy
            self.policy_factory = make_lenient(DummyPolicy)


class BrownianMotionPolicyConfig(BasePolicyConfig):
    """Policy that applies Gaussian noise increments over noop control, resulting in Brownian motion."""

    policy_cls: type[BasePolicy] | None = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "dummy"
    std: float = 0.1

    def model_post_init(self, _context: Any) -> None:
        super().model_post_init(_context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy

            self.policy_cls = BrownianMotionPolicy
            self.policy_factory = make_lenient(BrownianMotionPolicy)
