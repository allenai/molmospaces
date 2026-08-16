"""Policy configuration classes for MolmoSpaces experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.planner.astar_planner import AStarPlannerConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

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

    policy_cls: type[BasePolicy]
    policy_factory: PolicyFactory
    """
    Factory function to create the policy instance from a config and task, can be same as ``policy_cls``.
    """
    policy_type: str  # Type of the policy, e.g., "planner", "teleop", "learned", etc.
    force_enable_depth: bool = False
    """
    If true, require all cameras to record depth.
    In eval the cameras will be overridden, otherwise it will just require the camera system config to enable depth.
    """


class ObjectManipulationPlannerPolicyConfig(BasePolicyConfig):
    """Configuration for Franka pick planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
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
            self.policy_factory = PickPlannerPolicy


class FetchmanPickPlannerPolicyConfig(PickPlannerPolicyConfig):
    """PickPlannerPolicy variant using mink's whole-body IK (waist + pelvis
    height assist reach) instead of arm-only analytical IK. G1-only; requires
    G1Config's default WBC-walking mode (use_holo_base=False). See
    FetchmanPickPlannerPolicy's docstring.
    """

    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    # g1_molmo has no mid-trajectory grip-quality abort at all: it never
    # checks (or retries a plan over) how tightly the gripper closed --
    # success is judged purely at the end, by the task's own reward/contact
    # check (see PickG1Task). Our own TCPMoveSequence.check_failure's
    # gripper_empty_threshold-based abort (base default 0.002) fires the
    # instant the "lift" phase starts if the gripper closed too near its own
    # fully-closed joint limit -- for a thin-rimmed object like a bowl, a
    # perfectly good grip can land right in that rejection zone (confirmed
    # empirically: real failures landed just 0.00003-0.0018m past threshold,
    # firing before the lift motion had even begun), aborting a pick before
    # it ever gets a chance to succeed. -inf disables the comparison
    # entirely (inter_finger_dist can never be less than -inf) rather than
    # just loosening the margin, matching g1_molmo's real "never abort for
    # this reason" behavior. Safe to disable: PickG1Task's own success check
    # independently requires real lift height AND real finger-object
    # contact, so a genuinely-dropped object still can't report success.
    gripper_empty_threshold: float = float("-inf")

    # g1_molmo's GraspPolicy.LIFT -- how far above the grasp pose the lift
    # target sits. Overrides PickPlannerPolicyConfig's postgrasp_z_offset=0.08
    # (a different, arm-only-IK-tuned pipeline's value; not g1_molmo-derived).
    postgrasp_z_offset: float = 0.15

    # Unlike PickPlannerPolicy's pregrasp_z_offset (a pure world-Z lift above the
    # grasp pose), g1_molmo's reference retreats the pregrasp pose along the
    # grasp pose's own local Z (approach) axis instead, by a randomly-drawn
    # distance -- see FetchmanPickPlannerPolicy.PREGRASP_OFFSET_RANGE and
    # _compute_target_poses (ported directly, including the randomization: not
    # just the direction) and g1_molmo's GraspPolicy.PREGRASP_OFFSET.

    # How many cost-ranked grasp candidates FetchmanPickPlannerPolicy tries
    # (falling through to the next if pregrasp/grasp/lift IK fails) before
    # giving up. Matches g1_molmo's GraspPolicy.plan() PATH_CHECK_K=5 ("the
    # lowest-error candidate is almost always the right pick -- the rest are
    # insurance"). select_grasp_pose's cost function has no notion of
    # "reachable from the robot's current pose", so the single geometrically-
    # best candidate can require more reach/twist than a nearby alternative.
    grasp_candidates_to_try: int = 5

    # Low-pass filter coefficient for the height/waist *command* sent to
    # G1WalkController each tick, applied in FetchmanPickPlannerPolicy.
    # _tcp_to_jp_fn: new = old + alpha * (fresh_ik_solution - old). Matches
    # g1_molmo's own execution loop exactly (see _advance_grasp: height_cmd =
    # self._height_cmd + 0.1 * (ik_h - self._height_cmd)): both sides now run
    # at the same real 5ms policy tick (G1Config.physics_timestep +
    # PickG1DataGenConfig.policy_dt_ms=5.0, see their own docstrings), so the
    # raw alpha is directly comparable again -- unlike an earlier version of
    # this port, which copied this same 0.1 while still running at a 66ms
    # (then 4ms) tick, giving it a ~13x (then ~1.25x) slower real-time time
    # constant than g1_molmo actually has. Re-derive this (and IK_DECIM)
    # together if either side's tick duration ever changes again.
    height_waist_smoothing_alpha: float = 0.1

    # PickPlannerPolicy._compute_trajectory uses speed_fast for the pregrasp
    # approach (safe/quick since not near the object yet) and speed_slow for
    # grasp/lift. That assumes near-instantaneous joint tracking, true for the
    # arm's JointPosController but not for G1's waist/height, which move via
    # G1WalkController's torque-PD-tracked WBC (a trained ONNX policy
    # converging over real simulated dynamics). Slowing the pregrasp approach
    # to match speed_slow reduces (but doesn't eliminate) the lag.
    speed_fast: float = 0.08

    # g1_molmo's reference GraspPolicy has no analogue of TCPMoveSequence's
    # mid-motion tracking-error abort at all (see _grasp_phase_done in
    # ~/code/g1_molmo/molmospaces/agents/policy.py): it just keeps refining
    # the WBC's IK target every tick and only *advances* a phase once the
    # gripper actually converges near it (pos_err<0.035, with a minimum-step
    # floor but no maximum) -- transient lag while the WBC is still catching
    # up is expected and never treated as a failure. TCPMoveSequence's fixed-
    # duration-then-settle model doesn't have a real equivalent of "wait for
    # convergence", so approximate it here: don't abort on transient tracking
    # error (WBC needs real time, not instant tracking) and give a generous
    # fixed settle window at the end of each move for it to actually catch up.
    tcp_pos_err_threshold: float = 1.0
    tcp_rot_err_threshold: float = np.radians(60.0)
    move_settle_time: float = 1.0

    # select_grasp_pose's "vertical_cost_weight * dists_up" term (its base
    # ObjectManipulationPlannerPolicyConfig default is 2.0) uses the *signed*
    # z-component of the grasp frame's approach axis, not abs/squared -- so it
    # doesn't penalize vertical orientations symmetrically as intended, it
    # actively rewards top-down approaches (dists_up ~ -1 gives cost ~ -2.0).
    # Confirmed by direct comparison against g1_molmo's real grasp choices for
    # the same object/library: g1_molmo (no cost-based re-ranking, just raw
    # IK-error sort) consistently lands on horizontal side grasps, while ours
    # was ranking top-down cap grasps highest for the identical candidate
    # pool. OpenClosePlannerPolicyConfig already independently discovered
    # this and works around it the same way: disable the buggy signed term
    # and use grasp_horizontal_cost_weight's squared, symmetric term instead.
    grasp_vertical_cost_weight: float = 0.0
    grasp_horizontal_cost_weight: float = 2.0

    # Walk phase (see FetchmanPickPlannerPolicy._plan_walk_path/_update_nav_command):
    # g1_molmo's G1Controller spawns the robot anywhere reachable in the scene
    # (its own initial-spawn radius is unrelated to arm reach) and A*-walks it
    # to a standoff point near the pickup object before ever attempting the
    # arm-only WBC grasp reach -- PickPlannerPolicy/FetchmanPickPlannerPolicy
    # previously had no such phase and assumed the task sampler's
    # place_robot_near already left the robot within arm's reach. These fields
    # mirror FetchManBasePlannerPolicyConfig's (the existing standalone port of
    # g1_molmo's navigation policy) field-for-field, so the two share identical
    # walk behavior/tuning; kept as a separate field set (not inherited from a
    # shared base) since the two config classes don't share a common ancestor.
    planner_config: AStarPlannerConfig = AStarPlannerConfig()
    downscale: int = 4
    wall_radius: int = 10
    wall_gain: float = 6.0
    wall_exp: float = 2.0
    simplify_clearance: int = 6
    waypoint_reach: float = 0.10
    # g1_molmo's own value (0.05) plus the smoothstep brake's min_speed floor
    # (needed to clear G1Robot's velocity deadband -- see
    # _update_nav_command's floor-at-min_speed comment) combine into an
    # effective minimum turning radius of roughly min_speed/drive_max_turn
    # (~0.15/0.3 = 0.5m here) that the robot physically cannot converge
    # tighter than while still translating -- demanding arrival within 0.09m
    # (final_reach+stop_pad) left it orbiting the goal forever instead of
    # ever satisfying the check. Raised well above that radius so "close
    # enough" is reachable; the terminal facing turn (pure rotation, no
    # minimum-speed constraint) handles final heading precision instead.
    final_reach: float = 0.3
    turn_kp: float = 2.0
    max_turn: float = 1.0
    face_turn: float = 1.2
    # See FetchManBasePlannerPolicyConfig's face_tol/face_wp_tol comment: both
    # loosened from g1_molmo's original 0.1/0.25 rad, which sit at/below
    # G1Robot's G1WalkController's own documented ~15deg yaw-tracking ceiling
    # and cause a turn/drive hunting oscillation that never converges.
    face_tol: float = 0.35
    face_wp_tol: float = 0.524
    speed: float = 0.4
    min_speed: float = 0.15
    brake_dist: float = 0.70
    stop_pad: float = 0.04
    drive_max_turn: float = 0.3
    # Standoff distance from the pickup object for the walk goal (NavGoalSampler's
    # distance_threshold) -- matches FetchManBasePlannerPolicy/NavGoalSampler's own
    # default rather than a Fetchman-specific value, since the standoff point just
    # needs to land within arm's reach, same requirement either policy has.
    walk_goal_distance_threshold: float = 0.5

    # g1_molmo's GRASP_PROFILE (spawn_at_grasp=True, which PickG1DataGenConfig's
    # short base_pose_sampling_radius_range=(0.2, 0.5) mirrors) never invokes A*
    # walk-planning at all -- the env spawns the robot directly at the intended
    # standoff pose, so start==goal by construction and there's nothing to walk.
    # Our own task sampler places the robot within a similarly short radius of
    # the object, but independently, via its own occupancy-based validity check
    # -- not the same computation NavGoalSampler/A* use to pick and route to a
    # standoff point, so the two don't always agree on what's "reachable" even
    # when the robot is already sitting almost exactly there. Confirmed
    # empirically: a real, deterministic case (procthor-10k-val house 0's bowl)
    # where the robot spawns ~0.85m from a valid NavGoalSampler standoff point
    # (comfortably within arm's reach, no meaningful walk needed) but A*'s
    # coarse costmap reports no path at all -- almost certainly the robot's own
    # spawn cell (right next to the same counter the bowl sits on) falling
    # inside the wall-clearance inflation buffer that keeps the coarse grid
    # simple. If the sampled standoff goal is already within this distance,
    # skip A* and treat the robot as arrived on the spot, matching what
    # g1_molmo's short-radius spawn effectively guarantees.
    direct_arrival_max_dist: float = 1.2

    # Walk goal for G1PickPlannerPolicy (the reference stack), which does its own
    # goal sampling + A* inside G1Controller rather than going through
    # NavGoalSampler/AStarPlanner -- so the walk_* fields above don't reach it.
    # Both values are g1_molmo's own, for the same two computations:
    #   - the standoff annulus around the pickup object that its env samples the
    #     grasp pose on (env.py's grasp_spawn_radius_min/max = 0.25/0.80)
    #   - occ_safe = occ.dilated(0.125): the extra inflation its A* plans on, on
    #     top of the occupancy map's own agent radius. Goal sampling uses the
    #     un-inflated map (a standoff pose hugging the counter the object sits on
    #     is fine to stand at, just not to route through).
    # Deliberately not folded into the task sampler's
    # base_pose_sampling_radius_range: that one decides where the robot *spawns*,
    # this one where it must end up to grasp, and datagen configs set the former
    # wide on purpose to exercise the walk phase.
    goal_standoff_radius_range: tuple[float, float] = (0.25, 0.80)
    nav_map_extra_inflation: float = 0.125
    # Candidate standoff poses to try before giving up (g1_molmo's
    # _sample_goal_pose attempts=25). Each rejected candidate costs one
    # occupancy lookup plus a connected-component query, no simulation.
    goal_sampling_attempts: int = 25
    # G1Controller gives up on the walk once env.time passes walk_timeout_s --
    # absolute sim time, not time-in-phase. g1_molmo's own 20.0 is fine for its
    # spawns (a few metres at most, ~0.3-0.5 m/s), but a walk goal further out
    # than roughly speed*20s would be abandoned mid-stride, so the timeout is
    # raised to fit the planned path: path_length / speed * walk_timeout_slack.
    # The slack covers what a straight-line estimate leaves out -- turning in
    # place at each waypoint, the smoothstep brake ramp, and the terminal facing
    # turn. Measured 2.05m in 7.5s against a commanded 0.308 m/s (~1.15x), so 2.0
    # is a margin, not a fit. Never lowers the timeout below walk_timeout_s.
    walk_timeout_s: float = 20.0
    walk_timeout_slack: float = 2.0

    def model_post_init(self, __context) -> None:
        # Skip PickPlannerPolicyConfig.model_post_init (it would set policy_cls
        # to PickPlannerPolicy) and go straight to its own parent.
        super(PickPlannerPolicyConfig, self).model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import (
                G1PickPlannerPolicy,
            )

            self.policy_cls = G1PickPlannerPolicy
            self.policy_factory = G1PickPlannerPolicy


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
            self.policy_factory = PickAndPlacePlannerPolicy


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
        self.policy_factory = PickAndPlaceNextToPlannerPolicy


class PickAndPlaceColorPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_color_planner_policy import (
            PickAndPlaceColorPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceColorPlannerPolicy
        self.policy_factory = PickAndPlaceColorPlannerPolicy


class DoorOpeningPolicyConfig(BasePolicyConfig):
    """Configuration for RBY1 door opening planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
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

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
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
    # g1_molmo's own values here (0.1/0.25 rad) sit at or below G1Robot's
    # G1WalkController's own documented yaw-tracking ceiling (~15deg -- see
    # g1.py's _YAW_GATE_THRESHOLD comment: "G1WalkController's yaw tracking
    # has its own residual convergence ceiling around 15deg"). At those tight
    # values the heading error can never settle inside tolerance, so the
    # turn/drive branches hunt back and forth indefinitely instead of
    # converging -- confirmed empirically via FetchmanPickPlannerPolicy's walk
    # phase stalling ~0.3m short of goal, oscillating between a pure-turn and
    # a drive command every few dozen ticks. Loosened to comfortably clear
    # that ceiling, matching (face_wp_tol) or exceeding (face_tol) the already
    # -proven _YAW_GATE_THRESHOLD=30deg used by G1Robot's own sibling
    # (_waypoint_to_velocity_target) nav-command path.
    face_tol: float = 0.35  # ~20deg -- heading error tolerance to end the terminal facing turn
    face_wp_tol: float = 0.524  # 30deg -- heading error above which translation is suppressed
    speed: float = 0.4  # Cruise linear speed (m/s)
    min_speed: float = 0.15  # Minimum linear speed while still short of a non-final waypoint
    brake_dist: float = 0.70  # Distance from the goal at which the smoothstep brake engages
    stop_pad: float = 0.04  # Extra margin added to final_reach to absorb walking inertia
    # Separate, tighter yaw-rate cap for the *simultaneous* turn correction
    # applied while already translating (once heading is within face_wp_tol) --
    # distinct from max_turn/face_turn, which are for pure in-place turning
    # with zero forward command. Confirmed empirically (reproduces identically
    # via the pre-existing, unmodified FetchManBasePlannerPolicy, so this is a
    # genuine G1WalkController characteristic, not specific to this port):
    # G1WalkController's real gait effectively stalls forward progress to a
    # crawl when commanded a forward speed together with a yaw_rate anywhere
    # close to max_turn (e.g. vx=0.2 + yaw_rate=0.5-0.6 measured near-zero net
    # displacement over 30+ seconds), even though either alone works fine.
    drive_max_turn: float = 0.3

    plan_max_retries: int = 3  # Number of alternate target candidates to try if planning fails

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
    policy_cls: type = None  # Set in model_post_init
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import DummyPolicy

            self.policy_cls = DummyPolicy
            self.policy_factory = make_lenient(DummyPolicy)


class BrownianMotionPolicyConfig(BasePolicyConfig):
    """Policy that applies Gaussian noise increments over noop control, resulting in Brownian motion."""

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "dummy"
    std: float = 0.1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy

            self.policy_cls = BrownianMotionPolicy
            self.policy_factory = make_lenient(BrownianMotionPolicy)
