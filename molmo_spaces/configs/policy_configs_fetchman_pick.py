"""Configuration for :class:`G1PickPlannerPolicy` (the "Fetchman" G1 pick)."""

from __future__ import annotations

import numpy as np

from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.planner.astar_planner import AStarPlannerConfig


class FetchmanPickPlannerPolicyConfig(PickPlannerPolicyConfig):
    """G1 pick with mink whole-body IK (waist + pelvis height assist the reach)
    and a walk phase. Requires G1Config's WBC-walking mode (use_holo_base=False).
    Values are g1_molmo's unless a comment says why they differ.
    """

    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    # Never abort on grip quality: g1_molmo judges success at the end by lift
    # and contact, and a thin-rimmed bowl closes the fingers near their limit
    # on a perfectly good grip.
    gripper_empty_threshold: float = float("-inf")

    # g1_molmo's GraspPolicy.LIFT (PickPlannerPolicyConfig's 0.08 is arm-only-IK tuned).
    postgrasp_z_offset: float = 0.15

    # Grasp candidates tried in cost order before giving up (g1_molmo's PATH_CHECK_K).
    grasp_candidates_to_try: int = 5

    # Low-pass alpha on the height/waist command per 5ms tick, as g1_molmo's
    # _advance_grasp. Re-derive together with IK_DECIM if the tick changes.
    height_waist_smoothing_alpha: float = 0.1

    # The WBC tracks waist/height through real dynamics, not instantly, so the
    # pregrasp approach runs at grasp speed.
    speed_fast: float = 0.08

    # g1_molmo never aborts on mid-motion tracking error; it advances a phase
    # once the gripper converges. Approximated here by a lenient error check
    # plus a settle window at the end of each move.
    tcp_pos_err_threshold: float = 1.0
    tcp_rot_err_threshold: float = np.radians(60.0)
    move_settle_time: float = 1.0

    # select_grasp_pose's signed vertical term rewards top-down grasps instead
    # of penalising them; g1_molmo picks horizontal side grasps for the same
    # candidates. Use the symmetric horizontal term (as OpenClosePlannerPolicyConfig).
    grasp_vertical_cost_weight: float = 0.0
    grasp_horizontal_cost_weight: float = 2.0

    # Walk phase: field-for-field the same as FetchManBasePlannerPolicyConfig
    # so both navigation paths share one tuning.
    planner_config: AStarPlannerConfig = AStarPlannerConfig()
    downscale: int = 4
    wall_radius: int = 10
    wall_gain: float = 6.0
    wall_exp: float = 2.0
    simplify_clearance: int = 6
    waypoint_reach: float = 0.10
    # g1_molmo's 0.05 is below the min_speed/drive_max_turn turning radius
    # (~0.5m) and left the robot orbiting the goal; the facing turn does the
    # final heading precision instead.
    final_reach: float = 0.3
    turn_kp: float = 2.0
    max_turn: float = 1.0
    face_turn: float = 1.2
    # Loosened from g1_molmo's 0.1/0.25 rad, which sit below the WBC's ~15deg
    # yaw-tracking ceiling and never converge (see FetchManBasePlannerPolicyConfig).
    face_tol: float = 0.35
    face_wp_tol: float = 0.524
    speed: float = 0.4
    min_speed: float = 0.15
    brake_dist: float = 0.70
    stop_pad: float = 0.04
    drive_max_turn: float = 0.3
    # Walk-goal standoff from the object (NavGoalSampler's distance_threshold).
    walk_goal_distance_threshold: float = 0.5

    # Within this distance of the goal, skip A* and count the robot as arrived:
    # the sampler may spawn it inside the costmap's wall inflation, where A*
    # finds no path although it stands at a valid standoff.
    direct_arrival_max_dist: float = 1.2

    # G1PickPlannerPolicy (the reference stack) samples its own walk goal on
    # this annulus around the object and plans on a map inflated by
    # nav_map_extra_inflation; the walk_* fields above do not reach it.
    # Narrowed from g1_molmo's (0.25, 0.80): measured on a table bowl, grasps
    # from 0.42m fold the arm and shove the object, from 0.60m+ the arm cannot
    # reach; 0.46-0.50m lifted it every time. The policy walks out to this
    # annulus whenever it starts outside it.
    goal_standoff_radius_range: tuple[float, float] = (0.45, 0.58)
    # Walk to a fresh standoff even when already inside the annulus
    # (InteractiveShellTask.pick sets it on a retry).
    force_standoff_walk: bool = False
    # G1Controller.direct_walk: prune waypoints within reach and drive short
    # forward goals holonomically instead of turn / drive / turn. Off by default
    # so the reference walk stays byte-identical to gold; the interactive shell
    # turns it on, where every in-place WBC turn costs 0.2-0.5m of drift.
    direct_walk: bool = False
    nav_map_extra_inflation: float = 0.125
    goal_sampling_attempts: int = 25
    # The walk timeout is raised to fit the planned path
    # (path_length / speed * slack), never below walk_timeout_s.
    walk_timeout_s: float = 20.0
    walk_timeout_slack: float = 2.0

    def model_post_init(self, __context) -> None:
        # Skip PickPlannerPolicyConfig's, which would set policy_cls to PickPlannerPolicy.
        super(PickPlannerPolicyConfig, self).model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import (
                G1PickPlannerPolicy,
            )

            self.policy_cls = G1PickPlannerPolicy
            self.policy_factory = G1PickPlannerPolicy
