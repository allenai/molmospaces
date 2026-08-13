"""Task sampler for direct (non-interactive) G1 pick tasks.

A plain PickTaskSampler for G1 -- no place target, no receptacle-size
filtering, unlike InteractiveShellTaskSampler (which always runs the full
pick-and-place pipeline even when only pick() is used; see
InteractiveShellTaskSampler._filter_place_target for why that's a poor fit
once pickup_types is restricted to one category).

Exists to be configured to reproduce g1_molmo's own working spawn setup
(~/code/g1_molmo/molmospaces/configs/bowl_mixed_grasponly.py's GRASP_PROFILE,
spawn_at_grasp=True): a moderate standoff annulus around the pickup object
(there, grasp_spawn_radius_min=0.2, grasp_spawn_radius_max=0.5) with many
placement retries (there, 25 annulus-sample attempts), rather than
InteractiveShell's tight (0, 0.4) disk-from-center range with only 10
retries -- which finds zero free occupancy-map cells far more often in
cluttered procthor scenes. See PickG1DataGenConfig for the actual
radius/retry values.
"""

import logging

import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.planner.astar_planner import AStarPlanner
from molmo_spaces.policy.solvers.navigation.fetchman_base_planner_policy import (
    _astar,
    _line_of_sight,
)
from molmo_spaces.tasks.pick_g1_task import PickG1Task
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.util_samplers.navgoal_sampler import NavGoalSampler

log = logging.getLogger(__name__)


class PickG1TaskSampler(PickTaskSampler):
    def _task_cls(self) -> type[PickTask]:
        return PickG1Task

    def _ensure_nav_reachability_checker(self, env: CPUMujocoEnv) -> None:
        """Lazily build (and cache per-house) the same AStarPlanner/
        NavGoalSampler pairing FetchmanPickPlannerPolicy builds for its own
        walk-phase planning -- reused here by _check_placement_walk_reachable
        to validate a placement against the exact machinery that will later
        have to walk from it, instead of an independent guess.
        """
        if getattr(self, "_nav_reachability_env", None) is env:
            return
        planner_config = self.config.policy_config.planner_config
        self._nav_reachability_planner = AStarPlanner(planner_config, env.current_model_path)
        self._nav_reachability_sampler = NavGoalSampler(
            self._nav_reachability_planner.map,
            check_target_in_view=False,
            camera_name="head_camera",
        )
        self._nav_reachability_env = env

    def _check_placement_walk_reachable(self, env: CPUMujocoEnv, pickup_obj_name: str) -> bool:
        """Reproduces g1_molmo's single-source-of-truth spawn guarantee
        (env computes one standoff point and spawns the robot directly
        there, so start==goal by construction) as a post-hoc check instead:
        reject this placement if FetchmanPickPlannerPolicy's own walk-
        planning (NavGoalSampler + direct-arrival/line-of-sight + A*) would
        also fail from here, so an unreachable (robot placement, walk goal)
        combination gets a fresh, independently-resampled placement instead
        of only being discovered as a "Walk-path planning failed" episode
        abort once policy.reset() runs.
        """
        if env.current_robot.controllers.get("legs_waist") is None:
            return True  # holo-base / non-walking robots have no walk phase to verify

        self._ensure_nav_reachability_checker(env)
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(pickup_obj_name)
        robot_view = env.current_robot.robot_view
        self._nav_reachability_sampler.set_target(pickup_obj)
        self._nav_reachability_sampler.set_robot_view(robot_view)

        cfg = self.config.policy_config
        target_pos_quat = None
        for _ in range(5):
            target_pos_quat = self._nav_reachability_sampler.sample(
                distance_threshold=cfg.walk_goal_distance_threshold
            )
            if target_pos_quat is not None:
                break
        if target_pos_quat is None:
            log.info(f"[PickG1TaskSampler] no standoff point found for {pickup_obj_name}")
            return False

        goal_xy = np.asarray(target_pos_quat[0][:2], dtype=np.float64)
        robot_xy = robot_view.base.pose[:2, 3]
        occ_map = self._nav_reachability_planner.map
        start_rc = occ_map.pos_m_to_px(np.array([*robot_xy, 0.0]))
        goal_rc = occ_map.pos_m_to_px(np.array([*goal_xy, 0.0]))

        if np.linalg.norm(robot_xy - goal_xy) <= cfg.direct_arrival_max_dist and _line_of_sight(
            occ_map.occupancy, start_rc, goal_rc, clearance=cfg.simplify_clearance
        ):
            return True

        path = _astar(
            occ_map.occupancy,
            start_rc,
            goal_rc,
            downscale=cfg.downscale,
            wall_radius=cfg.wall_radius,
            wall_gain=cfg.wall_gain,
            wall_exp=cfg.wall_exp,
        )
        if not path:
            log.info(
                f"[PickG1TaskSampler] placement not walk-reachable for {pickup_obj_name}: "
                f"robot={tuple(np.round(robot_xy, 2))} goal={tuple(np.round(goal_xy, 2))}"
            )
        return bool(path)
