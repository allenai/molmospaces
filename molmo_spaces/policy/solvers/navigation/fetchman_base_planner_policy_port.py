"""FetchManBasePlannerPolicyPort: FetchManBasePlannerPolicy driven by
G1PickAgent's grid helpers, constants and control law, so `nav_to` walks the
way a pick's own walk phase does. The A*/simplify helpers are imported from
pick_planner_policy_g1, not copied. Differences from the base class, all to match
G1PickAgent: no min_speed floor on the final brake, `_face_yaw_offset` is
honoured, the base command bypasses G1Robot's velocity floor (see get_action),
and the goal can be a grasping standoff (see _sample_goal).
"""

import logging

import numpy as np

from molmo_spaces.policy.solvers.navigation.fetchman_base_planner_policy import (
    FetchManBasePlannerPolicy,
)

# The SAME function objects the pick walk phase uses -- not copies.
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy_g1 import (
    HOLONOMIC_MAX_START_DIST,
    _astar,
    _simplify_path,
    goal_in_forward_cone,
    holonomic_cmd,
    prune_waypoints,
    sample_standoff_pose,
    trace_log,
)

log = logging.getLogger(__name__)


class FetchManBasePlannerPolicyPort(FetchManBasePlannerPolicy):
    """`FetchManBasePlannerPolicy` with pick_planner_policy_g1's grid code and control law."""

    # G1PickAgent's own nav constants (pick_planner_policy_g1.py, class body).
    WAYPOINT_REACH = 0.10
    FINAL_REACH = 0.05
    SPEED = 0.3
    MIN_SPEED = 0.3
    TURN_KP = 2.0
    MAX_TURN = 1.0
    FACE_TURN = 1.2
    FACE_TOL = 0.1
    FACE_WP_TOL = 0.25
    BRAKE_DIST = 0.70
    STOP_PAD = 0.04

    def _standoff_maps(self):
        """(goal-sampling map, A*-planning map) exactly as G1PickPlannerPolicy.
        _nav_maps builds them, so a standoff goal chosen here is one the pick
        would have chosen, and reachable on the map the pick plans on."""
        cfg = self.config.policy_config
        env = self.task.env
        occ = env.get_occupancy_map(
            agent_radius=self.config.task_sampler_config.robot_safety_radius, px_per_m=200
        )
        if getattr(self, "_standoff_occ_for", None) is not occ:
            self._standoff_occ_for = occ
            self._standoff_nav_occ = occ.dilated(cfg.standoff_map_extra_inflation)
        return occ, self._standoff_nav_occ

    def _sample_goal(self):
        """Returns (goal_xy, planning map) or (None, None).

        With `standoff_radius_range` set: a grasping standoff on the pick's
        annulus, via the pick's own sampler and maps. Otherwise the base
        class's NavGoalSampler goal on the AStarPlanner map.
        """
        cfg = self.config.policy_config
        if cfg.standoff_radius_range is not None:
            occ, nav_occ = self._standoff_maps()
            r_min, r_max = cfg.standoff_radius_range
            # Same seeding as G1PickPlannerPolicy.reset, so nav_to and a
            # following pick agree on what "the" standoff sample is.
            rng = np.random.default_rng(getattr(self.task, "episode_seed", 0) or 0)
            goal_xy, _ = sample_standoff_pose(
                occ,
                nav_occ,
                np.asarray(self.target_object.position[:2], dtype=np.float64),
                self._xy(),
                r_min,
                r_max,
                rng,
                here_yaw=self._yaw(),
            )
            if goal_xy is None:
                log.info(
                    "[FetchManBasePort PLAN FAIL] no reachable standoff pose on the "
                    f"{r_min:.2f}-{r_max:.2f}m annulus around {self.target_object.name!r}"
                )
                return None, None
            return goal_xy, nav_occ

        self.nav_goal_sampler.set_target(self.target_object)
        self.nav_goal_sampler.set_robot_view(self.robot_view)
        target_pos_quat = None
        for _ in range(5):
            target_pos_quat = self.nav_goal_sampler.sample()
            if target_pos_quat is not None:
                break
        if target_pos_quat is None:
            log.info("[FetchManBasePort PLAN FAIL] NavGoalSampler found no valid goal position")
            return None, None
        return np.asarray(target_pos_quat[0][:2], dtype=np.float64), self.nav_planner.map

    def _plan_path(self) -> bool:
        """As the base class, but A*/simplify through pick_planner_policy_g1's copies,
        with an optional grasping-standoff goal (see _sample_goal)."""
        cfg = self.config.policy_config
        goal_xy, occ_map = self._sample_goal()
        if goal_xy is None:
            return False

        self._object_xy = np.asarray(self.target_object.position[:2], dtype=np.float64)
        self._target_xy = goal_xy

        # _world_to_px/map_to_world: the pair both ProcTHORMap/iTHORMap and
        # AABBMap expose, and the one G1PickAgent._plan_path indexes with.
        start_rc = occ_map._world_to_px(self._xy())
        goal_rc = occ_map._world_to_px(goal_xy)

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
                f"[FetchManBasePort PLAN FAIL] A* found no path. "
                f"Robot {tuple(np.round(self._xy(), 2))} -> goal {tuple(np.round(goal_xy, 2))}"
            )
            return False

        path = _simplify_path(path, occ_map.occupancy, clearance=cfg.simplify_clearance)
        self._waypoints = [(occ_map.map_to_world @ np.array([r, c, 1.0]))[:2] for r, c in path]
        if self._waypoints:
            self._waypoints[-1] = goal_xy
        self._waypoints = prune_waypoints(self._waypoints, self._xy())
        # As G1PickAgent._plan_path: a short single-segment walk is driven
        # holonomically (see pick_planner_policy_g1.holonomic_cmd).
        self._holo_final = (
            len(self._waypoints) == 1
            and float(np.linalg.norm(self._waypoints[0] - self._xy())) < HOLONOMIC_MAX_START_DIST
            and goal_in_forward_cone(self._xy(), self._yaw(), self._waypoints[0])
        )

        self._wp_idx = 0
        self._has_path = True
        self._trace_mode = None
        log.info(
            f"[FetchManBasePort PLAN OK] from {tuple(np.round(self._xy(), 2))} to "
            f"{tuple(np.round(goal_xy, 2))} via {len(self._waypoints)} waypoints: "
            f"{[tuple(np.round(w, 2)) for w in self._waypoints]}"
        )
        return True

    def _update_nav_command(self):
        """G1PickAgent._update_nav_command, statement for statement.

        Only the command sink differs: G1PickAgent writes the WBC controller's
        `_low_level._cmd` directly, while a policy emits `self._cmd` as the
        "base_velocity" action (see the base class's get_action).
        """
        if self._arrived:
            self._cmd[:] = 0
            return
        if not self._waypoints:
            self._cmd[:] = 0
            return
        xy, yaw = self._xy(), self._yaw()
        if self._facing:
            face = self._object_xy if self._object_xy is not None else self._target_xy
            if face is not None:
                desired = np.arctan2(face[1] - xy[1], face[0] - xy[0]) + getattr(
                    self, "_face_yaw_offset", 0.0
                )
                ye = (desired - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(ye) > self.FACE_TOL:
                    self._trace("facing object", xy, yaw)
                    self._cmd[:] = [
                        0,
                        0,
                        np.clip(self.TURN_KP * ye, -self.FACE_TURN, self.FACE_TURN),
                    ]
                    return
            self._arrived = True
            self._trace("arrived", xy, yaw)
            self._cmd[:] = 0
            return
        wp = self._waypoints[self._wp_idx]
        if (
            np.linalg.norm(xy - wp) < self.WAYPOINT_REACH
            and self._wp_idx < len(self._waypoints) - 1
        ):
            self._wp_idx += 1
        wp = self._waypoints[self._wp_idx]
        delta = wp - xy
        dist = np.linalg.norm(delta)
        final = self._wp_idx >= len(self._waypoints) - 1
        # Smoothstep brake -- zero derivative at both ends so we "roll" into a stop
        # instead of stepping. Longer runway (0.70 m) gives ~2.3s of decel at
        # 0.3 m/s -- feels natural in real-life walking.
        stop_dist = self.FINAL_REACH + self.STOP_PAD
        # Arrive at stop_dist (where the brake zeros speed), else robot hangs in the
        # dead zone short of FINAL_REACH.
        if final and dist <= stop_dist:
            self._facing = True
            self._trace("at final waypoint", xy, yaw)
            self._cmd[:] = 0
            return
        if final and getattr(self, "_holo_final", False):
            self._trace("holonomic hop to goal", xy, yaw)
            face = self._object_xy if self._object_xy is not None else wp
            self._cmd[:] = holonomic_cmd(
                xy, yaw, wp, face, self.SPEED, stop_dist, self.BRAKE_DIST, self.TURN_KP
            )
            return
        ye = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(ye) > self.FACE_WP_TOL:
            self._trace(f"turning to waypoint {self._wp_idx}", xy, yaw)
            self._cmd[:] = [
                0,
                0,
                np.clip(self.TURN_KP * ye, -self.MAX_TURN, self.MAX_TURN),
            ]
            return
        self._trace(f"driving to waypoint {self._wp_idx}", xy, yaw)
        if final:
            if dist <= stop_dist:
                spd = 0.0
            elif dist >= self.BRAKE_DIST:
                spd = self.SPEED
            else:
                t = (dist - stop_dist) / (self.BRAKE_DIST - stop_dist)
                spd = self.SPEED * (3 * t * t - 2 * t * t * t)
        else:
            spd = np.clip(dist, self.MIN_SPEED, self.SPEED)
        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
        ln = max(np.sqrt(lx**2 + ly**2), 1e-6)
        ang = np.sign(self.TURN_KP * ye) * np.clip(abs(self.TURN_KP * ye), 0.05, self.MAX_TURN)
        self._cmd[:] = [spd * lx / ln, np.clip(spd * ly / ln, -0.5, 0.5), ang]

    def _trace(self, mode, xy, yaw):
        """Log the walk's control mode whenever it changes (diagnostics only)."""
        if mode == getattr(self, "_trace_mode", None):
            return
        self._trace_mode = mode
        trace_log.info(
            "[FetchManBasePort] t=%.2fs %s at %s yaw %.2f",
            float(self.task.env.current_data.time),
            mode,
            np.round(xy, 2),
            yaw,
        )

    def get_action(self, observation):
        """Hand G1 the command as a `legs_waist` target, bypassing the 0.15 m/s
        floor G1Robot.update_control applies to `base_velocity`; floored, the
        brake cannot roll to a stop and the robot overshoots the goal."""
        action = super().get_action(observation)
        base_velocity = action.pop("base_velocity", None)
        robot = self.task.env.current_robot
        if base_velocity is not None and hasattr(robot, "_legs_waist_target"):
            action["legs_waist"] = robot._legs_waist_target(base_velocity)
        elif base_velocity is not None:
            action["base_velocity"] = base_velocity
        return action
