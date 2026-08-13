import heapq
import logging
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.planner.astar_planner import AStarPlanner
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.util_samplers.navgoal_sampler import NavGoalSampler
from molmo_spaces.utils.linalg_utils import normalize_ang_error

log = logging.getLogger(__name__)


"""
Port of g1_molmo's navigation policy (molmospaces/agents/policy.py in the
g1_molmo reference repo): a single live control loop that recomputes a
[vx, vy, yaw_rate] base-velocity command from the robot's *current* pose every
step. This differs structurally from AStarPlannerPolicy, which pre-bakes an
explicit rotate-then-drive waypoint schedule into the plan itself
(build_policy_plan) and leaves a downstream per-robot controller (e.g.
G1Robot._waypoint_to_velocity_target) to reinterpret each absolute pose
waypoint as a velocity command -- two independent turn-then-drive gates that
can disagree. This policy owns that logic in one place instead, and emits an
already-computed velocity under the "base_velocity" action key (see
G1Robot.update_control) rather than an absolute "base" pose waypoint.

Ported pieces (algorithmically unchanged from g1_molmo, renamed for local
style): the coarsened distance-to-wall map (_coarsen_and_dist), the
heap-based A* with a nonlinear wall-clearance cost (_astar), the
line-of-sight path simplifier (_simplify_path), and the waypoint-following
control law (_update_nav_command). Object/goal-pose selection reuses this
codebase's own NavGoalSampler/AStarPlanner.map instead of g1_molmo's
standalone OccupancyMap -- both use the same [row, col] <-> world affine
convention (occupancy True == free), so the ported grid code needs no
adaptation beyond the map it's called with.
"""

_ASTAR_COARSE_CACHE: dict = {}


def _coarsen_and_dist(occ: np.ndarray, downscale: int):
    """Cached per (occ, downscale) -- outputs depend only on these and are constant within a scene."""
    key = (occ.tobytes(), downscale, occ.shape)
    cached = _ASTAR_COARSE_CACHE.get(key)
    if cached is not None:
        return cached
    if downscale <= 1:
        coarse = occ.copy()
    else:
        d = downscale
        h, w = occ.shape
        padded = np.pad(occ, ((0, (-h) % d), (0, (-w) % d)))
        coarse = padded.reshape(padded.shape[0] // d, d, padded.shape[1] // d, d).min(1).min(-1)
    h, w = coarse.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    q = deque()
    for r in range(h):
        for c in range(w):
            if not coarse[r, c]:
                dist[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        b = dist[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > b + 1:
                dist[nr, nc] = b + 1
                q.append((nr, nc))
    if len(_ASTAR_COARSE_CACHE) > 32:
        _ASTAR_COARSE_CACHE.clear()
    _ASTAR_COARSE_CACHE[key] = (coarse, dist)
    return coarse, dist


def _astar(occ, start_rc, goal_rc, downscale=4, wall_radius=10, wall_gain=6, wall_exp=2):
    coarse, dist = _coarsen_and_dist(occ, downscale)
    h, w, d = coarse.shape[0], coarse.shape[1], downscale

    def nearest(o, rc):
        sr, sc = max(0, min(int(rc[0]), h - 1)), max(0, min(int(rc[1]), w - 1))
        if o[sr, sc]:
            return sr, sc
        q, v = deque([(sr, sc)]), {(sr, sc)}
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in v:
                    v.add((nr, nc))
                    if o[nr, nc]:
                        return nr, nc
                    q.append((nr, nc))
        return None

    sf = nearest(coarse, (int(start_rc[0] // d), int(start_rc[1] // d)))
    gf = nearest(coarse, (int(goal_rc[0] // d), int(goal_rc[1] // d)))
    if sf is None or gf is None:
        return []
    sr, sc = sf
    gr, gc = gf
    if (sr, sc) == (gr, gc):
        return [(gr * d + d // 2, gc * d + d // 2)]

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

    def wp(r, c):
        dd = dist[r, c]
        return (
            100.0
            if dd <= 0
            else (
                0.0
                if dd >= wall_radius
                else wall_gain * (1 - max(dd, 1e-3) / wall_radius) ** wall_exp
            )
        )

    open_set = [(((sr - gr) ** 2 + (sc - gc) ** 2) ** 0.5, 0.0, sr, sc)]
    gs, cf, cl = {(sr, sc): 0.0}, {}, set()
    while open_set:
        _, g, r, c = heapq.heappop(open_set)
        if (r, c) in cl:
            continue
        cl.add((r, c))
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in cf:
                r, c = cf[(r, c)]
                path.append((r, c))
            path.reverse()
            return [(p * d + d // 2, q * d + d // 2) for p, q in path]
        for (dr, dc), cost in zip(dirs, costs):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and coarse[nr, nc] and (nr, nc) not in cl:
                ng = g + cost + wp(nr, nc)
                if ng < gs.get((nr, nc), float("inf")):
                    gs[(nr, nc)] = ng
                    cf[(nr, nc)] = (r, c)
                    heapq.heappush(
                        open_set, (ng + ((nr - gr) ** 2 + (nc - gc) ** 2) ** 0.5, ng, nr, nc)
                    )
    return []


def _line_of_sight(occ, a, b, clearance=6):
    """True if the straight pixel-space segment a->b stays clear of occupied
    cells, with a `clearance`-px square margin checked at every sampled point
    along it. Used both by _simplify_path (collapsing a multi-waypoint A*
    route down to fewer straight segments) and as a standalone check for
    deciding whether a direct walk is safe without running A* at all.
    """
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for t in np.linspace(0, 1, n):
        r, c = int(round((1 - t) * a[0] + t * b[0])), int(round((1 - t) * a[1] + t * b[1]))
        for dr in range(-clearance, clearance + 1):
            for dc in range(-clearance, clearance + 1):
                if not occ[
                    max(0, min(r + dr, occ.shape[0] - 1)), max(0, min(c + dc, occ.shape[1] - 1))
                ]:
                    return False
    return True


def _simplify_path(path, occ, clearance=6):
    if len(path) <= 2:
        return path

    s = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_of_sight(occ, path[i], path[j], clearance):
            j -= 1
        s.append(path[j])
        i = j
    return s


class FetchManBasePlannerPolicy(PlannerPolicy):
    """Live single-loop nav policy ported from g1_molmo. Emits a [vx, vy, yaw_rate]
    base velocity command every step (see get_action), recomputed fresh from the
    robot's current pose. Consumers must handle the "base_velocity" action key
    (see G1Robot.update_control); robots that only understand "base" [x, y,
    theta] pose waypoints (e.g. RBY1, FloatingRUM) ignore it.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)

        self.config.policy_config.planner_config.agent_radius = (
            self.config.task_sampler_config.robot_safety_radius
        )
        self.nav_planner = AStarPlanner(
            self.config.policy_config.planner_config, self.task.env.current_model_path
        )
        self.robot_view = task.env.current_robot.robot_view

        self._nav_goal_sampler = None
        self._target_object = None
        self._candidate_objs = None
        self._skipped_candidates = set()
        self._retries_left = self.config.policy_config.plan_max_retries

        self._waypoints: list[np.ndarray] = []
        self._wp_idx = 0
        self._arrived = False
        self._facing = False
        self._target_xy = None
        self._object_xy = None
        self._has_path = False

        self._cmd = np.zeros(3, dtype=np.float32)

    def reset(self):
        self._nav_goal_sampler = None
        self._target_object = None
        self._candidate_objs = None
        self._skipped_candidates = set()
        self._retries_left = self.config.policy_config.plan_max_retries

        self._waypoints = []
        self._wp_idx = 0
        self._arrived = False
        self._facing = False
        self._target_xy = None
        self._object_xy = None
        self._has_path = False

        self._cmd = np.zeros(3, dtype=np.float32)

    def planners(self):
        return self.nav_planner

    def get_phase(self) -> str:
        if self._arrived:
            return "arrived"
        if self._facing:
            return "facing"
        if self._has_path:
            return "navigating"
        return "planning"

    def get_all_phases(self) -> dict[str | int]:
        return {"planning": 0, "navigating": 1, "facing": 2, "arrived": 3}

    @property
    def retry_count(self) -> int:
        return self.config.policy_config.plan_max_retries - self._retries_left

    def skip_candidate(self, obj_name):
        self._skipped_candidates.add(obj_name)

    @property
    def candidate_objs(self) -> list[MlSpacesObject]:
        if self._candidate_objs is None:
            batch_idx = self.task.env.current_batch_index
            self._candidate_objs = self.task.nav_objs[batch_idx]
        return self._candidate_objs

    @property
    def target_object(self) -> MlSpacesObject:
        if self._target_object is None or self._target_object.name in self._skipped_candidates:
            if len(self.candidate_objs) == 1:
                self._target_object = self.candidate_objs[0]
            else:
                batch_idx = self.task.env.current_batch_index
                priority = self.task.get_nav_object_priority(batch_idx)
                for obj in priority:
                    if obj.name not in self._skipped_candidates:
                        self._target_object = obj
                        break
                else:
                    self._target_object = priority[0] if priority else None
        return self._target_object

    @property
    def nav_goal_sampler(self) -> NavGoalSampler:
        if self._nav_goal_sampler is None:
            self._nav_goal_sampler = NavGoalSampler(
                self.nav_planner.map, check_target_in_view=False, camera_name="head_camera"
            )
        return self._nav_goal_sampler

    def _xy(self) -> np.ndarray:
        return self.robot_view.base.pose[:2, 3]

    def _yaw(self) -> float:
        return float(R.from_matrix(self.robot_view.base.pose[:3, :3]).as_euler("xyz")[2])

    def _plan_path(self) -> bool:
        """Port of g1_molmo's _plan_path: sample a standing pose near the target
        object, A* to it on the occupancy grid, and keep only the line-of-sight-
        simplified corner waypoints -- no re-densification, unlike
        AStarPlannerPolicy.interpolate_waypoints/build_policy_plan.
        _update_nav_command handles the in-between steering live."""
        cfg = self.config.policy_config
        self.nav_goal_sampler.set_target(self.target_object)
        self.nav_goal_sampler.set_robot_view(self.robot_view)

        target_pos_quat = None
        for _ in range(5):
            target_pos_quat = self.nav_goal_sampler.sample()
            if target_pos_quat is not None:
                break
        if target_pos_quat is None:
            log.info(
                "[FetchManBase PLAN ATTEMPT FAIL] NavGoalSampler failed to find valid goal position"
            )
            return False

        goal_xy = np.asarray(target_pos_quat[0][:2], dtype=np.float64)
        self._object_xy = np.asarray(self.target_object.position[:2], dtype=np.float64)
        self._target_xy = goal_xy

        occ_map = self.nav_planner.map
        start_rc = occ_map.pos_m_to_px(np.array([*self._xy(), 0.0]))
        goal_rc = occ_map.pos_m_to_px(np.array([*goal_xy, 0.0]))

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
                f"[FetchManBase PLAN ATTEMPT FAIL] A* pathfinding failed - no valid path found. "
                f"Robot pos: {tuple(np.round(self._xy(), 2))}, Target pos: {tuple(np.round(goal_xy, 2))}"
            )
            return False

        path = _simplify_path(path, occ_map.occupancy, clearance=cfg.simplify_clearance)
        pixel_waypoints = np.array(path, dtype=np.float64)
        world_waypoints = occ_map.pos_px_to_m(pixel_waypoints)[:, :2]
        self._waypoints = list(world_waypoints)
        if self._waypoints:
            self._waypoints[-1] = goal_xy

        self._wp_idx = 0
        self._has_path = True
        log.info(
            f"[FetchManBase PLAN OK] Path planned successfully with {len(self._waypoints)} waypoints"
        )
        return True

    def _update_nav_command(self):
        """Direct port of g1_molmo's _update_nav_command: recompute a
        [vx, vy, yaw_rate] command from the robot's live pose every call,
        rather than delegating to a pre-baked waypoint schedule."""
        cfg = self.config.policy_config
        if self._arrived or not self._waypoints:
            self._cmd[:] = 0
            return

        xy, yaw = self._xy(), self._yaw()

        if self._facing:
            face = self._object_xy if self._object_xy is not None else self._target_xy
            desired = np.arctan2(face[1] - xy[1], face[0] - xy[0])
            ye = normalize_ang_error(desired - yaw)
            if abs(ye) > cfg.face_tol:
                self._cmd[:] = [0, 0, np.clip(cfg.turn_kp * ye, -cfg.face_turn, cfg.face_turn)]
                return
            self._arrived = True
            self._cmd[:] = 0
            return

        wp = self._waypoints[self._wp_idx]
        if np.linalg.norm(xy - wp) < cfg.waypoint_reach and self._wp_idx < len(self._waypoints) - 1:
            self._wp_idx += 1
        wp = self._waypoints[self._wp_idx]
        delta = wp - xy
        dist = np.linalg.norm(delta)
        final = self._wp_idx >= len(self._waypoints) - 1

        # Smoothstep brake -- zero derivative at both ends so the final approach
        # "rolls" into a stop instead of stepping (see g1_molmo._update_nav_command).
        stop_dist = cfg.final_reach + cfg.stop_pad
        if final and dist <= stop_dist:
            self._facing = True
            self._cmd[:] = 0
            return

        ye = normalize_ang_error(np.arctan2(delta[1], delta[0]) - yaw)
        if abs(ye) > cfg.face_wp_tol:
            self._cmd[:] = [0, 0, np.clip(cfg.turn_kp * ye, -cfg.max_turn, cfg.max_turn)]
            return

        if final:
            if dist <= stop_dist:
                spd = 0.0
            elif dist >= cfg.brake_dist:
                spd = cfg.speed
            else:
                t = (dist - stop_dist) / (cfg.brake_dist - stop_dist)
                # Floor at min_speed, matching the non-final branch below --
                # see FetchmanPickPlannerPolicy._update_nav_command's copy of
                # this comment for the full explanation (G1Robot's
                # _VELOCITY_DEADBAND/_MIN_LINEAR_VEL gap freezes the robot
                # partway through the smoothstep's low-speed tail otherwise).
                spd = max(cfg.speed * (3 * t * t - 2 * t * t * t), cfg.min_speed)
        else:
            spd = np.clip(dist, cfg.min_speed, cfg.speed)

        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
        ln = max(np.sqrt(lx**2 + ly**2), 1e-6)
        ang = np.sign(cfg.turn_kp * ye) * np.clip(abs(cfg.turn_kp * ye), 0.05, cfg.drive_max_turn)
        self._cmd[:] = [spd * lx / ln, np.clip(spd * ly / ln, -0.5, 0.5), ang]

    def get_action(self, observation):
        if not self._has_path:
            for _ in range(max(len(self.candidate_objs) - len(self._skipped_candidates), 1)):
                if self._plan_path():
                    break
                self.skip_candidate(self.target_object.name)
                self._retries_left -= 1
                if self._retries_left <= 0:
                    break
            if not self._has_path:
                log.warning("[FetchManBase DONE] Planning failed - terminating episode")
                return self._build_done_action()

        self._update_nav_command()
        if self._arrived:
            log.info("[FetchManBase DONE] Navigation complete")
            return self._build_done_action()
        return {"done": False, "base_velocity": self._cmd.copy()}

    def _build_done_action(self):
        """Build action to signal episode completion."""
        return {**self.robot_view.get_noop_ctrl_dict(["base"]), "done": True}
