"""Interactive shell task: exposes robot skills as plain Python functions in a REPL.

Rather than pursuing one fixed goal like other tasks, `InteractiveShellTask` drops
the user into a `code.interact()` session where calling `nav_to(object=...)`,
`pick(object=...)`, etc. builds the matching single-skill task + planner policy on
the fly and runs it to completion (reusing `ParallelRolloutRunner.run_single_rollout`)
against the same persistent env/robot, so each command's effect is visible to the next.
"""

import code
import copy
import difflib
import logging
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.env.sensors import get_core_sensors
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

# nav_to()'s relative-direction shortcut: (local_dx, local_dy) in the robot's
# own base frame (+x forward, +y left -- matches rotate()'s +ccw yaw
# convention: rotating "forward" by +90deg yaw lands on "left").
_NAV_DIRECTIONS = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}


class InteractiveShellTask(BaseMujocoTask):
    """Hands control of the robot to an interactive Python shell.

    Each skill method (`nav_to`, `pick`, `pick_and_place`, `open_object`,
    `close_object`) constructs a dedicated single-skill task and planner policy
    targeting the named object, runs it to completion, and returns whether it
    succeeded. `env` (and thus robot/object state) is shared and persists
    across calls, so skills can be chained interactively.
    """

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        # Cached occupancy map for nav_to()'s A* planner; built on first use.
        self.occupancy_map: Any | None = None
        self._held_object: str | None = None

    def _create_sensor_suite_from_config(self, exp_config: MlSpacesExpConfig) -> SensorSuite:
        return SensorSuite(get_core_sensors(exp_config))

    def get_task_description(self) -> str:
        return "Interactive shell session"

    def get_reward(self) -> np.ndarray:
        return np.zeros(self._env.n_batch, dtype=np.float32)

    def judge_success(self) -> bool:
        return False

    # -- Object discovery --

    def list_objects(self, limit: int = 200) -> list[str]:
        """Print and return a human-readable summary of interactable objects in the scene."""
        om = self._env.object_managers[self._env.current_batch_index]
        summaries = om.summarize_top_level_bodies(receptacle_types=[], limit=limit)
        for line in summaries:
            print(line)
        return summaries

    def _resolve_object_name(self, name: str) -> str:
        """Resolve `name` to an exact object name in the scene.

        Every skill (`nav_to`, `pick`, `pick_and_place`, `open_object`,
        `close_object`) routes its object argument(s) through this first, so
        you can type a short/approximate name (e.g. "tomato") instead of the
        full hashed instance name (e.g. "tomato_7024...1_0_0").

        If `name` is an exact match, it's returned immediately -- no prompt.
        Otherwise, the scene object name(s) most textually similar to `name`
        (case-insensitive) are found; a tie (e.g. two instances of the same
        object category, which share an identical category prefix and so
        score identically) is broken by picking whichever is spatially
        closest to the robot. Since this is a guess, it requires confirmation
        via a y/n prompt (blocks on stdin) before use -- raises if the user
        declines. Prefix `name` with "~" (e.g. "~tomato") to skip the prompt
        and auto-accept the closest match.
        """
        auto_accept = name.startswith("~")
        if auto_accept:
            name = name[1:]

        om = self._env.object_managers[self._env.current_batch_index]
        try:
            # ObjectManager.get_object_by_name's docstring claims it returns None
            # for an unknown name, but it actually raises KeyError from mujoco's
            # own body lookup (self.model.body(name)) -- pre-existing behavior,
            # not something to paper over beyond catching it here.
            if om.get_object_by_name(name) is not None:
                return name
        except KeyError:
            pass

        candidates = om.list_top_level_objects()
        if not candidates:
            raise ValueError(f"Unknown object {name!r}. Call list_objects() to see valid names.")

        scores = {
            obj.name: difflib.SequenceMatcher(None, name.lower(), obj.name.lower()).ratio()
            for obj in candidates
        }
        best_score = max(scores.values())
        best_names = [n for n, s in scores.items() if s == best_score]

        if len(best_names) > 1:
            robot_pos = self._env.current_robot.robot_view.base.pose[:3, 3]
            by_name = {obj.name: obj for obj in candidates}
            best_names.sort(
                key=lambda n: np.linalg.norm(np.asarray(by_name[n].position[:3]) - robot_pos)
            )

        resolved = best_names[0]

        if auto_accept:
            print(
                f"Auto-accepting closest match for {name!r}: {resolved!r} (similarity {best_score:.2f})"
            )
            return resolved

        answer = (
            input(
                f"No object named {name!r}. Closest match: {resolved!r} "
                f"(similarity {best_score:.2f}). Use it? [y/N] "
            )
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            raise ValueError(f"Aborted: {name!r} not found and match not confirmed.")

        print(f"Using {resolved!r}")
        return resolved

    # -- Shared machinery --

    def _current_robot_base_pose(self) -> list[float]:
        from molmo_spaces.utils.pose import pose_mat_to_7d

        robot_view = self._env.current_robot.robot_view
        return pose_mat_to_7d(robot_view.base.pose).tolist()

    def _run_subtask(self, sub_task: BaseMujocoTask, policy_factory) -> bool:
        """Construct the policy, register it, and run `sub_task` to completion."""
        from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

        policy = policy_factory(sub_task.config, sub_task)
        sub_task.register_policy(policy)
        success = ParallelRolloutRunner.run_single_rollout(
            episode_seed=0,
            task=sub_task,
            policy=policy,
            viewer=self.viewer,
            end_on_success=True,
        )
        print(f"{'done - ' if success else 'FAILED - '}{sub_task.get_task_description()}")
        sub_task.close()
        return success

    # -- Skills --

    def nav_to(self, object: str, planner: str | None = None, dist: float = 0.25) -> bool:
        """Navigate the robot base to within range of `object`, or take a single
        `dist`-meter step in the robot's own base frame if `object` is
        "forward"/"backward"/"left"/"right" instead of an object name.

        planner: "fetchman" (FetchManBasePlannerPolicy -- the g1_molmo-ported
            live single-loop controller) or "astar" (AStarPlannerPolicy).
            Defaults to "fetchman" for G1 in WBC mode (the only consumer of
            its "base_velocity" action, see G1Robot.update_control) and
            "astar" for every other robot/mode, where "fetchman" would
            silently do nothing (no controller reads "base_velocity"). Unused
            for the forward/backward/left/right case (see _nav_to_direction).
        dist: step size in meters for the forward/backward/left/right case.
            Unused when `object` names an actual object.
        """
        if object in _NAV_DIRECTIONS:
            return self._nav_to_direction(object, dist)

        from molmo_spaces.configs.policy_configs import (
            AStarNavToObjPolicyConfig,
            FetchManBasePlannerPolicyConfig,
        )
        from molmo_spaces.configs.task_configs import NavToObjTaskConfig
        from molmo_spaces.tasks.nav_task import NavToObjTask

        if planner is None:
            planner = (
                "fetchman"
                if self.config.robot_config.name == "g1"
                and not self.config.robot_config.use_holo_base
                else "astar"
            )
        if planner not in ("astar", "fetchman"):
            raise ValueError(f"Unknown planner {planner!r}, expected 'astar' or 'fetchman'")

        object = self._resolve_object_name(object)

        if self.occupancy_map is None:
            log.info("Building occupancy map for navigation (first nav_to() call)...")
            # env.get_thormap() picks iTHORMap vs ProcTHORMap based on the scene's
            # model path (see AStarPlanner.map) - iTHOR floor plans have no "room_"
            # prefixed floor geoms, so a hardcoded ProcTHORMap fails to find a floor.
            self.occupancy_map = self._env.get_thormap(
                agent_radius=self.config.task_sampler_config.robot_safety_radius,
                px_per_m=200,
            )

        nav_config = copy.deepcopy(self.config)
        nav_config.task_type = "nav_to_obj"
        if planner == "fetchman":
            nav_config.policy_config = FetchManBasePlannerPolicyConfig()
        elif self.config.robot_config.name == "g1":
            # G1WalkController converges markedly slower than FloatingRUM's mocap-weld
            # base (see G1RobotView.is_close_to's higher default threshold for the
            # same reason). AStarPlannerPolicy's default plan_fail_after_waypoint_steps
            # (10) triggers a "failure to progress" replan before G1 has had time to
            # actually catch up on a waypoint requiring a real heading change --
            # confirmed via debug trace: convergence often takes 15-40+ steps.
            # Default waypoint spacing (path_max_inter_waypoint_dist=0.25m,
            # path_max_inter_waypoint_angle=10deg) chops the path into many small
            # move/rotate segments, each requiring a full stop-and-reconverge (see
            # is_close_to threshold above) before advancing -- the robot rarely
            # reaches cruising speed (measured ~0.5 m/s in a straight-line test)
            # before the next segment forces it to slow down and reorient again.
            # Widen the spacing so segments are long enough to actually cruise.
            nav_config.policy_config = AStarNavToObjPolicyConfig(
                plan_fail_after_waypoint_steps=50,
                plan_max_retries=5,
                path_max_inter_waypoint_dist=1.0,
                path_max_inter_waypoint_angle=np.radians(30),
            )
        else:
            nav_config.policy_config = AStarNavToObjPolicyConfig()
        nav_config.task_config = NavToObjTaskConfig(
            task_cls=NavToObjTask,
            pickup_obj_name=object,
            robot_base_pose=self._current_robot_base_pose(),
            succ_pos_threshold=0.5,  # meters (default 1.5m)
        )

        sub_task = NavToObjTask(self._env, nav_config)
        sub_task.occupancy_map = self.occupancy_map

        success = self._run_subtask(sub_task, nav_config.policy_config.policy_factory)

        robot_view = self._env.current_robot.robot_view
        target_obj = self._env.object_managers[self._env.current_batch_index].get_object_by_name(
            object
        )
        distance = float(
            np.linalg.norm(np.asarray(target_obj.position[:2]) - robot_view.base.pose[:2, 3])
        )
        print(f"Distance to {object!r}: {distance:.3f}m")

        return success

    def _nav_to_direction(
        self, direction: str, dist: float, max_ticks: int = 150, threshold: float = 0.1
    ) -> bool:
        """Step `dist` meters along `direction` in the robot's own base frame
        (see _NAV_DIRECTIONS), holding heading fixed.

        Same direct-drive, no-path-planning approach as rotate() -- a small
        fixed-distance nudge doesn't need nav_to()'s full A*/replanning
        machinery (which also requires a named object target, not a raw
        offset).
        """
        from scipy.spatial.transform import Rotation as R

        robot_view = self._env.current_robot.robot_view
        pose = robot_view.base.pose
        x, y = pose[0, 3], pose[1, 3]
        yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2]
        local_dx, local_dy = _NAV_DIRECTIONS[direction]
        world_dx = local_dx * np.cos(yaw) - local_dy * np.sin(yaw)
        world_dy = local_dx * np.sin(yaw) + local_dy * np.cos(yaw)
        target = np.array([x + dist * world_dx, y + dist * world_dy, yaw])

        for i in range(max_ticks):
            if robot_view.is_close_to(["base"], target, threshold=threshold):
                break
            self._apply_action({"base": target})
            if self.viewer is not None:
                self.viewer.sync()
            if i % 10 == 0:
                p = robot_view.base.pose
                dist_remaining = robot_view.distance_to(["base"], target)
                log.debug(
                    f"[nav_to:{direction}] i={i} pos=({p[0, 3]:.3f},{p[1, 3]:.3f}) "
                    f"dist_remaining={dist_remaining:.4f}"
                )

        success = robot_view.is_close_to(["base"], target, threshold=threshold)
        print(f"{'done' if success else 'FAILED'} - Nav {direction} {dist:.2f}m")
        return success

    def rotate(self, angle_deg: float, max_ticks: int = 150, threshold: float = 0.1) -> bool:
        """Rotate the robot base in place by `angle_deg` degrees (+ccw), holding x/y fixed.

        Bypasses nav_to()'s A* path planning entirely, driving the same "base"
        action interface (robot.update_control({"base": [x, y, theta]})) directly
        with a target heading only -- isolates whether the underlying base/WBC
        controller can turn in place at all, independent of path planning/replanning.
        """
        from scipy.spatial.transform import Rotation as R

        robot_view = self._env.current_robot.robot_view
        pose = robot_view.base.pose
        x, y = pose[0, 3], pose[1, 3]
        current_yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2]
        target = np.array([x, y, current_yaw + np.radians(angle_deg)])

        for i in range(max_ticks):
            if robot_view.is_close_to(["base"], target, threshold=threshold):
                break
            self._apply_action({"base": target})
            if self.viewer is not None:
                self.viewer.sync()
            if i % 10 == 0:
                p = robot_view.base.pose
                yaw = R.from_matrix(p[:3, :3]).as_euler("xyz")[2]
                dist = robot_view.distance_to(["base"], target)
                log.debug(
                    f"[rotate] i={i} yaw_deg={np.degrees(yaw):.2f} "
                    f"pos=({p[0, 3]:.3f},{p[1, 3]:.3f}) dist={dist:.4f}"
                )

        success = robot_view.is_close_to(["base"], target, threshold=threshold)
        print(f"{'done' if success else 'FAILED'} - Rotate by {angle_deg:.1f} deg")
        return success

    def noop(self, ticks: int = 50) -> None:
        """Step the simulation for `ticks` policy steps without commanding any motion.

        An empty action dict makes every move group's controller fall back to
        its own "hold current state" behavior (see Robot.update_control's
        set_to_stationary() fallback) -- e.g. G1's WBC keeps actively balancing
        in place rather than literally freezing. Useful for letting the robot
        settle (e.g. right after a reset/placement, or between commands) without
        driving it anywhere.
        """
        for _ in range(ticks):
            self._apply_action({})
            if self.viewer is not None:
                self.viewer.sync()

    def pick(self, object: str, planner_policy_config_cls: type | None = None) -> bool:
        """Pick up and lift `object` with the robot's gripper.

        Args:
            object: Name (or short/approximate name) of the object to pick.
            planner_policy_config_cls: Planner policy config class to use.
                Defaults to FetchmanPickPlannerPolicyConfig (mink-based,
                waist+height-assisted whole-body IK) for G1 in its default
                WBC-walking mode (use_holo_base=False) -- PickPlannerPolicy's
                arm-only analytical IK is unreliable for G1 (see
                FetchmanPickPlannerPolicy's docstring). Every other
                robot/mode defaults to PickPlannerPolicyConfig.
        """
        from molmo_spaces.configs.policy_configs import (
            FetchmanPickPlannerPolicyConfig,
            PickPlannerPolicyConfig,
        )
        from molmo_spaces.configs.task_configs import PickTaskConfig
        from molmo_spaces.tasks.pick_task import PickTask
        from molmo_spaces.utils.pose import pose_mat_to_7d

        if planner_policy_config_cls is None:
            planner_policy_config_cls = (
                FetchmanPickPlannerPolicyConfig
                if self.config.robot_config.name == "g1"
                and not self.config.robot_config.use_holo_base
                else PickPlannerPolicyConfig
            )

        object = self._resolve_object_name(object)
        om = self._env.object_managers[self._env.current_batch_index]
        pickup_obj = om.get_object_by_name(object)

        start_pose = pose_mat_to_7d(pickup_obj.pose)
        goal_pose = start_pose.copy()
        goal_pose[2] += 0.1  # lift 10cm above the start pose

        pick_config = copy.deepcopy(self.config)
        pick_config.task_type = "pick"
        pick_config.policy_config = planner_policy_config_cls()
        pick_config.task_config = PickTaskConfig(
            task_cls=PickTask,
            pickup_obj_name=object,
            robot_base_pose=self._current_robot_base_pose(),
            pickup_obj_start_pose=start_pose.tolist(),
            pickup_obj_goal_pose=goal_pose.tolist(),
        )
        pick_config.task_config.referral_expressions["pickup_obj_name"] = object

        sub_task = PickTask(self._env, pick_config)
        success = self._run_subtask(sub_task, pick_config.policy_config.policy_factory)
        if success:
            self._held_object = object
        return success

    def pick_and_place(self, object: str, receptacle: str) -> bool:
        """Pick up `object` (from its current resting pose) and place it on `receptacle`."""
        from molmo_spaces.configs.policy_configs import PickAndPlacePlannerPolicyConfig
        from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
        from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
        from molmo_spaces.utils.pose import pose_mat_to_7d

        object = self._resolve_object_name(object)
        receptacle = self._resolve_object_name(receptacle)
        om = self._env.object_managers[self._env.current_batch_index]
        pickup_obj = om.get_object_by_name(object)

        pp_config = copy.deepcopy(self.config)
        pp_config.task_type = "pick_and_place"
        pp_config.policy_config = PickAndPlacePlannerPolicyConfig()
        pp_config.task_config = PickAndPlaceTaskConfig(
            task_cls=PickAndPlaceTask,
            pickup_obj_name=object,
            place_receptacle_name=receptacle,
            robot_base_pose=self._current_robot_base_pose(),
            pickup_obj_start_pose=pose_mat_to_7d(pickup_obj.pose).tolist(),
        )
        pp_config.task_config.referral_expressions["pickup_name"] = object
        pp_config.task_config.referral_expressions["place_name"] = receptacle

        sub_task = PickAndPlaceTask(self._env, pp_config)
        success = self._run_subtask(sub_task, pp_config.policy_config.policy_factory)
        if success:
            self._held_object = None
        return success

    def _open_or_close(self, object: str, task_type: str, joint_index: int) -> bool:
        from molmo_spaces.configs.policy_configs import OpenClosePlannerPolicyConfig
        from molmo_spaces.configs.task_configs import OpeningTaskConfig
        from molmo_spaces.tasks.opening_tasks import OpeningTask

        object = self._resolve_object_name(object)

        open_config = copy.deepcopy(self.config)
        open_config.task_type = task_type
        open_config.policy_config = OpenClosePlannerPolicyConfig()
        open_config.task_config = OpeningTaskConfig(
            task_cls=OpeningTask,
            pickup_obj_name=object,
            joint_index=joint_index,
            any_inst_of_category=False,
            robot_base_pose=self._current_robot_base_pose(),
        )
        open_config.task_config.referral_expressions["pickup_obj_name"] = object

        sub_task = OpeningTask(self._env, open_config)
        return self._run_subtask(sub_task, open_config.policy_config.policy_factory)

    def open_object(self, object: str, joint_index: int = 0) -> bool:
        """Open `object` (e.g. a drawer, cabinet, or door)."""
        return self._open_or_close(object, "open", joint_index)

    def close_object(self, object: str, joint_index: int = 0) -> bool:
        """Close `object` (e.g. a drawer, cabinet, or door)."""
        return self._open_or_close(object, "close", joint_index)

    # -- Shell --

    def run_shell(self, commands: list[str] | None = None) -> None:
        """Drop into an interactive Python shell exposing robot skills as functions.

        Args:
            commands: Optional statements (e.g. 'nav_to(object="apple_...")') to run,
                in order, before handing control to the interactive prompt -- useful
                for seeding a session (or scripting/headless testing) without retyping
                setup each time. Results stay bound in the shell's namespace, so e.g.
                `result = nav_to(...)` leaves `result` available once you drop in.
        """
        banner = "\n".join(
            [
                "",
                "Interactive robot shell. Available commands:",
                "  list_objects()                        - list interactable objects in the scene",
                "  nav_to(object=name)                   - navigate to an object",
                "  nav_to(object=dir, dist=.25)           - step dist meters in the base frame;",
                "                                           dir is 'forward'/'backward'/'left'/'right'",
                "  rotate(angle_deg)                     - rotate the base in place (+ccw), no path planning",
                "  noop(ticks=50)                         - hold current position/pose, do nothing",
                "  pick(object=name)                     - pick up and lift an object",
                "  pick_and_place(object=x, receptacle=y) - pick up an object and place it on/in y",
                "  open_object(object=name)               - open a drawer/cabinet/door",
                "  close_object(object=name)              - close a drawer/cabinet/door",
                "  help()                                 - re-print this message",
                "Each command runs the robot to completion and returns True/False for success.",
                "Object names don't need to be exact -- an approximate name (e.g. 'tomato') is",
                "matched to the closest full instance name and confirmed with you before use.",
                "Prefix with '~' (e.g. '~tomato') to auto-accept the closest match, no prompt.",
                "Press Ctrl-D to exit.",
                "",
            ]
        )

        def nav_to(object, **kwargs):
            return self.nav_to(object, **kwargs)

        def rotate(angle_deg, **kwargs):
            return self.rotate(angle_deg, **kwargs)

        def noop(**kwargs):
            return self.noop(**kwargs)

        def pick(object, planner_policy_config_cls=None):
            return self.pick(object, planner_policy_config_cls)

        def pick_and_place(object, receptacle):
            return self.pick_and_place(object, receptacle)

        def open_object(object, joint_index=0):
            return self.open_object(object, joint_index)

        def close_object(object, joint_index=0):
            return self.close_object(object, joint_index)

        def list_objects():
            return self.list_objects()

        def help():
            print(banner)

        task = self
        env = self._env

        namespace = dict(globals(), **locals())
        for cmd in commands or []:
            print(f">>> {cmd}")
            exec(cmd, namespace)

        try:
            code.interact(banner=banner, local=namespace)
        except (SystemExit, KeyboardInterrupt):
            # exit()/quit() raise SystemExit, but when a passive viewer is attached
            # its background render thread can post a KeyboardInterrupt to the main
            # thread around the same time (e.g. when the viewer window is closed),
            # which code.interact()'s own loop catches and reports without exiting,
            # requiring a second exit() to actually leave. Whichever one surfaces
            # here, treat it as "the user is done" and exit the shell in one shot.
            print()
