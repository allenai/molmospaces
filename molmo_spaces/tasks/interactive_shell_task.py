"""Interactive shell task: exposes robot skills as plain Python functions in a REPL.

Rather than pursuing one fixed goal like other tasks, `InteractiveShellTask` drops
the user into a `code.interact()` session where calling `nav_to(object=...)`,
`pick(object=...)`, etc. builds the matching single-skill task + planner policy on
the fly and runs it to completion (reusing `ParallelRolloutRunner.run_single_rollout`)
against the same persistent env/robot, so each command's effect is visible to the next.

Known-good sequence (G1, InteractiveShellG1 config, house 1)::

    PYTHONPATH=. mjpython scripts/datagen/run_pipeline.py --config InteractiveShellG1 --viewer --house_inds 1

    >>> nav_to("right", dist=.3)   # or "left"/"forward(s)"/"backward(s)"
    >>> pick("~bowl")

The side-step puts the bowl in front of the right hand at a workable standoff;
`pick` then walks itself onto its 0.45-0.58m standoff annulus if needed, grasps,
and retries from a fresh standoff on a miss. `nav_to(B)` with the exact bowl
name followed by `pick(B)` also works; note that `~bowl` resolves to the
*nearest* "bowl"-labelled object, which from the spawn is the place receptacle.

`nav_to` also reaches doors and windows (`nav_to("~door")`), which the
ObjectManager classes as structural and so hides from every other candidate
list; see `InteractiveShellTask.NAV_STRUCTURAL_TYPES`.
"""

import atexit
import code
import copy
import difflib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.env import BaseMujocoEnv
from molmo_spaces.env.sensors import get_core_sensors
from molmo_spaces.policy.solvers.navigation.astar_planner_policy import split_by_max_dist
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

# nav_to()'s relative-direction shortcut: (local_dx, local_dy) in the robot's
# own base frame (+x forward, +y left -- matches rotate()'s +ccw yaw
# convention: rotating "forward" by +90deg yaw lands on "left").
_NAV_DIRECTIONS = {
    "forward": (1.0, 0.0),
    "forwards": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "backwards": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}

# Where the shell's command history is persisted across sessions (override with
# MOLMO_SPACES_SHELL_HISTORY).
_HISTORY_FILE = Path(
    os.environ.get(
        "MOLMO_SPACES_SHELL_HISTORY",
        Path.home() / ".cache" / "molmospaces" / "interactive_shell_history",
    )
).expanduser()
_HISTORY_LENGTH = 10000


def _setup_readline_history(namespace: dict[str, Any]):
    """Load `_HISTORY_FILE` into readline, wire tab completion against
    `namespace`, and return a "flush history to disk" callable (a no-op without
    readline; a missing history is never a failed shell)."""
    try:
        import readline
        import rlcompleter
    except ImportError:  # readline is optional (e.g. bare Windows)
        log.debug("readline unavailable; interactive shell history disabled")
        return lambda: None

    readline.set_completer(rlcompleter.Completer(namespace).complete)
    # libedit (macOS' stock readline) spells the completion binding differently.
    if "libedit" in (getattr(readline, "__doc__", "") or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _HISTORY_FILE.exists():
            readline.read_history_file(str(_HISTORY_FILE))
    except OSError as e:
        log.warning(f"Could not read shell history from {_HISTORY_FILE}: {e}")

    readline.set_history_length(_HISTORY_LENGTH)

    def save_history():
        try:
            readline.write_history_file(str(_HISTORY_FILE))
        except OSError as e:
            log.warning(f"Could not write shell history to {_HISTORY_FILE}: {e}")

    atexit.register(save_history)
    return save_history


class InteractiveShellTask(BaseMujocoTask):
    """Hands control of the robot to an interactive Python shell.

    Each skill method (`nav_to`, `pick`, `pick_and_place`, `open_object`,
    `close_object`) constructs a dedicated single-skill task and planner policy
    targeting the named object, runs it to completion, and returns whether it
    succeeded. `env` (and thus robot/object state) is shared and persists
    across calls, so skills can be chained interactively.
    """

    # Categories that ObjectManager.STRUCTURAL_TYPES hides from
    # list_top_level_objects() -- and so from every sampler's candidate pool --
    # but that are still worth walking up to from the shell. Structural targets
    # are nav-only: they have no free joint, so pick/place never sees them.
    NAV_STRUCTURAL_TYPES = frozenset({"door", "doorway", "doorframe", "window"})

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        # Cached occupancy map for nav_to()'s A* planner; built on first use.
        self.occupancy_map: Any | None = None
        self._held_object: str | None = None
        self._last_subtask_info: dict = {}
        # Named sim states saved by snapshot(), replayed by restore().
        self._snapshots: dict[str, tuple[np.ndarray, str | None]] = {}
        # Objects the robot was already in contact with as of the last command,
        # so _warn_new_collisions() reports only transitions. Populated lazily.
        self._collisions: set[str] = set()
        self._robot_geoms: set[int] | None = None
        # Scanning contacts costs ~0.5ms in a furnished scene (~370 contacts),
        # against a ~36ms policy tick -- so a per-tick check is only ~1% of run
        # time, but there's no reason to pay it that often. Poll at most once
        # per this many seconds of simulated time; set to 0.0 for every tick.
        self.collision_check_interval: float = 1.0
        self._last_collision_check: float = -np.inf
        # Direct-drive commands (nav_to(direction), rotate, noop, grasp/release)
        # poll the sensor suite once per policy tick and append the step to the
        # task's caches, exactly as a planner rollout does -- their trajectories
        # are then BC training data on the same footing as a planned skill's.
        # Set False for an exploratory session where the sensor cost isn't worth it.
        self.record_direct_drive: bool = True
        # Observation from the last recorded direct-drive tick, for inspection.
        self.last_observation: list[dict[str, Any]] | None = None

    def _create_sensor_suite_from_config(self, exp_config: MlSpacesExpConfig) -> SensorSuite:
        return SensorSuite(get_core_sensors(exp_config))

    def get_task_description(self) -> str:
        return "Interactive shell session"

    def get_reward(self) -> np.ndarray:
        return np.zeros(self._env.n_batch, dtype=np.float32)

    def judge_success(self) -> bool:
        return False

    # -- Object discovery --

    def list_objects(
        self, limit: int = 200, dist: float | None = None, structural: bool = True
    ) -> list[str]:
        """Print and return a human-readable summary of interactable objects in the scene.

        Args:
            limit: Maximum number of objects to report.
            dist: If given, only report objects whose center is within `dist` meters
                of the robot base, closest first, with the distance appended to each
                line. Otherwise all objects are reported, ordered by name.
            structural: Also report the nav-only structural targets (doors,
                doorframes, windows) that `nav_to` can reach but no other skill
                can act on; they are suffixed "(nav-only)".
        """
        om = self._env.object_managers[self._env.current_batch_index]
        nav_only = self._nav_structural_objects(om) if structural else []
        nav_only_names = {obj.name for obj in nav_only}

        def summarize(name: str) -> str:
            line = om.object_summary_str(name, receptacle_types=[])
            return f"{line} (nav-only)" if name in nav_only_names else line

        if dist is None:
            summaries = om.summarize_top_level_bodies(receptacle_types=[], limit=limit)
            summaries += [summarize(obj.name) for obj in nav_only][: max(0, limit - len(summaries))]
        else:
            robot_pos = self._env.current_robot.robot_view.base.pose[:3, 3]
            near: list[tuple[float, str]] = []
            for obj in om.list_top_level_objects() + nav_only:
                d = float(np.linalg.norm(np.asarray(obj.position[:3]) - robot_pos))
                if d <= dist:
                    near.append((d, obj.name))
            near.sort()
            summaries = [f"{summarize(name)} [{d:.2f}m]" for d, name in near[:limit]]

        for line in summaries:
            print(line)
        return summaries

    def _nav_structural_objects(self, om: Any) -> list[Any]:
        """Top-level bodies that `list_top_level_objects()` drops as structural
        but whose category is in `NAV_STRUCTURAL_TYPES`, e.g. the
        `doorway_<hash>_...` bodies ProcTHOR emits for doors and doorframes.

        Kept separate from the ObjectManager's own listing so the samplers'
        candidate pools are unaffected -- this only widens what the shell will
        resolve a name against.
        """
        objs: list[Any] = []
        for b in om.top_level_bodies():
            name = om.get_object_name(b)
            if not name or om.is_excluded(name) or not om.is_structural(name):
                continue
            if om.category_from_name(name) not in self.NAV_STRUCTURAL_TYPES:
                continue
            try:
                objs.append(om.get_object_by_name(name))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Same tolerance as ObjectManager.summarize_top_level_bodies:
                # a body that won't build into an object is skipped, not fatal.
                log.debug(f"Skipping structural nav candidate {name!r}: {e}")
        return sorted(objs, key=lambda o: o.name)

    def _resolve_object_name(self, name: str) -> str:
        """Resolve a short or approximate `name` ("tomato") to an exact object
        name. Exact matches return at once; otherwise the best `_match_score`
        wins, ties going to the object closest to the robot, and the guess is
        confirmed on stdin unless `name` starts with "~".
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

        # Doors et al are structural, so they never reach list_top_level_objects();
        # add them back here or "door" can only ever fuzzy-match furniture.
        candidates = om.list_top_level_objects() + self._nav_structural_objects(om)
        if not candidates:
            raise ValueError(f"Unknown object {name!r}. Call list_objects() to see valid names.")

        query = name.lower()
        robot_pos = self._env.current_robot.robot_view.base.pose[:3, 3]
        scored = [
            (
                self._match_score(query, self._object_labels(om, obj)),
                float(np.linalg.norm(np.asarray(obj.position[:3]) - robot_pos)),
                obj.name,
            )
            for obj in candidates
        ]
        best_score = max(s for s, _, _ in scored)
        # Highest score first, then closest to the robot -- with category-level
        # scoring every instance of the queried category ties at the top, so
        # this is what actually decides between them.
        tied = sorted((d, n) for s, d, n in scored if s == best_score)
        dist, resolved = tied[0]

        detail = f"similarity {best_score:.2f}, {dist:.2f}m away"
        if len(tied) > 1:
            detail += f", closest of {len(tied)} equally-good matches"

        if auto_accept:
            print(f"Auto-accepting closest match for {name!r}: {resolved!r} ({detail})")
            return resolved

        answer = (
            input(
                f"No object named {name!r}. Closest match: {resolved!r} ({detail}). Use it? [y/N] "
            )
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            raise ValueError(f"Aborted: {name!r} not found and match not confirmed.")

        print(f"Using {resolved!r}")
        return resolved

    @staticmethod
    def _object_labels(om: Any, obj: Any) -> set[str]:
        """The lowercase strings an object should be matchable by.

        Its category with the 32-char asset hash stripped (`category_from_name`,
        e.g. "plate"), its annotated category (e.g. "Bowl" -> "bowl", which for
        a `place_receptacle/...` instance is the only informative label), and the
        full instance name so typing a long partial name still works.
        """
        labels = {obj.name.lower(), om.category_from_name(obj.name).lower()}
        try:
            labels.add(om.get_annotation_category(obj).lower())
        except KeyboardInterrupt:
            raise
        except Exception:
            # Annotation lookup goes to object metadata that not every instance
            # has; the name-derived labels are enough on their own.
            pass
        return {label for label in labels if label}

    @staticmethod
    def _match_score(query: str, labels: set[str]) -> float:
        """Best match of `query` against any of `labels`, in [0, 1]. Tiered:
        exact, then substring (a prefix scores no higher, so "table" ties
        `tablelamp` with `diningtable` and distance decides), then fuzzy. A raw
        difflib ratio let the 32-char asset hash dominate and never tied.
        """
        best = 0.0
        for label in labels:
            if label == query:
                score = 1.0
            elif query in label or label in query:
                score = 0.85
            else:
                score = 0.7 * difflib.SequenceMatcher(None, query, label).ratio()
            best = max(best, score)
        return best

    # -- Inspection --

    def _gripper_move_groups(self) -> list[tuple[str, Any]]:
        """(move_group_id, GripperGroup) for every gripper this robot has.

        Empty for a robot with no gripper move group -- callers report that
        rather than raising, so `where()`/`contacts()` still work on a base-only
        robot.
        """
        robot_view = self._env.current_robot.robot_view
        return [
            (mg_id, robot_view.get_gripper(mg_id))
            for mg_id in robot_view.get_gripper_movegroup_ids()
        ]

    def _base_xy_yaw(self) -> tuple[float, float, float]:
        pose = self._env.current_robot.robot_view.base.pose
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        return float(pose[0, 3]), float(pose[1, 3]), yaw

    def _robot_geom_ids(self) -> set[int]:
        """Every geom belonging to the robot, cached (the model never changes)."""
        from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

        if self._robot_geoms is None:
            robot_view = self._env.current_robot.robot_view
            self._robot_geoms = set(
                descendant_geoms(
                    robot_view.mj_model, robot_view.base.root_body_id, visible_only=False
                )
            )
        return self._robot_geoms

    def _robot_object_collisions(self) -> dict[str, tuple[str, float]]:
        """Current robot-vs-object collisions, {object root body: (robot link,
        deepest penetration)}. Robot self-contacts and the floor are excluded;
        the floor is matched by geom name, since its geom hangs off the world
        body and a body-name check would count every footfall.
        """
        robot_view = self._env.current_robot.robot_view
        model, data = robot_view.mj_model, robot_view.mj_data
        robot_geoms = self._robot_geom_ids()

        worst: dict[str, tuple[str, float]] = {}
        for cid in range(data.ncon):
            c = data.contact[cid]
            if c.dist > 0:  # proximity record, not an actual touch
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            in1, in2 = g1 in robot_geoms, g2 in robot_geoms
            if in1 == in2:  # neither is the robot, or both are (self-collision)
                continue
            robot_geom, other_geom = (g1, g2) if in1 else (g2, g1)

            body = model.body(model.geom_bodyid[other_geom]).name or ""
            root = model.body(model.body_rootid[model.geom_bodyid[other_geom]]).name or ""
            geom_name = model.geom(other_geom).name or ""
            if "floor" in f"{geom_name} {body} {root}".lower():
                continue
            # Objects are their own root body; scene fixtures (walls, etc.) hang
            # off the world body, where the geom name is the only useful label.
            other = root if root and root != "world" else (geom_name or body or f"geom{other_geom}")

            link = model.body(model.geom_bodyid[robot_geom]).name or f"geom{robot_geom}"
            depth = -float(c.dist)
            if other not in worst or depth > worst[other][1]:
                worst[other] = (link, depth)
        return worst

    def _warn_new_collisions(self, rebaseline: bool = False) -> None:
        """Warn once per object per command about new robot collisions.
        Throttled to `collision_check_interval` of simulated time between
        checks; `rebaseline=True` (the end of a command) always checks and
        re-syncs the reported set, so a flickering contact does not re-warn
        every tick and a contact carried into the next command stays quiet.
        """
        now = float(self._env.current_robot.robot_view.mj_data.time)
        if not rebaseline:
            elapsed = now - self._last_collision_check
            # elapsed < 0 means restore() wound the clock back; treat as due.
            if 0.0 <= elapsed < self.collision_check_interval:
                return
        self._last_collision_check = now

        current = self._robot_object_collisions()
        for other, (link, depth) in sorted(current.items()):
            if other not in self._collisions:
                print(
                    f"!! COLLISION: robot ({link}) hit {other} (penetration {depth * 1000:.1f}mm)"
                )
                self._collisions.add(other)
        if rebaseline:
            self._collisions = set(current)

    def where(self) -> dict[str, Any]:
        """Print (and return) the robot's own state: base pose, gripper pose, grasp state.

        Complements `list_objects(dist=...)`, which reports the scene from the
        *base*'s point of view -- this reports the gripper, which is what
        actually has to reach an object.
        """
        robot_view = self._env.current_robot.robot_view
        x, y, yaw = self._base_xy_yaw()
        z = float(robot_view.base.pose[2, 3])
        print(f"base: x={x:.3f} y={y:.3f} z={z:.3f} yaw={np.degrees(yaw):.1f}deg")

        info: dict[str, Any] = {"base": (x, y, z, yaw), "grippers": {}}
        for mg_id, gripper in self._gripper_move_groups():
            ee = robot_view.get_move_group(mg_id).leaf_frame_to_world
            ee_pos = ee[:3, 3]
            closed_dist, open_dist = gripper.inter_finger_dist_range
            print(
                f"{mg_id}: ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) "
                f"fingers={gripper.inter_finger_dist:.4f}m "
                f"(closed={closed_dist:.4f} open={open_dist:.4f}) "
                f"{'OPEN' if gripper.is_open else 'CLOSED'}"
            )
            info["grippers"][mg_id] = {
                "ee_pos": ee_pos.copy(),
                "inter_finger_dist": gripper.inter_finger_dist,
                "is_open": gripper.is_open,
            }
        if not info["grippers"]:
            print("(robot has no gripper move group)")

        print(f"held object: {self._held_object!r}")
        info["held_object"] = self._held_object
        return info

    def whereis(self, object: str) -> dict[str, Any]:
        """Print (and return) where `object` is relative to the robot.

        Reports the object's world position, its distance from the base (xy) and
        from each gripper (3D), and the bearing to it in the base frame -- i.e.
        the argument to hand `rotate()` to face it.
        """
        object = self._resolve_object_name(object)
        om = self._env.object_managers[self._env.current_batch_index]
        obj = om.get_object_by_name(object)
        obj_pos = np.asarray(obj.position[:3], dtype=float)

        robot_view = self._env.current_robot.robot_view
        x, y, yaw = self._base_xy_yaw()
        d_xy = float(np.linalg.norm(obj_pos[:2] - np.array([x, y])))
        # Bearing in the base frame: +ccw, so it feeds straight into rotate().
        bearing = float(np.arctan2(obj_pos[1] - y, obj_pos[0] - x) - yaw)
        bearing = float(np.arctan2(np.sin(bearing), np.cos(bearing)))
        print(f"{object}")
        print(f"  world pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"  base: dist_xy={d_xy:.3f}m bearing={np.degrees(bearing):.1f}deg (rotate() arg)")

        info: dict[str, Any] = {
            "name": object,
            "position": obj_pos,
            "base_dist_xy": d_xy,
            "base_bearing_deg": float(np.degrees(bearing)),
            "grippers": {},
        }
        for mg_id, _ in self._gripper_move_groups():
            ee = robot_view.get_move_group(mg_id).leaf_frame_to_world
            delta = obj_pos - ee[:3, 3]
            # Also express the offset in the gripper's own frame -- "0.1m in
            # front of the fingers" is more actionable than a world-frame delta.
            local = ee[:3, :3].T @ delta
            print(
                f"  {mg_id}: dist={np.linalg.norm(delta):.3f}m "
                f"world_delta=({delta[0]:.3f},{delta[1]:.3f},{delta[2]:.3f}) "
                f"ee_frame_delta=({local[0]:.3f},{local[1]:.3f},{local[2]:.3f})"
            )
            info["grippers"][mg_id] = {
                "dist": float(np.linalg.norm(delta)),
                "world_delta": delta,
                "ee_frame_delta": local,
            }
        return info

    def contacts(self, object: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Print and return the robot's active contacts, strongest first. With
        `object`, only contacts with it, plus whether a gripper touches it.
        Negative `dist` is penetration depth; gripper geoms are tagged.
        """
        import mujoco

        from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

        robot_view = self._env.current_robot.robot_view
        model, data = robot_view.mj_model, robot_view.mj_data

        robot_geoms = set(descendant_geoms(model, robot_view.base.root_body_id, visible_only=False))
        gripper_geoms: set[int] = set()
        for mg_id, _ in self._gripper_move_groups():
            root = robot_view.get_move_group(mg_id).root_body_id
            gripper_geoms.update(descendant_geoms(model, root, visible_only=False))

        object_geoms: set[int] = set()
        if object is not None:
            object = self._resolve_object_name(object)
            om = self._env.object_managers[self._env.current_batch_index]
            object_geoms = set(
                descendant_geoms(model, om.get_object_by_name(object).body_id, visible_only=False)
            )

        def geom_label(gid: int) -> str:
            name = model.geom(gid).name or f"geom{gid}"
            body = model.body(model.geom_bodyid[gid]).name or f"body{model.geom_bodyid[gid]}"
            tag = " [gripper]" if gid in gripper_geoms else ""
            return f"{name}({body}){tag}"

        force_buf = np.zeros(6, dtype=np.float64)
        rows: list[dict[str, Any]] = []
        # range(data.ncon), not iteration over data.contact -- the latter walks
        # the full preallocated buffer including stale slots past ncon.
        for cid in range(data.ncon):
            c = data.contact[cid]
            g1, g2 = int(c.geom1), int(c.geom2)
            if object_geoms:
                # Contacts of the target object with anything (robot or not) --
                # "the object is still resting on the table" is a real answer.
                if not (g1 in object_geoms or g2 in object_geoms):
                    continue
            elif not (g1 in robot_geoms or g2 in robot_geoms):
                continue
            mujoco.mj_contactForce(model, data, cid, force_buf)
            rows.append(
                {
                    "geom1": g1,
                    "geom2": g2,
                    "dist": float(c.dist),
                    "force": float(np.linalg.norm(force_buf[:3])),
                }
            )

        rows.sort(key=lambda r: -r["force"])
        header = f"{len(rows)} contact(s)" + (f" involving {object!r}" if object else " on robot")
        print(header)
        for r in rows[:limit]:
            print(
                f"  {geom_label(r['geom1'])} <-> {geom_label(r['geom2'])}  "
                f"dist={r['dist']:+.5f}m force={r['force']:.2f}N"
            )
        if len(rows) > limit:
            print(f"  ... {len(rows) - limit} more (raise limit= to see them)")

        if object_geoms:
            touching = any(
                (r["geom1"] in gripper_geoms and r["geom2"] in object_geoms)
                or (r["geom2"] in gripper_geoms and r["geom1"] in object_geoms)
                for r in rows
            )
            other = any(
                r["geom1"] not in gripper_geoms and r["geom2"] not in gripper_geoms for r in rows
            )
            print(
                f"gripper touching {object!r}: {touching}; "
                f"still touching something else (unlifted): {other}"
            )
        return rows

    def look(
        self, camera: str | None = None, save: str | None = None, show: bool = False
    ) -> np.ndarray:
        """Render one RGB frame from a scene camera; return it and save it to a PNG.

        Shows what the *policy's* sensors see, which is not what the free-fly
        viewer shows.

        Args:
            camera: Camera name (see the printed list of registry names on a bad
                name). Defaults to the first robot-mounted camera, falling back
                to the first registered camera.
            save: PNG path to write. Defaults to `look_<camera>.png` in the cwd.
            show: Also open the image in the system viewer.
        """
        from PIL import Image

        from molmo_spaces.env.camera_manager import RobotMountedCamera

        env = self._env
        # Robot-mounted cameras are attached to bodies whose poses have moved
        # since the last env.step(); refresh before rendering.
        env.camera_manager.registry.update_all_cameras(env)
        registry = env.camera_manager.registry

        if camera is None:
            mounted = [c.name for c in registry if isinstance(c, RobotMountedCamera)]
            names = mounted or list(registry.keys())
            if not names:
                raise ValueError("No cameras registered in this scene.")
            camera = names[0]
        elif camera not in registry:
            raise KeyError(f"Unknown camera {camera!r}. Available: {sorted(registry.keys())}")

        frame = env.render_rgb_frame(camera)
        img = frame
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(img)

        path = Path(save) if save is not None else Path.cwd() / f"look_{camera}.png"
        image.save(path)
        print(f"{camera}: {frame.shape[1]}x{frame.shape[0]} -> {path}")
        if show:
            image.show()
        return frame

    # -- Sim state --

    _SNAPSHOT_SPEC: int | None = None

    def _snapshot_spec(self) -> int:
        """mj state bitmask: full physics plus ctrl and mocap, without which a
        restore snaps the bodies back and the held targets drive them away."""
        import mujoco

        if InteractiveShellTask._SNAPSHOT_SPEC is None:
            InteractiveShellTask._SNAPSHOT_SPEC = int(
                mujoco.mjtState.mjSTATE_FULLPHYSICS
                | mujoco.mjtState.mjSTATE_CTRL
                | mujoco.mjtState.mjSTATE_MOCAP_POS
                | mujoco.mjtState.mjSTATE_MOCAP_QUAT
            )
        return InteractiveShellTask._SNAPSHOT_SPEC

    def snapshot(self, name: str = "default") -> str:
        """Save the sim state under `name` for `restore(name)`: A/B attempts
        from an identical start. Sim state only; each skill builds a fresh
        policy anyway, and the occupancy map is unaffected by robot motion.
        """
        import mujoco

        data = self._env.current_robot.robot_view.mj_data
        spec = self._snapshot_spec()
        state = np.zeros(mujoco.mj_stateSize(self._env.mj_model, spec), dtype=np.float64)
        mujoco.mj_getState(self._env.mj_model, data, state, spec)
        self._snapshots[name] = (state, self._held_object)
        print(f"Saved snapshot {name!r} (t={data.time:.3f}s)")
        return name

    def restore(self, name: str = "default") -> None:
        """Restore the sim state saved by `snapshot(name)`."""
        import mujoco

        if name not in self._snapshots:
            raise KeyError(f"No snapshot {name!r}. Saved: {sorted(self._snapshots)}")
        state, held = self._snapshots[name]
        model = self._env.mj_model
        data = self._env.current_robot.robot_view.mj_data
        mujoco.mj_setState(model, data, state, self._snapshot_spec())
        mujoco.mj_forward(model, data)
        self._held_object = held
        self._env.camera_manager.registry.update_all_cameras(self._env)
        if self.viewer is not None:
            self.viewer.sync()
        print(f"Restored snapshot {name!r} (t={data.time:.3f}s)")
        # Rebaseline rather than warn: whatever this state is in contact with is
        # a property of the snapshot, not something the robot just did.
        self._collisions = set(self._robot_object_collisions())

    def snapshots(self) -> list[str]:
        """List the names saved by `snapshot()`."""
        names = sorted(self._snapshots)
        print(f"snapshots: {names}" if names else "no snapshots saved")
        return names

    def teleport(self, object: str, dist: float = 1.0, max_tries: int = 10) -> bool:
        """Place the base at a collision-free pose within `dist` of `object`, facing it.

        The same placement `pick`'s task sampler uses (`env.place_robot_near`) --
        no walking, no A*. When you're debugging manipulation, navigating there
        each iteration is pure overhead and a nav failure blocks the thing you
        actually wanted to test.
        """
        object = self._resolve_object_name(object)
        om = self._env.object_managers[self._env.current_batch_index]
        placed = self._env.place_robot_near(
            robot_view=self._env.current_robot.robot_view,
            target=om.get_object_by_name(object),
            max_tries=max_tries,
            sampling_radius_range=(0.0, dist),
            robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
            face_target=True,
        )
        self._env.camera_manager.registry.update_all_cameras(self._env)
        if self.viewer is not None:
            self.viewer.sync()
        print(f"{'done' if placed else 'FAILED'} - teleport near {object!r} (within {dist}m)")
        if placed:
            x, y, yaw = self._base_xy_yaw()
            print(f"base now at x={x:.3f} y={y:.3f} yaw={np.degrees(yaw):.1f}deg")
        # place_robot_near collision-checks candidate poses, so a warning here
        # means it had to settle for one -- worth seeing.
        self._warn_new_collisions(rebaseline=True)
        return placed

    def _set_gripper(self, open: bool, ticks: int, move_group: str | None) -> list[tuple[str, Any]]:
        """Drive every (or one) gripper to its fully open/closed ctrl target.

        Writes the target through `GripperGroup.set_gripper_ctrl_open` (the same
        call `GripperAction` uses) and then commands it for `ticks` policy steps
        so the fingers actually travel -- a one-shot ctrl write would be undone
        by the next command's stationary fallback.
        """
        grippers = self._gripper_move_groups()
        if move_group is not None:
            grippers = [(mg_id, g) for mg_id, g in grippers if mg_id == move_group]
            if not grippers:
                raise ValueError(
                    f"Unknown gripper move group {move_group!r}. "
                    f"Available: {[mg for mg, _ in self._gripper_move_groups()]}"
                )
        if not grippers:
            raise RuntimeError("This robot has no gripper move group.")

        targets = {}
        for mg_id, gripper in grippers:
            gripper.set_gripper_ctrl_open(open)
            targets[mg_id] = np.asarray(gripper.ctrl, dtype=np.float32).copy()

        for _ in range(ticks):
            # Only the gripper groups are commanded; everything else falls back
            # to its own "hold current state" behavior (see noop()).
            self._step_and_record(dict(targets))
            self._warn_new_collisions()
            if self.viewer is not None:
                self.viewer.sync()
        self._warn_new_collisions(rebaseline=True)

        for mg_id, gripper in grippers:
            print(
                f"{mg_id}: fingers={gripper.inter_finger_dist:.4f}m "
                f"{'OPEN' if gripper.is_open else 'CLOSED'}"
            )
        return grippers

    def grasp(
        self, ticks: int = 50, move_group: str | None = None, empty_threshold: float = 0.004
    ) -> bool:
        """Close the gripper in place, no reaching. Returns True if the fingers
        stopped more than `empty_threshold` m short of fully closed, i.e. on
        something -- the pick planners' own empty-gripper test.
        """
        grippers = self._set_gripper(open=False, ticks=ticks, move_group=move_group)
        return all(
            g.inter_finger_dist > g.inter_finger_dist_range[0] + empty_threshold
            for _, g in grippers
        )

    def release(self, ticks: int = 50, move_group: str | None = None) -> bool:
        """Open the gripper in place, without any reaching or IK.

        Returns whether every driven gripper actually reached the open end of
        its travel (False means something is jammed between the fingers).
        """
        grippers = self._set_gripper(open=True, ticks=ticks, move_group=move_group)
        opened = all(g.is_open for _, g in grippers)
        if opened:
            self._held_object = None
        return opened

    # -- Shared machinery --

    def _current_robot_base_pose(self) -> list[float]:
        from molmo_spaces.utils.pose import pose_mat_to_7d

        robot_view = self._env.current_robot.robot_view
        return pose_mat_to_7d(robot_view.base.pose).tolist()

    def _step_and_record(self, action: dict[str, Any]) -> None:
        """One policy tick: apply `action`, then poll the sensor suite and
        append the step to the task's caches.

        These are the same two calls `BaseMujocoTask.step` makes, minus its
        terminal gate (the shell's episode never ends and its horizon is
        meaningless). Direct-drive commands therefore observe at one tick per
        action, the same cadence `ParallelRolloutRunner` gets from a planner
        that emits one action per step -- so their trajectories can be mixed
        with planned ones in a BC dataset.
        """
        self._apply_action(action)
        if self.record_direct_drive:
            self.last_observation = self._observe_and_cache()[0]

    def _nav_planner_name(self, nav_only: bool = False) -> str:
        """Which base planner `nav_to(object)` drives this robot with. Also
        tells the direct-drive moves whose velocity law they should match."""
        if (
            self.config.robot_config.name == "g1"
            and not self.config.robot_config.use_holo_base
            and not nav_only
        ):
            return "fetchman"
        return "astar"

    def _astar_nav_policy_config(self):
        """The A* nav config `nav_to(object)` uses on this robot -- and so the
        source of the waypoint spacing the direct-drive moves chunk with."""
        from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig

        if self.config.robot_config.name == "g1":
            # G1's WBC converges on a waypoint in 15-40 steps, so the default
            # 10-step "no progress" replan fires too early, and the default
            # 0.25m/10deg waypoint spacing never lets it reach cruising speed.
            return AStarNavToObjPolicyConfig(
                plan_fail_after_waypoint_steps=50,
                plan_max_retries=5,
                path_max_inter_waypoint_dist=1.0,
                path_max_inter_waypoint_angle=np.radians(30),
            )
        return AStarNavToObjPolicyConfig()

    def _direct_drive_pacing(self, speed: float | None = None) -> tuple[float, float | None]:
        """(waypoint spacing in metres, pure-pursuit lead in metres or None)
        for a straight direct-drive base move -- how `nav_to(direction)` paces
        itself to walk like `nav_to(object)` does on this robot.

        Where fetchman drives the base (G1 in WBC mode) the object path walks at
        a fixed cruise speed, so the direction step does too: a fine waypoint
        chain consumed as a carrot held `speed` metres ahead. That distance is
        the speed because G1 bridges an absolute `base` waypoint to the WBC as a
        velocity equal to the position error itself
        (`G1Robot.waypoint_to_velocity_target`, unit gain, 1s time constant), so
        a carrot `speed` metres out commands exactly `speed` m/s -- and stays
        well clear of that bridge's 0.08m deadband, below which the robot simply
        does not walk. Pacing by waypoint spacing alone stalls there.

        Otherwise the object path is A*, which paces the base purely by how far
        apart its waypoints are, so the direction step chunks its segment at
        that same spacing and holds each waypoint until the base arrives.

        An explicit `speed` (m/s) forces the carrot form on any robot.
        """
        dt_s = self.config.policy_dt_ms / 1000.0
        if speed is None and self._nav_planner_name() == "fetchman":
            from molmo_spaces.policy.solvers.navigation.fetchman_base_planner_policy_port import (
                FetchManBasePlannerPolicyPort,
            )

            speed = float(FetchManBasePlannerPolicyPort.SPEED)
        if speed is not None:
            return max(speed * dt_s, 1e-3), speed
        return self._astar_nav_policy_config().path_max_inter_waypoint_dist, None

    def _drive_base_waypoints(
        self,
        waypoints: np.ndarray,
        label: str,
        threshold: float = 0.1,
        max_ticks: int | None = None,
        lead: float | None = None,
    ) -> bool:
        """Command `waypoints` (N x [x, y, yaw], world frame) to the base in
        order, one policy tick each, and drive the last one to convergence.

        This is the shared execution half of both `nav_to` paths: the same
        `{"base": [x, y, yaw]}` action interface, the same
        `robot_view.is_close_to` arrival test, the same per-tick observation
        recording, and the same chain-of-waypoints pacing an
        `AStarPlannerPolicy` plan gets (see `split_by_max_dist`). Only the
        planning half differs -- A*/fetchman route around obstacles, a
        direct-drive move goes straight.

        `lead`: with a fine waypoint chain, advance to keep the commanded
        waypoint this far ahead of the base, i.e. pure pursuit with a fixed
        carrot distance. Without it, a waypoint is held until the base reaches
        it, which is what the A* plan's own coarse waypoints do.
        """
        robot_view = self._env.current_robot.robot_view
        if max_ticks is None:
            max_ticks = 150 + len(waypoints)
        last = len(waypoints) - 1
        idx = 0
        ticks = 0

        while ticks < max_ticks:
            if lead is None:
                while idx < last and robot_view.is_close_to(
                    ["base"], waypoints[idx], threshold=threshold
                ):
                    idx += 1
            else:
                xy = robot_view.base.pose[:2, 3]
                while idx < last and float(np.linalg.norm(waypoints[idx][:2] - xy)) < lead:
                    idx += 1
            if idx == last and robot_view.is_close_to(
                ["base"], waypoints[last], threshold=threshold
            ):
                break

            self._step_and_record({"base": waypoints[idx]})
            self._warn_new_collisions()
            if self.viewer is not None:
                self.viewer.sync()
            ticks += 1
            if ticks % 10 == 0:
                pose = robot_view.base.pose
                log.debug(
                    f"[{label}] tick={ticks} waypoint={idx + 1}/{len(waypoints)} "
                    f"pos=({pose[0, 3]:.3f},{pose[1, 3]:.3f}) "
                    f"dist_remaining={robot_view.distance_to(['base'], waypoints[last]):.4f}"
                )

        self._warn_new_collisions(rebaseline=True)
        return bool(robot_view.is_close_to(["base"], waypoints[last], threshold=threshold))

    def _run_subtask(
        self,
        sub_task: BaseMujocoTask,
        policy_factory,
        end_on_success: bool = True,
        settle_time_s: float = 0.0,
    ) -> bool:
        """Build the policy, register it, and run `sub_task` to completion.
        `end_on_success=False` runs the full motion instead of stopping when
        the success criterion first trips (pick's lift). `settle_time_s` holds
        the robot's final command for that much simulated time before judging
        success: the WBC lags its targets, so at the policy's done tick a
        grasped object can still be rising towards the lift target. Collisions
        are checked once at the end; only the direct-drive commands check
        every tick.
        """
        from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner

        policy = policy_factory(sub_task.config, sub_task)
        sub_task.register_policy(policy)
        success = ParallelRolloutRunner.run_single_rollout(
            episode_seed=0,
            task=sub_task,
            policy=policy,
            viewer=self.viewer,
            end_on_success=end_on_success,
        )
        if settle_time_s > 0 and not success:
            # An empty action holds every controller's last target (see noop).
            for _ in range(max(1, round(settle_time_s * 1000.0 / self.config.policy_dt_ms))):
                self._step_and_record({})
                if self.viewer is not None:
                    self.viewer.sync()
            success = bool(sub_task.judge_success())
        print(f"{'done - ' if success else 'FAILED - '}{sub_task.get_task_description()}")
        # The task's final metrics, read before close() drops its env reference.
        self._last_subtask_info = sub_task.get_info()[0] if hasattr(sub_task, "get_info") else {}
        # Every skill builds a fresh policy against the one long-lived robot, so
        # anything a policy switched on for its own control loop has to come
        # back off before the next skill runs (see BasePolicy.close).
        policy.close()
        sub_task.close()
        self._warn_new_collisions(rebaseline=True)
        return success

    # -- Skills --

    def nav_to(
        self,
        object: str,
        planner: str | None = None,
        dist: float = 0.25,
        speed: float | None = None,
    ) -> bool:
        """Navigate the base to `object`, or step `dist` meters if `object` is
        a direction: "forward"/"forwards", "backward"/"backwards", "left" or
        "right" (`speed` in m/s caps the step's pace; it defaults to the pace of
        whichever planner drives this robot, and is ignored for an object goal).

        `object` may also name a nav-only structural target -- a door, doorframe
        or window (see `NAV_STRUCTURAL_TYPES`); those always walk with "astar".

        planner: "fetchman" (the g1_molmo-ported velocity controller, default
            for G1 in WBC mode, the only robot that consumes its command) or
            "astar" (default for every other robot, and for structural targets).
        """
        if isinstance(object, str) and object.lower() in _NAV_DIRECTIONS:
            return self._nav_to_direction(object.lower(), dist, speed=speed)

        from molmo_spaces.configs.policy_configs import FetchManBasePlannerPolicyConfig
        from molmo_spaces.configs.task_configs import NavToObjTaskConfig
        from molmo_spaces.policy.solvers.navigation.fetchman_base_planner_policy_port import (
            FetchManBasePlannerPolicyPort,
        )
        from molmo_spaces.tasks.nav_task import NavToObjTask

        object = self._resolve_object_name(object)

        om = self._env.object_managers[self._env.current_batch_index]
        # A door or window is wall-embedded and ungraspable, so fetchman's
        # grasping standoff is the wrong goal for it; ProcTHORMap already clears
        # the open-door path, so A* can plan straight into the opening.
        nav_only = om.is_structural(object)

        if planner is None:
            planner = self._nav_planner_name(nav_only=nav_only)
        if planner not in ("astar", "fetchman"):
            raise ValueError(f"Unknown planner {planner!r}, expected 'astar' or 'fetchman'")

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
            # The Port walks with the pick's own grid helpers and control law,
            # and parks on the pick's standoff annulus facing the object, so a
            # following `pick` grasps from where it stands instead of walking
            # out again.
            from molmo_spaces.configs.policy_configs_fetchman_pick import (
                FetchmanPickPlannerPolicyConfig,
            )

            standoff = FetchmanPickPlannerPolicyConfig().goal_standoff_radius_range
            nav_config.policy_config = FetchManBasePlannerPolicyConfig(
                policy_cls=FetchManBasePlannerPolicyPort,
                policy_factory=FetchManBasePlannerPolicyPort,
                standoff_radius_range=standoff,
            )
        else:
            nav_config.policy_config = self._astar_nav_policy_config()
        # The fetchman planner walks to a grasping standoff and then turns to
        # face the object, so its rollout runs to the planner's own done action
        # and success is judged where it actually stops: within the standoff
        # annulus (plus the brake's stop pad) of the object. The A* planner has
        # no such terminal phase and keeps the "within 0.5m" early exit.
        if planner == "fetchman":
            # Distance is judged to the object's origin, so a large object (a
            # bed) is "reached" from much further out than a mug: add its
            # horizontal half-diagonal, which is also roughly how far the
            # planner's closest-reachable fallback has to stop short.
            target_obj = om.get_object_by_name(object)
            half = np.asarray(target_obj.aabb_size[:2], dtype=np.float64)
            succ_pos_threshold = standoff[1] + 0.15 + float(np.linalg.norm(half))
            end_on_success = False
        else:
            succ_pos_threshold = 0.5
            end_on_success = True
        nav_config.task_config = NavToObjTaskConfig(
            task_cls=NavToObjTask,
            pickup_obj_name=object,
            robot_base_pose=self._current_robot_base_pose(),
            succ_pos_threshold=succ_pos_threshold,  # meters (default 1.5m)
        )

        sub_task = NavToObjTask(self._env, nav_config)
        sub_task.occupancy_map = self.occupancy_map

        success = self._run_subtask(
            sub_task, nav_config.policy_config.policy_factory, end_on_success=end_on_success
        )

        robot_view = self._env.current_robot.robot_view
        target_obj = om.get_object_by_name(object)
        distance = float(
            np.linalg.norm(np.asarray(target_obj.position[:2]) - robot_view.base.pose[:2, 3])
        )
        print(f"Distance to {object!r}: {distance:.3f}m")

        return success

    def _nav_to_direction(
        self,
        direction: str,
        dist: float,
        max_ticks: int | None = None,
        threshold: float = 0.1,
        speed: float | None = None,
    ) -> bool:
        """Step `dist` meters along `direction` in the base frame, heading fixed.

        Straight line, no path planning -- but chunked into the same kind of
        waypoint chain `nav_to(object)`'s A* plan is, executed by the same
        `_drive_base_waypoints`, and paced to whichever planner would drive
        this robot (see `_direct_drive_pacing`; `speed` in m/s overrides). So a
        hand-driven step walks at the same speed as a planned one and records
        the same per-tick observations.

        Routing this through the planner policies themselves is not on: the
        fetchman law turns to face each waypoint before walking to it (and its
        holonomic hop is capped at a ~20deg bearing, because the WBC went
        unstable on sideways commands), which would turn a 0.3m side-step into
        turn-walk-turn.
        """
        x, y, yaw = self._base_xy_yaw()
        local_dx, local_dy = _NAV_DIRECTIONS[direction]
        world_dx = local_dx * np.cos(yaw) - local_dy * np.sin(yaw)
        world_dy = local_dx * np.sin(yaw) + local_dy * np.cos(yaw)
        start_xy = np.array([x, y])
        target_xy = start_xy + dist * np.array([world_dx, world_dy])

        step_dist, lead = self._direct_drive_pacing(speed)
        # The same segment-chunking AStarPlannerPolicy applies to its own path.
        xys = split_by_max_dist(np.stack([start_xy, target_xy]), step_dist)
        waypoints = np.concatenate([xys, np.full((len(xys), 1), yaw)], axis=1)

        success = self._drive_base_waypoints(
            waypoints,
            label=f"nav_to:{direction}",
            threshold=threshold,
            max_ticks=max_ticks,
            lead=lead,
        )
        print(f"{'done' if success else 'FAILED'} - Nav {direction} {dist:.2f}m")
        return success

    def rotate(
        self, angle_deg: float, max_ticks: int | None = None, threshold: float = 0.1
    ) -> bool:
        """Rotate the robot base in place by `angle_deg` degrees (+ccw), holding x/y fixed.

        Bypasses nav_to()'s A* path planning entirely, driving the same "base"
        action interface (robot.update_control({"base": [x, y, theta]})) directly
        with a target heading only -- isolates whether the underlying base/WBC
        controller can turn in place at all, independent of path planning/replanning.
        Deliberately a single waypoint rather than the slerped chain a plan would
        use, so the raw controller is what's under test; execution and per-tick
        observation recording are shared with every other base move
        (`_drive_base_waypoints`).
        """
        x, y, current_yaw = self._base_xy_yaw()
        target = np.array([[x, y, current_yaw + np.radians(angle_deg)]])

        success = self._drive_base_waypoints(
            target, label="rotate", threshold=threshold, max_ticks=max_ticks
        )
        print(f"{'done' if success else 'FAILED'} - Rotate by {angle_deg:.1f} deg")
        return success

    def noop(self, ticks: int = 50) -> None:
        """Step `ticks` policy steps with an empty action: every controller
        holds its current state (G1's WBC keeps balancing). Lets the robot settle."""
        for _ in range(ticks):
            self._step_and_record({})
            self._warn_new_collisions()
            if self.viewer is not None:
                self.viewer.sync()
        self._warn_new_collisions(rebaseline=True)

    def pick(
        self, object: str, planner_policy_config_cls: type | None = None, max_attempts: int = 3
    ) -> bool:
        """Pick up and lift `object`.

        Args:
            object: Name (or short/approximate name) of the object to pick.
            max_attempts: Retries. Each retry reseeds the policy and walks to a
                fresh standoff pose instead of replaying the failed approach;
                stops early if the object has fallen off its surface.
            planner_policy_config_cls: Defaults to FetchmanPickPlannerPolicyConfig
                (whole-body IK) for G1 in WBC mode, PickPlannerPolicyConfig otherwise.
        """
        from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
        from molmo_spaces.configs.policy_configs_fetchman_pick import (
            FetchmanPickPlannerPolicyConfig,
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

        pick_config = copy.deepcopy(self.config)
        pick_config.task_type = "pick"
        pick_config.policy_config = planner_policy_config_cls()
        if planner_policy_config_cls is FetchmanPickPlannerPolicyConfig:
            # G1PickPlannerPolicy advances the WBC gait clock once per
            # get_action; at the shell's 70ms policy rate that is 14x too slow
            # and the robot falls over, so pick runs at the control rate.
            pick_config.policy_dt_ms = pick_config.ctrl_dt_ms
            # ... which makes each step 14x shorter, so keep the same simulated
            # time budget rather than cutting it to a fraction of a pick.
            pick_config.task_horizon = int(
                self.config.task_horizon * self.config.policy_dt_ms / pick_config.policy_dt_ms
            )
        # The lift is measured from where the object was when pick() began,
        # on every attempt: a retry may start with the object already in the
        # hand (a grasp that held but had not risen 10cm when judged), and
        # measuring from that in-hand pose reported 2-5cm lifts for an object
        # that ended up 12cm above the table.
        start_pose = pose_mat_to_7d(pickup_obj.pose)
        initial_z = float(start_pose[2])
        goal_pose = start_pose.copy()
        goal_pose[2] += 0.1  # lift 10cm above the start pose
        success = False
        for attempt in range(max(1, max_attempts)):
            pick_config.task_config = PickTaskConfig(
                task_cls=PickTask,
                pickup_obj_name=object,
                robot_base_pose=self._current_robot_base_pose(),
                pickup_obj_start_pose=start_pose.tolist(),
                pickup_obj_goal_pose=goal_pose.tolist(),
            )
            pick_config.task_config.referral_expressions["pickup_obj_name"] = object

            # A 1cm lift (PickTaskConfig's default) is cleared while the gripper is
            # still closing; combined with end_on_success that ended the rollout
            # before post_close/lift ever ran, so `pick` returned True on a barely
            # -moved object. Require the lift to be real, and run the motion out.
            pick_config.task_config.succ_pos_threshold = 0.10
            sub_task = PickTask(self._env, pick_config)
            # G1PickPlannerPolicy seeds its standoff-pose and grasp sampling from
            # the task's episode_seed, and on a retry is told to walk to that
            # fresh standoff even if it is already inside the annulus, so it
            # approaches from a different pose instead of replaying the attempt
            # that just failed from this one.
            sub_task.episode_seed = attempt
            if hasattr(pick_config.policy_config, "force_standoff_walk"):
                pick_config.policy_config.force_standoff_walk = attempt > 0
            if hasattr(pick_config.policy_config, "direct_walk"):
                pick_config.policy_config.direct_walk = True
            success = self._run_subtask(
                sub_task,
                pick_config.policy_config.policy_factory,
                end_on_success=False,
                settle_time_s=1.0,
            )
            if success:
                break
            info = self._last_subtask_info
            if "lift_height" in info:
                print(
                    f"pick: lift {info['lift_height'] * 100:.1f}cm (need "
                    f"{pick_config.task_config.succ_pos_threshold * 100:.0f}), "
                    f"robot contact {info['robot_contact']}, touching something else "
                    f"{info['robot_contact'] and not info['only_robot_contact']}"
                )
            dropped = initial_z - float(pose_mat_to_7d(pickup_obj.pose)[2])
            if dropped > 0.3:
                print(
                    f"pick: {object!r} has fallen {dropped:.2f}m (knocked off its surface?)"
                    " -- not retrying"
                )
                break
            if attempt + 1 < max_attempts:
                print(
                    f"pick: attempt {attempt + 1}/{max_attempts} failed"
                    " -- retrying from a fresh standoff pose"
                )
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
        """Drop into an interactive Python shell exposing the skills as functions.

        Args:
            commands: Statements to run, in order, before the prompt (seeding a
                session, or headless scripting); their bindings stay in the namespace.
        """
        banner = "\n".join(
            [
                "",
                "Interactive robot shell. Available commands:",
                " skills:",
                "  nav_to(object=name)                    - navigate to an object, or to a door/window",
                "  nav_to(object=dir, dist=.25)           - step dist meters in the base frame, at the",
                "                                           same pace as a planned nav; dir is",
                "                                           'forward(s)'/'backward(s)'/'left'/'right'",
                "  rotate(angle_deg)                      - rotate the base in place (+ccw), no path planning",
                "  noop(ticks=50)                         - hold current position/pose, do nothing",
                "  pick(object=name, max_attempts=3)      - pick up and lift an object, retrying",
                "                                           from a new standoff pose on failure",
                "  pick_and_place(object=x, receptacle=y) - pick up an object and place it on/in y",
                "  open_object(object=name)               - open a drawer/cabinet/door",
                "  close_object(object=name)              - close a drawer/cabinet/door",
                "  grasp() / release()                    - close/open the gripper in place, no reaching",
                "  teleport(object=name, dist=1.0)        - place the base near an object, no walking",
                " inspection:",
                "  list_objects()                         - list interactable objects in the scene,",
                "                                           plus nav-only doors/windows",
                "  list_objects(dist=1.0)                 - only objects within 1m of the robot, closest first",
                "  where()                                - base pose, gripper pose, grasp state",
                "  whereis(object=name)                   - object pos + distance/bearing from base and gripper",
                "  contacts(object=None)                  - active robot contacts w/ forces; is the gripper touching?",
                "  look(camera=None, save=None)           - render a camera frame to a PNG",
                " state:",
                "  snapshot(name) / restore(name)         - save/replay the sim state; snapshots() lists them",
                "  help()                                 - re-print this message",
                "Each skill runs the robot to completion and returns True/False for success.",
                "Any command that moves the robot prints '!! COLLISION' when it newly hits an",
                "object (floor and robot self-contacts excluded), once per object per command.",
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

        def pick(object, planner_policy_config_cls=None, **kwargs):
            return self.pick(object, planner_policy_config_cls, **kwargs)

        def pick_and_place(object, receptacle):
            return self.pick_and_place(object, receptacle)

        def open_object(object, joint_index=0):
            return self.open_object(object, joint_index)

        def close_object(object, joint_index=0):
            return self.close_object(object, joint_index)

        def list_objects(**kwargs):
            return self.list_objects(**kwargs)

        def where():
            return self.where()

        def whereis(object):
            return self.whereis(object)

        def contacts(object=None, **kwargs):
            return self.contacts(object, **kwargs)

        def look(camera=None, **kwargs):
            return self.look(camera, **kwargs)

        def snapshot(name="default"):
            return self.snapshot(name)

        def restore(name="default"):
            return self.restore(name)

        def snapshots():
            return self.snapshots()

        def teleport(object, **kwargs):
            return self.teleport(object, **kwargs)

        def grasp(**kwargs):
            return self.grasp(**kwargs)

        def release(**kwargs):
            return self.release(**kwargs)

        def help():
            print(banner)

        task = self
        env = self._env

        # The pick/walk trace lines are the shell's main diagnostic.
        from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy_g1 import (
            enable_g1_trace,
        )

        enable_g1_trace()

        namespace = dict(globals(), **locals())
        # Printed here, not handed to code.interact(), which writes its banner
        # to stderr -- so under a pipe it landed out of order with everything
        # the seeded commands print to stdout (a startup list_objects() looked
        # like it had never run).
        print(banner)
        for cmd in commands or []:
            print(f">>> {cmd}")
            exec(cmd, namespace)

        save_history = _setup_readline_history(namespace)

        try:
            code.interact(banner="", local=namespace)
        except (SystemExit, KeyboardInterrupt):
            # exit()/quit() raise SystemExit, but when a passive viewer is attached
            # its background render thread can post a KeyboardInterrupt to the main
            # thread around the same time (e.g. when the viewer window is closed),
            # which code.interact()'s own loop catches and reports without exiting,
            # requiring a second exit() to actually leave. Whichever one surfaces
            # here, treat it as "the user is done" and exit the shell in one shot.
            print()
        finally:
            # atexit covers process teardown; this covers "shell exited but the
            # process lives on" (e.g. an embedding script that keeps going).
            save_history()
