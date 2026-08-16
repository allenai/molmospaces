"""Interactive shell task: exposes robot skills as plain Python functions in a REPL.

Rather than pursuing one fixed goal like other tasks, `InteractiveShellTask` drops
the user into a `code.interact()` session where calling `nav_to(object=...)`,
`pick(object=...)`, etc. builds the matching single-skill task + planner policy on
the fly and runs it to completion (reusing `ParallelRolloutRunner.run_single_rollout`)
against the same persistent env/robot, so each command's effect is visible to the next.
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
    """Make up-arrow recall commands from previous shell sessions.

    Loads `_HISTORY_FILE` into readline's in-memory history and registers an
    atexit hook to write it back, so a freshly opened shell starts with the
    previous session's commands. Also wires tab-completion against `namespace`
    while we're holding readline. A missing/unusable readline (or history file)
    just means no history -- never a failed shell.

    Returns the "flush history to disk" callable (a no-op if readline is
    unavailable) so the caller can also save when the shell exits but the
    process keeps running.
    """
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

    def __init__(self, env: BaseMujocoEnv, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(env, exp_config)
        # Cached occupancy map for nav_to()'s A* planner; built on first use.
        self.occupancy_map: Any | None = None
        self._held_object: str | None = None
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

    def _create_sensor_suite_from_config(self, exp_config: MlSpacesExpConfig) -> SensorSuite:
        return SensorSuite(get_core_sensors(exp_config))

    def get_task_description(self) -> str:
        return "Interactive shell session"

    def get_reward(self) -> np.ndarray:
        return np.zeros(self._env.n_batch, dtype=np.float32)

    def judge_success(self) -> bool:
        return False

    # -- Object discovery --

    def list_objects(self, limit: int = 200, dist: float | None = None) -> list[str]:
        """Print and return a human-readable summary of interactable objects in the scene.

        Args:
            limit: Maximum number of objects to report.
            dist: If given, only report objects whose center is within `dist` meters
                of the robot base, closest first, with the distance appended to each
                line. Otherwise all objects are reported, ordered by name.
        """
        om = self._env.object_managers[self._env.current_batch_index]

        if dist is None:
            summaries = om.summarize_top_level_bodies(receptacle_types=[], limit=limit)
        else:
            robot_pos = self._env.current_robot.robot_view.base.pose[:3, 3]
            near: list[tuple[float, str]] = []
            for obj in om.list_top_level_objects():
                d = float(np.linalg.norm(np.asarray(obj.position[:3]) - robot_pos))
                if d <= dist:
                    near.append((d, obj.name))
            near.sort()
            summaries = [
                f"{om.object_summary_str(name, receptacle_types=[])} [{d:.2f}m]"
                for d, name in near[:limit]
            ]

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
        Otherwise every object is scored against `name` by `_match_score`, and
        of all objects sharing the top score the one spatially closest to the
        robot wins. Since this is a guess, it requires confirmation via a y/n
        prompt (blocks on stdin) before use -- raises if the user declines.
        Prefix `name` with "~" (e.g. "~tomato") to skip the prompt and
        auto-accept the closest match.
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
        """Best match of `query` against any of `labels`, in [0, 1].

        Deliberately tiered rather than a raw `difflib` ratio: scoring against
        the full instance name lets the 32-char asset hash dominate the ratio,
        so "bowl" scored a *parked* `place_receptacle/2_0/Bowl_25` (0.25) over
        the actual `bowl_be78...` sitting 0.84m away, and no two objects ever
        tied -- which meant the closest-wins tie-break below never ran. Exact
        and substring hits are pinned above the fuzzy tier so a real category
        match always beats coincidental character overlap.

        A prefix hit is scored the *same* as any other substring hit rather than
        above it, deliberately: ranking prefixes higher made "table" resolve to
        a `tablelamp` 11m away instead of the `diningtable` 0.8m in front of the
        robot. Leaving them tied hands that call to the distance tie-break,
        which is the disambiguation the rest of this shell already leans on.
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
        """Current robot-vs-object collisions, as {object: (robot link, penetration depth)}.

        Keyed by the other side's *root* body -- i.e. the object as a whole, so
        one entry per object however many of its geoms are touching, reporting
        the deepest. Excluded: robot self-collisions (the robot's own links
        touching each other, constantly present on G1 and not what "hit
        something" means) and the floor, following
        `env.check_robot_collision_in_current_pose`'s "allow floor contacts but
        reject walls/obstacles" rule.

        The floor test has to look at the *geom* name, not just the body: the
        floor geom is named "floor" but hangs directly off the world body, so a
        body-name-only check (which is all the env-level helper does) reports
        every footfall as a collision with "world".
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
        """Print a warning for each object the robot has *newly* collided with.

        Called every tick by the direct-drive loops, but throttled to run at
        most once per `collision_check_interval` seconds of *simulated* time
        (so the rate is independent of tick size). The `rebaseline=True` call at
        the end of a command always runs, so a collision still standing when the
        command finishes is never missed -- only ones made and released inside
        one interval are.

        The "already reported" set only grows between rebaselines, and is
        re-synced to reality by the rebaseline call. Without that, a contact
        that flickers on and off across ticks -- which is exactly what a walking
        gait brushing a table does -- re-warns on every re-contact, tens of
        times per command. Net effect: at most one warning per object per
        command, and a contact that persists into the next command stays quiet.
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
        """Print (and return) the robot's active contacts, strongest force first.

        With `object`, restricts to contacts involving that object and adds a
        verdict line on whether the gripper is actually touching it -- the
        question `pick()`'s True/False can't answer on its own.

        Each row is `geom_a <-> geom_b  dist=... force=...N`, where a negative
        `dist` is penetration depth and `force` is the contact normal force
        magnitude. Geoms belonging to a gripper are tagged `[gripper]`.
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
        """mj state bitmask: full physics plus the ctrl/mocap the controllers need.

        mjSTATE_FULLPHYSICS alone (time+qpos+qvel+act+warmstart+plugin) restores
        the bodies but not the actuator targets currently being held, nor the
        mocap targets a weld-driven base (e.g. FloatingRUM, G1 in holo mode)
        follows -- restoring without them snaps the robot back and then
        immediately drives it somewhere else.
        """
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
        """Save the current sim state under `name`, for `restore(name)` to replay.

        Lets you A/B two variants (planner configs, approach poses, ...) from a
        byte-identical start without restarting the process: snapshot once the
        robot is in position, then restore between attempts.

        Caveat: this saves *sim* state (mujoco), not python-side policy state.
        Each skill call builds a fresh sub-task/policy anyway, so that's usually
        the whole story -- but a stale `occupancy_map` is deliberately kept
        (the scene geometry it maps is unchanged by robot motion).
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
            self._apply_action(dict(targets))
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
        """Close the gripper in place, without any reaching or IK.

        Decomposes `pick()` into reach vs. close-fingers: pair with `contacts()`
        to tell "the arm never got there" apart from "the fingers closed on
        nothing".

        Returns True if the fingers stopped short of fully closed by more than
        `empty_threshold` meters -- i.e. they stopped on *something*. This is
        the same empty-gripper test the pick planners use (see
        `gripper_empty_threshold` in BaseObjectManipulationPlannerPolicy, and
        FetchmanPickPlannerPolicy's own 0.004m margin); a False means the
        fingers closed on empty air.
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

    def _run_subtask(self, sub_task: BaseMujocoTask, policy_factory) -> bool:
        """Construct the policy, register it, and run `sub_task` to completion.

        The rollout loop lives in `ParallelRolloutRunner.run_single_rollout` and
        steps `sub_task`, not this task, so collisions are checked once at the
        end rather than per tick -- a contact made and released mid-rollout goes
        unreported. The direct-drive commands (`rotate`, `noop`, the
        `nav_to` direction step, `grasp`/`release`) do drive their own loop here
        and so check every tick.
        """
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
        self._warn_new_collisions(rebaseline=True)
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
            self._warn_new_collisions()
            if self.viewer is not None:
                self.viewer.sync()
            if i % 10 == 0:
                p = robot_view.base.pose
                dist_remaining = robot_view.distance_to(["base"], target)
                log.debug(
                    f"[nav_to:{direction}] i={i} pos=({p[0, 3]:.3f},{p[1, 3]:.3f}) "
                    f"dist_remaining={dist_remaining:.4f}"
                )

        self._warn_new_collisions(rebaseline=True)
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
            self._warn_new_collisions()
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

        self._warn_new_collisions(rebaseline=True)
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
            self._warn_new_collisions()
            if self.viewer is not None:
                self.viewer.sync()
        self._warn_new_collisions(rebaseline=True)

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
                " skills:",
                "  nav_to(object=name)                    - navigate to an object",
                "  nav_to(object=dir, dist=.25)           - step dist meters in the base frame;",
                "                                           dir is 'forward'/'backward'/'left'/'right'",
                "  rotate(angle_deg)                      - rotate the base in place (+ccw), no path planning",
                "  noop(ticks=50)                         - hold current position/pose, do nothing",
                "  pick(object=name)                      - pick up and lift an object",
                "  pick_and_place(object=x, receptacle=y) - pick up an object and place it on/in y",
                "  open_object(object=name)               - open a drawer/cabinet/door",
                "  close_object(object=name)              - close a drawer/cabinet/door",
                "  grasp() / release()                    - close/open the gripper in place, no reaching",
                "  teleport(object=name, dist=1.0)        - place the base near an object, no walking",
                " inspection:",
                "  list_objects()                         - list interactable objects in the scene",
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

        def pick(object, planner_policy_config_cls=None):
            return self.pick(object, planner_policy_config_cls)

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

        namespace = dict(globals(), **locals())
        for cmd in commands or []:
            print(f">>> {cmd}")
            exec(cmd, namespace)

        save_history = _setup_readline_history(namespace)

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
        finally:
            # atexit covers process teardown; this covers "shell exited but the
            # process lives on" (e.g. an embedding script that keeps going).
            save_history()
