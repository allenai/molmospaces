"""IEEE Humanoids 2026 Loco-Manipulation Challenge, manipulation task 3: lightbulb turning.

Rules PDF: "Commonly available light fixtures will be affixed to a board. 3D-printed lightbulbs
must be either tightened or picked up and screwed into place." The boards are on the wall and at
the floor, so the robot has to bend over for the low ones.

The scene (scripts/data/format_conversion/convert_testbed_to_mjcf.py) has ten fixtures: 0-4 on a
vertical wall board (0 centre, 1 top right, 2 bottom right, 3 top left, 4 bottom left, seen from
the robot) and 5-9 on a 45-degree floor board. Fixtures 0, 1, 3 and 5-9 hold a bulb that starts
loose (turn 0); `EMPTY_FIXTURES` (2 and 4, the bottom row of the wall board) are empty. Ten spare
bulbs stand in a rack on the right-hand rail (the CAD has no separate box; this is where its spare
bulbs are). Each fixture has a right-hand thread, joint `thread{i}_turn` (positive = tightening)
coupled to a 1 mm-per-turn slide, with a hard stop at 285 degrees.

* an installed bulb is **tightened** once its turn passes `BULB_SEATED_DEG`;
* a spare bulb is **captured** when its base is inside an empty fixture's mouth (within
  `CAPTURE_DIST`, axis within `CAPTURE_ANGLE`): the task then activates the weld `seat_{spare}_{fixture}`
  tying it to the fixture's thread (a free bulb has no thread until it is engaged), and it counts as
  **placed** once it has been turned past `BULB_SEATED_DEG` as well.

Use via the interactive shell (config InteractiveShellG1Dex3ChallengeLightbulb): `status()`,
`turn_bulb(i)`, `place_bulb(i)`, `solve_lightbulb()`.

Sim parameters the CAD does not give (thread friction 0.25 N m, bulb surface friction 6, joint
inertia and damping) were tuned until a three-digit grip could turn a bulb at all; see the
constants in the converter. The Dex3 hand rolls a bulb ~40-60 degrees per grasp, so a bulb takes
several grasp-twist-release cycles.
"""

import logging

import mujoco
import numpy as np

from molmo_spaces.tasks.interactive_shell_task import InteractiveShellTask

log = logging.getLogger(__name__)

N_FIXTURES = 10
WALL_FIXTURES = (0, 1, 2, 3, 4)
EMPTY_FIXTURES = (2, 4)  # keep in sync with the converter
N_SPARES = 10
BULB_SEATED_DEG = 270.0  # turn at which a bulb is tightened / seated (converter: SEAT_DEG)
BULB_BOTTOMED_DEG = 281.0  # the thread's hard stop is at 285 (converter: STOP_DEG)
FALLEN_Z = 0.25  # m, pelvis height below which the robot has fallen (a deep squat is ~0.35 m)
FALLEN_UP = 0.6  # pelvis up-axis z component below which the robot has tipped over
CAPTURE_DIST = 0.012  # m, spare base to the empty fixture's thread origin
CAPTURE_ANGLE = np.radians(12.0)


class ChallengeLightbulbTask(InteractiveShellTask):
    """Per-bulb progress, scored from sim state every tick the shell steps the robot."""

    def __init__(self, env, exp_config) -> None:
        super().__init__(env, exp_config)
        # A full lightbulb solve is tens of thousands of contact-rich ticks: polling the camera
        # suite (and caching every observation) on each one costs most of the wall time and grows
        # to GBs. Set True again to record the solve as a trajectory.
        self.record_direct_drive = False
        model = env.current_model
        name2id = lambda kind, n: mujoco.mj_name2id(model, kind, n)  # noqa: E731
        self._turn_qadr = [
            int(model.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, f"thread{i}_turn")])
            for i in range(N_FIXTURES)
        ]
        self._thread_body = [name2id(mujoco.mjtObj.mjOBJ_BODY, f"thread{i}") for i in range(N_FIXTURES)]
        self._spare_body = [name2id(mujoco.mjtObj.mjOBJ_BODY, f"spare{k}") for k in range(N_SPARES)]
        self._weld = {
            (k, j): name2id(mujoco.mjtObj.mjOBJ_EQUALITY, f"seat_{k}_{j}")
            for k in range(N_SPARES)
            for j in EMPTY_FIXTURES
        }
        self.max_turn = np.zeros(N_FIXTURES)  # deg, furthest tightened so far
        self.captured_by: dict[int, int] = {}  # empty fixture -> spare bulb index engaged in it
        # Fixtures that were tightened / seated at some point. MuJoCo resets the whole state when
        # the physics blows up (e.g. after a fall), which also un-turns every bulb; this keeps the
        # record of what had been achieved up to then (status() reports it as "ever_done").
        self.done_ever: set[int] = set()
        self.fell = False
        self.phase = ""  # label the solver sets, kept with `trace` for debugging
        self.trace: list[tuple[str, np.ndarray]] = []
        # The arm poses the episode starts in, which the solver returns to.
        view = env.current_robot.robot_view
        self.rest_arms = {
            side: np.asarray(view.get_move_group(f"{side}_arm").joint_pos, dtype=np.float32).copy()
            for side in ("left", "right")
        }

    # ---- state readers -------------------------------------------------------------------
    def turn_deg(self, fixture: int) -> float:
        return float(np.degrees(self._env.current_data.qpos[self._turn_qadr[fixture]]))

    def is_empty(self, fixture: int) -> bool:
        return fixture in EMPTY_FIXTURES

    def tightened(self, fixture: int) -> bool:
        """Installed bulb turned past the seating angle, or a placed bulb screwed in."""
        if self.is_empty(fixture) and fixture not in self.captured_by:
            return False
        return self.turn_deg(fixture) >= BULB_SEATED_DEG - 1.0

    def spare_pose(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        data = self._env.current_data
        return data.xpos[self._spare_body[k]].copy(), data.xmat[self._spare_body[k]].reshape(3, 3).copy()

    def fixture_pose(self, fixture: int) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the fixture's thread frame (origin = bulb base, z = bulb axis, toward the tip)."""
        data = self._env.current_data
        return (
            data.xpos[self._thread_body[fixture]].copy(),
            data.xmat[self._thread_body[fixture]].reshape(3, 3).copy(),
        )

    def base_fallen(self) -> bool:
        """The pelvis is on the floor or tipped over. The whole-body IK squats on its own to reach
        the floor board (pelvis down to ~0.35 m), so that alone is not a fall; the keypad task's
        pelvis z < 0.5 m test would flag it."""
        pose = self._env.current_robot.robot_view.base.pose
        return bool(pose[2, 3] < FALLEN_Z or pose[2, 2] < FALLEN_UP)

    # ---- scoring -------------------------------------------------------------------------
    def _update_progress(self) -> None:
        data = self._env.current_data
        self.trace.append((self.phase, np.degrees(data.qpos[self._turn_qadr])))
        for i in range(N_FIXTURES):
            self.max_turn[i] = max(self.max_turn[i], self.turn_deg(i))
        # A free spare bulb has no thread: engage it once it is inside an empty fixture's mouth.
        for j in EMPTY_FIXTURES:
            if j in self.captured_by:
                continue
            p_f, R_f = self.fixture_pose(j)
            for k in range(N_SPARES):
                if k in self.captured_by.values():
                    continue
                p_s, R_s = self.spare_pose(k)
                if (
                    np.linalg.norm(p_s - p_f) < CAPTURE_DIST
                    and np.arccos(np.clip(R_s[:, 2] @ R_f[:, 2], -1, 1)) < CAPTURE_ANGLE
                ):
                    self.captured_by[j] = k
                    data.eq_active[self._weld[(k, j)]] = 1
                    print(f"lightbulb: spare {k} engaged in fixture {j}")
                    break
        self.fell = self.fell or self.base_fallen()
        self.done_ever |= {i for i in range(N_FIXTURES) if self.tightened(i)}

    def _step_and_record(self, action) -> None:
        super()._step_and_record(action)
        self._update_progress()

    def complete(self) -> list[int]:
        return [i for i in range(N_FIXTURES) if self.tightened(i)]

    def judge_success(self) -> bool:
        return bool(len(self.complete()) == N_FIXTURES and not self.fell)

    def get_reward(self) -> np.ndarray:
        """Mean over fixtures of progress: turn / seat angle (a fixture still empty scores 0)."""
        prog = [
            0.0
            if (self.is_empty(i) and i not in self.captured_by)
            else float(np.clip(self.turn_deg(i) / BULB_SEATED_DEG, 0, 1))
            for i in range(N_FIXTURES)
        ]
        return np.array([float(np.mean(prog))])

    def get_task_description(self) -> str:
        return (
            "Tighten the loose light bulbs on the wall and floor boards, and screw spare bulbs "
            "from the rack into the empty fixtures."
        )

    def bulb_status(self) -> dict[int, dict]:
        out = {}
        for i in range(N_FIXTURES):
            if self.is_empty(i):
                state = (
                    ("placed" if self.tightened(i) else "engaged")
                    if i in self.captured_by
                    else "empty"
                )
            else:
                state = "tightened" if self.tightened(i) else "loose"
            out[i] = {
                "board": "wall" if i in WALL_FIXTURES else "floor",
                "state": state,
                "turn_deg": round(self.turn_deg(i), 1),
            }
        return out

    def status(self) -> dict:
        bulbs = self.bulb_status()
        s = {
            "bulbs": bulbs,
            "tightened": [i for i, b in bulbs.items() if b["state"] == "tightened"],
            "placed": [i for i, b in bulbs.items() if b["state"] == "placed"],
            "ever_done": sorted(self.done_ever),
            "fell": self.fell,
            "success": self.judge_success(),
        }
        print(
            "bulbs: "
            + " ".join(f"{i}:{b['state']}({b['turn_deg']:.0f})" for i, b in bulbs.items())
            + f" | done={len(self.complete())}/{N_FIXTURES} ever_done={sorted(self.done_ever)} "
            + f"fell={self.fell} success={s['success']}"
        )
        return s

    # ---- shell hooks ---------------------------------------------------------------------
    def _shell_extras(self) -> dict:
        return {
            "status": self.status,
            "turn_bulb": self.turn_bulb,
            "place_bulb": self.place_bulb,
            "solve_lightbulb": self.solve_lightbulb,
        }

    def _bulb_solver(self):
        from molmo_spaces.policy.solvers.object_manipulation.lightbulb_solver import (
            Dex3LightbulbSolver,
        )

        if "left_hand" not in self._env.current_robot.robot_view.move_group_ids():
            raise NotImplementedError("lightbulb solver needs the dual-Dex3 G1 (config *G1Dex3*)")
        if getattr(self, "_solver", None) is None:
            self._solver = Dex3LightbulbSolver(self)
        return self._solver

    def turn_bulb(self, fixture: int) -> bool:
        """Tighten the loose bulb in fixture `fixture` (an installed one: 0, 1, 3 or 5-9)."""
        return self._bulb_solver().turn_bulb(fixture)

    def place_bulb(self, fixture: int) -> bool:
        """Fetch the nearest spare bulb from the rack and screw it into empty fixture `fixture`."""
        return self._bulb_solver().place_bulb(fixture)

    def solve_lightbulb(self) -> bool:
        """Tighten every installed bulb, then place a spare in each empty fixture."""
        return self._bulb_solver().solve()

    def _shell_banner_extra(self) -> list[str]:
        return [
            " challenge (lightbulb):",
            f"  task: {self.get_task_description()}",
            "  status()                               - per-bulb state and turn angle",
            "  turn_bulb(i)                           - tighten the loose bulb in fixture i",
            "  place_bulb(i)                          - fetch a spare from the rack, screw into empty fixture i",
            "  solve_lightbulb()                      - tighten all, then fill the empty fixtures",
        ]
