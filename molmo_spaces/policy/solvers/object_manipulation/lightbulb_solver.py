"""Scripted lightbulb turning for the dual-Dex3 G1 (`g1_dex3`): challenge manipulation task 3.

* `turn_bulb(i)`: stand so the right hand lines up with the bulb in fixture `i`, then repeat
  grasp -> twist -> release -> hand back until the bulb has been turned until it stops. A cycle wraps
  the three digits around the globe from the side (fingers along the bulb toward the board, palm
  toward the globe), rolls the hand rigidly about the bulb's axis so the globe stays put in the hand,
  opens, lifts clear and un-rotates. A wrist rolls ~100 degrees, so a bulb needs several cycles. A
  cycle that does not turn the bulb stops the action; nothing is repeated blindly.
* `place_bulb(i)`: pick the nearest spare bulb out of the rack from above (palm down), tuck the arm
  and walk to the empty fixture, re-pose the held bulb (in-hand pose is rigid, so the hand pose
  follows from the bulb's pose) to line it up with the fixture's mouth, push it in until the task
  engages the thread, let go and screw it in with `turn_bulb`.

Everything drives the hand with closed-loop whole-body IK from the live state (the arm sags 1-3 cm,
so `_servo` integrates the remaining error), reusing KeypadPressSolver's `_move` / `_servo`. Low
fixtures are reached by the IK's own standing-height command and waist bend.

Status (see the task docstring and the commit message for the seed-by-seed numbers): turning works on
the wall board, but not reliably (the grip is a two- or three-contact friction grip on a smooth
sphere, and the arm is at the edge of its reach for the top row); the floor-board bulbs fail (the
squat with an arm out topples the robot); `place_bulb` does not work yet: the fingers closing on a
free spare push it out of its rack hole before they grip it.
"""

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.policy.solvers.object_manipulation.keypad_press_solver import KeypadPressSolver
from molmo_spaces.tasks.challenge_lightbulb_task import (
    BULB_BOTTOMED_DEG,
    EMPTY_FIXTURES,
    N_SPARES,
)

log = logging.getLogger(__name__)

# Right hand (Dex3-1) joints: thumb_0..2, middle_0..1, index_0..1. Right curls are positive for the
# fingers; the thumb closes toward the fingers with negative thumb_1 / thumb_2. GRASP_POSE wraps a
# 62 mm globe with the three digits and a few mm of squeeze (found by checking the hand meshes'
# distance to the globe over a grid of poses).
OPEN_POSE = np.zeros(7, dtype=np.float32)
GRASP_POSE = np.array([0.3, -0.9, -0.9, 1.2, 1.6, 1.2, 1.6], dtype=np.float32)

# Where the globe's centre sits in the grasp-site frame (z along the fingers, y out of the palm,
# x from index to middle finger) for that wrap grasp.
BALL_IN_SITE = np.array([0.028, 0.075, -0.08])
GLOBE_T = 0.072  # m from a bulb's base to the centre of its globe
# Grasp-site rotations (columns = site axes in the world), fingers pointing forward (+x):
PALM_LEFT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # handshake: globe to the left
PALM_DOWN = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # globe below the hand
# Taking a spare out of the rack: the hand comes in over the rack pointing across it (-y), palm down, so
# the thumb (which hangs 7 cm behind the globe) is not over the neighbouring spares 10 cm along the rack.
PICK_YAW = -90.0
PICK_ROT = R.from_euler("z", PICK_YAW, degrees=True).as_matrix() @ PALM_DOWN


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    return R.from_rotvec(np.asarray(axis, dtype=float) * np.radians(deg)).as_matrix()


def _grip_rot(w: np.ndarray, psi_deg: float) -> np.ndarray:
    """Grasp-site rotation for a bulb whose tightening axis is `w` (pointing into the board): the
    handshake pose PALM_LEFT carried over so the fingers point along `w` (toward the fixture), as
    they do for a bulb on the wall, then rolled `psi_deg` about `w`."""
    x = np.array([1.0, 0.0, 0.0])
    axis = np.cross(x, w)
    if np.linalg.norm(axis) < 1e-9:
        align = np.eye(3)
    else:
        align = _rot(axis / np.linalg.norm(axis), np.degrees(np.arccos(np.clip(x @ w, -1, 1))))
    return _rot(w, psi_deg) @ align @ PALM_LEFT


class Dex3LightbulbSolver(KeypadPressSolver):
    hand = "right"
    keep_station = False  # base corrections with the arm out tip the robot over
    TWIST_DEG = 90.0  # hand roll per cycle (deg)
    HIGH_Z = 1.1  # m, bulbs whose globe is above this are at the arm's reach limit ...
    HIGH_TWIST_DEG = 60.0  # ... and use a smaller roll that starts at ROLL_START_HIGH
    ROLL_START_HIGH = -90.0
    TWIST_TICKS = 50
    MAX_CYCLES = 14
    MIN_PROGRESS_DEG = 3.0  # a cycle that turns the bulb less than this ends the action
    ROLL_START = -110.0  # hand roll (deg about the bulb axis, from the handshake pose) a twist starts at: palm down and beyond
    REACH_BACK = 0.38  # m the base stands behind the grasp site
    RIGHT_OFFSET = 0.21  # m the right hand sits to the right of the base centre
    GRIP_SETTLE_TICKS = 20
    SETTLE_TICKS = 40  # policy ticks the legs get to stop after a walk
    STAND_TICKS = 400  # policy ticks the walk may take
    WALK_MIN, WALK_MAX = 0.15, 0.3  # m/s, the WBC does not walk below ~0.1 m/s
    LIFT = 0.07  # m the open hand lifts clear of the globe between twists
    RACK_PICK_X = 7.6  # m, where along the rack spares are taken from
    PICK_CLOSE_TICKS = 60  # a free bulb is pushed off its hole by a fast close
    PICK_LIFT = 0.12  # m a spare is raised out of its rack hole

    _pose = OPEN_POSE

    def _hand_action(self) -> dict:
        return {"right_hand": self._pose}

    # ---- helpers ---------------------------------------------------------------------------
    def _set_pose(self, pose: np.ndarray, ticks: int) -> None:
        """Ramp the right hand's fingers to `pose` while the arm holds where it is."""
        a = self._pose.copy()
        pos, rot = self._site()
        for i in range(1, ticks + 1):
            self._pose = a + (pose - a) * (i / ticks)
            self._move(pos, rot, 1, precision=True, comp=False)
        self._pose = pose.copy()

    def _hold_still(self, ticks: int) -> None:
        pos, rot = self._site()
        for _ in range(ticks):
            self._move(pos, rot, 1, precision=True, comp=False)

    def _site_for(self, centre: np.ndarray, rot: np.ndarray) -> np.ndarray:
        return centre - rot @ BALL_IN_SITE

    def _check_fell(self) -> bool:
        if self.task.base_fallen():
            print("lightbulb: the robot has fallen")
            return True
        return False

    def _stand_for(self, site: np.ndarray, yaw_deg: float = 0.0) -> None:
        """Walk so the right hand (at rest, 0.21 m right of the base centre, REACH_BACK ahead of the
        site) lines up with `site`, facing `yaw_deg`. The deck around the boards is tight, so it is
        a straight line, no grid planner, walked with the base velocity command until the base is
        within 4 cm / 4 deg (the shell's waypoint walk only stops within 15 cm)."""
        task = self.task
        robot = task._env.current_robot
        x, y, yaw = task._base_xy_yaw()
        c, s_ = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        base_to_site = np.array(
            [c * self.REACH_BACK + s_ * self.RIGHT_OFFSET, s_ * self.REACH_BACK - c * self.RIGHT_OFFSET]
        )
        goal = site[:2] - base_to_site
        yaw_goal = np.radians(yaw_deg)
        d_yaw = (np.degrees(yaw) - yaw_deg + 180) % 360 - 180
        if np.linalg.norm(goal - [x, y]) > 0.03 or abs(d_yaw) > 4:
            height = float(robot._low_level._height_cmd)
            for _ in range(self.STAND_TICKS):
                x, y, yaw = task._base_xy_yaw()
                to_goal = goal - [x, y]
                err = float(np.linalg.norm(to_goal))
                yaw_err = (yaw_goal - yaw + np.pi) % (2 * np.pi) - np.pi
                if err < 0.04 and abs(yaw_err) < np.radians(4):
                    break
                # Slow constant-speed walk toward the goal (turn first if heading is far off).
                speed = float(np.clip(err, self.WALK_MIN, self.WALK_MAX)) if err > 0.04 else 0.0
                v_world = to_goal / max(err, 1e-6) * speed
                vx = np.cos(yaw) * v_world[0] + np.sin(yaw) * v_world[1]
                vy = -np.sin(yaw) * v_world[0] + np.cos(yaw) * v_world[1]
                wz = float(np.clip(yaw_err, -0.3, 0.3)) if abs(yaw_err) > np.radians(4) else 0.0
                if abs(yaw_err) > np.radians(30):
                    vx = vy = 0.0
                cmd = np.array([vx, vy, wz], dtype=np.float32)
                task._step_and_record(
                    {"legs_waist": np.array([*cmd, height, 0, 0, 0], dtype=np.float32)}
                )
            # The base keeps chasing its last waypoint and can spin on the spot; hold the pose it
            # stopped at so the gait comes to rest before the arm moves.
            hold = np.array(task._base_xy_yaw())
            for _ in range(self.SETTLE_TICKS):
                task._step_and_record({"base": hold})
        self._hold = None
        self._height_cmd = None

    def _rest(self, ticks: int = 40, keep_grip: bool = False) -> None:
        """Return the right arm and the standing height to rest; the hand opens unless `keep_grip`."""
        task = self.task
        robot = task._env.current_robot
        view = robot.robot_view
        a0 = {"right": np.asarray(view.get_move_group("right_arm").joint_pos, np.float32)}
        h0 = float(robot._low_level._height_cmd)
        p0 = self._pose.copy()
        end = p0 if keep_grip else OPEN_POSE
        for i in range(1, ticks + 1):
            f = i / ticks
            action = {
                "legs_waist": np.array([0, 0, 0, h0 + f * (0.793 - h0), 0, 0, 0], dtype=np.float32),
                "right_hand": p0 + f * (end - p0),
            }
            action["right_arm"] = a0["right"] + f * (task.rest_arms["right"] - a0["right"])
            task._step_and_record(action)
        self._pose = np.array(end, dtype=np.float32)
        self._height_cmd = None

    # ---- turning ---------------------------------------------------------------------------
    def _approach(self, centre: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Open hand to a pre-grasp pose beside and behind the globe, then onto it. Returns the
        integrated IK offset of the final pose."""
        grasp = self._site_for(centre, rot)
        pre = grasp - 0.07 * rot[:, 1] - 0.10 * rot[:, 2]
        self._pose = OPEN_POSE.copy()
        self._move(pre, rot, self.APPROACH_TICKS, comp=True)
        off, err = self._servo(pre, lambda: rot)
        self._move(grasp + off, rot, self.PUSH_TICKS, precision=True, comp=False)
        off, err = self._servo(grasp, lambda: rot, offset=off)
        if self.debug:
            print(f"  grasp pose: err={err * 100:.1f}cm")
        return off

    def _twist_about(self, centre: np.ndarray, w: np.ndarray, deg: float, off: np.ndarray) -> None:
        """Roll the hand rigidly by `deg` about the line through `centre` with direction `w`."""
        pos0, rot0 = self._site()
        rv = R.from_rotvec(w * np.radians(deg))
        goal = centre + rv.apply(pos0 - centre)
        self._move(
            goal + off, rv.as_matrix() @ rot0, self.TWIST_TICKS, precision=True, comp=False
        )  # fmt: skip
        if self.debug:
            pos, rot = self._site()
            roll = np.degrees(R.from_matrix(rot @ rot0.T).as_rotvec() @ w)
            print(f"  twist: hand rolled {roll:.0f} deg about the axis (asked {deg:.0f}), {np.linalg.norm(pos - goal) * 100:.1f}cm off the arc")

    def _cycle(self, fixture: int, centre: np.ndarray, w: np.ndarray, psi: float, twist: float) -> float:
        """One grasp -> twist -> release -> return cycle; returns the bulb's turn (deg) gained."""
        task = self.task
        t0 = task.turn_deg(fixture)
        rot = _grip_rot(w, psi)
        task.phase = "approach"
        off = self._approach(centre, rot)
        task.phase = "grip"
        self._set_pose(GRASP_POSE, 30)
        self._hold_still(self.GRIP_SETTLE_TICKS)  # let the contacts settle before turning
        if self.debug:
            q = task._env.current_robot.robot_view.get_move_group("right_hand").joint_pos
            print(f"  grip: finger joints {np.round(q, 2)} (commanded {np.round(GRASP_POSE, 2)})")
        pos0, rot0 = self._site()
        task.phase = "twist"
        self._twist_about(centre, w, twist, off)
        gained = task.turn_deg(fixture) - t0
        task.phase = "open"
        self._set_pose(OPEN_POSE, 30)
        self._hold_still(10)  # let the fingers finish opening
        pos, rot_now = self._site()
        # Lift clear (away from the globe) and un-rotate, open, to the starting roll.
        task.phase = "lift"
        away = -rot0[:, 1]
        self._move(pos + self.LIFT * away, rot_now, 25, precision=True, comp=False)
        task.phase = "unrotate"
        self._move(pos0 + self.LIFT * away, rot0, self.TWIST_TICKS, comp=False)
        print(f"  twist cycle: bulb {fixture} turned {gained:.0f} deg (now {task.turn_deg(fixture):.0f})")
        return gained

    def _bulb_frame(self, fixture: int) -> tuple[np.ndarray, np.ndarray]:
        """(globe centre, tightening axis) of the bulb in `fixture` at its current turn."""
        p, Rm = self.task.fixture_pose(fixture)
        return p + Rm[:, 2] * GLOBE_T, -Rm[:, 2]

    def turn_bulb(self, fixture: int) -> bool:
        """Tighten the bulb in `fixture` (an installed one, or a spare already engaged in it)."""
        task = self.task
        if task.is_empty(fixture) and fixture not in task.captured_by:
            print(f"turn_bulb({fixture}): fixture is empty, use place_bulb")
            return False
        centre, w = self._bulb_frame(fixture)
        self._hold = None
        high = centre[2] > self.HIGH_Z
        psi = self.ROLL_START_HIGH if high else self.ROLL_START
        twist = self.HIGH_TWIST_DEG if high else self.TWIST_DEG
        self._stand_for(self._site_for(centre, _grip_rot(w, psi)))
        for _ in range(self.MAX_CYCLES):
            if task.turn_deg(fixture) >= BULB_BOTTOMED_DEG:  # turned until it stops: tight
                break
            centre, w = self._bulb_frame(fixture)
            gained = self._cycle(fixture, centre, w, psi, twist)
            if self._check_fell() or gained < self.MIN_PROGRESS_DEG:
                break
        self._rest()
        ok = task.tightened(fixture)
        print(f"turn_bulb({fixture}): {'tightened' if ok else 'NOT tightened'} ({task.turn_deg(fixture):.0f} deg)")
        return ok

    # ---- placing ---------------------------------------------------------------------------
    def _free_spare(self) -> int | None:
        """The rack spare nearest the middle of the rack (its ends are next to a post and to the floor
        board, where the legs would hit) that is not engaged anywhere and still stands in the rack."""
        task = self.task
        best = None
        for k in range(N_SPARES):
            if k in task.captured_by.values():
                continue
            p, Rm = task.spare_pose(k)
            if Rm[2, 2] < 0.9 or p[1] > 0.3:  # knocked over / out of the rack
                continue
            if best is None or abs(p[0] - self.RACK_PICK_X) < best[0]:
                best = (abs(p[0] - self.RACK_PICK_X), k)
        return None if best is None else best[1]

    def _pick_spare(self, k: int) -> bool:
        """Palm down over spare `k`, close on it, lift it out of the rack."""
        task = self.task
        p, Rm = task.spare_pose(k)
        centre = p + Rm[:, 2] * GLOBE_T
        self._hold = None
        self._stand_for(self._site_for(centre, PICK_ROT), PICK_YAW)
        task.phase = "pick"
        self._approach(centre, PICK_ROT)
        self._set_pose(0.55 * GRASP_POSE, self.PICK_CLOSE_TICKS)  # first touch the globe,
        self._hold_still(15)
        self._set_pose(GRASP_POSE, self.PICK_CLOSE_TICKS)  # then squeeze
        pos, rot = self._site()
        if self.debug:
            p_in = task.spare_pose(k)[0]
            print(f"  pick: after closing the bulb is {np.round(R.from_matrix(rot).inv().apply(p_in - pos), 3)} from the site (site frame), z={p_in[2]:.3f} (was {p[2]:.3f})")
        for i in range(4):
            self._move(pos + [0, 0, self.PICK_LIFT * (i + 1) / 4], rot, 10, precision=True, comp=False)
            if self.debug:
                print(f"  pick: lift step {i}: spare z={task.spare_pose(k)[0][2]:.3f} hand z={self._site()[0][2]:.3f}")
        lifted = task.spare_pose(k)[0][2] - p[2]
        print(f"  pick: spare {k} lifted {lifted * 100:.0f} cm")
        return lifted > 0.6 * self.PICK_LIFT

    def _in_hand(self, k: int) -> np.ndarray:
        """Pose of spare `k` in the grasp-site frame (4x4)."""
        p_s, R_s = self.task.spare_pose(k)
        p_h, R_h = self._site()
        T = np.eye(4)
        T[:3, :3] = R_h.T @ R_s
        T[:3, 3] = R_h.T @ (p_s - p_h)
        return T

    def _insert(self, k: int, fixture: int) -> bool:
        """Line the held spare up with the fixture's mouth and push it in until the thread engages."""
        task = self.task
        p_f, R_f = task.fixture_pose(fixture)
        a_f = R_f[:, 2]
        T_hs = self._in_hand(k)
        R_hs, p_hs = T_hs[:3, :3], T_hs[:3, 3]

        def site_for_bulb(p_bulb: np.ndarray) -> np.ndarray:
            return p_bulb - (R_f @ R_hs.T) @ p_hs

        rot = R_f @ R_hs.T
        task.phase = "insert"
        pre = p_f + 0.08 * a_f
        self._move(site_for_bulb(pre), rot, 60, comp=True)
        off = np.zeros(3)
        for _ in range(4):  # servo the bulb (not the hand) onto the pre-insert point
            off += 0.9 * (pre - task.spare_pose(k)[0])
            self._move(site_for_bulb(pre) + off, rot, 8, precision=True, comp=False)
        engaged = lambda: fixture in task.captured_by  # noqa: E731
        for depth in (0.05, 0.02, 0.0):  # slide in along the axis, correcting from the bulb's pose
            tgt = p_f + depth * a_f
            off += 0.9 * (tgt - task.spare_pose(k)[0])
            self._move(site_for_bulb(tgt) + off, rot, 20, precision=True, comp=False, until=engaged)
            if engaged():
                break
        print(f"  insert: spare {k} {'engaged in' if engaged() else 'did NOT engage'} fixture {fixture}")
        return engaged()

    def place_bulb(self, fixture: int) -> bool:
        """Fetch the nearest spare from the rack and screw it into empty fixture `fixture`."""
        task = self.task
        if not task.is_empty(fixture) or fixture in task.captured_by:
            print(f"place_bulb({fixture}): fixture is not empty")
            return False
        k = self._free_spare()
        if k is None:
            print("place_bulb: no spare bulb left in the rack")
            return False
        if not self._pick_spare(k):
            print(f"place_bulb({fixture}): could not pick up spare {k}")
            self._rest()
            return False
        self._rest(keep_grip=True)  # arm tucked, bulb in hand, for the walk
        p_f, R_f = task.fixture_pose(fixture)
        self._stand_for(p_f + R_f[:, 2] * 0.10)
        if not self._insert(k, fixture):
            self._set_pose(OPEN_POSE, 20)
            self._rest()
            return False
        self._set_pose(OPEN_POSE, 20)
        self._hold_still(10)
        pos, rot = self._site()
        self._move(pos - 0.08 * rot[:, 2], rot, 30, precision=True, comp=False)  # back off along the fingers
        return self.turn_bulb(fixture)

    def solve(self) -> bool:
        """Wall board first (tighten, then fill its empty fixtures), then the floor board. Each bulb
        is one action; the sequence stops if the robot falls."""
        task = self.task
        installed = [i for i in range(10) if not task.is_empty(i)]
        steps = (
            [("turn", i) for i in installed if i < 5]
            + [("place", j) for j in EMPTY_FIXTURES]
            + [("turn", i) for i in installed if i >= 5]
        )
        for kind, i in steps:
            (self.turn_bulb if kind == "turn" else self.place_bulb)(i)
            if task.fell:
                print("solve_lightbulb: stopping, the robot fell")
                break
        return task.status()["success"]
