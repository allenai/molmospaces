"""Scripted solver for the challenge keypad: press the code's buttons with a closed gripper.

Structured like the G1 grasp planner (pick_planner_policy_g1.GraspPlanner): per button an
APPROACH to a pre-press standoff in front of it, a PUSH along the keypad normal, and a
RETRACT back to the standoff. Every policy tick the hand target is interpolated in Cartesian
space (position lerp, orientation slerp) starting from the hand's *current* pose, and whole-body
IK (`G1Robot.kinematics_wbc`, self-collision avoidance on) turns it into arm + waist joint
targets and a standing-height command, which is smoothed like the planner does. The gripper
stays closed throughout, so the fingertips do the pushing.
"""

import logging

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from molmo_spaces.tasks.challenge_keypad_door_task import KEY_RELEASED_DEPTH

log = logging.getLogger(__name__)

# Grasp-site frame for pushing along the base's forward axis: z forward, x up (right hand).
PUSH_ROT_LOCAL = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
# Hand pointing straight down with the jaws (which separate along the site's y axis) opening
# along world x: one finger goes in front of a thin vertical-plane bar, the other behind it, so
# pulling toward +x hooks the bar instead of relying on friction. World-fixed (no yaw term).
GRASP_ROT_WORLD = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
HEIGHT_GAIN = 0.35  # standing-height command smoothing per policy tick (planner: 0.1 per 15ms)


class KeypadPressSolver:
    STANDOFF = 0.18  # m in front of the key face for the pre-press pose (clears the wrist link)
    PUSH_DEPTH = 0.010  # m past the face the push target sits (button travel is 8mm)
    APPROACH_TICKS = 40
    PUSH_TICKS = 30
    RETRACT_TICKS = 20
    HOLD_TICKS = 4
    PUSH_STEP = 0.002  # m the push target advances per tick
    LATERAL_GAIN = 0.06  # share of the live lateral error added to the push target per tick
    MAX_LATERAL = 0.04  # m, cap on the correction accumulated during one push
    MAX_OVERTRAVEL = 0.03  # m past the key face the hand may go before the push is abandoned
    MAX_ATTEMPTS = 1  # one push per key; no retries
    debug = False
    gripper_open = False  # the keypad is pressed with a closed gripper
    hand = "right"  # which arm/hand does the work ("left" or "right")
    keep_station = True  # walk the base back toward where it stood; off = stand still
    COMP_GAIN = 0.8  # share of the tracking error added to the IK target
    MAX_COMP = 0.05  # m, cap on that correction

    def __init__(self, task) -> None:
        self.task = task
        self._height_cmd: float | None = None
        self._hold: np.ndarray | None = None  # base [x, y, yaw] to keep while the arm works

    # ---- geometry --------------------------------------------------------------------------
    def _face(self, digit: int) -> np.ndarray:
        model, data = self.task._env.current_model, self.task._env.current_data
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"key{digit}_col")
        return data.geom_xpos[gid] + [model.geom_size[gid][0], 0.0, 0.0]  # front face (+x)

    def _site(self) -> tuple[np.ndarray, np.ndarray]:
        model, data = self.task._env.current_model, self.task._env.current_data
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot_0/{self.hand}_grasp")
        return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()

    def _yaw_rot(self) -> np.ndarray:
        yaw = self.task._base_xy_yaw()[2]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _push_rot(self) -> np.ndarray:
        return self._yaw_rot() @ PUSH_ROT_LOCAL

    def _grasp_rot(self) -> np.ndarray:
        return GRASP_ROT_WORLD.copy()

    # ---- one controlled move ---------------------------------------------------------------
    def _closed_gripper_ctrl(self) -> np.ndarray:
        """Gripper target for the current mode (closed for pressing; `gripper_open` to grasp)."""
        gripper = self.task._env.current_robot.robot_view.get_gripper("right_gripper")
        gripper.set_gripper_ctrl_open(self.gripper_open)
        return np.asarray(gripper.ctrl, dtype=np.float32).copy()

    def _hand_action(self) -> dict:
        """Action entries that command the end effector (Dex1: its closed/open gripper)."""
        return {"right_gripper": self._closed_gripper_ctrl()}

    def _move(
        self, goal_pos: np.ndarray, goal_rot: np.ndarray, ticks: int, precision: bool = False, until=None, base_cmd=None,
        comp: bool | None = None,
    ):
        """Interpolate the hand from where it is to the goal over `ticks` policy ticks (stopping
        early once `until()` is true). `precision` is the contact phase: tighter IK, and no
        error compensation (winding the target up against a wall would shove the robot)."""
        task = self.task
        robot = task._env.current_robot
        base_cmd_in = base_cmd
        start_pos, start_rot = self._site()
        slerp = Slerp([0, 1], R.concatenate([R.from_matrix(start_rot), R.from_matrix(goal_rot)]))
        hand_action = self._hand_action()
        for i in range(1, ticks + 1):
            t = i / ticks
            pos = start_pos * (1 - t) + goal_pos * t
            # The arm tracks with a few cm of lag/droop: aim past the target by the error.
            lag = np.zeros(3) if (precision if comp is None else not comp) else np.clip(pos - self._site()[0], -self.MAX_COMP, self.MAX_COMP)
            arm, waist, ik_h, ik_err = robot.kinematics_wbc(
                pos + self.COMP_GAIN * lag,
                slerp(t).as_matrix(),
                hand=self.hand,
                avoid_self_collision=True,
                precision=precision,
            )
            if self._hold is None:
                self._hold = np.array(task._base_xy_yaw())
            # Station-keeping: the legs walk back to where the solve started, so the reaching
            # arm does not drag the base (and its heading) around.
            if base_cmd_in is not None:
                base_cmd = base_cmd_in
            elif self.keep_station:
                base_cmd = robot.waypoint_to_velocity_target(self._hold)
            else:
                base_cmd = np.zeros(3, dtype=np.float32)
            if self._height_cmd is None:
                self._height_cmd = ik_h
            self._height_cmd += HEIGHT_GAIN * (ik_h - self._height_cmd)
            task._step_and_record(
                {
                    f"{self.hand}_arm": arm,
                    "legs_waist": np.array([*base_cmd, self._height_cmd, *waist], dtype=np.float32),
                    **hand_action,
                }
            )
            if task.viewer is not None:
                task.viewer.sync()
            if until is not None and until():
                return True
            if self.debug:
                bx, by, byaw = task._base_xy_yaw()
                z = robot.robot_view.base.pose[2, 3]
                print(
                    f"  t={i}/{ticks} base=({bx:.3f},{by:.3f},{np.degrees(byaw):.0f}deg,z={z:.2f}) "
                    f"site_err={np.linalg.norm(self._site()[0] - pos) * 100:.1f}cm "
                    f"d=({np.round((self._site()[0] - pos) * 100, 1)}) "
                    f"ik_err={ik_err * 100:.1f}cm ik_h={ik_h:.2f} h_cmd={self._height_cmd:.2f} base_cmd={np.round(base_cmd, 2)}"
                )

        return False

    def _servo(self, goal: np.ndarray, rot_fn, iters: int = 6, tol: float = 0.005, offset=None):
        """Settle the hand on `goal` to within `tol`: the arm sags/lags by a steady cm or two, so
        keep integrating the remaining error into the IK target. Returns the integrated target
        offset (add it to later goals nearby so the sag does not come back) and the error."""
        off = np.zeros(3) if offset is None else np.asarray(offset, dtype=float).copy()
        err = np.inf
        for _ in range(iters):
            self._move(goal + off, rot_fn(), 8, precision=True, comp=False)
            err_vec = goal - self._site()[0]
            err = float(np.linalg.norm(err_vec))
            if err < tol:
                break
            off = off + 0.9 * err_vec
        return off, err

    # ---- skills ----------------------------------------------------------------------------
    def press(self, digit: int) -> bool:
        """Press button `digit` once."""
        task = self.task
        self._hold = None  # hold wherever the base stands now
        face = self._face(digit)
        presses_before = len(task.press_log)
        rot = self._push_rot()
        pre = face + [self.STANDOFF, 0.0, 0.0]
        self._move(pre, rot, self.APPROACH_TICKS)
        registered = lambda: len(task.press_log) > presses_before  # noqa: E731
        ok = self._push(face, registered)
        self._release(digit, pre)
        print(f"press({digit}): {'registered' if ok else 'MISSED'}")
        return ok

    def _push(self, face: np.ndarray, registered) -> bool:
        """One continuous push: line the fingertip up with the key, then advance the target along
        -x each tick from the live state until the key's own slide joint registers the press.
        Aborts if the base falls or the hand goes `MAX_OVERTRAVEL` past the face."""
        task = self.task
        pre, _ = self._site()
        off, _ = self._servo(np.array([pre[0], face[1], face[2]]), self._push_rot)
        off = off0 = off * [0, 1, 1]
        x = pre[0]
        while x > face[0] - self.MAX_OVERTRAVEL:
            x -= self.PUSH_STEP
            # The arm sags further as it pushes: keep integrating the lateral error.
            off = off + self.LATERAL_GAIN * (face - self._site()[0]) * [0, 1, 1]
            off = np.clip(off, off0 - self.MAX_LATERAL, off0 + self.MAX_LATERAL)
            if self._move(np.array([x, face[1], face[2]]) + off, self._push_rot(), 1, precision=True, until=registered):
                return True
            if task._env.current_robot.robot_view.base.pose[2, 3] < 0.5:
                return False
        return registered()

    def _release(self, digit: int, pre: np.ndarray) -> None:
        """Retract to the standoff, then keep going until the key is back up past the release depth."""
        task = self.task
        up = lambda: -task._q(f"key{digit}_slide") <= KEY_RELEASED_DEPTH  # noqa: E731
        self._move(pre, self._push_rot(), self.RETRACT_TICKS, until=up)
        self._move(pre, self._push_rot(), self.RETRACT_TICKS)

    def solve(self, code: list[int] | None = None) -> bool:
        """Press the whole code in order, one push per digit. No retries: a missed or wrong
        key is left as is and the caller moves on (the door does not depend on the keypad)."""
        code = list(self.task.code if code is None else code)
        for digit in code:
            self.press(digit)
        done = self.task.keypad_done
        print(f"keypad {'solved' if done else 'NOT solved'}: entered={self.task.entered}")
        return done


class Dex3KeypadPressSolver(KeypadPressSolver):
    """The same approach / push / retract, with a Dex3-1 hand: index finger extended and
    pointing (the `right_grasp` site is its fingertip), middle finger and thumb curled out of
    the way so only the index touches the keys."""

    # thumb_0..2, index_0..1, middle_0..1 (right hand: positive flexion for index / middle,
    # negative for thumb_2). Index extended = 0; middle fully curled.
    POINT_POSE = np.array([0.0, -0.5, -1.2, 0.0, 0.0, 1.55, 1.7], dtype=np.float32)

    def _hand_action(self) -> dict:
        return {"right_hand": self.POINT_POSE}
