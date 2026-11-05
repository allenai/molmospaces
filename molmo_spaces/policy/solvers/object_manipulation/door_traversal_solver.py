"""Scripted door traversal for the dual-Dex3 G1 (`g1_dex3`): lever, open, walk through.

The door opens toward the start side (rules PDF, "view from the starting location"): hinged on
the left (low y), lever on the right, so from the start it is a pull door. From a stand to the
right of the doorway the robot's *left* hand lines up with the lever bar.

1. Left hand: palm down, fingers forward, over the thin grip bar; close the fingers so the
   curled fingertips go behind the bar (a hook) and the palm presses on top of it.
2. Pull the bar down (turning the lever, releasing the latch), then drag it back along the
   door's arc with the base following, until the door stands open.
3. Release, arms back to rest, walk around the leaf's free edge into the doorway and through,
   clear of the swing arc; the hinge spring closes the door behind.

Reuses KeypadPressSolver's closed-loop hand controller (`_move`, `_servo`).
"""

import logging

import mujoco
import numpy as np

from molmo_spaces.policy.solvers.object_manipulation.keypad_press_solver import KeypadPressSolver
from molmo_spaces.tasks.challenge_keypad_door_task import LEVER_TURNED

log = logging.getLogger(__name__)

# Left-hand site frame while hooking the lever: z = forward (finger direction), y = the side the
# fingers curl toward = down, x = y cross z. Robot-local (forward +x, left +y, up +z).
HOOK_ROT_LOCAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)

# thumb_0..2, index_0..1, middle_0..1 -- left hand curls with negative index/middle angles.
OPEN_POSE = np.zeros(7, dtype=np.float32)
HOOK_POSE = np.array([0.0, 0.0, 0.0, -1.35, -1.5, -1.35, -1.5], dtype=np.float32)


class DoorTraversalSolver(KeypadPressSolver):
    hand = "left"
    keep_station = False  # base corrections with the arm out tip the robot over
    LEVER_PRESS_ANGLE = 0.45  # rad the lever is turned to (latch needs LEVER_TURNED)
    LEVER_DROP = 0.045  # m the hooked grip is pulled down to turn the lever
    OVER_BAR = 0.025  # m the fingertip site starts above the bar's top surface
    FINGER_REACH = 0.04  # m the fingertip sits past the bar, toward the door, before closing
    PULL_TO = 1.15  # rad (~66 deg): the free edge is then well clear of the right jamb
    PULL_RATE = 0.012  # rad of door angle commanded per policy tick
    PULL_MAX_TICKS = 300
    WALK_GAIN = 2.0  # base step toward the hand's target per metre of hand error
    LEVER_HOLD = 0.45  # rad the lever is held at while pulling (limit is 0.9)
    LIFT_GAIN = 0.02  # m of hand height correction per rad of lever error per tick
    THROUGH_X = 8.0  # base x just past the door plane (x=8.59); the swing arc is on the start side.
    # Behind the door the dexterity layout's deck is flat (mobility terrain removed in the scene converter).
    THROUGH_Y = 1.40  # walk-through line in the doorway (centre 1.30, lever side is y=1.75)

    _pose = OPEN_POSE

    def _push_rot(self) -> np.ndarray:
        return self._yaw_rot() @ HOOK_ROT_LOCAL

    def _hand_action(self) -> dict:
        return {"left_hand": self._pose}

    def _set_pose(self, pose: np.ndarray, ticks: int) -> None:
        """Ramp the left hand's fingers to `pose` while the arm holds where it is."""
        a = self._pose.copy()
        pos, rot = self._site()
        for i in range(1, ticks + 1):
            self._pose = a + (pose - a) * (i / ticks)
            self._move(pos, rot, 1, precision=True, comp=False)
        self._pose = pose.copy()

    # ---- geometry ---------------------------------------------------------------------------
    def _lever_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World AABB (min, max) of the lever's grip bar: the handle collision geom reaching
        furthest out from the door (+x). It extends toward the hinge (-y) from the neck."""
        model, data = self.task._env.current_model, self.task._env.current_data
        best = None
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.startswith("handle_col_"):
                continue
            rot = data.geom_xmat[gid].reshape(3, 3)
            c = data.geom_xpos[gid] + rot @ model.geom_aabb[gid, :3]
            half = np.abs(rot) @ model.geom_aabb[gid, 3:]
            if best is None or c[0] + half[0] > best[1][0]:
                best = (c - half, c + half)
        return best

    def _hinge_xy(self) -> np.ndarray:
        model, data = self.task._env.current_model, self.task._env.current_data
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door")][:2].copy()

    # ---- phases -----------------------------------------------------------------------------
    def hook_lever(self) -> bool:
        """Stand at the lever, hand over the bar, close the fingers behind it."""
        task = self.task
        lo, hi = self._lever_bounds()
        self._bar = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])  # bar centre, top face
        # Stand so the left fingertip is in line with the bar.
        base = np.array(task._base_xy_yaw()[:2])
        off = self._site()[0][:2] - base
        stand = np.array([base[0], self._bar[1] - off[1]])
        x, y, yaw = task._base_xy_yaw()
        if abs(stand[1] - y) > 0.06 or abs(np.degrees(yaw) % 360 - 180) > 6:
            task.nav_to_xy(stand[0], stand[1], yaw_deg=180)
        self._hold = None
        self._pose = OPEN_POSE
        over = self._bar + [-self.FINGER_REACH, 0.0, self.OVER_BAR + 0.05]
        self._move(over, self._push_rot(), self.APPROACH_TICKS, comp=True)
        off3, e = self._servo(over, self._push_rot)
        if self.debug:
            print(
                f"  hook pre: err={e * 100:.1f}cm site={np.round(self._site()[0], 3)} "
                f"bar={np.round(self._bar, 3)}"
            )
        low = over - [0.0, 0.0, 0.05]
        self._move(low + off3, self._push_rot(), self.PUSH_TICKS, precision=True, comp=False)
        off3, e = self._servo(low, self._push_rot, offset=off3)
        self._grasp_offset = off3
        if self.debug:
            print(f"  hook low: err={e * 100:.1f}cm site={np.round(self._site()[0], 3)}")
        self._set_pose(HOOK_POSE, 30)
        data = task._env.current_data
        model = task._env.current_model
        q = [
            data.qpos[
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_0/left_hand_{n}_joint")
                ]
            ]
            for n in ("index_0", "index_1", "middle_0", "middle_1")
        ]
        print(f"door: fingers closed to {np.round(q, 2)} (commanded {HOOK_POSE[3:]})")
        # Fingers stopped short of their command => something (the bar) is in the curl.
        caught = bool(abs(q[0]) < abs(HOOK_POSE[3]) - 0.15 or abs(q[2]) < abs(HOOK_POSE[5]) - 0.15)
        if not caught:
            print("door: fingers closed fully (bar probably not in the curl); continuing anyway")
        return caught

    def turn_lever(self) -> bool:
        task = self.task
        site = self._site()[0]
        target = site - [0.0, 0.0, self.LEVER_DROP]
        self._move(
            target + self._grasp_offset, self._push_rot(), self.PUSH_TICKS * 2, precision=True,
            comp=False, until=lambda: task.lever_angle() >= self.LEVER_PRESS_ANGLE,
        )  # fmt: skip
        print(f"door: lever at {task.lever_angle():.2f} rad (needs {LEVER_TURNED})")
        return task.lever_angle() >= LEVER_TURNED

    def pull_door(self) -> bool:
        """Carry the hooked grip along the door's arc by walking the base, arm frozen.

        Dragging the hand with the arm alone runs out of reach after ~15 deg (the arm folds
        toward the body). Instead the arm/waist/fingers hold the hooked configuration and the
        base walks and turns so the hand tracks a point moving along the arc; the base also yaws
        with the door so the hook keeps its orientation to the bar.
        """
        task = self.task
        robot = task._env.current_robot
        view = robot.robot_view
        hinge = self._hinge_xy()
        site = self._site()[0]
        rel = site[:2] - hinge
        radius = float(np.linalg.norm(rel))
        phi0 = np.arctan2(rel[0], rel[1])  # angle from +y toward +x
        q0 = task.door_angle()
        arm_hold = np.asarray(view.get_move_group("left_arm").joint_pos, dtype=np.float32)
        model, data = task._env.current_model, task._env.current_data
        waist_qa = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_0/waist_{a}_joint")]
            for a in ("yaw", "roll", "pitch")
        ]
        waist_hold = data.qpos[waist_qa].astype(np.float32)
        height = float(robot._low_level._height_cmd)
        q_cmd = q0
        for i in range(self.PULL_MAX_TICKS):
            # Only run the command ahead of the door by a little, so a slipping hand stalls it.
            q_cmd = min(q_cmd + self.PULL_RATE, task.door_angle() + 0.12, self.PULL_TO)
            phi = phi0 + (q_cmd - q0)
            want = hinge + radius * np.array([np.sin(phi), np.cos(phi)])
            err = want - self._site()[0][:2]
            bx, by, byaw = task._base_xy_yaw()
            goal = np.array([bx, by]) + np.clip(self.WALK_GAIN * err, -0.5, 0.5)
            yaw_goal = np.pi - (q_cmd - q0)
            cmd = robot.waypoint_to_velocity_target([goal[0], goal[1], yaw_goal])
            task._step_and_record(
                {
                    "left_arm": arm_hold,
                    "legs_waist": np.array([*cmd, height, *waist_hold], dtype=np.float32),
                    "left_hand": self._pose,
                }
            )
            if task.viewer is not None:
                task.viewer.sync()
            if self.debug and (i % 10 == 0 or i < 8):
                print(
                    f"  pull q={np.degrees(task.door_angle()):.0f} cmd={np.degrees(q_cmd):.0f} "
                    f"lever={task.lever_angle():.2f} base=({bx:.2f},{by:.2f},{np.degrees(byaw):.0f}) "
                    f"z={robot.robot_view.base.pose[2, 3]:.2f} hand_err={np.linalg.norm(err) * 100:.0f}cm"
                )
            if self.debug and task.lever_angle() < 0.25 and not getattr(self, "_dumped", False):
                self._dumped = True
                self._dump_hand_vs_handle()
            if task.door_angle() >= self.PULL_TO:
                break
        ok = task.door_angle() > 0.8
        print(f"door: pulled to {np.degrees(task.door_angle()):.0f} deg {'ok' if ok else 'FAILED'}")
        return ok

    def _dump_hand_vs_handle(self) -> None:
        """Debug: left-hand link positions in the (unturned) handle frame at the moment of a slip."""
        model, data = self.task._env.current_model, self.task._env.current_data
        hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door_handle")
        R, p = data.xmat[hb].reshape(3, 3), data.xpos[hb]
        print("  SLIP: handle frame; bar box (local) y[-0.095,-0.015] x[0.068,0.088] z+-0.012")
        for n in ("left_wrist_yaw_link", "left_hand_index_0_link", "left_hand_index_1_link",
                  "left_hand_middle_0_link", "left_hand_middle_1_link"):  # fmt: skip
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_0/{n}")
            print("   ", n, np.round(R.T @ (data.xpos[b] - p), 3))

    def _rest_arms(self, ticks: int = 30) -> None:
        """Release the lever and return both arms, fingers and standing height to rest."""
        task = self.task
        robot = task._env.current_robot
        view = robot.robot_view
        a0 = {
            s: np.asarray(view.get_move_group(f"{s}_arm").joint_pos, np.float32)
            for s in ("left", "right")
        }
        h0 = float(robot._low_level._height_cmd)
        p0 = self._pose.copy()
        for i in range(1, ticks + 1):
            f = i / ticks
            action = {
                "legs_waist": np.array([0, 0, 0, h0 + f * (0.793 - h0), 0, 0, 0], dtype=np.float32),
                "left_hand": p0 + f * (OPEN_POSE - p0),
            }
            for s in ("left", "right"):
                action[f"{s}_arm"] = a0[s] + f * (task.rest_arms[s] - a0[s])
            task._step_and_record(action)
            if task.viewer is not None:
                task.viewer.sync()
        self._pose = OPEN_POSE
        self._height_cmd = None

    def walk_through(self) -> bool:
        """Arms to rest, then around the leaf's free edge and straight through the doorway.

        Direct waypoint driving, not the grid planner: that treats the door leaf as absent, and
        the CAD's rough-terrain section (which the rules remove for manipulation trials) starts
        right behind the door and reads as blocked. The path goes from where the pull left the
        robot, round the +y side of the leaf's free edge, then through the doorway.
        """
        task = self.task
        self._rest_arms()
        x, y, _ = task._base_xy_yaw()
        tip = self._hinge_xy() + 0.91 * np.array([np.sin(task.door_angle()), np.cos(task.door_angle())])
        via = np.array([tip[0] + 0.15, tip[1] + 0.40])  # clear of the free edge, on its +y side
        path = np.array([[x, y], via, [9.0, self.THROUGH_Y], [self.THROUGH_X, self.THROUGH_Y]])
        ok = task._follow_xy_path(path, 180, label="door_walk")
        print(f"door: walk-through {'reached' if ok else 'stopped'} at x={task._base_xy_yaw()[0]:.2f}")
        return ok

    def _release_and_lift(self) -> None:
        """Open the fingers and raise the hand clear of the lever, ready to re-approach."""
        self._set_pose(OPEN_POSE, 20)
        pos, rot = self._site()
        self._move(pos + [0.05, 0.0, 0.10], self._push_rot(), 20, comp=True)

    def traverse(self, attempts: int = 3) -> bool:
        """Hook the lever, turn it, pull the door open, walk through, wait for it to close.

        The finger-stall check in `hook_lever` is advisory (the palm turns the lever even when the
        fingers miss it); what decides success is whether the lever turned and the door opened,
        so a miss is retried from a fresh approach.
        """
        task = self.task
        for attempt in range(attempts):
            self.hook_lever()
            if self.turn_lever():
                break
            print(f"door: lever not turned (attempt {attempt + 1}/{attempts}), re-hooking")
            self._release_and_lift()
        else:
            print("door: giving up, the lever never turned")
            return False
        if not self.pull_door():
            print("door: giving up, the door did not open far enough")
            return False
        self.walk_through()
        for _ in range(60):  # the weak, damped hinge spring takes a couple of minutes to shut
            task.noop(ticks=50)
            if task.door_closed_after:
                break
        task.status()
        return task.traversed and task.door_closed_after
