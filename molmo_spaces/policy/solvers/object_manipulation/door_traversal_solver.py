"""Scripted solver for the challenge door: turn the lever, pull the door open, walk through.

WORK IN PROGRESS: grasping and turning the lever works (Dex1 gripper, jaws vertical); the
pull-through is not finished or verified end to end.

The door opens toward the start side (rules PDF, "view from the starting location"), hinged on
the left (low y) with the lever on the right, so from the start it is a pull door. The hand is
one-handed, so the lever has to be *grasped*: the jaws straddle the thin grip bar, close on it,
turn it down, and then carry the door open by the bar. Reuses KeypadPressSolver's closed-loop
hand controller (`_move`).

1. Walk to a stand in front of the lever, straddle the grip bar with the open jaws and close.
2. Pull the bar down (turning the lever, releasing the latch), then drag it back along the arc
   to ~70 deg, walking the base after it.
3. Return the arm to rest, walk around the leaf's free edge into the doorway and through it,
   clearing the swing arc. The hinge spring then closes the door behind the robot.
"""

import logging

import mujoco
import numpy as np

from molmo_spaces.policy.solvers.object_manipulation.keypad_press_solver import KeypadPressSolver
from molmo_spaces.tasks.challenge_keypad_door_task import LEVER_TURNED

log = logging.getLogger(__name__)

HAND_Y_OFFSET = 0.226  # right hand sits this far to the robot's right of the base origin


class DoorTraversalSolver(KeypadPressSolver):
    keep_station = False  # base corrections with the arm out tip the robot over
    STAND_X = 9.0  # base x for the lever press (door plane is at x~8.59)
    LEVER_PRESS_ANGLE = 0.45  # rad the lever is turned to (latch needs LEVER_TURNED)
    LEVER_DROP = 0.04  # m the grip is pulled down to turn the lever
    EMPTY_WIDTH = -0.0195  # m: inter-finger distance reads -0.0245 when closed on nothing
    GRASP_DEPTH = 0.015  # m the fingertips go below the bar's centre
    PULL_TO = 1.2  # rad (~70 deg): the free edge is then clear of the right jamb
    PULL_RATE = 0.012  # rad of door angle commanded per policy tick
    THROUGH_X = 7.3  # base x that is clear of the door's swing arc
    THROUGH_Y = 1.40  # walk-through line in the doorway (centre 1.30; keep off the hinge side)

    # ---- geometry ---------------------------------------------------------------------------
    def _lever_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World AABB (min, max) of the lever grip: the handle collision geom reaching furthest
        out from the door (+x). It extends toward the hinge (-y) from the spindle."""
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
    def _rest_arm(self, ticks: int = 30) -> None:
        """Ramp the right arm and waist back to the pose the episode started in, gripper open,
        and the standing height back up."""
        task = self.task
        robot = task._env.current_robot
        a0 = np.asarray(robot.robot_view.get_move_group("right_arm").joint_pos, dtype=np.float32)
        gripper = robot.robot_view.get_gripper("right_gripper")
        gripper.set_gripper_ctrl_open(True)
        grip = np.asarray(gripper.ctrl, dtype=np.float32).copy()
        h0 = float(robot._low_level._height_cmd)
        for i in range(1, ticks + 1):
            f = i / ticks
            task._step_and_record(
                {
                    "right_arm": a0 + f * (task.rest_arm - a0),
                    "legs_waist": np.array(
                        [0, 0, 0, h0 + f * (0.793 - h0), 0, 0, 0], dtype=np.float32
                    ),
                    "right_gripper": grip,
                }
            )
            if task.viewer is not None:
                task.viewer.sync()
        self._height_cmd = None

    def _hold_hand(self, ticks: int) -> None:
        """Keep the hand where it is for `ticks` (e.g. while the gripper opens or closes)."""
        pos, rot = self._site()
        self._move(pos, rot, ticks)

    def grasp_lever(self) -> bool:
        """Walk to the lever, straddle the grip bar with the open jaws (hand rolled so they
        open vertically) and close on it. True once holding."""
        task = self.task
        lo, hi = self._lever_bounds()
        bar = (lo + hi) / 2
        self._bar = np.array([hi[0] - 0.02, lo[1] + 0.04, bar[2]])  # mid-bar, clear of the neck at y>1.74  # the bar is ~1cm thick at x~8.67
        self._bar_z = bar[2]
        stand = np.array([self.STAND_X, self._bar[1] - HAND_Y_OFFSET])
        x, y, yaw = task._base_xy_yaw()
        if np.linalg.norm(stand - [x, y]) > 0.1 or abs(np.degrees(yaw) % 360 - 180) > 8:
            task.nav_to_xy(stand[0], stand[1], yaw_deg=180)
        x, y, yaw = task._base_xy_yaw()
        err = (np.pi - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(err) > np.radians(4):
            task.rotate(np.degrees(err))
        self._hold = None
        self.gripper_open = True
        rot = self._grasp_rot()
        pre = self._bar + [0.0, 0.0, 0.12]  # above the bar, hand pointing down
        self._move(pre, rot, self.APPROACH_TICKS, comp=True)
        off, e = self._servo(pre, self._grasp_rot)  # exact height in free air, before going in
        if self.debug:
            print(f"  pre servo: err={e * 100:.1f}cm off={np.round(off * 100, 1)} site={np.round(self._site()[0], 3)} pre={np.round(pre, 3)}")
        # Site (the fingertips) goes past the bar toward the door so the bar sits between the
        # fingers, which straddle it above and below.
        grasp = self._bar - [0.0, 0.0, self.GRASP_DEPTH]  # fingertips just below the bar
        self._move(grasp + off, self._grasp_rot(), self.PUSH_TICKS, precision=True, comp=False)
        self._servo(grasp, self._grasp_rot, offset=off)
        if self.debug:
            sp, sR = self._site()
            print(f"  at grasp: site={np.round(sp, 3)} bar={np.round(self._bar, 3)} err={np.round((sp - grasp) * 100, 1)}cm "
                  f"finger_w={task._env.current_robot.robot_view.get_gripper('right_gripper').inter_finger_dist:.3f}")
        if self.debug:
            m, d = task._env.current_model, task._env.current_data
            sp, sR = self._site()
            print("  site R (rows=site axes in world):", np.round(sR.T, 2).tolist())
            for b in range(m.nbody):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if n.startswith("robot_0/right_Link"):
                    print("   ", n, "world", np.round(d.xpos[b], 3))
        self.gripper_open = False
        self._hold_hand(25)
        gripper = task._env.current_robot.robot_view.get_gripper("right_gripper")
        width = float(gripper.inter_finger_dist)
        held = width > self.EMPTY_WIDTH  # fingers stopped on the bar instead of closing fully
        print(f"door: grip width {width * 1000:.0f} mm -> {'holding the lever' if held else 'MISSED the lever'}")
        return held

    def turn_lever(self) -> bool:
        """Pull the grasped grip down until the latch releases."""
        task = self.task
        target = self._bar - [0.0, 0.0, self.GRASP_DEPTH + self.LEVER_DROP]
        self._move(
            target, self._grasp_rot(), self.PUSH_TICKS * 2, precision=True,
            until=lambda: task.lever_angle() >= self.LEVER_PRESS_ANGLE,
        )  # fmt: skip
        print(f"door: lever at {task.lever_angle():.2f} rad (needs {LEVER_TURNED})")
        return task.lever_angle() >= LEVER_TURNED

    def pull_door(self) -> bool:
        """Drag the held grip back along the door's arc, base following, to PULL_TO."""
        task = self.task
        robot = task._env.current_robot
        hinge = self._hinge_xy()
        site = self._site()[0]
        rel = site[:2] - hinge
        radius = float(np.linalg.norm(rel))
        phi0 = np.arctan2(rel[0], rel[1])  # angle from +y toward +x
        q0 = task.door_angle()
        base_off = np.array(task._base_xy_yaw()[:2]) - site[:2]  # keep the same stance
        z = site[2]
        rot0 = self._grasp_rot()
        q_cmd = q0
        for i in range(300):
            # Only run the command ahead of the door by a little, so a slipping hand stalls it.
            q_cmd = min(q_cmd + self.PULL_RATE, task.door_angle() + 0.12, self.PULL_TO)
            phi = phi0 + (q_cmd - q0)
            hand = np.array([hinge[0] + radius * np.sin(phi), hinge[1] + radius * np.cos(phi), z])
            c, s_ = np.cos(-(q_cmd - q0)), np.sin(-(q_cmd - q0))  # hand turns with the door
            rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ rot0
            base_goal = hand[:2] + base_off
            cmd = robot.waypoint_to_velocity_target([base_goal[0], base_goal[1], np.pi])
            self._move(hand, rot, 1, precision=True, base_cmd=cmd)
            if self.debug and (i % 10 == 0 or i < 14):
                bx, by, byaw = task._base_xy_yaw()
                print(
                    f"  pull q={np.degrees(task.door_angle()):.0f} cmd={np.degrees(q_cmd):.0f} "
                    f"lever={task.lever_angle():.2f} base=({bx:.2f},{by:.2f},{np.degrees(byaw):.0f}) "
                    f"z={robot.robot_view.base.pose[2, 3]:.2f} w={robot.robot_view.get_gripper('right_gripper').inter_finger_dist * 1000:.0f}mm "
                    f"site_err={np.linalg.norm(self._site()[0] - hand) * 100:.0f}cm"
                )
            if task.door_angle() >= self.PULL_TO:
                break
        ok = task.door_angle() > 0.8
        print(f"door: pulled to {np.degrees(task.door_angle()):.0f} deg {'ok' if ok else 'FAILED'}")
        return ok

    def walk_through(self) -> bool:
        """Arm to rest, then around the free edge into the doorway and straight through it."""
        from molmo_spaces.policy.solvers.navigation.astar_planner_policy import split_by_max_dist

        task = self.task
        self._rest_arm()
        x, y, yaw = task._base_xy_yaw()
        step, lead = task._direct_drive_pacing()
        # Around the leaf's free edge (right of it), then line up on the doorway and go through.
        tip_y = self._hinge_xy()[1] + 0.91 * np.cos(task.door_angle())
        via_y = max(tip_y + 0.45, 1.75)
        path = np.array(
            [[x, y], [max(x, 9.3), via_y], [9.0, self.THROUGH_Y], [self.THROUGH_X, self.THROUGH_Y]]
        )
        xys = np.concatenate(
            [path[:1]] + [split_by_max_dist(path[i : i + 2], step)[1:] for i in range(len(path) - 1)]
        )
        d = np.diff(xys, axis=0, prepend=xys[:1])
        yaws = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        yaws[0] = yaw
        wps = np.concatenate([xys, yaws[:, None]], axis=1)
        ok = True
        for k in range(0, len(wps), 15):
            ok = task._drive_base_waypoints(
                wps[k : k + 15], "door_walk", threshold=0.12, max_ticks=80, lead=lead
            )
        print(f"door: walk-through {'reached' if ok else 'stopped'} at x={task._base_xy_yaw()[0]:.2f}")
        return ok

    def traverse(self) -> bool:
        """Lever, pull, walk through, wait for the door to close."""
        task = self.task
        if not self.grasp_lever() or not self.turn_lever():
            return False
        if not self.pull_door():
            return False
        self.walk_through()
        task.noop(ticks=100)  # the hinge spring shuts the door
        task.status()
        return task.traversed and task.door_closed_after
