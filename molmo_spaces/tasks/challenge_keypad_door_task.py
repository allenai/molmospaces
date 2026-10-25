"""IEEE Humanoids 2026 Loco-Manipulation Challenge, manipulation tasks 1 and 2.

1. Keypad: a 5-digit code is shown on the screen above the keypad; press the matching
   buttons (3x3 grid, 1 top-left .. 9 bottom-right) in order. A button "lights up"
   (turns green) once pushed in far enough.
2. Door traversal: turn the lever handle, push the door open, walk through the doorway,
   and let the door close behind you. The hinge has a return spring, so the door closes
   itself once released.

The scene comes from scripts/data/format_conversion/convert_testbed_to_mjcf.py, which cuts
the door ("door", "door_handle") and the nine buttons ("key1".."key9") out of the CAD as
articulated bodies. The scene uses its default `dexterity` layout (mobility terrain removed, flat
deck behind the door). The latch is simulated here: while the door is shut and the lever is
not turned, the hinge's joint range is clamped to [0, 0].

Use via the interactive shell (config InteractiveShellG1ChallengeKeypadDoor): `status()`
plus `press(digit)` / `solve_keypad()` and the usual `nav_to_xy` / `rotate` / ... skills. The
scripted solvers: `press` / `solve_keypad` and `solve_door` / `solve_all`.
"""

import logging

import mujoco
import numpy as np

from molmo_spaces.tasks.interactive_shell_task import InteractiveShellTask

log = logging.getLogger(__name__)

KEY_PRESSED_DEPTH = 0.005  # m the button must travel to register / light up
KEY_RELEASED_DEPTH = 0.002  # m it must come back to before it can register again
LEVER_TURNED = 0.35  # rad the lever must be turned to release the latch
DOOR_AJAR = 0.05  # rad past which the latch no longer matters
DOOR_CLOSED = 0.05  # rad below which the door counts as shut
TRAVERSED_X = 8.2  # base x (m) past which the robot is through the doorway (wall at 8.58)
DOOR_OPEN_MAX = 1.9  # rad, hinge range upper limit (matches the MJCF)

LIT_RGBA = np.array([0.1, 0.9, 0.2, 1.0])
UNLIT_RGBA = np.array([1.0, 1.0, 1.0, 1.0])


class ChallengeKeypadDoorTask(InteractiveShellTask):
    """Keypad then door, scored from sim state every tick the shell steps the robot."""

    def __init__(self, env, exp_config, code: list[int]) -> None:
        super().__init__(env, exp_config)
        model = env.current_model
        self.code = list(code)
        self._jid = {
            n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ["door_hinge", "door_handle_turn", *[f"key{d}_slide" for d in range(1, 10)]]
        }
        self._plate_geoms = {
            d: [
                g
                for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                    f"key{d}_plate"
                )
            ]
            for d in range(1, 10)
        }
        self._door_range = model.jnt_range[self._jid["door_hinge"]].copy()
        self._key_down = {d: False for d in range(1, 10)}
        self.entered: list[int] = []  # digits registered since the last wrong press
        self.press_log: list[int] = []  # every registered press, for scoring/debugging
        self.keypad_done = False
        self.wrong_presses = 0
        self.lever_max = 0.0
        self.door_max = 0.0
        self.traversed = False
        self.door_closed_after = False
        self.fell = False
        # The arm pose the episode starts in, which the door solver returns to.
        view = env.current_robot.robot_view
        self.rest_arms = {
            side: np.asarray(view.get_move_group(f"{side}_arm").joint_pos, dtype=np.float32).copy()
            for side in ("left", "right")
        }
        self.rest_arm = self.rest_arms["right"]

    # ---- state readers -------------------------------------------------------------------
    def _q(self, joint: str) -> float:
        return float(self._env.current_data.qpos[self._env.current_model.jnt_qposadr[self._jid[joint]]])

    def door_angle(self) -> float:
        return self._q("door_hinge")

    def lever_angle(self) -> float:
        return abs(self._q("door_handle_turn"))

    # ---- scoring -------------------------------------------------------------------------
    def _update_progress(self) -> None:
        model = self._env.current_model

        # Keypad: edge-detect presses, advance/reset the sequence, light the plate.
        for d in range(1, 10):
            depth = -self._q(f"key{d}_slide")
            # Hysteresis: a pushed button chatters against the fingertip; it must come
            # back up before the next press counts.
            if depth >= KEY_PRESSED_DEPTH and not self._key_down[d]:
                self._key_down[d] = True
                self._register_press(d)
            elif depth <= KEY_RELEASED_DEPTH:
                self._key_down[d] = False
            for g in self._plate_geoms[d]:
                model.geom_rgba[g] = LIT_RGBA if self._key_down[d] else UNLIT_RGBA

        # Latch: shut door + unturned lever => hinge locked; otherwise free.
        lever = self.lever_angle()
        angle = self.door_angle()
        self.lever_max = max(self.lever_max, lever)
        self.door_max = max(self.door_max, angle)
        jid = self._jid["door_hinge"]
        locked = angle < DOOR_AJAR and lever < LEVER_TURNED
        model.jnt_range[jid] = [0.0, 0.0] if locked else self._door_range

        # Traversal / closing.
        x = self._env.current_robot.robot_view.base.pose[0, 3]
        z = self._env.current_robot.robot_view.base.pose[2, 3]
        self.fell = self.fell or z < 0.5
        if self.door_max > 0.3 and x < TRAVERSED_X:
            self.traversed = True
        if self.traversed and angle < DOOR_CLOSED:
            self.door_closed_after = True

    def _register_press(self, digit: int) -> None:
        self.press_log.append(digit)
        if self.keypad_done:
            return
        if digit == self.code[len(self.entered)]:
            self.entered.append(digit)
            print(f"keypad: {digit} OK ({len(self.entered)}/{len(self.code)})")
            if len(self.entered) == len(self.code):
                self.keypad_done = True
                print("keypad: code accepted")
        else:
            self.wrong_presses += 1
            self.entered = []
            print(f"keypad: {digit} wrong, sequence reset (expected {self.code[0]} first)")

    def _step_and_record(self, action) -> None:
        super()._step_and_record(action)
        self._update_progress()

    def judge_success(self) -> bool:
        return bool(
            self.keypad_done and self.traversed and self.door_closed_after and not self.fell
        )

    def get_reward(self) -> np.ndarray:
        steps = [
            len(self.entered) / len(self.code) if not self.keypad_done else 1.0,
            float(self.lever_max >= LEVER_TURNED),
            float(self.door_max > 0.3),
            float(self.traversed),
            float(self.door_closed_after),
        ]
        return np.array([sum(steps) / len(steps)])

    def get_task_description(self) -> str:
        return (
            f"Enter code {''.join(map(str, self.code))} on the keypad, then open the door, "
            "walk through and let it close."
        )

    def status(self) -> dict:
        s = {
            "code": "".join(map(str, self.code)),
            "entered": "".join(map(str, self.entered)),
            "keypad_done": self.keypad_done,
            "wrong_presses": self.wrong_presses,
            "lever_turned": self.lever_max >= LEVER_TURNED,
            "door_angle_deg": round(float(np.degrees(self.door_angle())), 1),
            "door_opened": self.door_max > 0.3,
            "traversed": self.traversed,
            "door_closed_after": self.door_closed_after,
            "success": self.judge_success(),
        }
        print(", ".join(f"{k}={v}" for k, v in s.items()))
        return s

    # ---- shell hooks ---------------------------------------------------------------------
    def _shell_extras(self) -> dict:
        return {
            "status": self.status,
            "press": self.press,
            "solve_keypad": self.solve_keypad,
            "solve_door": self.solve_door,
            "solve_all": self.solve_all,
        }

    def _press_solver(self):
        from molmo_spaces.policy.solvers.object_manipulation.keypad_press_solver import (
            Dex3KeypadPressSolver,
            KeypadPressSolver,
        )

        if getattr(self, "_solver", None) is None:
            dex3 = "right_hand" in self._env.current_robot.robot_view.move_group_ids()
            self._solver = (Dex3KeypadPressSolver if dex3 else KeypadPressSolver)(self)
        return self._solver

    def press(self, digit: int) -> bool:
        """Press keypad button `digit` (1-9) with the right hand, gripper closed."""
        return self._press_solver().press(digit)

    def solve_keypad(self) -> bool:
        """Press this episode's code, in order (see KeypadPressSolver)."""
        return self._press_solver().solve(self.code)

    def solve_door(self) -> bool:
        """Turn the lever, push through the door, let it close (see DoorTraversalSolver)."""
        from molmo_spaces.policy.solvers.object_manipulation.door_traversal_solver import (
            DoorTraversalSolver,
        )

        if "left_hand" not in self._env.current_robot.robot_view.move_group_ids():
            raise NotImplementedError("solve_door needs the dual-Dex3 G1 (config *G1Dex3*)")
        if getattr(self, "_door_solver", None) is None:
            self._door_solver = DoorTraversalSolver(self)
        return self._door_solver.traverse()

    def solve_all(self) -> bool:
        """Keypad, then door."""
        self.solve_keypad()
        self.solve_door()
        return self.status()["success"]

    def _shell_banner_extra(self) -> list[str]:
        return [
            " challenge (keypad + door):",
            f"  task: {self.get_task_description()}",
            "  status()                               - keypad / lever / door / traversal progress",
            "  press(digit)                           - right-hand press of keypad button 1-9",
            "  solve_keypad()                         - press the whole code, in order",
            "  solve_door()                           - lever, push through the door, let it close",
            "  solve_all()                            - keypad, then door",
        ]
