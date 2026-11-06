"""The G1 humanoid with a Unitree Dex3-1 hand on each wrist (`g1_dex3`).

Same body, walking controller, arms and cameras as `G1Robot`; the right Dex1 gripper is replaced
by two 7-joint Dex3 hands (`left_hand` / `right_hand` move groups) and the left arm is actively
controlled. Built from g1_dex.xml by scripts/data/format_conversion/build_g1_dex3.py.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np

from molmo_spaces.robots.g1 import DEFAULT_QPOS, JOINT_NAMES, G1Robot

# legs(12) + waist(3) + arms(14); the hands are position-actuated move groups, not WBC joints.
DEX3_JOINT_NAMES = JOINT_NAMES[:29]
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


class G1Dex3Robot(G1Robot):
    JOINT_NAMES = DEX3_JOINT_NAMES
    DEFAULT_QPOS = DEFAULT_QPOS[:29]
    ACTUATOR_NAME_MAP: dict[str, str] = {}
    # waist(3) + right arm(7): the Dex1 variant's upper-body randomisation minus its gripper.
    _UPPER_RAND_IDX = np.array([12, 13, 14, 22, 23, 24, 25, 26, 27, 28], dtype=np.int64)

    @staticmethod
    def _low_level_cls() -> type:
        from molmo_spaces.controllers.g1_dex3_wbc import G1Dex3WholeBodyController

        return G1Dex3WholeBodyController

    @property
    def robot_view(self):
        if self._native_robot_view is None:
            from molmo_spaces.robots.robot_views.g1_dex3_view import G1Dex3RobotView

            self._native_robot_view = G1Dex3RobotView(self.data, self._namespace)
        return self._native_robot_view

    def create_robot_sensors(self):
        from molmo_spaces.env.sensors_g1 import dex3_robot_sensors

        return dex3_robot_sensors()

    def set_defaults(self):
        super().set_defaults()
        for side in ("left", "right"):
            for j in HAND_JOINTS:
                name = f"{self._namespace}{side}_hand_{j}_joint"
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
                aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
                self.data.ctrl[aid] = 0.0

    def sample_upper_pose(self, np_random, radius=0.15):
        idx = self._UPPER_RAND_IDX
        per_dim = np.full(len(idx), radius, dtype=np.float64)
        per_dim[:3] = radius * 0.4  # waist: closer to neutral
        sampled = self.DEFAULT_QPOS[idx] + np_random.uniform(-per_dim, per_dim)
        for i, jidx in enumerate(idx):
            jid = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, f"{self._namespace}{self.JOINT_NAMES[jidx]}"
            )
            lo, hi = self.model.jnt_range[jid]
            sampled[i] = float(np.clip(sampled[i], lo, hi))
        return sampled

    def apply_upper_pose(self, values):
        idx = self._UPPER_RAND_IDX
        self.data.qpos[self._qpos_ids[idx]] = values
        aids = self.act_ids[idx]
        valid = aids >= 0
        if valid.any():
            self.data.ctrl[aids[valid]] = values[valid]
