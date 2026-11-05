"""Whole-body controller for the dual-Dex3 G1 (`g1_dex3`).

The legs/waist walking law, ONNX policy and right-arm passthrough are `G1WholeBodyController`'s,
unchanged. What differs from the Dex1 G1: the *left* arm is actively position-controlled like
the right (the Dex1 G1 leaves it limp), and the right Dex1 gripper is replaced by two 7-joint
Dex3-1 hands, each a plain position-actuator passthrough move group (`left_hand`, `right_hand`).
"""

from __future__ import annotations

import mujoco
import numpy as np

from molmo_spaces.controllers.g1_wbc import (
    G1WholeBodyController,
    MoveGroupController,
    RightArmController,
    WbcJointGroup,
)

HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
ARM_KP, ARM_KD = 2000.0, 60.0  # the Dex1 G1's right-arm gains (see G1WholeBodyController.setup)


class HandController(MoveGroupController):
    """Dex3 hand: 7 position-actuator targets, passed straight through."""

    def compute_ctrl_inputs(self, data, step_counter):
        return np.asarray(self.target, dtype=np.float32)


class G1Dex3WholeBodyController(G1WholeBodyController):
    def setup(self, model, data, prefix="robot_0/"):
        super().setup(model, data, prefix)
        # Left arm: the base setup zeroes its gains (limp). Give it the right arm's PD.
        for aid in self._act_ids[15:22]:
            if aid >= 0:
                model.actuator_gainprm[aid, 0] = ARM_KP
                model.actuator_biasprm[aid, :3] = [0.0, -ARM_KP, -ARM_KD]
                model.actuator_forcerange[aid] = [-400, 400]
        left_arm_mg = self._left_arm_ctrl.move_group
        self._left_arm_ctrl = RightArmController(left_arm_mg)

        def hand_group(side: str) -> WbcJointGroup:
            jqpos, jdof, acts = [], [], []
            for j in HAND_JOINTS:
                name = f"{prefix}{side}_hand_{j}_joint"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                jqpos.append(model.jnt_qposadr[jid])
                jdof.append(model.jnt_dofadr[jid])
                acts.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
            return WbcJointGroup(
                f"{side}_hand",
                np.array(jqpos, dtype=np.int32),
                np.array(jdof, dtype=np.int32),
                np.array(acts),
            )

        self._left_hand_ctrl = HandController(hand_group("left"))
        self._right_hand_ctrl = HandController(hand_group("right"))
        self._controllers = [
            self._legs_waist_ctrl,
            self._left_arm_ctrl,
            self._right_arm_ctrl,
            self._left_hand_ctrl,
            self._right_hand_ctrl,
        ]
