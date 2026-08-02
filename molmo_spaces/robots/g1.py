from typing import TYPE_CHECKING

import mujoco
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.g1_view import G1RobotView

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


class G1Robot(Robot):
    """G1 humanoid robot (Phase 2: standing only, no walking or arm IK yet).

    Every actuated move group (legs, waist, both arms, the right gripper) is
    driven by a plain JointPosController -- the MJCF's own tuned PD actuator
    gains (biastype="affine") do the rest. The base has no actuators; the
    pelvis is a free-floating body whose dynamics MuJoCo integrates directly.
    """

    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)

        self._namespace = self.exp_config.robot_config.robot_namespace
        self._robot_view = G1RobotView(mj_data, self.namespace)
        self._kinematics = MlSpacesKinematics(self.exp_config.robot_config)

        self._controllers = {
            "legs": JointPosController(self.robot_view.get_move_group("legs")),
            "waist": JointPosController(self.robot_view.get_move_group("waist")),
            "left_arm": JointPosController(self.robot_view.get_move_group("left_arm")),
            "right_arm": JointPosController(self.robot_view.get_move_group("right_arm")),
            "right_gripper": JointPosController(self.robot_view.get_move_group("right_gripper")),
        }
        assert set(self._controllers.keys()).issubset(set(self._robot_view.move_group_ids())), (
            "All controller keys must be move group IDs"
        )

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        raise NotImplementedError("Parallel kinematics not implemented for G1")

    def get_arm_move_group_ids(self) -> list[str]:
        """Both arms get independent TCP-bounded action noise."""
        return ["left_arm", "right_arm"]

    def reset(self) -> None:
        """Reset the robot to its initial standing state."""
        init_qpos_dict = self.exp_config.robot_config.init_qpos
        self.set_joint_pos(init_qpos_dict)
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        return "pelvis"

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "BaseRobotConfig"):
        super().apply_control_overrides(spec, robot_config)
        # Match the solver settings the source G1 stack validated for stable
        # bipedal contact: implicit integration, pyramidal friction cones,
        # extra no-slip iterations, and unit impedance ratio. The robot's own
        # MJCF <option> block can't take effect on its own -- MjSpec.attach_body
        # only merges body subtrees, not <option> blocks -- so whatever the
        # target scene's <option> says would otherwise silently win.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.noslip_iterations = 5
        spec.option.impratio = 1.0
