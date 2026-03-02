import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)

MOVE_DOWN_STEP = 0.001  # 1mm per step


class StereoVLA_Policy(InferencePolicy):
    """StereoVLA policy stub (Franka Panda).

    Currently just moves the EEF straight down.
    """

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        super().__init__(exp_config, exp_config.task_type)
        self.starting_time = None

    def reset(self):
        self.starting_time = None

    def prepare_model(self):
        pass

    def _get_tcp_pose_and_euler(self):
        robot_view = self.task.env.current_robot.robot_view
        arm_mg = robot_view.get_move_group("arm")
        tcp_pose = arm_mg.leaf_frame_to_robot
        position = tcp_pose[:3, 3].copy()
        euler = R.from_matrix(tcp_pose[:3, :3]).as_euler("xyz")
        return position, euler

    def obs_to_model_input(self, obs):
        return obs

    def inference_model(self, model_input):
        if self.starting_time is None:
            self.starting_time = time.time()

        position, euler = self._get_tcp_pose_and_euler()
        # Move down (negative z)
        target_pos = position.copy()
        target_pos[2] -= MOVE_DOWN_STEP
        return np.concatenate([target_pos, euler, [0.0]])

    def model_output_to_action(self, model_output):
        target_pos = model_output[:3]
        target_euler = model_output[3:6]
        gripper_val = model_output[6]

        target_pose = np.eye(4)
        target_pose[:3, :3] = R.from_euler("xyz", target_euler).as_matrix()
        target_pose[:3, 3] = target_pos

        kinematics = self.task.env.current_robot.kinematics
        robot_view = self.task.env.current_robot.robot_view

        gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
        arm_mgs = [mg for mg in robot_view.move_group_ids() if mg not in gripper_mgs]

        jp = kinematics.ik(
            "arm",
            target_pose,
            arm_mgs,
            robot_view.get_qpos_dict(),
            robot_view.base.pose,
            rel_to_base=True,
        )

        action = robot_view.get_ctrl_dict()
        if jp is not None:
            action.update({mg_id: jp[mg_id] for mg_id in arm_mgs})
        else:
            log.warning("IK failed, holding current position")

        action["gripper"] = np.array([255.0]) if gripper_val > 0.5 else np.array([0.0])
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "stereo_vla"
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
