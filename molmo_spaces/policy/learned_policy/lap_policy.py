import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import PromptSampler, resize_with_pad

log = logging.getLogger(__name__)


def euler_to_rot6d(euler_angles: np.ndarray) -> np.ndarray:
    rot_matrix = R.from_euler("xyz", euler_angles, degrees=False).as_matrix()
    return np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]])


def add_euler(curr: np.ndarray, delta: np.ndarray) -> np.ndarray:
    r_curr = R.from_euler("xyz", curr)
    r_delta = R.from_euler("xyz", delta)
    return (r_curr * r_delta).as_euler("xyz")


class LAP_Policy(InferencePolicy):
    """LAP (Language-Action Pre-training) policy client.
    """

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        super().__init__(exp_config, exp_config.task_type)
        self.remote_config = exp_config.policy_config.remote_config
        self.prompt_sampler = PromptSampler(
            task_type=exp_config.task_type,
            prompt_templates=exp_config.policy_config.prompt_templates,
            prompt_object_word_num=exp_config.policy_config.prompt_object_word_num,
        )
        self.grasping_type = exp_config.policy_config.grasping_type
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.chunk_size = exp_config.policy_config.chunk_size
        self.model = None

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.prompt_sampler.next()
        self.starting_time = None

    def prepare_model(self):
        try:
            from openpi_client import websocket_client_policy
        except ImportError as e:
            log.warning(
                "openpi_client package is required for remote model inference. "
                "Install it with: pip install openpi-client"
            )
            raise e

        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8000)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = websocket_client_policy.WebsocketClientPolicy(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to LAP server at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to LAP server after {max_retries} attempts")
                    raise

    def _get_tcp_pose_and_euler(self):
        """Get current TCP position and euler angles relative to robot base."""
        robot_view = self.task.env.current_robot.robot_view
        arm_mg = robot_view.get_move_group("arm")
        tcp_pose = arm_mg.leaf_frame_to_robot  # 4x4 relative to base
        position = tcp_pose[:3, 3].copy()
        euler = R.from_matrix(tcp_pose[:3, :3]).as_euler("xyz")
        return position, euler

    def obs_to_model_input(self, obs):
        prompt = self.prompt_sampler.get_prompt(self.task).lower()

        exo_key = "droid_shoulder_light_randomization" if "droid_shoulder_light_randomization" in obs else "exo_camera_1"
        wrist_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"

        position, euler = self._get_tcp_pose_and_euler()
        rot6d = euler_to_rot6d(euler)
        cartesian_position = np.concatenate([position, rot6d])

        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        grip = 1.0 if grip > 0.5 else 0.0
        gripper_position = np.array([grip])

        return {
            "observation": {
                "base_0_rgb": resize_with_pad(obs[exo_key], 224, 224),
                "left_wrist_0_rgb": resize_with_pad(obs[wrist_key], 224, 224),
                "cartesian_position": cartesian_position,
                "gripper_position": gripper_position,
                "joint_position": np.array(obs["qpos"]["arm"][:7]),
                "state": np.concatenate([cartesian_position, gripper_position]),
                "euler": euler,
            },
            "prompt": prompt,
        }

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()

        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:

            response = self.model.infer(model_input)
            actions = np.array(response["actions"], dtype=float)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            query_pos, query_euler = self._get_tcp_pose_and_euler()

            actions[:, :3] += query_pos
            for i in range(len(actions)):
                actions[i, 3:6] = add_euler(query_euler, actions[i, 3:6])

            self.actions_buffer = actions
            self.current_buffer_index = 0

        action = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return action

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

        if self.grasping_type == "binary":
            action["gripper"] = (
                np.array([255.0]) if gripper_val > self.grasping_threshold else np.array([0.0])
            )
        else:
            action["gripper"] = gripper_val * np.array([255.0])

        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "lap"
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.prompt_sampler.get_prompt(self.task)
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
