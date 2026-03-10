import io
import logging
import time
from collections import deque

import numpy as np
import zmq
from PIL import Image
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
import cv2

from molmo_spaces.policy.learned_policy.utils import PromptSampler

log = logging.getLogger(__name__)


class StereoVLA_Policy(InferencePolicy):

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
        self.proprio_buffer_size = exp_config.policy_config.proprio_buffer_size
        self.image_buffer_size = exp_config.policy_config.image_buffer_size
        self.socket = None
        self.starting_time = None

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.prompt_sampler.next()
        self.starting_time = None
        self.gripper_open = True

        self.proprio_history = deque(maxlen=self.proprio_buffer_size)
        self.left_image_history = deque(maxlen=self.image_buffer_size)
        self.right_image_history = deque(maxlen=self.image_buffer_size)

        self.current_target_pos = None
        self.current_target_rot = None

    def prepare_model(self):
        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 6666)

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        log.info(f"Connected to StereoVLA server at {host}:{port}")

    @staticmethod
    def _center_crop_and_resize(img: np.ndarray, output_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h != w:
            crop = min(h, w)
            start_x = w // 2 - crop // 2
            start_y = h // 2 - crop // 2
            img = img[start_y:start_y + crop, start_x:start_x + crop]
        if img.shape[0] != output_size:
            img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _compress_image(img: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG")
        return buf.getvalue()

    def _get_tcp_pose_and_euler(self):
        robot_view = self.task.env.current_robot.robot_view
        arm_mg = robot_view.get_move_group("arm")
        tcp_pose = arm_mg.leaf_frame_to_root
        position = tcp_pose[:3, 3].copy()
        euler = R.from_matrix(tcp_pose[:3, :3]).as_euler("XYZ")
        return position, euler

    def obs_to_model_input(self, obs):
        prompt = self.prompt_sampler.get_prompt(self.task).lower()

        left_img = self._center_crop_and_resize(obs["stereo_left"], 256)
        right_img = self._center_crop_and_resize(obs["stereo_right"], 256)

        position, euler = self._get_tcp_pose_and_euler()
        gripper_state = np.array([1.0 if self.gripper_open else -1.0], dtype=np.float32)
        proprio = np.concatenate([position, euler, gripper_state]).astype(np.float32)

        left_jpg = self._compress_image(left_img)
        right_jpg = self._compress_image(right_img)

        self.left_image_history.append(left_jpg)
        self.right_image_history.append(right_jpg)
        self.proprio_history.append(proprio.tolist())

        while len(self.proprio_history) < self.proprio_buffer_size:
            self.proprio_history.appendleft(self.proprio_history[0])
        while len(self.left_image_history) < self.image_buffer_size:
            self.left_image_history.appendleft(self.left_image_history[0])
        while len(self.right_image_history) < self.image_buffer_size:
            self.right_image_history.appendleft(self.right_image_history[0])

        return {
            "text": prompt,
            "image_wrist_array": list(self.left_image_history),
            "image_array": list(self.right_image_history),
            "proprio_array": list(self.proprio_history),
            "compressed": True,
            "traj_metadata": None,
        }

    def inference_model(self, model_input):
        if self.socket is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()

        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            self.socket.send_pyobj(model_input)
            response = self.socket.recv_pyobj()

            if response.get("info") != "success":
                log.error(f"Server error: {response}")
                position, euler = self._get_tcp_pose_and_euler()
                return np.concatenate([position, euler, [0.0]])

            self.actions_buffer = response["result"]
            self.current_buffer_index = 0

            position, euler = self._get_tcp_pose_and_euler()
            self.current_target_pos = position.copy()
            self.current_target_rot = R.from_euler("XYZ", euler)

        delta_action = np.array(self.actions_buffer[self.current_buffer_index])
        self.current_buffer_index += 1

        self.current_target_pos = self.current_target_pos + delta_action[:3]

        delta_rot = R.from_euler("XYZ", delta_action[3:6])
        self.current_target_rot = delta_rot * self.current_target_rot

        target_euler = self.current_target_rot.as_euler("XYZ")

        gripper_cmd = delta_action[6]
        if gripper_cmd < -0.5:
            self.gripper_open = False
        elif gripper_cmd > 0.5:
            self.gripper_open = True

        gripper_val = 1.0 if self.gripper_open else 0.0
        return np.concatenate([self.current_target_pos, target_euler, [gripper_val]])

    def model_output_to_action(self, model_output):
        target_pos = model_output[:3]
        target_euler = model_output[3:6]
        gripper_val = model_output[6]

        target_in_root = np.eye(4)
        target_in_root[:3, :3] = R.from_euler("XYZ", target_euler).as_matrix()
        target_in_root[:3, 3] = target_pos

        robot_view = self.task.env.current_robot.robot_view
        arm_mg = robot_view.get_move_group("arm")
        root_to_robot = arm_mg.root_frame_to_robot
        target_pose = root_to_robot @ target_in_root

        kinematics = self.task.env.current_robot.kinematics

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
        info["prompt"] = self.prompt_sampler.get_prompt(self.task)
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
