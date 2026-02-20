import logging
import os
import time

import cv2
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import PromptSampler, resize_with_pad

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAMPLE_PROMPTS = False 

class PI_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        super().__init__(exp_config, exp_config.task_type)
        self.remote_config = exp_config.policy_config.remote_config
        if SAMPLE_PROMPTS:
            self.prompt_sampler = PromptSampler(
                task_type=exp_config.task_type,
                prompt_templates=exp_config.policy_config.prompt_templates,
                prompt_object_word_num=exp_config.policy_config.prompt_object_word_num,
            )
        else:
            self.prompt_sampler = None
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        self.chunk_size = exp_config.policy_config.chunk_size
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.model = None  # don't init model till inference to allow multiprocessing

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        if self.prompt_sampler is not None:
            self.prompt_sampler.next()
        self.starting_time = None

    def prepare_model(self):
        self.model_name = os.path.basename(self.checkpoint_path)
        if self.remote_config is not None:
            self._prepare_remote_model(self.checkpoint_path)
        else:
            self._prepare_local_model(self.checkpoint_path)
        # self.reset()

    def _prepare_local_model(self, checkpoint_path: str):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        self.config = _config.get_config(os.path.basename(checkpoint_path))
        self.model = _policy_config.create_trained_policy(self.config, checkpoint_path)

    def _prepare_remote_model(self, checkpoint_path: str):
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
        self.checkpoint_path = checkpoint_path

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = websocket_client_policy.WebsocketClientPolicy(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to remote model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        # self.render(obs)
        if self.prompt_sampler is None:
            prompt = self.task.get_task_description()
        else:
            prompt = self.prompt_sampler.get_prompt(self.task).lower()
        log.info(f"The prompt is: {self._cached_prompt}")

        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        exo_camera_key = "droid_shoulder_light_randomization" if "droid_shoulder_light_randomization" in obs else "exo_camera_1"
        wrist_camera_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        model_input = {
            "observation/exterior_image_1_left": resize_with_pad(obs[exo_camera_key], 224, 224),
            "observation/wrist_image_left": resize_with_pad(obs[wrist_camera_key], 224, 224),
            "observation/joint_position": np.array(obs["qpos"]["arm"][:7]).reshape(
                7,
            ),
            "observation/gripper_position": np.array(grip).reshape(
                1,
            ),
            "prompt": prompt,
        }
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            self.actions_buffer = self.model.infer(model_input)["actions"]
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        if self.grasping_type == "continuous":
            gripper_pos = model_output[7] * np.array([255.0])
        else:  # binary
            gripper_pos = (
                np.array([255.0]) if model_output[7] > self.grasping_threshold else np.array([0.0])
            )

        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = (
            self.model.get_server_metadata().get("policy_name", "pi")
            if hasattr(self.model, "get_server_metadata")
            else "pi"
        )
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        if self.prompt_sampler is not None:
            info["prompt"] = self.prompt_sampler.get_prompt(self.task)
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
