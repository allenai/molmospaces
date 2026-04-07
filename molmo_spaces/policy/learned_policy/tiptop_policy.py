import logging
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import websockets.exceptions
import websockets.sync.client
import msgpack_numpy

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import PromptSampler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class TiptopWebsocketClient:
    """Websocket client that adds endpoint field for a Tiptop server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        self._connected_uri = self._uri

    def _connect_once(self, uri: str) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn, metadata = self._connect_once(self._uri)
            return conn, metadata
        except Exception:
            logging.info("Connection with ws:// failed. Trying wss:// ...")

        wss_uri = "wss://" + self._uri.split("//")[1]
        conn, metadata = self._connect_once(wss_uri)
        self._uri = wss_uri
        return conn, metadata

    def _reconnect(self) -> None:
        retry_delay = 2
        while True:
            logging.warning(f"WebSocket connection closed. Reconnecting to {self._connected_uri}...")
            try:
                self._ws, self._server_metadata = self._connect_once(self._connected_uri)
                logging.info("Reconnected to server.")
                return
            except Exception as e:
                logging.warning(f"Reconnect failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    def infer(self, obs: Dict) -> Dict:
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("ConnectionClosedError during infer. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: Dict = None) -> None:
        if reset_info is None:
            reset_info = {}
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("ConnectionClosedError during reset. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        return response

    def get_server_metadata(self) -> Dict:
        return self._server_metadata


class Tiptop_Policy(InferencePolicy):
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
        self.chunk_size = exp_config.policy_config.chunk_size
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.cam_obs_qpos = exp_config.policy_config.cam_obs_qpos
        self.cam_obs_n_steps = exp_config.policy_config.cam_obs_n_steps
        self.repeat_waypoints_by_dt = getattr(exp_config.policy_config, "repeat_waypoints_by_dt", True)
        self.trajectory_settle_steps = getattr(exp_config.policy_config, "trajectory_settle_steps", 8)
        self.policy_dt_ms = float(exp_config.policy_dt_ms)
        self.model = None  # don't init model till inference to allow multiprocessing

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.prompt_sampler.next()
        self.starting_time = None
        self._in_pre_obs_phase = self.cam_obs_qpos is not None
        self._pre_obs_buffer = None
        self._pre_obs_index = 0
        self._plan_exhausted = False

    def prepare_model(self):
        self.model_name = "tiptop"
        if self.remote_config is not None:
            self._prepare_remote_model()
        else:
            self._prepare_local_model(self.checkpoint_path)

    def _prepare_local_model(self, checkpoint_path: str):
        raise Exception("Tiptop policy only supports remote model inference for now")

    def _prepare_remote_model(self):
        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8765)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = TiptopWebsocketClient(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to Tiptop model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def render(self, obs):
        # Tiptop uses just the wrist camera for now
        wrist_camera_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        views = obs[wrist_camera_key]
        # shoulder_camera_key = "droid_shoulder_light_randomization"
        # views = obs[shoulder_camera_key]
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    # Input obs is a dict of size 30 containing:
    #   - wrist_camera / wrist_camera_zed_mini: uint8 (H, W, 3) RGB image
    #   - {camera_name}_depth: float32 (H, W) depth in meters
    #   - sensor_param_{camera_name}: dict with "intrinsic_cv" (3,3) and "cam2world_gl" (4,4)
    #   - qpos["arm"]: 7 joint positions
    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            obs = obs[0]
        prompt = self.prompt_sampler.get_prompt(self.task).lower()

        wrist_camera_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        camera_params = obs[f"sensor_param_{wrist_camera_key}"]
        # shoulder_camera_key = "droid_shoulder_light_randomization"
        # camera_params = obs[f"sensor_param_{shoulder_camera_key}"]

        # Collect all available cameras from sensor_param_* keys
        cameras = {}
        for key in obs:
            if not key.startswith("sensor_param_"):
                continue
            cam_name = key[len("sensor_param_"):]
            if cam_name not in obs:
                continue
            cam_params = obs[key]
            cam_data = {
                "rgb": np.array(obs[cam_name], dtype=np.uint8),
                "intrinsics": np.array(cam_params["intrinsic_cv"], dtype=np.float32),
                "world_from_cam": np.array(cam_params["cam2world_gl"], dtype=np.float32),
            }
            depth_key = f"{cam_name}_depth"
            if depth_key in obs:
                cam_data["depth"] = np.array(obs[depth_key], dtype=np.float32)
            cameras[cam_name] = cam_data

        model_input = {
            "rgb": np.array(obs[wrist_camera_key], dtype=np.uint8),
            "depth": np.array(obs[f"{wrist_camera_key}_depth"], dtype=np.float32),
            "intrinsics": np.array(camera_params["intrinsic_cv"], dtype=np.float32),
            "world_from_cam": np.array(camera_params["cam2world_gl"], dtype=np.float32),
            "cameras": cameras,
            "task": prompt,
            "q_init": np.array(obs["qpos"]["arm"][:7], dtype=np.float32),
            "fr3_link0_pose": np.array(obs["fr3_link0_pose"], dtype=np.float32),
        }
        return model_input

    def _unroll_plan(self, plan: list) -> list:
        """Unroll a serialized tiptop plan into a flat list of (8,) [arm(7) | gripper(1)] arrays.

        Gripper is encoded as 0.0 (open) or 1.0 (closed), matching the scale expected by
        model_output_to_action (continuous: *255, binary: threshold comparison).
        """
        actions = []
        current_gripper = 0.0  # start open
        last_arm_pos = None

        for step in plan["steps"]:
            step_type = step["type"] if isinstance(step, dict) else step.get(b"type", b"").decode()
            if step_type == "metadata":
                q_init = step.get("q_init") if step.get("q_init") is not None else step.get(b"q_init")
                last_arm_pos = np.array(q_init, dtype=np.float32)[:7]
            elif step_type == "trajectory":
                positions = step.get("positions") if step.get("positions") is not None else step.get(b"positions")
                positions = np.array(positions, dtype=np.float32)
                dt = step.get("dt") if step.get("dt") is not None else step.get(b"dt")
                repeats = 1
                if self.repeat_waypoints_by_dt and dt is not None:
                    repeats = max(1, int(round(float(dt) * 1000.0 / self.policy_dt_ms)))
                for waypoint in positions:
                    last_arm_pos = waypoint[:7]
                    action = np.concatenate([last_arm_pos, [current_gripper]]).astype(np.float32)
                    actions.extend([action.copy() for _ in range(repeats)])
            elif step_type == "gripper":
                if last_arm_pos is not None and self.trajectory_settle_steps > 0:
                    settle_action = np.concatenate([last_arm_pos, [current_gripper]]).astype(np.float32)
                    actions.extend([settle_action.copy() for _ in range(self.trajectory_settle_steps)])
                action_val = step.get("action") if step.get("action") is not None else step.get(b"action")
                if isinstance(action_val, bytes):
                    action_val = action_val.decode()
                current_gripper = 1.0 if action_val == "close" else 0.0
                if last_arm_pos is not None:
                    actions.append(np.concatenate([last_arm_pos, [current_gripper]]).astype(np.float32))

        return actions

    # model_input dictionary:
    # {
    #   "rgb": np.array (H, W, 3) uint8          # wrist camera
    #   "depth": np.array (H, W) float32 meters  # wrist camera
    #   "intrinsics": np.array (3, 3) float32    # wrist camera
    #   "world_from_cam": np.array (4, 4) float32  # wrist camera
    #   "cameras": {                             # all cameras keyed by name
    #     "<cam_name>": {
    #       "rgb": np.array (H, W, 3) uint8,
    #       "depth": np.array (H, W) float32 (if available),
    #       "intrinsics": np.array (3, 3) float32,
    #       "world_from_cam": np.array (4, 4) float32,
    #     }, ...
    #   }
    #   "task": "pick up the cup."
    #   "q_init": np.array (7,) float32
    # }
    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()

        # Pre-observation phase: move arm to cam_obs_qpos before sending camera data to Tiptop.
        # We interpolate from the current joint positions (q_init) to cam_obs_qpos over
        # cam_obs_n_steps steps. Only after the arm reaches the observation pose do we call
        # infer() — ensuring Tiptop sees the scene from the elevated camera position.
        if self._in_pre_obs_phase:
            if self._pre_obs_buffer is None:
                q_current = np.array(model_input["q_init"], dtype=np.float32)
                q_target = np.array(self.cam_obs_qpos, dtype=np.float32)
                n = self.cam_obs_n_steps
                self._pre_obs_buffer = [
                    np.concatenate([
                        q_current + (q_target - q_current) * (i + 1) / n,
                        [0.0],  # gripper open throughout
                    ])
                    for i in range(n)
                ]
                self._pre_obs_index = 0
                log.info(f"Pre-obs phase: moving to cam_obs_qpos over {n} steps")
            if self._pre_obs_index < len(self._pre_obs_buffer):
                action = self._pre_obs_buffer[self._pre_obs_index]
                self._pre_obs_index += 1
                return action
            # Arm is at observation position — fall through to call infer() with
            # camera data captured from this pose.
            self._in_pre_obs_phase = False
            log.info("Pre-obs phase complete; sending observation-position camera data to Tiptop")

        if self.actions_buffer is None:
            result = self.model.infer(model_input)
            if not result["success"]:
                raise RuntimeError(f"Tiptop planning failed: {result.get('error', 'unknown error')}")
            self.actions_buffer = self._unroll_plan(result["plan"])
            self.current_buffer_index = 0
            log.info(
                "Tiptop plan unrolled into %d policy actions (repeat_by_dt=%s, settle_steps=%d)",
                len(self.actions_buffer),
                self.repeat_waypoints_by_dt,
                self.trajectory_settle_steps,
            )
        if self.current_buffer_index >= len(self.actions_buffer):
            log.warning("Tiptop plan exhausted; holding last waypoint and sending done action")
            self._plan_exhausted = True
            return self.actions_buffer[-1]
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    # model_output is an ndarray of shape (8,): arm joints (7,) + gripper scalar (1,)
    def model_output_to_action(self, model_output):
        if self.grasping_type == "continuous":
            gripper_pos = model_output[7] * np.array([255.0])
        else:
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
        if self._plan_exhausted:
            action["done"] = True
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = (
            self.model.get_server_metadata().get("policy_name", "tiptop")
            if hasattr(self.model, "get_server_metadata")
            else "tiptop"
        )
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.prompt_sampler.get_prompt(self.task)
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
