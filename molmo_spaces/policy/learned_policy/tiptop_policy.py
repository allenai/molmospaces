import logging
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import websockets.exceptions
import websockets.sync.client
import msgpack_numpy

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class TiptopWebsocketClient:
    """Websocket client that adds endpoint field for a TiPToP server."""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
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
            logging.warning(
                f"WebSocket connection closed. Reconnecting to {self._connected_uri}..."
            )
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


class TiptopPolicy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.cam_obs_qpos = exp_config.policy_config.cam_obs_qpos
        self.cam_obs_n_steps = exp_config.policy_config.cam_obs_n_steps
        self.model = None  # don't init model till inference to allow multiprocessing
        self.reset()

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None
        self._in_pre_obs_phase = self.cam_obs_qpos is not None
        self._pre_obs_buffer = None
        self._pre_obs_index = 0
        self._plan_exhausted = False

    def prepare_model(self):
        if not self.remote_config:
            raise ValueError("TiPToP policy only supports remote model inference")
        host = self.remote_config["host"]
        port = self.remote_config["port"]
        max_retries = self.remote_config["max_retries"]

        for attempt in range(max_retries):
            try:
                self.model = TiptopWebsocketClient(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to TiPToP model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def render(self, obs):
        # TiPToP uses just the wrist camera
        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )
        views = obs[wrist_camera_key]
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        """
        Args:
            obs: a single-element list whose element is
                a dict containing:
                - wrist_camera / wrist_camera_zed_mini: uint8 (H, W, 3) RGB image
                - {camera_name}_depth: float32 (H, W) depth in meters
                - sensor_param_{camera_name}: dict with "intrinsic_cv" (3, 3) and
                  "cam2world_gl" (4, 4)
                - qpos["arm"]: 7 joint positions
        """
        obs = obs[0]

        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )
        camera_params = obs[f"sensor_param_{wrist_camera_key}"]

        # TiPToP's planning frame is the robot base link frame. The world coordinate frame differs from the
        # robot base link frame, so we need to compute the transformation matrix from the camera to
        # the robot base link frame.
        world_from_base = self.task.env.current_robot.robot_view.get_move_group(
            "arm"
        ).root_frame_to_world
        base_from_world = np.linalg.inv(world_from_base).astype(np.float32)

        # TiPToP only uses the wrist camera. The TiPToP API contract (world_from_cam) assumes we are passing
        # in the transformation matrix from the camera to the robot base link frame.
        base_from_cam = base_from_world @ np.asarray(
            camera_params["cam2world_gl"], dtype=np.float32
        )

        model_input = {
            "rgb": np.array(obs[wrist_camera_key], dtype=np.uint8),
            "depth": np.array(obs[f"{wrist_camera_key}_depth"], dtype=np.float32),
            "intrinsics": np.array(camera_params["intrinsic_cv"], dtype=np.float32),
            "world_from_cam": base_from_cam,
            "task": self.task.get_task_description(),
            "q_init": np.array(obs["qpos"]["arm"][:7], dtype=np.float32),
        }
        return model_input

    def _unroll_plan(self, plan: dict) -> np.ndarray:
        """Unroll a serialized tiptop plan into an (N, 8) [arm(7) | gripper(1)] array."""
        action_segments = []
        current_gripper = 0.0  # start open
        last_arm_position = np.asarray(plan["q_init"], dtype=np.float32)[:7]

        for step in plan["steps"]:
            step_type = step["type"]
            if step_type == "trajectory":
                arm_positions = np.asarray(step["positions"], dtype=np.float32)[:, :7]  # (M, 7)
                gripper_col = np.full((len(arm_positions), 1), current_gripper, dtype=np.float32)
                action_segment = np.hstack([arm_positions, gripper_col])  # (M, 8)
                action_segments.append(action_segment)
                last_arm_position = arm_positions[-1]
            elif step_type == "gripper":
                current_gripper = 1.0 if step["action"] == "close" else 0.0
                action = np.concatenate([last_arm_position, [current_gripper]]).astype(np.float32)  # (8,)
                action_segments.append(action[None])  # (1, 8)

        return np.concatenate(action_segments, axis=0)

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()

        # Pre-observation phase: move arm to cam_obs_qpos before sending camera data to TiPToP.
        # We interpolate from the current joint positions (q_init) to cam_obs_qpos over
        # cam_obs_n_steps steps. Only after the arm reaches the observation pose do we call
        # infer() — ensuring Tiptop sees the scene from the elevated camera position.
        if self._in_pre_obs_phase:
            if self._pre_obs_buffer is None:
                q_current = np.array(model_input["q_init"], dtype=np.float32)
                q_target = np.array(self.cam_obs_qpos, dtype=np.float32)
                n = self.cam_obs_n_steps
                positions = np.linspace(q_current, q_target, n + 1, dtype=np.float32)[1:]
                gripper_col = np.zeros((n, 1), dtype=np.float32)
                self._pre_obs_buffer = np.hstack([positions, gripper_col])
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
                log.warning(
                    "Tiptop planning failed: %s. Returning no-op (hold current pose) and marking done.",
                    result.get("error", "unknown error"),
                )
                noop = np.concatenate([model_input["q_init"][:7], [0.0]]).astype(np.float32)
                self.actions_buffer = noop[None]  # (1, 8)
                self.current_buffer_index = 0
                self._plan_exhausted = True
                return noop
            self.actions_buffer = self._unroll_plan(result["plan"])
            self.current_buffer_index = 0
            log.info(
                "Tiptop plan unrolled into %d policy actions",
                len(self.actions_buffer),
            )
        if self.current_buffer_index >= len(self.actions_buffer):
            log.warning("Tiptop plan exhausted; holding last waypoint and sending done action")
            self._plan_exhausted = True
            return self.actions_buffer[-1]
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        """
        Args:
            model_output: an ndarray of shape (8,): arm joints (7,) + gripper scalar (1,)
        """
        gripper_pos = np.array([model_output[7] * 255.0])
        arm_output = model_output[:7]
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        if self._plan_exhausted:
            action["done"] = True
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "tiptop"
        info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
