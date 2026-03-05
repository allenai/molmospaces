"""Policy via OpenPI server (--policy_type pi_remote). No molmo_spaces/mujoco in this process."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Optional: openpi_client for remote inference
try:
    from openpi_client import websocket_client_policy
    _HAS_OPENPI_CLIENT = True
except ImportError:
    _HAS_OPENPI_CLIENT = False
    websocket_client_policy = None


def _resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image to (height, width) with letterbox padding. Requires cv2."""
    import cv2
    h, w = img.shape[:2]
    if h == height and w == width:
        return np.asarray(img, dtype=np.uint8)
    ratio = max(w / width, h / height)
    new_w, new_h = int(w / ratio), int(h / ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((height, width, img.shape[2]) if img.ndim == 3 else (height, width), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


class PiRemotePolicy:
    """Policy that calls OpenPI server. Obs: qpos, wrist_camera, exo_camera_1; prompt = task description."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        task_description: str = "pick up the object.",
        grasping_threshold: float = 0.5,
        chunk_size: int = 8,
        connect_timeout_s: float = 10.0,
        inference_timeout_s: float = 120.0,
    ):
        if not _HAS_OPENPI_CLIENT:
            raise ImportError(
                "openpi_client is required for Pi remote policy. "
                "Install with: pip install openpi-client"
            )
        # Fail fast if server is not reachable (avoid hang in env setup / first step)
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(max(1.0, connect_timeout_s))
        try:
            sock.connect((host, port))
        except (OSError, socket.timeout) as e:
            sock.close()
            raise ConnectionError(
                f"Cannot connect to Pi server at {host}:{port}. "
                "Start the OpenPI server first in another terminal, then run this script."
            ) from e
        sock.close()
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._task_description = task_description
        self._grasping_threshold = grasping_threshold
        self._chunk_size = chunk_size
        self._inference_timeout_s = inference_timeout_s
        self._actions_buffer: np.ndarray | None = None
        self._current_buffer_index = 0
        self._starting_time: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        log.info("Pi remote policy connected to %s:%s (inference_timeout_s=%s)", host, port, inference_timeout_s)

    def reset(self) -> None:
        self._actions_buffer = None
        self._current_buffer_index = 0
        self._starting_time = None

    def _obs_to_model_input(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Build the dict the OpenPI server expects (same as PI_Policy.obs_to_model_input)."""
        qpos = obs.get("qpos", {})
        arm = np.asarray(qpos.get("arm", np.zeros(7)), dtype=np.float64).reshape(7)
        gripper = qpos.get("gripper", np.zeros(2))
        grip = float(np.clip(np.atleast_1d(gripper)[0] / 0.824033, 0, 1))
        exo_key = "exo_camera_1"
        wrist_key = "wrist_camera"
        exo_img = obs.get(exo_key, np.zeros((224, 224, 3), dtype=np.uint8))
        wrist_img = obs.get(wrist_key, np.zeros((224, 224, 3), dtype=np.uint8))
        if exo_img.shape[:2] != (224, 224):
            exo_img = _resize_with_pad(np.asarray(exo_img), 224, 224)
        if wrist_img.shape[:2] != (224, 224):
            wrist_img = _resize_with_pad(np.asarray(wrist_img), 224, 224)
        prompt = (self._task_description or "pick up the object.").lower()
        return {
            "observation/exterior_image_1_left": exo_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": arm,
            "observation/gripper_position": np.array([grip], dtype=np.float64),
            "prompt": prompt,
        }

    def _model_output_to_action(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
        """Same as PI_Policy.model_output_to_action: arm 7, gripper 1."""
        if model_output.size >= 8:
            gripper_val = float(model_output[7])
            gripper_pos = np.array([255.0 if gripper_val > self._grasping_threshold else 0.0])
        else:
            gripper_pos = np.array([0.0])
        arm = np.asarray(model_output[:7], dtype=np.float64).reshape(7)
        return {"arm": arm, "gripper": gripper_pos}

    def get_action(self, observation: list[dict]) -> dict[str, np.ndarray]:
        """observation is a list of one obs dict (policy_obs format pi)."""
        if not observation:
            return {"arm": np.zeros(7), "gripper": np.array([0.0])}
        obs = observation[0]
        model_input = self._obs_to_model_input(obs)
        if self._starting_time is None:
            self._starting_time = time.time()
        if self._actions_buffer is None or self._current_buffer_index >= self._chunk_size:
            log.debug("Requesting action from Pi server (timeout=%ss)...", self._inference_timeout_s)
            future = self._executor.submit(lambda: self._client.infer(model_input)["actions"])
            try:
                raw = future.result(timeout=self._inference_timeout_s)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"Pi server did not respond within {self._inference_timeout_s}s. "
                    "Check that the server is running and not stuck (e.g. first inference can be slow)."
                ) from None
            self._actions_buffer = np.asarray(raw) if not isinstance(raw, np.ndarray) else raw
            self._current_buffer_index = 0
        model_output = self._actions_buffer[self._current_buffer_index]
        if not isinstance(model_output, np.ndarray):
            model_output = np.asarray(model_output, dtype=np.float64)
        self._current_buffer_index += 1
        return self._model_output_to_action(model_output)


def get_pi_remote_action(
    pi_policy: PiRemotePolicy,
    arena_obs: dict,
    camera_key_map: dict[str, str],
    default_image_shape: tuple[int, int, int],
    device,
) -> "torch.Tensor":
    """Build policy obs (pi format), get action from Pi server, return Arena 7D action tensor."""
    import torch
    from molmo_spaces_isaac.arena.molmospaces_learned_policy_adapter import (
        arena_obs_to_policy_obs,
        policy_action_to_arena_action,
    )
    policy_obs = arena_obs_to_policy_obs(
        arena_obs,
        camera_key_map=camera_key_map,
        action_move_group_names=["arm", "gripper"],
        default_image_shape=default_image_shape,
        device=torch.device(device) if not isinstance(device, torch.device) else device,
        policy_obs_format="pi",
    )
    action_dict = pi_policy.get_action([policy_obs])
    return policy_action_to_arena_action(
        action_dict,
        action_spec={"arm": 7, "gripper": 2},
        action_scale_pos=0.5,
        action_scale_rot=0.3,
        action_gripper_open_threshold=0.5,
        device=torch.device(device) if not isinstance(device, torch.device) else device,
    )
