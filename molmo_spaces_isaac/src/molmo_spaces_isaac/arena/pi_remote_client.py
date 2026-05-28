"""Policy via OpenPI server (--policy_type pi_remote). No molmo_spaces/mujoco in this process."""

from __future__ import annotations

import logging
import os
import site
import sys
import time
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Isaac Sim bundles websockets 12.x via omni.kit.pip_archive; the openpi server requires 15.x.
# Inject the openpi venv's site-packages so openpi_client finds the correct websockets version.
_OPENPI_VENV_SITE = os.environ.get(
    "MOLMO_OPENPI_VENV_SITE",
    os.path.expanduser("~/openpi/.venv/lib/python3.11/site-packages"),
)
if os.path.isdir(_OPENPI_VENV_SITE) and _OPENPI_VENV_SITE not in sys.path:
    site.addsitedir(_OPENPI_VENV_SITE)
if os.path.isdir(_OPENPI_VENV_SITE):
    if _OPENPI_VENV_SITE in sys.path:
        sys.path.remove(_OPENPI_VENV_SITE)
    sys.path.insert(0, _OPENPI_VENV_SITE)
    log.debug("Prepended openpi venv site-packages for websockets 15.x: %s", _OPENPI_VENV_SITE)

# Optional: openpi_client for remote inference
try:
    from openpi_client import websocket_client_policy
    _HAS_OPENPI_CLIENT = True
except ImportError:
    _HAS_OPENPI_CLIENT = False
    websocket_client_policy = None


if _HAS_OPENPI_CLIENT:

    class _LongInferenceWebsocketClientPolicy(websocket_client_policy.WebsocketClientPolicy):
        """OpenPI websocket client variant that tolerates slow first inference.

        The default websockets keepalive closes the connection if the OpenPI
        server spends longer than 20s compiling or running an inference. This
        wrapper leaves inference timeouts to PiRemotePolicy instead.
        """

        def _wait_for_server(self):
            from openpi_client import msgpack_numpy
            import websockets.sync.client

            logging.info("Waiting for server at %s...", self._uri)
            while True:
                try:
                    headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                    connect_kwargs = {
                        "compression": None,
                        "max_size": None,
                        "additional_headers": headers,
                        "ping_interval": None,
                        "ping_timeout": None,
                    }
                    supported = inspect.signature(websockets.sync.client.connect).parameters
                    connect_kwargs = {k: v for k, v in connect_kwargs.items() if k in supported}
                    conn = websockets.sync.client.connect(self._uri, **connect_kwargs)
                    metadata = msgpack_numpy.unpackb(conn.recv())
                    return conn, metadata
                except ConnectionRefusedError:
                    logging.info("Still waiting for server...")
                    time.sleep(5)
else:
    _LongInferenceWebsocketClientPolicy = None


def _normalize_task_prompt(task_description: str | None) -> str:
    """Match MolmoSpaces PI_Policy prompt text without inventing punctuation."""
    prompt = (task_description or "pick up the object.").strip().lower()
    return prompt


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
        grasping_threshold: float = 0.01,
        chunk_size: int = 15,
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
        self._client = _LongInferenceWebsocketClientPolicy(host=host, port=port)
        self._task_description = _normalize_task_prompt(task_description)
        self._grasping_threshold = grasping_threshold
        self._chunk_size = chunk_size
        self._inference_timeout_s = inference_timeout_s
        self._actions_buffer: np.ndarray | None = None
        self._current_buffer_index = 0
        self._current_chunk_index = -1
        self._starting_time: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        trace_dir = (os.environ.get("MOLMO_PI_TRACE_DIR") or "").strip()
        self._trace_dir = Path(trace_dir).expanduser().resolve() if trace_dir else None
        self._trace_call_index = 0
        self._trace_chunk_index = 0
        if self._trace_dir is not None:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            log.info("Pi remote trace enabled: %s", self._trace_dir)
        log.info(
            "Pi remote policy connected to %s:%s (chunk_size=%s, grasping_threshold=%s, inference_timeout_s=%s)",
            host,
            port,
            chunk_size,
            grasping_threshold,
            inference_timeout_s,
        )

    def reset(self) -> None:
        self._actions_buffer = None
        self._current_buffer_index = 0
        self._current_chunk_index = -1
        self._starting_time = None

    def set_task_description(self, task_description: str) -> None:
        self._task_description = _normalize_task_prompt(task_description)

    def _write_trace_chunk(self, chunk_index: int, model_input: dict[str, Any], raw_actions: np.ndarray) -> None:
        if self._trace_dir is None:
            return
        chunk_dir = self._trace_dir / f"chunk_{chunk_index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        np.save(chunk_dir / "actions.npy", np.asarray(raw_actions))
        np.save(chunk_dir / "joint_position.npy", np.asarray(model_input.get("observation/joint_position")))
        np.save(chunk_dir / "gripper_position.npy", np.asarray(model_input.get("observation/gripper_position")))
        metadata = {
            "chunk_index": chunk_index,
            "prompt": model_input.get("prompt"),
            "actions_shape": list(np.asarray(raw_actions).shape),
        }
        try:
            from PIL import Image

            for key, name in (
                ("observation/exterior_image_1_left", "exterior_image_1_left.png"),
                ("observation/wrist_image_left", "wrist_image_left.png"),
            ):
                img = np.asarray(model_input.get(key))
                metadata[f"{key}_shape"] = list(img.shape)
                if img.ndim == 3 and img.shape[-1] >= 3:
                    Image.fromarray(img[:, :, :3].astype(np.uint8)).save(chunk_dir / name)
        except Exception as e:
            metadata["image_write_error"] = str(e)
            for key, name in (
                ("observation/exterior_image_1_left", "exterior_image_1_left.npy"),
                ("observation/wrist_image_left", "wrist_image_left.npy"),
            ):
                np.save(chunk_dir / name, np.asarray(model_input.get(key)))
        with open(chunk_dir / "metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

    def _write_trace_call(
        self,
        call_index: int,
        chunk_index: int,
        buffer_index: int,
        model_output: np.ndarray,
        action: dict[str, np.ndarray],
    ) -> None:
        if self._trace_dir is None:
            return
        import json

        out = {
            "call_index": call_index,
            "chunk_index": chunk_index,
            "buffer_index": buffer_index,
            "model_output": np.asarray(model_output, dtype=float).tolist(),
            "arm": np.asarray(action.get("arm", []), dtype=float).tolist(),
            "gripper": np.asarray(action.get("gripper", []), dtype=float).tolist(),
        }
        with open(self._trace_dir / f"call_{call_index:05d}.json", "w") as f:
            json.dump(out, f, indent=2)

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
        prompt = _normalize_task_prompt(self._task_description)
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
        call_index = self._trace_call_index
        self._trace_call_index += 1
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
            self._current_chunk_index = self._trace_chunk_index
            self._trace_chunk_index += 1
            self._write_trace_chunk(self._current_chunk_index, model_input, self._actions_buffer)
        model_output = self._actions_buffer[self._current_buffer_index]
        if not isinstance(model_output, np.ndarray):
            model_output = np.asarray(model_output, dtype=np.float64)
        buffer_index = self._current_buffer_index
        self._current_buffer_index += 1
        action = self._model_output_to_action(model_output)
        self._write_trace_call(call_index, self._current_chunk_index, buffer_index, model_output, action)
        return action


def get_pi_remote_action(
    pi_policy: PiRemotePolicy,
    arena_obs: dict,
    camera_key_map: dict[str, str],
    default_image_shape: tuple[int, int, int],
    device,
    use_joint_pos_control: bool = False,
    use_joint_velocity_control: bool = False,
) -> "torch.Tensor":
    """Build policy obs (pi format), get action from Pi server, return Arena action tensor.

    use_joint_pos_control=True: 8D (7 arm joint angles + 1 binary gripper), for pi0 DROID
    joint-position policies. Requires Arena Franka configured with JointPositionActionCfg.
    use_joint_velocity_control=True: 8D (7 arm joint velocities + 1 binary gripper), for
    OpenPI DROID checkpoints such as pi05_droid. Requires Arena DROID configured with
    JointVelocityActionCfg.
    use_joint_pos_control=False (default): 7D delta EEF pose, for normalized EEF policies.
    """
    import torch
    from molmo_spaces_isaac.arena.molmospaces_learned_policy_adapter import (
        arena_obs_to_policy_obs,
        policy_action_to_arena_action,
        policy_joint_pos_action_to_arena_action,
        policy_joint_velocity_action_to_arena_action,
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
    _dev = torch.device(device) if not isinstance(device, torch.device) else device
    if use_joint_velocity_control:
        return policy_joint_velocity_action_to_arena_action(
            action_dict,
            action_gripper_open_threshold=0.5,
            device=_dev,
        )
    if use_joint_pos_control:
        return policy_joint_pos_action_to_arena_action(
            action_dict,
            action_gripper_open_threshold=0.5,
            device=_dev,
        )
    return policy_action_to_arena_action(
        action_dict,
        action_spec={"arm": 7, "gripper": 2},
        action_scale_pos=0.5,
        action_scale_rot=0.3,
        action_gripper_open_threshold=0.5,
        device=_dev,
    )
