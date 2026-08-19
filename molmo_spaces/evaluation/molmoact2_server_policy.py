from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _ensure_lerobot_env() -> None:
    if "HF_LEROBOT_HOME" not in os.environ and "LEROBOT_DATA_ROOT" in os.environ:
        os.environ["HF_LEROBOT_HOME"] = os.environ["LEROBOT_DATA_ROOT"]
    if "LEROBOT_HOME" in os.environ:
        os.environ.setdefault("HF_LEROBOT_HOME", os.environ["LEROBOT_HOME"])
        del os.environ["LEROBOT_HOME"]


def _ensure_import_paths(mm_olmo_path: str) -> None:
    if not mm_olmo_path:
        raise ValueError(
            "mm_olmo_path is not set. Point it at a local checkout of the molmoact2 repo "
            "(config field mm_olmo_path or env var MOLMOACT2_MM_OLMO_PATH)."
        )
    mm_olmo_root = Path(mm_olmo_path).expanduser().resolve()
    if not mm_olmo_root.exists():
        raise FileNotFoundError(f"mm_olmo_path does not exist: {mm_olmo_root}")

    lerobot_src = mm_olmo_root / "lerobot_v050" / "src"
    for candidate in (mm_olmo_root, lerobot_src):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


@dataclass
class Molmoact2PolicyState:
    model_obs_history: dict[int, list[Any]] = field(default_factory=dict)
    model_action_queues: dict[int, list[Any]] = field(default_factory=dict)
    actions_buffer: np.ndarray | None = None
    current_buffer_index: int = 0
    starting_time: float | None = None
    obs_history: list[dict[str, Any]] = field(default_factory=list)


class Molmoact2ServerPolicy(InferencePolicy, StatefulPolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        # BasePolicy's second parameter is `task: BaseMujocoTask | None`, not a task-type
        # string. Passing task_type here set self.task = "pick", so _resolve_prompt hit
        # "pick".get_task_description(). In server mode there is no task object -- the
        # prompt arrives in the observation -- and both self.task call sites already
        # guard on None, which is the path they were written for.
        super().__init__(exp_config)
        policy_config = exp_config.policy_config
        self.checkpoint_path = policy_config.checkpoint_path
        self.mm_olmo_path = policy_config.mm_olmo_path
        self.device = policy_config.device
        self.exo_camera_key = str(getattr(policy_config, "exo_camera_key", "") or "")
        checkpoint_view_match = re.search(r"(\d+)_views", str(self.checkpoint_path))
        self.expected_exo_image_count = (
            int(checkpoint_view_match.group(1)) if checkpoint_view_match is not None else None
        )
        self.seq_len = policy_config.seq_len
        self.num_steps = policy_config.num_steps
        self.n_action_steps = (
            policy_config.n_action_steps
            if policy_config.n_action_steps is not None
            else policy_config.chunk_size
        )
        self.action_mode = policy_config.action_mode
        self.discrete_action_tokenizer = policy_config.discrete_action_tokenizer
        self.discrete_generation_max_steps = policy_config.discrete_generation_max_steps
        self.style = policy_config.style
        self.norm_tag = policy_config.norm_tag
        self.verbose = policy_config.verbose
        self.grasping_type = policy_config.grasping_type
        self.grasping_threshold = policy_config.grasping_threshold

        self.model = None
        self.robot_processor = None
        self.n_obs_steps = 1
        self.resolved_norm_tag = str(self.norm_tag or "")
        self.resolved_style = str(self.style or "")
        self.policy_inference_path = "per_frame_chunk_regen"
        self._logged_image_selection = False

        self.actions_buffer: np.ndarray | None = None
        self.current_buffer_index = 0
        self.starting_time: float | None = None
        self.obs_history: list[dict[str, Any]] = []

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        if self.task is not None:
            return str(self.task.get_task_description() or "")

        for key in ("task", "instruction", "prompt", "question"):
            value = str(obs.get(key, "") or "").strip()
            if value:
                return value
        return ""

    def get_state(self):
        model_obs_history: dict[int, list[Any]] = {}
        model_action_queues: dict[int, list[Any]] = {}
        if self.model is not None:
            raw_obs_history = getattr(self.model, "_obs_history", {})
            raw_action_queues = getattr(self.model, "_action_queues", {})
            model_obs_history = {
                int(idx): [_clone_value(item) for item in history]
                for idx, history in raw_obs_history.items()
            }
            model_action_queues = {
                int(idx): [_clone_value(item) for item in queue]
                for idx, queue in raw_action_queues.items()
            }
        return Molmoact2PolicyState(
            model_obs_history=model_obs_history,
            model_action_queues=model_action_queues,
            actions_buffer=None if self.actions_buffer is None else self.actions_buffer.copy(),
            current_buffer_index=self.current_buffer_index,
            starting_time=self.starting_time,
            obs_history=[_clone_value(step_obs) for step_obs in self.obs_history],
        )

    def set_state(self, state: Molmoact2PolicyState):
        self.actions_buffer = None if state.actions_buffer is None else state.actions_buffer.copy()
        self.current_buffer_index = state.current_buffer_index
        self.starting_time = state.starting_time
        self.obs_history = [_clone_value(step_obs) for step_obs in state.obs_history]
        if self.model is not None:
            self.model._obs_history = defaultdict(
                lambda: deque(),
                {
                    int(idx): deque(
                        (_clone_value(item) for item in history),
                        maxlen=self.n_obs_steps,
                    )
                    for idx, history in state.model_obs_history.items()
                },
            )
            self.model._action_queues = defaultdict(
                lambda: deque(),
                {
                    int(idx): deque(_clone_value(item) for item in queue)
                    for idx, queue in state.model_action_queues.items()
                },
            )

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None
        self.obs_history = []
        if self.model is not None:
            self.model.reset()

    def _select_image_keys(self, obs: dict[str, Any]) -> list[str]:
        available_image_keys = [
            key for key, value in obs.items() if isinstance(value, np.ndarray) and value.ndim >= 2
        ]
        if not available_image_keys:
            raise KeyError("Observation does not contain any image-like numpy arrays.")

        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )
        if wrist_camera_key not in obs:
            raise KeyError(
                "Could not resolve a wrist camera for MolmoAct2 policy. "
                f"Available image-like keys: {sorted(available_image_keys)}"
            )

        ordered_candidates: list[str] = []
        if self.exo_camera_key:
            ordered_candidates.append(self.exo_camera_key)
            if self.exo_camera_key.endswith("_1"):
                ordered_candidates.append(f"{self.exo_camera_key[:-2]}_2")
            if self.exo_camera_key.endswith("_left"):
                ordered_candidates.append(f"{self.exo_camera_key[:-4]}right")

        ordered_candidates.extend(
            [
                "exo_camera_1",
                "exo_camera_2",
                "droid_shoulder_light_randomization",
                "randomized_zed2_analogue_1",
                "randomized_zed2_analogue_2",
                "randomized_gopro_analogue_1",
            ]
        )
        ordered_candidates = _dedupe_preserve_order(ordered_candidates)

        selected_image_keys = [key for key in ordered_candidates if key in obs]
        if not selected_image_keys:
            raise KeyError(
                "Could not resolve any exocentric camera for MolmoAct2 policy. "
                f"Tried {ordered_candidates}, available image-like keys: {sorted(available_image_keys)}"
            )
        if self.expected_exo_image_count is not None:
            selected_image_keys = selected_image_keys[: self.expected_exo_image_count]

        selected_image_keys.append(wrist_camera_key)
        return _dedupe_preserve_order(selected_image_keys)

    def prepare_model(self, model_name: str = ""):
        del model_name

        _ensure_lerobot_env()
        _ensure_import_paths(self.mm_olmo_path)

        from lerobot.policies.molmoact.configuration_molmoact import MolmoActConfig
        from lerobot.policies.molmoact.modeling_molmoact import MolmoActPolicy
        from olmo.extra_tokens import style_uses_depth_output

        self.model_name = os.path.basename(self.checkpoint_path.rstrip("/"))
        config = MolmoActConfig(
            checkpoint_path=str(Path(self.checkpoint_path).expanduser()),
            mm_olmo_path=str(Path(self.mm_olmo_path).expanduser()),
            device=self.device,
            seq_len=self.seq_len,
            num_steps=self.num_steps,
            action_mode=self.action_mode,
            discrete_action_tokenizer=self.discrete_action_tokenizer,
            discrete_generation_max_steps=int(self.discrete_generation_max_steps),
            style=self.style,
            norm_tag=str(self.norm_tag or ""),
            normalization_repo_id=str(self.norm_tag or ""),
            verbose=bool(self.verbose),
        )
        self.model = MolmoActPolicy(config)
        self.model.eval()

        handles = self.model._handles
        if handles is None:
            raise RuntimeError("MolmoActPolicy did not initialize handles.")

        self.robot_processor = handles.robot_processor
        self.n_obs_steps = int(handles.n_obs_steps)
        self.resolved_norm_tag = str(self.norm_tag or handles.norm_tag or "")
        self.resolved_style = str(self.style or handles.default_style or "")
        self.policy_inference_path = (
            "per_frame_chunk_regen"
            if self.action_mode != "continuous" or style_uses_depth_output(self.resolved_style)
            else "select_action_queue"
        )

        log.info(
            "[Molmoact2ServerPolicy] Loaded MolmoAct2 checkpoint %s on %s "
            "(action_mode=%s, style=%s, norm_tag=%s, n_obs_steps=%d, n_action_steps=%s, inference_path=%s)",
            self.checkpoint_path,
            self.device,
            self.action_mode,
            self.resolved_style or "<checkpoint default>",
            self.resolved_norm_tag or "<none>",
            self.n_obs_steps,
            self.n_action_steps,
            self.policy_inference_path,
        )

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            if len(obs) > 1:
                log.warning(
                    "WARNING: obs list has %d elements but only using the first one!", len(obs)
                )
            obs = obs[0]

        selected_image_keys = self._select_image_keys(obs)
        if not self._logged_image_selection:
            log.info(
                "[Molmoact2ServerPolicy] Using image keys in order: %s",
                selected_image_keys,
            )
            self._logged_image_selection = True

        joint_position = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        gripper_position = np.asarray(obs["qpos"]["gripper"][:1], dtype=np.float32)
        state = np.concatenate([joint_position, gripper_position], axis=0)

        model_input = {
            "task": self._resolve_prompt(obs),
            "observation.state": state,
        }
        for image_key in selected_image_keys:
            model_input[f"observation.images.{image_key}"] = obs[image_key]
        if self.style:
            model_input["style"] = self.style
        return model_input

    def _refresh_action_buffer(self) -> None:
        if self.model is None:
            raise RuntimeError("MolmoAct2 local model has not been prepared.")

        observations = self.obs_history[-self.n_obs_steps :]
        result = self.model.generate_inference_result_from_observations(
            observations,
            norm_tag=self.resolved_norm_tag or None,
            num_steps=self.num_steps,
            n_action_steps=self.n_action_steps,
        )
        if result.actions is None:
            raise RuntimeError(f"MolmoAct2 style '{result.style}' did not produce actions.")

        action_chunk = result.actions
        if self.robot_processor is not None:
            action_chunk = self.robot_processor.unnormalize_action(
                action_chunk,
                repo_id=self.resolved_norm_tag,
            )

        actions = np.asarray(action_chunk.detach().cpu().numpy(), dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise RuntimeError(f"Unexpected MolmoAct2 action chunk shape: {actions.shape}")

        self.actions_buffer = actions
        self.current_buffer_index = 0

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()

        if self.policy_inference_path == "select_action_queue":
            with torch.inference_mode():
                action_tensor = self.model.select_action(
                    model_input,
                    norm_tag=self.resolved_norm_tag or None,
                    num_steps=self.num_steps,
                    n_action_steps=self.n_action_steps,
                )
            model_output = np.asarray(action_tensor.detach().cpu().numpy(), dtype=np.float32)
            if model_output.ndim > 1:
                model_output = model_output[0]
            return model_output

        self.obs_history.append(_clone_value(model_input))
        if len(self.obs_history) > self.n_obs_steps:
            self.obs_history = self.obs_history[-self.n_obs_steps :]

        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            self._refresh_action_buffer()

        if self.actions_buffer is None:
            raise RuntimeError("MolmoAct2 local inference did not produce an action buffer.")

        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        if self.grasping_type == "continuous":
            gripper_pos = model_output[7] * np.array([255.0], dtype=np.float32)
        else:
            gripper_pos = (
                np.array([255.0], dtype=np.float32)
                if model_output[7] > self.grasping_threshold
                else np.array([0.0], dtype=np.float32)
            )

        action = {
            "arm": np.asarray(model_output[:7], dtype=np.float32).reshape(7),
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "molmoact2_server"
        info["policy_checkpoint"] = getattr(self, "model_name", self.checkpoint_path)
        queue_length = None
        if self.model is not None:
            action_queues = getattr(self.model, "_action_queues", {})
            if 0 in action_queues:
                queue_length = len(action_queues[0])
        info["policy_buffer_length"] = (
            queue_length
            if self.policy_inference_path == "select_action_queue"
            else (None if self.actions_buffer is None else len(self.actions_buffer))
        )
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["policy_inference_path"] = self.policy_inference_path
        info["policy_style"] = self.resolved_style or None
        info["prompt"] = self.task.get_task_description() if self.task is not None else None
        info["norm_tag"] = self.resolved_norm_tag or None
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
