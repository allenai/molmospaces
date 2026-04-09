"""Observation and action conversion between Isaac Lab Arena and remote policies (e.g. Pi0)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def arena_obs_to_policy_obs(
    arena_obs: dict,
    camera_key_map: dict[str, str],
    action_move_group_names: list[str],
    default_image_shape: tuple[int, int, int],
    device: torch.device,
    policy_obs_format: str = "standard",
) -> dict[str, Any]:
    """Build the obs dict for remote policies from Arena obs. pi format adds top-level qpos key."""
    policy_obs: dict[str, Any] = {}
    if "policy" not in arena_obs:
        raise KeyError("arena_obs must contain 'policy' group")

    p = arena_obs["policy"]
    joint_pos = p.get("joint_pos")
    gripper_pos = p.get("gripper_pos")
    if joint_pos is None:
        raise KeyError("arena_obs['policy'] must contain 'joint_pos'")

    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.cpu().numpy()
    if isinstance(gripper_pos, torch.Tensor):
        gripper_pos = gripper_pos.cpu().numpy()

    if joint_pos.ndim == 2:
        joint_pos = joint_pos[0]
        if gripper_pos is not None:
            gripper_pos = gripper_pos[0]

    if joint_pos.size >= 9:
        arm = np.asarray(joint_pos[:7], dtype=np.float32)
        gripper = np.asarray(joint_pos[7:9], dtype=np.float32) if gripper_pos is None else np.asarray(gripper_pos, dtype=np.float32)
    else:
        arm = np.asarray(joint_pos[:7], dtype=np.float32)
        gripper = np.asarray(gripper_pos, dtype=np.float32) if gripper_pos is not None else np.zeros(2, dtype=np.float32)

    policy_obs["robot_state"] = {"qpos": {"arm": arm, "gripper": gripper}}
    for g in action_move_group_names:
        if g not in policy_obs["robot_state"]["qpos"]:
            if g == "arm":
                policy_obs["robot_state"]["qpos"]["arm"] = arm
            elif g == "gripper":
                policy_obs["robot_state"]["qpos"]["gripper"] = gripper
            else:
                policy_obs["robot_state"]["qpos"][g] = np.zeros(1, dtype=np.float32)

    for cam_name, arena_key in camera_key_map.items():
        parts = arena_key.split(".", 1)
        if len(parts) == 2:
            group, term = parts[0], parts[1]
            if group in arena_obs and term in arena_obs[group]:
                img = arena_obs[group][term]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                while img.ndim > 3:  # strip (num_envs, ...) batch dims
                    img = img[0]
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[:, :, :3]  # RGBA → RGB
                policy_obs[cam_name] = np.asarray(img, dtype=np.uint8)
            else:
                policy_obs[cam_name] = np.zeros(default_image_shape, dtype=np.uint8)
        else:
            policy_obs[cam_name] = np.zeros(default_image_shape, dtype=np.uint8)

    # Pi format: top-level "qpos" and keys wrist_camera, exo_camera_1.
    if policy_obs_format == "pi":
        policy_obs["qpos"] = policy_obs["robot_state"]["qpos"]

    return policy_obs


def policy_action_to_arena_action(
    policy_action: dict[str, np.ndarray],
    action_spec: dict[str, int],
    action_scale_pos: float,
    action_scale_rot: float,
    action_gripper_open_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert delta-EEF policy action to Arena 7D (3 pos delta + 3 rot delta + 1 gripper)."""
    arm = policy_action.get("arm")
    gripper = policy_action.get("gripper")
    if arm is None:
        arm = np.zeros(action_spec.get("arm", 7), dtype=np.float32)
    if gripper is None:
        gripper = np.zeros(action_spec.get("gripper", 2), dtype=np.float32)
    arm = np.atleast_1d(arm).astype(np.float32)
    gripper = np.atleast_1d(gripper).astype(np.float32)

    if arm.size >= 6:
        delta_pos = (arm[:3] - 0.5) * action_scale_pos
        delta_rot = (arm[3:6] - 0.5) * action_scale_rot
    else:
        delta_pos = (arm[:3] - 0.5) * action_scale_pos if arm.size >= 3 else np.zeros(3, dtype=np.float32)
        delta_rot = np.zeros(3, dtype=np.float32)

    gripper_cmd = float(np.mean(gripper) > action_gripper_open_threshold)
    action_7d = np.concatenate([delta_pos, delta_rot, [gripper_cmd]], axis=0).astype(np.float32)
    return torch.from_numpy(action_7d).unsqueeze(0).to(device)


def policy_joint_pos_action_to_arena_action(
    policy_action: dict[str, np.ndarray],
    action_gripper_open_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Convert joint-position policy output to Arena 8D (7 absolute joint angles + 1 binary gripper)."""
    arm = policy_action.get("arm")
    gripper = policy_action.get("gripper")
    if arm is None:
        arm = np.zeros(7, dtype=np.float32)
    if gripper is None:
        gripper = np.zeros(1, dtype=np.float32)
    arm = np.atleast_1d(arm).astype(np.float32)[:7]
    if arm.size < 7:
        arm = np.pad(arm, (0, 7 - arm.size))
    gripper = np.atleast_1d(gripper).astype(np.float32)
    gripper_cmd = float(np.mean(gripper) > action_gripper_open_threshold)
    action_8d = np.concatenate([arm, [gripper_cmd]], axis=0).astype(np.float32)
    return torch.from_numpy(action_8d).unsqueeze(0).to(device)
