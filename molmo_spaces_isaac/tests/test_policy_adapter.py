import numpy as np
import torch

from molmo_spaces_isaac.arena.molmospaces_learned_policy_adapter import (
    policy_joint_velocity_action_to_arena_action,
)


def test_policy_joint_velocity_action_to_arena_action_clips_arm_and_binarizes_gripper():
    action = {
        "arm": np.array([2.0, -2.0, 0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float32),
        "gripper": np.array([255.0], dtype=np.float32),
    }

    arena_action = policy_joint_velocity_action_to_arena_action(
        action,
        action_gripper_open_threshold=0.5,
        device=torch.device("cpu"),
    )

    assert arena_action.shape == (1, 8)
    np.testing.assert_allclose(
        arena_action.cpu().numpy()[0],
        np.array([1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 1.0], dtype=np.float32),
    )
