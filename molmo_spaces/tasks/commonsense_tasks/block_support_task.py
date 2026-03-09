from typing import Any

import numpy as np

from molmo_spaces.env.data_views import MjThorObject
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom


class BlockSupportTask(PickTask):
    """Block support task where blocks must be stacked in order.
    With N blocks, the reward is the fraction of correctly stacked pairs:
    block_2 on block_1, block_3 on block_2, etc.
    """

    def get_task_description(self) -> str:
        n = len(self.config.task_config.block_names)
        return f"Stack {n} blocks on top of each other."

    def get_info(self) -> list[dict[str, Any]]:
        """Override to use block-stacking success instead of pick-lift success."""
        parent_info = super().get_info()
        stacking_success = self.get_reward()[0] >= 1
        for info in parent_info:
            info["success"] = stacking_success
        return parent_info

    def get_reward(self) -> np.ndarray:
        """Calculate reward based on whether blocks are stacked in order.
        Returns reward = (number of correctly stacked pairs) / (total pairs needed).
        Full reward (1.0) means block_2 is on block_1, block_3 on block_2, etc.
        """
        block_names = self.config.task_config.block_names
        if len(block_names) < 2:
            return np.ones(self._env.n_batch)

        num_pairs = len(block_names) - 1
        rewards = np.zeros(self._env.n_batch)

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            model = data.model

            correct_pairs = 0
            for pair_idx in range(num_pairs):
                lower_block = MjThorObject(data=data, object_name=block_names[pair_idx])
                upper_block = MjThorObject(data=data, object_name=block_names[pair_idx + 1])

                # Check if upper block is supported by lower block
                upper_supporting_geom = get_supporting_geom(data, upper_block.object_id)
                if upper_supporting_geom is not None:
                    supporting_body_id = model.body_rootid[model.geom_bodyid[upper_supporting_geom]]
                    if supporting_body_id == lower_block.object_id:
                        correct_pairs += 1

            rewards[i] = correct_pairs / num_pairs

        return rewards

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""
        if self.config.task_type in ("block_support", "block_stacking"):
            return self.get_reward()[0] >= 1
        else:
            raise ValueError(f"Invalid action_type {self.config.task_type}")
