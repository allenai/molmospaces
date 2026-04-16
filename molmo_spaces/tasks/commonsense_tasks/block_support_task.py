import logging
from typing import Any

import numpy as np

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body

log = logging.getLogger(__name__)


class BlockSupportTask(PickTask):
    """Block support task where blocks must be stacked in order.
    With N blocks, the reward is the fraction of correctly stacked pairs:
    block_2 on block_1, block_3 on block_2, etc.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        # Gates the terminal support-summary log so it fires exactly once per
        # episode, at the actual terminal step (not at any earlier is_done check).
        self._support_summary_logged = False

    def reset(self):
        self._support_summary_logged = False
        return super().reset()

    def step(self, action):
        result = super().step(action)
        # Log the support summary exactly once, on the step where is_done
        # first becomes True — this captures the terminal state rather than
        # the initial or any mid-episode state.
        if not self._support_summary_logged and self.is_done():
            self._log_terminal_summary()
            self._support_summary_logged = True
        return result

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

        Uses is_object_supported_by_body, which integrates actual contact
        forces across all contacts between the pair and requires the upward
        component to exceed 50% of the held block's weight. This is more
        reliable than the first-matching-contact + normal-heuristic used by
        get_supporting_geom, which on stacked blocks can miss valid supports.
        """
        block_names = self.config.task_config.block_names
        if len(block_names) < 2:
            return np.ones(self._env.n_batch)

        num_pairs = len(block_names) - 1
        rewards = np.zeros(self._env.n_batch)

        # pair_statuses: list[tuple[str, str, bool]] = []
        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]

            correct_pairs = 0
            for pair_idx in range(num_pairs):
                lower_block = MlSpacesObject(data=data, object_name=block_names[pair_idx])
                upper_block = MlSpacesObject(data=data, object_name=block_names[pair_idx + 1])

                supported = is_object_supported_by_body(
                    data=data,
                    object_id=upper_block.object_id,
                    support_id=lower_block.object_id,
                    geometric_fallback=True,
                )
                # if i == 0:
                #     pair_statuses.append(
                #         (block_names[pair_idx + 1], block_names[pair_idx], supported)
                #     )
                if supported:
                    correct_pairs += 1

            rewards[i] = correct_pairs / num_pairs

        # log.info(
        #     "[BLOCK SUPPORT] step pairs: "
        #     + ", ".join(f"{u}->on->{l}={s}" for u, l, s in pair_statuses)
        # )

        return rewards

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""
        if self.config.task_type in ("block_support", "block_stacking"):
            return self.get_reward()[0] >= 1
        else:
            raise ValueError(f"Invalid action_type {self.config.task_type}")

    def _log_terminal_summary(self) -> None:
        """Called once at the step where the task first becomes done.
        Reports, for each pair, whether the upper block is currently supported
        by the lower block, plus the final reward.
        """
        block_names = self.config.task_config.block_names
        if len(block_names) < 2:
            reward = float(self.get_reward()[0])
            log.info(f"[BLOCK SUPPORT] final reward={reward:.3f} (no pairs)")
            return

        data = self._env.mj_datas[0]

        for pair_idx in range(len(block_names) - 1):
            lower_name = block_names[pair_idx]
            upper_name = block_names[pair_idx + 1]
            upper_block = MlSpacesObject(data=data, object_name=upper_name)
            lower_block = MlSpacesObject(data=data, object_name=lower_name)

            registered = is_object_supported_by_body(
                data=data,
                object_id=upper_block.object_id,
                support_id=lower_block.object_id,
                geometric_fallback=True,
            )
            log.info(
                f"[BLOCK SUPPORT] pair {pair_idx}: "
                f"upper='{upper_name}' supported_by lower='{lower_name}' -> registered={registered}"
            )

        reward = float(self.get_reward()[0])
        num_pairs = len(block_names) - 1
        log.info(
            f"[BLOCK SUPPORT] final reward={reward:.3f} "
            f"(correct_pairs={round(reward * num_pairs)}/{num_pairs})"
        )
