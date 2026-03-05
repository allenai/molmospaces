import numpy as np

from molmo_spaces.env.data_views import MjThorObject
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom


class BlockSupportTask(PickTask):
    """Block support task where one block must be placed on top of another.
    The reward is nonzero when one of the blocks is resting on top of the other block.
    """

    def get_task_description(self) -> str:
        return "Stack the blocks."

    def get_reward(self) -> np.ndarray:
        """Calculate reward based on whether one block is resting on the other.
        Returns a positive reward when one of the blocks is supported by the other block.
        """
        # FIXME: may need to update this to also include that the
        rewards = np.zeros(self._env.n_batch)

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            model = data.model

            # Get both block objects
            red_block = MjThorObject(data=data, object_name="red_cube")
            blue_block = MjThorObject(data=data, object_name="red_cube_2")

            # Check if red block is supported by blue block
            red_supporting_geom = get_supporting_geom(data, red_block.object_id)
            if red_supporting_geom is not None:
                supporting_body_id = model.body_rootid[model.geom_bodyid[red_supporting_geom]]
                if supporting_body_id == blue_block.object_id:
                    # Red block is on top of blue block
                    rewards[i] = 1.0
                    continue

            # Check if blue block is supported by red block
            blue_supporting_geom = get_supporting_geom(data, blue_block.object_id)
            if blue_supporting_geom is not None:
                supporting_body_id = model.body_rootid[model.geom_bodyid[blue_supporting_geom]]
                if supporting_body_id == red_block.object_id:
                    # Blue block is on top of red block
                    rewards[i] = 1.0
                    continue

        return rewards

    def judge_success(self) -> bool:
        """Judge if the task was successful (for data generation)."""

        # Get pickup object using Object class for proper positioning
        # data = self._env.mj_datas[0]
        # pickup_obj = MjThorObject(data=data, object_name=self.config.task_config.pickup_obj_name)
        if self.config.task_type == "block_support":
            return self.get_reward()[0] >= 1
        else:
            raise ValueError(f"Invalid action_type {self.config.task_type}")
