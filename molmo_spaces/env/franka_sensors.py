import gymnasium.spaces as gyms
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.utils.pose import pose_mat_to_7d


class Fr3Link0PoseSensor(Sensor):
    """Sensor for fr3_link0 world pose in 7D format (the actual arm kinematic root, above the pedestal)."""

    def __init__(self, uuid: str = "fr3_link0_pose", namespace: str = "") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self._namespace = namespace
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get fr3_link0 world pose."""
        try:
            robot_view = env.robots[batch_index].robot_view
            arm_group = robot_view.get_move_group("arm")
            pose = arm_group.root_frame_to_world
            return pose_mat_to_7d(pose).astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not get fr3_link0 pose: {e}")
            raise e
