from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs import PickTaskSamplerConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
)
from molmo_spaces.configs.robot_configs import (
    FrankaRobotConfig,
)
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import (
    register_user_asset_library,
    register_user_grasp_library,
)

from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.pose import pose_mat_to_7d


register_user_asset_library("custom_assets", Path("asset_library"))

register_user_grasp_library("custom_grasps", Path("asset_library"), "custom_assets")


def random_pose(x_noise: float, y_noise: float, yaw_noise: float):
    pose = np.eye(4)
    pose[0, 3] = np.random.uniform(-x_noise, x_noise)
    pose[1, 3] = np.random.uniform(-y_noise, y_noise)
    pose[:3, :3] = R.from_euler(
        "z", np.random.uniform(-yaw_noise, yaw_noise), degrees=True
    ).as_matrix()
    return pose


class BlockPickupTaskSampler(PickTaskSampler):
    """
    Since we're using a custom scene, we need to implement custom sampling logic
    """

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        task_cfg = self.config.task_config
        robot_view = env.current_robot.robot_view

        robot_view.base.pose = robot_view.base.pose @ random_pose(0.1, 0.1, 30)
        mujoco.mj_fwdPosition(env.current_model, env.current_data)
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj = create_mlspaces_body(env.current_data, task_cfg.pickup_obj_name)
        pickup_obj.pose = pickup_obj.pose @ random_pose(0.05, 0.05, 45)
        mujoco.mj_fwdPosition(env.current_model, env.current_data)

        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()


@register_config("CustomAssetsDataGenConfig")
class FrankaPickDroidDataGenConfig(PickBaseConfig):
    scene_dataset: str = "user"
    num_workers: int = 1
    robot_config: FrankaRobotConfig = FrankaRobotConfig(base_size=None)
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=BlockPickupTaskSampler,
        dataset_name="user",
        scene_xml_paths=["scene.xml"],
        house_inds=None,
        samples_per_house=2,
        house_variant="base",
    )
    output_dir: Path = Path("experiment_output")

    @property
    def tag(self) -> str:
        return "tutorial_franka_pick_custom_assets"
