from pathlib import Path

from molmo_spaces.configs import PickTaskSamplerConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaDroidCameraSystem,
)
from molmo_spaces.configs.robot_configs import (
    FrankaRobotConfig,
)
from molmo_spaces.molmo_spaces_constants import register_user_asset_library, register_user_grasp_library

from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler


register_user_asset_library("custom_assets", Path("asset_library"))

register_user_grasp_library("custom_grasps", Path("asset_library"), "custom_assets")

# TODO: write custom task sampler for custom scene

# TODO: write script to generate asset and grasp indices

# TODO: write script to compute asset metadata

@register_config("CustomAssetsDataGenConfig")
class FrankaPickDroidDataGenConfig(PickBaseConfig):
    scene_dataset: str = "user"
    num_workers: int = 1
    robot_config: FrankaRobotConfig = FrankaRobotConfig(base_size=None)
    use_passive_viewer: bool = True
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        dataset_name="user",
        scene_xml_paths=["scene.xml"],
        house_inds=[0],
        samples_per_house=2,
        robot_object_z_offset=0.0,
        robot_object_z_offset_random_max=0.0,
        robot_object_z_offset_random_min=0.0,
        house_variant="base",
    )
    output_dir: Path = Path("experiment_output")

    @property
    def tag(self) -> str:
        return "tutorial_franka_pick_custom_assets"
