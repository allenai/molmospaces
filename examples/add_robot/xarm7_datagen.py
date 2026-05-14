from pathlib import Path

from molmo_spaces.configs import PickTaskSamplerConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import AllCameraTypes, CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.data_generation.config_registry import register_config

from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from xarm7_config import XArm7RobotConfig


class XArm7CameraSystem(CameraSystemConfig):
    img_resolution: tuple[int, int] = (624, 352)

    cameras: list[AllCameraTypes] = [
        MjcfCameraConfig(
            name="wrist_camera_zed_mini",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/",
        ),
        MjcfCameraConfig(
            name="exo_camera_zed_2",
            mjcf_name="exo_camera",
            robot_namespace="robot_0/",
        ),
    ]


@register_config("XArm7PickDataGenConfig")
class XArm7PickDataGenConfig(PickBaseConfig):
    robot_config: XArm7RobotConfig = XArm7RobotConfig()
    camera_config: XArm7CameraSystem = XArm7CameraSystem()
    output_dir: Path = Path("experiment_output") / "datagen" / "xarm7_pick_v1"
    num_workers: int = 4  # number of rollout processes
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        dataset_name="procthor-10k",  # Which house dataset to use
        house_inds=list(range(4)),  # Run in first 4 houses
        samples_per_house=2,  # Number of episodes to sample per house
        # The XArm7 has a max reach of 0.7m, constrain to 0.6m for safety
        base_pose_sampling_radius_range=(0.15, 0.6),
        # Offset between bottom of robot base and pickup object (see the base size in the robot config)
        robot_object_z_offset=-0.25,
        # Randomize the robot z around the offset
        robot_object_z_offset_random_min=-0.2,
        robot_object_z_offset_random_max=0.2,
    )

    @property
    def tag(self) -> str:
        return "xarm7_pick_datagen"
