import logging
from abc import ABC, abstractmethod

from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.camera_configs import CameraSystemConfig, MjcfCameraConfig
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

log = logging.getLogger(__name__)


class BaseRobotEvalOverride(ABC):

    @abstractmethod
    def apply_overrides(
        self,
        episode_spec: EpisodeSpec,
        camera_config: CameraSystemConfig,
    ) -> None:
        pass


class CAPRobotEvalOverride(BaseRobotEvalOverride):

    def apply_overrides(
        self,
        episode_spec: EpisodeSpec,
        camera_config: CameraSystemConfig,
    ) -> None:
        log.info("Applying CAP robot evaluation overrides")
        
        camera_config.cameras[0] = MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/gripper/",
            fov=53.0,
            fov_noise_degrees=(0.0, 0.0),
            pos_noise_range=(0.0, 0.0),
            orientation_noise_degrees=0.0,
            record_depth=True,
        )
        
        camera_config.cameras[1].record_depth = True
        camera_config.cameras[1].fov = 71
        
        rot_base = R.from_quat(
            episode_spec.task["robot_base_pose"][3:7], scalar_first=True
        ).as_matrix()
        episode_spec.task["robot_base_pose"][:3] += 0.05 * rot_base[0:3, 0]
        episode_spec.task["robot_base_pose"][2] -= 0.2
        
        camera_config.img_resolution = (960, 720)
        
        episode_spec.robot.init_qpos = {
            "base": [],
            "arm": [[0, -1.5, 0.116, -2.45, 0, 0.842, 0.965]],
            "gripper": [0.00296, 0.00296],
        }


ROBOT_OVERRIDE_REGISTRY: dict[str, type[BaseRobotEvalOverride]] = {
    "FrankaCAPRobotConfig": CAPRobotEvalOverride,
}

def get_robot_override(robot_config) -> BaseRobotEvalOverride | None:
    robot_class_name = robot_config.__class__.__name__
    override_class = ROBOT_OVERRIDE_REGISTRY.get(robot_class_name)
    
    if override_class is not None:
        log.info(f"Found robot override for {robot_class_name}")
        return override_class()
    
    return None