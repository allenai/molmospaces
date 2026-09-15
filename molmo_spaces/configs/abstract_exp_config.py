from __future__ import annotations

import base64
import datetime
import logging
import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.camera_configs import (
    CameraSystemConfig,
    FixedExocentricCameraConfig,
    MjcfCameraConfig,
    RandomizedExocentricCameraConfig,
    RobotMountedCameraConfig,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.configs.task_configs import AllTaskConfigs
from molmo_spaces.configs.task_sampler_configs import BaseMujocoTaskSamplerConfig
from molmo_spaces.env.camera_manager import RobotMountedCamera
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Profiler

log = logging.getLogger(__name__)


class MlSpacesExpConfig(Config, ABC):
    """Base configuration class for experiments.

    This should be extended to create specific experiment configurations.
    """

    config_version: str = "0.1"

    num_envs: int
    """Number of batched environments per worker (for vectorized physics in CPUMujocoEnv)"""

    num_workers: int = 1
    """Number of worker processes for parallel data generation (episode-level parallelism)"""

    task_type: str
    """Task type: e.g. pick, pick_and_place, etc."""

    use_passive_viewer: bool
    """Launch passive viewer for rendering"""

    viewer_cam_dict: dict
    """Dictionary containing viewer camera parameters"""

    policy_dt_ms: float
    """Default policy time step"""

    ctrl_dt_ms: float
    """Default control time step"""

    sim_dt_ms: float
    """Default simulation time step"""

    seed: int | None = None
    """Random seed for task sampling (if None, generates random seed)"""

    task_horizon: int | None = None
    """Maximum number of steps per episode (if None, no time limit)"""

    end_on_success: bool = False
    """Whether to end episode immediately upon success (overrides task_horizon if True)"""

    collision_free_pose_limit: int = 3

    # Scene configuration  -------------------

    scene_dataset: str
    """Scenes to use, e.g. ithor, procthor-10k, procthor-objaverse. If "user", use the scene_xml_paths in task_sampler_config"""

    data_split: str = "train"
    """Data split to use, e.g. train, val, test"""
    # ----------------------------------------

    @property
    def fps(self) -> float:
        return 1000.0 / self.policy_dt_ms

    # Configuration fields using imported base classes ---------------------

    camera_config: CameraSystemConfig | None = None
    """Configuration for cameras and sensors"""

    robot_config: BaseRobotConfig
    """Configuration for the robot"""

    task_sampler_config: BaseMujocoTaskSamplerConfig
    """Configuration for the task sampler"""

    task_config: AllTaskConfigs
    """Configuration for tasks"""

    task_config_preset_exp: AllTaskConfigs | None = None
    """Cached config for whole experiment"""

    task_config_preset_scn: AllTaskConfigs | None = None
    """Cached config for scene"""

    policy_config: BasePolicyConfig
    """Configuration for policies"""

    # -----------------------------------------------------------------------

    benchmark_path: Path | None = None
    """Contains a json with a list of fully-specified episodes"""

    eval_runtime_params: Any = None
    """Evaluation runtime parameters (optional, set during evaluation initialization)"""

    # Output and profiling --------------------------------------------------

    output_dir: Path
    """Output directory for experiment results"""

    profile: bool = False
    """Whether to enable profiling"""

    profiler: Profiler | None = None
    """Profiler instance (auto-created if profile=True)"""

    datagen_profiler: bool = True
    """Whether or not to use the datagen profiler"""

    # -----------------------------------------------------------------------

    # Logging ---------------------------------------------------------------

    log_level: str = "info"
    """Global logging level: "debug", "info", "warning", "error", "none"""

    use_wandb: bool = False
    """Whether or not to use wandb logging"""

    wandb_project: str | None = None
    """(Optional) The name of the wandb project to use"""

    wandb_name: str | None = None
    """(Optional) The name of the wandb run"""

    # -----------------------------------------------------------------------

    # Backward compatibility aliases for nested class references (ClassVar so Pydantic ignores them)

    CameraConfig: ClassVar[type] = CameraSystemConfig

    RobotConfig: ClassVar[type] = BaseRobotConfig

    PolicyConfig: ClassVar[type] = BasePolicyConfig

    # -----------------------------------------------------------------------

    filter_for_successful_trajectories: bool = True
    """If True, only save successful trajectories to main output directory (failed episodes may be sampled 1% for debug directory). If False, save all trajectories to main output directory"""

    environment_light_intensity: float = 15000.0
    """The environment intensity value for the IBL when using the filament-based renderer"""

    no_cached_map: bool = False
    """Whether or not to cache the generated thormap to disk after creating it for a datagen run"""

    def model_post_init(self, _context: Any) -> None:
        def is_multiple(val_a: float, val_b: float) -> bool:
            if val_b == 0:
                return val_a == 0

            multiplier = round(val_a / val_b)
            return math.isclose(val_a, multiplier * val_b)

        assert is_multiple(self.policy_dt_ms, self.ctrl_dt_ms), (
            "policy_dt_ms must be a multiple of ctrl_dt_ms"
        )
        assert is_multiple(self.ctrl_dt_ms, self.sim_dt_ms), (
            "ctrl_dt_ms must be a multiple of sim_dt"
        )

    @property
    @abstractmethod
    def tag(self) -> str:
        """A string describing the experiment."""

    def save_config(self, output_dir: Path | str | None = None) -> None:
        if output_dir is None:
            output_dir = self.output_dir

        output_dir = Path(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO(wilbert): maybe we should save to another format as well. If using pickle, the
        # context of the module is also included, so to import the pkl outside of the project (like
        # to just validate it outside or reuse it) we would get import errors
        config_path = output_dir / f"experiment_config_{timestamp}.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(self, f)

        log.info(f"Saved experiment configuration to {output_dir}")

    @staticmethod
    def load_config(output_dir: Path) -> MlSpacesExpConfig:
        config_path = output_dir / "experiment_config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        log.info(f"Loaded experiment configuration from {output_dir}")
        return config

    class SavedEpisode(Config):
        """Config information describing a single episode

        The code below this is used for saving episode state so that it can be re-loaded w/o sampling
        """

        camera_config: CameraSystemConfig | None = None
        """Configuration for cameras and sensors"""

        robot_config: BaseRobotConfig | None = None
        """Configuration for the robot"""

        task_config: AllTaskConfigs | None = None
        """Configuration for tasks"""

        task_cls_str: str | None = None
        """(Optional) The name of the task class to be used"""

    def freeze_task_config(self, observation, task: BaseMujocoTask | None = None) -> str:
        sc = MlSpacesExpConfig.SavedEpisode()

        # NOTE(RMH): deep argument VERY IMPORTANT. Mutates config for future episodes otherwise
        sc.robot_config = self.robot_config.model_copy(deep=True)

        sc.robot_config.robot_cls = None
        sc.robot_config.robot_factory = None
        sc.robot_config.robot_view_factory = None

        sc.robot_config.init_qpos_noise_range = None
        sc.robot_config.init_qpos = observation[0]["qpos"]

        if task and self.camera_config:
            sc.camera_config = self.camera_config.model_copy(deep=True)
            for i, camera in enumerate(sc.camera_config.cameras):
                # Some cameras can contain random sampling, e.g. of positions
                # Read the camera's positions and convert them to fixed cameras
                if isinstance(camera, MjcfCameraConfig | RobotMountedCameraConfig):
                    cam = task.env.camera_manager.registry[camera.name]

                    # TODO(wilbert): uhmm, most of the attributes below are only defined in the
                    # RobotMountedCamera cls, not the base Camera cls. For now we're assumming
                    # that here we always get a RobotMountedCamera instance, otherwise it would
                    # crash anyway bc the parameters expected wouldn't be defined
                    assert isinstance(cam, RobotMountedCamera), (
                        "Something is wrong here, the 'cam' instance should be of class 'RobotMountedCamera'"
                    )

                    camera_quat: list[float] = (
                        cam.camera_quaternion.tolist()
                        if cam.camera_quaternion is not None
                        else [1.0, 0.0, 0.0, 0.0]
                    )
                    new_camera = RobotMountedCameraConfig(
                        name=cam.name,
                        reference_body_names=list(cam.reference_body_names),
                        camera_offset=list(cam.camera_offset),
                        lookat_offset=list(cam.lookat_offset),
                        camera_quaternion=camera_quat,
                        fov=cam.fov,
                    )
                    sc.camera_config.cameras[i] = new_camera

                elif isinstance(
                    camera, RandomizedExocentricCameraConfig | FixedExocentricCameraConfig
                ):
                    cam = task.env.camera_manager.registry[camera.name]
                    new_camera = FixedExocentricCameraConfig(
                        name=cam.name,
                        fov=cam.fov,
                        pos=list(cam.pos),
                        up=list(cam.up),
                        forward=list(cam.forward),
                    )
                    sc.camera_config.cameras[i] = new_camera
                else:
                    raise NotImplementedError(
                        f"Cannot freeze camera of type {type(camera).__name__}"
                    )

        if task:
            obj_poses = {}
            om = task.env.object_managers[task.env.current_batch_index]
            task_objects = om.get_mobile_objects()
            for task_object in task_objects:
                obj_poses[task_object.name] = pose_mat_to_7d(task_object.pose).tolist()
            task.config.task_config.object_poses = obj_poses

            sc.task_config = self.task_config.model_copy(deep=True)
            sc.task_config.task_cls = None
            if self.task_config.task_cls is not None:
                sc.task_cls_str = (
                    self.task_config.task_cls.__module__ + "." + self.task_config.task_cls.__name__
                )

        if sc.task_config and sc.task_config.robot_base_pose is None:
            raise RuntimeError("'robot_base_pose' attribute must be in the task_config")

        sc_bytes = pickle.dumps(sc)
        sc_b64 = base64.b64encode(sc_bytes).decode("utf-8")
        return sc_b64
