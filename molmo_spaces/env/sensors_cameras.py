import gymnasium.spaces as gyms
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor


class CameraSensor(Sensor):
    """Sensor for RGB camera images from MuJoCo."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"camera_{camera_name}"

        # Define observation space for RGB images
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get camera image from environment rendering."""

        # Use camera-specific frame access for multi-camera support
        # if hasattr(env, 'render_rgb_frame') and callable(env.render_rgb_frame):
        frame = env.render_rgb_frame(self.camera_name)

        if frame is not None:
            return frame

        # Return black image if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 3), dtype=np.uint8)


class DepthSensor(Sensor):
    """Sensor for depth images from MuJoCo.

    Returns raw metric depth in meters as float32. Encoding to RGB for video storage
    happens at save time. See molmo_spaces.utils.depth_utils for encoding/decoding functions.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"depth_{camera_name}"

        # Define observation space for raw depth (float32 in meters)
        width, height = img_resolution
        observation_space = gyms.Box(low=0.0, high=10.0, shape=(height, width), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get depth image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "render_depth_frame") and callable(env.render_depth_frame):
            frame = env.render_depth_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "depth_frame") and env.depth_frame is not None:
            return env.depth_frame

        # Return zero depth if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width), dtype=np.float32)


class SegmentationSensor(Sensor):
    """Sensor for segmentation images from MuJoCo, outputs video-compatible arrays."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"segmentation_{camera_name}"

        # Define observation space for uint8 images with channel dimension
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get segmentation image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "segmentation_frame") and callable(env.segmentation_frame):
            frame = env.segmentation_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "segmentation_frame") and env.segmentation_frame is not None:
            return env.segmentation_frame

        # Return zero segmentation if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 1), dtype=np.uint8)


class CameraParameterSensor(Sensor):
    """Sensor for camera parameters (intrinsics and extrinsics)."""

    def __init__(
        self,
        camera_name: str,
        img_resolution: tuple[int, int],
        uuid: str | None = None,
    ) -> None:
        self.img_resolution = img_resolution
        self.camera_name = camera_name

        if uuid is None:
            uuid = f"camera_params_{camera_name}"

        observation_space = gyms.Dict(
            {
                "extrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32),
                "cam2world_gl": gyms.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32),
                "intrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32),
            }
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        """Get camera parameters for a specific environment."""
        camera = env.camera_manager.registry[self.camera_name]
        world2cam = camera.get_pose()
        # Create extrinsic_cv (Computer Vision convention - world2cam)
        extrinsic_cv = np.linalg.inv(world2cam)[:3, :]  # 3x4 matrix
        cam2world_gl = world2cam

        width, height = self.img_resolution
        fovy_degrees = camera.fov

        # Convert field of view to focal length
        focal_length = (height / 2.0) / np.tan(np.radians(fovy_degrees / 2.0))

        # Create intrinsic matrix (assuming square pixels and centered principal point)
        intrinsic_cv = np.array(
            [[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )

        # Ensure consistent structure and ordering
        data = {
            "cam2world_gl": cam2world_gl.tolist(),
            "extrinsic_cv": extrinsic_cv.tolist(),
            "intrinsic_cv": intrinsic_cv.tolist(),
        }
        return data


class ObjectPointInCameraSensor(Sensor):
    """Analytic (no-render) pixel projection of a task object's 3D position
    into each configured camera view -- normalized (u, v) in [0, 1], or
    (-1, -1) if the object is behind the camera. Matches g1_molmo's own
    target_point_in_head() (~/code/g1_molmo/molmospaces/env.py): a pure
    pinhole projection using the camera's known intrinsics/extrinsics, not a
    render. Reuses the exact extrinsic_cv/intrinsic_cv math
    CameraParameterSensor already computes analytically (env.camera_manager
    poses, not a rendered frame).

    Contrast with ObjectImagePointsSensor, which calls
    env.get_segmentation_mask_of_object() -- a real render -- every time
    it's polled; that's the right tool when you need the object's actual
    on-screen *extent* (its silhouette), but this sensor is for when only a
    single representative point is needed and a render's cost isn't
    justified (e.g. an oracle/scripted policy that never looks at pixels at
    all, only records them for a downstream consumer -- see PickG1Task).
    """

    def __init__(
        self,
        exp_config,
        object_name_attr: str = "pickup_obj_name",
        camera_names: list[str] | None = None,
        uuid: str = "object_point_in_camera",
    ) -> None:
        self.object_name_attr = object_name_attr
        all_camera_specs = {c.name: c for c in exp_config.camera_config.cameras}
        self.camera_names = (
            list(all_camera_specs.keys())
            if camera_names is None
            else [n for n in camera_names if n in all_camera_specs]
        )
        self.img_resolution = exp_config.camera_config.img_resolution
        observation_space = gyms.Dict(
            {
                name: gyms.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                for name in self.camera_names
            }
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(
        self, env, task, batch_index: int = 0, *args, **kwargs
    ) -> dict[str, np.ndarray]:
        from molmo_spaces.env.data_views import MlSpacesObject

        result = {name: np.array([-1.0, -1.0], dtype=np.float32) for name in self.camera_names}
        object_name = getattr(task.config.task_config, self.object_name_attr, None)
        if not object_name:
            return result

        data = env.mj_datas[batch_index]
        obj = MlSpacesObject(data=data, object_name=object_name)
        world_point = np.append(obj.position.astype(np.float64), 1.0)
        width, height = self.img_resolution

        for name in self.camera_names:
            camera = env.camera_manager.registry[name]
            cam2world_gl = camera.get_pose()
            focal = (height / 2.0) / np.tan(np.radians(camera.fov / 2.0))
            p_cam = np.linalg.inv(cam2world_gl) @ world_point
            xc, yc, zc = p_cam[:3]
            if zc <= 1e-6:
                continue  # behind the camera, not imageable
            u = width / 2.0 + focal * xc / zc
            v = height / 2.0 + focal * yc / zc
            result[name] = np.array([u / width, v / height], dtype=np.float32)
        return result
