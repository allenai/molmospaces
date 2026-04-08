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
        camera_name: str = "camera",
        uuid: str | None = None,
        img_resolution: tuple[int, int] = (480, 480),
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


# # Legacy sensors from other project (keeping for reference)
# class AgentsCameraParametersSensor(Sensor):
#     def __init__(
#         self,
#         uuid: str = "agent_camera_params",
#         str_max_len: Union[str, int] = 2000,
#     ) -> None:
#         assert isinstance(str_max_len, int)
#         self.str_max_len = str_max_len
#         observation_space = self._get_observation_space()
#         super().__init__(uuid=uuid, observation_space=observation_space)

#     def _get_observation_space(self) -> gyms.MultiDiscrete:
#         return gyms.Discrete(self.str_max_len)

#     def get_observation(self, env, task, *args, **kwargs) -> np.ndarray:
#         # Legacy implementation - would need adaptation for molmo-spaces
#         agent_parameter_sensors = {}
#         for which_cam in ["front", "left", "right", "down"]:
#             params = round_floats_in_dict(
#                 task.controller.camera_registry[which_cam]["camera_parameters"]
#             )
#             if params is not None and "camera_intrinsic" in params:
#                 del params[
#                     "camera_intrinsic"
#                 ]  # alternately, make it json-friendly. this seems fine though
#             agent_parameter_sensors[which_cam] = params
#         param_string = json.dumps(agent_parameter_sensors)
#         # Convert string to bytes array for gym compatibility
#         byte_array = np.zeros(self.str_max_len, dtype=np.uint8)
#         encoded = param_string.encode('utf-8')[:self.str_max_len]
#         byte_array[:len(encoded)] = list(encoded)
#         return byte_array


# class RawRGBCameraSensor(Sensor):
#     def __init__(self, uuid: str, height: int, width: int, which_camera: str):
#         self.height = height
#         self.width = width
#         self.which_camera = which_camera

#         observation_space = gyms.Box(
#             low=0, high=255,
#             shape=(height, width, 3),
#             dtype=np.uint8
#         )
#         super().__init__(uuid=uuid, observation_space=observation_space)

#     def get_observation(self, env, task, *args, **kwargs) -> Any:
#         # Legacy implementation - would need adaptation for molmo-spaces
#         frame = env.camera_registry[self.which_camera]["rgb"].copy()
#         if frame.shape[0] != self.height or frame.shape[1] != self.width:
#             import platform
#             if platform.system() != "Darwin":
#                 raise NotImplementedError(
#                     "Resizing the raw frames is a temp hack to get the warped and raw frames at "
#                     "the same time for visual comparison. If you are actually generating data, "
#                     "do not just bypass this check, fix get_core_sensors to actually be "
#                     "what you want."
#                 )
#             import cv2
#             frame = cv2.resize(frame, (self.width, self.height))
#         return frame
