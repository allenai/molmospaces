"""Camera management for MolmoSpaces environments.

This module handles all camera-related functionality including camera registration,
pose updates, rendering, and setup from camera configurations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from molmo_spaces.utils.camera_utils import CameraNoiseModel, CameraResetCadence

if TYPE_CHECKING:
    from molmo_spaces.configs.camera_configs import (
        CameraSystemConfig,
        FixedExocentricCameraConfig,
        MjcfCameraConfig,
        RandomizedExocentricCameraConfig,
        RobotMountedCameraConfig,
    )
    from molmo_spaces.env.env import CPUMujocoEnv

log = logging.getLogger(__name__)


@dataclass
class MjcfCameraInfo:
    """What a Camera registered from an MjcfCameraConfig remembers about the
    MJCF camera it stands for: its id in the model, its un-noised mounting
    (the base a per-episode randomizer perturbs model.cam_* around), and, for
    a FisheyeMjcfCameraConfig, the co-located tile cameras the cubemap
    composite is rendered through. `fisheye` caches the FisheyeRenderer once
    CameraManager.fisheye_renderer has built it.
    """

    camera_id: int
    base_pos: NDArray
    base_quat: NDArray
    base_fovy: float
    config: MjcfCameraConfig
    tile_camera_ids: list[int] = field(default_factory=list)
    tile_base_pos: list[NDArray] = field(default_factory=list)
    tile_base_quat: list[NDArray] = field(default_factory=list)
    fisheye: object | None = None

    @property
    def is_fisheye(self) -> bool:
        return bool(self.tile_camera_ids)

    @property
    def rig_camera_ids(self) -> list[int]:
        """The MJCF camera plus its fisheye tiles: every camera that has to move
        together when the mounting is perturbed."""
        return [self.camera_id, *self.tile_camera_ids]

    @property
    def rig_base_pos(self) -> list[NDArray]:
        return [self.base_pos, *self.tile_base_pos]

    @property
    def rig_base_quat(self) -> list[NDArray]:
        return [self.base_quat, *self.tile_base_quat]


class Camera:
    """Base camera class with position and orientation."""

    # Set by CameraManager._setup_mjcf_camera for cameras defined in the MJCF.
    mjcf: MjcfCameraInfo | None = None

    # default fov set to 45, which is mujoco default
    def __init__(
        self,
        name: str,
        pos: NDArray[np.float32] | None = None,
        forward: NDArray[np.float32] | None = None,
        up: NDArray[np.float32] | None = None,
        fov: float = 45.0,
    ) -> None:
        self.name: str = name
        self.pos: NDArray[np.float32] = (
            pos if pos is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        self.forward: NDArray[np.float32] = (
            forward if forward is not None else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        self.up: NDArray[np.float32] = (
            up if up is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        self.fov: float = fov

    def update_pose(self, env: CPUMujocoEnv) -> bool:
        """Update camera pose. Returns True if pose changed, False otherwise."""
        return False  # by default cameras don't update

    def get_pose(self) -> NDArray[np.float32]:
        """
        return 4x4 pose
        """
        # Validate and normalize camera vectors
        forward_norm = np.linalg.norm(self.forward)
        up_norm = np.linalg.norm(self.up)

        if forward_norm < 1e-6 or up_norm < 1e-6:
            print(
                f"Warning: Camera '{self.camera_name}' has degenerate vectors (forward_norm={forward_norm}, up_norm={up_norm})"
            )
            return np.eye(4, 4, dtype=np.float32)

        forward = self.forward / forward_norm
        up = self.up / up_norm
        right = np.cross(forward, up)

        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            print(f"Warning: Camera '{self.self}' has collinear forward/up vectors")
            return np.eye(4, 4, dtype=np.float32)

        right = right / right_norm

        # Recompute orthogonal up to ensure proper orthogonal basis
        up = np.cross(right, forward)

        # Create cam2world matrix (standard camera convention)
        world2cam = np.eye(4)
        world2cam[:3, 0] = right  # X-axis (right)
        world2cam[:3, 1] = -up  # Y-axis (up)
        world2cam[:3, 2] = forward  # Z-axis - camera looks down negative Z
        world2cam[:3, 3] = self.pos  # Translation
        return world2cam


class RobotMountedCamera(Camera):
    """Generic robot-mounted camera that can attach to any body/joint with configurable offsets."""

    def __init__(
        self,
        name: str,
        reference_body_names: str | list[str],
        camera_offset: NDArray[np.float32] | list[float] | None = None,
        lookat_offset: NDArray[np.float32] | list[float] | None = None,
        up_axis: str = "z",
        camera_quaternion: NDArray[np.float32] | list[float] | None = None,
        camera_fov: float = 45,
    ) -> None:
        """
        Args:
            name: Camera name
            reference_body_names: Body name(s) to attach camera to. If list, tries each until one works.
            camera_offset: Camera position relative to reference body frame
            lookat_offset: Offset from reference body to look at (used when camera_quaternion is None)
            up_axis: Which local axis of reference frame is "up" ("x", "y", or "z") (used when camera_quaternion is None)
            camera_quaternion: Quaternion [w, x, y, z] relative to reference body frame. If provided, overrides lookat_offset and up_axis
        """
        # Store configuration for dynamic updates
        self.reference_body_names: list[str] = (
            [reference_body_names]
            if isinstance(reference_body_names, str)
            else reference_body_names
        )
        self.camera_offset: NDArray[np.float32] = (
            np.array(camera_offset, dtype=np.float32)
            if camera_offset is not None
            else np.array([0.10, 0.0, -0.15], dtype=np.float32)
        )
        self.lookat_offset: NDArray[np.float32] = (
            np.array(lookat_offset, dtype=np.float32)
            if lookat_offset is not None
            else np.array([0.0, 0.0, 0.08], dtype=np.float32)
        )
        self.up_axis: str = up_axis
        self.camera_quaternion: NDArray[np.float32] | None = (
            np.array(camera_quaternion, dtype=np.float32) if camera_quaternion is not None else None
        )

        # Initialize with default pose (will be updated on first update_pose call)
        super().__init__(name, fov=camera_fov)

        # Cache for avoiding unnecessary recalculations
        self._last_reference_pose: NDArray[np.float32] | None = None
        self._active_reference_body_name: str | None = None

    def _find_reference_body(self, env: CPUMujocoEnv) -> tuple[object | None, str | None]:
        """Find the first valid reference body from the list of candidates."""
        from molmo_spaces.env.data_views import create_mlspaces_body

        for body_name in self.reference_body_names:
            try:
                body = create_mlspaces_body(env.current_data, body_name)
                return body, body_name
            except (AttributeError, KeyError):
                continue

        return None, None

    def update_pose(self, env: CPUMujocoEnv) -> bool:
        """Update camera pose based on current reference body state. Returns True if pose changed."""

        # Find reference body
        reference_body, body_name = self._find_reference_body(env)
        if reference_body is None:
            log.warning(
                f"No valid reference body found for camera {self.name} {self.reference_body_names}"
            )
            return False  # Cannot find any reference body

        # Update active reference if it changed
        if self._active_reference_body_name != body_name:
            self._active_reference_body_name = body_name
            self._last_reference_pose = None  # Force update

        current_reference_pose = reference_body.pose.copy()

        # Check if reference pose has changed (with small tolerance for numerical precision)
        if self._last_reference_pose is not None:
            pose_diff = np.linalg.norm(current_reference_pose - self._last_reference_pose)
            if pose_diff < 1e-6:  # No significant change
                return False

        # Reference pose has changed, recalculate camera pose
        if self.camera_quaternion is not None:
            pos, forward, up = env.camera_manager.create_quaternion_camera_pose(
                env, self._active_reference_body_name, self.camera_offset, self.camera_quaternion
            )
        else:
            pos, forward, up = env.camera_manager.create_robot_mounted_camera_pose(
                env,
                self._active_reference_body_name,
                self.camera_offset,
                self.lookat_offset,
                self.up_axis,
            )

        # Update our pose
        self.pos = pos
        self.forward = forward
        self.up = up

        # Cache the reference pose
        self._last_reference_pose = current_reference_pose

        return True


class CameraRegistry:
    """Registry for camera objects with auto-updating support."""

    def __init__(self) -> None:
        self.cameras: dict[str, Camera] = {}

    def add_camera(self, camera: Camera) -> None:
        """Adds a camera object to the registry."""
        self.cameras[camera.name] = camera

    def add_static_camera(
        self,
        name: str,
        pos: NDArray[np.float32],
        forward: NDArray[np.float32],
        up: NDArray[np.float32],
    ) -> None:
        """Adds a static camera to the registry."""
        self.cameras[name] = Camera(name=name, pos=pos, forward=forward, up=up)

    def update_all_cameras(self, env: CPUMujocoEnv) -> list[str]:
        """Update all cameras that need updating. Returns list of cameras that changed."""
        updated_cameras: list[str] = []
        for camera in self.cameras.values():
            if camera.update_pose(env):
                updated_cameras.append(camera.name)
        return updated_cameras

    def __getitem__(self, key: str) -> Camera:
        try:
            return self.cameras[key]
        except KeyError as e:
            log.warning(f"For KeyError, camera options are options are: {self.cameras.keys()}")
            raise e

    def __setitem__(self, key: str, value: Camera) -> None:
        self.cameras[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.cameras

    def keys(self):
        return self.cameras.keys()

    def __iter__(self):
        return iter(self.cameras.values())


class CameraManager:
    """Manages all camera-related operations for an environment.

    This class encapsulates camera setup, registration, and rendering operations,
    keeping camera logic separate from core environment concerns.

    Note: This class does not store a reference to the environment to avoid
    circular references and enable pickling for multiprocessing. Instead,
    the environment is passed as a parameter to methods that need it.
    """

    def __init__(self) -> None:
        """Initialize the camera manager with an empty registry."""
        self.registry: CameraRegistry = CameraRegistry()

    def setup_cameras(
        self,
        env,
        camera_system_config: CameraSystemConfig,
        workspace_center=None,
        visibility_resolver: Callable[[str], list[str]] | None = None,
        deterministic_only: bool = False,
        apply_mjcf_noise: bool = True,
    ) -> None:
        """Set up all cameras from a CameraSystemConfig.

        This is the main entry point for camera setup. It processes each camera spec
        and delegates to the appropriate setup method based on camera type.

        Args:
            env: The environment instance (CPUMujocoEnv)
            camera_system_config: CameraSystemConfig instance with all camera specs
            workspace_center: Optional workspace center position (np.ndarray) for camera placement
            visibility_resolver: Optional callable(key: str) -> str that resolves special visibility keys
            apply_mjcf_noise: Apply the MJCF cameras' configured noise at setup.
                False registers them at their MJCF pose without drawing RNG,
                for an owner that re-noises them every episode (G1TaskSampler).
        """
        from molmo_spaces.configs.camera_configs import (
            FixedExocentricCameraConfig,
            MjcfCameraConfig,
            RandomizedExocentricCameraConfig,
            RobotMountedCameraConfig,
        )

        # Store workspace center and visibility resolver for use in camera setup methods
        self._workspace_center = workspace_center
        self._visibility_resolver = visibility_resolver

        log.info(f"[CAMERA SETUP] Setting up {len(camera_system_config.cameras)} cameras")

        # allow runtime errors to propagate here instead of catching them and logging
        for camera_spec in camera_system_config.cameras:
            self._check_reset_cadence_supported(camera_spec)
            if isinstance(camera_spec, MjcfCameraConfig):
                self._setup_mjcf_camera(env, camera_spec, apply_noise=apply_mjcf_noise)
            elif isinstance(camera_spec, RobotMountedCameraConfig):
                self._setup_robot_mounted_camera(env, camera_spec)
            elif isinstance(camera_spec, FixedExocentricCameraConfig):
                self._setup_fixed_exocentric_camera(env, camera_spec)
            elif isinstance(camera_spec, RandomizedExocentricCameraConfig):
                if deterministic_only:
                    log.info(
                        f"[CAMERA SETUP] skipping randomized exocentric camera '{camera_spec.name}' for pre-workspace setup"
                    )
                else:
                    self._setup_randomized_exocentric_camera(env, camera_spec)
            else:
                log.warning(
                    f"[CAMERA SETUP] Unknown camera spec type: {type(camera_spec).__name__}"
                )

        # Clean up
        self._workspace_center = None
        self._visibility_resolver = None

        log.info(f"[CAMERA SETUP] Successfully set up {len(self.registry.cameras)} cameras")

    def fisheye_renderer(self, env, camera_name: str, output_h: int, output_w: int):
        """The cubemap FisheyeRenderer for a FisheyeMjcfCameraConfig camera at
        this output size, built on first use from the config's own lens
        parameters (FisheyeMjcfCameraConfig.cubemap_renderer_kwargs) and cached
        on the camera; a different output size rebuilds it. Its K/D can be
        perturbed in place afterwards (set_intrinsics), which is why the cache
        is per camera rather than per call.
        """
        camera = self.registry[camera_name]
        info = camera.mjcf
        if info is None or not info.is_fisheye:
            raise ValueError(f"camera {camera_name!r} is not a fisheye MJCF camera")
        fisheye = info.fisheye
        if fisheye is None or fisheye.output_h != output_h or fisheye.output_w != output_w:
            from molmo_spaces.utils.fisheye_cubemap import FisheyeRenderer

            fisheye = FisheyeRenderer(
                env.current_model, **info.config.cubemap_renderer_kwargs(output_h, output_w)
            )
            info.fisheye = fisheye
        return fisheye

    @staticmethod
    def apply_mjcf_camera_noise(
        camera_pos: np.ndarray,
        camera_quat: np.ndarray,
        camera_fov: float,
        camera_config: MjcfCameraConfig,
        rng=np.random,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply position, orientation, and FOV noise to MJCF camera parameters.

        Args:
            camera_pos: Camera position (body-frame offset), modified in place.
            camera_quat: Camera quaternion [w,x,y,z] (body-frame), modified in place.
            camera_fov: Base FOV in degrees.
            camera_config: Config carrying noise ranges.
            rng: Random generator (np.random module or np.random.RandomState).

        Returns:
            Tuple of (noised_pos, noised_quat, noised_fov).
        """
        if camera_config.noise_model == CameraNoiseModel.BODY_FRAME_AXIS_ANGLE:
            return CameraManager._apply_body_frame_axis_angle_noise(
                camera_pos, camera_quat, camera_fov, camera_config, rng
            )

        if camera_config.fov_noise_degrees is not None:
            noise = rng.uniform(
                camera_config.fov_noise_degrees[0], camera_config.fov_noise_degrees[1]
            )
            camera_fov += noise
            log.debug(
                f"[CAMERA SETUP] Applied FOV noise to '{camera_config.name}': {noise} degrees"
            )

        if camera_config.pos_noise_range is not None:
            noise = rng.uniform(
                camera_config.pos_noise_range[0], camera_config.pos_noise_range[1], size=3
            )
            original_rot = R.from_quat(camera_quat, scalar_first=True)
            camera_pos = camera_pos + original_rot.apply(noise)
            log.debug(f"[CAMERA SETUP] Applied position noise to '{camera_config.name}': {noise}")

        if camera_config.orientation_noise_degrees is not None:
            noise_euler = rng.uniform(
                -np.array(camera_config.orientation_noise_degrees),
                np.array(camera_config.orientation_noise_degrees),
                size=3,
            )
            noise_rotation = R.from_euler("xyz", noise_euler, degrees=True)
            original_rotation = R.from_quat(camera_quat, scalar_first=True)
            noisy_rotation = original_rotation * noise_rotation
            camera_quat = noisy_rotation.as_quat(scalar_first=True)
            log.debug(
                f"[CAMERA SETUP] Applied orientation noise to '{camera_config.name}': {noise_euler} degrees"
            )

        return camera_pos, camera_quat, camera_fov

    @staticmethod
    def _check_reset_cadence_supported(camera_config) -> None:
        """Fail loudly on a reset_cadence this manager cannot honor: noise is
        applied once at registration, so "episode" would silently narrow the
        camera distribution to "setup". The G1 is exempt -- G1TaskSampler
        redraws its cameras itself every reset.

        TODO(max): make "episode" the default -- redraw on reset here, point
        the G1 at it, and both the exemption and the "setup" default go away.
        """
        # noise_model lives on MjcfCameraConfig only; the other camera types have
        # no G1 counterpart, so for them EPISODE is always unimplemented.
        if (
            camera_config.reset_cadence == CameraResetCadence.EPISODE
            and getattr(camera_config, "noise_model", None)
            != CameraNoiseModel.BODY_FRAME_AXIS_ANGLE
        ):
            raise NotImplementedError(
                f"camera {camera_config.name!r} asks for reset_cadence=EPISODE, which "
                "CameraManager does not implement -- it applies noise once at setup. Use "
                "SETUP, or drive the per-reset randomization outside this class the way "
                "fetchman's G1TaskSampler does for the G1 cameras."
            )

    @staticmethod
    def _apply_body_frame_axis_angle_noise(
        camera_pos: np.ndarray,
        camera_quat: np.ndarray,
        camera_fov: float,
        camera_config: MjcfCameraConfig,
        rng,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """g1_molmo's `_perturb_camera`: position offset in the parent body
        frame, rotation as one random axis-angle composed parent-side, draws
        in the order position -> axis -> angle -> FOV. Opt-in per camera; it
        differs from "camera_local_euler" in each of those respects.
        """
        if camera_config.pos_noise_range is not None:
            lo, hi = camera_config.pos_noise_range
            camera_pos = camera_pos + rng.uniform(lo, hi, 3)

        if camera_config.orientation_noise_degrees is not None:
            axis = rng.uniform(-1, 1, 3)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
            limit = np.radians(camera_config.orientation_noise_degrees)
            angle = float(rng.uniform(-limit, limit))
            delta = R.from_rotvec(axis * angle)
            camera_quat = (delta * R.from_quat(camera_quat, scalar_first=True)).as_quat(
                scalar_first=True
            )

        if camera_config.fov_noise_degrees is not None:
            camera_fov += float(
                rng.uniform(camera_config.fov_noise_degrees[0], camera_config.fov_noise_degrees[1])
            )

        return camera_pos, camera_quat, camera_fov

    def _setup_mjcf_camera(
        self, env, camera_config: MjcfCameraConfig, apply_noise: bool = True
    ) -> None:
        """Register a camera defined in the MJCF as a tracking
        RobotMountedCamera, and record its MJCF facts on it (MjcfCameraInfo)."""
        # Build full camera name with namespace if provided
        if camera_config.robot_namespace:
            full_mjcf_name = f"{camera_config.robot_namespace}{camera_config.mjcf_name}"
        else:
            full_mjcf_name = camera_config.mjcf_name

        # Find camera in model
        model = env.current_model
        camera_id = None
        camera_names = []
        for i in range(model.ncam):
            camera_names.append(model.cam(i).name)
            if model.cam(i).name == full_mjcf_name:
                camera_id = i
                break

        if camera_id is None:
            log.warning(
                f"[CAMERA SETUP] MJCF camera '{full_mjcf_name}' not found in model, options are {camera_names}"
            )
            return

        # Extract camera parameters from MJCF
        # NOTE: if there's an fov in the camera_config, use that instead of the MJCF fov
        camera_obj = model.cam(camera_id)
        camera_pos = camera_obj.pos.copy()
        camera_quat = camera_obj.quat.copy()
        camera_fov = (
            camera_config.fov if camera_config.fov is not None else camera_obj.fovy[0]
        )  # this will raise an error if the fov is not set - desired behavior

        # BODY_FRAME_AXIS_ANGLE also specifies WHICH stream it draws from -- the
        # env's seeded one, so cameras reproduce from the episode seed as
        # g1_molmo's do. The default model still draws from global np.random;
        # changing that would reshuffle every other robot's cameras.
        noise_rng = np.random
        if camera_config.noise_model == CameraNoiseModel.BODY_FRAME_AXIS_ANGLE:
            noise_rng = getattr(env, "np_random", np.random)

        base_pos, base_quat, base_fov = camera_pos.copy(), camera_quat.copy(), float(camera_fov)
        if apply_noise:
            camera_pos, camera_quat, camera_fov = self.apply_mjcf_camera_noise(
                camera_pos, camera_quat, camera_fov, camera_config, rng=noise_rng
            )

        # Set up as robot-mounted camera (will track the body it's attached to)
        camera_obj_bodyid = camera_obj.bodyid.item()
        self.add_robot_mounted_camera_with_quaternion(
            env,
            camera_name=camera_config.name,
            reference_body_names=[env.current_model.body(camera_obj_bodyid).name],
            camera_offset=camera_pos,
            camera_quaternion=camera_quat,
            camera_fov=camera_fov,
        )

        # The MJCF side of this camera, for renderers that go through the
        # model's own camera (mujoco.Renderer.update_scene(camera=id)) and for
        # per-episode randomizers that perturb model.cam_* around the base.
        camera = self.registry.cameras[camera_config.name]
        tile_ids: list[int] = []
        from molmo_spaces.configs.camera_configs import FisheyeMjcfCameraConfig

        if isinstance(camera_config, FisheyeMjcfCameraConfig):
            for tile_name in camera_config.tile_camera_names():
                tile_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, tile_name)
                if tile_id < 0:
                    log.warning(
                        f"[CAMERA SETUP] fisheye tile camera '{tile_name}' of "
                        f"'{camera_config.name}' not found in model"
                    )
                    continue
                tile_ids.append(int(tile_id))
        camera.mjcf = MjcfCameraInfo(
            camera_id=int(camera_id),
            base_pos=base_pos,
            base_quat=base_quat,
            base_fovy=base_fov,
            config=camera_config,
            tile_camera_ids=tile_ids,
            tile_base_pos=[model.cam_pos[i].copy() for i in tile_ids],
            tile_base_quat=[model.cam_quat[i].copy() for i in tile_ids],
        )

        # Store visibility constraints on the camera object for later use during robot placement
        if camera_config.visibility_constraints is not None:
            camera = self.registry.cameras[camera_config.name]
            camera.visibility_constraints = camera_config.visibility_constraints
            log.debug(
                f"[CAMERA SETUP] Stored visibility constraints for '{camera_config.name}': "
                f"{camera_config.visibility_constraints}"
            )

        log.info(
            f"[CAMERA SETUP] Set up MJCF camera '{camera_config.name}' (MJCF: {full_mjcf_name})"
        )

    def _setup_robot_mounted_camera(self, env, camera_config: RobotMountedCameraConfig) -> None:
        """Set up a dynamically robot-mounted camera."""
        # Apply position noise if configured
        camera_offset = np.array(camera_config.camera_offset, dtype=np.float32)
        if camera_config.pos_noise_range is not None:
            noise = np.random.uniform(
                camera_config.pos_noise_range[0], camera_config.pos_noise_range[1], size=3
            )
            camera_offset += noise

        # Apply lookat noise if configured and using lookat method
        lookat_offset = camera_config.lookat_offset
        if lookat_offset is not None and camera_config.lookat_noise_range is not None:
            lookat_offset = np.array(lookat_offset, dtype=np.float32)
            noise = np.random.uniform(
                camera_config.lookat_noise_range[0], camera_config.lookat_noise_range[1], size=3
            )
            lookat_offset += noise

        # Use quaternion or lookat method based on what's provided
        if camera_config.camera_quaternion is not None:
            camera_quaternion = np.array(camera_config.camera_quaternion, dtype=np.float32)

            # Apply orientation noise if configured
            if camera_config.orientation_noise_degrees is not None:
                noise_euler = np.random.uniform(
                    -camera_config.orientation_noise_degrees,
                    camera_config.orientation_noise_degrees,
                    size=3,
                )
                noise_rotation = R.from_euler("xyz", noise_euler, degrees=True)

                # Compose noise with original quaternion
                original_rotation = R.from_quat(camera_quaternion, scalar_first=True)
                noisy_rotation = noise_rotation * original_rotation
                camera_quaternion = noisy_rotation.as_quat(scalar_first=True)

                log.debug(
                    f"[CAMERA SETUP] Applied orientation noise to '{camera_config.name}': {noise_euler} degrees"
                )

            self.add_robot_mounted_camera_with_quaternion(
                env,
                camera_name=camera_config.name,
                reference_body_names=camera_config.reference_body_names,
                camera_offset=camera_offset,
                camera_quaternion=camera_quaternion,
                camera_fov=camera_config.fov,
            )
        else:
            self.add_robot_mounted_camera(
                env,
                camera_name=camera_config.name,
                reference_body_names=camera_config.reference_body_names,
                camera_offset=camera_offset,
                lookat_offset=lookat_offset,
                up_axis=camera_config.up_axis,
            )

        # Store visibility constraints on the camera object for later use during robot placement
        if camera_config.visibility_constraints is not None:
            camera = self.registry.cameras[camera_config.name]
            camera.visibility_constraints = camera_config.visibility_constraints
            log.debug(
                f"[CAMERA SETUP] Stored visibility constraints for '{camera_config.name}': "
                f"{camera_config.visibility_constraints}"
            )

        log.info(f"[CAMERA SETUP] Set up robot-mounted camera '{camera_config.name}'")

    def _setup_fixed_exocentric_camera(
        self, env, camera_config: FixedExocentricCameraConfig
    ) -> None:
        """Set up a fixed exocentric (external) camera."""
        pos = np.array(camera_config.pos, dtype=np.float32)
        forward = np.array(camera_config.forward, dtype=np.float32)
        up = np.array(camera_config.up, dtype=np.float32)

        # Apply position noise if camera_configified
        if camera_config.pos_noise_range is not None:
            noise = np.random.uniform(
                camera_config.pos_noise_range[0], camera_config.pos_noise_range[1], size=3
            )
            y_ax = -up
            z_ax = forward
            x_ax = np.cross(y_ax, z_ax)
            rotmat = np.column_stack([x_ax, y_ax, z_ax])
            pos += rotmat @ noise
            log.debug(f"[CAMERA SETUP] Applied position noise to '{camera_config.name}': {noise}")

        # Apply orientation noise if camera_configified
        if camera_config.orientation_noise_degrees is not None:
            # TODO: Implement orientation noise using scipy.spatial.transform.Rotation
            log.warning(
                f"[CAMERA SETUP] Orientation noise not yet implemented for '{camera_config.name}'"
            )

        self.add_camera(camera_config.name, pos, forward, up, camera_config.fov)

        # Store visibility constraints on the camera object for later use during robot placement
        if camera_config.visibility_constraints is not None:
            camera = self.registry.cameras[camera_config.name]
            camera.visibility_constraints = camera_config.visibility_constraints
            log.debug(
                f"[CAMERA SETUP] Stored visibility constraints for '{camera_config.name}': "
                f"{camera_config.visibility_constraints}"
            )

        log.info(f"[CAMERA SETUP] Set up fixed exocentric camera '{camera_config.name}'")

    def _setup_randomized_exocentric_camera(
        self, env, camera_config: RandomizedExocentricCameraConfig
    ) -> None:
        """Set up a randomized exocentric (external) camera.

        RandomizedExocentricCameraConfig is designed to always get workspace_center from
        the task sampler and always look at the workspace center (with optional noise).
        """

        # Randomized camera - need to sample a position
        camera_fov = camera_config.fov
        if camera_config.fov_range is not None:
            camera_fov = np.random.uniform(camera_config.fov_range[0], camera_config.fov_range[1])

        # Determine workspace center - use value from task sampler or default
        workspace_center = self._workspace_center
        if workspace_center is not None:
            log.debug(
                f"[CAMERA SETUP] Using workspace_center for '{camera_config.name}': "
                f"({workspace_center[0]:.3f}, {workspace_center[1]:.3f}, {workspace_center[2]:.3f})"
            )
        else:
            # Default: use robot gripper position or world origin
            gripper_positions = env.get_robot_gripper_positions(0)
            if gripper_positions:
                positions = list(gripper_positions.values())
                workspace_center = np.mean(positions, axis=0)
            else:
                workspace_center = np.array([0.0, 0.0, 0.8])  # Default table height

        # Sample camera position
        best_pos, best_forward, best_up = None, None, None
        best_visibility_score = -1.0

        for attempt in range(camera_config.max_placement_attempts):
            # Sample spherical coordinates
            distance = np.random.uniform(
                camera_config.distance_range[0], camera_config.distance_range[1]
            )
            azimuth = np.random.uniform(
                camera_config.azimuth_range[0], camera_config.azimuth_range[1]
            )
            height = np.random.uniform(camera_config.height_range[0], camera_config.height_range[1])

            # Convert to Cartesian
            camera_pos = workspace_center.copy()
            camera_pos[0] += distance * np.cos(azimuth)
            camera_pos[1] += distance * np.sin(azimuth)
            camera_pos[2] += height

            # Always look at workspace center (design assumption for this camera type)
            lookat_target = workspace_center.copy()

            # Add lookat noise if configured
            if camera_config.lookat_noise_range is not None:
                noise = np.random.uniform(
                    camera_config.lookat_noise_range[0], camera_config.lookat_noise_range[1], size=3
                )
                lookat_target += noise

            # Calculate camera orientation
            pos, forward, up = self.create_lookat_pose_world(
                env, camera_pos, np.zeros(3), lookat_target=lookat_target
            )

            # Check visibility constraints if specified
            if camera_config.visibility_constraints:
                # Resolve special visibility keys (e.g., __task_objects__, __gripper__)
                resolved_constraints = {}
                for key, threshold in camera_config.visibility_constraints.items():
                    if key.startswith("__") and key.endswith("__"):
                        # Special key - resolve via callback
                        if self._visibility_resolver is not None:
                            resolved_keys = self._visibility_resolver(key)
                            if resolved_keys:
                                for resolved_key in resolved_keys:
                                    resolved_constraints[resolved_key] = threshold
                            else:
                                log.warning(
                                    f"[CAMERA SETUP] Could not resolve visibility key '{key}' for camera '{camera_config.name}'"
                                )
                        else:
                            log.warning(
                                f"[CAMERA SETUP] No visibility resolver provided for key '{key}' in camera '{camera_config.name}'"
                            )
                    else:
                        # Regular body name
                        resolved_constraints[key] = threshold

                if not resolved_constraints:
                    # No valid constraints, place camera without visibility check
                    log.debug(
                        f"[CAMERA SETUP] No valid visibility constraints for '{camera_config.name}', placing camera without visibility check"
                    )
                    self.add_camera(camera_config.name, pos, forward, up, camera_fov)
                    log.info(
                        f"[CAMERA SETUP] Set up randomized exocentric camera '{camera_config.name}' (no visibility constraints)"
                    )
                    return

                # Temporarily add camera to check visibility
                temp_camera = Camera(f"_temp_{camera_config.name}", pos, forward, up, camera_fov)
                self.registry.add_camera(temp_camera)

                try:
                    visibility_results = env.check_visibility(
                        f"_temp_{camera_config.name}", *resolved_constraints.keys()
                    )
                    if isinstance(visibility_results, dict):
                        visibility_scores = [
                            visibility_results[obj] for obj in resolved_constraints
                        ]
                    else:
                        visibility_scores = [visibility_results]

                    # Check if all constraints are satisfied
                    constraints_met = all(
                        visibility_results.get(obj, 0.0) >= threshold
                        if isinstance(visibility_results, dict)
                        else visibility_results >= threshold
                        for obj, threshold in resolved_constraints.items()
                    )

                    # Track best attempt
                    avg_visibility = np.mean(visibility_scores)
                    if avg_visibility > best_visibility_score:
                        best_visibility_score = avg_visibility
                        best_pos, best_forward, best_up = pos, forward, up

                    # Remove temporary camera
                    del self.registry.cameras[f"_temp_{camera_config.name}"]

                    if constraints_met:
                        # Found a valid placement
                        self.add_camera(camera_config.name, pos, forward, up, camera_fov)
                        log.info(
                            f"[CAMERA SETUP] Set up randomized exocentric camera '{camera_config.name}' "
                            f"(attempt {attempt + 1}, visibility: {avg_visibility:.5f})"
                        )
                        return

                except Exception as e:
                    # Clean up temp camera on error
                    if f"_temp_{camera_config.name}" in self.registry.cameras:
                        del self.registry.cameras[f"_temp_{camera_config.name}"]
                    log.debug(f"[CAMERA SETUP] Visibility check failed: {e}")
            else:
                # No visibility constraints - use first sample
                self.add_camera(camera_config.name, pos, forward, up, camera_fov)
                log.info(
                    f"[CAMERA SETUP] Set up randomized exocentric camera '{camera_config.name}' (no constraints)"
                )
                return

        # Failed to meet constraints - use best attempt if allowed
        if camera_config.allow_relaxed_constraints and best_pos is not None:
            self.add_camera(camera_config.name, best_pos, best_forward, best_up, camera_fov)
            log.warning(
                f"[CAMERA SETUP] Set up exocentric camera '{camera_config.name}' with relaxed constraints "
                f"(best visibility: {best_visibility_score:.5f})"
            )
        else:
            raise RuntimeError(
                f"[CAMERA SETUP] Failed to place exocentric camera '{camera_config.name}' "
                f"within visibility constraints after {camera_config.max_placement_attempts} attempts"
            )

    def add_camera(
        self,
        camera_name: str,
        pos: NDArray[np.float32],
        forward: NDArray[np.float32],
        up: NDArray[np.float32],
        fov: float = 45.0,
    ) -> None:
        """Adds or updates a static camera in the registry with its world pose."""
        self.registry.add_camera(Camera(camera_name, pos, forward, up, fov))

    def add_robot_mounted_camera(
        self,
        env,
        camera_name: str,
        reference_body_names: str | list[str],
        camera_offset: NDArray[np.float32] | None = None,
        lookat_offset: NDArray[np.float32] | None = None,
        up_axis: str = "z",
    ) -> None:
        """
        Add a robot-mounted camera that follows a specific body/joint.

        Args:
            env: The environment instance (CPUMujocoEnv)
            camera_name: Name for the camera
            reference_body_names: Body name(s) to attach camera to. If list, tries each until one works.
            camera_offset: Camera position relative to reference body frame
            lookat_offset: Offset from reference body to look at
            up_axis: Which local axis of reference frame is "up" ("x", "y", or "z")
        """
        robot_camera = RobotMountedCamera(
            camera_name, reference_body_names, camera_offset, lookat_offset, up_axis
        )
        # Initialize pose
        robot_camera.update_pose(env)
        self.registry.add_camera(robot_camera)

    def add_robot_mounted_camera_with_quaternion(
        self,
        env,
        camera_name: str,
        reference_body_names: str | list[str],
        camera_offset: NDArray[np.float32] | None = None,
        camera_quaternion: NDArray[np.float32] | None = None,
        camera_fov: float = 45,
    ) -> None:
        """
        Add a robot-mounted camera that follows a specific body/joint using quaternion orientation.

        Args:
            env: The environment instance (CPUMujocoEnv)
            camera_name: Name for the camera
            reference_body_names: Body name(s) to attach camera to. If list, tries each until one works.
            camera_offset: Camera position relative to reference body frame
            camera_quaternion: Quaternion [w, x, y, z] relative to reference body frame
        """
        robot_camera = RobotMountedCamera(
            camera_name,
            reference_body_names,
            camera_offset=camera_offset,
            camera_quaternion=camera_quaternion,
            camera_fov=camera_fov,
        )
        # Initialize pose
        robot_camera.update_pose(env)
        self.registry.add_camera(robot_camera)

    # ========================================================================
    # Camera Pose Calculation Methods
    # ========================================================================

    def create_lookat_pose_world(
        self,
        env,
        camera_pos_world: np.ndarray,
        rpy: np.ndarray,
        lookat_target: np.ndarray | None = None,
        lookat_body_name: str | None = None,
        camera_up: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera position and orientation vectors using enhanced lookat approach.
        Functionally equivalent to create_lookat_pose, but without use of a reference body.

        Args:
            env: The environment instance (CPUMujocoEnv)
            camera_pos_world: Camera position in world coordinates
            rpy: Roll, pitch, yaw (currently unused but kept for compatibility)
            lookat_target: Optional 3D point (np.array) to look at in world coordinates
            lookat_body_name: Optional body name to look at (used if lookat_target is None)
            camera_up: Optional desired up direction for camera (np.array in world coordinates)
                      If None, uses world Z-up [0, 0, 1]

        Returns:
            Tuple of (camera_pos_world, forward_vector, up_vector), each of shape (3,)
        """
        from molmo_spaces.utils.pose import compute_lookat_forward_up

        if lookat_target is None:
            if lookat_body_name is not None:
                from molmo_spaces.env.data_views import create_mlspaces_body

                lookat_target = create_mlspaces_body(env.current_data, lookat_body_name).position
            else:
                raise ValueError("Either lookat_target or lookat_body_name must be provided")

        forward, up = compute_lookat_forward_up(camera_pos_world, lookat_target, camera_up)
        return camera_pos_world, forward, up

    def create_lookat_pose(
        self,
        env,
        camera_relative_pos: np.ndarray,
        rpy: np.ndarray,
        reference_body_name: str,
        lookat_target: np.ndarray = None,
        lookat_body_name: str = None,
        camera_up: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera position and orientation vectors using enhanced lookat approach.

        Args:
            env: The environment instance (CPUMujocoEnv)
            camera_relative_pos: Camera position relative to reference body
            rpy: Roll, pitch, yaw (currently unused but kept for compatibility)
            reference_body_name: Body name for camera positioning reference
            lookat_target: Optional 3D point (np.array) to look at in world coordinates
            lookat_body_name: Optional body name to look at (used if lookat_target is None)
            camera_up: Optional desired up direction for camera (np.array in world coordinates)
                      If None, uses world Z-up [0, 0, 1]
        """
        from molmo_spaces.env.data_views import create_mlspaces_body

        data = env.current_data

        coordinate_reference_body = create_mlspaces_body(data, reference_body_name)
        camera_pos_world = (
            coordinate_reference_body.pose[:3, :3] @ camera_relative_pos
            + coordinate_reference_body.pose[:3, 3]
        )

        # Determine what to look at
        if lookat_target is not None:
            # Look at the provided 3D point
            lookat_pos_world = lookat_target
        elif lookat_body_name is not None:
            # Look at the specified body
            lookat_reference_body = create_mlspaces_body(data, lookat_body_name)
            lookat_pos_world = lookat_reference_body.position
        else:
            raise ValueError("Either lookat_target or lookat_body_name must be provided")

        # Calculate forward direction (from camera to target)
        forward = lookat_pos_world - camera_pos_world
        forward = forward / np.linalg.norm(forward)

        # Calculate right and up vectors using the desired up direction
        if camera_up is None:
            camera_up = np.array([0.0, 0.0, 1.0])  # Default world Z-up

        # Calculate right vector (perpendicular to both forward and desired up)
        right = np.cross(forward, camera_up)
        right_norm = np.linalg.norm(right)

        # Handle case where forward is parallel to desired up
        if right_norm < 1e-6:
            # Use a fallback reference direction
            fallback_ref = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(forward, fallback_ref)) > 0.9:
                fallback_ref = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, fallback_ref)
            right = right / np.linalg.norm(right)
        else:
            right = right / right_norm

        # Calculate actual up vector (perpendicular to forward and right)
        up = np.cross(right, forward)

        return camera_pos_world, forward, up

    def create_robot_mounted_camera_pose(
        self,
        env,
        reference_body_name: str,
        camera_offset: np.ndarray,
        lookat_offset: np.ndarray = None,
        up_axis: str = "z",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create generic robot-mounted camera pose using the existing create_lookat_pose method.
        Camera positioned relative to reference body, looking at a specific point offset from the body,
        with camera up aligned with the specified local axis.

        Args:
            env: The environment instance (CPUMujocoEnv)
            reference_body_name: Name of the body to attach camera to
            camera_offset: Camera position relative to reference body frame
            lookat_offset: Offset from reference body to look at (default: 8cm forward along local Z axis)
            up_axis: Which local axis of reference frame is "up" ("x", "y", or "z")
        """
        from molmo_spaces.env.data_views import create_mlspaces_body

        # Default lookat offset: look forward along reference body Z axis
        if lookat_offset is None:
            lookat_offset = np.array([0.0, 0.0, 0.08])  # 8cm forward along local Z axis

        # Get reference body for transformations
        reference_body = create_mlspaces_body(env.current_data, reference_body_name)

        # Calculate the target point in world coordinates
        lookat_target_world = (
            reference_body.pose[:3, :3] @ lookat_offset + reference_body.pose[:3, 3]
        )

        # Get the reference body's up direction in world coordinates
        local_up_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        if up_axis not in local_up_map:
            raise ValueError(f"up_axis must be 'x', 'y', or 'z', got {up_axis}")

        local_up = local_up_map[up_axis]
        reference_up_world = reference_body.pose[:3, :3] @ local_up

        # Use create_lookat_pose with the 3D target point and reference body's up direction
        return self.create_lookat_pose(
            env,
            camera_offset,
            np.zeros(3),
            reference_body_name,
            lookat_target=lookat_target_world,
            camera_up=reference_up_world,
        )

    def create_quaternion_camera_pose(
        self,
        env,
        reference_body_name: str,
        camera_offset: np.ndarray,
        camera_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create camera pose using quaternion-based orientation relative to reference body.

        Args:
            env: The environment instance (CPUMujocoEnv)
            reference_body_name: Name of the body to attach camera to
            camera_offset: Camera position relative to reference body frame
            camera_quaternion: Quaternion [w, x, y, z] relative to reference body frame

        Returns:
            Tuple of (camera_pos_world, forward_vector, up_vector)
        """
        from scipy.spatial.transform import Rotation as R

        from molmo_spaces.env.data_views import create_mlspaces_body

        # Get reference body for transformations
        reference_body = create_mlspaces_body(env.current_data, reference_body_name)

        # Calculate camera position in world coordinates
        camera_pos_world = reference_body.pose[:3, :3] @ camera_offset + reference_body.pose[:3, 3]

        # Calculate camera orientation in world coordinates
        norm_quat = np.linalg.norm(camera_quaternion)
        if norm_quat > 0:
            camera_quaternion = camera_quaternion / norm_quat

        camera_rotation_matrix = R.from_quat(camera_quaternion, scalar_first=True).as_matrix()
        world_rotation = reference_body.pose[:3, :3] @ camera_rotation_matrix

        # Extract forward and up vectors from the rotation matrix
        # Camera convention: forward is negative Z-axis, up is Y-axis
        forward = -world_rotation[:, 2]  # Negative Z-axis
        up = world_rotation[:, 1]  # Y-axis

        return camera_pos_world, forward, up
