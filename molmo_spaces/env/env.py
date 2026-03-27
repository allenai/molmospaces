import gc
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.camera_manager import CameraManager
from molmo_spaces.env.data_views import (
    Door,
    MlSpacesArticulationObject,
    MlSpacesObject,
    create_mlspaces_body,
)
from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer
from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.utils.rendering_utils import get_geom_seg_mask
from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap, sample_around_point
from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class BaseMujocoEnv(ABC):
    object_managers: list["ObjectManager"]

    def __init__(self, exp_config: "MlSpacesExpConfig", mj_model: MjModel = None) -> None:
        self._mj_model = mj_model
        self._current_batch_index = 0

        self.config = exp_config

        # Rendering state
        # TODO(anyone): can we remove these? too simple, we're on camera manager now.
        self._renderer = None
        self._rgb_frame = None
        self._depth_frame = None
        self._segmentation_frame = None
        self._camera_name = "camera"

    def is_loaded(self) -> bool:
        """Check if a scene is currently loaded."""
        return self._mj_model is not None

    @property
    def mj_model(self) -> MjModel:
        if not self.is_loaded():
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        return self._mj_model

    @property
    @abstractmethod
    def mj_datas(self) -> Sequence[MjData]:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_batch(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def robots(self) -> Sequence[Robot]:
        raise NotImplementedError

    @property
    def current_batch_index(self) -> int:
        """Current batch index for accessing data and robots."""
        return self._current_batch_index

    @current_batch_index.setter
    def current_batch_index(self, idx: int) -> None:
        """Set the current batch index."""
        if idx < 0 or idx >= self.n_batch:
            raise ValueError(f"Batch index {idx} out of range [0, {self.n_batch - 1}]")
        self._current_batch_index = idx

    @property
    def current_data(self) -> MjData:
        """Current MjData instance based on current_batch_index."""
        if not self.is_loaded():
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        return self.mj_datas[self.current_batch_index]

    @property
    def current_model(self) -> MjModel:
        """Current MjModel instance (always the same across batches)."""
        return self.mj_model

    @property
    def current_model_path(self) -> str:
        """Current string xml path instance (always the same across batches)."""
        return self._mj_base_scene_path

    @property
    def current_scene_metadata(self) -> dict:
        """Current scene metadata instance (always the same across batches)."""
        return self._scene_metadata

    @property
    def current_robot(self) -> Robot:
        """Current robot instance based on current_batch_index."""
        if not self.is_loaded():
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        return self.robots[self.current_batch_index]

    @property
    def rgb_frame(self) -> np.ndarray:
        """Get the latest RGB frame."""
        return self._rgb_frame

    @property
    def depth_frame(self) -> np.ndarray:
        """Get the latest depth frame."""
        return self._depth_frame

    @property
    def segmentation_frame(self) -> np.ndarray:
        """Get the latest segmentation frame."""
        return self._segmentation_frame

    @abstractmethod
    def reset(self, idxs: Collection[int] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        raise NotImplementedError


class CPUMujocoEnv(BaseMujocoEnv):
    def __init__(
        self,
        exp_config: "MlSpacesExpConfig",
        robot_factory: Callable[[MjData], Robot],
        mj_model: MjModel,
        mj_base_scene_path: str,
        parallelize: bool = True,
        use_filament: bool = False,
    ) -> None:
        super().__init__(exp_config, mj_model)

        # Store configuration for scene loading
        self._robot_factory = robot_factory
        self._n_batch = exp_config.task_sampler_config.task_batch_size
        self._parallelize = parallelize

        # Initialize empty - will be populated when scene is loaded
        self._mj_datas = None
        self._robots = None
        self._executor = None
        self._mj_base_scene_path = None
        self._scene_metadata = None

        self.camera_manager = CameraManager()
        self._renderer: MjAbstractRenderer | None = None
        self._use_filament = use_filament

        self.object_managers = []

        # Cached occupancy map for robot placement (expensive to create)
        self._cached_thormap = None
        self._cached_thormap_key = None  # (model_path, agent_radius, px_per_m)

        self._initialize_with_model(mj_model, mj_base_scene_path)

    def _initialize_with_model(self, mj_model: MjModel, mj_base_scene_path: str) -> None:
        """Initialize the environment with a MuJoCo model."""
        # Clean up old renderer if it exists (important for GPU texture cleanup when loading new scenes)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        # Invalidate cached thormap when scene changes
        self._cached_thormap = None
        self._cached_thormap_key = None

        # scenes
        self._mj_model = mj_model
        self._mj_base_scene_path = mj_base_scene_path
        self._scene_metadata = get_scene_metadata(mj_base_scene_path)

        # data for each batch
        self._mj_datas = [MjData(mj_model) for _ in range(self._n_batch)]
        for mj_data in self._mj_datas:
            mujoco.mj_forward(mj_model, mj_data)
            for _ in range(
                self.config.task_sampler_config.sim_settle_timesteps
            ):  # let objects settle
                mujoco.mj_step(mj_model, mj_data)
        self._robots = tuple(self._robot_factory(mj_data) for mj_data in self._mj_datas)

        # Initialize the single renderer
        # TODO HERE: need to set devices here
        # likely pattern - include device in env constructor, pass to renderer here
        if self.config.camera_config is not None:
            width, height = self.config.camera_config.img_resolution
        else:
            width, height = (640, 480)  # Default resolution
        if self._use_filament:
            self._renderer = MjFilamentRenderer(model=self.mj_model, width=width, height=height)
        else:
            self._renderer = MjOpenGLRenderer(model=self.mj_model, width=width, height=height)

        if self._parallelize and self._n_batch > 1:
            self._executor = ThreadPoolExecutor(max_workers=self._n_batch)
        else:
            self._executor = None

        # For now, instantiate a new ObjectManager per data
        from molmo_spaces.env.object_manager import ObjectManager

        for idx in range(len(self._mj_datas)):
            self.object_managers.append(ObjectManager(self, idx))

    @property
    def mj_datas(self) -> Sequence[MjData]:
        if not self.is_loaded():
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        return self._mj_datas

    @property
    def n_batch(self) -> int:
        return self._n_batch  # This is always available (stored from constructor)

    @property
    def robots(self) -> Sequence[Robot]:
        if not self.is_loaded():
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        return self._robots

    def _render_frame(
        self,
        pos: np.ndarray,
        forward: np.ndarray,
        up: np.ndarray,
        fov: float,
        segmentation: bool = False,
        depth: bool = False,
    ) -> np.ndarray:
        """Helper to render a single frame using pos, forward, up vectors."""
        if not self._renderer:
            raise RuntimeError("Renderer not initialized. Call _initialize_with_model first.")

        prev_fov = self.mj_model.vis.global_.fovy
        self.mj_model.vis.global_.fovy = fov  # set global fov
        # Create a camera view object (from simple_camera_test.py render_scene)
        cam = mujoco.MjvCamera()
        # note that passing cam to update() is not actually required, but doesn't hurt
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._renderer.update(self.current_data, cam)

        for camera in self._renderer.scene.camera:  # the for loop is necessary!
            camera: mujoco.MjvGLCamera
            camera.pos = pos
            camera.forward = forward
            camera.up = up

        if segmentation:
            self._renderer.enable_segmentation_rendering()
            frame = self._renderer.render()
            self._renderer.disable_segmentation_rendering()
        elif depth:
            self._renderer.enable_depth_rendering()
            frame = self._renderer.render()
            self._renderer.disable_depth_rendering()
        else:
            frame = self._renderer.render()

        self.mj_model.vis.global_.fovy = prev_fov  # set global fov back
        return frame

    def render_rgb_frame(self, camera_name: str) -> np.ndarray:
        """Renders an RGB frame from the perspective of the specified camera."""
        if camera_name not in self.camera_manager.registry:
            raise KeyError(f"Camera '{camera_name}' not found in registry.")

        camera = self.camera_manager.registry[camera_name]
        return self._render_frame(
            camera.pos, camera.forward, camera.up, camera.fov, segmentation=False
        )

    def render_depth_frame(self, camera_name: str) -> np.ndarray:
        """Renders a depth frame from the perspective of the specified camera.

        Returns raw metric depth values in meters as float32 array.
        Depth encoding to RGB happens at save time for video storage.

        Returns:
            np.ndarray: (H, W) float32 array of depth values in meters
        """
        if camera_name not in self.camera_manager.registry:
            raise KeyError(f"Camera '{camera_name}' not found in registry.")

        camera = self.camera_manager.registry[camera_name]
        depth_frame = self._render_frame(
            camera.pos, camera.forward, camera.up, camera.fov, depth=True
        )

        # Return raw depth in meters (encoding to RGB happens at save time)
        return depth_frame.astype(np.float32)

    def render_segmentation_frame(self, camera_name: str) -> np.ndarray:
        """Renders a segmentation frame from the perspective of the specified camera."""
        if camera_name not in self.camera_manager.registry:
            raise KeyError(f"Camera '{camera_name}' not found in registry.")

        camera = self.camera_manager.registry[camera_name]
        return self._render_frame(
            camera.pos, camera.forward, camera.up, camera.fov, segmentation=True
        )

    def get_camera_parameters(self, camera_name: str) -> dict:
        """Get camera parameters for a specific camera."""
        if camera_name not in self.camera_manager.registry:
            raise KeyError(f"Camera '{camera_name}' not found.")
        camera = self.camera_manager.registry[camera_name]
        return {"pos": camera.pos, "forward": camera.forward, "up": camera.up}

    def get_visible_objects(self, camera_name: str) -> list:
        """Get list of visible objects for a specific camera (placeholder)."""
        # This would require proper visibility calculation using segmentation - TODO
        return []

    def get_robot_gripper_positions(self, robot_index: int = 0) -> dict[str, np.ndarray]:
        """Get current world positions of all robot grippers/end-effectors.

        It's particularly useful for camera positioning to ensure good gripper visibility.

        Args:
            robot_index: Index of robot in batch (default 0)

        Returns:
            Dict mapping gripper names to their world positions as numpy arrays.
            Returns empty dict if the robot does not have any grippers.
        """
        assert 0 <= robot_index < self.n_batch, (
            f"Robot index {robot_index} out of range [0, {self.n_batch})"
        )

        robot = self.robots[robot_index]
        robot_view = robot.robot_view

        gripper_positions = {}
        for gripper_name in robot_view.get_gripper_movegroup_ids():
            mg = robot_view.get_move_group(gripper_name)
            gripper_positions[gripper_name] = mg.leaf_frame_to_world[:3, 3]

        return gripper_positions

    def cleanup_rendering(self) -> None:
        """Clean up rendering resources."""
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    def _reset_single(self, idx: int) -> None:
        mujoco.mj_resetData(self._mj_model, self._mj_datas[idx])
        mujoco.mj_forward(self._mj_model, self._mj_datas[idx])

    def reset(self, idxs: Collection[int] | None = None) -> None:
        if idxs is None:
            idxs = range(self.n_batch)
        if self._executor is not None and len(idxs) > 1:
            futures = [self._executor.submit(self._reset_single, idx) for idx in idxs]
            for future in as_completed(futures):
                future.result()
        else:
            for idx in idxs:
                self._reset_single(idx)

    def step(self, n_steps: int = 1) -> None:
        if self._executor is not None:
            futures = [
                self._executor.submit(mujoco.mj_step, self._mj_model, mj_data, n_steps)
                for mj_data in self._mj_datas
            ]
            for future in as_completed(futures):
                future.result()
        else:
            for mj_data in self._mj_datas:
                mujoco.mj_step(self._mj_model, mj_data, n_steps)

        # We got new scene state, so anything depending on data must be refreshed
        for om in self.object_managers:
            om.invalidate_data_cache()

        self.camera_manager.registry.update_all_cameras(self)

    def segmentation_fraction(self, seg: np.ndarray, body_name_or_id: str | int) -> float:
        """Calculate visibility of a body in the segmentation image."""
        model = self.current_model
        if isinstance(body_name_or_id, str):
            body_id = model.body(body_name_or_id).id
        else:
            body_id = body_name_or_id

        return np.mean(get_geom_seg_mask(model, seg[..., :2], body_id)).item()

    def get_segmentation_mask_of_object(
        self, object_name: str, camera_name: str, batch_index: int = 0
    ) -> np.ndarray | None:
        """Get binary segmentation mask for a specific object from a camera view.

        Args:
            object_name: Name of the object body to segment
            camera_name: Name of the camera to render from
            batch_index: Batch index for the environment

        Returns:
            Binary mask (HxW) where True indicates object pixels, or None if object not found
        """
        # Set current batch index for rendering
        prev_batch_index = self.current_batch_index
        try:
            self.current_batch_index = batch_index

            # Get the body ID for the object
            model = self.current_model
            try:
                body_id = model.body(object_name).id
            except KeyError:
                log.warning(f"Object '{object_name}' not found in model")
                return None

            # Render segmentation frame for the camera
            seg_frame = self.render_segmentation_frame(camera_name)

            # Extract binary mask for this specific object using get_geom_seg_mask
            object_mask = get_geom_seg_mask(model, seg_frame[..., :2], body_id)

            return object_mask.astype(bool)

        finally:
            # Restore previous batch index
            self.current_batch_index = prev_batch_index

    def check_visibility(self, camera_name: str, *target_objects) -> float | dict[str, float]:
        """
        Check visibility of one or more target objects from a specific camera.

        Args:
            camera_name: Name of camera in the registry
            *target_objects: Variable number of object/body names to check visibility for

        Returns:
            float: If single object provided, returns visibility fraction (0.0 to 1.0)
            dict: If multiple objects provided, returns mapping of object names to visibility fractions
        """
        if camera_name not in self.camera_manager.registry:
            raise KeyError(f"Camera '{camera_name}' not found in registry.")

        if len(target_objects) == 0:
            raise ValueError("At least one target object must be provided")

        try:
            # Render segmentation frame once for all objects
            seg_frame = self.render_segmentation_frame(camera_name)

            results = {}
            # Check visibility for each target object
            for obj_name in target_objects:
                try:
                    visibility = self.segmentation_fraction(seg_frame, obj_name)
                    results[obj_name] = visibility
                except ValueError:
                    results[obj_name] = 0.0

            # Return single float if only one object, dict if multiple
            if len(target_objects) == 1:
                return results[target_objects[0]]
            else:
                return results

        except Exception as e:
            # Return 0 visibility for all objects if camera render fails
            log.warning(
                f"[VISIBILITY CHECK] Failed to check visibility for camera {camera_name}: {e}"
            )
            if len(target_objects) == 1:
                return 0.0
            else:
                return {obj_name: 0.0 for obj_name in target_objects}

    def check_camera_visibility_constraints(
        self,
        visibility_resolver: Callable[[str], list[str]] | None = None,
        save_frames_dir: Path | str | None = None,
    ) -> tuple[bool, dict[str, dict[str, float]]]:
        """
        Check if all cameras with visibility constraints can see their target objects.

        This is used during robot placement to ensure that cameras (fixed exo, robot-mounted base, etc.)
        have good views of important objects (e.g., gripper, target objects).

        Args:
            visibility_resolver: Optional callable that resolves special visibility keys
                               like "__gripper__" to (possibly multiple) actual body names
            save_frames_dir: Optional directory to save RGB frames from each camera for debugging

        Returns:
            Tuple of (all_satisfied, detailed_results)
            - all_satisfied: bool indicating if ALL constraints on ALL cameras are met
            - detailed_results: dict mapping camera_name -> {object_name: visibility_fraction}
        """
        all_satisfied = True
        detailed_results = {}

        # Prepare save directory if specified
        if save_frames_dir is not None:
            save_frames_dir = Path(save_frames_dir)
            visibility_frames_dir = save_frames_dir / "visibility_check_frames"
            visibility_frames_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through all cameras in the registry
        for camera in self.camera_manager.registry:
            # Check if this camera has visibility constraints
            if (
                not hasattr(camera, "visibility_constraints")
                or camera.visibility_constraints is None
            ):
                continue

            camera_name = camera.name
            constraints = camera.visibility_constraints

            # Resolve special visibility keys
            resolved_constraints = {}
            for key, threshold in constraints.items():
                if key.startswith("__") and key.endswith("__"):
                    # Special key - resolve via callback
                    if visibility_resolver is not None:
                        resolved_keys = visibility_resolver(key)
                        if resolved_keys:
                            for resolved_key in resolved_keys:
                                resolved_constraints[resolved_key] = threshold
                        else:
                            log.warning(
                                f"[VISIBILITY CHECK] Could not resolve visibility key '{key}' for camera '{camera_name}'"
                            )
                    else:
                        log.warning(
                            f"[VISIBILITY CHECK] No visibility resolver provided for key '{key}' in camera '{camera_name}'"
                        )
                else:
                    # Regular body name
                    resolved_constraints[key] = threshold

            if not resolved_constraints:
                # No valid constraints for this camera
                continue

            # Check visibility for all objects
            try:
                visibility_results = self.check_visibility(
                    camera_name, *resolved_constraints.keys()
                )

                # Convert to dict if single object
                if not isinstance(visibility_results, dict):
                    visibility_results = {list(resolved_constraints.keys())[0]: visibility_results}

                detailed_results[camera_name] = visibility_results

                # Save frame if directory specified
                if save_frames_dir is not None:
                    try:
                        import time

                        from PIL import Image

                        rgb_frame = self.render_rgb_frame(camera_name)
                        # Convert from float [0,1] to uint8 if needed
                        if rgb_frame.dtype == np.float32 or rgb_frame.dtype == np.float64:
                            rgb_frame = (rgb_frame * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_frame)
                        # Add timestamp to filename to avoid overwriting
                        timestamp = int(time.time() * 1000)
                        frame_path = visibility_frames_dir / f"{camera_name}_{timestamp}.png"
                        img.save(frame_path)
                        log.debug(f"[VISIBILITY CHECK] Saved frame to {frame_path}")
                    except Exception as e:
                        log.warning(
                            f"[VISIBILITY CHECK] Failed to save frame for camera '{camera_name}': {e}"
                        )

                # Check if all constraints are satisfied for this camera
                for obj_name, threshold in resolved_constraints.items():
                    actual_visibility = visibility_results.get(obj_name, 0.0)
                    if actual_visibility < threshold:
                        all_satisfied = False
                        log.debug(
                            f"[VISIBILITY CHECK] Camera '{camera_name}': object '{obj_name}' "
                            f"visibility {actual_visibility:.5f} < threshold {threshold:.5f}"
                        )

            except Exception as e:
                log.warning(
                    f"[VISIBILITY CHECK] Failed to check visibility for camera '{camera_name}': {e}"
                )
                all_satisfied = False

        return all_satisfied, detailed_results

    def check_if_robot_collision_at_base_pose(
        self, robot_view, robot_pose: np.ndarray, robot_namespace: str = "robot_0/"
    ) -> bool:
        """Check if robot placement would result in collision with environment."""
        # TODO: there has to be a better way to do this
        model = self.current_model
        data = self.current_data

        # Store current robot pose
        original_pose = robot_view.base.pose.copy()

        try:
            # Temporarily place robot at candidate position
            robot_view.base.pose = robot_pose
            mujoco.mj_forward(model, data)
            return self.check_robot_collision_in_current_pose(robot_namespace)

        finally:
            # Restore original robot pose
            robot_view.base.pose = original_pose
            mujoco.mj_forward(model, data)

    def check_robot_collision_in_current_pose(self, robot_namespace: str = "robot_0/") -> bool:
        model = self.current_model
        data = self.current_data

        # Check for contacts
        contacts = data.contact
        collision_found = False

        for i in range(data.ncon):
            contact = contacts[i]
            if contact.dist > 0:  # Only consider actual contacts
                continue

            # Get body IDs for the contacting geometries
            body1_id = model.geom_bodyid[contact.geom1]
            body2_id = model.geom_bodyid[contact.geom2]

            # Get root body IDs to identify which system each contact belongs to
            root1_id = model.body_rootid[body1_id]
            root2_id = model.body_rootid[body2_id]

            # Check if this involves the robot by looking at body names
            body1_name = model.body(root1_id).name
            body2_name = model.body(root2_id).name

            # Check if either body belongs to the robot (using namespace)
            robot_involved = body1_name.startswith(robot_namespace) or body2_name.startswith(
                robot_namespace
            )

            if robot_involved:
                # Get the non-robot body name
                other_body_name = (
                    body2_name if body1_name.startswith(robot_namespace) else body1_name
                )

                actual_body1_name = model.body(body1_id).name
                actual_body2_name = model.body(body2_id).name
                log.debug(
                    f"Collision is between {actual_body1_name} and {actual_body2_name} (roots: {body1_name}, {body2_name})"
                )

                # Allow floor contacts but reject walls/obstacles
                if "floor" not in other_body_name.lower() and body1_name != body2_name:
                    collision_found = True
                    break

        return collision_found

    def get_thormap(
        self, agent_radius: float = 0.35, px_per_m: int = 200
    ) -> "ProcTHORMap | iTHORMap":
        """
        Get or create a cached occupancy map for robot placement.

        The map is cached per scene and parameters to avoid expensive recreation.
        Creating the map involves re-parsing XML, re-compiling MuJoCo model, and
        rendering multiple depth views - typically taking 10-30 seconds.

        Args:
            agent_radius: Safety radius for occupancy dilation (default 0.35m)
            px_per_m: Pixels per meter resolution (default 200)

        Returns:
            ProcTHORMap or iTHORMap depending on scene type
        """
        cache_key = (self.current_model_path, agent_radius, px_per_m)

        # Return cached map if available and matches parameters
        if self._cached_thormap is not None and self._cached_thormap_key == cache_key:
            log.debug("[THORMAP] Using cached occupancy map")
            return self._cached_thormap

        # Create new map
        log.info(
            f"[THORMAP] Creating occupancy map (agent_radius={agent_radius}, px_per_m={px_per_m})"
        )

        if "ithor" in self.current_model_path:
            thormap = iTHORMap.from_mj_model_path(
                model_path=self.current_model_path,
                agent_radius=agent_radius,
                px_per_m=px_per_m,
                device_id=None,
            )
        elif "procthor" in self.current_model_path or "holodeck" in self.current_model_path:
            thormap = ProcTHORMap.from_mj_model_path(
                model_path=self.current_model_path,
                px_per_m=px_per_m,
                agent_radius=agent_radius,
                device_id=None,
            )
        else:
            raise ValueError(f"Unknown scene type: {self.current_model_path}")

        # Cache the map
        self._cached_thormap = thormap
        self._cached_thormap_key = cache_key
        log.info("[THORMAP] Occupancy map created and cached")

        return thormap

    def place_robot_near(
        self,
        robot_view,
        target,
        max_tries: int = 10,
        sampling_radius_range: tuple[float, float] = (0.0, 1.0),
        robot_safety_radius: float = 0.35,
        preserve_z: float = None,
        face_target: bool = True,
        check_camera_visibility: bool = False,
        visibility_resolver=None,
        excluded_positions: list[np.ndarray] | None = None,
        exclusion_threshold: float | None = None,
        save_visibility_frames_dir: Path | str | None = None,
    ) -> bool:
        """
        Place robot near a target point or object with collision checking.

        Args:
            robot_view: Robot view object to position
            target: Either a 3D point (np.ndarray) or an Object instance or object name (str)
            max_tries: Maximum number of placement attempts
            sampling_radius_range: Radius range around target to sample robot positions. For picking up objects, the min radius is 0.0.
            robot_safety_radius: Safety radius for occupancy map collision checking
            preserve_z: Z height to preserve for robot (if None, uses current robot Z)
            face_target: Whether to orient robot to face the target
            check_camera_visibility: Whether to check fixed camera visibility constraints after placing robot
            visibility_resolver: Optional callable(key: str) -> str for resolving special visibility keys
            excluded_positions: List of positions to avoid (e.g. previously used positions)
            exclusion_threshold: Minimum distance from any excluded position
            save_visibility_frames_dir: Optional directory to save camera frames during visibility check
        Returns:
            bool: True if placement was successful, False otherwise
        """
        if exclusion_threshold is None:
            exclusion_threshold = (
                self.config.task_sampler_config.robot_placement_exclusion_threshold
            )

        log.debug(
            f"[PLACE_ROBOT_NEAR] Starting robot placement near target (max_tries={max_tries})"
        )

        # Extract target position
        if isinstance(target, np.ndarray):
            if target.shape != (3,):
                raise ValueError("Target point must be a 3D numpy array")
            target_pos = target
            log.debug(
                f"[PLACE_ROBOT_NEAR] Target point: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
            )
        elif isinstance(target, MlSpacesArticulationObject):
            target_pos = target.get_joint_leaf_body_position(0)

        elif isinstance(target, MlSpacesObject):
            target_pos = target.position
            log.debug(
                f"[PLACE_ROBOT_NEAR] Target object '{target.name}': ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
            )
        elif isinstance(target, str):
            # Assume it's an object name
            target_obj = create_mlspaces_body(self.current_data, target)
            target_pos = target_obj.position
            log.debug(
                f"[PLACE_ROBOT_NEAR] Target object '{target}': ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
            )
        else:
            raise ValueError(
                "Target must be a 3D point (np.ndarray), Object instance, or object name (str)"
            )

        # Get robot Z height to preserve
        initial_robot_z = preserve_z if preserve_z is not None else robot_view.base.pose[2, 3]
        log.debug(f"[PLACE_ROBOT_NEAR] Robot Z height to preserve: {initial_robot_z:.6f}m")
        log.debug(
            f"[PLACE_ROBOT_NEAR] Sampling radius range: {sampling_radius_range[0]:.3f}m - {sampling_radius_range[1]:.3f}m"
        )

        # Track timing breakdown for profiling
        import time

        place_start_time = time.perf_counter()
        map_time_ms = 0.0  # Will be set after map creation
        attempts_made = 0

        # Try occupancy map approach first (using cached map for performance)
        try:
            map_start_time = time.perf_counter()
            thormap = self.get_thormap(agent_radius=robot_safety_radius, px_per_m=200)
            map_time_ms = (time.perf_counter() - map_start_time) * 1000

            # Get free points within sampling radius of target
            free_points = thormap.get_free_points()
            target_dist = np.linalg.norm(free_points[:, :2] - target_pos[:2], axis=1)
            # Filter points within the radius range [min, max]
            valid_mask = (target_dist > sampling_radius_range[0]) & (
                target_dist < sampling_radius_range[1]
            )
            valid_points = free_points[valid_mask]

            if len(valid_points) > 0:
                log.debug(
                    f"[PLACE_ROBOT_NEAR] Found {len(valid_points)} free points within sampling radius"
                )

                # PHASE 1: Find all collision-free candidate poses (no visibility checks yet)
                # This avoids expensive rendering on every attempt
                collision_free_poses = []
                for attempt in range(max_tries):
                    attempts_made = attempt + 1
                    # Sample a random point from valid points using the radius range
                    sampled_point = sample_around_point(
                        thormap, target_pos[:2], sampling_radius_range
                    )

                    robot_base_pos = np.array([sampled_point[0], sampled_point[1], initial_robot_z])

                    # Check against excluded positions
                    if excluded_positions:
                        is_excluded = np.any(
                            np.linalg.norm(
                                np.stack(excluded_positions)[:, :2] - robot_base_pos[None, :2],
                                axis=-1,
                            )
                            < exclusion_threshold
                        )
                        if is_excluded:
                            if attempt < 5:
                                log.debug(
                                    f"[PLACE_ROBOT_NEAR] Attempt {attempt + 1}: Position excluded, trying another..."
                                )
                            continue  # Skip this point

                    # Reject pose if target is a Door and point is inside swing arc
                    if isinstance(target, Door):
                        if target.is_point_in_swing_arc(
                            robot_base_pos, safety_margin=robot_safety_radius
                        ):
                            if attempt < 5:
                                log.debug(
                                    f"[PLACE_ROBOT_NEAR] Attempt {attempt + 1}: Point in door swing arc, rejecting..."
                                )
                            continue  # Skip this point

                    # Calculate robot orientation
                    if face_target:
                        # Orient robot to face the target
                        xy_vec_robot_to_target = target_pos[:2] - robot_base_pos[:2]
                        if np.linalg.norm(xy_vec_robot_to_target) > 1e-6:  # Avoid division by zero
                            robot_base_yaw = np.arctan2(
                                xy_vec_robot_to_target[1], xy_vec_robot_to_target[0]
                            )
                        else:
                            robot_base_yaw = (
                                0.0  # Default orientation if target is at same XY position
                            )
                        # NOTE(yejin): is this robot-specific logic okay?
                        if "rum" in self.config.robot_config.robot_cls.__name__.lower():
                            # Orient robot pitch to face the target
                            robot_base_pitch = -np.arctan2(
                                target_pos[2] - robot_base_pos[2],
                                np.linalg.norm(xy_vec_robot_to_target),
                            )
                        else:
                            robot_base_pitch = 0.0  # Default pitch
                    else:
                        robot_base_yaw = 0.0  # Default orientation
                        robot_base_pitch = 0.0  # Default pitch

                    # Apply randomization to yaw
                    randomization_range = (
                        self.config.task_sampler_config.robot_placement_rotation_range_rad
                    )
                    if randomization_range > 0:
                        robot_base_yaw += np.random.uniform(
                            -randomization_range, randomization_range
                        )

                    robot_base_quat = R.from_euler(
                        "xyz", [0, robot_base_pitch, robot_base_yaw], degrees=False
                    ).as_quat(scalar_first=True)

                    # Create robot pose matrix
                    robot_pose = np.eye(4)
                    robot_pose[:3, 3] = robot_base_pos
                    robot_pose[:3, :3] = R.from_quat(robot_base_quat, scalar_first=True).as_matrix()

                    log.debug(
                        f"[PLACE_ROBOT_NEAR] Attempt {attempt + 1}: pos={robot_base_pos} yaw={np.degrees(robot_base_yaw):.1f}deg"
                    )

                    # Check for collisions
                    if not self.check_if_robot_collision_at_base_pose(
                        robot_view, robot_pose, "robot_0/"
                    ):
                        # Valid collision-free placement found - add to candidates
                        collision_free_poses.append((robot_pose, robot_base_pos, robot_base_yaw))
                        log.debug(
                            f"[PLACE_ROBOT_NEAR] Found collision-free pose #{len(collision_free_poses)}"
                        )
                        # If not checking visibility, we can return immediately with first valid pose
                        if not check_camera_visibility:
                            break
                        # Otherwise, collect a few candidates for visibility checking
                        # Stop early once we have enough candidates to avoid unnecessary collision checks
                        if len(collision_free_poses) >= self.config.collision_free_pose_limit:
                            break
                    elif attempt < 5:
                        log.debug(
                            "[PLACE_ROBOT_NEAR]   Collision detected, trying another point..."
                        )

                # PHASE 2: Check visibility only on collision-free candidates (much fewer renders)
                for pose_idx, (robot_pose, robot_base_pos, robot_base_yaw) in enumerate(
                    collision_free_poses
                ):
                    robot_view.base.pose = robot_pose
                    mujoco.mj_forward(self.current_model, self.current_data)

                    # Check visibility constraints if requested
                    if check_camera_visibility:
                        # Update camera poses (important for robot-mounted cameras)
                        self.camera_manager.registry.update_all_cameras(self)

                        visibility_satisfied, visibility_results = (
                            self.check_camera_visibility_constraints(
                                visibility_resolver=visibility_resolver
                            )
                        )

                        if not visibility_satisfied:
                            log.debug(
                                f"[PLACE_ROBOT_NEAR] Candidate {pose_idx + 1}/{len(collision_free_poses)}: Visibility constraints not met"
                            )
                            continue  # Try next collision-free candidate
                        else:
                            log.debug(
                                f"[PLACE_ROBOT_NEAR] Visibility constraints satisfied: {visibility_results}"
                            )

                    # Success!
                    total_time_ms = (time.perf_counter() - place_start_time) * 1000
                    retry_time_ms = total_time_ms - map_time_ms
                    log.info(
                        f"[PLACE_ROBOT_NEAR] Success after {attempts_made} samples, {len(collision_free_poses)} collision-free, {pose_idx + 1} visibility checks | "
                        f"map={map_time_ms:.1f}ms, retries={retry_time_ms:.1f}ms, total={total_time_ms:.1f}ms"
                    )
                    log.debug(
                        f"[PLACE_ROBOT_NEAR] Final position: ({robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f})"
                    )
                    log.debug(
                        f"[PLACE_ROBOT_NEAR] Final orientation: {np.degrees(robot_base_yaw):.1f} deg"
                    )
                    distance_to_target = np.linalg.norm(robot_base_pos[:2] - target_pos[:2])
                    log.debug(f"[PLACE_ROBOT_NEAR] Distance to target: {distance_to_target:.3f}m")
                    return True

                if len(collision_free_poses) > 0:
                    log.debug(
                        f"[PLACE_ROBOT_NEAR] Found {len(collision_free_poses)} collision-free poses but none satisfied visibility"
                    )
            else:
                log.debug(
                    "[PLACE_ROBOT_NEAR] No free points found within sampling radius, trying fallback..."
                )

        except Exception as e:
            log.exception(e)
            log.debug(f"[PLACE_ROBOT_NEAR] ❌ Occupancy map failed: {e}, failed to place robot...")

        # Log timing even on failure
        total_time_ms = (time.perf_counter() - place_start_time) * 1000
        retry_time_ms = total_time_ms - map_time_ms
        log.info(
            f"[PLACE_ROBOT_NEAR] ❌ Failed after {attempts_made} attempts | "
            f"map={map_time_ms:.1f}ms, retries={retry_time_ms:.1f}ms, total={total_time_ms:.1f}ms"
        )
        return False

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)

        # clean up object managers
        for om in self.object_managers:
            om.clear()
        self.object_managers.clear()

        # Clean up camera renderers
        self.cleanup_rendering()

        # add garbage collection to ensure all resources are released
        gc.collect()

    def __del__(self) -> None:
        self.close()
