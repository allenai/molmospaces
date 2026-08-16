"""Camera configuration classes for MolmoSpaces experiments."""

import logging
from abc import ABC
from typing import ClassVar, TypeAlias, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

from molmo_spaces.configs.abstract_config import Config

T = TypeVar("T")
Triple: TypeAlias = tuple[T, T, T]

# Ways of rendering a fisheye camera; see CameraConfig.fisheye_backend for what
# each one does and when to pick it.
FISHEYE_BACKENDS = ("cubemap", "warping")

# How a camera's pos/orientation/FOV noise is drawn and applied; see
# MjcfCameraConfig.noise_model. Named for the convention each one uses, since
# that -- not which robot happens to want it -- is what makes them differ.
CAMERA_NOISE_MODELS = ("camera_local_euler", "body_frame_axis_angle")

# How long a camera's sampled noise lasts; see MjcfCameraConfig.reset_cadence.
CAMERA_RESET_CADENCES = ("setup", "episode")

# OpenCV fisheye calibration of the G1's real head lens, plus the capture size
# (W, H) it was calibrated at. These are the values g1_molmo renders with.
# They lived in utils/fisheye_cubemap.py as module-level HEAD_FISHEYE_*
# defaults; a renderer that carries its own lens leaves two sources of truth the
# moment a config declares one too, and the renderer's copy silently wins for
# any caller that forgets to pass K/D. Config owns them now -- the renderer
# requires them and has no fallback.
G1_HEAD_FISHEYE_K = [
    [801.6382129934864, 0.0, 976.1246839545557],
    [0.0, 802.1081824931498, 542.7122090223202],
    [0.0, 0.0, 1.0],
]
G1_HEAD_FISHEYE_D = [
    -0.02559442829261663,
    0.008371943913215045,
    -0.006921566406199126,
    0.0010132813066123071,
]
G1_HEAD_FISHEYE_IMAGE_SIZE = (1920, 1080)


class CameraConfig(Config, ABC):
    """Base specification for a single camera.

    Each camera spec defines how one camera should be created and configured.
    Subclasses implement different camera types (MJCF, robot-mounted, exocentric).
    """

    name: str  # Unique identifier for this camera in the registry
    fov: float | None = None  # Field of view in degrees
    is_warped: bool = False  # Whether camera has lens distortion (e.g., GoPro fisheye)
    record_depth: bool = False  # Whether to record depth images for this camera
    skip_erosion: bool = (
        False  # Skip erosion for object point sampling (useful for wide FOV cameras)
    )

    # Which implementation produces the image when `is_warped` -- ignored
    # otherwise, so it is safe to leave at the default on a pinhole camera.
    #   "warping"  utils/fisheye_warping.py. Post-distorts a single pinhole
    #              render with a radial k1..k4 model. One render, cheap, and
    #              the right fit for a mild lens (the GoPro-style cameras every
    #              non-G1 system here uses). It can only redistribute detail
    #              the pinhole already captured, so beyond that field of view
    #              the periphery is missing rather than curved.
    #   "cubemap"  utils/fisheye_cubemap.py. Renders five wide tile cameras
    #              sharing an optical center and composites them through an
    #              OpenCV fisheye model calibrated on the real lens. Costs five
    #              renders per frame, and needs the tile cameras to exist in
    #              the MJCF -- so it is opt-in, for a true fisheye wide enough
    #              that a single pinhole cannot cover it. The G1 head is the
    #              only such camera today; see FisheyeMjcfCameraConfig.
    fisheye_backend: str = "warping"

    # How often this camera's position/orientation/FOV noise is redrawn. Every
    # camera type here carries noise fields, so the choice lives on the base.
    #   "setup"    Drawn once, when the camera is registered, and kept for the
    #              camera's lifetime -- every episode in a run sees the same
    #              mounting error, a fixed miscalibration rather than a varying
    #              one. This repo's behavior and the default.
    #   "episode"  Redrawn on every episode reset, around the un-noised pose
    #              each time (not around the previous episode's, which would
    #              random-walk). What g1_molmo does, and so what reproducing its
    #              camera *distribution* takes -- matching the per-draw formula
    #              alone is not enough. Both G1 cameras select it.
    #
    # Only honored where something actually redraws: env/camera_manager.py
    # raises NotImplementedError on "episode" rather than silently treating it
    # as "setup" (the G1 cameras are exempt -- g1_molmo_port's G1TaskSampler
    # redraws those itself).
    # TODO(max): switch to episode reset cadence per default.
    reset_cadence: str = "setup"

    # Visibility constraints for robot placement validation (optional)
    # Maps body names to minimum visibility thresholds (0.0 to 1.0)
    # Can use special keys like "__gripper__" or "__task_objects__" (resolved at placement time)
    # If specified, these constraints will be checked during robot placement when enabled
    visibility_constraints: dict[str, float] | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.fisheye_backend not in FISHEYE_BACKENDS:
            raise ValueError(
                f"unknown fisheye_backend {self.fisheye_backend!r} on camera "
                f"{self.name!r}; expected one of {FISHEYE_BACKENDS}"
            )
        if self.reset_cadence not in CAMERA_RESET_CADENCES:
            raise ValueError(
                f"unknown reset_cadence {self.reset_cadence!r} on camera {self.name!r}; "
                f"expected one of {CAMERA_RESET_CADENCES}"
            )


class MjcfCameraConfig(CameraConfig):
    """Camera defined in the MJCF file.

    This references a camera that already exists in the scene MJCF or robot MJCF.
    Useful for cameras with fixed mounting in robot models.
    """

    mjcf_name: str  # Full name of camera in MJCF (may include namespace)
    robot_namespace: str | None = None  # If specified, prepends to mjcf_name (e.g., "robot_0/")

    # Optional noise for MJCF cameras (applied to their fixed mounting)
    pos_noise_range: tuple[float, float] | tuple[Triple[float], Triple[float]] | None = (
        None  # Add noise to camera position (min, max)
    )
    orientation_noise_degrees: float | Triple[float] | None = (
        None  # Random rotation noise in degrees
    )
    fov_noise_degrees: tuple[float, float] | None = None  # Add noise to FOV (min, max)

    # How the three noise fields above are drawn and applied. Same magnitudes,
    # different conventions -- they do NOT produce the same cameras.
    #   "camera_local_euler"     The offset is rotated into the camera frame
    #                            before being added, the rotation is a per-axis
    #                            xyz euler triple composed on the right
    #                            (camera-local), and FOV is drawn first. This
    #                            repo's own convention and the default.
    #   "body_frame_axis_angle"  The offset is added in the parent body frame
    #                            unrotated, the rotation is a single uniformly-
    #                            random axis turned by a uniform angle and
    #                            composed on the left (parent frame), and the
    #                            draw order is position -> axis -> angle -> FOV.
    #                            Also draws from the env's seeded RNG rather
    #                            than global np.random, so the cameras are
    #                            reproducible from the episode seed.
    #
    # Used by the G1: both its cameras select "body_frame_axis_angle" because it
    # is g1_molmo's own `_perturb_camera` convention, and the G1 camera system
    # exists to reproduce that stack's recordings (see G1CameraSystem).
    # Everything else should stay on the default.
    noise_model: str = "camera_local_euler"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.noise_model not in CAMERA_NOISE_MODELS:
            raise ValueError(
                f"unknown noise_model {self.noise_model!r} on camera {self.name!r}; "
                f"expected one of {CAMERA_NOISE_MODELS}"
            )


class FisheyeMjcfCameraConfig(MjcfCameraConfig):
    """MJCF camera carrying the parameters the "cubemap" backend needs.

    Only worth using when a camera sets `fisheye_backend="cubemap"` -- the
    "warping" default takes none of these fields, so a plain MjcfCameraConfig
    is enough for it. The G1 head is the one camera here wide enough to need
    the cubemap path; see G1CameraSystem.

    `fov` stays the pinhole camera's own field of view and `tile_fov` the tile
    cameras', so both render paths take their field of view from this config
    and neither can drift from it silently.
    """

    # (cubemap) Suffixes appended to mjcf_name to find the tile cameras. Order is
    # load-bearing: FisheyeRenderer's LUT assumes center/up/down/left/right.
    tile_suffixes: tuple[str, str, str, str, str] = ("center", "up", "down", "left", "right")
    # Vertical FOV of each tile camera, degrees. FisheyeRenderer requires >=90
    # (it has to cover the fisheye circle with five tiles); g1_dex.xml declares
    # 100 on each head_pov_tile_* camera and g1_molmo renders them as authored.
    tile_fov: float = 100.0
    # (cubemap) Off-screen render size per tile. g1_molmo's render_fisheye default.
    tile_size: int = 512
    # (cubemap) Cosine-falloff exponent blending overlapping tiles; FisheyeRenderer default.
    weight_power: float = 4.0
    # (cubemap) OpenCV fisheye intrinsics/distortion and the (W, H) capture size
    # they were calibrated at. Default to the G1 head lens, the only one this
    # backend renders today; a different lens must say so here, because
    # utils/fisheye_cubemap.py no longer keeps a calibration of its own.
    fisheye_K: list[list[float]] = G1_HEAD_FISHEYE_K
    fisheye_D: list[float] = G1_HEAD_FISHEYE_D
    fisheye_image_size: tuple[int, int] = G1_HEAD_FISHEYE_IMAGE_SIZE

    # (cubemap) Per-episode randomization the fisheye lens has and a pinhole one
    # does not, so it has nowhere to live on the base config:
    #
    # `distortion_noise` scales every OpenCV distortion coefficient by
    # 1 +/- this fraction of its OWN magnitude -- proportional, not absolute, so
    # k1 moves ~20x more than k4 and the model stays self-consistent instead of
    # scrambling the image edge. g1_molmo calls this head_camera_distortion_noise.
    distortion_noise: float | None = None
    #
    # On a fisheye, `fov_noise_degrees` cannot move cam_fovy -- the pinhole FOV
    # is not what sets the projection, the calibrated focal length is. g1_molmo
    # instead scales K's fx/fy by 1 + u/90 for u drawn over that range, which is
    # what this divisor is. Both are applied by rebuilding the renderer's LUT
    # (FisheyeRenderer.set_intrinsics), not by touching the MJCF camera.
    fov_noise_focal_divisor: float = 90.0

    def tile_camera_names(self) -> list[str]:
        """Fully-qualified MJCF names of the five tile cameras ("cubemap" backend)."""
        prefix = self.robot_namespace or ""
        return [f"{prefix}{self.mjcf_name}_tile_{s}" for s in self.tile_suffixes]

    def cubemap_renderer_kwargs(self, output_h: int, output_w: int) -> dict:
        """Every FisheyeRenderer argument except the MuJoCo model, so callers
        construct one straight from config rather than restating its parameters.

        Output size stays a call argument: it is the resolution the caller wants
        this frame (CameraSystemConfig.img_resolution), not a property of the
        lens.
        """
        return {
            "tile_cam_names": self.tile_camera_names(),
            "K": self.fisheye_K,
            "D": self.fisheye_D,
            "image_size": self.fisheye_image_size,
            "tile_size": self.tile_size,
            "tile_fovy": self.tile_fov,
            "weight_power": self.weight_power,
            "output_h": output_h,
            "output_w": output_w,
        }


class RobotMountedCameraConfig(CameraConfig):
    """Camera dynamically mounted to a robot body.

    Camera follows the specified reference body with configurable offset and orientation.
    Can use either lookat-based positioning or quaternion-based orientation.
    """

    reference_body_names: list[str]  # Body names to try (uses first that exists)
    camera_offset: list[float] = [0.10, 0.0, -0.15]  # Position relative to reference body

    # Orientation method 1: Look-at based (simpler, more intuitive)
    lookat_offset: list[float] = [0.0, 0.0, 0.08]  # Point to look at (relative to reference body)
    up_axis: str = "z"  # Which local axis is "up" ("x", "y", or "z")

    # Orientation method 2: Quaternion-based (more precise control)
    camera_quaternion: list[float] | None = None  # [w, x, y, z] relative to reference body

    # Optional randomization at camera setup time
    pos_noise_range: tuple[float, float] | None = None  # Add noise to camera_offset (min, max)
    lookat_noise_range: tuple[float, float] | None = None  # Add noise to lookat_offset (min, max)
    orientation_noise_degrees: float | None = (
        None  # Random rotation noise in degrees (for quaternion-based)
    )


class FixedExocentricCameraConfig(CameraConfig):
    """Fixed external camera at a specific world position.

    Useful for consistent third-person views, overhead cameras, or monitoring positions.
    Can optionally add small amounts of noise for data augmentation.
    # TODO: should this also have a quaternion option? was figuring this would be most useful for fixed eval episodes
    """

    pos: list[float]  # World position [x, y, z]
    forward: list[float]  # Forward direction vector
    up: list[float]  # Up direction vector

    # Optional noise for data augmentation
    pos_noise_range: tuple[float, float] | tuple[Triple[float], Triple[float]] | None = (
        None  # Add noise to position (min, max)
    )
    orientation_noise_degrees: float | Triple[float] | None = (
        None  # Random rotation noise in degrees
    )


class RandomizedExocentricCameraConfig(CameraConfig):
    """Randomized external camera positioned around a workspace center.

    Samples camera position within specified ranges around a workspace center.
    Can use visibility constraints to ensure good views of important objects.
    CORE ASSUMPTION: workspace center will be sourced from task sampler callback function get_workspace_center
    you will always be looking at the workspace center (with optional noise).
    """

    # Sampling ranges (spherical coordinates around workspace center)
    distance_range: tuple[float, float]  # (min, max) distance from workspace center
    height_range: tuple[float, float]  # (min, max) height above workspace
    azimuth_range: tuple[float, float]  # (min, max) azimuth angle in radians
    fov_range: tuple[float, float] | None = None  # (min, max) FOV range in degrees

    # # Lookat configuration
    # NOTE CHANGE: always look at workspace center (workspace center can just be a 3d point or a body position). simplifies logic.
    lookat_noise_range: tuple[float, float] | None = None  # Add noise to lookat point

    # Visibility constraints for camera placement (optional)
    # Maps body names to minimum visibility thresholds (0.0 to 1.0)
    # Can use special keys like "__gripper__" or "__task_objects__" (resolved at setup time)
    # Note: ALL constraints must be met (no "at least one of" logic currently supported)
    # TODO: Add support for "at least one of" groups in visibility constraints
    # task sampler should implement resolve_visibility_object to provide body names for special keys if you add any
    visibility_constraints: dict[str, float] | None = None
    max_placement_attempts: int = 100  # Max attempts to satisfy visibility constraints
    allow_relaxed_constraints: bool = False  # Use best attempt if constraints not met


class EvalRobotMountedCameraConfig(RobotMountedCameraConfig):
    """Robot-mounted camera with additional intrinsics perturbation for eval."""

    fov_noise_degrees: tuple[float, float] | None = None


class EvalExocentricCameraConfig(FixedExocentricCameraConfig):
    """Eval exocentric camera whose level-0 pose is derived from a shoulder mount.

    At runtime the reference body pose is resolved into world-frame
    pos/forward/up, decomposed into spherical coordinates relative to the
    workspace center, perturbed according to the spherical noise params, and
    then placed via the normal fixed-exocentric path.

    ``pos``, ``forward``, ``up`` default to ``None`` here (overriding the
    required parent fields) because they are computed at runtime.
    """

    # Override parent required fields with None defaults (resolved at runtime)
    pos: list[float] | None = None
    forward: list[float] | None = None
    up: list[float] | None = None

    # Spherical perturbation ranges (around the reference shoulder-mount pose)
    azimuth_range: tuple[float, float] | None = None  # radians, symmetric around ref azimuth
    distance_range: tuple[float, float] | None = None  # meters, offset from ref distance
    height_range: tuple[float, float] | None = None  # meters, offset from ref height
    workspace_center_weight: float | None = (
        None  # [0,1] blend from calibrated target → workspace center
    )
    lookat_noise_range: tuple[float, float] | None = None  # meters, per-axis noise on lookat target
    fov_range: tuple[float, float] | None = None  # degrees, FOV sampling range

    max_placement_attempts: int = 50

    # Reference shoulder-mounted pose (used to compute level-0 pos/forward/up)
    reference_body_names: list[str] = ["robot_0/fr3_link0"]
    camera_offset: list[float] = [0.1, 0.57, 0.66]
    camera_quaternion: list[float] = [-0.3633, -0.1241, 0.4263, 0.8191]


AllCameraTypes: TypeAlias = (
    # Before MjcfCameraConfig: it is a subclass, and pydantic resolves a union
    # left to right, so the base would swallow it and drop the fisheye fields.
    FisheyeMjcfCameraConfig
    | MjcfCameraConfig
    | RobotMountedCameraConfig
    | FixedExocentricCameraConfig
    | RandomizedExocentricCameraConfig
    | EvalRobotMountedCameraConfig
    | EvalExocentricCameraConfig
)


class CameraSystemConfig(Config):
    """Complete camera system configuration.

    Defines all cameras that should be set up in the environment,
    along with shared settings like resolution.
    """

    # Shared settings for all cameras
    img_resolution: tuple[int, int] = (640, 480)  # (width, height)

    # Individual camera specifications
    cameras: list[AllCameraTypes] = []

    def add_camera(self, camera_spec: CameraConfig) -> None:
        """Add a camera specification to the system."""
        self.cameras.append(camera_spec)

    def get_camera_by_name(self, name: str) -> CameraConfig | None:
        """Get a camera spec by name."""
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None


class RBY1MjcfCameraSystem(CameraSystemConfig):
    """Camera system using RBY1's built-in MJCF cameras."""

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        MjcfCameraConfig(
            name="head_camera",
            mjcf_name="head_camera",
            robot_namespace="robot_0/",
            fov=139.0,
            skip_erosion=True,
        ),
        MjcfCameraConfig(
            name="wrist_camera_l",
            mjcf_name="wrist_camera_l",
            robot_namespace="robot_0/",
            record_depth=True,
        ),
        MjcfCameraConfig(
            name="wrist_camera_r",
            mjcf_name="wrist_camera_r",
            robot_namespace="robot_0/",
            record_depth=True,
        ),
        MjcfCameraConfig(
            name="camera_follower",
            mjcf_name="camera_follower",
            robot_namespace="robot_0/",
        ),
        # MjcfCameraConfig(
        #     name="camera_thirdview_follower_1",
        #     mjcf_name="camera_thirdview_follower_1",
        #     robot_namespace="robot_0/",
        # ),
        # MjcfCameraConfig(
        #     name="camera_thirdview_follower_2",
        #     mjcf_name="camera_thirdview_follower_2",
        #     robot_namespace="robot_0/",
        # ),
    ]


class RBY1GoProD455CameraSystem(CameraSystemConfig):
    """Camera system for RBY1 with GoPro head camera and D455 wrist cameras.

    Renders at 1024x576 (16:9) to accommodate both:
    - Head camera: GoPro analogue (4:3, crop to 768x576 in post-processing)
    - Wrist cameras: D455 analogue (16:9, use full frame)

    All cameras include randomization for sim-to-real transfer.
    """

    img_resolution: tuple[int, int] = (1024, 576)
    cameras: list[AllCameraTypes] = [
        # Head camera - GoPro analogue (4:3, VFOV ~94deg for wide mode)
        # Crop to 768x576 in post-processing to get 4:3 aspect ratio
        MjcfCameraConfig(
            name="head_camera",
            mjcf_name="head_camera",
            robot_namespace="robot_0/",
            fov=139.0,  # GoPro wide mode VFOV
            fov_noise_degrees=(-3.0, 3.0),
            pos_noise_range=((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            orientation_noise_degrees=(4.0, 4.0, 4.0),
            skip_erosion=True,
        ),
        # Left wrist camera - D455 analogue (16:9, VFOV ~58deg)
        MjcfCameraConfig(
            name="wrist_camera_l",
            mjcf_name="wrist_camera_l",
            robot_namespace="robot_0/",
            fov=58.0,  # D455 depth VFOV
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
        # Right wrist camera - D455 analogue (16:9, VFOV ~58deg)
        MjcfCameraConfig(
            name="wrist_camera_r",
            mjcf_name="wrist_camera_r",
            robot_namespace="robot_0/",
            fov=58.0,  # D455 depth VFOV
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
    ]


class G1CameraSystem(CameraSystemConfig):
    """Camera system for the G1 humanoid: head + right wrist camera.

    Matched against g1_molmo's own camera setup for this robot (its
    `configs/bowl_mixed_grasponly.py` plus the base FOVs in `g1_dex.xml`), so
    frames rendered here are comparable to the ones its LeRobotRecorder writes.
    See `scripts/g1_molmo_port_comparison/NEXT_STEPS.md`.

    `head_camera` owns every parameter of the G1 head fisheye (see
    FisheyeMjcfCameraConfig): tile cameras, tile size and FOV, blend exponent,
    and the lens calibration itself. `g1_molmo_port`'s G1Env renders from these
    -- `_ensure_fisheye` builds its FisheyeRenderer out of
    `cubemap_renderer_kwargs()`, and the task sampler's fisheye noise perturbs
    around `fisheye_K`/`fisheye_D` here -- so utils/fisheye_cubemap.py holds no
    parameters of its own to drift from.

    NOTE: molmo_spaces' *own* live render path (env/camera_manager.py, as
    opposed to g1_molmo_port's G1Env) still renders the plain `head_pov` pinhole
    camera, so head frames from that path differ in projection from g1_molmo's
    until it too is wired to the cubemap backend.
    """

    # g1_molmo's camera_size=(224, 384) is (height, width); img_resolution is
    # (width, height) -- see CameraSensor, which unpacks `width, height`.
    img_resolution: tuple[int, int] = (384, 224)
    cameras: list[AllCameraTypes] = [
        FisheyeMjcfCameraConfig(
            name="head_camera",
            mjcf_name="head_pov",
            robot_namespace="robot_0/",
            # fov=None takes the MJCF's own fovy (68 for head_pov, 37.956 for
            # right_wrist_camera), which is the base g1_molmo perturbs around
            # -- rather than restating it here and silently drifting from the
            # robot asset. The previous wrist value (70.0) came from the
            # commented-out camera_mount block in g1_dex.xml, not the live
            # camera, and was ~1.8x too wide.
            fov=None,
            # The real head camera is a fisheye.
            is_warped=True,
            # g1_molmo renders this head as a cubemap composite, so "warping"
            # would not reproduce its recordings however well it is tuned.
            fisheye_backend="cubemap",
            # g1_molmo's render_fisheye(tile_size=512) over the five fovy=100
            # head_pov_tile_* cameras. The lens is the G1_HEAD_FISHEYE_* default
            # of fisheye_K/D/image_size -- the same calibration g1_molmo warps
            # with, now declared at the top of this module.
            tile_fov=100.0,
            tile_size=512,
            # head_camera_pos_noise=0.01, head_camera_rot_noise=0.0349 rad,
            # drawn and applied g1_molmo's way rather than this repo's default,
            # and redrawn every episode as g1_molmo does -- which is already
            # what g1_molmo_port's G1TaskSampler does for this camera. NOTE:
            # molmo_spaces' own camera_manager still applies noise once at setup
            # and does not yet honor reset_cadence, so on that path this
            # currently describes the intent rather than driving it.
            noise_model="body_frame_axis_angle",
            reset_cadence="episode",
            orientation_noise_degrees=2.0,
            pos_noise_range=(-0.01, 0.01),
            # head_camera_distortion_noise=0.2, and the /90 focal scaling that
            # g1_molmo's fovy noise uses on a fisheye.
            distortion_noise=0.2,
            fov_noise_focal_divisor=90.0,
            # head_camera_fovy_noise=2.0. g1_molmo applies this to the fisheye
            # by scaling K's focal lengths by 1 + u/90 rather than by moving
            # cam_fovy, and additionally jitters D by
            # head_camera_distortion_noise=0.2; G1TaskSampler does the same,
            # perturbing around fisheye_K/fisheye_D above. On molmo_spaces' own
            # pinhole render path this perturbs the FOV instead.
            fov_noise_degrees=(-2.0, 2.0),
        ),
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="right_wrist_camera",
            robot_namespace="robot_0/",
            fov=None,
            # g1_molmo records RGB video only -- its LeRobotRecorder builds one
            # "video" feature per camera and no depth feature at all.
            record_depth=False,
            # wrist_camera_pos_noise=0.01, wrist_camera_rot_noise=0.0349 rad,
            # drawn and applied g1_molmo's way rather than this repo's default,
            # and redrawn every episode as g1_molmo does -- which is already
            # what g1_molmo_port's G1TaskSampler does for this camera. NOTE:
            # molmo_spaces' own camera_manager still applies noise once at setup
            # and does not yet honor reset_cadence, so on that path this
            # currently describes the intent rather than driving it.
            noise_model="body_frame_axis_angle",
            reset_cadence="episode",
            orientation_noise_degrees=2.0,
            pos_noise_range=(-0.01, 0.01),
            # wrist_camera_fovy_noise=2.0
            fov_noise_degrees=(-2.0, 2.0),
        ),
    ]


class FrankaRandomizedD405D455CameraSystem(CameraSystemConfig):
    """Camera system for Franka pick-and-place tasks with wrist cam and 2 randomized exo cams.

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (640, 368)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_cam",
            robot_namespace="robot_0/",
            fov=58.0,
            fov_noise_degrees=(-10.0, 10.0),  # ±10° FOV noise
            pos_noise_range=(-0.015, 0.015),  # ±1.5cm position noise
            orientation_noise_degrees=8.0,  # ±8° rotation noise
        ),
        # Two randomized exocentric cameras positioned around workspace center
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.2, 0.8),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(50, 90),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.2, 0.8),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(50, 90),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaDroidCameraSystem(CameraSystemConfig):
    """Camera system for Franka with DROID-style fixed cameras.

    Uses wrist camera plus DROID-style exocentric camera mounted to robot base.
    All cameras are deterministic (no noise) for consistent, reproducible viewpoints.
    This matches the behavior of the old `cameras_fixed_droid=True` setting.
    """

    img_resolution: tuple[int, int] = (
        640,
        368,
    )  # 16:9 aspect ratio and divisible by 16px for video encoding
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera (with depth for D405 simulation)
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=56.74,
            # record_depth=True,  # Enable depth recording for wrist camera
        ),
        # DROID-style exocentric camera (mounted to robot base)
        RobotMountedCameraConfig(
            name="exo_camera_1",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=71.0,
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
                # "__gripper__": 0.001,  # not necessarily visible from this cam - just focus on the object
            },
        ),
    ]


class FrankaEasyRandomizedDroidCameraSystem(CameraSystemConfig):
    """Camera system for Franka DROID system with wrist cam (ZED mini) and 2 randomized exo cams (ZED 2/ZED 2i).

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (640, 368)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.0,
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
        RobotMountedCameraConfig(  # left shoulder
            name="exo_camera_1",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=71.0,
            pos_noise_range=(-0.05, 0.05),
            orientation_noise_degrees=8.0,
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
            },
        ),
        # only use one camera at a time (having both at the same time tanks placement success)
        # RobotMountedCameraConfig(  # right shoulder
        #     name="exo_camera_2",
        #     reference_body_names=["robot_0/fr3_link0"],
        #     camera_offset=[0.1, -0.57, 0.66],
        #     camera_quaternion=[0.8190819, -0.42629058, 0.12409726, -0.36329197],
        #     fov=71.0,
        #     pos_noise_range=(-0.05, 0.05),
        #     orientation_noise_degrees=8.0,
        #     visibility_constraints={
        #         "__task_objects__": 0.001,  # Resolved by task sampler
        #     },
        # ),
    ]


class FrankaOmniPurposeCameraSystem(CameraSystemConfig):
    """Camera system for Franka DROID system with wrist cam (ZED mini), droid-alike left shoulder cam,
    2 randomized Zed2 cams, and 1 randomized GoPro cam. Intended such that data with this camera system
    can be used for a wide variety of purposes and maximally consistent ablations.

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (640, 368)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera_zed_mini",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.0,
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.02), (0.015, 0.005, 0.02)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
        RobotMountedCameraConfig(  # left shoulder
            name="droid_shoulder_light_randomization",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=71.0,
            pos_noise_range=(-0.05, 0.05),
            orientation_noise_degrees=8.0,
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
            },
        ),
        # Two randomized exocentric cameras positioned around workspace center
        RandomizedExocentricCameraConfig(
            name="randomized_zed2_analogue_1",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="randomized_zed2_analogue_2",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="randomized_gopro_analogue_1",
            distance_range=(0.2, 0.5),
            height_range=(0.1, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(137, 140),  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaRandomizedDroidCameraSystem(CameraSystemConfig):
    """Camera system for Franka DROID system with wrist cam (ZED mini) and 2 randomized exo cams (ZED 2/ZED 2i).

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (640, 368)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.0,
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
        ),
        # Two randomized exocentric cameras positioned around workspace center
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_3",
            distance_range=(0.2, 0.5),
            height_range=(0.1, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(137, 140),  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaGoProD405D455CameraSystem(CameraSystemConfig):
    """Camera system for Franka with GoPro and D405 analogue cameras with noise.

    Uses:
    - D405 analogue wrist camera: VFOV=58°, resolution 640x480, with position and orientation noise
    - 455 analogue exo camera: VFOV=58°, resolution 640x480, with position and orientation noise but around droid shoulder
    - GoPro analogue exo camera: VFOV=139°, resolution 640x480, with position and orientation noise

    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # D405-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=58.0,  # D405 vertical FOV
            record_depth=True,  # D405 has depth capability
            pos_noise_range=(-0.01, 0.01),  # ±1cm position noise
            orientation_noise_degrees=2.0,  # ±2° rotation noise
        ),
        # 455 analogue in noisy droid position
        RobotMountedCameraConfig(
            name="exo_camera_1",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=58.0,  # 455 vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            pos_noise_range=(-0.10, 0.10),  # ±2cm position noise
            orientation_noise_degrees=3.0,  # ±3° rotation noise
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
                # "__gripper__": 0.001,  # not necessarily visible from this cam - just focus on the object
            },
        ),
        # fully randomized gopro-analogue exo camera
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.2, 0.5),
            height_range=(0.1, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaGoProD405RandomizedCameraSystem(CameraSystemConfig):
    """Camera system for Franka with D405 wrist cam and 2 randomized GoPro exo cams.

    Uses:
    - D405 analogue wrist camera: VFOV=58°, resolution 640x480, with position and orientation noise
    - Two randomized GoPro exo cameras: VFOV=139°, resolution 640x480, with visibility constraints

    Workspace center sourced from task sampler, exo cameras positioned to maximize visibility
    of pickup object and gripper.
    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # D405-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_cam",
            robot_namespace="robot_0/",
            fov=58.0,  # D405 vertical FOV
            record_depth=True,  # D405 has depth capability
            pos_noise_range=(-0.01, 0.01),  # ±1cm position noise
            orientation_noise_degrees=2.0,  # ±2° rotation noise
        ),
        # Two randomized GoPro-style exocentric cameras
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.4, 1.0),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.4, 1.0),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaRobotiq2f85CameraSystem(CameraSystemConfig):
    """Camera system for Franka with Robotiq 2f85 wrist cam and 2 randomized GoPro exo cams.

    Uses:
    - Robotiq 2f85 wrist camera: VFOV=56.74°, resolution 1280x720, with position and orientation noise
    - Two randomized GoPro exo cameras: VFOV=139°, resolution 640x480, with visibility constraints
    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # Robotiq 2f85-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/",
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.4, 1.0),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__pickup_object__": 0.001,  # Resolved by task sampler
            },
            max_placement_attempts=200,
            allow_relaxed_constraints=True,
        ),
    ]


class I2rtYamCameraSystem(CameraSystemConfig):
    """Camera system for i2rt YAM robot.

    Uses robot-mounted exo camera since YAM doesn't have built-in MJCF cameras.
    The exo camera is mounted relative to the robot base (mocap body at ground level).

    Note: Camera offset z must account for the base platform height (0.7m).
    To achieve similar viewing angle as Franka DROID (camera at ~1.24m total height),
    we use z offset = 0.7 (platform) + 0.5 (above platform) = 1.2m
    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # Robotiq 2f85-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/",
        ),
        # Exocentric camera mounted relative to robot base (mocap body at ground level)
        RobotMountedCameraConfig(
            name="exo_camera_1",
            reference_body_names=["robot_0/base", "robot_0/arm"],  # Try base first (mocap body)
            camera_offset=[0.1, 0.5, 1.2],  # z=1.2 to account for 0.7m platform + 0.5m above
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],  # Similar angle to Franka DROID
            fov=71.0,
            visibility_constraints={
                "__task_objects__": 0.001,
            },
        ),
    ]


class BimanualYamCameraSystem(CameraSystemConfig):
    """Camera system for bimanual YAM robot.

    Includes wrist cameras on both arms (defined in yam.xml MJCF) and a robot-mounted
    exo camera positioned to see both arms and the workspace between them.

    Note: Camera offset z must account for the base platform height (0.7m).
    The exo camera is positioned slightly back and higher to capture both arms.
    """

    img_resolution: tuple[int, int] = (640, 368)
    cameras: list[AllCameraTypes] = [
        # Left wrist camera (defined in yam.xml, attached to left_link_6)
        MjcfCameraConfig(
            name="left_wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/left_",
            fov=58.0,
        ),
        # Right wrist camera (defined in yam.xml, attached to right_link_6)
        MjcfCameraConfig(
            name="right_wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/right_",
            fov=58.0,
        ),
        # Exocentric camera mounted relative to robot base (mocap body at ground level)
        RobotMountedCameraConfig(
            name="exo_camera",  # Name matches policy camera_mapping expectation
            reference_body_names=[
                "robot_0/base",
                "robot_0/left_arm",
            ],  # Try base first (mocap body)
            camera_offset=[0.0, 0.0, 1.56],  # Centered between both arms, 86cm above platform
            camera_quaternion=[
                0.6870,
                0.1675,
                -0.1675,
                -0.6870,
            ],  # 90° CW roll + 46.5° pitch down, arms left/right
            fov=58.0,
            visibility_constraints={
                "__task_objects__": 0.001,
            },
        ),
    ]


class FrankaEvalCameraSystem(CameraSystemConfig):
    """Unified Franka eval camera system with progressive perturbation (0-100 level).

    Cameras:
    - 1 wrist camera (FrankaDroid-like)
    - 1 ZED2-like exocentric cameras (stored pose from episode + perturbation)

    The cameras list holds only calibrated (level 0) specs with no randomization ranges.
    All randomization ranges are in per-reference level, per-camera dicts.
    Level scaling is applied by apply_eval_camera_randomization_level() in eval_camera_randomization_utils.py
    using an N-piece linear curve (e.g. 0→low→high→100 for N=3).

    Exo camera pos/forward/up are placeholders; the runtime loads stored poses from episode specs.
    """

    img_resolution: tuple[int, int] = (640, 368)

    ref_level_ranges: ClassVar[
        list[tuple[float, dict[str, dict[str, float | tuple[float, ...]]]]]
    ] = [
        (
            0.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    "orientation_noise_degrees": (0.0, 0.0, 0.0),
                    "fov_noise_degrees": (0.0, 0.0),
                },
                "exo_camera_1": {
                    "azimuth_range": (0.0, 0.0),
                    "distance_range": (0.0, 0.0),
                    "height_range": (0.0, 0.0),
                    "workspace_center_weight": 0.0,  # keep calibrated orientation
                    "lookat_noise_range": (0.0, 0.0),
                    "fov_range": (71.0, 71.0),
                },
            },
        ),
        (
            10.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": {
                    "azimuth_range": (-np.pi / 4, np.pi / 4),
                    "distance_range": (-0.05, 0.05),
                    "height_range": (-0.05, 0.05),
                    "workspace_center_weight": 1.0,  # look at workspace center
                    "lookat_noise_range": (-0.01, 0.01),
                    "fov_range": (71.0, 71.0),
                },
            },
        ),
        (
            40.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": {
                    "azimuth_range": (-np.pi / 2, np.pi / 2),
                    "distance_range": (-0.5, 0.5),
                    "height_range": (-0.1, 0.5),
                    "workspace_center_weight": 1.0,  # look at workspace center
                    "lookat_noise_range": (-0.05, 0.05),
                    "fov_range": (64.0, 72.0),
                },
            },
        ),
        (
            75.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": dict(
                    azimuth_range=(-np.pi, np.pi),
                    distance_range=(-0.5, 1.0),
                    height_range=(-0.2, 0.7),
                    workspace_center_weight=1.0,  # look at workspace center
                    lookat_noise_range=(-0.1, 0.1),
                    fov_range=(64.0, 72.0),
                ),
            },
        ),
        (
            100.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": dict(
                    azimuth_range=(-np.pi, np.pi),
                    distance_range=(-0.5, 1.5),
                    height_range=(-0.3, 0.8),
                    workspace_center_weight=1.0,  # look at workspace center
                    lookat_noise_range=(-0.15, 0.15),
                    fov_range=(64.0, 78.0),
                ),
            },
        ),
    ]

    # Calibrated (level 0) camera specs only; no randomization params.
    cameras: list[AllCameraTypes] = [
        # Wrist camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.00,
        ),
        # Shoulder camera
        EvalExocentricCameraConfig(
            name="exo_camera_1",
            fov=71.0,
            visibility_constraints={
                "__task_objects__": 0.0001,
                "__gripper__": 0.0001,
            },
        ),
    ]


AllCameraSystems: TypeAlias = (
    RBY1MjcfCameraSystem
    | RBY1GoProD455CameraSystem
    | G1CameraSystem
    | FrankaRandomizedD405D455CameraSystem
    | FrankaEasyRandomizedDroidCameraSystem
    | FrankaDroidCameraSystem
    | FrankaOmniPurposeCameraSystem
    | FrankaRandomizedDroidCameraSystem
    | FrankaGoProD405D455CameraSystem
    | FrankaGoProD405RandomizedCameraSystem
    | FrankaRobotiq2f85CameraSystem
    | FrankaEvalCameraSystem
    | I2rtYamCameraSystem
    | BimanualYamCameraSystem
    | FrankaEvalCameraSystem
)
