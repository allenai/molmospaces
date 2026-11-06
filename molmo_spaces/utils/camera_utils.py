from enum import StrEnum

import numpy as np
from scipy.ndimage import binary_erosion


def normalize_points(
    points: np.ndarray,
    img_width: int,
    img_height: int,
    distortion_map: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize image points to 0-1 range, optionally applying distortion correction.

    Args:
        points: Array of shape (N, 2) containing (x, y) pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
        distortion_map: Optional distortion map for warped cameras (e.g., GoPro)
                       Currently not implemented - will be added in future

    Returns:
        Normalized points in 0-1 range as array of shape (N, 2)
    """
    # Apply distortion correction if provided
    if distortion_map is not None:
        raise NotImplementedError("Distortion map correction not yet implemented")

    # Normalize to 0-1 range
    normalized_points = points.copy().astype(np.float32)
    normalized_points[:, 0] /= img_width  # x coordinate
    normalized_points[:, 1] /= img_height  # y coordinate

    return normalized_points


def erode_segmentation_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Apply binary erosion to a segmentation mask.

    Args:
        mask: Binary segmentation mask
        iterations: Number of erosion iterations

    Returns:
        Eroded binary mask
    """
    return binary_erosion(mask, iterations=iterations)


# Camera noise/cadence enums live here, not in configs/camera_configs.py, so that
# env/camera_manager.py can name them without importing the configs package: any
# `molmo_spaces.configs.*` import runs configs/__init__.py, which pulls in the exp/
# policy config tree -> policy layer -> task layer -> back into env/env.py, and that
# cycle is fatal whenever env/env.py is the module entered first. configs/
# camera_configs.py re-exports both names, so config code can keep importing them
# from there.


class CameraNoiseModel(StrEnum):
    """How a camera's pos/orientation/FOV noise is drawn and applied; see
    MjcfCameraConfig.noise_model. Named for the convention each one uses, since
    that -- not which robot wants it -- is what makes them differ. Same
    magnitudes either way: these do NOT produce the same cameras.

    CAMERA_LOCAL_EULER     Offset rotated into the camera frame before being
                           added; rotation is an xyz euler triple composed on
                           the right; FOV drawn first. This repo's convention.
    BODY_FRAME_AXIS_ANGLE  Offset added unrotated in the parent body frame;
                           rotation is one uniform random axis turned by a
                           uniform angle, composed on the left; draw order is
                           position -> axis -> angle -> FOV. Draws from the
                           env's seeded RNG, not global np.random, so the
                           cameras reproduce from the episode seed. Both G1
                           cameras select this -- it is g1_molmo's own
                           `_perturb_camera` convention, which G1CameraSystem
                           exists to reproduce. Everything else stays on the
                           default.
    """

    CAMERA_LOCAL_EULER = "camera_local_euler"
    BODY_FRAME_AXIS_ANGLE = "body_frame_axis_angle"


class CameraResetCadence(StrEnum):
    """How long a camera's sampled noise lasts; see CameraConfig.reset_cadence.

    SETUP    Drawn once at registration and kept -- a fixed miscalibration
             shared by every episode in a run. This repo's behavior.
    EPISODE  Redrawn each episode reset around the un-noised pose (not the
             previous episode's, which would random-walk). What g1_molmo does,
             and what reproducing its camera *distribution* takes -- matching
             the per-draw formula alone is not enough.

    Only honored where something actually redraws: camera_manager raises on
    EPISODE rather than silently downgrading it (the G1 cameras are exempt,
    G1TaskSampler redraws those itself).
    """

    SETUP = "setup"
    EPISODE = "episode"
