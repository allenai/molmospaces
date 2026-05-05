import math

import numpy as np


def mujoco_depth_to_pointcloud(depth, fovx=58.0, fovy=None, aspect=None, scalingFactor=1):
    """
    Convert a depth image to a point cloud using the camera intrinsic matrix.

    Args:
        depth_image: np.ndarray, depth image
        camera_info: dict, camera information # TODO: add to arguments

    Returns:
        pointcloud: np.ndarray, point cloud

    NOTE: Stretch onboard camera is rotated image
    """
    scalingFactor = 1
    width = depth.shape[1]  #
    height = depth.shape[0]  #
    height / width  # depth.shape[1] / depth.shape[0]

    if fovx is not None and fovy is None:
        fovy = (
            2
            * math.atan(height * 0.5 / (width * 0.5 / math.tan(fovx * math.pi / 360 / 2)))
            / math.pi
            * 360
        )
    elif fovx is None and fovy is not None:
        fovx = (
            2
            * math.atan(width * 0.5 / (height * 0.5 / math.tan(fovy * math.pi / 360 / 2)))
            / math.pi
            * 360
        )

    fx = width / 2 / (math.tan(fovx * math.pi / 360))
    fy = height / 2 / (math.tan(fovy * math.pi / 360))

    points = []
    for v in range(0, height, 10):
        for u in range(0, width, 10):
            Z = depth[v][u] / scalingFactor
            if Z == 0:
                continue
            X = (u - width / 2) * Z / fx
            Y = (v - height / 2) * Z / fy
            points.append([Y, -X, Z])  # [X, Y, Z])
    return np.array(points)
