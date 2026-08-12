from dataclasses import dataclass, field
from enum import StrEnum

import msgspec
import mujoco as mj
import numpy as np
from pxr import Gf
from scipy.spatial.transform import Rotation as R

QUAT_TOLERANCE = 1e-10


class SceneObjectType(StrEnum):
    THOR_OBJ = "thor_obj"
    OBJAVERSE_OBJ = "objaverse_obj"
    CUSTOM_OBJ = "custom_obj"  # type associated to custom geometry from iTHOR houses


@dataclass
class AssetGenMetadata:
    asset_id: str
    hash_id: str
    articulated: bool
    bbox_size: list[float] = field(default_factory=list)


class MetadataObjInfoNameMap(msgspec.Struct):
    bodies: dict[str, str]
    joints: dict[str, str]
    sites: dict[str, str]


class MetadataObjInfo(msgspec.Struct):
    hash_name: str
    asset_id: str
    object_id: str
    category: str
    object_enum: SceneObjectType
    is_static: bool
    room_id: int
    name_map: MetadataObjInfoNameMap


@dataclass
class SceneObjectInfo:
    spec: mj.MjsBody
    hash_id: str
    metadata: MetadataObjInfo | None
    articulated: bool = False


def to_usd_quat(quat: np.ndarray) -> Gf.Quatf:
    return Gf.Quatf(quat[0], quat[1], quat[2], quat[3])


def from_usd_quat(quat: Gf.Quatf | Gf.Quatd) -> np.ndarray:
    return np.array([quat.real, quat.imaginary[0], quat.imaginary[1], quat.imaginary[2]])


def vec_to_quat(vec: Gf.Vec3d) -> Gf.Quatf:
    z_axis = Gf.Vec3d(0, 0, 1)
    vec.Normalize()

    # Cross product of z-axis and vector
    cross = z_axis.GetCross(vec)
    s = cross.GetLength()

    if s < QUAT_TOLERANCE:
        return Gf.Quatf(0, 1, 0, 0)
    else:
        # Normalize cross product
        cross.Normalize()

        # Calculate angle between z-axis and vector
        ang = np.arctan2(s, vec[2]).item()

        # Construct quaternion
        return Gf.Quatf(
            np.cos(ang / 2.0),
            cross[0] * np.sin(ang / 2.0),
            cross[1] * np.sin(ang / 2.0),
            cross[2] * np.sin(ang / 2.0),
        ).GetNormalized()


def get_orientation(body_spec: mj.MjsBody) -> np.ndarray:
    match body_spec.alt.type:
        case mj.mjtOrientation.mjORIENTATION_QUAT:
            return body_spec.quat.copy()
        case mj.mjtOrientation.mjORIENTATION_AXISANGLE:
            axisangle = body_spec.alt.axisangle
            return R.from_rotvec(axisangle[-1] * axisangle[:-1]).as_quat(scalar_first=True)
        case mj.mjtOrientation.mjORIENTATION_XYAXES:
            raise NotImplementedError("Support for xyaxes in orientation is not supported yet")
        case mj.mjtOrientation.mjORIENTATION_ZAXIS:
            raise NotImplementedError("Support for zaxis in orientation is not supported yet")
        case mj.mjtOrientation.mjORIENTATION_EULER:
            euler = body_spec.alt.euler
            return R.from_euler("xyz", euler, degrees=False).as_quat(scalar_first=True)
        case _:
            raise ValueError(f"Orientation type {body_spec.alt.type} is not valid")
