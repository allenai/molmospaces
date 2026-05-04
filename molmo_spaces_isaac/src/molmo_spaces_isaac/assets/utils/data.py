from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import msgspec
import mujoco as mj
import numpy as np
import usdex.core
from pxr import Gf, Usd
from scipy.spatial.transform import Rotation as R

QUAT_TOLERANCE = 1e-10

# TODO(wilbert): refactor this part, had to copy some stuff to common.py, bc importing usdex.core
# breaks when using isaaclab for some reason T_T


@dataclass
class AssetGenMetadata:
    asset_id: str
    hash_id: str
    articulated: bool
    bbox_size: list[float] = field(default_factory=list)


class Tokens(StrEnum):
    ASSET = usdex.core.getAssetToken()
    LIBRARY = usdex.core.getLibraryToken()
    CONTENTS = usdex.core.getContentsToken()
    GEOMETRY = usdex.core.getGeometryToken()
    MATERIALS = usdex.core.getMaterialsToken()
    TEXTURES = usdex.core.getTexturesToken()
    PAYLOAD = usdex.core.getPayloadToken()
    PHYSICS = usdex.core.getPhysicsToken()


@dataclass
class BodyData:
    body_spec: mj.MjsBody
    body_name: str
    body_safename: str
    body_tf: np.ndarray
    body_prim: Usd.Prim


class AssetParametersCore(msgspec.Struct):
    stiffness: float
    damping: float
    targetPosition: float | None = None


class AssetParametersPhysx(msgspec.Struct):
    armature: float
    jointFriction: float


class AssetParameters(msgspec.Struct):
    core: AssetParametersCore
    physx: AssetParametersPhysx


@dataclass
class BaseConversionData:
    spec: mj.MjSpec
    stage: Usd.Stage
    usd_path: Path

    content: dict[Tokens, Usd.Stage] = field(default_factory=dict)
    libraries: dict[Tokens, Usd.Stage] = field(default_factory=dict)
    references: dict[Tokens, dict[str, Usd.Prim]] = field(default_factory=dict)
    name_cache: usdex.core.NameCache = field(default_factory=usdex.core.NameCache)

    bodies: dict[str, BodyData] = field(default_factory=dict)
    bodies_to_fix: list[tuple[str, str]] = field(default_factory=list)

    export_scene: bool = True
    export_sites: bool = False

    root_rotation: np.ndarray | None = None

    start_sleep: bool = False

    use_physx: bool = False
    use_newton: bool = False

    thor_parameters: dict[str, AssetParameters] = field(default_factory=dict)
    thor_id_to_category: dict[str, str] = field(default_factory=dict)

    comment: str = ""


class SceneObjectType(StrEnum):
    THOR_OBJ = "thor_obj"
    OBJAVERSE_OBJ = "objaverse_obj"
    CUSTOM_OBJ = "custom_obj"  # type associated to custom geometry from iTHOR houses


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

def to_usd_quatd(quat: np.ndarray) -> Gf.Quatd:
    return Gf.Quatd(quat[0], quat[1], quat[2], quat[3])

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


def vec_to_quat_np(vec: np.ndarray) -> np.ndarray:
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    vec_norm = np.linalg.norm(vec)
    vec /= vec_norm

    # Cross product of z-axis and vector
    cross = np.cross(z_axis, vec)
    s = np.linalg.norm(cross)

    if s < QUAT_TOLERANCE:
        return np.array([0, 1, 0, 0], dtype=np.float32)
    else:
        # Normalize cross product
        cross_norm = np.linalg.norm(cross)
        cross /= cross_norm

        # Calculate angle between z-axis and vector
        ang = np.arctan2(s, vec[2]).item()

        # Construct quaternion
        return np.array(
            [
                np.cos(ang / 2.0),
                cross[0] * np.sin(ang / 2.0),
                cross[1] * np.sin(ang / 2.0),
                cross[2] * np.sin(ang / 2.0),
            ]
        )


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
