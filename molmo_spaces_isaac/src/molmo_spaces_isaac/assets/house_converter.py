import json
import os
import re
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version as imp_version
from pathlib import Path
from typing import Literal, cast

import msgspec
import mujoco as mj
import numpy as np
import tyro
import usdex.core
from p_tqdm import p_uimap
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from molmo_spaces_isaac import MOLMO_SPACES_ISAAC_BASE_DIR
from molmo_spaces_isaac.assets.utils.contacts import convert_contact_excludes
from molmo_spaces_isaac.assets.utils.data import (
    AssetParameters,
    BaseConversionData,
    MetadataObjInfo,
    SceneObjectInfo,
    SceneObjectType,
    Tokens,
    from_usd_quat,
    to_usd_quat,
)
from molmo_spaces_isaac.assets.utils.flatten import export_flatten
from molmo_spaces_isaac.assets.utils.geom import (
    ARTICULABLE_DYNAMIC_CLASS,
    STRUCTURAL_CLASS,
    VISUAL_CLASS,
)
from molmo_spaces_isaac.assets.utils.ithor import (
    ITHOR_IBL_COLOR,
    convert_ithor_materials,
    convert_ithor_meshes,
    convert_ithor_objects,
)
from molmo_spaces_isaac.assets.utils.material import convert_material
from molmo_spaces_isaac.assets.utils.mesh import convert_mesh
from molmo_spaces_isaac.assets.utils.skybox import create_skybox

ROOT_DIR = Path(__file__).parent.parent.parent.parent

DEFAULT_ERRORS_FILE_TEMPLATE = "mjcf_usd_scenes_conversion_errors_{identifier}.json"

ASSETS_METADATA_FILE = MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "usd_assets_metadata.json"

PARAMETERS_FILE = MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "usd_assets_parameters.yaml"

ASSET_ID_TO_CATEGORY_FILE = (
    MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "asset_id_to_object_type.json"
)

SCENE_OBJECTS_PATTERN = r"\w+_([0-9a-f]{32})_\d+_\d+_\d+"

FIX_ROTATION = R.from_rotvec([90, 0, 0], degrees=True)

DEFAULT_DIR_LIGHT_INTENSITY = 1000
DEFAULT_DIR_LIGHT_ROTATION_X = -10.0

JOINT_REF_TOLERANCE = 1e-8

X_AXIS = np.array([1.0, 0.0, 0.0], np.float64)
Y_AXIS = np.array([0.0, 1.0, 0.0], np.float64)
Z_AXIS = np.array([0.0, 0.0, 1.0], np.float64)


def get_tf_from_body(body: mj.MjsBody) -> Gf.Transform:
    curr_rot = R.from_quat(body.quat, scalar_first=True)
    new_quat = (FIX_ROTATION * curr_rot).as_quat(scalar_first=True)
    return Gf.Transform(translation=body.pos.tolist(), rotation=Gf.Rotation(to_usd_quat(new_quat)))


def get_fixed_quat(quat: np.ndarray) -> Gf.Quatf:
    curr_rot = R.from_quat(quat, scalar_first=True)
    new_quat = (FIX_ROTATION * curr_rot).as_quat(scalar_first=True)
    return to_usd_quat(new_quat)


SCENES_SUFFIXES_TO_SKIP = (
    "_orig",
    "_non_settled",
)


THOR_ASSETS_IDS_TO_SKIP = (
    "Light_Switch",
    "RoboTHOR_dresser_aneboda",
    "Laptop_20",
    "bin_6",
    "bin_11",
)

ITHOR_DEBUG_PREFIXES_TO_KEEP = (
    "mesh_",
    "floor_",
    "window_",
    "wall_",
    "standardislandheight_",
    "ceiling_",
)

THOR_ASSETS_IDS_TO_SET_REFS = ["Doorway_Double_", "Doorway_"]

ITHOR_DEBUG_PREFIXES_ARTICULATED_TO_KEEP = ("dishwasher_", "cabinet_")

TYPES_TO_USE_ONLY_PRIM: list[str] = [
    "pen",
    "pencil",
    "plate",
    "key",
    "cd",
    "book",
    "phone",
    "card",
    "bedsheet",
    "spoon",
    "fork",
    "remote",
    "laptop",
    "box",
    "statue",
    "lamp",
    "light_switch",
    "houseplant",
    "stool",
    "garbagebin",
    "garbagecan",
    "window",
    "bed",
    "shelving",
    "table",
    "dresser",
    "desk",
    "countertop",
    "cabinet",
    "bin",
    "chair",
    "painting",
    "stand",
    "room_decor",
    "tennis",
    "pan",
    "watch",
    "basketball",
    "candle",
    "dish_sponge",
]


def must_use_prim(asset_id: str) -> bool:
    asset_id_lower = asset_id.lower()
    return any([ignore_type in asset_id_lower for ignore_type in TYPES_TO_USE_ONLY_PRIM])


def print_subtree_hierarchy(prim: Usd.Prim, depth: int = 0) -> None:
    has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)  # type: ignore
    print(
        f"{'  ' * depth}name: {prim.GetName()}, type: {prim.GetTypeName()}, has_rb: {has_rigid_body}"
    )
    for child in prim.GetChildren():
        print_subtree_hierarchy(child, depth + 1)


def make_non_articulated_static(prim: Usd.Prim) -> None:
    if prim.HasAPI("PhysicsRigidBodyAPI"):
        prim.RemoveAPI("PhysicsRigidBodyAPI")
    for child_prim in prim.GetChildren():
        assert isinstance(child_prim, Usd.Prim)
        if child_prim.GetTypeName() == "PhysicsFixedJoint":
            child_prim.SetActive(False)
        else:
            make_non_articulated_static(child_prim)


def is_articulated(body: mj.MjsBody) -> bool:
    to_explore = [body]
    while len(to_explore) > 0:
        curr_body = to_explore.pop()
        if any(jnt.type != mj.mjtJoint.mjJNT_FREE for jnt in curr_body.joints):
            return True
        to_explore.extend(curr_body.bodies)
    return False


class AssetGenMetadata(msgspec.Struct):
    asset_id: str
    hash_id: str
    articulated: bool


@dataclass
class Args:
    mode: Literal["convert-single", "convert-all"]
    scene_path: Path | None = None
    dataset: Literal["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"] = "ithor"
    split: Literal["train", "val", "test"] = "train"
    scenes_dir: Path | None = None
    start: int = -1
    end: int = -1
    output_dir: Path | None = None
    thor_usd_dir: Path | None = None
    objaverse_usd_dir: Path | None = None

    is_ithor: bool = False

    start_sleep: bool = False

    bundle: bool = False

    max_workers: int = 1

    use_physx: bool = False

    use_newton: bool = False

    verbose: bool = False


G_ARGS: Args | None = None

G_PARAMETERS: dict[str, AssetParameters] = {}
G_ASSET_ID_TO_CATEGORY: dict[str, str] = {}


@dataclass
class SceneConversionData(BaseConversionData):
    collision_groups: dict[str, UsdPhysics.CollisionGroup] = field(default_factory=dict)
    collider_collections: dict[str, Usd.CollectionAPI] = field(default_factory=dict)

    structural_prims: list[Usd.Prim] = field(default_factory=list)
    articulable_dynamic_prims: list[Usd.Prim] = field(default_factory=list)

    has_ceiling: bool = False

    usd_assets_metadata: dict[str, AssetGenMetadata] = field(default_factory=dict)

    scene_metadata: dict[str, dict[str, MetadataObjInfo]] = field(default_factory=dict)


@dataclass
class SceneConversionResult:
    success: bool
    mjcf_path: Path
    usd_path: Path
    error_msg: str = ""
    error_traceback: str = ""


def create_physics_scene(data: BaseConversionData) -> None:
    asset_stage: Usd.Stage = data.content[Tokens.ASSET]
    content_stage: Usd.Stage = data.content[Tokens.CONTENTS]
    physics_stage: Usd.Stage = data.content[Tokens.PHYSICS]

    # ensure the name is valid across all layers
    safe_name = data.name_cache.getPrimName(asset_stage.GetPseudoRoot(), "PhysicsScene")

    # author the scene in the physics layer
    scene: UsdPhysics.Scene = UsdPhysics.Scene.Define(
        physics_stage, asset_stage.GetPseudoRoot().GetPath().AppendChild(safe_name)
    )

    # reference the scene in the asset layer, but from the content layer
    content_scene: Usd.Prim = content_stage.GetPseudoRoot().GetChild(safe_name)
    usdex.core.definePayload(asset_stage.GetPseudoRoot(), content_scene, safe_name)

    gravity_vector: Gf.Vec3d = Gf.Vec3d(data.spec.option.gravity.astype(np.float64).tolist())
    scene.CreateGravityDirectionAttr().Set(gravity_vector.GetNormalized())
    scene.CreateGravityMagnitudeAttr().Set(gravity_vector.GetLength())


def get_authoring_metadata(house_name: str) -> str:
    msg = "AI2-THOR mjcf-usd scene converter\n"
    msg += f"    time: {datetime.now().strftime('%m-%d-%y %H:%M:%S')}\n"
    msg += f"    scene: {house_name}\n"
    msg += f"    usd-exchange-version: {imp_version('usd-exchange')}\n"
    return msg


def bind_material(geom_prim: UsdGeom.Gprim, name: str, data: SceneConversionData) -> None:
    local_materials = data.content[Tokens.MATERIALS].GetDefaultPrim().GetChild(Tokens.MATERIALS)
    ref_material = data.references[Tokens.MATERIALS].get(name)
    if ref_material is None:
        raise RuntimeError(f"Material '{name}' not found in material library")

    material_prim = UsdShade.Material(local_materials.GetChild(ref_material.GetName()))
    if not material_prim:
        material_prim = UsdShade.Material(
            usdex.core.defineReference(local_materials, ref_material, ref_material.GetName())
        )

    prim = geom_prim.GetPrim()
    if not prim.IsA(UsdGeom.Mesh):  # type: ignore
        has_diffuse_texture = False
        shader = usdex.core.computeEffectivePreviewSurfaceShader(material_prim)
        if shader:
            if diffuse_input := shader.GetInput("diffuseColor"):
                value_attrs = diffuse_input.GetValueProducingAttributes()
                for attr in value_attrs:
                    source_prim = attr.GetPrim()
                    if source_prim and source_prim.GetTypeName() == "Shader":
                        shader_type = source_prim.GetAttribute("info:id").Get()
                        if shader_type == "UsdUVTexture":
                            has_diffuse_texture = True
                            break

        if has_diffuse_texture:
            print(
                f"[WARN]: binding a textured material '{material_prim.GetPath()}' to a {prim.GetTypeName()} Prim ('{prim.GetPath()}') "
                "will discard textures at render time"
            )

    geom_over = data.content[Tokens.MATERIALS].OverridePrim(geom_prim.GetPath())
    usdex.core.bindMaterial(geom_over, material_prim)


def create_lights(data: SceneConversionData, is_ithor: bool) -> None:
    content_stage = data.content[Tokens.CONTENTS]

    # NOTE(wilbert): for now, just create a default directional light that kind of look fine
    dir_light = UsdLux.DistantLight.Define(
        content_stage, content_stage.GetDefaultPrim().GetPath().AppendChild("scene_dir_light")
    )
    dir_light.GetIntensityAttr().Set(DEFAULT_DIR_LIGHT_INTENSITY)
    dir_light_prim = dir_light.GetPrim()
    dir_light_xform = UsdGeom.Xformable(dir_light_prim)
    dir_light_xform.AddRotateXOp().Set(DEFAULT_DIR_LIGHT_ROTATION_X)

    # TODO(wilbert): refactor this part, we just copied it to make it work for now
    if is_ithor:
        ibl_light = usdex.core.defineDomeLight(
            parent=content_stage.GetDefaultPrim(),
            name="scene_ibl_light",
            intensity=500,
        )
        ibl_light.GetColorAttr().Set(Gf.Vec3f(ITHOR_IBL_COLOR))

        for prim in data.content[Tokens.GEOMETRY].Traverse():
            if not prim.IsA(UsdGeom.Gprim):  # type: ignore
                continue
            if all(substr not in prim.GetName() for substr in ("wall_", "ceiling_")):
                continue
            if "visual" in prim.GetName():
                primvar_api = UsdGeom.PrimvarsAPI(prim)
                shadow_primvar = primvar_api.CreatePrimvar(
                    "doNotCastShadows", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant
                )
                shadow_primvar.Set(True)
    else:
        for prim in data.content[Tokens.GEOMETRY].Traverse():
            if not prim.IsA(UsdGeom.Gprim):  # type: ignore
                continue
            if not prim.GetName().startswith("wall_"):
                continue
            if "visual" in prim.GetName():
                primvar_api = UsdGeom.PrimvarsAPI(prim)
                shadow_primvar = primvar_api.CreatePrimvar(
                    "doNotCastShadows", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant
                )
                shadow_primvar.Set(True)

    if data.has_ceiling:
        for prim in data.content[Tokens.GEOMETRY].Traverse():
            if not prim.IsA(UsdGeom.Gprim):  # type: ignore
                continue
            if not prim.GetName().startswith("ceiling_"):
                continue
            if "visual" in prim.GetName():
                primvar_api = UsdGeom.PrimvarsAPI(prim)
                shadow_primvar = primvar_api.CreatePrimvar(
                    "doNotCastShadows", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant
                )
                shadow_primvar.Set(True)


def convert_wall_and_room_meshes(
    data: SceneConversionData,
    walls: list[mj.MjsBody],
    rooms: list[mj.MjsBody],
    ceilings: list[mj.MjsBody] = [],
) -> None:
    bodies = walls + rooms + ceilings
    meshes: dict[str, mj.MjsMesh] = {}
    for body in bodies:
        for geom in body.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type == mj.mjtGeom.mjGEOM_MESH:
                if mesh_handle := data.spec.mesh(geom.meshname):
                    meshes[mesh_handle.name] = mesh_handle

    geometry_scope = data.libraries[Tokens.GEOMETRY].GetDefaultPrim()

    orig_names = list(meshes.keys())
    safe_names = data.name_cache.getPrimNames(geometry_scope, orig_names)

    for orig_name, safe_name in zip(orig_names, safe_names):
        mesh = meshes[orig_name]
        mesh_prim = usdex.core.defineXform(geometry_scope, safe_name).GetPrim()
        data.references[Tokens.GEOMETRY][orig_name] = mesh_prim
        convert_mesh(mesh_prim, mesh, data.spec, normalize_mesh_scale=False)


def convert_wall_and_room_materials(
    data: SceneConversionData,
    walls: list[mj.MjsBody],
    rooms: list[mj.MjsBody],
    ceilings: list[mj.MjsBody] = [],
) -> None:
    bodies = walls + rooms + ceilings
    materials: dict[str, mj.MjsMaterial] = {}
    for body in bodies:
        for geom in body.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.classname.name == VISUAL_CLASS:
                if mat_handle := data.spec.material(geom.material):
                    materials[mat_handle.name] = mat_handle

    materials_scope = data.libraries[Tokens.MATERIALS].GetDefaultPrim()

    orig_names = list(materials.keys())
    safe_names = data.name_cache.getPrimNames(materials_scope, orig_names)

    for orig_name, safe_name in zip(orig_names, safe_names):
        material = materials[orig_name]
        material_prim = convert_material(
            materials_scope, safe_name, material, data.spec, data.libraries[Tokens.MATERIALS]
        ).GetPrim()
        data.references[Tokens.MATERIALS][orig_name] = material_prim


def convert_walls_and_rooms(
    data: SceneConversionData,
    walls: list[mj.MjsBody],
    rooms: list[mj.MjsBody],
    ceilings: list[mj.MjsBody] = [],
) -> None:
    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )
    bodies = walls + rooms + ceilings
    for body in bodies:
        for geom in body.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                # NOTE(wilbert): all walls|rooms are mesh colliders, but just in case skip other types
                continue

            ref_mesh = data.references[Tokens.GEOMETRY].get(geom.meshname)
            if ref_mesh is None:
                raise RuntimeError(
                    f"Geom '{geom.name}' has an associated mesh '{geom.meshname}' that wasn't converted yet"
                )
            mesh_handle = data.spec.mesh(geom.meshname)
            if mesh_handle is None:
                continue

            safe_name = data.name_cache.getPrimName(geo_scope, geom.name)
            prim = usdex.core.defineReference(geo_scope, ref_mesh, safe_name)
            geom_prim = UsdGeom.Mesh(prim)

            usd_pos = Gf.Vec3d(geom.pos.tolist())
            usd_orient = to_usd_quat(geom.quat)
            # TODO(wilbert): for some reason the scale is not being used in the mesh geometry, so we
            # have to force it here, otherwise the walls|rooms will face a different direction
            usd_scale = Gf.Vec3f(mesh_handle.scale.tolist())

            usdex.core.setLocalTransform(
                geom_prim, translation=usd_pos, orientation=usd_orient, scale=usd_scale
            )

            if geom.classname.name != VISUAL_CLASS:
                geom_prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

            if geom.material:
                bind_material(geom_prim, geom.material, data)

            if geom.classname.name == VISUAL_CLASS and not np.array_equal(
                geom.rgba, data.spec.default.geom.rgba
            ):
                color, opacity = Gf.Vec3f(*geom.rgba[:3].tolist()), geom.rgba[-1]
                usdex.core.Vec3fPrimvarData(
                    UsdGeom.Tokens.constant, Vt.Vec3fArray([color])
                ).setPrimvar(geom_prim.CreateDisplayColorPrimvar())
                usdex.core.FloatPrimvarData(
                    UsdGeom.Tokens.constant, Vt.FloatArray([opacity])
                ).setPrimvar(geom_prim.CreateDisplayOpacityPrimvar())

            if geom.classname.name != VISUAL_CLASS:
                geom_over: Usd.Prim = data.content[Tokens.PHYSICS].OverridePrim(
                    geom_prim.GetPrim().GetPath()
                )
                UsdPhysics.CollisionAPI.Apply(geom_over)
                mesh_collider_api = UsdPhysics.MeshCollisionAPI.Apply(geom_over)
                mesh_collider_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

                data.structural_prims.append(geom_over)


def check_is_class_type(root_body: mj.MjsBody, desired_classname: str) -> bool:
    is_articulable_dyn = False
    to_explore = [root_body]
    while len(to_explore) > 0:
        curr_body = to_explore.pop()
        if any(geom.classname.name == desired_classname for geom in curr_body.geoms):
            is_articulable_dyn = True
            break
        to_explore.extend(curr_body.bodies)
    return is_articulable_dyn


def set_articulated_object_init_qpos(  # noqa: PLR0915
    scene_obj: SceneObjectInfo,
    root_body_prim: Usd.Prim,
    root_body_spec: mj.MjsBody,
    asset_id: str,
) -> None:
    if all(substr not in asset_id for substr in THOR_ASSETS_IDS_TO_SET_REFS):
        return

    if scene_obj.metadata is None:
        print(f"[WARN]: scene object '{root_body_spec.name}' should have valid metadata")
        return

    joints_handles: list[mj.MjsJoint] = []

    def collect_joints(body: mj.MjsBody) -> None:
        for jnt in body.joints:
            assert isinstance(jnt, mj.MjsJoint)
            if jnt.type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                continue
            joints_handles.append(jnt)

        for child in body.bodies:
            assert isinstance(child, mj.MjsBody)
            collect_joints(child)

    collect_joints(root_body_spec)

    joints_names: list[str] = [jnt.name for jnt in joints_handles]
    joints_refs: list[float] = [np.rad2deg(jnt.ref).item() for jnt in joints_handles]
    joints_axes: list[np.ndarray] = [cast(np.ndarray, jnt.axis) for jnt in joints_handles]

    revolute_joints: dict[str, UsdPhysics.RevoluteJoint] = {}
    fixed_joints: list[UsdPhysics.FixedJoint] = []
    rigid_bodies: dict[str, Usd.Prim] = {}

    for curr_prim in Usd.PrimRange(root_body_prim):
        if revolute_joint := UsdPhysics.RevoluteJoint.Get(
            root_body_prim.GetStage(), curr_prim.GetPath()
        ):
            revolute_joints[revolute_joint.GetPath().name] = revolute_joint
        elif fixed_joint := UsdPhysics.FixedJoint.Get(
            root_body_prim.GetStage(), curr_prim.GetPath()
        ):
            fixed_joints.append(fixed_joint)
        elif curr_prim.HasAPI(UsdPhysics.RigidBodyAPI):  # type: ignore
            rigid_bodies[curr_prim.GetName()] = curr_prim

    @dataclass
    class BodyNode:
        spec: mj.MjsBody
        local_tf: np.ndarray
        orig_name: str
        target_name: str
        parent: str = ""
        children: list[str] = field(default_factory=list)
        new_local_tf: np.ndarray | None = None
        prim: Usd.Prim | None = None

    bodies_name_map: dict[str, str] = scene_obj.metadata.name_map.bodies
    joints_name_map: dict[str, str] = scene_obj.metadata.name_map.joints

    body_nodes: dict[str, BodyNode] = {}

    def collect_kintree(body: mj.MjsBody) -> None:
        is_root = body.parent.name == "world"
        local_tf = np.eye(4)
        if not is_root:
            local_tf[:3, :3] = R.from_quat(body.quat, scalar_first=True).as_matrix()
            local_tf[:3, 3] = body.pos

        assert scene_obj.metadata is not None, (
            f"Must have metadata for '{root_body_spec.name}' by now, something went wrong"
        )
        target_body_name = bodies_name_map[body.name]
        body_nodes[target_body_name] = BodyNode(
            spec=body,
            local_tf=local_tf,
            orig_name=body.name,
            target_name=target_body_name,
            parent=body.parent.name,
            children=[bodies_name_map[child.name] for child in body.bodies],
        )
        for child in body.bodies:
            collect_kintree(child)

    collect_kintree(root_body_spec)

    def get_axis(joint: UsdPhysics.RevoluteJoint) -> np.ndarray:
        jnt_axis_id = joint.GetAxisAttr().Get()
        match jnt_axis_id:
            case "X":
                return X_AXIS
            case "Y":
                return Y_AXIS
            case "Z":
                return Z_AXIS
        return Y_AXIS

    for jnt_name, jnt_ref, jnt_axis in zip(joints_names, joints_refs, joints_axes):
        if abs(jnt_ref) < JOINT_REF_TOLERANCE:
            continue
        target_jnt_name = joints_name_map[jnt_name]
        if revolute_joint := revolute_joints.get(target_jnt_name):
            joint_prim = revolute_joint.GetPrim()
            jnt_axis_usd = get_axis(revolute_joint)
            axis_sign = 1.0 if np.dot(jnt_axis_usd, jnt_axis) > 0 else -1.0
            orig_quat = from_usd_quat(revolute_joint.GetLocalRot1Attr().Get())
            orig_rot = R.from_quat(orig_quat, scalar_first=True)
            extra_rot = R.from_rotvec([0.0, axis_sign * jnt_ref, 0.0], degrees=True)
            new_quat = (extra_rot * orig_rot).as_quat(scalar_first=True)
            revolute_joint.GetLocalRot1Attr().Set(to_usd_quat(new_quat))

            lower_limit = revolute_joint.GetLowerLimitAttr().Get()
            upper_limit = revolute_joint.GetUpperLimitAttr().Get()

            new_lower_limit = lower_limit - jnt_ref
            new_upper_limit = upper_limit - jnt_ref

            revolute_joint.GetLowerLimitAttr().Set(new_lower_limit)
            revolute_joint.GetUpperLimitAttr().Set(new_upper_limit)

            body_prim = joint_prim.GetParent()
            body_xform = UsdGeom.Xformable(body_prim)
            body_local_tf = body_xform.GetLocalTransformation()
            body_orig_pos = body_local_tf.ExtractTranslation()
            body_orig_quat = from_usd_quat(body_local_tf.ExtractRotation().GetQuat())
            body_orig_rot = R.from_quat(body_orig_quat, scalar_first=True)
            body_extra_rot = R.from_rotvec([0.0, axis_sign * jnt_ref, 0.0], degrees=True)
            body_new_quat = (body_extra_rot * body_orig_rot).as_quat(scalar_first=True)

            body_new_tf = np.eye(4)
            body_new_tf[:3, :3] = R.from_quat(body_new_quat, scalar_first=True).as_matrix()
            body_new_tf[:3, 3] = np.array([*body_orig_pos])

            if body_prim.GetName() in body_nodes:
                body_nodes[body_prim.GetName()].new_local_tf = body_new_tf
                body_nodes[body_prim.GetName()].prim = body_prim

    # NOTE(wilbert): have to recompute the transforms along the kinematic tree, as one of the nodes
    # has changed, so all its children must change as well (recall we don't have a recursive
    # structure, but a flattened one)
    if root_body_name := bodies_name_map.get(root_body_spec.name):
        root_body_node = body_nodes[root_body_name]

        def update_nodes(node: BodyNode) -> None:
            if node.new_local_tf is not None and node.prim is not None:
                new_local_pos = node.new_local_tf[:3, 3]
                new_local_quat = R.from_matrix(node.new_local_tf[:3, :3]).as_quat(scalar_first=True)
                xform = UsdGeom.Xformable(node.prim)
                xform.GetTranslateOp().Set(Gf.Vec3d(*new_local_pos))
                xform.GetOrientOp().Set(to_usd_quat(new_local_quat))
                # diff_tf = node.new_local_tf @ np.linalg.inv(node.local_tf)
                for child_name in node.children:
                    child_node = body_nodes[child_name]
                    if child_name not in rigid_bodies:
                        continue
                    child_prim = rigid_bodies[child_name]
                    child_new_local_tf = node.new_local_tf @ child_node.local_tf
                    child_xform = UsdGeom.Xformable(child_prim)
                    child_new_pos = Gf.Vec3d(*child_new_local_tf[:3, 3])
                    child_new_quat = R.from_matrix(child_new_local_tf[:3, :3]).as_quat(
                        scalar_first=True
                    )
                    child_xform.GetTranslateOp().Set(child_new_pos)
                    child_xform.GetOrientOp().Set(to_usd_quat(child_new_quat))
            for child_name in node.children:
                update_nodes(body_nodes[child_name])

        update_nodes(root_body_node)


def convert_scene_objects(data: SceneConversionData, scene_objects: list[SceneObjectInfo]) -> None:  # noqa: PLR0915
    global G_ARGS  # noqa: PLW0602

    assert G_ARGS is not None, "Must have 'args' defined by now, something went wrong"

    assert G_ARGS.thor_usd_dir is not None, (
        "Must have a valid thor_usd_dir by now, something went wrong"
    )

    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )
    for scene_obj in scene_objects:
        if not scene_obj.metadata:
            # TODO(wilbert): should we just skip?, both thor, objaverse, and custom should have metadata
            continue
        asset_id = scene_obj.metadata.asset_id
        is_static = scene_obj.metadata.is_static
        obj_enum = scene_obj.metadata.object_enum

        if any(substr in asset_id for substr in THOR_ASSETS_IDS_TO_SKIP):
            continue

        usd_pos = Gf.Vec3d(scene_obj.spec.pos.tolist())
        usd_orient = to_usd_quat(scene_obj.spec.quat)

        safe_name = data.name_cache.getPrimName(geo_scope, scene_obj.spec.name)

        root_body_xform = usdex.core.defineXform(geo_scope, safe_name)
        root_body_prim = root_body_xform.GetPrim()
        usdex.core.setLocalTransform(root_body_prim, translation=usd_pos, orientation=usd_orient)

        if check_is_class_type(scene_obj.spec, ARTICULABLE_DYNAMIC_CLASS):
            data.articulable_dynamic_prims.append(root_body_prim)
        elif check_is_class_type(scene_obj.spec, STRUCTURAL_CLASS):
            data.structural_prims.append(root_body_prim)

        match obj_enum:
            case SceneObjectType.THOR_OBJ:
                can_use_mesh = not must_use_prim(asset_id)
                suffix = "mesh" if can_use_mesh else "prim"
                model_path = (
                    G_ARGS.thor_usd_dir / f"{asset_id}_{suffix}" / f"{asset_id}_{suffix}.usda"
                )
                if not model_path.is_file():
                    model_path = G_ARGS.thor_usd_dir / asset_id / f"{asset_id}.usda"
                if not model_path.is_file():
                    print(f"[WARN]: model '{model_path}' not found")
                    continue
                rel_model_path = os.path.relpath(model_path, start=data.usd_path)
                root_body_prim.GetReferences().AddReference(rel_model_path)

                if is_static:
                    geom_layer = root_body_prim.GetChild(Tokens.GEOMETRY.value)
                    asset_root_prim = geom_layer.GetChild(asset_id)
                    if asset_root_prim.IsValid():
                        if not UsdPhysics.ArticulationRootAPI(asset_root_prim):
                            make_non_articulated_static(geom_layer)
                        else:
                            # TODO(wilbert): check if we can make it rel to body0 instead, as this
                            # kind of makes it difficult to move the whole scene, as the bodies
                            # transforms are in world space
                            fixed_joint = UsdPhysics.FixedJoint.Define(
                                data.content[Tokens.GEOMETRY],
                                asset_root_prim.GetPath().AppendChild("FixedJointRoot"),
                            )
                            fixed_joint.GetBody1Rel().SetTargets([asset_root_prim.GetPath()])
                            fixed_joint.CreateLocalPos0Attr().Set(usd_pos)
                            fixed_joint.CreateLocalRot0Attr().Set(usd_orient)
                            fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3d(0, 0, 0))
                            fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf.GetIdentity())

                if scene_obj.articulated:
                    set_articulated_object_init_qpos(
                        scene_obj,
                        root_body_prim.GetChild(Tokens.GEOMETRY.value),
                        scene_obj.spec,
                        scene_obj.metadata.asset_id,
                    )
            case SceneObjectType.OBJAVERSE_OBJ:
                if G_ARGS.objaverse_usd_dir is None:
                    continue
                model_path = G_ARGS.objaverse_usd_dir / f"obja_{asset_id}" / f"obja_{asset_id}.usda"
                if not model_path.is_file():
                    print(f"[WARN]: model '{model_path}' not found")
                    continue
                rel_model_path = os.path.relpath(model_path, start=data.usd_path)
                root_body_prim.GetReferences().AddReference(rel_model_path)
                if is_static:
                    geom_layer = root_body_prim.GetChild(Tokens.GEOMETRY.value)
                    asset_root_prim = geom_layer.GetChild(f"obja_{asset_id}")
                    if asset_root_prim.IsValid():
                        make_non_articulated_static(geom_layer)
            case SceneObjectType.CUSTOM_OBJ:
                raise RuntimeError(
                    "Shouldn't get here, custom objects can't be handled by this function"
                )


def add_ground_plane(data: SceneConversionData) -> None:
    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )
    half_width = UsdGeom.GetStageMetersPerUnit(geo_scope.GetStage()) * 10
    half_depth = UsdGeom.GetStageMetersPerUnit(geo_scope.GetStage()) * 10
    plane_prim: UsdGeom.Plane = usdex.core.definePlane(
        geo_scope, "floor", half_width * 2.0, half_depth * 2.0, UsdGeom.Tokens.z
    )
    usdex.core.setLocalTransform(
        plane_prim, translation=Gf.Vec3d(0, 0, 0), orientation=Gf.Quatf.GetIdentity()
    )
    geom_over: Usd.Prim = data.content[Tokens.PHYSICS].OverridePrim(plane_prim.GetPrim().GetPath())
    UsdPhysics.CollisionAPI.Apply(geom_over)
    plane_prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    data.structural_prims.append(geom_over)


def create_collision_groups(data: SceneConversionData) -> None:
    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )

    structural_cls_group = UsdPhysics.CollisionGroup.Define(
        geo_scope.GetStage(), geo_scope.GetPath().AppendChild("structural_cls_group")
    )
    structural_cls_collection = Usd.CollectionAPI.Apply(structural_cls_group.GetPrim(), "colliders")

    articulable_dynamic_cls_group = UsdPhysics.CollisionGroup.Define(
        geo_scope.GetStage(), geo_scope.GetPath().AppendChild("articulable_dynamic_cls_group")
    )
    articulable_dynamic_cls_collection = Usd.CollectionAPI.Apply(
        articulable_dynamic_cls_group.GetPrim(), "colliders"
    )

    structural_cls_group.CreateFilteredGroupsRel().AddTarget(
        articulable_dynamic_cls_group.GetPath()
    )
    structural_cls_group.GetFilteredGroupsRel().AddTarget(structural_cls_group.GetPath())
    articulable_dynamic_cls_group.CreateFilteredGroupsRel().AddTarget(
        structural_cls_group.GetPath()
    )

    data.collision_groups[STRUCTURAL_CLASS] = structural_cls_group
    data.collider_collections[STRUCTURAL_CLASS] = structural_cls_collection

    data.collision_groups[ARTICULABLE_DYNAMIC_CLASS] = articulable_dynamic_cls_group
    data.collider_collections[ARTICULABLE_DYNAMIC_CLASS] = articulable_dynamic_cls_collection


def populate_collision_groups(data: SceneConversionData) -> None:
    def collect_collider_prims(prim: Usd.Prim, colliders: list[Usd.Prim]) -> None:
        if prim.HasAPI(UsdPhysics.CollisionAPI):  # type: ignore
            colliders.append(prim)
        for child_prim in prim.GetChildren():
            collect_collider_prims(child_prim, colliders)

    for structural_prim in data.structural_prims:
        data.collider_collections[STRUCTURAL_CLASS].GetIncludesRel().AddTarget(
            structural_prim.GetPath()
        )

    articulable_targets: list[Sdf.Path] = []
    for articulable_dyn_prim in data.articulable_dynamic_prims:
        collider_prims: list[Usd.Prim] = []
        collect_collider_prims(articulable_dyn_prim, collider_prims)
        targets = [col_prim.GetPath() for col_prim in collider_prims]
        articulable_targets.extend(targets)

    for art_target in articulable_targets:
        data.collider_collections[ARTICULABLE_DYNAMIC_CLASS].GetIncludesRel().AddTarget(art_target)


def convert(scene_path: Path) -> SceneConversionResult:  # noqa: PLR0915
    global G_ARGS, G_PARAMETERS, G_ASSET_ID_TO_CATEGORY  # noqa: PLW0602

    assert G_ARGS is not None, "Must have 'args' defined by now, something went wrong"

    assert G_ARGS.output_dir, "Must have a valid output_dir by now, something went wrong"

    house_name = scene_path.stem.replace("_ceiling", "")
    output_dir = G_ARGS.output_dir / scene_path.stem
    if G_ARGS.bundle:
        output_dir = G_ARGS.output_dir / f"{scene_path.stem}-bundle"
    output_dir.mkdir(exist_ok=True)
    usd_path = output_dir / "scene.usda"

    success = True
    error_msg = ""
    error_traceback = ""
    try:
        # if True:
        with open(ASSETS_METADATA_FILE, "rb") as fhandle:
            usd_assets_metadata = msgspec.json.decode(
                fhandle.read(), type=dict[str, AssetGenMetadata]
            )

        metadata_file = scene_path.parent / f"{house_name}_metadata.json"
        with open(metadata_file, "rb") as fhandle:
            scene_metadata = msgspec.json.decode(
                fhandle.read(), type=dict[str, dict[str, MetadataObjInfo]]
            )

        shutil.copyfile(metadata_file, output_dir / "scene_metadata.json")

        stage = usdex.core.createStage(
            usd_path.absolute().as_posix(),
            defaultPrimName=house_name,
            upAxis=UsdGeom.Tokens.z,
            linearUnits=UsdGeom.LinearUnits.meters,
            authoringMetadata=get_authoring_metadata(house_name),
        )
        if stage is None:
            msg = f"[ERROR]: couldn't create a usd stage for scene '{house_name}' at path '{usd_path}'"
            return SceneConversionResult(
                success=False, mjcf_path=scene_path, usd_path=usd_path, error_msg=msg
            )

        stage.SetMetadata(UsdPhysics.Tokens.kilogramsPerUnit, 1)
        _ = usdex.core.defineXform(stage, stage.GetDefaultPrim().GetPath()).GetPrim()

        spec = mj.MjSpec.from_file(scene_path.as_posix())

        data = SceneConversionData(
            spec=spec,
            stage=stage,
            usd_path=usd_path,
            export_scene=False,
            export_sites=False,
            root_rotation=None,
            start_sleep=G_ARGS.start_sleep,
            has_ceiling=scene_path.stem.endswith("ceiling"),
            usd_assets_metadata=usd_assets_metadata,
            scene_metadata=scene_metadata,
            thor_parameters=G_PARAMETERS,
            thor_id_to_category=G_ASSET_ID_TO_CATEGORY,
            use_physx=G_ARGS.use_physx,
            use_newton=G_ARGS.use_newton,
        )

        ceiling_mj_arr: list[mj.MjsBody] = []
        if not G_ARGS.is_ithor and data.has_ceiling:
            for root_body in spec.worldbody.bodies:
                assert isinstance(root_body, mj.MjsBody)
                if root_body.name.startswith("ceiling_"):
                    ceiling_mj_arr.append(root_body)

        asset_payload = usdex.core.createAssetPayload(stage)
        assert asset_payload is not None, (
            "Must be able to create an asset payload for the asset stage"
        )
        data.content[Tokens.CONTENTS] = asset_payload

        # Make libraries to store both material and geometry information from the scene ----------------
        material_lib = usdex.core.addAssetLibrary(
            asset_payload, Tokens.MATERIALS.value, format="usda"
        )
        assert material_lib is not None, (
            f"Must be able to create material library, something went wrong for house {house_name}"
        )

        data.libraries[Tokens.MATERIALS] = material_lib
        data.references[Tokens.MATERIALS] = {}

        geometry_lib = usdex.core.addAssetLibrary(
            asset_payload, Tokens.GEOMETRY.value, format="usdc"
        )
        assert geometry_lib is not None, (
            f"Must be able to create geometry library, something went wrong for house {house_name}"
        )

        data.libraries[Tokens.GEOMETRY] = geometry_lib
        data.references[Tokens.GEOMETRY] = {}
        # ----------------------------------------------------------------------------------------------

        walls: list[mj.MjsBody] = []
        rooms: list[mj.MjsBody] = []
        scene_objects: list[SceneObjectInfo] = []
        ithor_custom_objects: list[SceneObjectInfo] = []
        for root_body in spec.worldbody.bodies:
            assert isinstance(root_body, mj.MjsBody)
            name_sub = re.sub(r"[()\[\]{}]", "", root_body.name)
            if p_match := re.search(SCENE_OBJECTS_PATTERN, name_sub):
                hash_id = p_match.group(1)
                if obj_metadata := scene_metadata["objects"].get(root_body.name, None):
                    if obj_metadata.object_enum == SceneObjectType.CUSTOM_OBJ:
                        ithor_custom_objects.append(
                            SceneObjectInfo(
                                spec=root_body,
                                hash_id=hash_id,
                                metadata=obj_metadata,
                                articulated=is_articulated(root_body),
                            )
                        )
                    else:
                        scene_objects.append(
                            SceneObjectInfo(
                                spec=root_body,
                                hash_id=hash_id,
                                metadata=obj_metadata,
                                articulated=is_articulated(root_body),
                            )
                        )
            elif "wall_" in root_body.name:
                walls.append(root_body)
            elif "room_" in root_body.name:
                rooms.append(root_body)

        # Convert all materials into a separate usd library --------------------------------------------
        convert_wall_and_room_materials(data, walls, rooms, ceiling_mj_arr)
        if G_ARGS.is_ithor:
            convert_ithor_materials(data, ithor_custom_objects)

        usdex.core.saveStage(
            data.libraries[Tokens.MATERIALS], comment=f"Material library for house {house_name}"
        )

        material_asset_content = usdex.core.addAssetContent(
            asset_payload, Tokens.MATERIALS.value, format="usda"
        )
        assert material_asset_content is not None, (
            f"Couldn't create asset content for material, for model {data.spec.modelname}"
        )

        data.content[Tokens.MATERIALS] = material_asset_content
        # ----------------------------------------------------------------------------------------------

        # Convert all meshes into a separate usd library -----------------------------------------------
        convert_wall_and_room_meshes(data, walls, rooms, ceiling_mj_arr)
        if G_ARGS.is_ithor:
            convert_ithor_meshes(data, ithor_custom_objects)

        usdex.core.saveStage(
            data.libraries[Tokens.GEOMETRY], comment=f"Geometry library for house {house_name}"
        )
        # ----------------------------------------------------------------------------------------------

        asset_content_geo = usdex.core.addAssetContent(
            asset_payload, Tokens.GEOMETRY.value, format="usda"
        )
        assert asset_content_geo is not None, (
            f"Couldn't create asset content for geometries for model '{house_name}'"
        )
        data.content[Tokens.GEOMETRY] = asset_content_geo

        asset_content_physics = usdex.core.addAssetContent(
            asset_payload, Tokens.PHYSICS.value, format="usda"
        )
        assert asset_content_physics is not None, (
            f"Couldn't create asset content for physics for model '{house_name}'"
        )
        data.content[Tokens.PHYSICS] = asset_content_physics
        data.content[Tokens.PHYSICS].SetMetadata(UsdPhysics.Tokens.kilogramsPerUnit, 1)
        data.references[Tokens.PHYSICS] = {}

        create_collision_groups(data)

        if not G_ARGS.is_ithor:
            convert_walls_and_rooms(data, walls, rooms, ceiling_mj_arr)
        convert_scene_objects(data, scene_objects)
        if G_ARGS.is_ithor:
            _ = convert_ithor_objects(data, ithor_custom_objects, data.collider_collections)

        add_ground_plane(data)

        populate_collision_groups(data)

        usdex.core.addAssetInterface(stage, source=asset_payload)

        if not G_ARGS.is_ithor:
            create_skybox(data)
        create_lights(data, G_ARGS.is_ithor)

        convert_contact_excludes(data)

        if G_ARGS.bundle:
            export_flatten(data, usd_path, "usda", is_scene=True)
        else:
            usdex.core.saveStage(stage, comment="")

        # usdex.core.saveStage(stage, comment="")
    except Exception as e:
        success = False
        error_msg = f"Couldn't convert mjcf file '{scene_path.stem}', error: {e}"
        error_traceback = traceback.format_exc()
        print(f"[ERROR]: {error_msg}")

    return SceneConversionResult(
        success=success,
        mjcf_path=scene_path,
        usd_path=usd_path,
        error_msg=error_msg,
        error_traceback=error_traceback,
    )


def can_convert_scene(scene_path: Path) -> bool:
    return scene_path.is_file() and all(
        substr not in scene_path.stem for substr in SCENES_SUFFIXES_TO_SKIP
    )


def get_ithor_houses_paths(scenes_dir: Path, start: int = -1, end: int = -1) -> list[Path]:
    houses_xmls: list[Path] = []

    use_range = start != -1 and end != -1 and start < end
    for i in range(1, 431, 1):
        if use_range and (i < start or i >= end):
            continue
        scene_path = scenes_dir / "ithor" / f"FloorPlan{i}_physics.xml"
        if not scene_path.is_file():
            continue
        houses_xmls.append(scene_path)

    return houses_xmls


def get_procthor_like_houses_paths(
    scenes_dir: Path,
    dataset: str,
    split: str,
    start: int = -1,
    end: int = -1,
) -> list[Path]:
    houses_xmls: list[Path] = []

    scenes_dataset_dir = scenes_dir / f"{dataset}-{split}"
    use_range = start != -1 and end != -1 and start < end
    if use_range:
        for house_idx in range(start, end):
            house_path = scenes_dataset_dir / f"{split}_{house_idx}.xml"
            if house_path.is_file():
                houses_xmls.append(house_path)
            house_path_ceiling = scenes_dataset_dir / f"{split}_{house_idx}_ceiling.xml"
            if house_path_ceiling.is_file():
                houses_xmls.append(house_path_ceiling)
    else:
        for house_path in scenes_dataset_dir.rglob("*.xml"):
            if can_convert_scene(house_path):
                houses_xmls.append(house_path)

    return houses_xmls


def main() -> int:  # noqa: PLR0915
    global G_ARGS, G_PARAMETERS, G_ASSET_ID_TO_CATEGORY  # noqa: PLW0603

    G_ARGS = tyro.cli(Args)

    # Validate all given paths ---------------------------------------------------------------------
    if not PARAMETERS_FILE.is_file():
        print(
            f"Couldn't find parameters file @ '{PARAMETERS_FILE}', which is required for generating the houses"
        )
        return 1

    with open(PARAMETERS_FILE, "rb") as fhandle:
        G_PARAMETERS = msgspec.yaml.decode(fhandle.read(), type=dict[str, AssetParameters])

    if not ASSET_ID_TO_CATEGORY_FILE.is_file():
        print(
            f"Couldn't find id-to-category file @ '{ASSET_ID_TO_CATEGORY_FILE}', which is required for generating the houses"
        )
        return 1

    with open(ASSET_ID_TO_CATEGORY_FILE, "rb") as fhandle:
        G_ASSET_ID_TO_CATEGORY = msgspec.json.decode(fhandle.read(), type=dict[str, str])

    if G_ARGS.thor_usd_dir is None:
        print("Must provide a valid path to the USD version of the thor assets via --thor-usd-dir")
        return 1
    if not G_ARGS.thor_usd_dir.is_dir():
        print(f"Given path to thor-usd-dir @ '{G_ARGS.thor_usd_dir}' is not a valid directory")
        return 1

    if G_ARGS.output_dir is None:
        print("Must provide a path to the save folder where to generate the USD houses")
        return 1
    if not G_ARGS.output_dir.is_dir():
        G_ARGS.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------------------

    usdex.core.activateDiagnosticsDelegate()
    usdex.core.setDiagnosticsLevel(
        usdex.core.DiagnosticsLevel.eStatus
        if G_ARGS.verbose
        else usdex.core.DiagnosticsLevel.eWarning
    )

    match G_ARGS.mode:
        case "convert-single":
            if not G_ARGS.scene_path:
                print("[ERROR]: must provide a scene via --scene_path")
                return 1
            if not G_ARGS.scene_path.is_file():
                print(f"[ERROR]: the given scene path '{G_ARGS.scene_path}' is not a valid file")
                return 1

            if not G_ARGS.output_dir:
                print("[ERROR]: must provide an output directory via --output-dir")
                return 1
            if not G_ARGS.output_dir.is_dir():
                print(
                    f"[ERROR]: the given output dir '{G_ARGS.output_dir}' is not a valid directory"
                )
                return 1

            result = convert(G_ARGS.scene_path)
            if result.success:
                print(
                    f"[INFO]: successfully converted mjcf scene '{G_ARGS.scene_path.stem}' to USD"
                )
            else:
                print(
                    f"[ERROR]: got an error while converting mjcf scene '{G_ARGS.scene_path.stem}' to USD"
                )

        case "convert-all":
            if G_ARGS.scenes_dir is None:
                print(
                    "[ERROR]: must provide a folder to the scenes to be converted via --scenes_dir"
                )
                return 1
            if not G_ARGS.scenes_dir.is_dir():
                print(
                    f"[ERROR]: given scenes folder path '{G_ARGS.scenes_dir}' is not a valid directory"
                )
                return 1

            scenes_xmls = (
                get_ithor_houses_paths(G_ARGS.scenes_dir, start=G_ARGS.start, end=G_ARGS.end)
                if G_ARGS.dataset == "ithor"
                else get_procthor_like_houses_paths(
                    G_ARGS.scenes_dir,
                    G_ARGS.dataset,
                    G_ARGS.split,
                    start=G_ARGS.start,
                    end=G_ARGS.end,
                )
            )

            error_messages: list[str] = []
            error_tracebacks: list[str] = []
            error_scenes: list[str] = []
            if G_ARGS.max_workers > 1:
                results = p_uimap(convert, scenes_xmls, num_cpus=G_ARGS.max_workers)
                for result in results:
                    if not result.success:
                        error_messages.append(result.error_msg)
                        error_tracebacks.append(result.error_traceback)
                        error_scenes.append(result.mjcf_path.stem)
            else:
                for scene_xml in tqdm(scenes_xmls):
                    result = convert(scene_xml)
                    if not result.success:
                        error_messages.append(result.error_msg)
                        error_tracebacks.append(result.error_traceback)
                        error_scenes.append(result.mjcf_path.stem)

            date = datetime.now().strftime("%m%d%y")
            identifier = f"{G_ARGS.dataset}-{G_ARGS.split}-{date}"
            if G_ARGS.start != -1 and G_ARGS.end != -1:
                identifier = f"{G_ARGS.dataset}-{G_ARGS.split}_{G_ARGS.start}-{G_ARGS.end}-{date}"
            errors_filepath = Path(DEFAULT_ERRORS_FILE_TEMPLATE.format(identifier=identifier))

            if len(error_messages) > 0:
                errors_data = {}
                if errors_filepath.is_file():
                    with open(errors_filepath) as fhandle:
                        errors_data = json.load(fhandle)

                for msg, trace, scene_name in zip(error_messages, error_tracebacks, error_scenes):
                    errors_data[scene_name] = dict(message=msg, trace=trace)

                with open(errors_filepath, "w") as fhandle:
                    json.dump(errors_data, fhandle, indent=4)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
