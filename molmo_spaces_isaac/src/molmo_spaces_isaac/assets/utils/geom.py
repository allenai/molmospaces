import mujoco as mj
import numpy as np
import usdex.core
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from scipy.spatial.transform import Rotation as R

from molmo_spaces_isaac.assets.utils.data import (
    BaseConversionData,
    Tokens,
    to_usd_quat,
    vec_to_quat_np,
)

VISUAL_CLASS = "__VISUAL_MJT__"
DYNAMIC_CLASS = "__DYNAMIC_MJT__"
ARTICULABLE_DYNAMIC_CLASS = "__ARTICULABLE_DYNAMIC_MJT__"
STRUCTURAL_CLASS = "__STRUCTURAL_MJT__"
STRUCTURAL_WALL_CLASS = "__STRUCTURAL_WALL_MJT__"

MJ_VISUAL_CLASSES = {VISUAL_CLASS, "visual"}

MASS_TOLERANCE = 1e-12
DENSITY_TOLERANCE = 1e-8

VISUAL_GROUPS: tuple[int, ...] = (0, 1, 2)


def is_visual(geom: mj.MjsGeom) -> bool:
    if geom.classname.name in MJ_VISUAL_CLASSES:
        return True
    elif geom.contype == 0 and geom.conaffinity == 0:
        return True
    return False


def create_prim_box(parent: Usd.Prim, geom: mj.MjsGeom, safe_name: str) -> UsdGeom.Cube:
    cube: UsdGeom.Cube = usdex.core.defineCube(parent, safe_name, size=1)

    width = 2.0 * geom.size[0]
    depth = 2.0 * geom.size[1]
    height = 2.0 * geom.size[2]
    cube.AddScaleOp().Set(Gf.Vec3f(width, depth, height))

    return cube


def create_prim_sphere(parent: Usd.Prim, geom: mj.MjsGeom, safe_name: str) -> UsdGeom.Sphere:
    sphere: UsdGeom.Sphere = usdex.core.defineSphere(parent, safe_name, geom.size[0])
    return sphere


def create_prim_cylinder(parent: Usd.Prim, geom: mj.MjsGeom, safe_name: str) -> UsdGeom.Cylinder:
    radius = geom.size[0]
    height = geom.size[1] * 2
    if not np.isnan(geom.fromto[0]):
        start = geom.fromto[:3]
        end = geom.fromto[3:]
        height = np.linalg.norm(end - start).item()

    cylinder: UsdGeom.Cylinder = usdex.core.defineCylinder(
        parent, safe_name, radius, height, UsdGeom.Tokens.z
    )

    return cylinder


def create_prim_capsule(parent: Usd.Prim, geom: mj.MjsGeom, safe_name: str) -> UsdGeom.Capsule:
    radius = geom.size[0]
    height = geom.size[1] * 2
    if not np.isnan(geom.fromto[0]):
        start = geom.fromto[:3]
        end = geom.fromto[3:]
        height = np.linalg.norm(end - start).item()

    capsule: UsdGeom.Capsule = usdex.core.defineCapsule(
        parent, safe_name, radius, height, UsdGeom.Tokens.z
    )

    return capsule


def create_prim_mesh(
    parent: Usd.Prim, geom: mj.MjsGeom, safe_name: str, data: BaseConversionData
) -> UsdGeom.Mesh:
    ref_mesh = data.references[Tokens.GEOMETRY].get(geom.meshname)

    if ref_mesh is None:
        raise RuntimeError(
            f"Geom '{geom.name}' has an associated mesh '{geom.meshname}' that wasn't converted yet"
        )
    prim = usdex.core.defineReference(parent, ref_mesh, safe_name)
    return UsdGeom.Mesh(prim)


def apply_physics(
    geom_prim: Usd.Prim,
    geom: mj.MjsGeom,
    data: BaseConversionData,
    override_box_with_mesh: bool = False,
) -> None:
    if is_visual(geom):
        return

    # if not np.isnan(geom.mass) and geom.mass < MASS_TOLERANCE:
    #     return

    geom_over: Usd.Prim = data.content[Tokens.PHYSICS].OverridePrim(geom_prim.GetPrim().GetPath())

    UsdPhysics.CollisionAPI.Apply(geom_over)
    if data.use_physx:
        geom_over.AddAppliedSchema("PhysxCollisionAPI")

    if geom.type == mj.mjtGeom.mjGEOM_MESH or (
        override_box_with_mesh and geom.type == mj.mjtGeom.mjGEOM_BOX
    ):
        mesh_collider_api = UsdPhysics.MeshCollisionAPI.Apply(geom_over)
        mesh_collider_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        if data.use_physx:
            geom_over.AddAppliedSchema("PhysxSDFMeshCollisionAPI")

    geom_mass_api = UsdPhysics.MassAPI.Apply(geom_over)
    if not np.isnan(geom.mass) and geom.mass >= MASS_TOLERANCE:
        geom_mass_api.CreateMassAttr().Set(geom.mass)
    elif geom.density >= DENSITY_TOLERANCE:
        geom_mass_api.CreateDensityAttr().Set(geom.density)

    # TODO(wilbert): set material properties like friction


def bind_material(geom_prim: UsdGeom.Gprim, name: str, data: BaseConversionData) -> None:
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
    if not prim.IsA(UsdGeom.Mesh):
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


def convert_geom(  # noqa: PLR0915
    parent: Usd.Prim,
    geom: mj.MjsGeom,
    safe_name: str,
    data: BaseConversionData,
    local_tf: np.ndarray | None = None,
    opt_col_collection: Usd.CollectionAPI | None = None,
    normalize_mesh_scale: bool = False,
    override_box_with_mesh: bool = False,
) -> UsdGeom.Gprim:
    geom_pos = geom.pos.copy()
    geom_quat = geom.quat.copy()
    geom_scale = np.ones(3)

    geom_prim: UsdGeom.Gprim | None = None
    match geom.type:
        case mj.mjtGeom.mjGEOM_BOX:
            if override_box_with_mesh:
                h_width, h_depth, h_height = geom.size.tolist()
                # fmt: off
                points = [
                    Gf.Vec3f(-h_width, -h_depth,  h_height), Gf.Vec3f( h_width, -h_depth,  h_height),
                    Gf.Vec3f( h_width,  h_depth,  h_height), Gf.Vec3f(-h_width,  h_depth,  h_height),
                    Gf.Vec3f(-h_width, -h_depth, -h_height), Gf.Vec3f( h_width, -h_depth, -h_height),
                    Gf.Vec3f( h_width,  h_depth, -h_height), Gf.Vec3f(-h_width,  h_depth, -h_height)
                ]
                faceVertexCounts = Vt.IntArray([4, 4, 4, 4, 4, 4])
                faceVertexIndices = Vt.IntArray([
                    0, 1, 2, 3,
                    4, 7, 6, 5,
                    0, 4, 5, 1,
                    3, 2, 6, 7,
                    0, 3, 7, 4,
                    1, 5, 6, 2
                ])
                # fmt: on
                geom_prim = usdex.core.definePolyMesh(
                    parent,
                    safe_name,
                    faceVertexCounts=faceVertexCounts,
                    faceVertexIndices=faceVertexIndices,
                    points=Vt.Vec3fArray(points),
                )
            else:
                geom_prim = create_prim_box(parent, geom, safe_name)
                geom_scale = 2.0 * geom.size
        case mj.mjtGeom.mjGEOM_SPHERE:
            geom_prim = create_prim_sphere(parent, geom, safe_name)
        case mj.mjtGeom.mjGEOM_CYLINDER:
            geom_prim = create_prim_cylinder(parent, geom, safe_name)
            if not np.isnan(geom.fromto[0]):
                start, end = Gf.Vec3d(geom.fromto[:3].tolist()), Gf.Vec3d(geom.fromto[3:].tolist())
                geom_pos = (end + start) / 2
                geom_quat = vec_to_quat_np(end - start)
        case mj.mjtGeom.mjGEOM_CAPSULE:
            geom_prim = create_prim_capsule(parent, geom, safe_name)
            if not np.isnan(geom.fromto[0]):
                start, end = Gf.Vec3d(geom.fromto[:3].tolist()), Gf.Vec3d(geom.fromto[3:].tolist())
                geom_pos = (end + start) / 2
                geom_quat = vec_to_quat_np(end - start)
        case mj.mjtGeom.mjGEOM_MESH:
            geom_prim = create_prim_mesh(parent, geom, safe_name, data)
            mesh_spec = data.spec.mesh(geom.meshname)
            if not normalize_mesh_scale and mesh_spec is not None:
                geom_scale = mesh_spec.scale.copy()
        case _:
            raise RuntimeError(f"Geom '{geom.name}' has unsupported type '{geom.type}'")

    assert geom_prim is not None, f"Must have a valid prim by now for geom '{geom.name}'"

    if local_tf is not None:
        geom_tf = np.eye(4)
        geom_tf[:3, :3] = R.from_quat(geom_quat, scalar_first=True).as_matrix()
        geom_tf[:3, 3] = geom_pos
        total_tf = geom_tf @ local_tf

        geom_pos = total_tf[:3, 3].copy()
        geom_quat = R.from_matrix(total_tf[:3, :3]).as_quat(scalar_first=True)

    usd_pos = Gf.Vec3d(*geom_pos)
    usd_orient = to_usd_quat(geom_quat)
    usd_scale = Gf.Vec3f(*geom_scale)

    usdex.core.setLocalTransform(
        geom_prim, translation=usd_pos, orientation=usd_orient, scale=usd_scale
    )  # type: ignore

    if geom.group not in VISUAL_GROUPS:
        geom_prim.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    if geom.material:
        bind_material(geom_prim, geom.material, data)

    # TODO(wilbert): must allow to set color and opacity if no material is defined as well
    if is_visual(geom) and not np.array_equal(geom.rgba, data.spec.default.geom.rgba):
        color, _ = Gf.Vec3f(*geom.rgba[:3].tolist()), geom.rgba[-1]
        usdex.core.Vec3fPrimvarData(UsdGeom.Tokens.constant, Vt.Vec3fArray([color])).setPrimvar(
            geom_prim.CreateDisplayColorPrimvar()
        )
        usdex.core.FloatPrimvarData(UsdGeom.Tokens.constant, Vt.FloatArray([1.0])).setPrimvar(
            geom_prim.CreateDisplayOpacityPrimvar()
        )

    apply_physics(geom_prim.GetPrim(), geom, data, override_box_with_mesh)

    if geom.group not in VISUAL_GROUPS and opt_col_collection is not None:
        opt_col_collection.GetIncludesRel().AddTarget(geom_prim.GetPath())

    return geom_prim
