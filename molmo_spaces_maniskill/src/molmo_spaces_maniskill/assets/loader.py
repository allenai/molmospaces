from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypeVar

import mujoco as mj
import numpy as np
import sapien
from mani_skill.envs.scene import ManiSkillScene
from sapien import ActorBuilder, ArticulationBuilder, Pose
from sapien.physx import PhysxArticulation, PhysxMaterial
from sapien.render import RenderMaterial, RenderTexture2D
from sapien.wrapper.articulation_builder import LinkBuilder
from scipy.spatial.transform import Rotation as R

VISUAL_CLASSES = {"__VISUAL_MJT__", "visual"}
VISUAL_THRESHOLD_DENSITY = 1e-5
VISUAL_THRESHOLD_MASS = 1e-6

CAPSULE_FIX_POSE = Pose(q=R.from_euler("xyz", [0, np.pi / 2, 0]).as_quat(scalar_first=True))
CYLINDER_FIX_POSE = Pose(q=R.from_euler("xyz", [0, np.pi / 2, 0]).as_quat(scalar_first=True))

X_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

WORLD_UP = Z_AXIS.copy()

SceneT = TypeVar("SceneT", sapien.Scene, ManiSkillScene)

THOR_COLLISION_GROUPS: dict[str, list[int]] = {
    "__STRUCTURAL_MJT__": [0b1000, 0b1111, 1, 0],
    "__STRUCTURAL_WALL_MJT__": [0b1000, 0b1111, 1, 0],
    # "__DYNAMIC_MJT__": [0b0001, 0b1111, 1, 0],
    "__ARTICULABLE_DYNAMIC_MJT__": [0b0000, 0b0111, 1, 0],
}


@dataclass
class MjcfTextureInfo:
    name: str
    type: mj.mjtTexture
    rgb1: list
    rgb2: list
    file: Path


@dataclass
class MjcfJointInfo:
    name: str
    type: Literal["free", "fixed", "hinge", "slide", "ball"]
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: X_AXIS.copy())
    limited: int = False
    limits: np.ndarray = field(default_factory=lambda: np.array([-np.inf, np.inf]))
    frictionloss: float = 0.0
    damping: float = 0.0


def mjc_joint_type_to_str(
    jnt_type: mj.mjtJoint,
) -> Literal["free", "fixed", "hinge", "slide", "ball"]:
    match jnt_type:
        case mj.mjtJoint.mjJNT_FREE:
            return "free"
        case mj.mjtJoint.mjJNT_HINGE:
            return "hinge"
        case mj.mjtJoint.mjJNT_SLIDE:
            return "slide"
        case mj.mjtJoint.mjJNT_BALL:
            return "ball"
        case _:
            raise RuntimeError(f"Joint of type '{jnt_type}' is not valid")


QUAT_TOLERANCE = 1e-10
AXIS_NORM_TOLERANCE = 1e-3


def vec_to_quat(vec: np.ndarray) -> np.ndarray:
    vec /= np.linalg.norm(vec)

    cross = np.cross(Z_AXIS, vec)
    s = np.linalg.norm(cross)

    if s < QUAT_TOLERANCE:
        return np.array([0.0, 1.0, 0.0, 0.0])
    else:
        cross /= np.linalg.norm(cross)
        ang = np.arctan2(s, vec[2]).item()
        quat = np.array(
            [
                np.cos(ang / 2.0),
                cross[0] * np.sin(ang / 2.0),
                cross[1] * np.sin(ang / 2.0),
                cross[2] * np.sin(ang / 2.0),
            ]
        )
        quat /= np.linalg.norm(quat)
        return quat


def get_frame_axes(axis_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(np.dot(axis_x, [1.0, 0.0, 0.0])) > 0.9:  # noqa: PLR2004
        axis_y = np.cross(axis_x, WORLD_UP)
        axis_y = axis_y / np.linalg.norm(axis_y)
    else:
        axis_y = np.cross(axis_x, X_AXIS)
        axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    return axis_x, axis_y, axis_z


def is_visual(geom: mj.MjsGeom) -> bool:
    return (geom.classname.name in VISUAL_CLASSES) or (geom.contype == 0 and geom.conaffinity == 0)


def get_visual_specs_from_body(mjs_body: mj.MjsBody) -> list[mj.MjsGeom]:
    return [geom for geom in mjs_body.geoms if is_visual(geom)]


def get_collider_specs_from_body(mjs_body: mj.MjsBody) -> list[mj.MjsGeom]:
    return [geom for geom in mjs_body.geoms if not is_visual(geom)]


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


def add_colliders_to_sapien_link(
    link_builder: LinkBuilder,
    mj_spec: mj.MjSpec,
    mj_body_name: str,
    mj_model_dir: Path,
) -> None:
    mjs_body = mj_spec.body(mj_body_name)
    if mjs_body is None:
        return
    colliders_specs = get_collider_specs_from_body(mjs_body)
    for idx in range(len(colliders_specs)):
        col_spec: mj.MjsGeom = colliders_specs[idx]
        local_pose = Pose(p=tuple(col_spec.pos), q=tuple(col_spec.quat))
        physx_material: PhysxMaterial | None = None
        if col_spec.condim == 3:  # noqa: PLR2004
            physx_material = PhysxMaterial(
                static_friction=col_spec.friction[0],
                dynamic_friction=col_spec.friction[0],
                restitution=0,
            )
        elif col_spec.condim == 1:
            physx_material = PhysxMaterial(static_friction=0, dynamic_friction=0, restitution=0)

        if col_spec.type == mj.mjtGeom.mjGEOM_BOX:
            link_builder.add_box_collision(
                pose=local_pose,
                half_size=tuple(col_spec.size),
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_SPHERE:
            link_builder.add_sphere_collision(
                pose=local_pose,
                radius=col_spec.size[0],
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_CAPSULE:
            radius = col_spec.size[0]
            half_length = col_spec.size[1]
            if not np.isnan(col_spec.fromto[0]):
                start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                pos = (end + start) / 2
                quat = vec_to_quat(end - start)
                local_pose = Pose(p=tuple(pos), q=tuple(quat))
                half_length = np.linalg.norm(end - start).item() / 2.0
            link_builder.add_capsule_collision(
                pose=local_pose * CAPSULE_FIX_POSE,
                radius=radius,
                half_length=half_length,
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_CYLINDER:
            radius = col_spec.size[0]
            half_length = col_spec.size[1]
            if not np.isnan(col_spec.fromto[0]):
                start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                pos = (end + start) / 2
                quat = vec_to_quat(end - start)
                local_pose = Pose(p=tuple(pos), q=tuple(quat))
                half_length = np.linalg.norm(end - start).item() / 2.0
            link_builder.add_cylinder_collision(
                pose=local_pose * CYLINDER_FIX_POSE,
                radius=radius,
                half_length=half_length,
                density=col_spec.density,
                material=physx_material,
            )
        elif col_spec.type == mj.mjtGeom.mjGEOM_MESH:
            mesh_spec = mj_spec.mesh(col_spec.meshname)
            if mesh_spec is not None:
                mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                link_builder.add_convex_collision_from_file(
                    pose=local_pose,
                    filename=mesh_path.as_posix(),
                    scale=tuple(mesh_spec.scale),
                    density=col_spec.density,
                    material=physx_material,
                )
        else:
            raise ValueError(f"Collider geom type {col_spec.type} not supported")

        if col_groups := THOR_COLLISION_GROUPS.get(col_spec.classname.name):
            link_builder.collision_groups = col_groups


def get_rgba_from_geom(mj_spec: mj.MjSpec, mj_geom: mj.MjsGeom) -> np.ndarray:
    rgba = mj_geom.rgba.copy()
    if mj_geom.material:
        mj_mat = mj_spec.material(mj_geom.material)
        if mj_mat is not None:
            rgba = mj_mat.rgba.copy()
    return rgba


def add_visuals_to_sapien_link(
    link: LinkBuilder,
    mj_spec: mj.MjSpec,
    mj_body_name: str,
    mj_model_dir: Path,
    materials: dict[str, RenderMaterial],
    colliders_are_visuals: bool = False,
) -> None:
    mjs_body = mj_spec.body(mj_body_name)
    if mjs_body is None:
        return
    visual_specs = (
        get_visual_specs_from_body(mjs_body)
        if not colliders_are_visuals
        else get_collider_specs_from_body(mjs_body)
    )
    for idx in range(len(visual_specs)):
        vis_spec: mj.MjsGeom = visual_specs[idx]
        local_pose = Pose(p=tuple(vis_spec.pos), q=tuple(vis_spec.quat))

        match vis_spec.type:
            case mj.mjtGeom.mjGEOM_BOX:
                link.add_box_visual(
                    pose=local_pose,
                    half_size=tuple(vis_spec.size),
                    material=tuple(get_rgba_from_geom(mj_spec, vis_spec)),
                )
            case mj.mjtGeom.mjGEOM_SPHERE:
                link.add_sphere_visual(
                    pose=local_pose,
                    radius=vis_spec.size[0],
                    material=tuple(get_rgba_from_geom(mj_spec, vis_spec)),
                )
            case mj.mjtGeom.mjGEOM_CAPSULE:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    local_pose = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                link.add_capsule_visual(
                    pose=local_pose * CAPSULE_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(get_rgba_from_geom(mj_spec, vis_spec)),
                )
            case mj.mjtGeom.mjGEOM_CYLINDER:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    local_pose = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                link.add_cylinder_visual(
                    pose=local_pose * CYLINDER_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_MESH:
                mesh_spec = mj_spec.mesh(vis_spec.meshname)
                if mesh_spec is not None:
                    mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                    material = materials.get(vis_spec.material)
                    link.add_visual_from_file(
                        pose=local_pose,
                        filename=mesh_path.as_posix(),
                        scale=tuple(mesh_spec.scale),
                        material=material,
                    )
            case _:
                raise ValueError(f"Visual geom type {vis_spec.type} not supported")


def add_visuals_to_actor_builder(
    builder: ActorBuilder,
    mj_spec: mj.MjSpec,
    mj_body: mj.MjsBody,
    rel_pose_to_parent: Pose,
    mj_model_dir: Path,
    materials: dict[str, RenderMaterial],
) -> None:
    visual_specs = get_visual_specs_from_body(mj_body)
    for vis_spec in visual_specs:
        tf_geom_to_body = Pose(p=tuple(vis_spec.pos), q=tuple(vis_spec.quat))
        match vis_spec.type:
            case mj.mjtGeom.mjGEOM_BOX:
                builder.add_box_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    half_size=tuple(vis_spec.size),
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_SPHERE:
                builder.add_sphere_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    radius=vis_spec.size[0],
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_CAPSULE:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_capsule_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body * CAPSULE_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_CYLINDER:
                radius = vis_spec.size[0]
                half_length = vis_spec.size[1]
                if not np.isnan(vis_spec.fromto[0]):
                    start, end = vis_spec.fromto[:3], vis_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_cylinder_visual(
                    pose=rel_pose_to_parent * tf_geom_to_body * CYLINDER_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    material=tuple(vis_spec.rgba[:3]),
                )
            case mj.mjtGeom.mjGEOM_MESH:
                mesh_spec = mj_spec.mesh(vis_spec.meshname)
                if mesh_spec is not None:
                    mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                    material = materials.get(vis_spec.material)
                    builder.add_visual_from_file(
                        pose=rel_pose_to_parent * tf_geom_to_body,
                        filename=mesh_path.as_posix(),
                        scale=tuple(mesh_spec.scale),
                        material=material,
                    )
            case _:
                raise ValueError(
                    f"Visual geom type {vis_spec.type} not supported, for geom {vis_spec.name}"
                )


def add_colliders_to_actor_builder(
    builder: ActorBuilder,
    mj_spec: mj.MjSpec,
    mj_body: mj.MjsBody,
    rel_pose_to_parent: Pose,
    mj_model_dir: Path,
) -> None:
    colliders_specs = get_collider_specs_from_body(mj_body)
    for col_spec in colliders_specs:
        tf_geom_to_body = Pose(p=tuple(col_spec.pos), q=tuple(col_spec.quat))
        physx_material: PhysxMaterial | None = None
        if col_spec.condim == 3:  # noqa: PLR2004
            physx_material = PhysxMaterial(
                static_friction=col_spec.friction[0],
                dynamic_friction=col_spec.friction[0],
                restitution=0,
            )
        elif col_spec.condim == 1:
            physx_material = PhysxMaterial(static_friction=0, dynamic_friction=0, restitution=0)

        match col_spec.type:
            case mj.mjtGeom.mjGEOM_BOX:
                builder.add_box_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    half_size=tuple(col_spec.size),
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_SPHERE:
                builder.add_sphere_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body,
                    radius=col_spec.size[0],
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_CAPSULE:
                radius = col_spec.size[0]
                half_length = col_spec.size[1]
                if not np.isnan(col_spec.fromto[0]):
                    start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_capsule_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body * CAPSULE_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_CYLINDER:
                radius = col_spec.size[0]
                half_length = col_spec.size[1]
                if not np.isnan(col_spec.fromto[0]):
                    start, end = col_spec.fromto[:3], col_spec.fromto[3:]
                    pos = (end + start) / 2
                    quat = vec_to_quat(end - start)
                    tf_geom_to_body = Pose(p=tuple(pos), q=tuple(quat))
                    half_length = np.linalg.norm(end - start).item() / 2.0
                builder.add_cylinder_collision(
                    pose=rel_pose_to_parent * tf_geom_to_body * CYLINDER_FIX_POSE,
                    radius=radius,
                    half_length=half_length,
                    density=col_spec.density,
                    material=physx_material,
                )
            case mj.mjtGeom.mjGEOM_MESH:
                mesh_spec = mj_spec.mesh(col_spec.meshname)
                if mesh_spec is not None:
                    mesh_path = mj_model_dir / mj_spec.meshdir / mesh_spec.file
                    builder.add_convex_collision_from_file(
                        pose=rel_pose_to_parent * tf_geom_to_body,
                        filename=mesh_path.as_posix(),
                        scale=tuple(mesh_spec.scale),
                        density=col_spec.density,
                        material=physx_material,
                    )
            case _:
                raise ValueError(
                    f"Collider geom type {col_spec.type} not supported, for geom {col_spec.name}"
                )

        if col_groups := THOR_COLLISION_GROUPS.get(col_spec.classname.name):
            builder.collision_groups = col_groups


def parse_textures(spec: mj.MjSpec, model_dir: Path) -> dict[str, MjcfTextureInfo]:
    textures_info: dict[str, MjcfTextureInfo] = {}
    for tex_spec in spec.textures:
        assert isinstance(tex_spec, mj.MjsTexture)
        if tex_spec.name in textures_info:
            print(f"[WARN]: texture with name {tex_spec.name} already parsed")
            continue
        textures_info[tex_spec.name] = MjcfTextureInfo(
            name=tex_spec.name,
            type=tex_spec.type,
            rgb1=tex_spec.rgb1.tolist(),
            rgb2=tex_spec.rgb2.tolist(),
            file=model_dir / tex_spec.file,
        )

    return textures_info


def parse_materials(spec: mj.MjSpec, model_dir: Path) -> dict[str, RenderMaterial]:
    textures_info = parse_textures(spec, model_dir)

    materials: dict[str, RenderMaterial] = {}
    for mat_spec in spec.materials:
        assert isinstance(mat_spec, mj.MjsMaterial)
        if mat_spec.name in materials:
            print(f"[WARN]: material with name {mat_spec.name} already parsed")
            continue

        rgba = mat_spec.rgba.copy()
        em = mat_spec.emission
        emission_arr = [rgba[0] * em, rgba[1] * em, rgba[2] * em, 1]
        render_material = RenderMaterial(
            emission=emission_arr,
            base_color=mat_spec.rgba.tolist(),
            specular=mat_spec.specular,
            roughness=1.0 - mat_spec.reflectance,
            metallic=mat_spec.shininess,
        )

        texture: RenderTexture2D | None = None
        texture_id = mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB]
        if texture_id != "":
            if texture_id in textures_info:
                texture_filepath = textures_info[texture_id].file
                if texture_filepath.exists() and texture_filepath.is_file():
                    texture = RenderTexture2D(filename=texture_filepath.as_posix())

        if texture is not None:
            render_material.base_color_texture = texture
        materials[mat_spec.name] = render_material

    return materials


def has_any_non_free_joint(root_body: mj.MjsBody) -> bool:
    has_non_free_joint = False
    stack = [root_body]
    while len(stack) > 0:
        body = stack.pop()
        if len(body.joints) > 0:
            if any([jnt.type != mj.mjtJoint.mjJNT_FREE for jnt in body.joints]):
                has_non_free_joint = True
                break
        for child in body.bodies:
            stack.append(child)

    return has_non_free_joint


class MjcfAssetArticulationLoader(Generic[SceneT]):
    def __init__(self, scene: SceneT | None = None):
        self._scene: SceneT | None = scene

        self._spec: mj.MjSpec | None = None
        self._model_dir: Path | None = None
        self._materials: dict[str, RenderMaterial] = {}

    def set_scene(self, scene: SceneT) -> MjcfAssetArticulationLoader:
        self._scene = scene
        return self

    @property
    def mjspec(self) -> mj.MjSpec | None:
        return self._spec

    def load_from_spec(
        self,
        scene_spec: mj.MjSpec,
        model_dir: Path,
        root_body_name: str | None = None,
        floating_base: bool | None = None,
        materials: dict[str, RenderMaterial] | None = None,
        is_part_of_scene: bool = False,
    ) -> ArticulationBuilder:
        """Loads an articulation from a given MjSpec for a given scene

        The given spec is assumed to be the full MjSpec for a whole scene, and the section that
        corresponds to the articulation we want to create starts at the body that has name given
        by the `root_body_name` parameter. If no root_body_name parameter is given, it's assumed
        that the whole spec corresponds for a single articulated object.

        Args:
            scene_spec: The spec for a whole scene, or for a single articulation
            model_dir: Path to the folder that contains the xml model from which the spec was parsed
            root_body_name (optional): The name of the root of the articulation
            floating_base (optional): Whether or not the base should be free to move
            materials (optional): A cache of parsed materials. If not given, will parse again here

        Returns:
            PhysxArticulation: The generated articulation object

        Raises:
            ValueError: If the root body was not found in the given spec

        """
        assert self._scene is not None, "Must assign a valid Sapien scene before loading"

        self._spec = scene_spec
        self._model_dir = model_dir

        self._materials = parse_materials(scene_spec, model_dir) if materials is None else materials

        articulation_builder = self._scene.create_articulation_builder()

        mj_root_body: mj.MjsBody | None = None
        if root_body_name is not None:
            mj_root_body = self._spec.body(root_body_name)
            if mj_root_body is None:
                raise ValueError(
                    f"Couldn't find root-body with name {root_body_name} in the given MjSpec"
                )
        else:
            mj_root_body = self._spec.worldbody.first_body()

        assert mj_root_body is not None, "Something went wrong when loading articulated object"

        if not has_any_non_free_joint(mj_root_body):
            print(
                f"[WARN]: the given model {self._spec.modelname} doesn't have any non-free joints. "
                + "You might be better off using MjcfAssetActorLoader instead"
            )

        has_freejoint = any(jnt.type == mj.mjtJoint.mjJNT_FREE for jnt in mj_root_body.joints)

        if has_freejoint:
            for jnt in mj_root_body.joints:
                assert isinstance(jnt, mj.MjsJoint)
                if jnt.type == mj.mjtJoint.mjJNT_FREE:
                    self._spec.delete(jnt)

        dummy_root_link: LinkBuilder = articulation_builder.create_link_builder(None)
        dummy_root_link.name = root_body_name or "dummy_root_0"

        self._parse_body(mj_root_body, articulation_builder, dummy_root_link, True)

        if not has_freejoint and not floating_base:
            dummy_root_link.set_joint_properties(
                type="fixed",
                limits=None,
                pose_in_parent=Pose(),
                pose_in_child=Pose(),
            )

        # If is not part of a scene, then the whole spec corresponds to this articulation, so we
        # have to parse the other stuff here (not part of the scene loader)
        if not is_part_of_scene:
            # TODO: parse constraints and other stuff here
            pass

        return articulation_builder

    def load_from_xml(
        self,
        xml_model: Path,
        floating_base: bool | None = None,
    ) -> ArticulationBuilder:
        """Loads an articulation from a given mjcf model

        The given model is assumed to correspond to a single articulation. If the mjcf model
        contains more than 1 articulation or actor, you should use MjcfSceneLoader instead. If
        the model corresponds to a single actor that doesn't contain joints, then you should use
        MjcfAssetActorLoader instead.

        Args:
            xml_model: The path to the mjcf model corresponding to an articulation
            floating_base: Whether or not the base should be free

        Returns:
            PhysxArticulation: The generated articulation object

        Raises:
            ValueError: If the mjcf model couldn't be parsed

        """
        spec = mj.MjSpec.from_file(xml_model.as_posix())
        return self.load_from_spec(spec, xml_model.parent, floating_base=floating_base)

    def _parse_body(
        self,
        mjs_body: mj.MjsBody,
        articulation_builder: ArticulationBuilder,
        parent_link_builder: LinkBuilder,
        is_root: bool = False,
    ) -> LinkBuilder:
        assert self._spec is not None, "A valid mjspec object is required for the loader"
        assert self._model_dir is not None, (
            "A valid path to the folder containing the model should be provided"
        )

        joints_info: list[MjcfJointInfo] = []
        for jnt_spec in mjs_body.joints:
            assert isinstance(jnt_spec, mj.MjsJoint)
            joints_info.append(
                MjcfJointInfo(
                    name=jnt_spec.name,
                    type=mjc_joint_type_to_str(jnt_spec.type),
                    pos=jnt_spec.pos,
                    axis=jnt_spec.axis,
                    limited=jnt_spec.limited,
                    limits=jnt_spec.range,
                    frictionloss=float(jnt_spec.frictionloss),
                    damping=float(jnt_spec.damping),
                )
            )
        if len(mjs_body.joints) == 0:
            joints_info.append(
                MjcfJointInfo(
                    name=f"{mjs_body.name}_fixed",
                    type="fixed",
                )
            )

        link_builder = parent_link_builder

        for i, jnt_info in enumerate(joints_info):
            link_builder = articulation_builder.create_link_builder(parent=link_builder)
            link_builder.set_joint_name(jnt_info.name)
            if i == len(joints_info) - 1:
                link_builder.set_name(f"{mjs_body.name}")
                has_any_visuals = any(is_visual(geom) for geom in mjs_body.geoms)
                add_colliders_to_sapien_link(
                    link_builder,
                    self._spec,
                    mjs_body.name,
                    self._model_dir,
                )
                add_visuals_to_sapien_link(
                    link_builder,
                    self._spec,
                    mjs_body.name,
                    self._model_dir,
                    self._materials,
                    colliders_are_visuals=not has_any_visuals,
                )
            else:
                link_builder.set_name(f"{mjs_body.name}_dummy_{i}")

            tf_joint2parent = np.eye(4)
            if i == 0 and not is_root:
                tf_joint2parent[:3, 3] = mjs_body.pos
                tf_joint2parent[:3, :3] = R.from_quat(
                    get_orientation(mjs_body), scalar_first=True
                ).as_matrix()

            axis = jnt_info.axis
            axis_norm = np.linalg.norm(axis)
            if axis_norm < AXIS_NORM_TOLERANCE:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis /= axis_norm
            axis_x, axis_y, axis_z = get_frame_axes(axis)

            tf_axis2joint = np.eye(4)
            tf_axis2joint[:3, 3] = jnt_info.pos
            tf_axis2joint[:3, 0] = axis_x
            tf_axis2joint[:3, 1] = axis_y
            tf_axis2joint[:3, 2] = axis_z

            tf_axis2parent = tf_joint2parent @ tf_axis2joint

            jnt_range_min, jnt_range_max = jnt_info.limits
            jnt_limited = (
                jnt_range_min < jnt_range_max
                if jnt_info.limited == mj.mjtLimited.mjLIMITED_AUTO
                else jnt_info.limited == mj.mjtLimited.mjLIMITED_TRUE
            )
            jnt_range = [jnt_range_min, jnt_range_max] if jnt_limited else [-np.inf, np.inf]

            match jnt_info.type:
                case "hinge":
                    link_builder.set_joint_properties(
                        "revolute_unwrapped" if jnt_limited else "revolute",
                        limits=[jnt_range],
                        pose_in_parent=Pose(tf_axis2parent),
                        pose_in_child=Pose(tf_axis2joint),
                        friction=jnt_info.frictionloss,
                        damping=jnt_info.damping,
                    )
                case "slide":
                    link_builder.set_joint_properties(
                        "prismatic",
                        limits=[jnt_range],
                        pose_in_parent=Pose(tf_axis2parent),
                        pose_in_child=Pose(tf_axis2joint),
                        friction=jnt_info.frictionloss,
                        damping=jnt_info.damping,
                    )
                case "fixed":
                    link_builder.set_joint_properties(
                        "fixed",
                        limits=[],
                        pose_in_parent=Pose(tf_axis2parent),
                        pose_in_child=Pose(tf_axis2joint),
                    )

        for mjs_child in mjs_body.bodies:
            assert isinstance(mjs_child, mj.MjsBody)
            self._parse_body(mjs_child, articulation_builder, link_builder, False)
        return link_builder


class MjcfAssetActorLoader(Generic[SceneT]):
    def __init__(self, scene: SceneT | None = None):
        self._scene: SceneT | None = scene

        self._spec: mj.MjSpec | None = None
        self._model_dir: Path | None = None
        self._materials: dict[str, RenderMaterial] = {}

    def set_scene(self, scene: SceneT) -> MjcfAssetActorLoader:
        self._scene = scene
        return self

    @property
    def mjspec(self) -> mj.MjSpec | None:
        return self._spec

    def load_from_spec(
        self,
        scene_spec: mj.MjSpec,
        model_dir: Path,
        root_body_name: str | None = None,
        floating_base: bool | None = None,
        materials: dict[str, RenderMaterial] | None = None,
        is_part_of_scene: bool = False,
    ) -> ActorBuilder:
        """Loads an actor from a given MjSpec for a given scene

        The given spec is assumed to be the full MjSpec for a whole scene, and the section that
        corresponds to the actor we want to create starts at the body that has name given
        by the :root_body_name: parameter. If no root_body_name parameter is given, it's assumed
        that the whole spec corresponds for a single actor.

        Args:
            scene_spec: The spec for a whole scene, or for a single actor
            model_dir: Path to the folder that contains the xml model from which the spec was parsed
            root_body_name (optional): The name of the root of the actor
            floating_base (optional): Whether or not the base should be free to move
            materials (optional): A cache of parsed materials. If not given, will parse again here

        Returns:
            sapien.Entity: The generated actor

        Raises:
            ValueError: If the root body was not found in the given spec

        """
        assert self._scene is not None, "Must assign a valid Sapien scene before loading"

        self._spec = scene_spec
        self._model_dir = model_dir
        assert self._model_dir is not None, (
            "Must provide valid model_dir to the folder containing the mjcf model"
        )

        self._materials = parse_materials(scene_spec, model_dir) if materials is None else materials

        actor_builder = self._scene.create_actor_builder()

        mj_root_body: mj.MjsBody | None = None
        if root_body_name is not None:
            mj_root_body = self._spec.body(root_body_name)
            if mj_root_body is None:
                raise ValueError(
                    f"Couldn't find root-body with name {root_body_name} in the given MjSpec"
                )
        else:
            mj_root_body = self._spec.worldbody.first_body()

        assert mj_root_body is not None, "Something went wrong when loading an actor"

        if has_any_non_free_joint(mj_root_body):
            print(
                f"[WARN]: the given body {mj_root_body.name} has some non-free joints. "
                + "You might be better off using MjcfAssetArticulationLoader instead"
            )

        has_freejoint = any([jnt.type == mj.mjtJoint.mjJNT_FREE for jnt in mj_root_body.joints])
        body_type = "dynamic" if has_freejoint or floating_base else "static"

        def make_tree_recursive(
            builder: ActorBuilder,
            spec: mj.MjSpec,
            body: mj.MjsBody,
            rel_pose: Pose,
            model_dir: Path,
            materials: dict[str, RenderMaterial],
        ) -> None:
            add_visuals_to_actor_builder(builder, spec, body, rel_pose, model_dir, materials)
            add_colliders_to_actor_builder(builder, spec, body, rel_pose, model_dir)
            builder.set_physx_body_type(body_type)

            for child in body.bodies:
                body_pose = Pose(p=tuple(child.pos), q=tuple(get_orientation(child)))
                make_tree_recursive(
                    builder, spec, child, rel_pose * body_pose, model_dir, materials
                )

        make_tree_recursive(
            actor_builder, self._spec, mj_root_body, Pose(), self._model_dir, self._materials
        )

        return actor_builder

    def load_from_xml(
        self,
        xml_model: Path,
        floating_base: bool | None = None,
    ) -> ActorBuilder:
        """Loads an actor from a given mjcf model

        The given model is assumed to correspond to a single actor. If the mjcf model constains
        more than one articulation or actor, you should use :MjcfSceneLoader: instead. If the
        model corresponds to an articulation , then you should use :MjcfAssetArticulationLoader:

        Args:
            xml_model: The path to the mjcf model corresponding to an articulation
            floating_base: Whether or not the base should be free

        Returns:
            PhysxArticulation: The generated articulation object

        Raises:
            ValueError: If the mjcf model couldn't be parsed

        """
        spec = mj.MjSpec.from_file(xml_model.as_posix())
        return self.load_from_spec(spec, xml_model.parent, floating_base=floating_base)


class MjcfSceneLoader(Generic[SceneT]):
    def __init__(self, scene: SceneT | None = None) -> None:
        self._scene: SceneT | None = scene
        self._spec: mj.MjSpec | None = None
        self._num_actors: int = 0
        self._num_articulations: int = 0

    def set_scene(self, scene: SceneT) -> MjcfSceneLoader:
        self._scene = scene
        return self

    @property
    def mjspec(self) -> mj.MjSpec | None:
        return self._spec

    def load(
        self, scene_path: Path
    ) -> tuple[dict[str, sapien.Entity], dict[str, PhysxArticulation]]:
        assert self._scene is not None, "Must assign a valid Sapien scene before loading"

        actors: dict[str, sapien.Entity] = {}
        articulations: dict[str, PhysxArticulation] = {}

        articulation_loader = MjcfAssetArticulationLoader[SceneT](self._scene)
        actor_loader = MjcfAssetActorLoader[SceneT](self._scene)

        if isinstance(self._scene, sapien.Scene):
            self._scene.add_ground(altitude=0, render=False)
        else:
            for subscene in self._scene.sub_scenes:
                subscene.add_ground(altitude=0, render=False)

        spec = mj.MjSpec.from_file(scene_path.as_posix())
        for root_body in spec.worldbody.bodies:
            assert isinstance(root_body, mj.MjsBody)

            world_pose = Pose(p=tuple(root_body.pos), q=tuple(get_orientation(root_body)))
            if has_any_non_free_joint(root_body):
                name = (
                    root_body.name
                    if root_body.name != ""
                    else f"articulation_{self._num_articulations}"
                )
                builder = articulation_loader.load_from_spec(
                    scene_spec=spec,
                    model_dir=scene_path.parent,
                    root_body_name=root_body.name,
                    is_part_of_scene=True,
                )
                builder.set_initial_pose(world_pose)
                articulations[name] = builder.build()
                articulations[name].set_name(name)
                self._num_articulations += 1
            else:
                if root_body.name == "floor":
                    continue
                name = root_body.name if root_body.name != "" else f"actor_{self._num_actors}"
                builder = actor_loader.load_from_spec(
                    scene_spec=spec,
                    model_dir=scene_path.parent,
                    root_body_name=root_body.name,
                    is_part_of_scene=True,
                )
                builder.set_name(name)
                builder.set_initial_pose(world_pose)
                actors[name] = builder.build()
                self._num_actors += 1

        for light in spec.worldbody.lights:
            assert isinstance(light, mj.MjsLight)
            match light.type:
                case mj.mjtLightType.mjLIGHT_DIRECTIONAL:
                    self._scene.add_directional_light(
                        direction=light.dir,
                        color=light.diffuse,
                        shadow=bool(light.castshadow),
                    )
                case mj.mjtLightType.mjLIGHT_POINT:
                    pass
                case _:
                    pass

        return actors, articulations
