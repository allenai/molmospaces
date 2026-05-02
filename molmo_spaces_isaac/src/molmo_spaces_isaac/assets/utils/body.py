from dataclasses import dataclass

import mujoco as mj
import numpy as np
import usdex.core
from pxr import Gf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation as R

from molmo_spaces_isaac.assets.utils.data import BaseConversionData, BodyData, Tokens, to_usd_quat
from molmo_spaces_isaac.assets.utils.geom import convert_geom, is_visual
from molmo_spaces_isaac.assets.utils.joint import convert_joint_flatten

FIX_ROTATION = R.from_rotvec([90, 0, 0], degrees=True)
COLLAPSE_NON_ARTICULATED = True


def get_fixed_quat(quat: np.ndarray) -> Gf.Quatf:
    curr_rot = R.from_quat(quat, scalar_first=True)
    new_quat = (FIX_ROTATION * curr_rot).as_quat(scalar_first=True)
    return to_usd_quat(new_quat)


def get_tf_from_body(body: mj.MjsBody, fix_rotation: bool = True) -> Gf.Transform:
    if fix_rotation:
        return Gf.Transform(
            translation=body.pos.tolist(), rotation=Gf.Rotation(get_fixed_quat(body.quat))
        )
    return Gf.Transform(translation=body.pos.tolist(), rotation=Gf.Rotation(to_usd_quat(body.quat)))


def extract_inertia(fullinertia: np.ndarray) -> tuple[Gf.Quatf, Gf.Vec3f]:
    mat = np.zeros((3, 3))
    mat[0, 0] = fullinertia[0]
    mat[1, 1] = fullinertia[1]
    mat[2, 2] = fullinertia[2]
    mat[0, 1] = fullinertia[3]
    mat[1, 0] = fullinertia[3]
    mat[0, 2] = fullinertia[4]
    mat[2, 0] = fullinertia[4]
    mat[1, 2] = fullinertia[5]
    mat[2, 1] = fullinertia[5]

    # mju_eig3 expects flattened column-major matrix
    flat_mat = mat.flatten("F")

    eigval = np.zeros(3)
    eigvec = np.zeros(9)
    quat = np.zeros(4)

    # Call mju_eig3 to get principal axes and diagonal inertia
    mj.mju_eig3(eigval, eigvec, quat, flat_mat)
    diag_inertia = Gf.Vec3f(*eigval)

    return to_usd_quat(quat), diag_inertia


@dataclass
class GeomCollapsedInfo:
    geom_spec: mj.MjsGeom
    local_tf: Gf.Transform


def convert_bodies_flatten_collapsed(
    data: BaseConversionData,
    prefix: str = "",
    opt_col_collection: Usd.CollectionAPI | None = None,
    normalize_mesh_scale: bool = False,
) -> Usd.Prim:
    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )

    root_body: mj.MjsBody = data.spec.worldbody.first_body()

    collapsed_geoms: list[GeomCollapsedInfo] = []

    def collect_collapsed_geoms(body: mj.MjsBody, parent_tf: Gf.Transform) -> None:
        local_body_tf = get_tf_from_body(body, fix_rotation=False)
        local_tf = parent_tf * local_body_tf
        for geom in body.geoms:
            assert isinstance(geom, mj.MjsGeom)
            collapsed_geoms.append(GeomCollapsedInfo(geom_spec=geom, local_tf=local_tf))
        for child_body in body.bodies:
            assert isinstance(child_body, mj.MjsBody)
            collect_collapsed_geoms(child_body, local_tf)

    collect_collapsed_geoms(root_body, Gf.Transform().SetIdentity())

    safe_name = data.name_cache.getPrimName(geo_scope, f"{prefix}{root_body.name}")
    body_xform = usdex.core.defineXform(geo_scope, safe_name)
    body_prim = body_xform.GetPrim()

    orig_names = [f"{prefix}{info.geom_spec.name}" for info in collapsed_geoms]
    safe_names = data.name_cache.getPrimNames(geo_scope, orig_names)
    for geom_info, safe_name in zip(collapsed_geoms, safe_names):
        convert_geom(
            parent=body_prim,
            geom=geom_info.geom_spec,
            safe_name=safe_name,
            data=data,
            local_tf=geom_info.local_tf,
            opt_col_collection=opt_col_collection,
            normalize_mesh_scale=normalize_mesh_scale,
        )

    body_over = data.content[Tokens.PHYSICS].OverridePrim(body_prim.GetPath())
    data.references[Tokens.PHYSICS][root_body.name] = body_over

    _ = UsdPhysics.RigidBodyAPI.Apply(body_over)

    data.bodies[root_body.name] = BodyData(
        body_spec=root_body,
        body_name=root_body.name,
        body_safename=safe_name,
        body_tf=get_tf_from_body(root_body, fix_rotation=False),
        body_prim=body_prim,
    )

    return body_prim


def convert_bodies_flatten(
    data: BaseConversionData,
    articulated: bool,
    prefix: str = "",
    opt_col_collection: Usd.CollectionAPI | None = None,
    normalize_mesh_scale: bool = False,
    asset_id: str = "",
) -> Usd.Prim:
    data.bodies_to_fix = []

    geo_scope = (
        data.content[Tokens.GEOMETRY].GetDefaultPrim().GetChild(Tokens.GEOMETRY.value).GetPrim()
    )
    root_prim = convert_body_flatten(
        body=data.spec.worldbody,
        root=geo_scope,
        parent_tf=np.eye(4),
        data=data,
        articulated=articulated,
        prefix=prefix,
        start_sleep=data.start_sleep,
        opt_col_collection=opt_col_collection,
        normalize_mesh_scale=normalize_mesh_scale,
    )

    joints_per_body: dict[str, list[mj.MjsJoint]] = {}

    def collect_mj_joints(body: mj.MjsBody, joints_map: dict[str, list[mj.MjsJoint]]) -> None:
        jnts = [jnt for jnt in body.joints if jnt.type != mj.mjtJoint.mjJNT_FREE]
        if body.name not in joints_map:
            joints_map[body.name] = []
        joints_map[body.name].extend(jnts)
        for child_body in body.bodies:
            collect_mj_joints(child_body, joints_map)

    collect_mj_joints(data.spec.worldbody.first_body(), joints_per_body)

    for b_name, joints_of_body in joints_per_body.items():
        if len(joints_of_body) < 1:
            continue
        if len(joints_of_body) > 1:
            print(f"[WARN]: Body '{b_name}' has more than one joint, will use only the first one")

        # TODO(wilbert): for now we're not setting stiffness and damping, bc the ranges from MuJoCo
        # and PhysX don't match (we get weird behavior with those values in isaac)
        convert_joint_flatten(
            joints_of_body[0],
            data,
            prefix=prefix,
            asset_id=asset_id,
            thor_parameters=data.thor_parameters,
        )

    for body_0_name, body_1_name in data.bodies_to_fix:
        body_0_data = data.bodies[body_0_name]
        body_1_data = data.bodies[body_1_name]
        if body_0_data is None or body_1_data is None:
            print(f"[WARN]: Couldn't create fixed joint for pair '({body_0_name}, {body_1_name})'")
            pass

        body_0_prim = body_0_data.body_prim
        body_1_prim = body_1_data.body_prim

        body_0_spec = body_0_data.body_spec
        body_1_spec = body_1_data.body_spec

        if body_1_spec == data.spec.worldbody:
            body_1_prim = body_1_prim.GetStage().GetDefaultPrim()

        usd_pos = Gf.Vec3d(body_0_spec.pos.tolist())
        usd_quat = Gf.Quatd(*body_0_spec.quat.tolist())

        name = data.name_cache.getPrimName(body_0_prim, UsdPhysics.Tokens.PhysicsFixedJoint)
        frame = usdex.core.JointFrame(usdex.core.JointFrame.Space.Body0, usd_pos, usd_quat)
        usdex.core.definePhysicsFixedJoint(body_0_prim, name, body_0_prim, body_1_prim, frame)

    if data.root_rotation is not None:
        new_quat = R.from_matrix(data.root_rotation).as_quat(scalar_first=True)
        new_orient = to_usd_quat(new_quat)
        usdex.core.setLocalTransform(
            root_prim.GetParent(),
            translation=Gf.Vec3d(),
            orientation=new_orient,
            scale=Gf.Vec3f(1.0),
        )

    return root_prim


def convert_body_flatten(  # noqa: PLR0915
    body: mj.MjsBody,
    root: Usd.Prim,
    parent_tf: np.ndarray,
    data: BaseConversionData,
    articulated: bool,
    prefix: str = "",
    start_sleep: bool = False,
    opt_col_collection: Usd.CollectionAPI | None = None,
    normalize_mesh_scale: bool = False,
) -> Usd.Prim:
    if body == data.spec.worldbody:
        body_prim = root
        final_tf = parent_tf
        safe_name = f"{prefix}{body.name}"
    else:
        safe_name = data.name_cache.getPrimName(root, f"{prefix}{body.name}")
        body_xform = usdex.core.defineXform(root, safe_name)
        body_prim = body_xform.GetPrim()

        # if body.name == "fr3_link5":
        #     breakpoint()

        local_tf = np.eye(4)
        local_tf[:3, :3] = R.from_quat(body.quat, scalar_first=True).as_matrix()
        local_tf[:3, 3] = body.pos

        final_tf = local_tf @ parent_tf

        final_usd_pos = Gf.Vec3d(*final_tf[:3, 3])
        final_usd_rot = to_usd_quat(R.from_matrix(final_tf[:3, :3]).as_quat(scalar_first=True))

        usdex.core.setLocalTransform(
            body_prim, translation=final_usd_pos, orientation=final_usd_rot
        )

    if body != data.spec.worldbody:
        orig_names = [f"{prefix}{geom.name}" for geom in body.geoms]
        safe_names = data.name_cache.getPrimNames(body_prim, orig_names)
        for geom, safe_name in zip(body.geoms, safe_names):
            assert isinstance(geom, mj.MjsGeom)
            convert_geom(
                body_prim, geom, safe_name, data, None, opt_col_collection, normalize_mesh_scale
            )

        body_over = data.content[Tokens.PHYSICS].OverridePrim(body_prim.GetPath())
        data.references[Tokens.PHYSICS][body.name] = body_over

        rb_api = UsdPhysics.RigidBodyAPI.Apply(body_over)
        if data.use_physx:
            body_over.AddAppliedSchema("PhysxRigidBodyAPI")
        if start_sleep:
            rb_api.GetStartsAsleepAttr().Set(True)

        if body.explicitinertial:
            mass_api = UsdPhysics.MassAPI.Apply(body_over)
            mass_api.CreateMassAttr().Set(body.mass)
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*body.ipos))
            if np.isnan(body.fullinertia[0]):
                quat = to_usd_quat(body.iquat)
                inertia = Gf.Vec3f(*body.inertia)
            else:
                quat, inertia = extract_inertia(body.fullinertia)
            if inertia != Gf.Vec3f(0, 0, 0):
                mass_api.CreatePrincipalAxesAttr().Set(quat)
                mass_api.CreateDiagonalInertiaAttr().Set(inertia)
        elif len(body.geoms) == 0 or all(is_visual(geom) for geom in body.geoms):
            mass_api = UsdPhysics.MassAPI.Apply(body_over)
            mass_api.CreateMassAttr().Set(1e-8)

        if (body.parent == data.spec.worldbody) and articulated:
            UsdPhysics.ArticulationRootAPI.Apply(body_over)

    data.bodies[body.name] = BodyData(
        body_spec=body,
        body_name=body.name,
        body_safename=safe_name,
        body_tf=Gf.Transform(Gf.Matrix4d(final_tf)),
        body_prim=body_prim,
    )

    if (body.name != "world") and (body.parent != data.spec.worldbody and not body.joints):
        data.bodies_to_fix.append((body.name, body.parent.name))

    for child_body in body.bodies:
        assert isinstance(child_body, mj.MjsBody)
        convert_body_flatten(
            body=child_body,
            root=root,
            parent_tf=final_tf,
            data=data,
            articulated=articulated,
            prefix=prefix,
            start_sleep=start_sleep,
            opt_col_collection=opt_col_collection,
            normalize_mesh_scale=normalize_mesh_scale,
        )

    return body_prim
