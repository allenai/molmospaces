import copy
import re

import mujoco as mj
import numpy as np
import usdex.core
from pxr import Gf, Sdf, UsdPhysics

from .data import AssetParameters, BaseConversionData, BodyData, Tokens


def setup_physx_parameters(usd_joint: UsdPhysics.Joint, params: AssetParameters) -> None:
    jnt_prim = usd_joint.GetPrim()
    jnt_prim.AddAppliedSchema("PhysxJointAPI")

    armature_attr = jnt_prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float)
    armature_attr.Set(params.physx.armature)
    friction_attr = jnt_prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float)
    friction_attr.Set(params.physx.jointFriction)


def get_asset_parameters(
    asset_id: str, thor_parameters: dict[str, AssetParameters]
) -> AssetParameters | None:
    params: AssetParameters | None = None
    if asset_id in thor_parameters:
        params = thor_parameters[asset_id]
    else:
        for candidate_id, candidate_params in thor_parameters.items():
            if "*" not in candidate_id:  # not a pattern to match
                continue
            if re.match(candidate_id, asset_id):
                params = copy.copy(candidate_params)
                break
    return params


def convert_joint_flatten(  # noqa: PLR0915
    joint: mj.MjsJoint,
    data: BaseConversionData,
    prefix: str = "",
    asset_id: str = "",
    thor_parameters: dict[str, AssetParameters] = {},
) -> None:
    body_0_name, body_1_name = joint.parent.name, ""
    if body_0_handle := data.spec.body(body_0_name):
        body_1_name = body_0_handle.parent.name

    body_data_0: BodyData | None = data.bodies.get(body_0_name)
    body_data_1: BodyData | None = data.bodies.get(body_1_name)

    if body_data_0 is None or body_data_1 is None:
        print(f"[WARN]: Couldn't create joint '{joint.name}'")
        return

    body_0_prim = body_data_0.body_prim
    body_1_prim = body_data_1.body_prim

    if body_data_1.body_spec == data.spec.worldbody:
        body_1_prim = body_1_prim.GetStage().GetDefaultPrim()

    safe_name = data.name_cache.getPrimName(body_0_prim, f"{prefix}{joint.name}")
    limits = get_limits(joint, data)
    axis = Gf.Vec3f(*(-joint.axis))
    frame = usdex.core.JointFrame(
        usdex.core.JointFrame.Space.Body0, Gf.Vec3d(*joint.pos), Gf.Quatd.GetIdentity()
    )

    usd_joint: UsdPhysics.Joint | None = None
    match joint.type:
        case mj.mjtJoint.mjJNT_HINGE:
            usd_joint = usdex.core.definePhysicsRevoluteJoint(
                body_0_prim, safe_name, body_0_prim, body_1_prim, frame, axis, limits[0], limits[1]
            )
            if usd_joint is not None:
                if params := get_asset_parameters(asset_id, thor_parameters):
                    drive_api = UsdPhysics.DriveAPI.Apply(usd_joint.GetPrim(), "angular")
                    drive_api.CreateTypeAttr("force")
                    drive_api.CreateStiffnessAttr(params.core.stiffness)
                    drive_api.CreateDampingAttr(params.core.damping)
                    if params.core.targetPosition is not None:
                        drive_api.CreateTargetPositionAttr(params.core.targetPosition)
                    if data.use_physx:
                        setup_physx_parameters(usd_joint, params)
                    elif data.use_newton:
                        print(
                            "[WARN]: Newton custom attributes are not used yet when generating to USD"
                        )
        case mj.mjtJoint.mjJNT_SLIDE:
            usd_joint = usdex.core.definePhysicsPrismaticJoint(
                body_0_prim, safe_name, body_0_prim, body_1_prim, frame, axis, limits[0], limits[1]
            )
            if usd_joint is not None:
                if params := get_asset_parameters(asset_id, thor_parameters):
                    drive_api = UsdPhysics.DriveAPI.Apply(usd_joint.GetPrim(), "linear")
                    drive_api.CreateTypeAttr("force")
                    drive_api.CreateStiffnessAttr(params.core.stiffness)
                    drive_api.CreateDampingAttr(params.core.damping)
                    if data.use_physx:
                        setup_physx_parameters(usd_joint, params)
                    elif data.use_newton:
                        print(
                            "[WARN]: Newton custom attributes are not used yet when generating to USD"
                        )
        case mj.mjtJoint.mjJNT_BALL:
            # only the upper limit is used for ball joints and it applies to both cone angles
            usd_joint = usdex.core.definePhysicsSphericalJoint(
                body_0_prim, safe_name, body_0_prim, body_1_prim, frame, axis, limits[1], limits[1]
            )

    if usd_joint is not None:
        data.references[Tokens.PHYSICS][joint.name] = usd_joint.GetPrim()


def is_limited(joint: mj.MjsJoint, data: BaseConversionData) -> bool:
    if joint.limited == mj.mjtLimited.mjLIMITED_TRUE:
        return True
    elif joint.limited == mj.mjtLimited.mjLIMITED_FALSE:
        return False
    elif data.spec.compiler.autolimits and joint.range[0] != joint.range[1]:
        return True
    return False


def get_limits(joint: mj.MjsJoint, data: BaseConversionData) -> tuple[float | None, float | None]:
    if not is_limited(joint, data):
        return (None, None)
    if joint.type == mj.mjtJoint.mjJNT_SLIDE or data.spec.compiler.degree:
        return tuple(joint.range)
    # for all other joint types, we need to convert the limits to degrees
    return (np.degrees(joint.range[0]), np.degrees(joint.range[1]))
