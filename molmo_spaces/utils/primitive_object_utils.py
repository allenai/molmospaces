"""Helpers for procedurally-created primitive bodies.

A "primitive" here is a MuJoCo body that is built at ``MjSpec`` time via
``add_body``/``add_freejoint``/``add_geom`` rather than loaded from an XML
asset. The JSON benchmark pipeline captures these via
``PrimitiveObjectSpec`` (see ``molmo_spaces/evaluation/benchmark_schema.py``)
so the eval-time loader can replay the exact same construction sequence.

This module provides the single canonical mapping between
``PrimitiveObjectSpec`` and ``MjSpec`` add-body calls, shared by the
datagen-time samplers (block support, mug-ball pick) and
``JsonEvalTaskSampler``.
"""

from typing import Any

import mujoco
from mujoco import MjSpec, mjtGeom

from molmo_spaces.evaluation.benchmark_schema import (
    PrimitiveGeomSpec,
    PrimitiveGeomType,
    PrimitiveObjectSpec,
)

# Canonical string → MuJoCo enum mapping for the geom types we serialize.
# Keep in sync with ``PrimitiveGeomType`` in benchmark_schema.py.
GEOM_TYPE_STR_TO_MJ: dict[PrimitiveGeomType, int] = {
    "box": mjtGeom.mjGEOM_BOX,
    "sphere": mjtGeom.mjGEOM_SPHERE,
    "capsule": mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mjtGeom.mjGEOM_ELLIPSOID,
}


def add_primitive_to_spec(spec: MjSpec, primitive: PrimitiveObjectSpec):
    """Replay a PrimitiveObjectSpec onto ``spec.worldbody``.

    Creates the body, optionally adds a freejoint, and adds every geom in
    the spec. Returns the newly-added ``MjsBody``. Optional ``None``-valued
    fields on the spec are intentionally not forwarded so MuJoCo's own
    defaults apply (matching the behavior of the original hand-written
    ``add_geom`` calls this replaces).
    """
    body = spec.worldbody.add_body(name=primitive.body_name, pos=list(primitive.initial_pos))

    if primitive.add_freejoint:
        if primitive.freejoint_name is None and primitive.freejoint_damping is None:
            # Matches the ``body.add_freejoint()`` call used by the
            # original sampler code — no name, no damping.
            body.add_freejoint()
        else:
            joint_kwargs: dict[str, Any] = {"type": mujoco.mjtJoint.mjJNT_FREE}
            if primitive.freejoint_name is not None:
                joint_kwargs["name"] = primitive.freejoint_name
            if primitive.freejoint_damping is not None:
                joint_kwargs["damping"] = primitive.freejoint_damping
            body.add_joint(**joint_kwargs)

    for geom in primitive.geoms:
        _add_geom_from_spec(body, primitive.body_name, geom)

    return body


def _add_geom_from_spec(body, body_name: str, geom: PrimitiveGeomSpec) -> None:
    """Add a single geom from a PrimitiveGeomSpec onto an MjsBody."""
    geom_kwargs: dict[str, Any] = {
        "name": f"{body_name}{geom.name_suffix}",
        "type": GEOM_TYPE_STR_TO_MJ[geom.geom_type],
        "size": list(geom.size),
        "rgba": list(geom.rgba),
    }
    if geom.contype is not None:
        geom_kwargs["contype"] = geom.contype
    if geom.conaffinity is not None:
        geom_kwargs["conaffinity"] = geom.conaffinity
    if geom.friction is not None:
        geom_kwargs["friction"] = list(geom.friction)
    if geom.mass is not None:
        geom_kwargs["mass"] = geom.mass
    body.add_geom(**geom_kwargs)


def primitive_spec_to_config_dict(primitive: PrimitiveObjectSpec) -> dict:
    """Convert a PrimitiveObjectSpec to the plain-dict form stored in
    ``BaseMujocoTaskConfig.primitive_objects`` (picklable with no
    evaluation-schema dependency)."""
    return primitive.model_dump()


def primitive_spec_from_config_dict(data: dict | PrimitiveObjectSpec) -> PrimitiveObjectSpec:
    """Inverse of ``primitive_spec_to_config_dict``. Accepts either an
    already-validated spec or the plain-dict form for convenience."""
    if isinstance(data, PrimitiveObjectSpec):
        return data
    return PrimitiveObjectSpec.model_validate(data)


def primitive_metadata_entry(primitive: PrimitiveObjectSpec) -> dict:
    """Build a synthetic ``scene_metadata['objects'][body_name]`` entry for
    a primitive body.

    Primitives have no Objaverse annotation, but downstream code (notably
    the learned-policy prompt sampler via
    ``ObjectMeta.get_target_object_uid``) assumes every pickup target has
    an entry keyed by body name with at least an ``asset_id``. We use the
    body name as a synthetic asset_id and the first underscore-separated
    token as the category — matching the ``target_name.split('_')[0]``
    fallback that the prompt sampler uses when ``short_descriptions``
    returns empty for an unknown asset_id.
    """
    return {
        "asset_id": primitive.body_name,
        "category": primitive.body_name.split("_")[0],
        "object_enum": "primitive_object",
        "is_static": False,
        "boundingBox": {},
    }
