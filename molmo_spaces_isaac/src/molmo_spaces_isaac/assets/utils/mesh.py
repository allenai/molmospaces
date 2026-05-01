# NOTE(wilbert): most of this code is addapted from the newton-physics mujoco-usd converter
# https://github.com/newton-physics/mujoco-usd-converter
#
# The key difference is that they are targetting the mujoco API in the USD schema, which is not yet
# supported in isaacsim, so the models we got by using their exporter couldn't be simulated

from __future__ import annotations

from pathlib import Path
from typing import cast

import mujoco as mj
import numpy as np
import stl
import tinyobjloader
import usdex.core
from pxr import Gf, Usd, UsdGeom, Vt

from .data import BaseConversionData, Tokens


def convert_meshes(
    data: BaseConversionData, prefix: str = "", normalize_mesh_scale: bool = False
) -> None:
    if len(data.spec.meshes) < 1:
        return

    geom_library = usdex.core.addAssetLibrary(
        data.content[Tokens.CONTENTS], Tokens.GEOMETRY.value, format="usdc"
    )
    if geom_library is None:
        raise RuntimeError(
            "Couldn't create asset library for storing meshes, something went wrong T_T"
        )
    if Tokens.GEOMETRY in data.libraries:
        raise RuntimeError(
            "Already has an asset library for meshes, must be doing something twice D:"
        )

    data.libraries[Tokens.GEOMETRY] = geom_library
    data.references[Tokens.GEOMETRY] = {}

    geo_scope = data.libraries[Tokens.GEOMETRY].GetDefaultPrim()
    mesh_names = [f"{prefix}{mesh.name}" for mesh in data.spec.meshes]
    safe_names = data.name_cache.getPrimNames(geo_scope, mesh_names)

    for mesh, safe_name in zip(data.spec.meshes, safe_names):
        assert isinstance(mesh, mj.MjsMesh)
        mesh_prim = usdex.core.defineXform(geo_scope, safe_name).GetPrim()
        data.references[Tokens.GEOMETRY][mesh.name] = mesh_prim
        convert_mesh(mesh_prim, mesh, data.spec, normalize_mesh_scale)

    usdex.core.saveStage(
        data.libraries[Tokens.GEOMETRY],
        comment=f"Mesh library for {data.spec.modelname}. {data.comment}",
    )


def convert_mesh(
    prim: Usd.Prim, mesh: mj.MjsMesh, spec: mj.MjSpec, normalize_mesh_scale: bool = False
) -> UsdGeom.Mesh:
    if not mesh.file:
        raise RuntimeError(f"Mesh '{mesh.name}' doesn't have a valid file string")

    mesh_file = Path(spec.modelfiledir) / spec.meshdir / mesh.file
    if not mesh_file.is_file():
        raise RuntimeError(f"Mesh '{mesh.name}' points to file '{mesh_file}' which doesn't exist")

    mesh_prim: UsdGeom.Mesh | None = None
    if mesh.content_type == "model/stl" or mesh_file.suffix.lower() == ".stl":
        mesh_prim = create_mesh_stl(
            prim,
            mesh_file,
            cast(np.ndarray, mesh.scale) if normalize_mesh_scale else np.ones(3, dtype=np.float64),
        )
        pass
    elif mesh.content_type == "model/obj" or mesh_file.suffix.lower() == ".obj":
        mesh_prim = create_mesh_obj(
            prim,
            mesh_file,
            cast(np.ndarray, mesh.scale) if normalize_mesh_scale else np.ones(3, dtype=np.float64),
        )
    else:
        raise RuntimeError(
            f"Mesh '{mesh.name}' has invalid file format. Must be either .stl or .obj"
        )

    assert mesh_prim is not None, f"Mesh '{mesh.name}' should be valid by now"

    if not normalize_mesh_scale:
        mesh_tf = Gf.Transform()
        mesh_tf.SetScale(mesh.scale.tolist())
        usdex.core.setLocalTransform(mesh_prim.GetPrim(), mesh_tf)

    return mesh_prim


def create_mesh_stl(prim: Usd.Prim, filepath: Path, scale: np.ndarray) -> UsdGeom.Mesh:
    stl_mesh = stl.mesh.Mesh.from_file(filepath.as_posix())

    breakpoint()

    pass


def create_mesh_obj(prim: Usd.Prim, filepath: Path, scale: np.ndarray) -> UsdGeom.Mesh:
    reader = tinyobjloader.ObjReader()
    if not reader.ParseFromFile(filepath.as_posix()):
        raise RuntimeError(f"Couldn't parse .obj file '{filepath}'")

    shapes = reader.GetShapes()
    if len(shapes) == 0:
        raise RuntimeError(f"Obj file '{filepath}' doesn't contain any meshes")
    elif len(shapes) > 1:
        print(f"[WARN]: Obj file '{filepath}'contains multiple meshes, will use only the first one")

    attrib = reader.GetAttrib()
    obj_mesh = shapes[0].mesh

    vertices = attrib.vertices
    face_vertex_counts = Vt.IntArray(obj_mesh.num_face_vertices)
    face_vertex_indices = Vt.IntArray(obj_mesh.vertex_indices())

    points = [
        Gf.Vec3f(scale[0] * vertices[i], scale[1] * vertices[i + 1], scale[2] * vertices[i + 2])
        for i in range(0, len(vertices), 3)
    ]

    # Process the normals (required by usdex.core API) ---------------------------------------------
    normals: usdex.core.Vec3fPrimvarData | None = None
    orig_normals = attrib.normals
    if len(orig_normals) > 0:
        normals_data = [
            Gf.Vec3f(orig_normals[i], orig_normals[i + 1], orig_normals[i + 2])
            for i in range(0, len(orig_normals), 3)
        ]
        normals = usdex.core.Vec3fPrimvarData(
            UsdGeom.Tokens.faceVarying,
            Vt.Vec3fArray(normals_data),
            Vt.IntArray(obj_mesh.normal_indices()),
        )
        normals.index()

    # Process the uvs (required by usdex.core API) -------------------------------------------------
    uvs: usdex.core.Vec2fPrimvarData | None = None
    orig_uvs = attrib.texcoords
    if len(orig_uvs) > 0:
        uv_data = [Gf.Vec2f(orig_uvs[i], orig_uvs[i + 1]) for i in range(0, len(orig_uvs), 2)]
        uvs = usdex.core.Vec2fPrimvarData(
            UsdGeom.Tokens.faceVarying,
            Vt.Vec2fArray(uv_data),
            Vt.IntArray(obj_mesh.texcoord_indices()),
        )
        uvs.index()

    usd_mesh = usdex.core.definePolyMesh(
        prim.GetParent(),
        prim.GetName(),
        faceVertexCounts=face_vertex_counts,
        faceVertexIndices=face_vertex_indices,
        points=Vt.Vec3fArray(points),
        normals=normals,
        uvs=uvs,
    )

    if usd_mesh is None:
        raise RuntimeError(f"Couldn't convert obj mesh from '{filepath}'")

    return usd_mesh
