"""Download and convert the Microsoft Rocketbox avatar library
(github.com/microsoft/Microsoft-Rocketbox, MIT license) into MJCF assets usable
by molmospaces' PickWithHumanRB task sampler.

Output is written to ``ROBOTS_DIR / "humans_rocketbox_static"`` (gitignored, like every other
locally-cached asset category under ``assets/``) rather than committed to the
repo -- there's no Git LFS here and no precedent for committing binary mesh/
texture data at this scale, so each machine that wants avatars regenerates
them locally by running this script.

Requires a separate, throwaway environment (NOT part of molmospaces'
`pyproject.toml` -- these are only needed for this one-time conversion):

    python3.11 -m venv /tmp/rocketbox_convert_venv
    /tmp/rocketbox_convert_venv/bin/pip install ufbx open3d numpy pillow mujoco

Usage:
    # 1. Download all three categories (Adults, Children, Professions -- ~115
    #    characters total, ~10GB+ with full-res textures).
    python convert_human_rb.py download

    # 2a. Convert to static mannequins (ROBOTS_DIR/humans_rocketbox_static) -- single mesh body
    #     with a free joint, no skeleton. Runs each avatar in its own
    #     subprocess -- MjSpec.to_xml()'s cleanup segfaults on process exit in
    #     this ufbx/mujoco combination (verified harmless: output files are
    #     already fully written by then), so isolating per-avatar keeps one
    #     crash from losing the whole batch.
    /tmp/rocketbox_convert_venv/bin/python convert_human_rb.py convert --all

    # 2b. Convert to articulated ragdolls (ROBOTS_DIR/humans_rocketbox_articulated) --
    #     a ~20-body chain (see CURATED_BONES) connected by ball joints,
    #     vertices assigned to bodies via the rig's own skin weights. Same
    #     per-avatar subprocess isolation as above, PLUS every avatar's own
    #     conversion further splits ufbx access into its own isolated
    #     subprocess (see _extract_articulated_data's docstring).
    /tmp/rocketbox_convert_venv/bin/python convert_human_rb.py articulate --all

    # 2c. Convert to skinned characters (ROBOTS_DIR/humans_rocketbox_skinned) -- the same
    #     ~20-body chain as `articulate`, but the mesh follows it as MuJoCo
    #     `<skin>` assets (linear blend skinning off the rig's own weights)
    #     rather than being cut into rigid per-bone chunks, so joints deform
    #     instead of tearing open. Same per-avatar subprocess isolation.
    /tmp/rocketbox_convert_venv/bin/python convert_human_rb.py skin --all

    # (internal, used by --all above): convert a single avatar in-process
    /tmp/rocketbox_convert_venv/bin/python convert_human_rb.py convert --uid Female_Adult_01
    /tmp/rocketbox_convert_venv/bin/python convert_human_rb.py articulate --uid Female_Adult_01
    /tmp/rocketbox_convert_venv/bin/python convert_human_rb.py skin --uid Female_Adult_01
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

AVATAR_CATEGORIES = ["Adults", "Children", "Professions"]

TEXTURE_MAX_DIM = 512
ROCKETBOX_REPO_URL = "https://github.com/microsoft/Microsoft-Rocketbox.git"


def default_rocketbox_dir() -> Path:
    return Path.home() / ".cache" / "molmospaces" / "rocketbox_src"


def default_out_dir() -> Path:
    from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR

    return ROBOTS_DIR / "humans_rocketbox_static"


def default_articulated_out_dir() -> Path:
    from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR

    return ROBOTS_DIR / "humans_rocketbox_articulated"


def default_skinned_out_dir() -> Path:
    from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR

    return ROBOTS_DIR / "humans_rocketbox_skinned"


def discover_avatars(rocketbox_dir: Path) -> dict:
    """Returns {uid: category} for every avatar folder that has an Export/<uid>.fbx."""
    avatars = {}
    for category in AVATAR_CATEGORIES:
        category_dir = rocketbox_dir / "Assets" / "Avatars" / category
        if not category_dir.is_dir():
            continue
        for d in sorted(category_dir.iterdir()):
            if d.is_dir() and (d / "Export" / f"{d.name}.fbx").exists():
                avatars[d.name] = category
    return avatars


def cmd_download(rocketbox_dir: Path) -> None:
    if not (rocketbox_dir / ".git").exists():
        rocketbox_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                ROCKETBOX_REPO_URL,
                str(rocketbox_dir),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "sparse-checkout", "set", "Assets/Avatars"],
        cwd=rocketbox_dir,
        check=True,
    )
    avatars = discover_avatars(rocketbox_dir)
    print(f"Rocketbox source ready at {rocketbox_dir} ({len(avatars)} avatars found)")


# ---------------------------------------------------------------------------
# Conversion (imports ufbx/open3d/mujoco lazily -- only needed for `convert`,
# and only present in the throwaway conversion venv, not the main project env)
# ---------------------------------------------------------------------------


def _round_obj_precision(path: Path, decimals: int) -> None:
    """Shrink OBJ file size by rounding vertex/normal/uv floats (Open3D writes full
    float64 precision, far more digits than useful at this mesh's real-world scale)."""
    fmt = f"{{:.{decimals}f}}"
    out_lines = []
    for line in path.read_text().splitlines():
        parts = line.split(" ")
        if parts and parts[0] in ("v", "vn", "vt"):
            head, nums = parts[0], parts[1:]
            rounded = [fmt.format(float(n)) for n in nums]
            out_lines.append(head + " " + " ".join(rounded))
        else:
            out_lines.append(line)
    path.write_text("\n".join(out_lines) + "\n")


def _matrix_cols(m):
    import numpy as np

    return (
        np.array([m.c0.x, m.c0.y, m.c0.z]),
        np.array([m.c1.x, m.c1.y, m.c1.z]),
        np.array([m.c2.x, m.c2.y, m.c2.z]),
        np.array([m.c3.x, m.c3.y, m.c3.z]),
    )


def _apply_matrix(points, m, translate=True):
    c0, c1, c2, c3 = _matrix_cols(m)
    out = points[:, 0:1] * c0 + points[:, 1:2] * c1 + points[:, 2:3] * c2
    if translate:
        out = out + c3
    return out


def _yup_to_zup(points):
    import numpy as np

    # (x, y, z)_Yup -> (x, -z, y)_Zup
    out = np.empty_like(points)
    out[:, 0] = points[:, 0]
    out[:, 1] = -points[:, 2]
    out[:, 2] = points[:, 1]
    return out


# Curated subset of Rocketbox's 81-bone Biped rig (which also includes ~50
# finger/facial bones for animation) -- a 20-body ragdoll chain covering the
# major limb segments is enough to be posable/articulated without the
# per-finger/per-eyebrow joints that would make the physics fragile for a
# scene-population avatar. (name, parent_name); parent=None is the free-jointed
# root. Every vertex still gets assigned to *some* body (see
# _nearest_curated_ancestor) -- fingers/face just inherit their parent's rigid
# segment instead of getting their own joint.
CURATED_BONES = [
    ("Bip01 Pelvis", None),
    ("Bip01 Spine", "Bip01 Pelvis"),
    ("Bip01 Spine1", "Bip01 Spine"),
    ("Bip01 Spine2", "Bip01 Spine1"),
    ("Bip01 Neck", "Bip01 Spine2"),
    ("Bip01 Head", "Bip01 Neck"),
    ("Bip01 L Clavicle", "Bip01 Spine2"),
    ("Bip01 L UpperArm", "Bip01 L Clavicle"),
    ("Bip01 L Forearm", "Bip01 L UpperArm"),
    ("Bip01 L Hand", "Bip01 L Forearm"),
    ("Bip01 R Clavicle", "Bip01 Spine2"),
    ("Bip01 R UpperArm", "Bip01 R Clavicle"),
    ("Bip01 R Forearm", "Bip01 R UpperArm"),
    ("Bip01 R Hand", "Bip01 R Forearm"),
    ("Bip01 L Thigh", "Bip01 Pelvis"),
    ("Bip01 L Calf", "Bip01 L Thigh"),
    ("Bip01 L Foot", "Bip01 L Calf"),
    ("Bip01 R Thigh", "Bip01 Pelvis"),
    ("Bip01 R Calf", "Bip01 R Thigh"),
    ("Bip01 R Foot", "Bip01 R Calf"),
]


def convert_one(uid: str, category: str, rocketbox_dir: Path, out_dir: Path) -> dict:
    import mujoco as mj
    import numpy as np
    import open3d as o3d
    import ufbx
    from PIL import Image

    src_dir = rocketbox_dir / "Assets" / "Avatars" / category / uid
    fbx_path = src_dir / "Export" / f"{uid}.fbx"
    textures_dir = src_dir / "Textures"

    scene = ufbx.load_file(str(fbx_path))
    if len(scene.meshes) != 1:
        print(f"  WARNING {uid}: expected 1 mesh, found {len(scene.meshes)}; using first")
    mesh = scene.meshes[0]
    node = mesh.instances[0]
    g2w = node.geometry_to_world
    unit_scale = scene.settings.unit_meters

    pos_vals = np.array(mesh.vertex_position.values, dtype=np.float64)
    pos_idx = np.array(mesh.vertex_position.indices, dtype=np.int64)
    uv_vals = np.array(mesh.vertex_uv.values, dtype=np.float64)
    uv_idx = np.array(mesh.vertex_uv.indices, dtype=np.int64)
    face_material = np.array(list(mesh.face_material), dtype=np.int64)
    num_indices = mesh.num_indices
    assert num_indices == mesh.num_faces * 3, f"{uid}: mesh is not fully triangulated"

    # World-space, meters, Z-up. (FBX per-corner normals are dropped -- ufbx never
    # dedups them, so they can't be welded; Open3D recomputes clean vertex normals
    # from the deduped topology instead, see below.)
    world_pos = _apply_matrix(pos_vals, g2w, translate=True) * unit_scale
    world_pos = _yup_to_zup(world_pos)

    # Center XY on the bounding-box center, put feet (min z) at z=0.
    bbox_min = world_pos.min(axis=0)
    bbox_max = world_pos.max(axis=0)
    center_xy = (bbox_min[:2] + bbox_max[:2]) / 2.0
    world_pos[:, 0] -= center_xy[0]
    world_pos[:, 1] -= center_xy[1]
    world_pos[:, 2] -= bbox_min[2]
    height = bbox_max[2] - bbox_min[2]
    half_width = max(bbox_max[0] - center_xy[0], center_xy[0] - bbox_min[0])

    corner_face_material = np.repeat(face_material, 3)
    triangles = np.arange(num_indices).reshape(-1, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = mj.MjSpec()
    spec.modelname = uid
    root_body = spec.worldbody.add_body(name=uid)
    root_body.add_joint(name=f"{uid}_jntfree", type=mj.mjtJoint.mjJNT_FREE, damping=0.5)

    visual_default = spec.add_default("avatar_visual", spec.default)
    visual_default.geom.contype = 0
    visual_default.geom.conaffinity = 0
    visual_default.geom.group = 0
    visual_default.geom.mass = 1e-8

    used_materials = sorted(set(face_material.tolist()))
    for mat_idx in used_materials:
        material = scene.materials[mat_idx]
        mat_name = material.name or f"mat{mat_idx}"
        face_mask = corner_face_material == mat_idx
        tri_mask = face_mask.reshape(-1, 3).all(axis=1)
        if not tri_mask.any():
            continue
        tri_subset = triangles[tri_mask]
        used_corners = np.unique(tri_subset.reshape(-1))
        # Dedup by (position, uv) only, not normal -- ufbx gives every triangle corner
        # its own normal slot by construction (never shared, even when numerically
        # identical), so including it would make every corner "unique" and defeat the
        # dedup entirely. Position+uv sharing is real and collapses the per-corner
        # "unwelded" OBJ vertex count close to the true topology (e.g. one avatar:
        # 15954 raw corners -> 3453 deduped) -- normals are recomputed from the
        # deduped geometry below instead of carrying over FBX's per-corner ones.
        keys = np.stack([pos_idx[used_corners], uv_idx[used_corners]], axis=1)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        remap = -np.ones(num_indices, dtype=np.int64)
        remap[used_corners] = inverse
        sub_triangles = remap[tri_subset]
        sub_pos = world_pos[unique_keys[:, 0]]
        sub_uv = uv_vals[unique_keys[:, 1]].copy()
        sub_uv[:, 1] = 1.0 - sub_uv[:, 1]

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(sub_pos)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(sub_triangles)
        o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(sub_uv[sub_triangles.reshape(-1)])
        o3d_mesh.compute_vertex_normals()

        texture_png = None
        diffuse_tex = None
        for tex_link in material.textures:
            if str(tex_link.material_prop) == "DiffuseColor" and tex_link.texture:
                diffuse_tex = tex_link.texture
                break
        if diffuse_tex is not None:
            basename = Path(diffuse_tex.relative_filename.replace("\\", "/")).name
            src_tex_path = textures_dir / basename
            if src_tex_path.exists():
                texture_png = out_dir / f"{uid}_{mat_name}.png"
                img = Image.open(src_tex_path).convert("RGBA")
                img.thumbnail((TEXTURE_MAX_DIM, TEXTURE_MAX_DIM), Image.LANCZOS)
                img.save(texture_png, optimize=True)
                o3d_mesh.textures = [o3d.io.read_image(str(texture_png))]
            else:
                print(f"  WARNING {uid}: texture {basename} not found in {textures_dir}")

        mesh_file = f"{uid}_{mat_name}.obj"
        o3d.io.write_triangle_mesh(str(out_dir / mesh_file), o3d_mesh)
        _round_obj_precision(out_dir / mesh_file, decimals=5)

        mesh_asset_name = f"{uid}_{mat_name}_mesh"
        spec.add_mesh(name=mesh_asset_name, file=mesh_file)

        vis_material_name = None
        if texture_png is not None:
            vis_texture_name = f"{uid}_{mat_name}_tex"
            spec.add_texture(
                name=vis_texture_name,
                file=texture_png.name,
                type=mj.mjtTexture.mjTEXTURE_2D,
            )
            vis_material_name = f"{uid}_{mat_name}_material"
            mat_spec = spec.add_material(name=vis_material_name)
            mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = vis_texture_name

        geom = root_body.add_geom(
            name=f"{uid}_{mat_name}_visual",
            type=mj.mjtGeom.mjGEOM_MESH,
            meshname=mesh_asset_name,
        )
        geom.classname = visual_default
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 0
        geom.mass = 1e-8
        if vis_material_name is not None:
            geom.material = vis_material_name

    # Simple capsule collision proxy sized from the bounding box -- full character
    # meshes are too expensive to collide against, and Rocketbox ships no pre-baked
    # simplified collider the way some object libraries do.
    capsule_radius = max(0.12, min(0.22, half_width * 0.55))
    capsule_half_height = max(0.05, height / 2.0 - capsule_radius)
    root_body.add_geom(
        name=f"{uid}_collision",
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        pos=[0, 0, height / 2.0],
        size=[capsule_radius, capsule_half_height, 0],
        contype=8,
        conaffinity=15,
        group=4,
        rgba=[0.5, 0.5, 0.5, 0.0],
    )

    xml_path = out_dir / f"{uid}.xml"
    prev_cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        xml_path.write_text(spec.to_xml())
    finally:
        os.chdir(prev_cwd)

    return {"uid": uid, "height": float(height), "half_width": float(half_width)}


def _bone_key(bone_name: str) -> str:
    # "Bip01 L Thigh" -> "L_Thigh" -- strip whichever BipNN prefix this rig
    # uses (Adults/Professions use Bip01, Children use Bip02) and normalize
    # remaining spaces.
    import re

    return re.sub(r"^Bip\d+ ?", "", bone_name).replace(" ", "_")


_EXTRACT_ARTICULATED_SCRIPT = """
import json
import numpy as np
import ufbx

CURATED_BONES = {curated_bones!r}
curated_names = {{name for name, _ in CURATED_BONES}}
fbx_path = {fbx_path!r}
textures_dir = {textures_dir!r}
out_npz = {out_npz!r}

scene = ufbx.load_file(fbx_path)
mesh = scene.meshes[0]
node = mesh.instances[0]
g2w = node.geometry_to_world
unit_scale = scene.settings.unit_meters


def apply_matrix(points, m, translate=True):
    c0 = np.array([m.c0.x, m.c0.y, m.c0.z])
    c1 = np.array([m.c1.x, m.c1.y, m.c1.z])
    c2 = np.array([m.c2.x, m.c2.y, m.c2.z])
    c3 = np.array([m.c3.x, m.c3.y, m.c3.z])
    out = points[:, 0:1] * c0 + points[:, 1:2] * c1 + points[:, 2:3] * c2
    if translate:
        out = out + c3
    return out


def yup_to_zup(points):
    out = np.empty_like(points)
    out[:, 0] = points[:, 0]
    out[:, 1] = -points[:, 2]
    out[:, 2] = points[:, 1]
    return out


# Ancestor resolution via .children-walk (see convert_one_articulated's
# docstring for why this whole extraction lives in an isolated subprocess).
# nodes_by_name is built *inline* during this same walk rather than via a
# separate `scene.nodes` pass afterward -- empirically, doing a .children
# walk and then separately re-iterating scene.nodes corrupts this ufbx
# binding's internal state (confirmed via bisection: a second collection
# pass over the same node set reliably crashes shortly after, regardless of
# what that pass does), even though either one alone is safe.
parent_of = {{}}
nodes_by_name = {{}}
def walk(n, parent_name):
    if n.name:
        parent_of[n.name] = parent_name
        nodes_by_name[n.name] = n
    for c in n.children:
        walk(c, n.name)
walk(scene.root_node, None)

# CURATED_BONES is written against the "Bip01" naming Adults/Professions use;
# Children's rig is otherwise identical but under a "Bip02" prefix. Detect
# whichever prefix this rig actually uses (from any bone name matching
# "BipNN ...") and translate curated names into that space for lookups --
# results are translated back to the canonical "Bip01..." form before output,
# so downstream code (convert_one_articulated) never needs to know the rig's
# real prefix.
import re as _re
_prefix_match = None
for _name in parent_of:
    _m = _re.match(r"^(Bip\\d+) ", _name)
    if _m:
        _prefix_match = _m.group(1)
        break
_actual_prefix = _prefix_match or "Bip01"
def to_actual(name):
    return name.replace("Bip01", _actual_prefix, 1) if name else name
def to_canonical(name):
    return name.replace(_actual_prefix, "Bip01", 1) if name else name
curated_names = {{to_actual(name) for name in curated_names}}

def resolve(name):
    seen = 0
    while name is not None and name not in curated_names and seen < 100:
        name = parent_of.get(name)
        seen += 1
    return to_canonical(name) if name in curated_names else CURATED_BONES[0][0]

all_resolved = {{name: resolve(name) for name in parent_of}}

bone_world_pos_raw = {{}}
for bone_name, _ in CURATED_BONES:
    bn = nodes_by_name[to_actual(bone_name)]
    m = bn.node_to_world
    p = np.array([[m.c3.x, m.c3.y, m.c3.z]]) * unit_scale
    bone_world_pos_raw[bone_name] = yup_to_zup(p)[0]

if not mesh.skin_deformers:
    raise SystemExit("NO_SKIN_DEFORMER")
skin = mesh.skin_deformers[0]
num_verts = mesh.num_vertices
curated_index = {{name: i for i, (name, _) in enumerate(CURATED_BONES)}}
# Two views of the same clusters, both built in this single pass over them (see
# the nodes_by_name comment above on why ufbx access is kept to one pass):
#   - argmax bone per vertex, for the rigid `articulate` partition
#   - the full per-vertex weight vector collapsed onto curated bones, for `skin`
#     (a vertex weighted 0.6/0.4 across two finger bones that both resolve to
#     Hand contributes 1.0 to Hand -- collapsing sums, it doesn't pick)
best_weight = np.zeros(num_verts)
best_bone_name = [CURATED_BONES[0][0]] * num_verts
curated_weights = np.zeros((num_verts, len(CURATED_BONES)), dtype=np.float64)
for cluster in skin.clusters:
    bone_node_name = cluster.bone_node.name
    verts = np.asarray(list(cluster.vertices), dtype=np.int64)
    weights = np.asarray(list(cluster.weights), dtype=np.float64)
    curated_col = curated_index[all_resolved.get(bone_node_name, CURATED_BONES[0][0])]
    np.add.at(curated_weights, (verts, curated_col), weights)
    for vi, w in zip(verts.tolist(), weights.tolist()):
        if w > best_weight[vi]:
            best_weight[vi] = w
            best_bone_name[vi] = bone_node_name
vertex_curated_bone = [all_resolved.get(n, CURATED_BONES[0][0]) for n in best_bone_name]

pos_vals = np.array(mesh.vertex_position.values, dtype=np.float64)
pos_idx = np.array(mesh.vertex_position.indices, dtype=np.int64)
uv_vals = np.array(mesh.vertex_uv.values, dtype=np.float64)
uv_idx = np.array(mesh.vertex_uv.indices, dtype=np.int64)
face_material = np.array(list(mesh.face_material), dtype=np.int64)

world_pos = apply_matrix(pos_vals, g2w, translate=True) * unit_scale
world_pos = yup_to_zup(world_pos)

bbox_min = world_pos.min(axis=0)
bbox_max = world_pos.max(axis=0)
center_xy = (bbox_min[:2] + bbox_max[:2]) / 2.0
world_pos[:, 0] -= center_xy[0]
world_pos[:, 1] -= center_xy[1]
world_pos[:, 2] -= bbox_min[2]
height = float(bbox_max[2] - bbox_min[2])
half_width = float(max(bbox_max[0] - center_xy[0], center_xy[0] - bbox_min[0]))

bone_world_pos = {{}}
for name, p in bone_world_pos_raw.items():
    p = p.copy()
    p[0] -= center_xy[0]
    p[1] -= center_xy[1]
    p[2] -= bbox_min[2]
    bone_world_pos[name] = p.tolist()

materials = []
for mi, material in enumerate(scene.materials):
    diffuse_path = None
    for tex_link in material.textures:
        if str(tex_link.material_prop) == "DiffuseColor" and tex_link.texture:
            import os as _os
            basename = _os.path.basename(tex_link.texture.relative_filename.replace(chr(92), "/"))
            candidate = _os.path.join(textures_dir, basename)
            if _os.path.exists(candidate):
                diffuse_path = candidate
            break
    materials.append({{"index": mi, "name": material.name or ("mat" + str(mi)), "diffuse_path": diffuse_path}})

np.savez(
    out_npz,
    pos_idx=pos_idx,
    uv_vals=uv_vals,
    uv_idx=uv_idx,
    face_material=face_material,
    world_pos=world_pos,
    vertex_curated_bone=np.array(vertex_curated_bone, dtype="<U64"),
    curated_weights=curated_weights.astype(np.float32),
)
print(json.dumps({{
    "materials": materials,
    "bone_world_pos": bone_world_pos,
    "height": height,
    "half_width": half_width,
}}))
"""


def _extract_articulated_data(uid: str, fbx_path: Path, textures_dir: Path, out_npz: Path) -> dict:
    """Runs ALL ufbx access (hierarchy walk, skin weights, mesh geometry,
    material/texture lookup) in one fresh, isolated subprocess, saving the
    numeric arrays to out_npz and returning the small metadata as a dict.

    Empirically, this ufbx Python binding (v0.0.5) corrupts its own state when
    a single process does enough combined Node-hierarchy / skin-cluster /
    mesh-geometry access -- confirmed via bisection that resolving even a
    handful of bone ancestors before later reading mesh.vertex_position
    reliably segfaults *mid-computation* (not just on exit, unlike the
    static convert_one() pipeline's harmless exit-time crash). Isolating the
    entire ufbx-touching phase in its own subprocess, with all downstream
    Open3D/MJCF work happening in a separate process that never imports ufbx,
    sidesteps the corruption entirely rather than trying to avoid whatever
    specific access pattern triggers it.
    """
    import json
    import tempfile

    script = _EXTRACT_ARTICULATED_SCRIPT.format(
        curated_bones=CURATED_BONES,
        fbx_path=str(fbx_path),
        textures_dir=str(textures_dir),
        out_npz=str(out_npz),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        # NOT check=True: even this isolated extraction's scene cleanup can
        # segfault on process exit (harmless -- stdout/out_npz are fully
        # written and flushed before it happens). Only missing output means
        # a real failure.
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    finally:
        os.unlink(script_path)
    if "NO_SKIN_DEFORMER" in (result.stderr or ""):
        raise ValueError(f"{uid}: no skin deformer found, cannot articulate")
    if not result.stdout.strip() or not out_npz.exists():
        raise RuntimeError(f"Extraction failed for {uid} (stderr: {result.stderr})")
    return json.loads(result.stdout)


def _load_extracted(uid: str, category: str, rocketbox_dir: Path, out_dir: Path):
    """Run the isolated ufbx extraction and return (arrays, metadata).

    Shared by the `articulate` and `skin` builders -- both need the same rig +
    geometry + skin-weight extraction and differ only in what they build from it.
    """
    import numpy as np

    src_dir = rocketbox_dir / "Assets" / "Avatars" / category / uid
    fbx_path = src_dir / "Export" / f"{uid}.fbx"
    textures_dir = src_dir / "Textures"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"_{uid}_extract.npz"
    metadata = _extract_articulated_data(uid, fbx_path, textures_dir, npz_path)
    try:
        extracted = np.load(npz_path, allow_pickle=False)
        arrays = {k: extracted[k] for k in extracted.files}
    finally:
        npz_path.unlink(missing_ok=True)
    return arrays, metadata


def _add_bone_bodies(spec, uid: str, bone_world_pos) -> dict:
    """Build the CURATED_BONES chain: free-jointed root, ball joints below.

    Every body is created with position only, so each one's rest orientation is
    the identity -- which is also what makes the skin bind pose trivial (see
    convert_one_skinned's bindquat).
    """
    import mujoco as mj

    bodies = {}
    for bone_name, parent_name in CURATED_BONES:
        key = _bone_key(bone_name)
        if parent_name is None:
            body = spec.worldbody.add_body(name=f"{uid}_{key}", pos=bone_world_pos[bone_name])
            body.add_joint(name=f"{uid}_{key}_jntfree", type=mj.mjtJoint.mjJNT_FREE, damping=0.5)
        else:
            parent_body = bodies[parent_name]
            rel_pos = bone_world_pos[bone_name] - bone_world_pos[parent_name]
            body = parent_body.add_body(name=f"{uid}_{key}", pos=rel_pos)
            body.add_joint(name=f"{uid}_{key}_jnt", type=mj.mjtJoint.mjJNT_BALL, damping=0.3)
        bodies[bone_name] = body
    return bodies


def _add_inertia_fallback_geoms(bodies: dict, uid: str) -> None:
    """Every curated bone gets a tiny invisible fallback sphere, unconditionally
    -- not just bones with zero mesh geoms (e.g. Neck, if every nearby vertex's
    dominant weight resolved to Head or Spine2 instead). A body whose only "real"
    geometry is a degenerate sliver (near-zero surface area, even under shell
    inertia) can still end up with an effectively singular inertia tensor and hit
    MuJoCo's rejection ("mass and inertia of moving bodies must be larger than
    mjMINVAL") despite having a nonzero-mass geom -- a well-formed sphere's
    inertia is never degenerate, so adding one to every body (redundant, harmless
    where geometry is already fine) guarantees every body clears mjMINVAL. In the
    skinned build these spheres are a body's *only* geometry, since a skin is an
    asset rather than a geom and contributes no inertia at all.
    """
    import mujoco as mj

    for bone_name, _ in CURATED_BONES:
        bodies[bone_name].add_geom(
            name=f"{uid}_{_bone_key(bone_name)}_fallback",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            contype=0,
            conaffinity=0,
            group=0,
            mass=1e-8,
            rgba=[0, 0, 0, 0],
        )


def _add_pelvis_collision(bodies: dict, uid: str, bone_world_pos, height, half_width) -> None:
    """Single collision proxy on the pelvis (root) -- a per-limb-segment collider
    would let a ~20-body chain self-collide and destabilize; a modest torso
    capsule is enough for PickWithHumanRBTaskSampler's place_object_near contact
    check, which only needs *some* geom on the kinematic tree."""
    import mujoco as mj
    import numpy as np

    capsule_radius = max(0.12, min(0.22, half_width * 0.55))
    capsule_half_height = max(0.05, height / 2.0 - capsule_radius)
    pelvis_local_center = np.array([0.0, 0.0, height / 2.0]) - bone_world_pos["Bip01 Pelvis"]
    bodies["Bip01 Pelvis"].add_geom(
        name=f"{uid}_collision",
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        pos=pelvis_local_center,
        size=[capsule_radius, capsule_half_height, 0],
        contype=8,
        conaffinity=15,
        group=4,
        rgba=[0.5, 0.5, 0.5, 0.0],
    )


def _write_xml(spec, uid: str, out_dir: Path) -> None:
    xml_path = out_dir / f"{uid}.xml"
    prev_cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        xml_path.write_text(spec.to_xml())
    finally:
        os.chdir(prev_cwd)


def _write_texture_png(diffuse_path, out_png: Path) -> None:
    from PIL import Image

    if out_png.exists():
        return
    img = Image.open(diffuse_path).convert("RGBA")
    img.thumbnail((TEXTURE_MAX_DIM, TEXTURE_MAX_DIM), Image.LANCZOS)
    img.save(out_png, optimize=True)


def convert_one_articulated(uid: str, category: str, rocketbox_dir: Path, out_dir: Path) -> dict:
    """Rigid-skinning articulation: assign each vertex to its highest-weighted
    skin cluster (an "equivalent strategy" to nearest-rigging-element cutting,
    using the rig's own authored weights instead of a from-scratch geometric
    heuristic), collapse to CURATED_BONES via nearest curated ancestor, and
    build one rigid body per curated bone connected by ball joints. No
    animation/pose retargeting -- this reproduces the bind (rest) pose as a
    posable ragdoll, same rest pose as the static convert_one() output.

    All ufbx access happens in _extract_articulated_data's isolated
    subprocess (see its docstring) -- this function only touches the
    extracted numpy arrays/JSON metadata, Open3D, and mujoco.MjSpec.
    """
    import mujoco as mj
    import numpy as np
    import open3d as o3d

    arrays, metadata = _load_extracted(uid, category, rocketbox_dir, out_dir)
    pos_idx = arrays["pos_idx"]
    uv_vals = arrays["uv_vals"]
    uv_idx = arrays["uv_idx"]
    face_material = arrays["face_material"]
    world_pos = arrays["world_pos"]
    vertex_curated_bone = arrays["vertex_curated_bone"]

    bone_world_pos = {name: np.array(pos) for name, pos in metadata["bone_world_pos"].items()}
    height = metadata["height"]
    half_width = metadata["half_width"]
    materials_meta = metadata["materials"]

    num_indices = len(pos_idx)
    corner_face_material = np.repeat(face_material, 3)
    corner_bone = vertex_curated_bone[pos_idx]  # per-corner, via position index
    triangles = np.arange(num_indices).reshape(-1, 3)

    spec = mj.MjSpec()
    spec.modelname = uid
    visual_default = spec.add_default("avatar_visual", spec.default)
    visual_default.geom.contype = 0
    visual_default.geom.conaffinity = 0
    visual_default.geom.group = 0
    visual_default.geom.mass = 1e-8

    bodies = _add_bone_bodies(spec, uid, bone_world_pos)

    # A triangle belongs to whichever bone its first corner is assigned to --
    # good enough for a rigid partition (a handful of triangles straddling a
    # joint boundary rendering with the "wrong" neighbor is a cosmetic, not a
    # functional, imperfection).
    used_materials = sorted(set(face_material.tolist()))
    for mat_idx in used_materials:
        mat_info = materials_meta[mat_idx]
        mat_name = mat_info["name"]
        mat_face_mask = corner_face_material == mat_idx
        mat_tri_mask = mat_face_mask.reshape(-1, 3).all(axis=1)
        if not mat_tri_mask.any():
            continue

        texture_png = None
        if mat_info["diffuse_path"] is not None:
            texture_png = out_dir / f"{uid}_{mat_name}.png"
            _write_texture_png(mat_info["diffuse_path"], texture_png)

        vis_material_name = None
        if texture_png is not None:
            vis_texture_name = f"{uid}_{mat_name}_tex"
            spec.add_texture(
                name=vis_texture_name, file=texture_png.name, type=mj.mjtTexture.mjTEXTURE_2D
            )
            vis_material_name = f"{uid}_{mat_name}_material"
            mat_spec = spec.add_material(name=vis_material_name)
            mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = vis_texture_name

        # First-corner bone assignment per triangle (a rigid partition -- see
        # docstring), restricted to this material's triangles.
        first_corner_bone = corner_bone[triangles[:, 0]]
        for bone_name, _ in CURATED_BONES:
            bone_key = _bone_key(bone_name)
            tri_mask = mat_tri_mask & (first_corner_bone == bone_name)
            if not tri_mask.any():
                continue
            tri_subset = triangles[tri_mask]
            used_corners = np.unique(tri_subset.reshape(-1))
            keys = np.stack([pos_idx[used_corners], uv_idx[used_corners]], axis=1)
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            if len(unique_keys) < 12:
                # A handful of stray triangles at a material/bone boundary.
                # Below 4 vertices MuJoCo's mesh compiler rejects it outright
                # ("at least 4 vertices required"); a few more than that but
                # still tiny (e.g. a 2-triangle "opacity" cutout card, ~6
                # vertices) tends to be a near-planar sliver whose inertia
                # tensor is singular/degenerate even under shell inertia
                # ("mass and inertia of moving bodies must be larger than
                # mjMINVAL") -- empirically these tiny fringe details (a
                # lapel corner, a card-thin accessory) are visually
                # negligible, so skip the whole small-vertex-count range
                # rather than trying to distinguish "small but fine" from
                # "small and degenerate" geometrically.
                continue
            inverse = inverse.reshape(-1)
            remap = -np.ones(num_indices, dtype=np.int64)
            remap[used_corners] = inverse
            sub_triangles = remap[tri_subset]
            # Local to this bone's body: subtract the bone's own world position.
            sub_pos = world_pos[unique_keys[:, 0]] - bone_world_pos[bone_name][None]
            sub_uv = uv_vals[unique_keys[:, 1]].copy()
            sub_uv[:, 1] = 1.0 - sub_uv[:, 1]

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(sub_pos)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(sub_triangles)
            o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(sub_uv[sub_triangles.reshape(-1)])
            o3d_mesh.compute_vertex_normals()
            if texture_png is not None:
                o3d_mesh.textures = [o3d.io.read_image(str(texture_png))]

            mesh_file = f"{uid}_{mat_name}_{bone_key}.obj"
            o3d.io.write_triangle_mesh(str(out_dir / mesh_file), o3d_mesh)
            _round_obj_precision(out_dir / mesh_file, decimals=5)

            mesh_asset_name = f"{uid}_{mat_name}_{bone_key}_mesh"
            # shell inertia: some per-bone slivers (e.g. a thin "opacity"
            # cutout layer) are thin enough that MuJoCo's default solid-volume
            # inertia calculation considers their enclosed volume degenerate
            # ("mesh volume is too small"). Geom mass is set explicitly to
            # 1e-8 regardless, so the inertia *shape* from a shell assumption
            # has no real physical consequence here.
            spec.add_mesh(
                name=mesh_asset_name, file=mesh_file, inertia=mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
            )

            geom = bodies[bone_name].add_geom(
                name=f"{uid}_{mat_name}_{bone_key}_visual",
                type=mj.mjtGeom.mjGEOM_MESH,
                meshname=mesh_asset_name,
            )
            geom.classname = visual_default
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 0
            geom.mass = 1e-8
            if vis_material_name is not None:
                geom.material = vis_material_name

    _add_inertia_fallback_geoms(bodies, uid)
    _add_pelvis_collision(bodies, uid, bone_world_pos, height, half_width)
    _write_xml(spec, uid, out_dir)

    return {"uid": uid, "height": float(height), "half_width": float(half_width)}


def _add_bbox_markers(bodies: dict, uid: str, bone_world_pos, height: float) -> None:
    """Two invisible marker spheres at the mesh bounding box's bottom and top.

    A skin is an asset, not a geom, so every geom-based measurement of the avatar
    misses it -- including body_aabb (utils/mj_model_and_data_utils.py), which
    place_object_near uses to work out how far above a floor point to drop the
    body. Without these, the only visible geoms are the per-bone inertia spheres,
    so the "bottom" of a skinned avatar reads as its ankle joints rather than the
    soles of its feet, the body lands ~9cm low, the collision capsule ends up
    through the floor, and every candidate placement is rejected as colliding
    (PickWithHumanRBTaskSampler: "Could not place ... after 20 attempts").

    Kept as two 1cm spheres rather than one bbox-sized box on purpose: they have
    to sit in a *visible* group (0-2) to count toward the AABB, and anything in
    those groups is drawn -- alpha 0 keeps them out of RGB, but a body-sized box
    would still be a body-sized silhouette in a segmentation render. The
    articulated build has no equivalent because its mesh geoms are the bbox.
    """
    import mujoco as mj
    import numpy as np

    pelvis = bone_world_pos["Bip01 Pelvis"]
    for label, z in (("bbox_min", 0.0), ("bbox_max", height)):
        bodies["Bip01 Pelvis"].add_geom(
            name=f"{uid}_{label}",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            pos=np.array([0.0, 0.0, z]) - pelvis,
            contype=0,
            conaffinity=0,
            group=0,
            mass=1e-8,
            rgba=[0, 0, 0, 0],
        )


# Linear blend skinning: bones per vertex kept after collapsing the rig's
# clusters onto CURATED_BONES, and the weight below which an influence is
# dropped as noise. 4 is the usual LBS budget and is plenty here -- the curated
# rig has no place where more than a Clavicle/UpperArm/Spine2-style trio meets.
SKIN_MAX_BONES_PER_VERTEX = 4
SKIN_MIN_WEIGHT = 0.01


def _fill_skin(skin, verts, texcoords, faces, bones) -> None:
    """Populate an ``<skin>`` element inline (vert/texcoord/face/bone attributes).

    MuJoCo also accepts a skin as a binary ``.skn`` file -- the form dm_control's
    dog_v2 `create_skin` writes, and the compact one: these avatars carry ~5-15k
    skinned vertices each. Inline is used anyway because a file-backed skin does
    not survive `MjSpec.attach`: the bone-to-body links of a `.skn` are resolved
    from the file at compile time, so attaching the avatar under a namespace
    (which is how PickWithHumanRBTaskSampler stages avatars into a scene) leaves
    them pointing at unprefixed body names and the scene fails to compile with
    "unknown body ... in skin". Inline bone names are held in the spec, so attach
    renames them along with the bodies.

    `bones` is a sequence of (body_name, bindpos, bindquat, vertex_ids, weights);
    bindpos/bindquat are that body's pose in the bind (rest) configuration, and
    the vertex positions are in that same frame.
    """
    import numpy as np

    skin.vert = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1).tolist()
    skin.texcoord = np.ascontiguousarray(texcoords, dtype=np.float32).reshape(-1).tolist()
    skin.face = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1).tolist()
    skin.bodyname = [name for name, _, _, _, _ in bones]
    skin.bindpos = np.concatenate([np.asarray(p, dtype=np.float64) for _, p, _, _, _ in bones])
    skin.bindquat = np.concatenate([np.asarray(q, dtype=np.float64) for _, _, q, _, _ in bones])
    skin.vertid = [np.asarray(ids, dtype=np.int32).tolist() for _, _, _, ids, _ in bones]
    skin.vertweight = [np.asarray(w, dtype=np.float32).tolist() for _, _, _, _, w in bones]


def _skin_vertex_weights(curated_weights):
    """Per-vertex (bone_index, weight) influences for a linear blend skin.

    Takes the dense (nvertex, len(CURATED_BONES)) weight matrix the extractor
    collapses the rig's clusters into, keeps the SKIN_MAX_BONES_PER_VERTEX
    largest influences per vertex, drops anything under SKIN_MIN_WEIGHT, and
    renormalizes so each vertex's weights sum to 1 (MuJoCo blends whatever it is
    given, so weights that don't sum to 1 shrink the vertex toward the origin).
    Vertices the rig left unweighted fall back to the root bone, rigidly.

    Returns (indices, weights), both (nvertex, SKIN_MAX_BONES_PER_VERTEX), with
    dropped influences carrying weight 0.
    """
    import numpy as np

    k = min(SKIN_MAX_BONES_PER_VERTEX, curated_weights.shape[1])
    top = np.argpartition(-curated_weights, k - 1, axis=1)[:, :k]
    weights = np.take_along_axis(curated_weights, top, axis=1).astype(np.float64)
    weights[weights < SKIN_MIN_WEIGHT] = 0.0
    totals = weights.sum(axis=1, keepdims=True)
    unweighted = totals[:, 0] <= 0.0
    if unweighted.any():
        top[unweighted] = 0  # CURATED_BONES[0] is the root (pelvis)
        weights[unweighted] = 0.0
        weights[unweighted, 0] = 1.0
        totals[unweighted] = 1.0
    return top, weights / totals


def convert_one_skinned(uid: str, category: str, rocketbox_dir: Path, out_dir: Path) -> dict:
    """Deformable skinning: the same CURATED_BONES ragdoll as
    convert_one_articulated, but the mesh rides on it as MuJoCo `<skin>` assets
    (linear blend skinning) instead of being cut into one rigid mesh geom per
    bone.

    The difference that matters is at the joints. The rigid partition assigns
    every vertex to exactly one bone, so bending an elbow tears the arm open at
    the seam between Forearm and UpperArm; a skin keeps the rig's own blended
    weights, so those vertices follow both bones and the surface stays closed.
    It is also much lighter: one skin per material instead of ~20 per-bone OBJs,
    and no per-bone mesh assets for MuJoCo to compile.

    Skins are visual-only -- they contribute no inertia and never collide -- so
    the bodies still carry the fallback spheres and the pelvis collision capsule
    the articulated build uses.

    All ufbx access happens in _extract_articulated_data's isolated subprocess
    (see its docstring); this function only touches the extracted arrays.
    """
    import mujoco as mj
    import numpy as np

    arrays, metadata = _load_extracted(uid, category, rocketbox_dir, out_dir)
    pos_idx = arrays["pos_idx"]
    uv_vals = arrays["uv_vals"]
    uv_idx = arrays["uv_idx"]
    face_material = arrays["face_material"]
    world_pos = arrays["world_pos"]
    curated_weights = arrays["curated_weights"]

    bone_world_pos = {name: np.array(pos) for name, pos in metadata["bone_world_pos"].items()}
    height = metadata["height"]
    half_width = metadata["half_width"]
    materials_meta = metadata["materials"]

    num_indices = len(pos_idx)
    corner_face_material = np.repeat(face_material, 3)
    triangles = np.arange(num_indices).reshape(-1, 3)
    vertex_bone_ids, vertex_bone_weights = _skin_vertex_weights(curated_weights)

    spec = mj.MjSpec()
    spec.modelname = uid
    bodies = _add_bone_bodies(spec, uid, bone_world_pos)

    # Bind pose. _add_bone_bodies gives every body position-only placement, so at
    # qpos0 each one's global orientation is the identity and its global position
    # is the bone's world position -- exactly the frame the skin's vertices are
    # written in, which is what makes the bind transforms these two constants.
    bind_quat = np.array([1.0, 0.0, 0.0, 0.0])
    bone_body_names = [f"{uid}_{_bone_key(name)}" for name, _ in CURATED_BONES]

    used_materials = sorted(set(face_material.tolist()))
    skin_count = 0
    for mat_idx in used_materials:
        mat_info = materials_meta[mat_idx]
        mat_name = mat_info["name"]
        mat_face_mask = corner_face_material == mat_idx
        tri_mask = mat_face_mask.reshape(-1, 3).all(axis=1)
        if not tri_mask.any():
            continue

        # One skin per material: a skin carries a single material, and these
        # avatars split body/hair/clothing across several textures.
        tri_subset = triangles[tri_mask]
        used_corners = np.unique(tri_subset.reshape(-1))
        # Dedup by (position, uv) -- same reasoning as convert_one(), plus a hard
        # requirement here: MuJoCo's skin format has one texcoord per vertex, so
        # a position appearing under two UVs has to become two skin vertices.
        keys = np.stack([pos_idx[used_corners], uv_idx[used_corners]], axis=1)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        if len(unique_keys) < 3:
            continue
        inverse = inverse.reshape(-1)
        remap = -np.ones(num_indices, dtype=np.int64)
        remap[used_corners] = inverse
        sub_faces = remap[tri_subset]
        sub_pos = world_pos[unique_keys[:, 0]]
        sub_uv = uv_vals[unique_keys[:, 1]].copy()
        sub_uv[:, 1] = 1.0 - sub_uv[:, 1]

        sub_bone_ids = vertex_bone_ids[unique_keys[:, 0]]
        sub_weights = vertex_bone_weights[unique_keys[:, 0]]
        bones = []
        for bone_i, body_name in enumerate(bone_body_names):
            influence = (sub_bone_ids == bone_i) & (sub_weights > 0.0)
            vert_ids = np.nonzero(influence.any(axis=1))[0]
            if len(vert_ids) == 0:
                continue  # this material's geometry never touches this bone
            weights = sub_weights[vert_ids][influence[vert_ids]]
            bones.append(
                (
                    body_name,
                    bone_world_pos[CURATED_BONES[bone_i][0]],
                    bind_quat,
                    vert_ids,
                    weights,
                )
            )

        skin = spec.add_skin(name=f"{uid}_{mat_name}_skin", group=0)
        _fill_skin(skin, sub_pos, sub_uv, sub_faces, bones)
        if mat_info["diffuse_path"] is not None:
            texture_png = out_dir / f"{uid}_{mat_name}.png"
            _write_texture_png(mat_info["diffuse_path"], texture_png)
            vis_texture_name = f"{uid}_{mat_name}_tex"
            spec.add_texture(
                name=vis_texture_name, file=texture_png.name, type=mj.mjtTexture.mjTEXTURE_2D
            )
            vis_material_name = f"{uid}_{mat_name}_material"
            mat_spec = spec.add_material(name=vis_material_name)
            mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = vis_texture_name
            skin.material = vis_material_name
        skin_count += 1

    if skin_count == 0:
        raise RuntimeError(f"{uid}: no skinnable geometry found")

    _add_inertia_fallback_geoms(bodies, uid)
    _add_bbox_markers(bodies, uid, bone_world_pos, height)
    _add_pelvis_collision(bodies, uid, bone_world_pos, height, half_width)
    _write_xml(spec, uid, out_dir)

    return {"uid": uid, "height": float(height), "half_width": float(half_width)}


def write_assets_index(out_dir: Path, avatars: dict) -> None:
    import json

    index = {}
    for uid in avatars:
        avatar_dir = out_dir / uid
        if not (avatar_dir / f"{uid}.xml").exists():
            continue
        metadata_path = avatar_dir / f"{uid}.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "assetId": uid,
                    "category": "avatar",
                    "description_long": "A static human avatar (Microsoft Rocketbox).",
                    "description": "A person.",
                    "description_short": {
                        "one_word": "person",
                        "two_words": "a person",
                        "three_words": "a standing person",
                    },
                    "synset": "person.n.01",
                },
                indent=2,
            )
        )
        index[uid] = {
            "uid": uid,
            "object_path": f"{uid}/{uid}.xml",
            "metadata_path": f"{uid}/{uid}.json",
            "metadata_npz_path": None,
        }
    (out_dir / "assets_index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {out_dir / 'assets_index.json'} with {len(index)} avatars")


def cmd_convert(args, rocketbox_dir: Path, out_dir: Path, command: str, convert_fn) -> None:
    if args.uid:
        info = convert_fn(args.uid, args.category, rocketbox_dir, out_dir / args.uid)
        print(f"OK {args.uid} height={info['height']:.2f}m half_width={info['half_width']:.2f}m")
        return

    # Driver mode: one subprocess per avatar (see module docstring -- MjSpec
    # cleanup segfaults on exit; isolating per-avatar avoids losing the batch).
    # A single avatar's conversion failing (unusual materials, missing texture,
    # no skin deformer for `articulate`, etc.) doesn't abort the rest of the
    # ~115-avatar batch -- failures are collected and reported at the end.
    out_dir.mkdir(parents=True, exist_ok=True)
    avatars = discover_avatars(rocketbox_dir)
    if not avatars:
        print(f"No avatars found under {rocketbox_dir}. Run `download` first.", file=sys.stderr)
        sys.exit(1)

    failed = []
    for i, (uid, category) in enumerate(avatars.items()):
        print(f"[{i + 1}/{len(avatars)}] Converting {uid} ({category})...")
        subprocess.run(
            [
                sys.executable,
                __file__,
                command,
                "--uid",
                uid,
                "--category",
                category,
                "--rocketbox-dir",
                str(rocketbox_dir),
                "--out-dir",
                str(out_dir),
            ],
            check=False,
        )
        if not (out_dir / uid / f"{uid}.xml").exists():
            print(f"  FAILED {uid}: no output xml written", file=sys.stderr)
            failed.append(uid)

    write_assets_index(out_dir, avatars)
    print(f"\nConverted {len(avatars) - len(failed)}/{len(avatars)} avatars into {out_dir}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download")
    p_download.add_argument(
        "--rocketbox-dir", type=Path, default=None, help="Where the Rocketbox source lives"
    )

    for name in ("convert", "articulate", "skin"):
        p = sub.add_parser(name)
        p.add_argument(
            "--rocketbox-dir", type=Path, default=None, help="Where the Rocketbox source lives"
        )
        p.add_argument("--out-dir", type=Path, default=None, help="Where to write MJCF output")
        p.add_argument(
            "--category", type=str, choices=AVATAR_CATEGORIES, help="Required with --uid"
        )
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--all", action="store_true")
        group.add_argument("--uid", type=str)

    args = parser.parse_args()
    if getattr(args, "uid", None) and not args.category:
        parser.error("--category is required with --uid")
    rocketbox_dir = args.rocketbox_dir or default_rocketbox_dir()

    if args.command == "download":
        cmd_download(rocketbox_dir)
    elif args.command == "convert":
        out_dir = args.out_dir or default_out_dir()
        cmd_convert(args, rocketbox_dir, out_dir, "convert", convert_one)
    elif args.command == "articulate":
        out_dir = args.out_dir or default_articulated_out_dir()
        cmd_convert(args, rocketbox_dir, out_dir, "articulate", convert_one_articulated)
    elif args.command == "skin":
        out_dir = args.out_dir or default_skinned_out_dir()
        cmd_convert(args, rocketbox_dir, out_dir, "skin", convert_one_skinned)


if __name__ == "__main__":
    main()
