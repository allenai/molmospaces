"""Convert the IEEE Humanoids 2026 Loco-Manipulation Challenge testbed (COLLADA) to MJCF.

    python scripts/data/format_conversion/convert_testbed_to_mjcf.py \
        --dae <challenge repo>/CAD-Files/LocoManipulationTestbed.dae --out_dir <dir>

Writes ``<out_dir>/testbed.xml`` plus ``<out_dir>/meshes/*.obj`` (and a loadable
``challenge_scene.xml`` + metadata for the InteractiveShellG1Challenge config):

* visual: one decimated mesh per material colour (contype/conaffinity 0). Textured pieces keep
  their UVs and texture images.
* collision: one convex hull per CAD instance (MuJoCo collides meshes as convex hulls),
  so concave parts (e.g. hollow cabinets) become solid. The CAD has no mass/friction data.

Task parts are cut out of the static mesh into articulated bodies (see TASK PARTS below):
the door (hinge, return spring) with its lever handle, and the nine keypad push buttons.

The DAE is in inches, Z-up; the output is in metres, Z-up, with the apparatus' min corner
at x=0, y=0 and its walking deck at z=0 (the CAD base is at z=-0.10).
"""

import argparse
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh


# ---- TASK PARTS: CAD node ids of the pieces that move. -----------------------------------------
DOOR_NODE = "ID3443"  # grey 3ft x 7ft slab, modelled in the CAD swung open ~54 deg
HANDLE_NODES = ["ID3457", "ID3465", "ID3471", "ID3477", "ID3483", "ID3489"]  # lever + spindle
DOORFRAME_NODES = ("ID3432", "ID3432_1")  # thin trim around the doorway, hull would fill it
# keypad digit -> (white label plate node, black push-button node); 3x3 grid, 1 top-left
KEYPAD_NODES = {
    1: ("ID5684", "ID5691"), 2: ("ID5705", "ID5712"), 3: ("ID5726", "ID5733"),
    4: ("ID5789", "ID5796"), 5: ("ID5747", "ID5754"), 6: ("ID5768", "ID5775"),
    7: ("ID5810", "ID5817"), 8: ("ID5831", "ID5838"), 9: ("ID5852", "ID5859"),
}  # fmt: skip
DECK_TOP = 0.10  # m, walking surface height above the CAD's base; the output puts it at z=0
KEY_TRAVEL = 0.008  # m, push depth (along -x)


def write_visual(prefix, items, mesh_dir, out, face_budget=None):
    """Write visual meshes for `items` [(mesh, rgba)]: textured pieces grouped per texture
    image (UVs kept), untextured ones per colour (decimated past `face_budget`). Appends
    MJCF snippets to `out` (dict of lists: textures, mats, assets, geoms) and returns the
    geom names.
    """
    groups: dict = {}
    for m, rgba in items:
        tex = getattr(getattr(m.visual, "material", None), "baseColorTexture", None)
        uv = getattr(m.visual, "uv", None)
        if tex is not None and uv is not None:
            groups.setdefault(id(tex), (tex, rgba, []))[2].append((m, uv))
        else:
            groups.setdefault(rgba, (None, rgba, []))[2].append((m, None))
    names = []
    for gi, (tex, rgba, members) in enumerate(groups.values()):
        name = f"{prefix}_{gi}"
        if tex is not None:
            verts = np.concatenate([m.vertices for m, _ in members])
            uvs = np.concatenate([uv for _, uv in members])
            offs = np.cumsum([0] + [len(m.vertices) for m, _ in members[:-1]])
            faces = np.concatenate([m.faces + o for (m, _), o in zip(members, offs)])
            with open(mesh_dir / f"{name}.obj", "w") as f:
                f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in verts)
                f.writelines(f"vt {u:.5f} {v:.5f}\n" for u, v in uvs)
                f.writelines(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in faces + 1)
            tex.convert("RGB").save(mesh_dir / f"{name}.png")
            out["textures"].append(f'<texture name="{name}" type="2d" file="meshes/{name}.png"/>')
            out["mats"].append(f'<material name="{name}" texture="{name}"/>')
        else:
            merged = trimesh.util.concatenate(
                [trimesh.Trimesh(m.vertices, m.faces, process=False) for m, _ in members]
            )
            if face_budget and len(merged.faces) > face_budget:
                v, f = fast_simplification.simplify(
                    merged.vertices,
                    merged.faces,
                    target_reduction=1 - face_budget / len(merged.faces),
                )
                merged = trimesh.Trimesh(v, f, process=False)
            merged.export(mesh_dir / f"{name}.obj")
            rgba_s = " ".join(f"{c:g}" for c in rgba)
            out["mats"].append(f'<material name="{name}" rgba="{rgba_s}"/>')
        out["assets"].append(f'<mesh name="{name}" file="meshes/{name}.obj" inertia="shell"/>')
        names.append(name)
    return names


def vis_geom(name, rgba_attr=""):
    return (
        f'<geom name="{name}" type="mesh" mesh="{name}" material="{name}" {rgba_attr} '
        'contype="0" conaffinity="0" density="0" group="2"/>'
    )


def hull_or_box(m: trimesh.Trimesh) -> trimesh.Trimesh:
    if m.extents.min() > 0.01:  # thinner pieces give MuJoCo degenerate (coplanar) hulls
        try:
            return m.convex_hull
        except Exception:
            pass
    obb = m.bounding_box_oriented
    return trimesh.creation.box(np.maximum(obb.primitive.extents, 0.01), obb.primitive.transform)


def to_frame(m: trimesh.Trimesh, T_world_frame: np.ndarray) -> trimesh.Trimesh:
    """Copy of `m` (world coords) expressed in the frame with world pose `T_world_frame`."""
    out = m.copy()
    out.apply_transform(np.linalg.inv(T_world_frame))
    return out


def material_rgba(geom: trimesh.Trimesh) -> tuple[float, ...]:
    mat = getattr(geom.visual, "material", None)
    tex = getattr(mat, "baseColorTexture", None)
    if tex is not None:
        rgb = np.asarray(tex.convert("RGB").resize((8, 8))).reshape(-1, 3).mean(0) / 255
        return (*np.round(rgb, 2), 1.0)
    col = getattr(mat, "main_color", None)
    return tuple(np.round(np.asarray(col if col is not None else [200, 200, 200, 255]) / 255, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dae", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--visual_faces", type=int, default=60000, help="face budget per colour group")
    args = ap.parse_args()

    mesh_dir = args.out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(args.dae, force="scene")
    scale = 0.0254  # inches -> metres
    world = {}
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(tf)
        m.apply_scale(scale)
        world[node] = (m, material_rgba(scene.geometry[gname]))

    lo = np.min([m.bounds[0] for m, _ in world.values()], axis=0)
    shift = -lo
    shift[2] -= DECK_TOP  # the G1 stack assumes the floor is at z=0
    for m, _ in world.values():
        m.apply_translation(shift)
    print(f"{len(world)} instances; extent {np.max([m.bounds[1] for m, _ in world.values()], axis=0)}")

    # Cut the task parts out of the static scene.
    taken = {}
    take_prefix = [DOOR_NODE, *HANDLE_NODES] + [n for pair in KEYPAD_NODES.values() for n in pair]
    for node in list(world):
        if any(node == t or node.startswith(t + "_") for t in take_prefix):
            taken[node] = world.pop(node)

    def parts(*ids):
        return [taken[n] for n in taken if any(n == i or n.startswith(i + "_") for i in ids)]

    out = {"textures": [], "mats": [], "assets": [], "geoms": []}
    assets, geoms = out["assets"], out["geoms"]

    # Static collision hulls (the doorframe trim is replaced by jambs + header: its hull
    # would plug the doorway).
    for i, (node, (m, _)) in enumerate(world.items()):
        name = f"col_{i}"
        if node in DOORFRAME_NODES:
            if node.endswith("_1"):
                continue
            (x0, y0, z0), (x1, y1, z1) = m.bounds
            ys = np.unique(np.round(m.vertices[:, 1], 3))
            zs = np.unique(np.round(m.vertices[:, 2], 3))
            jamb_l, jamb_r, head = ys[2], ys[-3], zs[-2]  # inner trim edges
            for k, (lo_, hi_) in enumerate(
                [((x0, y0, z0), (x1, jamb_l, z1)), ((x0, jamb_r, z0), (x1, y1, z1)),
                 ((x0, y0, head), (x1, y1, z1))]
            ):  # fmt: skip
                c, h = (np.array(lo_) + hi_) / 2, (np.array(hi_) - lo_) / 2
                geoms.append(
                    f'<geom name="doorframe_{k}" type="box" pos="{c[0]:g} {c[1]:g} {c[2]:g}" '
                    f'size="{h[0]:g} {h[1]:g} {h[2]:g}" group="3" rgba="0 0 0 0"/>'
                )
            continue
        hull_or_box(m).export(mesh_dir / f"{name}.obj")
        assets.append(f'<mesh name="{name}" file="meshes/{name}.obj"/>')
        geoms.append(f'<geom name="{name}" type="mesh" mesh="{name}" group="3" rgba="0 0 0 0"/>')

    # Static visual meshes.
    for name in write_visual("vis", list(world.values()), mesh_dir, out, args.visual_faces):
        geoms.append(vis_geom(name))

    # ---- Door: hinge on the CAD slab's hinge edge, re-posed from its open CAD pose to closed.
    slab = parts(DOOR_NODE)[0][0]
    v = slab.vertices
    hinge = v[v[:, 0] < v[:, 0].min() + 0.03].mean(0)
    far = v[v[:, 0] > v[:, 0].max() - 0.03].mean(0)
    d = far[:2] - hinge[:2]
    theta0 = np.pi / 2 - np.arctan2(d[1], d[0])  # rotate about +z so the door lies along +y
    c, s_ = np.cos(theta0), np.sin(theta0)
    R0 = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    hinge_pos = np.array([hinge[0], hinge[1], 0.0])
    T_door = np.eye(4)  # door frame (closed pose) -> world
    T_door[:3, :3], T_door[:3, 3] = np.eye(3), hinge_pos
    T_cad_to_closed = np.eye(4)  # world(CAD, open) -> world(closed)
    T_cad_to_closed[:3, :3] = R0
    T_cad_to_closed[:3, 3] = hinge_pos - R0 @ hinge_pos

    def closed_local(m, T_frame):
        m = m.copy()
        m.apply_transform(T_cad_to_closed)
        return to_frame(m, T_frame)

    door_slab = closed_local(slab, T_door)
    (bx0, by0, bz0), (bx1, by1, bz1) = door_slab.bounds
    dc, dh = (door_slab.bounds[0] + door_slab.bounds[1]) / 2, (door_slab.bounds[1] - door_slab.bounds[0]) / 2

    handle_items = parts(*HANDLE_NODES)
    spindles = [m for n, (m, _) in taken.items() if n in ("ID3471", "ID3483")]
    spindle = np.mean([m.bounds.mean(0) for m in spindles], axis=0)
    spindle_closed = (T_cad_to_closed @ np.append(spindle, 1))[:3]
    T_handle = np.eye(4)
    T_handle[:3, 3] = spindle_closed
    # Lever direction (in the y-z plane) decides which way "down" is.
    lever = closed_local(parts("ID3457")[0][0], T_handle)
    lever_dir = lever.bounds.mean(0)[1:]  # (dy, dz) from the spindle
    down_sign = -np.sign(lever_dir[0]) if abs(lever_dir[0]) > abs(lever_dir[1]) else np.sign(lever_dir[1])
    lo_a, hi_a = sorted([0.0, 0.9 * down_sign])
    print(f"door theta0={np.degrees(theta0):.1f}deg hinge={hinge_pos} spindle={spindle_closed} lever_dir={lever_dir}")

    door_geoms, handle_geoms = [], []
    dout = {"textures": out["textures"], "mats": out["mats"], "assets": out["assets"], "geoms": []}
    for name in write_visual(
        "door_vis", [(closed_local(m, T_door), r) for m, r in parts(DOOR_NODE)], mesh_dir, dout
    ):
        door_geoms.append(vis_geom(name))
    door_geoms.append(
        f'<geom name="door_col" type="box" pos="{dc[0]:g} {dc[1]:g} {dc[2]:g}" '
        f'size="{dh[0]:g} {dh[1]:g} {dh[2]:g}" mass="8" group="3" rgba="0 0 0 0"/>'
    )
    for name in write_visual(
        "handle_vis", [(closed_local(m, T_handle), r) for m, r in handle_items], mesh_dir, dout
    ):
        handle_geoms.append(vis_geom(name))
    for k, (m, _) in enumerate(handle_items):
        hull_or_box(closed_local(m, T_handle)).export(mesh_dir / f"handle_col_{k}.obj")
        assets.append(f'<mesh name="handle_col_{k}" file="meshes/handle_col_{k}.obj"/>')
        handle_geoms.append(
            f'<geom name="handle_col_{k}" type="mesh" mesh="handle_col_{k}" mass="0.05" '
            'group="3" rgba="0 0 0 0"/>'
        )
    sp = spindle_closed - hinge_pos  # handle origin in the door frame
    door_xml = (
        f'<body name="door" pos="{hinge_pos[0]:g} {hinge_pos[1]:g} 0">\n'
        f'        <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 {np.degrees(1.9):.1f}" '
        'stiffness="8" damping="4" springref="0"/>\n        '
        + "\n        ".join(door_geoms)
        + f'\n        <body name="door_handle" pos="{sp[0]:g} {sp[1]:g} {sp[2]:g}">\n'
        f'          <joint name="door_handle_turn" type="hinge" axis="1 0 0" '
        f'range="{np.degrees(lo_a):.1f} {np.degrees(hi_a):.1f}" stiffness="0.6" damping="0.05" springref="0"/>\n          '
        + "\n          ".join(handle_geoms)
        + "\n        </body>\n      </body>"
    )

    # ---- Keypad: nine spring-loaded buttons pushed along -x.
    key_xml, excludes = [], [
        '<exclude body1="testbed" body2="door"/>',
        '<exclude body1="testbed" body2="door_handle"/>',
    ]
    for digit, (plate_id, bezel_id) in KEYPAD_NODES.items():
        items = parts(plate_id, bezel_id)
        allv = np.concatenate([m.vertices for m, _ in items])
        bmin, bmax = allv.min(0), allv.max(0)
        origin = np.array([bmin[0], (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2])
        T_key = np.eye(4)
        T_key[:3, 3] = origin
        kg = []
        plate_items = [(to_frame(m, T_key), r) for n, (m, r) in taken.items() if n == plate_id]
        bezel_items = [(to_frame(m, T_key), r) for n, (m, r) in taken.items()
                       if n == bezel_id or n.startswith(bezel_id + "_")]  # fmt: skip
        for name in write_visual(f"key{digit}_plate", plate_items, mesh_dir, dout):
            kg.append(vis_geom(name, 'rgba="1 1 1 1"'))
        for name in write_visual(f"key{digit}_bezel", bezel_items, mesh_dir, dout):
            kg.append(vis_geom(name))
        h = (bmax - bmin) / 2
        kg.append(
            f'<geom name="key{digit}_col" type="box" pos="{h[0]:g} 0 0" '
            f'size="{h[0]:g} {h[1]:g} {h[2]:g}" mass="0.02" group="3" rgba="0 0 0 0"/>'
        )
        key_xml.append(
            f'<body name="key{digit}" pos="{origin[0]:g} {origin[1]:g} {origin[2]:g}">\n'
            f'        <joint name="key{digit}_slide" type="slide" axis="1 0 0" '
            f'range="{-KEY_TRAVEL:g} 0" stiffness="400" damping="25" springref="0"/>\n        '
            + "\n        ".join(kg)
            + "\n      </body>"
        )
        excludes.append(f'<exclude body1="testbed" body2="key{digit}"/>')

    textures, mats = out["textures"], out["mats"]

    xml = (
        '<mujoco model="loco_manipulation_testbed">\n'
        "  <asset>\n    " + "\n    ".join(textures + mats + assets) + "\n  </asset>\n"
        "  <contact>\n    " + "\n    ".join(excludes) + "\n  </contact>\n"
        "  <worldbody>\n"
        '    <body name="testbed">\n      ' + "\n      ".join(geoms) + "\n    </body>\n"
        "    " + door_xml + "\n    " + "\n    ".join(key_xml) + "\n"
        "  </worldbody>\n</mujoco>\n"
    )
    (args.out_dir / "testbed.xml").write_text(xml)
    print("wrote", args.out_dir / "testbed.xml")

    # Scene wrapper the molmospaces "user" scene loader consumes: testbed + ground + light,
    # an offscreen framebuffer big enough for the robot cameras, and empty scene metadata.
    cx, cy = (np.max([m.bounds[1] for m, _ in world.values()], axis=0)[:2] / 2)
    (args.out_dir / "challenge_scene.xml").write_text(
        f'''<mujoco model="ieee_humanoids_2026_challenge">
  <include file="testbed.xml"/>
  <visual><global offwidth="1920" offheight="1080"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.75 0.75 0.75" rgb2="0.65 0.65 0.65" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="12 12" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="{cx:g} {cy:g} 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="{cx:g} {cy:g} {-DECK_TOP:g}" size="12 8 0.1" material="floor_mat"/>
  </worldbody>
</mujoco>
'''
    )
    (args.out_dir / "challenge_scene_metadata.json").write_text('{"objects": {}}\n')
    print("wrote", args.out_dir / "challenge_scene.xml")


if __name__ == "__main__":
    main()
