"""Convert the IEEE Humanoids 2026 Loco-Manipulation Challenge testbed (COLLADA) to MJCF.

    python scripts/data/format_conversion/convert_testbed_to_mjcf.py \
        --dae <challenge repo>/CAD-Files/LocoManipulationTestbed.dae --out_dir <dir>

Add ``--layout dexterity|mobility|full`` (default dexterity: the manipulation-trial apparatus, with the
mobility terrain removed and a flat deck; see LAYOUTS below).

Writes ``<out_dir>/testbed.xml`` plus ``<out_dir>/meshes/*.obj`` (and a loadable
``challenge_scene.xml`` + metadata for the InteractiveShellG1Challenge config):

* visual: one decimated mesh per material colour (contype/conaffinity 0). Textured pieces keep
  their UVs and texture images.
* collision: one convex hull per CAD instance (MuJoCo collides meshes as convex hulls),
  so concave parts (e.g. hollow cabinets) become solid. The CAD has no mass/friction data.

Task parts are cut out of the static mesh into articulated bodies (see TASK PARTS below):
the door (hinge, return spring) with its lever handle, the nine keypad push buttons, and the
lightbulb task (task 3): ten threaded light fixtures on two boards (a wall board and a 45-degree
floor board), their bulbs, and ten spare bulbs standing in the rail rack.

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
# ---- LIGHTBULB TASK (rules PDF task 3): 3D-printed bulbs screwed into light fixtures on boards ----
# The CAD has ten fixtures: five on a vertical wall board (0-4, quincunx) and five on a 45-degree
# tilted floor board (5-9). Each carries a bulb (a white mesh plus an identical black outline shell),
# and ten more bulbs stand in a rail rack (the "box" they are retrieved from).
FIXTURE_NODES = [
    "ID5098", "ID5107", "ID5114", "ID5121", "ID5128",  # wall board
    "ID5397", "ID5404", "ID5411", "ID5418", "ID5425",  # floor board
]  # fmt: skip
INSTALLED_BULB_NODES = [  # (white, black outline) of the bulb the CAD screws into each fixture above
    ("ID5136", "ID5136_1"), ("ID5136_8", "ID5136_9"), ("ID5136_2", "ID5136_3"),
    ("ID5136_4", "ID5136_5"), ("ID5136_6", "ID5136_7"),
    ("ID5136_10", "ID5136_11"), ("ID5437",), ("ID5136_12", "ID5136_13"),
    ("ID5136_14", "ID5136_15"), ("ID5136_16", "ID5136_17"),
]  # fmt: skip
SPARE_BULB_NODES = [(f"ID5136_{k}", f"ID5136_{k + 1}") for k in range(18, 36, 2)] + [("ID5437_1",)]
# Fixtures that start empty (the bulb is not built): the bottom row of the wall board.
EMPTY_FIXTURES = (2, 4)
# Zero-size outline/dimension markers around the bulbs (2-6 faces); as 1 cm boxes they would collide.
MARKER_PREFIXES = ("ID5144", "ID5150", "ID5443", "ID5449")
RACK_PLATE_NODES = ["ID5897", "ID5897_1"]  # rack plate the spare bulbs hang through
RACK_COLLAR_NODES = [f"ID{n}{s}" for n in range(5912, 5992, 8) for s in ("", "_1")]
RACK_HOLE = 0.017  # m, half-width of the (square) hole each spare bulb stands in
THREAD_PITCH = 0.001  # m of travel into the fixture per full turn
SEAT_DEG = 270.0  # turn at which a bulb counts as tightened / seated (task module: BULB_SEATED_DEG)
STOP_DEG = 285.0  # hard stop of the thread (bottomed out)
# Sim values, not CAD data: the thread has to hold a bulb against the few newtons of incidental finger
# contact, yet be turnable by a three-digit grip, and the (light) bulb joint needs rotor inertia and
# damping to stay smooth under contact; hence a high-friction ("rubberised") bulb surface.
THREAD_FRICTION = 0.25  # N m, dry friction of the thread joint
THREAD_DAMPING = 0.05  # N m s/rad
THREAD_ARMATURE = 0.004  # kg m^2, rotor inertia of the thread joint
BULB_FRICTION = 6.0  # sliding friction coefficient of the bulb's contact geoms
BULB_MASS = 0.06  # kg, 3D-printed hollow bulb
BULB_TIP_LEN = 0.105  # m, base to tip
DECK_TOP = 0.10  # m, walking surface height above the CAD's base; the output puts it at z=0
KEY_TRAVEL = 0.008  # m, push depth (along -x)

# ---- LAYOUTS: which CAD nodes the scene keeps (rules PDF 'Mobility Tasks' / 'Manipulation Tasks'). ----
# The CAD is the *mobility* apparatus: manipulation stations plus the mobility terrain (ramps, stairs,
# wall obstacle). In Manipulation (dexterity) Trials the terrain is removed and the deck is flat from the
# start to the stations, so `dexterity` (default) edits the CAD with the tables below. `mobility` and
# `full` keep every CAD node (the mobility trial and the loco-manipulation trial use the same hardware;
# they only differ in the tasks scored). All node ids are exact CAD node names; heights are metres in
# the output frame (deck top z=0.003 on the 2.5-4.9 m and start-side sheets).
LAYOUTS = ("dexterity", "mobility", "full")
# 8ft x 8ft 15-degree hill/pinwheel section (x 4.93-7.37): wedge tiles (raised tops up to 0.19 m), their
# edge skins/plates and the cleat blocks under them. Deleted in `dexterity`. (Ids found by bounds: x 4.92-7.37,
# y 0.07-2.52, z 0.0135-0.2.)
PINWHEEL_TERRAIN_NODES = [
    "ID3522", "ID3536", "ID3550", "ID3559", "ID3567", "ID3576", "ID3576_1", "ID3576_10",
    "ID3576_11", "ID3576_12", "ID3576_13", "ID3576_14", "ID3576_15", "ID3576_16", "ID3576_17",
    "ID3576_18", "ID3576_19", "ID3576_2", "ID3576_20", "ID3576_21", "ID3576_22", "ID3576_23",
    "ID3576_24", "ID3576_25", "ID3576_26", "ID3576_27", "ID3576_28", "ID3576_29", "ID3576_3",
    "ID3576_30", "ID3576_31", "ID3576_4", "ID3576_5", "ID3576_6", "ID3576_7", "ID3576_8",
    "ID3576_9", "ID3587", "ID3596", "ID3605", "ID3614", "ID3622", "ID3632", "ID3641", "ID3650",
    "ID3659", "ID3667", "ID3677", "ID3686", "ID3695", "ID3704", "ID3712", "ID3732", "ID3741",
    "ID3750", "ID3759", "ID3767", "ID3777", "ID3786", "ID3795", "ID3804", "ID3812", "ID3822",
    "ID3831", "ID3840", "ID3849", "ID3857", "ID3867", "ID3876", "ID3885", "ID3894", "ID3902",
    "ID3922", "ID3931", "ID3940", "ID3949", "ID3957", "ID3967", "ID3976", "ID3985", "ID3994",
    "ID4002", "ID4012", "ID4021", "ID4030", "ID4039", "ID4047", "ID4057", "ID4066", "ID4075",
    "ID4084", "ID4092", "ID4112", "ID4121", "ID4130", "ID4139", "ID4147", "ID4157", "ID4166",
    "ID4175", "ID4184", "ID4192", "ID4202", "ID4211", "ID4220", "ID4229", "ID4237", "ID4247",
    "ID4256", "ID4265", "ID4274", "ID4282",
]  # fmt: skip
SMALL_WALL_NODES = ["ID5049"]  # mobility C: 0.79 m partition across half the flat section (x=3.71)
RAMP_EDGE_WALL_NODES = ["ID3174"]  # 0.8 m plate on the pinwheel's first edge (x=4.93, y<1.3); deleted too
# mobility A: the two 4"x4" lumber lengths laid across the walkway behind the door (each is two stacked
# 2x4-thick orange beams, 0.10 m tall, forming a V with its apex at the pinwheel partition).
LUMBER_NODES = ["ID4842", "ID4851", "ID4858", "ID4865"]
DEXTERITY_REMOVE_NODES = (
    PINWHEEL_TERRAIN_NODES + SMALL_WALL_NODES + RAMP_EDGE_WALL_NODES + LUMBER_NODES
)

# Deck panels that sit above the 0.003 m deck of the rest of the apparatus: the pinwheel's base sheets
# (top 0.014) and the sheet behind the door wall (top 0.029, on joists). Lowered so the floor is flat
# and continuous at the z of the start-side deck.
PINWHEEL_BASE_NODES = ["ID3722", "ID4102", "ID3507", "ID3912"]
BEHIND_DOOR_DECK_NODES = [
    "ID4302", "ID4293", "ID4336", "ID66_8", "ID75_8", "ID4319", "ID4310", "ID4327",  # joists
    "ID4347", "ID4347_1", "ID4355", "ID4355_1",  # sheet + its lines
]  # fmt: skip
# Pieces resting on the behind-the-door sheet (x 7.37-8.59): backing panel of the button boards, floor
# braces, wall sill blocks and the lightbulb board with its bulbs. Lowered with the sheet.
BEHIND_DOOR_FIXTURE_NODES = [
    "ID4805", "ID4814", "ID3408", "ID3353", "ID5059",
    "ID5068", "ID5077", "ID5089", "ID5098", "ID5107", "ID5114", "ID5121", "ID5128", "ID5136",
    "ID5136_1", "ID5136_10", "ID5136_11", "ID5136_12", "ID5136_13", "ID5136_14", "ID5136_15",
    "ID5136_16", "ID5136_17", "ID5136_2", "ID5136_3", "ID5136_4", "ID5136_5", "ID5136_6",
    "ID5136_7", "ID5136_8", "ID5136_9", "ID5144", "ID5144_1", "ID5144_2", "ID5144_3",
    "ID5144_4", "ID5144_5", "ID5144_6", "ID5144_7", "ID5144_8", "ID5150", "ID5150_1",
    "ID5150_2", "ID5150_3", "ID5150_4", "ID5150_5", "ID5150_6", "ID5150_7", "ID5150_8",
    "ID5388", "ID5397", "ID5404", "ID5411", "ID5418", "ID5425", "ID5437", "ID5443", "ID5449",
]  # fmt: skip
# Button boards (floor-level tilted board + wall board on a post) stand on a raised pinwheel tile
# (top 0.192). Lowered onto the deck once the tile is gone.
BUTTON_BOARD_NODES = [
    "ID4528", "ID4543", "ID4550", "ID4550_1", "ID4573", "ID4580", "ID4580_1", "ID4594",
    "ID4601", "ID4601_1", "ID4615", "ID4622", "ID4622_1", "ID4636", "ID4643", "ID4643_1",
    "ID4657", "ID4666", "ID4680", "ID4691", "ID4701", "ID4708", "ID4708_1", "ID4722", "ID4729",
    "ID4729_1", "ID4743", "ID4750", "ID4750_1", "ID4764", "ID4771", "ID4771_1", "ID4785",
    "ID4792", "ID4792_1",
]  # fmt: skip
# Door-wall studs/sheets (bottom at z=0.029, on the behind-the-door sheet). Their bottoms are extended
# down instead of moving them, so the keypad panel and doorframe keep their coordinates.
DOOR_WALL_NODES = ["ID3399", "ID3417", "ID3432", "ID3432_1", "ID3344", "ID3362", "ID3371", "ID3306"]
# Zero-thickness seam lines on the behind-the-door sheet; as 1 cm boxes they would be a 5 mm lip across the
# walkway, so in `dexterity` they are visual only (no collision).
SEAM_LINE_NODES = ["ID4355", "ID4355_1"]
DECK_LOW_TOP = 0.003  # m, deck top of the sheets that stay in place (ID85*, start-side)
PINWHEEL_BASE_DZ = 0.003 - 0.014  # sheet top -> deck top
BEHIND_DOOR_DZ = 0.003 - 0.029
BUTTON_BOARD_DZ = 0.003 - 0.192


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


def apply_dexterity_layout(world: dict) -> None:
    """Remove the mobility terrain from `world` {node: (mesh, rgba)} and flatten the deck (in place)."""
    missing = [n for n in DEXTERITY_REMOVE_NODES if n not in world]
    assert not missing, f"layout tables out of date, unknown nodes: {missing}"
    for n in DEXTERITY_REMOVE_NODES:
        del world[n]
    for ids, dz in (
        (PINWHEEL_BASE_NODES, PINWHEEL_BASE_DZ),
        (BEHIND_DOOR_DECK_NODES + BEHIND_DOOR_FIXTURE_NODES, BEHIND_DOOR_DZ),
        (BUTTON_BOARD_NODES, BUTTON_BOARD_DZ),
    ):
        for n in ids:
            world[n][0].apply_translation([0, 0, dz])
    for n in DOOR_WALL_NODES:  # extend the bottom (verts at the old deck height) down to the new deck
        v = world[n][0].vertices
        v[v[:, 2] < 0.0295, 2] += BEHIND_DOOR_DZ
    print(f"dexterity layout: removed {len(DEXTERITY_REMOVE_NODES)} terrain nodes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dae", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--layout",
        choices=LAYOUTS,
        default="dexterity",
        help="dexterity: manipulation-trial apparatus, terrain removed and deck flat (default); "
        "mobility / full: the CAD as modelled (mobility terrain + all stations)",
    )
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
    if args.layout == "dexterity":
        apply_dexterity_layout(world)
    print(f"layout={args.layout}")
    print(f"{len(world)} instances; extent {np.max([m.bounds[1] for m, _ in world.values()], axis=0)}")

    # Cut the task parts out of the static scene.
    taken = {}
    take_prefix = [DOOR_NODE, *HANDLE_NODES] + [n for pair in KEYPAD_NODES.values() for n in pair]
    for node in list(world):
        if any(node == t or node.startswith(t + "_") for t in take_prefix):
            taken[node] = world.pop(node)

    # Lightbulb task parts: fixtures, bulbs and the rack plate leave the static hull set.
    bulb_ids = [n for pair in INSTALLED_BULB_NODES + SPARE_BULB_NODES for n in pair]
    markers = [n for n in world if n.startswith(MARKER_PREFIXES) and len(world[n][0].faces) <= 6]
    bulb_src = {n: world.pop(n) for n in FIXTURE_NODES + bulb_ids + RACK_PLATE_NODES + RACK_COLLAR_NODES}
    for n in markers:
        del world[n]

    def parts(*ids):
        return [taken[n] for n in taken if any(n == i or n.startswith(i + "_") for i in ids)]

    out = {"textures": [], "mats": [], "assets": [], "geoms": []}
    assets, geoms = out["assets"], out["geoms"]

    # Static collision hulls (the doorframe trim is replaced by jambs + header: its hull
    # would plug the doorway).
    for i, (node, (m, _)) in enumerate(world.items()):
        name = f"col_{i}"
        if args.layout == "dexterity" and node in SEAM_LINE_NODES:
            continue
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
    # Flat white box shells (outline/edge duplicates lying in the deck and sheet tops, e.g. ID85_1,
    # ID4347_1) z-fight with the textured sheet they trace; they are collision-only.
    flat_shells = [
        n for n, (m, r) in world.items()
        if len(m.faces) <= 18 and m.extents[2] < 0.02 and tuple(r) == (1.0, 1.0, 1.0, 1.0)
    ]  # fmt: skip
    visible = [v for n, v in world.items() if n not in flat_shells]
    for name in write_visual("vis", visible, mesh_dir, out, args.visual_faces):
        geoms.append(vis_geom(name))

    # ---- Door: hinge on the CAD slab's hinge edge, re-posed from its open CAD pose to closed.
    # It opens toward +x, the start side (rules PDF, 'view from the starting location'), so the
    # hinge axis is -z: positive angle swings the free edge toward +x.
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
    # Collision: a convex hull of the L-shaped lever grip would fill the gap between the bar and
    # the door, which a finger has to hook into. Split the grip (the handle part with the longest
    # reach along y) into its neck + rose and its bar -- one box each -- and keep hulls for the
    # rest. In the handle frame the bar runs toward -y from the spindle.
    local_parts = [closed_local(m, T_handle) for m, _ in handle_items]
    grip_k = int(np.argmax([lm.extents[1] if lm.extents[0] > 0.05 else 0 for lm in local_parts]))
    for k, lm in enumerate(local_parts):
        if k == grip_k:
            v = lm.vertices
            for suffix, sel in (("bar", v[:, 1] < -0.015), ("neck", v[:, 1] >= -0.015)):
                lo_, hi_ = v[sel].min(0), v[sel].max(0)
                c_, h_ = (lo_ + hi_) / 2, np.maximum((hi_ - lo_) / 2, 0.003)
                handle_geoms.append(
                    f'<geom name="handle_col_{k}_{suffix}" type="box" '
                    f'pos="{c_[0]:g} {c_[1]:g} {c_[2]:g}" size="{h_[0]:g} {h_[1]:g} {h_[2]:g}" '
                    'mass="0.03" group="3" rgba="0 0 0 0"/>'
                )
            continue
        hull_or_box(lm).export(mesh_dir / f"handle_col_{k}.obj")
        assets.append(f'<mesh name="handle_col_{k}" file="meshes/handle_col_{k}.obj"/>')
        handle_geoms.append(
            f'<geom name="handle_col_{k}" type="mesh" mesh="handle_col_{k}" mass="0.02" '
            'group="3" rgba="0 0 0 0"/>'
        )
    sp = spindle_closed - hinge_pos  # handle origin in the door frame
    door_xml = (
        f'<body name="door" pos="{hinge_pos[0]:g} {hinge_pos[1]:g} 0">\n'
        f'        <joint name="door_hinge" type="hinge" axis="0 0 -1" range="0 {np.degrees(1.9):.1f}" '
        'stiffness="0.6" damping="25" springref="0"/>\n        '
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


    # ---- Lightbulb task: threaded fixtures + bulbs. Every bulb shares one local mesh (frame: origin
    # at the base end, +z along the bulb toward its tip). A fixture's carrier body `thread{i}` has a
    # hinge about -z (positive = tightening, a right-hand thread) and a slide along -z tied to it by
    # THREAD_PITCH per turn; at turn 0 it sits backed out, at SEAT_DEG its frame is the CAD pose. An
    # installed bulb's geoms ride on its carrier. A spare bulb is a free body; once the task sees it
    # aligned in an empty fixture it activates the weld `seat_{spare}_{fixture}` to that fixture's
    # carrier, so turning the bulb screws it in.
    from scipy.spatial.transform import Rotation as Rot

    def bulb_frame(meshes, expect):
        """(base-end centre, unit tip axis, 3x3 frame with z = axis) of a CAD bulb."""
        v = np.concatenate([m.vertices for m in meshes])
        c = v.mean(0)
        a = np.linalg.svd(v - c, full_matrices=False)[2][0]
        t = (v - c) @ a
        radius = np.linalg.norm((v - c) - np.outer(t, a), axis=1)
        end = lambda sel: radius[sel].max()  # noqa: E731
        if end(t > t.max() - 0.006) < end(t < t.min() + 0.006):  # the tip end is the round, wide one
            a = -a
            t = -t
        base = v[t < t.min() + 0.002].mean(0)
        assert a @ expect > 0.9, (a, expect)
        x = np.cross([0, 0, 1.0], a)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
        return base, a, np.stack([x, np.cross(a, x), a], axis=1)

    def mat4(Rm, origin):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rm, origin
        return T

    def quat_wxyz(Rm):
        q = Rot.from_matrix(Rm).as_quat()
        return f"{q[3]:g} {q[0]:g} {q[1]:g} {q[2]:g}"

    wall_up = np.array([-1.0, 0, 0])  # wall-board bulbs point out of the wall, toward the robot
    floor_up = np.array([-1.0, 0, 1.0]) / np.sqrt(2)
    frames = []  # per fixture: (base, axis, R)
    for i, ids in enumerate(INSTALLED_BULB_NODES):
        frames.append(bulb_frame([bulb_src[ids[0]][0]], wall_up if i < 5 else floor_up))
    spare_frames = [
        bulb_frame([bulb_src[ids[0]][0]], np.array([0, 0, 1.0])) for ids in SPARE_BULB_NODES
    ]
    # One shared local bulb mesh (decimated visual + a base hull and a globe hull for contact).
    t_bulb = mat4(spare_frames[0][2], spare_frames[0][0])
    bulb_local = to_frame(bulb_src[SPARE_BULB_NODES[0][0]][0], t_bulb)
    bulb_rgba = bulb_src[SPARE_BULB_NODES[0][0]][1]
    bulb_vis = [
        vis_geom(n)
        for n in write_visual("bulb_vis", [(bulb_local, bulb_rgba)], mesh_dir, dout, face_budget=8000)
    ]
    face_z = bulb_local.vertices[bulb_local.faces][:, :, 2].mean(1)
    bulb_col = []
    for k, (lo_z, hi_z) in enumerate([(-1.0, 0.045), (0.040, 1.0)]):  # thread + neck, globe
        sel = (face_z >= lo_z) & (face_z < hi_z)
        hull = trimesh.PointCloud(np.unique(bulb_local.vertices[bulb_local.faces[sel]].reshape(-1, 3), axis=0)).convex_hull
        hull.export(mesh_dir / f"bulb_col_{k}.obj")
        assets.append(f'<mesh name="bulb_col_{k}" file="meshes/bulb_col_{k}.obj"/>')
        bulb_col.append(
            f'<geom name="@@_col_{k}" type="mesh" mesh="bulb_col_{k}" mass="{BULB_MASS / 2:g}" '
            f'friction="{BULB_FRICTION:g} 0.02 0.002" group="3" rgba="0 0 0 0"/>'
        )
    print(f"bulb mesh: extents {np.round(bulb_local.extents, 4)}, tip at {bulb_local.bounds[1][2]:.3f}")

    seat_rad = np.radians(SEAT_DEG)
    slide_max = THREAD_PITCH * seat_rad / (2 * np.pi)
    inertial = f'<inertial pos="0 0 0.06" mass="{BULB_MASS:g}" diaginertia="3e-5 3e-5 1.5e-5"/>'
    fixture_xml, thread_xml, bulb_excludes, bulb_eq = [], [], [], []
    for i, node in enumerate(FIXTURE_NODES):
        fm, frgba = bulb_src[node]
        names = write_visual(f"fixture{i}_vis", [(fm, frgba)], mesh_dir, dout, face_budget=6000)
        fg = [vis_geom(n) for n in names]
        hull_or_box(fm).export(mesh_dir / f"fixture{i}_col.obj")
        assets.append(f'<mesh name="fixture{i}_col" file="meshes/fixture{i}_col.obj"/>')
        fg.append(
            f'<geom name="fixture{i}_col" type="mesh" mesh="fixture{i}_col" group="3" rgba="0 0 0 0"/>'
        )
        fixture_xml.append("\n      ".join(fg))
        base, a, Rm = frames[i]
        p0 = base + a * slide_max  # carrier frame at turn 0: backed out by the travel to the seat
        geoms_i = (
            [] if i in EMPTY_FIXTURES
            else [g.replace("<geom name=\"bulb_vis", f"<geom name=\"bulb{i}_vis") for g in bulb_vis]
            + [g.replace("@@", f"bulb{i}") for g in bulb_col]
        )  # fmt: skip
        thread_xml.append(
            f'<body name="thread{i}" pos="{p0[0]:g} {p0[1]:g} {p0[2]:g}" quat="{quat_wxyz(Rm)}">\n'
            f'        <joint name="thread{i}_slide" type="slide" axis="0 0 -1" range="0 {slide_max:g}"/>\n'
            f'        <joint name="thread{i}_turn" type="hinge" axis="0 0 -1" '
            f'range="0 {STOP_DEG:g}" frictionloss="{THREAD_FRICTION:g}" damping="{THREAD_DAMPING:g}" armature="{THREAD_ARMATURE:g}"/>\n        '
            + inertial + "\n        " + "\n        ".join(geoms_i) + "\n      </body>"
        )  # fmt: skip
        bulb_eq.append(
            f'<joint joint1="thread{i}_slide" joint2="thread{i}_turn" '
            f'polycoef="0 {THREAD_PITCH / (2 * np.pi):g} 0 0 0" solref="0.004 1" solimp="0.99 0.99 0.001"/>'
        )
        bulb_excludes.append(f'<exclude body1="fixture{i}" body2="thread{i}"/>')
    # `range` of a hinge is in degrees (compiler default); polycoef couples radians -- the slide
    # travel per turn is THREAD_PITCH, so q_slide = THREAD_PITCH / 2pi * q_turn(rad).

    spare_xml = []
    for k, (base, a, Rm) in enumerate(spare_frames):
        spare_xml.append(
            f'<body name="spare{k}" pos="{base[0]:g} {base[1]:g} {base[2]:g}" quat="{quat_wxyz(Rm)}">\n'
            f'        <freejoint name="spare{k}_free"/>\n        ' + inertial + "\n        "
            + "\n        ".join(
                [g.replace("<geom name=\"bulb_vis", f"<geom name=\"spare{k}_vis") for g in bulb_vis]
                + [g.replace("@@", f"spare{k}") for g in bulb_col]
            )
            + "\n      </body>"
        )  # fmt: skip
        for j in EMPTY_FIXTURES:
            bulb_excludes.append(f'<exclude body1="fixture{j}" body2="spare{k}"/>')
            bulb_eq.append(
                f'<weld name="seat_{k}_{j}" body1="spare{k}" body2="thread{j}" active="false" '
                'relpose="0 0 0 1 0 0 0" anchor="0 0 0" '
                'solref="0.008 1" solimp="0.95 0.99 0.001"/>'
            )

    # Rack plate: a plate with a square hole per spare bulb (a hull would be a solid slab).
    (px0, py0, pz0), (px1, py1, pz1) = bulb_src[RACK_PLATE_NODES[0]][0].bounds
    hx = sorted(b[0] for b, _, _ in spare_frames)
    hy = float(np.mean([b[1] for b, _, _ in spare_frames]))
    cuts = [(px0, hx[0] - RACK_HOLE)] + [
        (u + RACK_HOLE, v - RACK_HOLE) for u, v in zip(hx[:-1], hx[1:])
    ] + [(hx[-1] + RACK_HOLE, px1)]
    boxes = [((px0, px1), (py0, hy - RACK_HOLE)), ((px0, px1), (hy + RACK_HOLE, py1))]
    boxes += [((u, v), (hy - RACK_HOLE, hy + RACK_HOLE)) for u, v in cuts]
    for n, ((u0, u1), (v0, v1)) in enumerate(boxes):
        geoms.append(
            f'<geom name="rack_plate_{n}" type="box" pos="{(u0 + u1) / 2:g} {(v0 + v1) / 2:g} {(pz0 + pz1) / 2:g}" '
            f'size="{(u1 - u0) / 2:g} {(v1 - v0) / 2:g} {(pz1 - pz0) / 2:g}" group="3" rgba="0 0 0 0"/>'
        )
    for name in write_visual(
        "rack_vis",
        [bulb_src[n] for n in RACK_PLATE_NODES + RACK_COLLAR_NODES],
        mesh_dir,
        out,
        args.visual_faces,
    ):
        geoms.append(vis_geom(name))
    excludes += bulb_excludes

    textures, mats = out["textures"], out["mats"]

    xml = (
        '<mujoco model="loco_manipulation_testbed">\n'
        "  <asset>\n    " + "\n    ".join(textures + mats + assets) + "\n  </asset>\n"
        "  <contact>\n    " + "\n    ".join(excludes) + "\n  </contact>\n"
        "  <equality>\n    " + "\n    ".join(bulb_eq) + "\n  </equality>\n"
        "  <worldbody>\n"
        '    <body name="testbed">\n      ' + "\n      ".join(geoms) + "\n    </body>\n"
        + "".join(
            f'    <body name="fixture{i}">\n      {g}\n    </body>\n'
            for i, g in enumerate(fixture_xml)
        )
        + "    " + door_xml + "\n    " + "\n    ".join(key_xml) + "\n"
        "    " + "\n    ".join(thread_xml) + "\n    " + "\n    ".join(spare_xml) + "\n"
        "  </worldbody>\n</mujoco>\n"
    )
    (args.out_dir / "testbed.xml").write_text(xml)
    print("wrote", args.out_dir / "testbed.xml")

    # Scene wrapper the molmospaces "user" scene loader consumes: testbed + ground + light,
    # an offscreen framebuffer big enough for the robot cameras, and empty scene metadata.
    cx, cy = (np.max([m.bounds[1] for m, _ in world.values()], axis=0)[:2] / 2)
    (args.out_dir / "challenge_scene.xml").write_text(
        f'''<mujoco model="ieee_humanoids_2026_challenge">
  <!-- layout: {args.layout} -->
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
    (args.out_dir / "challenge_scene_metadata.json").write_text(f'{{"layout": "{args.layout}", "objects": {{}}}}\n')
    print("wrote", args.out_dir / "challenge_scene.xml")


if __name__ == "__main__":
    main()
