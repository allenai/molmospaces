"""AABBMap -- an occupancy map built from the axis-aligned bounding boxes of a
scene's floor geoms, rendered through MuJoCo's segmentation renderer.

From the FetchMan repo (`g1_molmo`, `components/occupancy_map.py::OccupancyMap`),
relocated here while dissolving `projects/fetchman/`. FetchMan goal/spawn sampling is
verified bit-exact against that repo and reads *this* grid, so cell-for-cell
output is load-bearing: re-run projects/fetchman/scripts/check_gold_parity.py
before changing it.

Same query API as `ProcTHORMap`/`iTHORMap` (utils/scene_maps.py, True = free) but a
*different* grid -- AABBMap frames on floor-geom AABBs and segments floor vs
non-floor; ProcTHORMap renders orthographic depth of the whole scene. They disagree
cell for cell, so they are selectable, not interchangeable (see
`OccupancyMapImpl` / `CPUMujocoEnv.get_occupancy_map(impl=...)`); "thor" is the
default implementation everywhere, and only G1/FetchMan should select "aabb".

How different, concretely: on procthor val_0 and val_1 the two agree on 98.6-99.6%
of free space (IoU), differing in a boundary sliver around obstacles and the odd
blob where the depth threshold in ProcTHORMap's `_get_occupancy_from_orthoview`
and this file's floor segmentation disagree. Neither is more correct. What always
differs is the framing -- `generate` below frames on the floor AABB + 2m of
padding, ProcTHORMap on the whole rendered scene -- so the same world point lands
on different (row, col) in each, and pixel indices from one are meaningless in the
other.

Two things this implementation does have over ProcTHORMap, neither of them a
reason to switch a non-FetchMan experiment on its own:

  * It finds floors more permissively. `_floor_geom_ids` takes "room|"/"room_"
    prefixes, plus any geom whose name contains "floor" with contype == 0, plus a
    fallback pass for a geom named exactly "floor". ProcTHORMap accepts only the
    "room|"/"room_" prefixes and asserts if it finds none -- it needs those names
    to build its room map. So on a scene outside that naming convention this may
    be the only one of the two that builds.
  * It is slightly faster, and its cache outlives the process. ~0.3s vs ~0.4s to
    build a 100px/m map of a procthor val scene (ProcTHORMap's first call in a
    process is ~2.1s, paying GL init), and `generate` writes the grid to a PNG
    next to the scene XML, so a later process reloads it for free where
    ProcTHORMap re-renders. `CPUMujocoEnv` memory-caches either one per scene, so
    in a single run this is worth a fraction of a second.

Both carry a room map, from the same floor segmentation, but they spell it
differently: ProcTHORMap has `room_ids_to_name`, `get_free_points_by_room` and
room-scoped `label_at`, while this one exposes the point queries
`room_at`/`room_near`/`room_of`/`same_room`. A pixel here holds the scene's own
room number -- `room_7` is 7 -- where ProcTHORMap stores `geom_id + 1`, which
does not survive the uint8 channel either map caches its rooms in. Only the
"room|"/"room_" floor geoms become rooms here; `_floor_geom_ids` also matches
anything merely named "...floor...", which on a procthor scene picks up every
floorlamp. `label_at` here stays what it was: a plain connected component of
free space.

Its own module because importing it monkeypatches `mujoco.renderer.Renderer.render`
process-globally (MuJoCo sizes its segid table by `scene.ngeom`, which IndexErrors
on dense scenes). scene_maps.py is imported by env.py and thus by everything, so
folding this in would apply that patch everywhere; here it is imported lazily
inside `get_occupancy_map`, only when "aabb" is selected.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy.spatial.transform import Rotation as R

# Patch MuJoCo's segmentation renderer to size segid2output by the actual max
# segid, not scene.ngeom (decorator/skybox ids exceed it on dense scenes and
# IndexError). Guarded: newer mujoco builds move or drop the class.
try:
    import mujoco.renderer as _mj_cls_renderer
except ImportError:
    _mj_cls_renderer = None

if _mj_cls_renderer is not None and not getattr(_mj_cls_renderer.Renderer, "_segid_patched", False):
    _orig_render = _mj_cls_renderer.Renderer.render

    def _safe_render(self, *, out=None):
        if not self._segmentation_rendering:
            return _orig_render(self, out=out)
        original_flags = np.copy(self._scene.flags)
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        if self._mjr_context is None:
            raise RuntimeError("render cannot be called after close.")
        if self._gl_context:
            self._gl_context.make_current()
        if out is None:
            out = np.empty((self._height, self._width, 3), dtype=np.uint8)
        mujoco.mjr_render(self._rect, self._scene, self._mjr_context)
        mujoco.mjr_readPixels(out, None, self._rect, self._mjr_context)
        image3 = out.astype(np.uint32)
        segimage = image3[:, :, 0] + image3[:, :, 1] * (2**8) + image3[:, :, 2] * (2**16)
        ngeoms = self._scene.ngeom
        max_id = int(segimage.max()) if segimage.size else 0
        table_size = max(ngeoms + 1, max_id + 1)
        segid2output = np.full((table_size, 2), fill_value=-1, dtype=np.int32)
        visible_geoms = [g for g in self._scene.geoms[:ngeoms] if g.segid != -1]
        if visible_geoms:
            vs = np.array([g.segid + 1 for g in visible_geoms], np.int32)
            segid2output[vs, 0] = np.array([g.objid for g in visible_geoms], np.int32)
            segid2output[vs, 1] = np.array([g.objtype for g in visible_geoms], np.int32)
        result = segid2output[segimage]
        np.copyto(self._scene.flags, original_flags)
        if self._gl_context:
            result = np.flipud(result)
        return result

    _mj_cls_renderer.Renderer.render = _safe_render
    _mj_cls_renderer.Renderer._segid_patched = True


def _strip_bodies(xml_path, name_contains):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stripped_names = set()

    def _collect_body_names(elem):
        name = elem.get("name", "")
        if name:
            stripped_names.add(name)
        for child in elem:
            if child.tag == "body":
                _collect_body_names(child)

    def _remove_matching(parent):
        to_remove = []
        for child in parent:
            if child.tag == "body":
                name = child.get("name", "")
                if any(kw in name.lower() for kw in name_contains):
                    to_remove.append(child)
                    _collect_body_names(child)
                    continue
            _remove_matching(child)
        for c in to_remove:
            parent.remove(c)

    _remove_matching(root)
    # Remove contact/exclude pairs that reference stripped bodies.
    if stripped_names:
        for tag in ("contact", "equality"):
            for section in root.iter(tag):
                to_remove = []
                for child in section:
                    attrs = (child.get("body1", ""), child.get("body2", ""), child.get("body", ""))
                    if any(a in stripped_names for a in attrs):
                        to_remove.append(child)
                for c in to_remove:
                    section.remove(c)
    out = xml_path.parent / (xml_path.stem + "_tmp_noceiling.xml")
    tree.write(str(out), xml_declaration=True)
    return out


def _floor_geom_ids(model):
    ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("room|") or name.startswith("room_"):
            ids.append(i)
        elif "floor" in name.lower() and model.geom_contype[i] == 0:
            ids.append(i)
    if not ids:
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name == "floor":
                ids.append(i)
    return ids


_ROOM_NUMBER_RE = re.compile(r"room[|_](\d+)")


def _room_number(geom_name: str, body_name: str) -> int | None:
    """The scene's own number for a room, from `room_7` / `room|7|...`, or None
    where the name carries no number a uint8 map channel can hold. Never a
    number of our own: a pixel value here is the room number in the scene."""
    for name in (body_name, geom_name):
        match = _ROOM_NUMBER_RE.match(name or "")
        if match:
            number = int(match.group(1))
            if 1 <= number <= 255:
                return number
    return None


def _geom_aabb(model, data, geom_ids):
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for gid in geom_ids:
        pos = data.geom_xpos[gid]
        size = model.geom_size[gid]
        gtype = model.geom_type[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = size
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            meshid = model.geom_dataid[gid]
            vert_start = model.mesh_vertadr[meshid]
            vert_count = model.mesh_vertnum[meshid]
            verts = model.mesh_vert[vert_start : vert_start + vert_count]
            mat = data.geom_xmat[gid].reshape(3, 3)
            world_verts = (mat @ verts.T).T + pos
            mins = np.minimum(mins, world_verts.min(axis=0))
            maxs = np.maximum(maxs, world_verts.max(axis=0))
            continue
        else:
            half = np.array([size[0]] * 3)
        mins = np.minimum(mins, pos - half)
        maxs = np.maximum(maxs, pos + half)
    center = (mins + maxs) / 2.0
    full_size = maxs - mins
    return center, full_size


def _circular_kernel(radius):
    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


def _fetchman_map_path(xml_path) -> Path:
    """FetchMan's own cache filename for a scene's map, kept exactly as it was
    in g1_molmo (`<scene>_thormap.png`) so existing caches stay valid. Note the
    name predates ProcTHORMap and has nothing to do with it."""
    xml_path = Path(xml_path)
    return xml_path.with_name(xml_path.stem + "_thormap.png")


class AABBMap:
    def __init__(
        self,
        occupancy,
        world_to_map,
        map_to_world,
        px_per_m,
        agent_radius=None,
        room_map=None,
        room_ids_to_name=None,
    ):
        self.occupancy = occupancy
        self.world_to_map = world_to_map
        self.map_to_world = map_to_world
        self.px_per_m = px_per_m
        # Per-pixel room id (0 = no room: an occluded cell, or a scene whose
        # floors carry no room names), and the id -> room body name it decodes
        # to. Same pair ProcTHORMap carries; see room_at/room_near.
        self.room_map = room_map
        self.room_ids_to_name = room_ids_to_name or {}
        # Radius the obstacles were already inflated by when this grid was
        # rendered, or None for a map cached before that was recorded. Read by
        # from_model_path to decide whether the cache is reusable.
        self.agent_radius = agent_radius

    @staticmethod
    def generate(xml_path, agent_radius=0.15, px_per_m=200, force=False, out_path=None) -> Path:
        xml_path = Path(xml_path)
        out_path = Path(out_path) if out_path is not None else _fetchman_map_path(xml_path)
        if out_path.exists() and not force:
            return out_path
        tmp_xml = _strip_bodies(xml_path, ["ceiling", "light"])
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        finally:
            tmp_xml.unlink(missing_ok=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        floor_ids = _floor_geom_ids(model)
        assert floor_ids, f"No floor geoms found in {xml_path}"

        # Door panels are hinge joints w/ nonzero qpos0; passable opening = doorframe - door.
        parent_to_child = {}
        for body_id in range(model.nbody):
            root_id = model.body_rootid[body_id]
            root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_id) or ""
            if root_name.startswith("door_") or root_name.startswith("doorway_"):
                parent_to_child.setdefault(root_id, []).append(body_id)

        door_body_ids = []
        doorway_body_ids = []
        for _root_body_id, children in parent_to_child.items():
            for child_id in children:
                jntadr = model.body(child_id).jntadr.item()
                if (
                    jntadr >= 0
                    and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE
                    and model.joint(jntadr).qpos0.item() != 0.0
                ):
                    door_body_ids.append(child_id)
                    doorway_body_ids.extend(children)
                if jntadr < 0 and len(children) == 2:
                    doorway_body_ids.append(child_id)

        door_geom_ids = []
        doorframe_geom_ids = []
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            parent_bid = model.body(bid).parentid.item()
            root_bid = model.body_rootid[bid]
            if bid in door_body_ids or parent_bid in door_body_ids:
                door_geom_ids.append(gid)
            if root_bid in doorway_body_ids:
                doorframe_geom_ids.append(gid)

        center, size = _geom_aabb(model, data, floor_ids)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
            all_geoms = [
                i for i in range(model.ngeom) if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
            ]
            center, size = _geom_aabb(model, data, all_geoms)
        size += np.array([2.0, 2.0, 0.0])
        h = max(round(px_per_m * size[0]), 1)
        w = max(round(px_per_m * size[1]), 1)
        effective_px = h / size[0]
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)
        renderer = mujoco.Renderer(model, height=h, width=w, max_geom=max(20000, model.ngeom * 4))
        renderer.enable_segmentation_rendering()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = center
        cam.distance = 5.0
        cam.azimuth = 0
        cam.elevation = -90
        scene = mujoco.MjvScene(model, maxgeom=10000)
        opt = mujoco.MjvOption()
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        for i in range(2):
            scene.camera[i].orthographic = 1
            scene.camera[i].frustum_bottom = -size[0] / 2
            scene.camera[i].frustum_top = size[0] / 2
        renderer.update_scene(data, camera=cam)
        for i in range(2):
            renderer._scene.camera[i].orthographic = 1
            renderer._scene.camera[i].frustum_bottom = -size[0] / 2
            renderer._scene.camera[i].frustum_top = size[0] / 2
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        seg_geom = seg[:, :, 0].astype(int)
        cam_to_world = np.eye(4)
        cam_to_world[:3, 3] = renderer._scene.camera[0].pos
        cam_x = np.cross(renderer._scene.camera[0].up, -renderer._scene.camera[0].forward)
        cam_to_world[:3, :3] = np.column_stack(
            (cam_x, renderer._scene.camera[0].up, -renderer._scene.camera[0].forward)
        )
        renderer.close()
        occupancy_occupied = np.ones(seg_geom.shape, dtype=bool)
        for fid in floor_ids:
            occupancy_occupied &= seg_geom != fid

        # Room map: the same floor segmentation, kept per room instead of
        # collapsed into one mask. A pixel holds the scene's own room number
        # (room_7 -> 7), not an index of our own and not ProcTHORMap's
        # geom_id + 1, which overflows the uint8 PNG channel this is cached in.
        # Only the "room|"/"room_" floor geoms are rooms: `_floor_geom_ids` also
        # takes anything whose name merely contains "floor", which on a procthor
        # scene picks up every floorlamp. Those stay free space, as they always
        # have, but they are not rooms.
        room_map = np.zeros(seg_geom.shape, dtype=np.uint8)
        room_ids_to_name = {}
        unnumbered = []
        for fid in floor_ids:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, fid) or ""
            if not geom_name.startswith(("room|", "room_")):
                continue
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[fid]))
                or geom_name
            )
            number = _room_number(geom_name, body_name)
            if number is None:
                unnumbered.append((fid, body_name))
                continue
            room_map[seg_geom == fid] = number
            room_ids_to_name[number] = body_name
        # A room the scene did not number (or numbered past 255) still needs an
        # id to be told apart from its neighbours; it gets the lowest free one.
        for fid, body_name in unnumbered:
            free = next((i for i in range(1, 256) if i not in room_ids_to_name), None)
            if free is None:
                print(f"AABBMap: no room id left for {body_name!r} in {xml_path.name}")
                break
            print(f"AABBMap: {body_name!r} in {xml_path.name} has no room number, using {free}")
            room_map[seg_geom == fid] = free
            room_ids_to_name[free] = body_name

        # Door opening mask = doorframe minus door panel. Cleared from obstacles below.
        occ_door = np.zeros(seg_geom.shape, dtype=bool)
        for did in door_geom_ids:
            occ_door[seg_geom == did] = True
        occ_doorframe = np.zeros(seg_geom.shape, dtype=bool)
        for did in doorframe_geom_ids:
            occ_doorframe[seg_geom == did] = True
        occ_door_path = occ_doorframe & ~occ_door
        occ_door_path = cv2.dilate(
            occ_door_path.astype(np.uint8),
            _circular_kernel(15),
        ).astype(bool)
        occupancy_occupied[occ_door_path] = False

        if agent_radius is not None and agent_radius > 0:
            rad_px = max(1, int(agent_radius * effective_px))
            kernel = _circular_kernel(rad_px)
            occupancy_occupied = cv2.dilate(occupancy_occupied.astype(np.uint8), kernel).astype(
                bool
            )
        occupancy = ~occupancy_occupied
        cam_to_map = np.array([[0, -effective_px, 0, h / 2], [effective_px, 0, 0, w / 2]])
        cam_inv = np.eye(4)
        cam_inv[:3, :3] = cam_to_world[:3, :3].T
        cam_inv[:3, 3] = -cam_to_world[:3, :3].T @ cam_to_world[:3, 3]
        world_to_map = cam_to_map @ cam_inv
        map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
        centered_to_cam = np.array([[1 / effective_px, 0, 0], [0, 1 / effective_px, 0], [0, 0, 1]])
        cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
        cam_to_world_floor[2, 2] = 0
        map_to_world = cam_to_world_floor @ centered_to_cam @ map_to_centered
        # RGB: occupancy in R (what every existing reader takes, unchanged),
        # a spare copy in G, the room ids in B. `load` treats a map without the
        # room channel as room-less, so an older grayscale cache still reads.
        occ_u8 = occupancy.astype(np.uint8) * 255
        img = Image.fromarray(np.stack([occ_u8, occ_u8, room_map], axis=2))
        metadata = PngInfo()
        metadata.add_text("world_to_map", json.dumps(world_to_map.tolist()))
        metadata.add_text("map_to_world", json.dumps(map_to_world.tolist()))
        metadata.add_text("px_per_m", json.dumps(float(effective_px)))
        metadata.add_text("agent_radius", json.dumps(float(agent_radius or 0.0)))
        metadata.add_text("room_ids_to_name", json.dumps(room_ids_to_name))
        img.save(str(out_path), pnginfo=metadata)
        return out_path

    @classmethod
    def from_scene(cls, scene, agent_radius=0.15):
        """FetchMan's own entry point, unchanged: its own `<scene>_thormap.png`
        cache file, and an existing one wins outright. Deliberately does NOT
        re-check the cached map's agent radius the way from_model_path below
        does -- this is the call the bit-exact gold rollout goes through, so it
        keeps gold's exact cache semantics. One consequence: a `_thormap.png`
        written before the room channel existed still wins, and loads room-less.
        Delete it to get rooms; the occupancy it regenerates is bit-identical."""
        map_path = _fetchman_map_path(scene.xml_path)
        if not map_path.exists():
            cls.generate(scene.xml_path, agent_radius=agent_radius, force=True)
        return cls.load(str(map_path))

    @classmethod
    def from_model_path(cls, xml_path, agent_radius=0.15, px_per_m=200):
        """molmo_spaces' entry point (get_occupancy_map with impl="aabb"), at
        any agent radius. Caches per (radius, px_per_m) in its own file, not in
        FetchMan's radius-less `<scene>_thormap.png`, which the gold rollout reads."""
        xml_path = Path(xml_path)
        # v2: carries the room channel. A v1 cache cannot be told apart from a
        # scene that genuinely has no rooms, so it gets a new name rather than
        # a silently room-less map.
        map_path = xml_path.with_name(
            f"{xml_path.stem}_aabbmap_v2_r{float(agent_radius):g}_p{int(px_per_m)}.png"
        )
        if map_path.exists():
            cached = cls.load(str(map_path))
            if cached.agent_radius is not None and np.isclose(
                cached.agent_radius, float(agent_radius)
            ):
                return cached
        cls.generate(
            xml_path,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
            force=True,
            out_path=map_path,
        )
        return cls.load(str(map_path))

    @classmethod
    def load(cls, path):
        img = Image.open(path)
        arr = np.array(img)
        room_map = None
        if arr.ndim == 3:
            occ_raw = arr[:, :, 0]
            if arr.shape[2] >= 3:
                room_map = arr[:, :, 2]
        else:
            occ_raw = arr
        occupancy = occ_raw > 0
        world_to_map = np.array(json.loads(img.info["world_to_map"]))
        map_to_world = np.array(json.loads(img.info["map_to_world"]))
        px_per_m = float(json.loads(img.info["px_per_m"]))
        radius_meta = img.info.get("agent_radius")
        agent_radius = float(json.loads(radius_meta)) if radius_meta is not None else None
        rooms_meta = img.info.get("room_ids_to_name")
        room_ids_to_name = (
            {int(k): v for k, v in json.loads(rooms_meta).items()} if rooms_meta else {}
        )
        if not room_ids_to_name:
            room_map = None
        return cls(
            occupancy,
            world_to_map,
            map_to_world,
            px_per_m,
            agent_radius=agent_radius,
            room_map=room_map,
            room_ids_to_name=room_ids_to_name,
        )

    def _world_to_px(self, xy):
        pos = np.array([xy[0], xy[1], 0.0, 1.0])
        rc = self.world_to_map @ pos
        return np.round(rc).astype(int)

    def is_free(self, xy):
        rc = self._world_to_px(xy)
        r, c = rc[0], rc[1]
        h, w = self.occupancy.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return bool(self.occupancy[r, c])

    def dilated(self, extra_radius_m):
        """Return a copy with obstacles inflated by extra_radius_m. World transforms shared."""
        if extra_radius_m <= 0:
            return self
        rad_px = max(1, int(extra_radius_m * self.px_per_m))
        kernel = _circular_kernel(rad_px)
        free = self.occupancy.astype(np.uint8)
        free_eroded = cv2.erode(free, kernel).astype(bool)
        return type(self)(
            free_eroded,
            self.world_to_map.copy(),
            self.map_to_world.copy(),
            self.px_per_m,
            agent_radius=(
                None if self.agent_radius is None else self.agent_radius + extra_radius_m
            ),
            # Inflation moves the obstacles, not the rooms.
            room_map=self.room_map,
            room_ids_to_name=self.room_ids_to_name,
        )

    def room_at(self, xy) -> int:
        """Room id of the world point `xy`; 0 where the map has no room there.

        The map is a top-down render, so anything that occludes the floor --
        a table, a wall, an object -- leaves 0 behind it. A point query is
        therefore only meaningful for open floor; use `room_near` for a point
        that may be under something.
        """
        if self.room_map is None:
            return 0
        rc = self._world_to_px(xy)
        r, c = int(rc[0]), int(rc[1])
        h, w = self.room_map.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return 0
        return int(self.room_map[r, c])

    def room_near(self, xy, radius_m: float = 1.0) -> int:
        """The room `xy` belongs to: the commonest non-zero room id within
        `radius_m` of it, or 0 if there is none. Resolves the occluded-pixel
        problem `room_at` documents -- an object's own footprint is never floor,
        so its room has to be read off the floor around it.
        """
        if self.room_map is None:
            return 0
        rc = self._world_to_px(xy)
        r0, c0 = int(rc[0]), int(rc[1])
        h, w = self.room_map.shape
        rad = max(1, int(radius_m * self.px_per_m))
        patch = self.room_map[
            max(0, r0 - rad) : min(h, r0 + rad + 1), max(0, c0 - rad) : min(w, c0 + rad + 1)
        ]
        ids, counts = np.unique(patch[patch > 0], return_counts=True)
        return int(ids[int(np.argmax(counts))]) if ids.size else 0

    def room_of(self, xy, radius_m: float = 1.0) -> int:
        """The room `xy` belongs to: the room map at that point, or, where the
        floor there is occluded -- under a table, under the object itself --
        the commonest room within `radius_m`. 0 if neither answers."""
        return self.room_at(xy) or self.room_near(xy, radius_m)

    def same_room(self, xy_a, xy_b, radius_m: float = 1.0) -> bool:
        """True unless `xy_a` and `xy_b` resolve to two different rooms. An
        unlabelled scene, or a point with no floor near it, answers True: this
        is a filter, and it declines to reject what it cannot see.
        """
        room_a = self.room_of(xy_a, radius_m)
        room_b = self.room_of(xy_b, radius_m)
        return room_a == 0 or room_b == 0 or room_a == room_b

    def _free_labels(self):
        lab = getattr(self, "_free_labels_cache", None)
        if lab is None:
            from scipy import ndimage

            lab, _ = ndimage.label(self.occupancy.astype(np.uint8))
            self._free_labels_cache = lab
        return lab

    def label_at(self, xy):
        """Connected-component id of the free cell at xy. 0 = occupied / out-of-bounds."""
        rc = self._world_to_px(xy)
        r, c = int(rc[0]), int(rc[1])
        h, w = self.occupancy.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return 0
        return int(self._free_labels()[r, c])

    def nearest_free_label(self, xy, max_radius_px=80):
        lab = self._free_labels()
        rc = self._world_to_px(xy)
        r0, c0 = int(rc[0]), int(rc[1])
        h, w = self.occupancy.shape
        if 0 <= r0 < h and 0 <= c0 < w and lab[r0, c0] > 0:
            return int(lab[r0, c0])
        for rad in range(1, max_radius_px + 1):
            r_lo, r_hi = max(0, r0 - rad), min(h - 1, r0 + rad)
            c_lo, c_hi = max(0, c0 - rad), min(w - 1, c0 + rad)
            sub = lab[r_lo : r_hi + 1, c_lo : c_hi + 1]
            free = sub[sub > 0]
            if free.size:
                return int(free.flat[0])
        return 0

    def same_free_component(self, xy_a, xy_b):
        """True if the free cells nearest xy_a and xy_b share a connected
        component. Both endpoints snap: on a dilated planning map the robot
        itself often stands on an "occupied" cell, and the A* snaps its start
        the same way."""
        la = self.nearest_free_label(xy_a)
        if la == 0:
            return False
        return la == self.nearest_free_label(xy_b)

    def any_free_in_annulus(self, center_xy, r_min, r_max):
        rc = self._world_to_px(center_xy)
        r0, c0 = int(rc[0]), int(rc[1])
        rad = int(np.ceil(r_max * self.px_per_m))
        h, w = self.occupancy.shape
        rlo, rhi = max(0, r0 - rad), min(h, r0 + rad + 1)
        clo, chi = max(0, c0 - rad), min(w, c0 + rad + 1)
        if rlo >= rhi or clo >= chi:
            return False
        ys, xs = np.ogrid[rlo:rhi, clo:chi]
        d2 = (ys - r0) ** 2 + (xs - c0) ** 2
        rmin_px2 = (r_min * self.px_per_m) ** 2
        ring = (d2 >= rmin_px2) & (d2 <= rad * rad)
        return bool(self.occupancy[rlo:rhi, clo:chi][ring].any())

    def sample_near(
        self, target_xy, radius_min=0.0, radius_max=0.7, max_attempts=500, np_random=None
    ):
        if np_random is None:
            np_random = np.random.default_rng()
        for _ in range(max_attempts):
            theta = np_random.uniform(0, 2 * np.pi)
            r = np_random.uniform(radius_min, radius_max)
            pt = target_xy[:2] + np.array([r * np.cos(theta), r * np.sin(theta)])
            if self.is_free(pt):
                return pt
        return None

    def sample_robot_pose(
        self,
        target_xyz,
        z_offset=0.0,
        radius_range=(0.0, 0.7),
        max_tries=10,
        np_random=None,
        yaw_noise=0.0,
    ):
        if np_random is None:
            np_random = np.random.default_rng()
        for _ in range(max_tries):
            xy = self.sample_near(
                target_xyz[:2], radius_range[0], radius_range[1], np_random=np_random
            )
            if xy is not None:
                z = target_xyz[2] + z_offset
                yaw = np.arctan2(target_xyz[1] - xy[1], target_xyz[0] - xy[0])
                if yaw_noise > 0:
                    yaw += np_random.normal(0, yaw_noise)
                pose = np.eye(4)
                pose[:3, 3] = [xy[0], xy[1], z]
                pose[:3, :3] = R.from_euler("xyz", [0, 0, yaw]).as_matrix()
                return pose
        return None

    def save_debug_image(
        self, path, target_xy, radius_min=0.15, radius_max=0.60, goal_xy=None, agent_radius=0.35
    ):
        h, w = self.occupancy.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # White=free(raw), red=agent-radius expansion, blue/orange=max/min radii, yellow=target, green=goal.
        rad_px = max(1, int(agent_radius * self.px_per_m))
        raw_free = cv2.dilate(self.occupancy.astype(np.uint8), _circular_kernel(rad_px)).astype(
            bool
        )

        img[raw_free] = [255, 255, 255]
        expanded = raw_free & ~self.occupancy
        img[expanded] = [0, 0, 200]

        tc = self._world_to_px(target_xy)
        r_min_px = int(radius_min * self.px_per_m)
        r_max_px = int(radius_max * self.px_per_m)
        cv2.circle(img, (tc[1], tc[0]), r_max_px, (200, 150, 50), 2)
        cv2.circle(img, (tc[1], tc[0]), r_min_px, (50, 150, 200), 2)
        agent_ref_px = int(0.35 * self.px_per_m)
        cv2.circle(img, (w // 2, h // 2), agent_ref_px, (150, 150, 150), 1)
        cv2.circle(img, (tc[1], tc[0]), 6, (0, 255, 255), -1)
        if goal_xy is not None:
            gc = self._world_to_px(goal_xy)
            cv2.circle(img, (gc[1], gc[0]), 6, (0, 255, 0), -1)
        cv2.imwrite(str(path), img)
