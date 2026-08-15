import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.g1_molmo_port import ASSETS_DIR
from molmo_spaces.g1_molmo_port.components.constants import ROBOT_PREFIX, is_pickup_type
from molmo_spaces.g1_molmo_port.components.object import Object
from molmo_spaces.g1_molmo_port.components.occupancy_map import OccupancyMap


def _strip_skybox(xml_path: Path) -> Path:
    """Cache a skybox-free copy of the scene XML (saves ~18 MB texture memory)."""
    out = xml_path.with_name(xml_path.stem + "_noskybox.xml")
    if out.exists():
        return out
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    removed = False
    for parent in root.iter():
        for tex in list(parent.findall("texture")):
            if tex.attrib.get("type") == "skybox":
                parent.remove(tex)
                removed = True
    if not removed:
        return xml_path
    tree.write(str(out))
    return out


class Scene:
    # Number of solid-color materials pre-allocated for the texture randomizer
    # to pick from as a non-textured alternative.
    _N_SOLID_COLOR_MATERIALS = 8

    def __init__(
        self, xml_path, robot_xml, mobile_regex=None, scene_textures=None, articulated_regex=None
    ):
        xml_path = Path(xml_path)
        if not xml_path.is_absolute():
            xml_path = ASSETS_DIR / xml_path
        self.xml_path = xml_path
        self.robot_xml = Path(robot_xml)
        self._robot_prefix = ROBOT_PREFIX
        spec = mujoco.MjSpec.from_file(str(_strip_skybox(xml_path)))
        robot_spec = mujoco.MjSpec.from_file(str(self.robot_xml))
        frame = spec.worldbody.add_frame()
        frame.attach_body(robot_spec.worldbody.first_body(), self._robot_prefix, "")
        spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)

        self.floor_gid = -1
        # Per-category material IDs: {"Wall": [matids], "Floor": [matids], ...}.
        # Populated below after compile.
        self.scene_matids: dict[str, list[int]] = {}
        # Solid-color placeholder materials. rgba scrambled per reset by the env.
        self.scene_color_matids: list[int] = []
        # `scene_textures` is a dict {category: [tex_paths]}. Old flat-list callers
        # are no longer supported.
        self._scene_texture_paths: dict[str, list[str]] = dict(scene_textures or {})
        for cat, paths in self._scene_texture_paths.items():
            for i, tex_path in enumerate(paths):
                tex_name = f"__scene_tex_{cat}_{i}"
                mat_name = f"__scene_mat_{cat}_{i}"
                spec.add_texture(
                    name=tex_name,
                    type=int(mujoco.mjtTexture.mjTEXTURE_2D),
                    file=str(Path(tex_path).resolve()),
                )
                spec.add_material(
                    name=mat_name,
                    textures=["", tex_name],
                    texrepeat=[2.0, 2.0],
                    texuniform=1,
                )
        if self._scene_texture_paths:
            for i in range(self._N_SOLID_COLOR_MATERIALS):
                spec.add_material(name=f"__scene_color_{i}", rgba=[1.0, 1.0, 1.0, 1.0])

        metadata = {}
        meta_path = self.xml_path.with_name(self.xml_path.stem + "_metadata.json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f).get("objects", {})
        # Shape ObjectManager.object_metadata expects: scene_metadata["objects"][name].
        # Set as current_scene_metadata by G1Env once the owning env exists.
        self.metadata: dict = {"objects": metadata}

        self._optimize(spec, metadata, mobile_regex, articulated_regex)
        self._add_grasp_probe(spec)

        # Weld to lock the robot base — inactive by default (G1 only).
        has_pelvis = any(b.name == f"{self._robot_prefix}pelvis" for b in spec.bodies)
        if has_pelvis:
            weld = spec.add_equality()
            weld.type = mujoco.mjtEq.mjEQ_WELD
            weld.name = "pelvis_weld"
            weld.objtype = mujoco.mjtObj.mjOBJ_BODY
            weld.name1 = f"{self._robot_prefix}pelvis"
            weld.name2 = ""
            weld.active = False
            weld.solref = [0.0002, 1.0]
            weld.solimp = [0.999, 0.9999, 0.0001, 0.5, 2.0]

        self._spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # Anti-tumble damping, rate-capped per DOF (flat 1.0 NaN-explodes gram-scale objects).
        LAMBDA_MAX = 20.0
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                d0 = int(self.model.jnt_dofadr[jid])
                for k in range(6):
                    self.model.dof_damping[d0 + k] = min(
                        1.0, LAMBDA_MAX * float(self.model.dof_M0[d0 + k])
                    )
        # Snapshot model arrays that reset mutates (body_pos for support-height, matid + lights for randomizers).
        self._init_body_pos = self.model.body_pos.copy()
        if self.model.nlight > 0:
            self._init_light_pos = self.model.light_pos.copy()
            self._init_light_dir = self.model.light_dir.copy()
            self._init_light_specular = self.model.light_specular.copy()
            self._init_light_ambient = self.model.light_ambient.copy()
            self._init_light_diffuse = self.model.light_diffuse.copy()
            self._init_light_active = self.model.light_active.copy()
            if hasattr(self.model, "light_castshadow"):
                self._init_light_castshadow = self.model.light_castshadow.copy()
        self._init_body_simple = self.model.body_simple.copy()
        self._init_body_sameframe = self.model.body_sameframe.copy()
        # Per-category material IDs and the geom IDs they should be applied to.
        # Pickables, robot, and probe geoms are excluded. Non-collidable only.
        self.scene_matids = {}
        self.scene_geom_ids: dict[str, list[int]] = {}
        self._init_geom_matid = self.model.geom_matid.copy()
        if self._scene_texture_paths:
            from molmo_spaces.g1_molmo_port.components.constants import classify_scene_geom

            self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            for cat, paths in self._scene_texture_paths.items():
                mids: list[int] = []
                for i in range(len(paths)):
                    mid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"__scene_mat_{cat}_{i}"
                    )
                    if mid >= 0:
                        mids.append(mid)
                if mids:
                    self.scene_matids[cat] = mids
                    self.scene_geom_ids[cat] = []
            for i in range(self._N_SOLID_COLOR_MATERIALS):
                mid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"__scene_color_{i}"
                )
                if mid >= 0:
                    self.scene_color_matids.append(mid)
            for gid in range(self.model.ngeom):
                if (
                    int(self.model.geom_contype[gid]) != 0
                    or int(self.model.geom_conaffinity[gid]) != 0
                ):
                    continue
                bname = (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[gid])
                    )
                    or ""
                ).lower()
                if bname.startswith(self._robot_prefix.lower()) or bname in (
                    "grasp_probe",
                    "gripper_probe",
                ):
                    continue
                gname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
                cat = classify_scene_geom(bname, gname)
                if cat is None or cat not in self.scene_geom_ids:
                    continue
                self.scene_geom_ids[cat].append(gid)
        self.grasp_probe_body_id = self.model.body("grasp_probe").id
        self.grasp_probe_qposadr = self.model.joint("grasp_probe_joint").qposadr[0]
        # Set by G1Env right after construction (needs self.data to exist first,
        # per ObjectManager's own `env.mj_datas[batch_idx]` construction). Object
        # views below are derived lazily through it -- no eager per-body scan.
        self.object_manager = None
        self._object_cache: dict[str, Object] = {}

    def _make_object(self, name: str) -> Object:
        if name not in self._object_cache:
            om = self.object_manager
            has_fj = om.has_free_joint(name)
            jxml_names, jids, jthor_names, jbody_ids = om.get_articulation_joints(name)
            meta = om.object_metadata(name)
            thor_name = (meta.get("name_map") or {}).get("bodies", {}).get(name, "")
            self._object_cache[name] = Object(
                body_id=om.get_object_body_id(name),
                name=name,
                category=om.get_annotation_category(name),
                asset_id=meta.get("asset_id", ""),
                is_static=meta.get("is_static", not has_fj),
                has_freejoint=has_fj,
                thor_name=thor_name,
                joint_xml_names=jxml_names,
                joint_ids=jids,
                joint_thor_names=jthor_names,
                joint_body_ids=jbody_ids,
            )
        return self._object_cache[name]

    @property
    def objects(self):
        # Deliberately not ObjectManager.get_objects_of_type: it drops
        # is_structural() bodies (doorways included -- STRUCTURAL_TYPES has
        # "doorway"), but the open/close task needs doors as legitimate
        # articulated targets. Only robot/grasp-probe bodies get excluded here,
        # same as the old eager scan -- reuse ObjectManager just for the
        # cached top-level-body listing and per-object derivation below. Order
        # matters too: task samplers draw target/spawn candidates by list
        # position against a seeded RNG, so this must stay in body-creation
        # order (top_level_bodies() is already ascending by body id).
        om = self.object_manager
        names = []
        for body_id in om.top_level_bodies():
            name = om.get_object_name(body_id)
            if not name or name.startswith(self._robot_prefix) or name in ("grasp_probe",):
                continue
            names.append(name)
        return [self._make_object(n) for n in names]

    @property
    def pickable(self):
        return [o for o in self.objects if o.has_freejoint]

    @property
    def static(self):
        return [o for o in self.objects if o.is_static]

    @property
    def articulated(self):
        return [o for o in self.objects if o.is_articulated]

    def get(self, name):
        return self._make_object(name)

    def by_category(self, category):
        c = category.lower()
        return [o for o in self.objects if o.category.lower() == c]

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        mujoco.mj_step(self.model, self.data, nstep=n)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[:] = self._init_body_pos
        self.model.body_simple[:] = self._init_body_simple
        self.model.body_sameframe[:] = self._init_body_sameframe
        self.model.geom_matid[:] = self._init_geom_matid
        if self.model.nlight > 0:
            self.model.light_pos[:] = self._init_light_pos
            self.model.light_dir[:] = self._init_light_dir
            self.model.light_specular[:] = self._init_light_specular
            self.model.light_ambient[:] = self._init_light_ambient
            self.model.light_diffuse[:] = self._init_light_diffuse
            self.model.light_active[:] = self._init_light_active
            if hasattr(self.model, "light_castshadow"):
                self.model.light_castshadow[:] = self._init_light_castshadow
        mujoco.mj_forward(self.model, self.data)

    def settle(self, steps=50):
        self.reset()
        self.step(steps)

    def enable_sleep(self):
        self.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)

    def disable_sleep(self):
        self.model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)

    def get_body_descendants(self, body_id):
        body_set = {body_id}
        for i in range(self.model.nbody):
            bid = i
            while bid > 0:
                if bid == body_id:
                    body_set.add(i)
                    break
                bid = self.model.body_parentid[bid]
        return body_set

    def check_held_by_robot(self, target_body_set):
        d = self.data
        m = self.model
        in_contact = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1_body = int(m.geom_bodyid[c.geom1])
            g2_body = int(m.geom_bodyid[c.geom2])
            g1_is_target = g1_body in target_body_set
            g2_is_target = g2_body in target_body_set
            if not g1_is_target and not g2_is_target:
                continue
            other_body = g2_body if g1_is_target else g1_body
            if not m.body(other_body).name.startswith(self._robot_prefix):
                return False
            in_contact = True
        return in_contact

    def occupancy_map(self, agent_radius=0.35):
        return OccupancyMap.from_scene(self, agent_radius)

    def _optimize(self, spec, metadata, mobile_regex, articulated_regex=None):
        """Strip joints + collisions from non-candidate bodies so they become inert static geometry.

        If `articulated_regex` is set, bodies (and their children) matching it
        are left intact — hinge/slide joints preserved so the open task can
        actuate drawers/doors. Default behaviour (None) is unchanged."""
        pattern = re.compile(mobile_regex, re.IGNORECASE) if mobile_regex is not None else None
        art_pattern = (
            re.compile(articulated_regex, re.IGNORECASE) if articulated_regex is not None else None
        )

        def _is_articulated_root(child):
            if art_pattern is None or not child.name:
                return False
            meta = metadata.get(child.name, {})
            cat = meta.get("category", child.name.split("_")[0])
            if not (art_pattern.search(child.name) or art_pattern.search(cat)):
                return False
            # Confirm there are actual non-free joints in the subtree per metadata.
            jmap = (meta.get("name_map") or {}).get("joints") or {}
            for thor_jname in jmap.values():
                if "free" not in thor_jname.lower():
                    return True
            return False

        def _strip(parent, in_articulated=False):
            for child in parent.bodies:
                if child.name and child.name.startswith(self._robot_prefix):
                    _strip(child)
                    continue

                if in_articulated or _is_articulated_root(child):
                    # Don't strip anything inside an articulated subtree.
                    _strip(child, in_articulated=True)
                    continue

                keep = False
                if pattern is not None:
                    meta = metadata.get(child.name, {})
                    cat = meta.get("category", child.name.split("_")[0])
                    name_match = bool(pattern.search(child.name))
                    # Keep if THOR pickup whitelist OR metadata explicitly marks it non-static (objaverse).
                    is_static_meta = meta.get("is_static", None)
                    if is_static_meta is False and name_match:
                        keep = True
                    elif is_pickup_type(cat) and name_match:
                        keep = True

                if not keep:
                    joints = list(child.joints)
                    for jnt in joints:
                        spec.delete(jnt)
                    if joints:
                        for geom in child.geoms:
                            geom.contype = 0
                            geom.conaffinity = 0

                _strip(child)

        _strip(spec.worldbody)

    def _add_grasp_probe(self, spec):
        W, L, H = 0.08, 0.03, 0.01
        BASE = np.array([0.0, 0.0, -0.04])
        hw = W / 2
        probe = spec.worldbody.add_body(name="grasp_probe", pos=[0, 0, 10], gravcomp=1)
        probe.add_freejoint(name="grasp_probe_joint")
        _kw = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=[1, 0, 0, 0.6],
            group=0,
            contype=0,
            conaffinity=0b1111,
        )
        for fromto in [
            ([0, -hw, 0], [0, hw, 0]),
            ([0, -hw, 0], [0, -hw, L]),
            ([0, hw, 0], [0, hw, L]),
        ]:
            g = probe.add_geom(**_kw)
            g.size[0] = H / 2
            g.fromto[:3] = np.array(fromto[0]) + BASE
            g.fromto[3:] = np.array(fromto[1]) + BASE

        gripper_probe_xml = self.robot_xml.parent / "gripper_probe.xml"
        if gripper_probe_xml.exists():
            gprobe_spec = mujoco.MjSpec.from_file(str(gripper_probe_xml))
            frame = spec.worldbody.add_frame()
            frame.attach_body(gprobe_spec.worldbody.first_body(), "", "")
