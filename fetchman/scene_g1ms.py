import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from molmo_spaces.env.arena import scene_spec_ops
from molmo_spaces.env.data_views import SceneObject
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.g1 import PREFIX as ROBOT_PREFIX
from molmo_spaces.utils.scene_maps_aabb import AABBMap


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
        scene_spec_ops.enable_sleep(spec)

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
        scene_spec_ops.add_base_weld(spec, robot_prefix=self._robot_prefix)

        self._spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        # Anti-tumble damping, rate-capped per DOF (flat 1.0 NaN-explodes gram-scale objects).
        scene_spec_ops.cap_freejoint_damping(self.model)
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
            from molmo_spaces.env.arena.randomization.texture import classify_scene_geom

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
                    scene_spec_ops.grasp_probe_body_name(0),
                    "gripper_probe",
                ):
                    continue
                gname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
                cat = classify_scene_geom(bname, gname)
                if cat is None or cat not in self.scene_geom_ids:
                    continue
                self.scene_geom_ids[cat].append(gid)
        self.grasp_probe_body_id = self.model.body(scene_spec_ops.grasp_probe_body_name(0)).id
        self.grasp_probe_qposadr = self.model.joint(
            scene_spec_ops.grasp_probe_joint_name(0)
        ).qposadr[0]
        # Set by G1Env right after construction (needs self.data to exist first,
        # per ObjectManager's own `env.mj_datas[batch_idx]` construction). Object
        # views below are derived lazily through it -- no eager per-body scan.
        self.object_manager = None
        self._object_cache: dict[str, Object] = {}

    def _make_object(self, name: str) -> SceneObject:
        if name not in self._object_cache:
            om = self.object_manager
            has_fj = om.has_free_joint(name)
            jxml_names, jids, jthor_names, jbody_ids = om.get_articulation_joints(name)
            meta = om.object_metadata(name)
            thor_name = (meta.get("name_map") or {}).get("bodies", {}).get(name, "")
            self._object_cache[name] = SceneObject(
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
        # Not ObjectManager.get_objects_of_type: it drops structural bodies,
        # and the open task needs doorways. Body-id order is load-bearing:
        # samplers index candidates by position against a seeded RNG.
        om = self.object_manager
        names = []
        for body_id in om.top_level_bodies():
            name = om.get_object_name(body_id)
            if (
                not name
                or name.startswith(self._robot_prefix)
                or name == scene_spec_ops.grasp_probe_body_name(0)
            ):
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
        return AABBMap.from_scene(self, agent_radius)

    def _optimize(self, spec, metadata, mobile_regex, articulated_regex=None):
        """Strip joints + collisions from non-candidate bodies so they become inert
        static geometry -- see scene_spec_ops.freeze_non_mobile_bodies, which the
        native scene build applies too (task_sampler._apply_scene_spec_ops)."""
        scene_spec_ops.freeze_non_mobile_bodies(
            spec,
            metadata,
            mobile_regex=mobile_regex,
            articulated_regex=articulated_regex,
            robot_prefix=self._robot_prefix,
        )

    def _add_grasp_probe(self, spec):
        """Gold's single grasp probe -- see scene_spec_ops.add_grasp_probes,
        the one producer of these bodies for both stacks."""
        scene_spec_ops.add_grasp_probes(
            spec, count=1, gripper_probe_xml=self.robot_xml.parent / "gripper_probe.xml"
        )
