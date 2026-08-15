import re

import ml_collections
import numpy as np

from molmo_spaces.g1_molmo_port import grasp_source_dir
from molmo_spaces.g1_molmo_port.components.constants import has_valid_grasp
from molmo_spaces.g1_molmo_port.components.prompt_sampler import PromptSampler, get_object_name
from molmo_spaces.g1_molmo_port.components.prompt_sampler import get_config as get_prompt_config


class PickTask:
    def __init__(self, config=None, object_regex=".*"):
        config = config or get_config()
        self._object_regex = object_regex or ".*"
        self._randomize_object = config.randomize_object
        self._object_noise = config.object_noise
        self._prompt_sampler = PromptSampler(config=config.prompts)
        self._prompt = ""

        self.target = None
        self._obj_idx = 0
        self._target_z0 = 0.0
        self._target_body_set = set()
        self._grasp_cache = {}
        self._target_grasps = None
        self.objects = []

    def set_objects(self, scene):
        pattern = re.compile(self._object_regex, re.IGNORECASE)
        # Pickable = freejoint body with a downloaded grasp file (no category whitelist).
        self.objects = [
            o for o in scene.pickable if has_valid_grasp(o.asset_id) and pattern.search(o.name)
        ]
        # objaverse assets listed 10x -> 10x draw probability
        obja = [o for o in self.objects if o.name.lower().startswith("obja")]
        self.objects += obja * 14
        if not self.objects:
            raise ValueError(f"No pickable objects matching '{self._object_regex}'")

    def perturb_objects(self, scene, rng):
        if self._object_noise > 0:
            for obj in scene.pickable:
                if not obj.has_freejoint:
                    continue
                jnt_adr = scene.model.body(obj.body_id).jntadr[0]
                qpos_adr = scene.model.jnt_qposadr[jnt_adr]
                scene.data.qpos[qpos_adr : qpos_adr + 2] += rng.normal(
                    0, self._object_noise, size=2
                )

    def select_target(self, rng, randomize=None):
        if not self.objects:
            raise RuntimeError("select_target called with no pickable objects")
        randomize = randomize if randomize is not None else self._randomize_object
        if randomize:
            self._obj_idx = int(rng.integers(0, 2**31 - 1))
            self.target = self.objects[self._obj_idx % len(self.objects)]
        elif self.target is None:
            self._obj_idx = int(rng.integers(0, 2**31 - 1))
            self.target = self.objects[self._obj_idx % len(self.objects)]
        self._target_grasps = self._load_grasps(getattr(self.target, "asset_id", ""))
        return self.target

    def _load_grasps(self, asset_id):
        if not asset_id:
            return None
        if asset_id in self._grasp_cache:
            return self._grasp_cache[asset_id]
        # New layout (grasps/<source>/<UID>/) first, then legacy flat layout.
        candidates = [
            grasp_source_dir("droid") / asset_id / f"{asset_id}_grasps_filtered.npz",
            grasp_source_dir("droid_objaverse") / asset_id / f"{asset_id}_grasps_filtered.npz",
            grasp_source_dir("") / asset_id / f"{asset_id}_grasps_filtered.npz",
        ]
        grasps = None
        for path in candidates:
            if path.exists():
                try:
                    grasps = np.load(path)["transforms"].astype(np.float64)
                except Exception:
                    grasps = None
                break
        self._grasp_cache[asset_id] = grasps
        return grasps

    def init_target_tracking(self, scene):
        import mujoco

        self._target_z0 = scene.data.xpos[self.target.body_id, 2]
        self._target_body_set = scene.get_body_descendants(self.target.body_id)
        # Articulated targets touch siblings/table during normal grasping — skip the
        # "no external contact" rule for them so benign contacts don't suppress reward.
        m = scene.model
        non_free = False
        for bid in self._target_body_set:
            jadr = int(m.body(bid).jntadr[0])
            jnum = int(m.body(bid).jntnum[0])
            for k in range(jnum):
                if m.jnt_type[jadr + k] != mujoco.mjtJoint.mjJNT_FREE:
                    non_free = True
                    break
            if non_free:
                break
        self._target_has_articulation = non_free

    def get_obs(self, scene):
        pos = self.target.position(scene.data)
        quat = self.target.quat(scene.data)
        return {
            "target_object_position": pos,
            "target_object_pose": np.concatenate([pos, quat]),
        }

    def compute_reward(self, scene):
        lift = scene.data.xpos[self.target.body_id, 2] - self._target_z0
        if lift <= 0:
            return 0.0
        # Require gripper contact — prevents rewarding cases where the object is knocked away.
        if not self._object_in_gripper(scene):
            return 0.0
        return float(lift)

    def _object_in_gripper(self, scene):
        """Require BOTH finger links (Link1 and Link2) to be in contact with the target."""
        import mujoco

        m, d = scene.model, scene.data
        target_bodies = self._target_body_set
        link1, link2 = False, False
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            if b1 not in target_bodies and b2 not in target_bodies:
                continue
            other = b2 if b1 in target_bodies else b1
            other_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if "Link1" in other_name:
                link1 = True
            elif "Link2" in other_name:
                link2 = True
            if link1 and link2:
                return True
        return False

    def make_info(self, scene, rng):
        obj = self.target
        self._prompt = self._prompt_sampler.sample(obj.asset_id, rng=rng)
        pos = obj.position(scene.data)
        quat = obj.quat(scene.data)
        info = {
            "target_name": obj.name,
            "target_asset_id": obj.asset_id,
            "target_category": obj.category,
            "target_object_position": pos,
            "target_object_pose": np.concatenate([pos, quat]),
            "prompt": self._prompt,
            "object_name": get_object_name(obj.asset_id, self._prompt_sampler.num_words),
        }
        self.attach_grasps(info)
        return info

    def step_info(self):
        return {
            "target_name": self.target.name,
            "target_category": self.target.category,
            "prompt": self._prompt,
        }

    def attach_grasps(self, info):
        info["valid_grasps"] = self._target_grasps


def get_config():
    return ml_collections.ConfigDict(
        dict(
            randomize_object=False,
            object_noise=0.0,
            prompts=get_prompt_config(),
        )
    )
