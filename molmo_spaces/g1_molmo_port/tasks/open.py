"""Articulated-object open / close task.

Target is a single non-freejoint joint inside an articulated object (drawer in a
dresser, door of a cabinet, lid on a toilet, ...).

- Reward is the absolute distance from the joint's closed value (qpos==0) divided
  by the joint's range, clamped to [0, 1]. No gripper-contact requirement —
  hooking the door open with the wrist back is a valid demonstration.
- `task_type="open"` (joint starts closed, goal = 1.0) and `task_type="close"`
  (joint starts open at `init_open_percent`, goal = 0.0) share the same class.
- The pick task is unaffected: this module is brand new and only loaded when
  the env is constructed with `task_type` in {"open", "close"}.
"""

import re

import ml_collections
import mujoco
import numpy as np

from molmo_spaces.g1_molmo_port.components.constants import joint_grasp_path
from molmo_spaces.g1_molmo_port.components.prompt_sampler import (
    PromptSampler,
    get_object_name,
)
from molmo_spaces.g1_molmo_port.components.prompt_sampler import (
    get_config as get_prompt_config,
)


class OpenTask:
    def __init__(self, config=None, object_regex=".*"):
        config = config or get_config()
        self._object_regex = object_regex or ".*"
        self._randomize_object = bool(config.randomize_object)
        self._task_type = str(config.get("task_type", "open")).lower()  # "open" | "close"
        self._success_threshold = float(config.success_threshold)
        self._terminate_threshold = float(config.get("terminate_threshold", 1.0))
        self._init_open_percent = float(config.init_open_percent)
        self._require_joint_grasp = bool(config.require_joint_grasp)
        # Force prompt templates to match task_type (open / close), overriding
        # the default 'pick' templates so the policy sees "open the drawer" etc.
        prompt_cfg = config.prompts
        if "mode" in prompt_cfg or hasattr(prompt_cfg, "mode"):
            prompt_cfg = type(prompt_cfg)({**dict(prompt_cfg), "mode": self._task_type})
        self._prompt_sampler = PromptSampler(config=prompt_cfg)
        self._prompt = ""

        self.target = None
        self._obj_idx = 0
        self._target_body_set: set = set()
        self._grasp_cache: dict = {}
        self._target_grasps = None
        # Active joint inside the target (only one driven per episode).
        self._target_joint_id = -1
        self._target_joint_xml = ""
        self._target_joint_thor = ""
        self._target_joint_body_id = -1
        self._target_joint_qa = -1
        self._target_joint_range = (0.0, 0.0)
        self._target_joint_init = 0.0
        self.objects = []

    # ---- selection ----

    def set_objects(self, scene):
        pattern = re.compile(self._object_regex, re.IGNORECASE)
        out = []
        for obj in scene.articulated:
            if not pattern.search(obj.name):
                continue
            if self._require_joint_grasp:
                # Object qualifies if AT LEAST one of its joints has a grasp file.
                has_any = any(joint_grasp_path(obj.thor_name, j) for j in obj.joint_thor_names)
                if not has_any:
                    continue
            out.append(obj)
        if not out:
            raise ValueError(
                f"No articulated objects matching '{self._object_regex}'"
                + ("" if not self._require_joint_grasp else " with any joint-grasp file")
            )
        self.objects = out

    def perturb_objects(self, scene, rng):
        # Joints are randomized per episode in init_target_tracking; nothing to do here.
        return

    def select_target(self, rng, randomize=None):
        randomize = randomize if randomize is not None else self._randomize_object
        if randomize or self.target is None:
            self.target = self.objects[self._obj_idx % len(self.objects)]
            self._obj_idx += 1
        # Pick which joint to drive this episode — prefer joints with a grasp
        # file on disk, otherwise fall back to a uniform random pick.
        joint_count = len(self.target.joint_ids)
        candidates = list(range(joint_count))
        if self._require_joint_grasp or randomize:
            with_grasp = [
                k
                for k in candidates
                if joint_grasp_path(self.target.thor_name, self.target.joint_thor_names[k])
            ]
            if with_grasp:
                candidates = with_grasp
        k = int(rng.integers(len(candidates))) if randomize else 0
        self._select_joint(candidates[k])
        # Load grasps for the SELECTED joint (per-joint, not per-object).
        self._target_grasps = self._load_joint_grasps(
            self.target.thor_name, self._target_joint_thor
        )
        return self.target

    def _select_joint(self, k: int):
        self._target_joint_id = int(self.target.joint_ids[k])
        self._target_joint_xml = self.target.joint_xml_names[k]
        self._target_joint_thor = self.target.joint_thor_names[k]
        self._target_joint_body_id = int(self.target.joint_body_ids[k])

    def _load_joint_grasps(self, thor_obj, thor_joint):
        key = (thor_obj, thor_joint)
        if key in self._grasp_cache:
            return self._grasp_cache[key]
        path = joint_grasp_path(thor_obj, thor_joint)
        grasps = None
        if path is not None:
            try:
                grasps = np.load(path)["transforms"].astype(np.float64)
            except Exception:
                grasps = None
        self._grasp_cache[key] = grasps
        return grasps

    # ---- per-episode init ----

    def init_target_tracking(self, scene):
        m, d = scene.model, scene.data
        self._target_body_set = scene.get_body_descendants(self.target.body_id)

        jid = self._target_joint_id
        self._target_joint_qa = int(m.jnt_qposadr[jid])
        lo, hi = float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1])
        # Use the larger-magnitude side if the joint isn't symmetric (rare but
        # safer when closed-pos is 0 and range is e.g. [-1.57, 0]).
        self._target_joint_range = (lo, hi)

        # Seed the joint at closed (open task) or partially open (close task).
        if self._task_type == "close":
            sign = 1.0 if abs(hi) >= abs(lo) else -1.0
            extent = hi if sign > 0 else lo
            init = float(extent) * float(self._init_open_percent)
        else:
            init = 0.0
        d.qpos[self._target_joint_qa] = init
        # Also zero its velocity so it doesn't drift after the env's settle step.
        dadr = int(m.jnt_dofadr[jid])
        d.qvel[dadr] = 0.0
        # Soften the joint so the WBC arm can actually pull/push it — procthor
        # ships drawers with frictionloss=1-2 which is a big dead-zone for the
        # ~few-Newton pull force our IK/WBC stack can sustain.
        m.dof_frictionloss[dadr] = 0.05
        m.dof_damping[dadr] = 0.01
        self._target_joint_init = init
        mujoco.mj_forward(m, d)

    # ---- runtime ----

    def get_obs(self, scene):
        # Observation uses the moving body pose — that's what the policy needs to
        # localize the handle (the dresser root barely moves; the drawer does).
        pos, quat = self.grasp_frame_pose(scene)
        return {
            "target_object_position": pos,
            "target_object_pose": np.concatenate([pos, quat]),
        }

    def grasp_frame_pose(self, scene):
        """World pose of the moving body the active joint actuates. Used by the
        controller as Tw for the joint-grasp transforms (stored in that body's
        local frame)."""
        d = scene.data
        bid = self._target_joint_body_id
        return d.xpos[bid].copy(), d.xquat[bid].copy()

    def articulation_info(self, scene):
        """Joint kinematics needed to drive the open / close motion: axis and
        pivot in world frame, current joint value, and the value to drive to.
        Mirrors upstream's gather_joint_info + open/close target selection."""
        m, d = scene.model, scene.data
        jid = self._target_joint_id
        bid = self._target_joint_body_id
        body_R = d.xmat[bid].reshape(3, 3).copy()
        axis_world = body_R @ np.asarray(m.jnt_axis[jid]).copy()
        n = float(np.linalg.norm(axis_world))
        if n > 1e-9:
            axis_world = axis_world / n
        pivot_world = body_R @ np.asarray(m.jnt_pos[jid]).copy() + d.xpos[bid].copy()
        q_now = float(d.qpos[self._target_joint_qa])
        lo, hi = self._target_joint_range
        target_q = 0.0 if self._task_type == "close" else (hi if abs(hi) >= abs(lo) else lo)
        jtype = int(m.jnt_type[jid])
        kind = "slide" if jtype == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge"
        return {
            "kind": kind,
            "axis_world": axis_world,
            "pivot_world": pivot_world,
            "q": q_now,
            "target_q": float(target_q),
        }

    def preferred_goal_directions(self, scene):
        """For slider joints, return the slider axis projected to xy (both signs)
        as preferred goal directions from the object — so the robot lines up with
        the slide axis to pull. Returns [] for hinge joints (no preference)."""
        m = scene.model
        jid = self._target_joint_id
        if jid < 0 or int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            return []
        bid = self._target_joint_body_id
        body_R = scene.data.xmat[bid].reshape(3, 3)
        axis_world = body_R @ np.asarray(m.jnt_axis[jid])
        axis_xy = axis_world[:2]
        n = float(np.linalg.norm(axis_xy))
        if n < 1e-6:
            return []
        axis_xy = axis_xy / n
        return [axis_xy, -axis_xy]

    def compute_reward(self, scene):
        return float(self._percent_open(scene))

    def _percent_open(self, scene) -> float:
        q = float(scene.data.qpos[self._target_joint_qa])
        lo, hi = self._target_joint_range
        span = abs(hi - lo)
        if span < 1e-6:
            return 0.0
        # Closed pos is 0; opening can be either +ve or -ve (handles [0,1.57]
        # AND [-1.57,0] equally).
        progress = abs(q) / span
        if self._task_type == "close":
            return float(max(0.0, 1.0 - progress / max(self._init_open_percent, 1e-6)))
        return float(min(1.0, progress))

    def is_success(self, scene) -> bool:
        return self._percent_open(scene) >= self._success_threshold

    def is_terminated(self, scene) -> bool:
        return self._percent_open(scene) >= self._terminate_threshold

    # ---- info ----

    def make_info(self, scene, rng):
        obj = self.target
        self._prompt = self._prompt_sampler.sample(obj.asset_id, rng=rng)
        # The grasp frame IS the moving body — grasps are stored in that frame.
        moving_pos, moving_quat = self.grasp_frame_pose(scene)
        # Root pose (dresser body) for goal_xy sampling (front of furniture).
        pos = obj.position(scene.data)
        quat = obj.quat(scene.data)
        info = {
            "target_name": obj.name,
            "target_asset_id": obj.asset_id,
            "target_category": obj.category,
            "target_object_position": moving_pos,
            "target_object_pose": np.concatenate([moving_pos, moving_quat]),
            "target_root_position": pos,
            "target_root_pose": np.concatenate([pos, quat]),
            "prompt": self._prompt,
            "object_name": get_object_name(obj.asset_id, self._prompt_sampler.num_words),
            "task_type": self._task_type,
            "target_joint_id": int(self._target_joint_id),
            "target_joint_name": self._target_joint_xml,
            "target_joint_thor_name": self._target_joint_thor,
            "target_joint_range": self._target_joint_range,
            "target_joint_init": self._target_joint_init,
            "target_joint_body_id": int(self._target_joint_body_id),
            "target_joint_body_pose": np.concatenate([moving_pos, moving_quat]),
        }
        self.attach_grasps(info)
        return info

    def step_info(self):
        return {
            "target_name": self.target.name,
            "target_category": self.target.category,
            "task_type": self._task_type,
            "target_joint_id": int(self._target_joint_id),
            "target_joint_name": self._target_joint_xml,
            "target_joint_body_id": int(self._target_joint_body_id),
            "target_joint_range": self._target_joint_range,
            "target_joint_init": self._target_joint_init,
            "prompt": self._prompt,
        }

    def attach_grasps(self, info):
        info["valid_grasps"] = self._target_grasps


def get_config():
    return ml_collections.ConfigDict(
        dict(
            randomize_object=True,
            task_type="open",  # "open" or "close"
            success_threshold=0.5,  # 50% of joint range — counts as success
            terminate_threshold=1.0,  # 100% of joint range — ends the episode
            init_open_percent=0.9,  # only used for task_type="close" (start mostly open)
            require_joint_grasp=False,  # set True to only accept objects with joint-grasp .npz
            prompts=get_prompt_config(),
        )
    )
