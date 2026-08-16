"""Fork of components/robot.py (gold), started byte-identical -- see
env_g1ms_abstraction_port's Stage 0 recipe (env_g1ms.py's own module
context) and scripts/generate_g1ms_rollout.py for the bit-exact
verification harness this fork is refactored under. Being reshaped to
match molmo_spaces/robots/abstract.py's Robot interface (namespace,
robot_view, kinematics, parallel_kinematics, controllers properties) and
molmo_spaces/robots/robot_views/g1_view.py's G1RobotView.
"""

import contextlib
import re

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as _R

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

XML_PATH = str(ASSETS_DIR / "robots/g1/g1_dex.xml")
PREFIX = "robot_0/"
ROOT_BODY = f"{PREFIX}pelvis"
STANDING_HEIGHT = 0.75

# Body-frame +x offset (m) shifting reported "robot xy" forward of the pelvis to better
# match the footprint center. Applied by get_xy()/set_pose() and the controller's _xy().
PELVIS_FORWARD_OFFSET = 0.05

# Dex gripper: positive qpos closes the fingers, negative opens.
GRIPPER_OPEN = -0.0222
GRIPPER_CLOSED = 0.0245

# 30 joints = 12 legs + 3 waist + 14 arms + 1 right gripper (Joint2_1 is <equality>-coupled).
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_Joint1_1",
]

ACTUATOR_NAME_MAP = {
    "right_Joint1_1": "right_grip",
}

# Left arm values are near the gravity-settled hanging pose (the arm is unactuated).
_DEFAULT_QPOS_PATTERNS = {
    r"left_shoulder_pitch_joint": 0.212,
    r"left_shoulder_roll_joint": -0.017,
    r"left_shoulder_yaw_joint": 0.062,
    r"left_elbow_joint": 1.216,
    r"left_wrist_roll_joint": 0.005,
    r"left_wrist_pitch_joint": 0.258,
    r"left_wrist_yaw_joint": 0.006,
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r"right_elbow_joint": -0.2,
    r"right_shoulder_roll_joint": -0.2,
    r"right_shoulder_pitch_joint": 0.2,
    r"right_Joint1_1": GRIPPER_OPEN,
    r"right_Joint2_1": GRIPPER_OPEN,
}


def _resolve_defaults():
    result = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        for pat, val in _DEFAULT_QPOS_PATTERNS.items():
            if re.fullmatch(pat, name):
                result[i] = val
                break
    return result


DEFAULT_QPOS = _resolve_defaults()


def _is_floor_geom(name: str) -> bool:
    name = (name or "").lower()
    return (
        name == "floor" or name.startswith("room|") or name.startswith("room_") or "floor" in name
    )


class G1RobotView:
    """Pose/contact/visibility helpers for the G1 robot, split out of G1Robot
    to match molmo_spaces/robots/robot_views/g1_view.py's G1RobotView --
    the interface molmo_spaces/robots/abstract.py's Robot.robot_view exposes.
    Same underlying code as before, just accessed through
    `robot.robot_view.get_xy()` etc. instead of directly on G1Robot (which
    keeps its own same-named methods as thin pass-throughs to this class, so
    existing call sites are unaffected by this split).
    """

    def __init__(
        self, model, data, body_id, freejoint_id, cam_ids, dof_ids, namespace: str = PREFIX
    ):
        self.model = model
        self.data = data
        self._body_id = body_id
        self._freejoint_id = freejoint_id
        self._cam_ids = cam_ids
        self._dof_ids = dof_ids
        self._namespace = namespace
        self._renderer = None

    def set_pose(self, xy, yaw, z=STANDING_HEIGHT):
        qadr = self.model.jnt_qposadr[self._freejoint_id]
        c, s = np.cos(yaw), np.sin(yaw)
        px = xy[0] - c * PELVIS_FORWARD_OFFSET
        py = xy[1] - s * PELVIS_FORWARD_OFFSET
        self.data.qpos[qadr : qadr + 3] = [px, py, z]
        self.data.qpos[qadr + 3 : qadr + 7] = _R.from_euler("z", yaw).as_quat(scalar_first=True)

    def zero_velocities(self):
        dadr = self.model.jnt_dofadr[self._freejoint_id]
        self.data.qvel[dadr : dadr + 6] = 0.0
        for i in range(len(JOINT_NAMES)):
            self.data.qvel[self._dof_ids[i]] = 0.0

    def get_xy(self):
        return self.data.xpos[self._body_id, :2].copy()

    def get_yaw(self):
        quat = self.data.xquat[self._body_id]
        return _R.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")[2]

    def pelvis_height(self):
        qadr = self.model.jnt_qposadr[self._freejoint_id]
        return float(self.data.qpos[qadr + 2])

    def has_bad_contacts(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]) or ""
            r1, r2 = n1.startswith(self._namespace), n2.startswith(self._namespace)
            if r1 == r2:
                continue
            scene_geom = g2 if r1 else g1
            if _is_floor_geom(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, scene_geom) or ""):
                continue
            return True
        return False

    def check_object_visibility(self, body_id, threshold=0.00002):
        # threshold = minimum fraction of the 224x224 segmentation frame the
        # object must occupy in at least one camera. ~0.00002 (~1px) restores
        # the original presence check: a hard 0.2% (~100px) gate is unsatisfiable
        # for small/distant objects at far nav spawns, so placement fails after
        # its retry budget and crashes rollout workers. Keep this near 1px.
        if not self._cam_ids:
            return True
        if self._renderer is None:
            # procthor scenes can exceed the default max_geom.
            self._renderer = mujoco.Renderer(
                self.model, 224, 224, max_geom=max(20000, self.model.ngeom * 4)
            )
        m = self.model
        geom_ids = set()
        stack = [body_id]
        while stack:
            bid = stack.pop()
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == bid:
                    geom_ids.add(gid)
            for cbid in range(m.nbody):
                if m.body_parentid[cbid] == bid and cbid != bid:
                    stack.append(cbid)

        for cam_id in self._cam_ids:
            try:
                self._renderer.update_scene(self.data, cam_id)
                self._renderer.enable_segmentation_rendering()
                seg = self._renderer.render()
                self._renderer.disable_segmentation_rendering()
            except IndexError:
                return True  # segid overflow — treat as visible to avoid crashing reset.
            seg0 = seg[:, :, 0]
            vis = np.isin(seg0, list(geom_ids)).sum()  # object's visible pixels
            if vis / seg0.size >= threshold:  # fraction of the frame
                return True
        return False

    def close(self):
        """Free the lazily-created visibility renderer (and its GL/EGL context).
        Must be called before dropping the robot on scene reload — otherwise the
        renderer's framebuffer leaks on the render GPU because __del__-based EGL
        teardown is unreliable, so VRAM creeps to OOM across reloads."""
        r = getattr(self, "_renderer", None)
        if r is not None:
            with contextlib.suppress(Exception):
                r.close()
            self._renderer = None


# IK constants/joint groups mirroring agents/policy_g1ms.py's module-level
# HEIGHT_MIN/HEIGHT_MAX, _WAIST, _hand()/_HANDS -- duplicated here (not
# imported) because policy_g1ms.py imports FROM this module's predecessor
# (components/robot.py) and would create a cycle; kept byte-identical to
# those definitions since G1Robot.kinematics below is a verbatim relocation
# of GraspPolicy._solve_ik, which depends on matching them exactly.
HEIGHT_MIN, HEIGHT_MAX = 0.35, 0.793
IK_DT = 1e-2
IK_HEIGHT_DAMPING = 5e5
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _hand_joints(side):
    return tuple(
        f"{side}_{j}"
        for j in (
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
        )
    )


HAND_ARM_JOINTS = {side: _hand_joints(side) for side in ("left", "right")}
HAND_SITE = {side: f"{side}_grasp" for side in ("left", "right")}


class G1Robot:
    def __init__(
        self,
        model,
        data,
        env=None,
        low_level=None,
        namespace: str = PREFIX,
        xml_path: str = XML_PATH,
    ):
        self.model = model
        self.data = data
        self._env = env
        # Namespace/asset path are per-instance rather than the module-level
        # PREFIX/XML_PATH the reference stack hardcoded, matching how every
        # other molmo_spaces Robot takes these from its BaseRobotConfig
        # (robot_namespace / get_robot_xml_path). The module constants remain
        # the defaults, so the reference stack's own call sites are unchanged.
        self._namespace = namespace
        self._xml_path = xml_path
        self._root_body = f"{namespace}pelvis"

        # The low-level WBC/PD controller (components/controller_g1ms.py) is
        # owned here, not by whichever policy attaches via env.set_agent() --
        # matches molmo_spaces/robots/abstract.py's Robot owning its own
        # control stack. `low_level` lets env_g1ms.py's _load_scene carry an
        # existing instance across scene reloads (G1Robot itself is rebuilt
        # every reload; reusing it avoids re-loading the groot_balance/
        # groot_walk ONNX sessions and matches how env.agent was never
        # reconstructed on reload either) -- setup() below rebinds its
        # qpos/qdof/actuator index arrays to this reload's model/data either
        # way, so a fresh instance and a reused one end up equivalent.
        if low_level is None:
            # Deferred import -- controller_g1ms.py imports JOINT_NAMES/
            # DEFAULT_QPOS from this module, so a module-level import here
            # would be circular.
            from molmo_spaces.g1_molmo_port.components.controller_g1ms import (
                G1Controller as _LowLevelController,
            )

            low_level = _LowLevelController()
        self._low_level = low_level
        self._low_level.setup(model, data, prefix=self._namespace)

        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._root_body)
        if self._body_id < 0:
            raise RuntimeError(f"G1 root body '{self._root_body}' not found")
        self._freejoint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}floating_base_joint"
        )

        # kinematics() state -- lazily-built mink.Configuration cache (see
        # kinematics' own docstring) plus the full scene joint-name list
        # _solve_ik reads back solved qpos through.
        self._stj = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
        ]
        self._fj_scene = int(model.jnt_qposadr[self._freejoint_id])
        self._ik_cfg = None
        self._ik_mdl = None
        self._ik_fj_qa = None

        # kinematics_wbc() state -- lazily-built standalone-model mink setup,
        # ported from agents/policy_g1ms.py's G1Controller.setup(). Built on
        # first use rather than here since it's WBC-controller-specific
        # (irrelevant to a G1Robot with no agent attached yet).
        self._wbc_ik_cfg = None
        self._wbc_ik_fj_dof = None
        self._wbc_ik_fj_qa = None
        self._wbc_scene_to_ik_qpos = None
        self._wbc_hand_cfg = None
        self._wbc_waist_qa = None
        self._wbc_posture_task = None
        self._wbc_pelvis_task = None
        self._wbc_limits = None
        self._wbc_self_collision_limit = None

        self._qpos_ids = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{n}")
                ]
                for n in JOINT_NAMES
            ]
        )
        self._dof_ids = np.array(
            [
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{n}")
                ]
                for n in JOINT_NAMES
            ]
        )

        def _find_act(jname):
            act_name = ACTUATOR_NAME_MAP.get(jname, jname)
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}walk_{act_name}"
            )
            if aid < 0:
                aid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}{act_name}"
                )
            return aid

        self.act_ids = np.array([_find_act(n) for n in JOINT_NAMES])

        self.right_gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}right_grip"
        )
        self.left_gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}left_grip"
        )

        self.n_substeps = max(1, round(0.02 / model.opt.timestep))

        # Visibility uses the egocentric head camera so checks match policy POV.
        self._cam_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{self._namespace}{n}")
            for n in ("head_pov",)
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{self._namespace}{n}") >= 0
        ]
        self._robot_view = G1RobotView(
            model,
            data,
            self._body_id,
            self._freejoint_id,
            self._cam_ids,
            self._dof_ids,
            namespace=self._namespace,
        )

        self._apply_solver_overrides()
        self._fix_contacts()

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def namespace(self):
        return self._namespace

    @property
    def controllers(self):
        """{move_group_name: Controller}, matching molmo_spaces/robots/
        abstract.py's Robot.controllers."""
        controllers = getattr(self._low_level, "_controllers", None)
        if not controllers:
            return {}
        return {c.move_group.name: c for c in controllers}

    def set_env(self, env):
        """Wires the low-level controller to `env` (mj model opt overrides,
        n_substeps, floor friction -- see controller_g1ms.G1Controller.
        set_env). Called by env_g1ms.py's _load_scene once `self` is
        assigned to `env.robot` (set_env reads env.robot.n_substeps, so it
        can't run from inside __init__, before that assignment exists)."""
        self._env = env
        self._low_level.set_env(env)

    def execute_action(self, action):
        return self._low_level.execute_action(action)

    def _set_groot_defaults(self):
        return self._low_level._set_groot_defaults()

    def kinematics(
        self, pos, rot=None, hand="right", ik_joints=None, col_limit=None, use_height=True
    ):
        """Exact relocation of agents/policy_g1ms.py's GraspPolicy._solve_ik
        onto G1Robot -- same statements, same order, same every early-break,
        just reading self.model/self.data (the SAME live scene MjModel/
        MjData GraspPolicy.setup() was separately handed -- not a
        standalone robot-only model; gold's own _solve_ik really does mink
        IK against the whole scene) instead of GraspPolicy's own copies, and
        taking as explicit arguments the two things that were caller
        (GraspPolicy)-specific rather than robot-generic:
        `ik_joints` (arm+waist for whichever hand is solving -- caller
        computes this from its own hand/site tables) and `col_limit` (an
        optional mink.CollisionAvoidanceLimit built by
        GraspPolicy._build_collision_limit, which stays caller-side since
        it's grasp-planning-specific, not robot-generic). `use_height`
        mirrors GraspPolicy._use_height (the WBC controller sets this False
        on its own grasp planner; the standalone grasp planner leaves it
        True).

        Deliberately NOT a property (unlike molmo_spaces' Robot.kinematics,
        which returns a solver object) -- this is the exact callable shape
        _solve_ik itself has, ported verbatim rather than reshaped to that
        contract.

        Returns {unprefixed_joint_name: qpos_value} for every joint in the
        WHOLE scene model (not just this robot's) -- callers filter to the
        ones they need, matching gold's own behavior (see _solve_ik's
        original output loop, which never actually filtered by prefix
        either).
        """
        if ik_joints is None:
            ik_joints = set(HAND_ARM_JOINTS[hand]) | set(WAIST_JOINTS)
        site = HAND_SITE[hand]

        if self._ik_cfg is None:
            self._ik_cfg = mink.Configuration(self.model)
            self._ik_mdl = self._ik_cfg.model
            self._ik_fj_qa = self._ik_mdl.jnt_qposadr[
                mujoco.mj_name2id(
                    self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + "floating_base_joint"
                )
            ]

        config = self._ik_cfg
        config.update(self.data.qpos.copy())

        mask = np.zeros(self._ik_mdl.nv)
        for jn in ik_joints:
            jid = mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + jn)
            if jid >= 0:
                mask[self._ik_mdl.jnt_dofadr[jid]] = 1.0
        if use_height:
            fj_dof = self._ik_mdl.jnt_dofadr[
                mujoco.mj_name2id(
                    self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + "floating_base_joint"
                )
            ]
            mask[fj_dof + 2] = 1.0

        ht = mink.FrameTask(
            frame_name=self._namespace + site,
            frame_type="site",
            position_cost=100,
            orientation_cost=1,
            lm_damping=1,
        )
        post = mink.PostureTask(self._ik_mdl, cost=1e-2)
        post.set_target_from_configuration(config)
        r = mink.SO3.from_matrix(rot) if rot is not None else mink.SO3.identity()
        ht.set_target(mink.SE3.from_rotation_and_translation(r, np.asarray(pos, dtype=np.float64)))

        limits = [mink.ConfigurationLimit(self._ik_mdl)]
        if col_limit is not None:
            limits.append(col_limit)
        prev = float("inf")
        for step in range(300):
            try:
                vel = mink.solve_ik(config, [ht, post], IK_DT, "daqp", damping=1e-1, limits=limits)
            except Exception:
                break
            vel *= mask
            config.integrate_inplace(vel, IK_DT)
            q = config.q.copy()
            q[self._ik_fj_qa + 2] = np.clip(q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
            config.update(q)
            err = np.linalg.norm(ht.compute_error(config)[:3])
            if err < 0.001:
                break
            if step == 100 and err > 0.1:
                break
            if step > 100 and err > prev - 1e-5:
                break
            prev = err

        if use_height:
            ik_h = np.clip(config.q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
            self.model.qpos_spring[self._fj_scene + 2] = ik_h
            self.model.dof_damping[self.model.jnt_dofadr[self._freejoint_id] + 2] = (
                IK_HEIGHT_DAMPING
            )

        out = {}
        for jn in self._stj:
            if jn == self._namespace + "floating_base_joint":
                continue
            jid = mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            key = jn[len(self._namespace) :] if jn.startswith(self._namespace) else jn
            out[key] = config.q[self._ik_mdl.jnt_qposadr[jid]]
        return out

    def _build_wbc_self_collision_limit(self, ik_model):
        """Exact relocation of G1Controller._build_self_collision_limit onto
        G1Robot -- self-collision limit for kinematics_wbc: arm<->torso/
        pelvis/waist/hip + arm<->arm geom pairs. Returns None if no
        collidable pairs exist."""
        right_arm, left_arm, body = [], [], []
        for gid in range(ik_model.ngeom):
            if ik_model.geom_contype[gid] == 0 and ik_model.geom_conaffinity[gid] == 0:
                continue
            bid = ik_model.geom_bodyid[gid]
            bname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(s in bname for s in ("shoulder", "elbow", "wrist", "gripper")):
                (right_arm if "right" in bname else left_arm if "left" in bname else []).append(gid)
            elif any(s in bname for s in ("pelvis", "torso", "hip", "waist")):
                body.append(gid)
        arm = right_arm + left_arm
        pairs = []
        if arm and body:
            pairs.append((arm, body))
        if right_arm and left_arm:
            pairs.append((right_arm, left_arm))
        if not pairs:
            return None
        return mink.CollisionAvoidanceLimit(
            model=ik_model,
            geom_pairs=pairs,
            minimum_distance_from_collisions=0.02,
            collision_detection_distance=0.08,
        )

    def _ensure_wbc_ik_setup(self):
        """Lazily build the standalone-model mink setup kinematics_wbc needs,
        exact relocation of G1Controller.setup()'s own WBC-IK construction
        block (agents/policy_g1ms.py). Built once per G1Robot instance (this
        robot is itself reconstructed on every scene (re)load -- see
        env_g1ms.py's _load_scene -- matching the old per-setup()-call
        lifetime the ported code had)."""
        if self._wbc_ik_cfg is not None:
            return
        self._wbc_ik_cfg = mink.Configuration(mujoco.MjModel.from_xml_path(self._xml_path))
        ik_model = self._wbc_ik_cfg.model
        ik_fj = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._wbc_ik_fj_dof = ik_model.jnt_dofadr[ik_fj]
        self._wbc_ik_fj_qa = ik_model.jnt_qposadr[ik_fj]
        self._wbc_scene_to_ik_qpos = []
        for jid in range(ik_model.njnt):
            jname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jname:
                continue
            scene_jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + jname
            )
            if scene_jid < 0:
                continue
            qsz = 7 if ik_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            self._wbc_scene_to_ik_qpos.append(
                (
                    self.model.jnt_qposadr[scene_jid],
                    ik_model.jnt_qposadr[jid],
                    qsz,
                )
            )

        self._wbc_hand_cfg = {}
        for hand in ("right", "left"):
            arm_joints = list(HAND_ARM_JOINTS[hand])
            site = HAND_SITE[hand]
            mask = np.zeros(ik_model.nv)
            for jn in arm_joints + list(WAIST_JOINTS):
                jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    mask[ik_model.jnt_dofadr[jid]] = 1.0
            mask_h = mask.copy()
            mask_h[self._wbc_ik_fj_dof + 2] = 1.0
            task = mink.FrameTask(
                frame_name=site,
                frame_type="site",
                position_cost=100,
                orientation_cost=1,
                lm_damping=1,
            )
            arm_qa = np.array(
                [
                    ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                    for jn in arm_joints
                ],
                dtype=np.int32,
            )
            self._wbc_hand_cfg[hand] = {
                "mask": mask,
                "mask_h": mask_h,
                "task": task,
                "arm_joints": arm_joints,
                "arm_qa": arm_qa,
            }

        posture_cost = np.full(ik_model.nv, 0.1)
        for jn in WAIST_JOINTS:
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                posture_cost[ik_model.jnt_dofadr[jid]] = 0.2
        posture_cost[self._wbc_ik_fj_dof + 2] = 0.1
        self._wbc_waist_qa = np.array(
            [
                ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in WAIST_JOINTS
            ],
            dtype=np.int32,
        )
        self._wbc_posture_task = mink.PostureTask(ik_model, cost=posture_cost)
        self._wbc_pelvis_task = mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=[5.0, 5.0, 0.3],
            orientation_cost=0,
            lm_damping=1,
        )
        for jn, lo, hi in [
            ("waist_yaw_joint", -0.5, 0.5),
            ("waist_roll_joint", -0.4, 0.4),
            ("waist_pitch_joint", -0.1, 0.4),
        ]:
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                ik_model.jnt_range[jid] = [lo, hi]
        self._wbc_limits = [mink.ConfigurationLimit(ik_model)]
        self._wbc_self_collision_limit = self._build_wbc_self_collision_limit(ik_model)

    def kinematics_wbc(
        self, target_pos, target_rot=None, hand="right", avoid_self_collision=False, precision=False
    ):
        """Exact relocation of agents/policy_g1ms.py's G1Controller.
        _solve_ik_wbc onto G1Robot -- same statements, same order, same
        every early-break. Unlike kinematics() above, this solves against a
        standalone robot-only model (mirrors G1Controller.setup()'s own
        `mujoco.MjModel.from_xml_path(self._xml_path)`), synced from the live
        scene qpos through a precomputed joint correspondence table rather
        than operating on the scene model directly. `precision` replaces
        the caller's own `self._grasp_phase in (...)` check (the caller
        computes that condition, since it's WBC-controller-state-specific,
        not robot-generic) -- True selects the tighter max_iters/conv_thresh
        gold used during PHASE_DESCEND/CLOSE/POST_CLOSE/LIFT.

        Returns (arm, waist, ik_h, err) exactly as _solve_ik_wbc did.
        """
        self._ensure_wbc_ik_setup()
        hcfg = self._wbc_hand_cfg[hand]

        ik_q = np.zeros(self._wbc_ik_cfg.model.nq, dtype=np.float64)
        for scene_qa, ik_qa, qsz in self._wbc_scene_to_ik_qpos:
            ik_q[ik_qa : ik_qa + qsz] = self.data.qpos[scene_qa : scene_qa + qsz]
        self._wbc_ik_cfg.update(ik_q)

        q_post = self._wbc_ik_cfg.q.copy()
        q_post[self._wbc_waist_qa] = 0.0
        self._wbc_posture_task.set_target(q_post)
        # Anchor pelvis xy to current (walking unaffected) but z to standing height
        # so the IK actively wants to stand back up when the wrist target allows it.
        pelvis_T = self._wbc_ik_cfg.get_transform_frame_to_world("pelvis", "body")
        pelvis_pos = pelvis_T.translation().copy()
        pelvis_pos[2] = HEIGHT_MAX
        self._wbc_pelvis_task.set_target(
            mink.SE3.from_rotation_and_translation(pelvis_T.rotation(), pelvis_pos)
        )
        rot = mink.SO3.from_matrix(target_rot) if target_rot is not None else mink.SO3.identity()
        hcfg["task"].set_target(
            mink.SE3.from_rotation_and_translation(rot, np.asarray(target_pos, dtype=np.float64))
        )

        max_iters = 60 if precision else 20
        conv_thresh = 0.0015 if precision else 0.005
        prev_err = float("inf")
        err = float("inf")
        limits = self._wbc_limits
        if avoid_self_collision and self._wbc_self_collision_limit is not None:
            limits = self._wbc_limits + [self._wbc_self_collision_limit]
        for step in range(max_iters):
            try:
                vel = mink.solve_ik(
                    self._wbc_ik_cfg,
                    [hcfg["task"], self._wbc_posture_task, self._wbc_pelvis_task],
                    1e-2,
                    "daqp",
                    damping=1e-1,
                    limits=limits,
                )
            except Exception:
                break
            vel *= hcfg["mask_h"]
            self._wbc_ik_cfg.integrate_inplace(vel, 1e-2)
            q_tmp = self._wbc_ik_cfg.q.copy()
            q_tmp[self._wbc_ik_fj_qa + 2] = np.clip(
                q_tmp[self._wbc_ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX
            )
            self._wbc_ik_cfg.update(q_tmp)
            err = float(np.linalg.norm(hcfg["task"].compute_error(self._wbc_ik_cfg)[:3]))
            if err < conv_thresh:
                break
            if step > 10 and err > prev_err - 1e-5:
                break
            prev_err = err
        q = self._wbc_ik_cfg.q
        arm = q[hcfg["arm_qa"]].astype(np.float32)
        waist = q[self._wbc_waist_qa].astype(np.float32)
        ik_h = float(np.clip(q[self._wbc_ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX))
        return arm, waist, ik_h, err

    @property
    def parallel_kinematics(self):
        raise NotImplementedError(
            "G1Robot.parallel_kinematics: not yet ported, see kinematics' docstring."
        )

    def _apply_solver_overrides(self):
        m = self.model
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        m.opt.noslip_iterations = 5
        m.opt.gravity[:] = [0, 0, -9.81]
        m.opt.impratio = 1.0
        m.opt.jacobian = 2  # auto
        m.opt.enableflags = 0
        for gid in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if _is_floor_geom(gname):
                m.geom_friction[gid] = [1.0, 0.005, 0.0001]

    def _fix_contacts(self):
        m = self.model
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(self._namespace) and "ankle_roll" in bname:
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    m.geom_conaffinity[gid] = 15

    def set_defaults(self):
        self.data.qpos[self._qpos_ids] = DEFAULT_QPOS
        valid = self.act_ids >= 0
        self.data.ctrl[self.act_ids[valid]] = DEFAULT_QPOS[valid]
        for i, name in enumerate(JOINT_NAMES):
            for prefix in ("walk_", "grasp_"):
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}{prefix}{name}"
                )
                if aid >= 0:
                    self.data.ctrl[aid] = DEFAULT_QPOS[i]
        if self.right_gripper_aid >= 0:
            self.data.ctrl[self.right_gripper_aid] = GRIPPER_OPEN
        for jname, qval in (
            ("right_Joint1_1", GRIPPER_OPEN),
            ("right_Joint2_1", GRIPPER_OPEN),
        ):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{jname}"
            )
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = qval

    # waist(3) + right arm(7) + right Joint1_1(1).
    _UPPER_RAND_IDX = np.array([12, 13, 14, 22, 23, 24, 25, 26, 27, 28, 29], dtype=np.int64)
    _UPPER_GRIPPER_LOCAL = 10

    def sample_upper_pose(self, np_random, radius=0.15):
        idx = self._UPPER_RAND_IDX
        # Waist gets 0.4x the arm's noise range so the trunk stays closer to neutral.
        per_dim = np.full(len(idx), radius, dtype=np.float64)
        per_dim[:3] = radius * 0.4
        sampled = DEFAULT_QPOS[idx] + np_random.uniform(-per_dim, per_dim)
        sampled[self._UPPER_GRIPPER_LOCAL] = float(np_random.uniform(GRIPPER_OPEN, GRIPPER_CLOSED))
        for i, jidx in enumerate(idx):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{JOINT_NAMES[jidx]}"
            )
            if jid >= 0:
                lo, hi = self.model.jnt_range[jid]
                sampled[i] = float(np.clip(sampled[i], lo, hi))
        return sampled

    def apply_upper_pose(self, values):
        idx = self._UPPER_RAND_IDX
        self.data.qpos[self._qpos_ids[idx]] = values
        aids = self.act_ids[idx]
        valid = aids >= 0
        if valid.any():
            self.data.ctrl[aids[valid]] = values[valid]
        # Mirror Joint2_1 to Joint1_1 — the equality only enforces during sim, not on init.
        j2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}right_Joint2_1"
        )
        if j2 >= 0:
            self.data.qpos[self.model.jnt_qposadr[j2]] = values[self._UPPER_GRIPPER_LOCAL]

    def apply_arm_pose(self, joints_dict):
        """Apply a {joint_name: qpos_value} dict produced by IK to the scene qpos.
        Only writes joints in JOINT_NAMES (skips legs/freejoint). Also mirrors
        right_Joint2_1 to right_Joint1_1."""
        if not joints_dict:
            return
        for i, jname in enumerate(JOINT_NAMES):
            if jname in joints_dict:
                self.data.qpos[self._qpos_ids[i]] = float(joints_dict[jname])
                aid = int(self.act_ids[i])
                if aid >= 0:
                    self.data.ctrl[aid] = float(joints_dict[jname])
        j2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}right_Joint2_1"
        )
        if j2 >= 0 and "right_Joint1_1" in joints_dict:
            self.data.qpos[self.model.jnt_qposadr[j2]] = float(joints_dict["right_Joint1_1"])

    def set_pose(self, xy, yaw, z=STANDING_HEIGHT):
        self._robot_view.set_pose(xy, yaw, z=z)

    def zero_velocities(self):
        self._robot_view.zero_velocities()

    def get_xy(self):
        return self._robot_view.get_xy()

    def get_yaw(self):
        return self._robot_view.get_yaw()

    def pelvis_height(self):
        return self._robot_view.pelvis_height()

    def place(self, xy, yaw):
        self.set_pose(np.asarray(xy, dtype=np.float64), float(yaw))
        self.set_defaults()
        self.zero_velocities()
        mujoco.mj_forward(self.model, self.data)

    def has_bad_contacts(self):
        return self._robot_view.has_bad_contacts()

    def check_object_visibility(self, body_id, threshold=0.00002):
        return self._robot_view.check_object_visibility(body_id, threshold=threshold)

    def close(self):
        """Free the lazily-created visibility renderer (and its GL/EGL context).
        Must be called before dropping the robot on scene reload — otherwise the
        renderer's framebuffer leaks on the render GPU because __del__-based EGL
        teardown is unreliable, so VRAM creeps to OOM across reloads."""
        self._robot_view.close()

    def state_is_finite(self):
        d = self.data
        return np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.ctrl).all()
