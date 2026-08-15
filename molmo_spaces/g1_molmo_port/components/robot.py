import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as _R

from molmo_spaces.g1_molmo_port import ASSETS_DIR

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


class G1Robot:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        if self._body_id < 0:
            raise RuntimeError(f"G1 root body '{ROOT_BODY}' not found")
        self._freejoint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}floating_base_joint"
        )

        self._qpos_ids = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{n}")
                ]
                for n in JOINT_NAMES
            ]
        )
        self._dof_ids = np.array(
            [
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{n}")
                ]
                for n in JOINT_NAMES
            ]
        )

        def _find_act(jname):
            act_name = ACTUATOR_NAME_MAP.get(jname, jname)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{PREFIX}walk_{act_name}")
            if aid < 0:
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{PREFIX}{act_name}")
            return aid

        self.act_ids = np.array([_find_act(n) for n in JOINT_NAMES])

        self.right_gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{PREFIX}right_grip"
        )
        self.left_gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{PREFIX}left_grip"
        )

        self.n_substeps = max(1, round(0.02 / model.opt.timestep))

        # Visibility uses the egocentric head camera so checks match policy POV.
        self._cam_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{PREFIX}{n}")
            for n in ("head_pov",)
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{PREFIX}{n}") >= 0
        ]
        self._renderer = None

        self._apply_solver_overrides()
        self._fix_contacts()

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
            if bname.startswith(PREFIX) and "ankle_roll" in bname:
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    m.geom_conaffinity[gid] = 15

    def set_defaults(self):
        self.data.qpos[self._qpos_ids] = DEFAULT_QPOS
        valid = self.act_ids >= 0
        self.data.ctrl[self.act_ids[valid]] = DEFAULT_QPOS[valid]
        for i, name in enumerate(JOINT_NAMES):
            for prefix in ("walk_", "grasp_"):
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{PREFIX}{prefix}{name}"
                )
                if aid >= 0:
                    self.data.ctrl[aid] = DEFAULT_QPOS[i]
        if self.right_gripper_aid >= 0:
            self.data.ctrl[self.right_gripper_aid] = GRIPPER_OPEN
        for jname, qval in (
            ("right_Joint1_1", GRIPPER_OPEN),
            ("right_Joint2_1", GRIPPER_OPEN),
        ):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{jname}")
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
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{JOINT_NAMES[jidx]}"
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
        j2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}right_Joint2_1")
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
        j2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}right_Joint2_1")
        if j2 >= 0 and "right_Joint1_1" in joints_dict:
            self.data.qpos[self.model.jnt_qposadr[j2]] = float(joints_dict["right_Joint1_1"])

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

    def place(self, xy, yaw):
        self.set_pose(np.asarray(xy, dtype=np.float64), float(yaw))
        self.set_defaults()
        self.zero_velocities()
        mujoco.mj_forward(self.model, self.data)

    def has_bad_contacts(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]) or ""
            r1, r2 = n1.startswith(PREFIX), n2.startswith(PREFIX)
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
            try:
                r.close()
            except Exception:
                pass
            self._renderer = None

    def state_is_finite(self):
        d = self.data
        return np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.ctrl).all()
