from __future__ import annotations

import heapq
import logging
import sys
from types import SimpleNamespace

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from molmo_spaces.g1_molmo_port.components.controller_g1ms import (
    ACT_ARM,
    ACT_CMD,
    ACT_GRIP,
    ACT_HEIGHT,
    ACT_UPPER,
    ACT_WAIST,
    ACTION_DIM,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import PickPlannerPolicy
from molmo_spaces.robots.g1 import JOINT_NAMES as _JOINTS
from molmo_spaces.robots.g1 import PELVIS_FORWARD_OFFSET as _PELVIS_FWD
from molmo_spaces.utils.grasps import get_pickup_grasps

# molmo_spaces' own FetchmanPickPlannerPolicy (the target shape this file is
# being reshaped towards) logs its G1_MOLMO_TRACE lines via `log.info(...)`
# rather than `print(..., flush=True)`. Matched here with a bare "%(message)s"
# handler on stdout so the printed bytes are unchanged -- gold's own
# agents/policy.py (unported reference) still uses print() with the exact
# same messages, and generate_gold_rollout.py/generate_our_rollout.py's
# side-by-side comparison depends on those bytes staying identical.
log = logging.getLogger(__name__)
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

HEIGHT_MIN, HEIGHT_MAX = 0.35, 0.793

_IDX = {n: i for i, n in enumerate(_JOINTS)}
_LEGS = _JOINTS[:12]
_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


def _hand(side):
    return dict(
        arm=[
            f"{side}_{j}"
            for j in [
                "shoulder_pitch_joint",
                "shoulder_roll_joint",
                "shoulder_yaw_joint",
                "elbow_joint",
                "wrist_roll_joint",
                "wrist_pitch_joint",
                "wrist_yaw_joint",
            ]
        ],
        gripper=f"{side}_Joint1_1",
        site=f"{side}_grasp",
    )


_HANDS = {s: _hand(s) for s in ("left", "right")}


_ASTAR_COARSE_CACHE: dict = {}


def _coarsen_and_dist(occ, downscale):
    """Cached per (occ, downscale) — outputs depend only on these and are constant within a scene."""
    from collections import deque

    key = (occ.tobytes(), downscale, occ.shape)
    cached = _ASTAR_COARSE_CACHE.get(key)
    if cached is not None:
        return cached
    if downscale <= 1:
        coarse = occ.copy()
    else:
        d = downscale
        H, W = occ.shape
        padded = np.pad(occ, ((0, (-H) % d), (0, (-W) % d)))
        coarse = padded.reshape(padded.shape[0] // d, d, padded.shape[1] // d, d).min(1).min(-1)
    h, w = coarse.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    q = deque()
    for r in range(h):
        for c in range(w):
            if not coarse[r, c]:
                dist[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        b = dist[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > b + 1:
                dist[nr, nc] = b + 1
                q.append((nr, nc))
    if len(_ASTAR_COARSE_CACHE) > 32:
        _ASTAR_COARSE_CACHE.clear()
    _ASTAR_COARSE_CACHE[key] = (coarse, dist)
    return coarse, dist


def _astar(occ, start_rc, goal_rc, downscale=4, wall_radius=10, wall_gain=6, wall_exp=2):
    from collections import deque

    coarse, dist = _coarsen_and_dist(occ, downscale)
    h, w, D = coarse.shape[0], coarse.shape[1], downscale

    def nearest(o, rc):
        sr, sc = max(0, min(int(rc[0]), h - 1)), max(0, min(int(rc[1]), w - 1))
        if o[sr, sc]:
            return sr, sc
        q, v = deque([(sr, sc)]), {(sr, sc)}
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in v:
                    v.add((nr, nc))
                    if o[nr, nc]:
                        return nr, nc
                    q.append((nr, nc))
        return None

    sf = nearest(coarse, (int(start_rc[0] // D), int(start_rc[1] // D)))
    gf = nearest(coarse, (int(goal_rc[0] // D), int(goal_rc[1] // D)))
    if sf is None or gf is None:
        return []
    sr, sc = sf
    gr, gc = gf
    if (sr, sc) == (gr, gc):
        return [(gr * D + D // 2, gc * D + D // 2)]

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

    def wp(r, c):
        d = dist[r, c]
        return (
            100.0
            if d <= 0
            else (
                0.0
                if d >= wall_radius
                else wall_gain * (1 - max(d, 1e-3) / wall_radius) ** wall_exp
            )
        )

    open_set = [(((sr - gr) ** 2 + (sc - gc) ** 2) ** 0.5, 0.0, sr, sc)]
    gs, cf, cl = {(sr, sc): 0.0}, {}, set()
    while open_set:
        _, g, r, c = heapq.heappop(open_set)
        if (r, c) in cl:
            continue
        cl.add((r, c))
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in cf:
                r, c = cf[(r, c)]
                path.append((r, c))
            path.reverse()
            return [(p * D + D // 2, q * D + D // 2) for p, q in path]
        for (dr, dc), cost in zip(dirs, costs):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and coarse[nr, nc] and (nr, nc) not in cl:
                ng = g + cost + wp(nr, nc)
                if ng < gs.get((nr, nc), float("inf")):
                    gs[(nr, nc)] = ng
                    cf[(nr, nc)] = (r, c)
                    heapq.heappush(
                        open_set, (ng + ((nr - gr) ** 2 + (nc - gc) ** 2) ** 0.5, ng, nr, nc)
                    )
    return []


def _simplify_path(path, occ, clearance=6):
    if len(path) <= 2:
        return path

    def free(a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for t in np.linspace(0, 1, n):
            r, c = int(round((1 - t) * a[0] + t * b[0])), int(round((1 - t) * a[1] + t * b[1]))
            for dr in range(-clearance, clearance + 1):
                for dc in range(-clearance, clearance + 1):
                    if not occ[
                        max(0, min(r + dr, occ.shape[0] - 1)), max(0, min(c + dc, occ.shape[1] - 1))
                    ]:
                        return False
        return True

    s = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not free(path[i], path[j]):
            j -= 1
        s.append(path[j])
        i = j
    return s


PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_REALIGN = "realign"
PHASE_DESCEND = "descend"
PHASE_OPEN_HOLD = "open_hold"
PHASE_CLOSE = "close"
PHASE_POST_CLOSE = "post_close"
PHASE_LIFT = "lift"
PHASE_DONE = "done"
_PHASE_ORDER = [
    PHASE_APPROACH,
    PHASE_DESCEND,
    PHASE_OPEN_HOLD,
    PHASE_CLOSE,
    PHASE_POST_CLOSE,
    PHASE_LIFT,
    PHASE_DONE,
]


class GraspPolicy:
    PREGRASP_OFFSET = (0.05, 0.125)
    LIFT = 0.15
    GRIPPER_OPEN = -0.0222
    GRIPPER_CLOSED = 0.0245
    STEPS = dict(approach=30, descend=30, open_hold=10, close=120, post_close=100, lift=30)
    IK_DT = 1e-2
    IK_SMOOTH = 0.3
    IK_THRESH = 0.06
    HEIGHT_DAMPING = 5e5

    def __init__(self):
        self._phase = PHASE_IDLE
        self._hand = "right"
        self._ik_error_fn = None

    def setup(self, model, data, prefix):
        self._model, self._data, self._prefix = model, data, prefix
        self._qposadr = {}
        for jn in _JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + jn)
            if jid >= 0:
                self._qposadr[jn] = model.jnt_qposadr[jid]
        self._freejoint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint"
        )
        self._fj_scene = model.jnt_qposadr[self._freejoint_id]
        self._sites = {
            h: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + c["site"])
            for h, c in _HANDS.items()
        }
        self._col_limit = None

    @property
    def done(self):
        return self._phase == PHASE_DONE

    def _build_collision_limit(self, target_pos, radius=1.0):
        m, d = self._model, self._data
        mujoco.mj_fwdPosition(m, d)
        robot_bids = {
            i
            for i in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(self._prefix)
        }
        right_arm_gids, left_arm_gids, body_gids, obs_gids = [], [], [], []
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bid in robot_bids:
                if any(s in bname for s in ("shoulder", "elbow", "wrist", "gripper")):
                    if "right" in bname:
                        right_arm_gids.append(gid)
                    elif "left" in bname:
                        left_arm_gids.append(gid)
                elif any(s in bname for s in ("pelvis", "torso", "hip", "waist")):
                    body_gids.append(gid)
                continue
            gtype = m.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if gname == "floor":
                continue
            if np.linalg.norm(d.geom_xpos[gid, :2] - target_pos[:2]) > radius:
                continue
            obs_gids.append(gid)
        arm_gids = right_arm_gids + left_arm_gids
        pairs = []
        if arm_gids and obs_gids:
            pairs.append((arm_gids, obs_gids))
        if arm_gids and body_gids:
            pairs.append((arm_gids, body_gids))
        if right_arm_gids and left_arm_gids:
            pairs.append((right_arm_gids, left_arm_gids))
        self._col_limit = (
            mink.CollisionAvoidanceLimit(
                model=m,
                geom_pairs=pairs,
                minimum_distance_from_collisions=0.03,
                collision_detection_distance=0.1,
            )
            if pairs
            else None
        )

    def _sync(self):
        return self._data.qpos.copy()

    def _solve_ik(self, pos, rot=None, hand=None):
        """Delegates to G1Robot.kinematics (components/robot_g1ms.py) -- a
        verbatim relocation of this method's own former body onto the robot
        object GraspPolicy.setup() was handed the same live scene model/data
        as (self._model/self._data ARE self._env.robot.model/data, not
        copies). ik_joints/col_limit/use_height are the caller-specific bits
        kinematics() takes as explicit arguments rather than owning itself.
        """
        hand = hand or self._hand
        hcfg = _HANDS[hand]
        ik_joints = set(hcfg["arm"] + _WAIST)
        env = getattr(self, "_env", None)
        return env.robot.solve_scene_ik(
            pos,
            rot=rot,
            hand=hand,
            ik_joints=ik_joints,
            col_limit=self._col_limit,
            use_height=getattr(self, "_use_height", True),
        )

    def _path_is_clear(self, dc, grasp_T, hand):
        grasp_pos = grasp_T[:3, 3]
        env = getattr(self, "_env", None)
        rng = env.np_random if env is not None else np.random
        offset = rng.uniform(0.10, 0.15)
        pregrasp_pos = grasp_pos - offset * grasp_T[:3, 2]
        if env is None:
            return True
        m, sd = env.scene.model, env.scene.data
        # CACHE: probe_bids + robot_bids are static for the lifetime of the scene
        # model. Recomputing them every call was the O(N²) hot spot during precheck.
        if getattr(self, "_cached_bids_for_model_id", None) != id(m):
            probe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_probe")
            if probe_bid < 0:
                self._cached_probe_bids = set()
            else:
                pb = set()
                for bid in range(m.nbody):
                    b = bid
                    while b > 0:
                        if b == probe_bid:
                            pb.add(bid)
                            break
                        b = m.body_parentid[b]
                self._cached_probe_bids = pb
            self._cached_robot_bids = {
                i
                for i in range(m.nbody)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(
                    self._prefix
                )
            }
            self._cached_bids_for_model_id = id(m)
            self._cached_target_body_id = None  # force target recompute on next target
        probe_bids = self._cached_probe_bids
        if not probe_bids:
            return True
        robot_bids = self._cached_robot_bids
        # CACHE: target_bids — invalidate when _target_body_id changes.
        tbid = getattr(self, "_target_body_id", -1)
        if getattr(self, "_cached_target_body_id", None) != tbid:
            tb = set()
            if tbid >= 0:
                stack = [tbid]
                while stack:
                    b = stack.pop()
                    tb.add(b)
                    for cb in range(m.nbody):
                        if m.body_parentid[cb] == b and cb != b:
                            stack.append(cb)
            self._cached_target_bids = tb
            self._cached_target_body_id = tbid
        target_bids = self._cached_target_bids
        ignore = robot_bids | target_bids | probe_bids
        qa = m.joint("gripper_probe_joint").qposadr[0]
        ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_probe_joint_a")
        jb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_probe_joint_b")
        if ja >= 0:
            sd.qpos[m.jnt_qposadr[ja]] = 0.04
        if jb >= 0:
            sd.qpos[m.jnt_qposadr[jb]] = -0.04
        grasp_quat = R.from_matrix(grasp_T[:3, :3]).as_quat(scalar_first=True)
        # Precheck only needs a smell-test (grasp pose isn't in a wall). At runtime
        # (committed to executing this grasp), sweep the full pregrasp→mid→grasp
        # path to catch "approach blocked" cases.
        if getattr(self, "_fast_precheck", False):
            checkpoints = [grasp_pos]
        else:
            checkpoints = [
                pregrasp_pos,
                0.5 * pregrasp_pos + 0.5 * grasp_pos,
                grasp_pos,
            ]
        for pt in checkpoints:
            sd.qpos[qa : qa + 3] = pt
            sd.qpos[qa + 3 : qa + 7] = grasp_quat
            mujoco.mj_forward(m, sd)
            mujoco.mj_collision(m, sd)
            for k in range(sd.ncon):
                c = sd.contact[k]
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if b1 in probe_bids or b2 in probe_bids:
                    other = b2 if b1 in probe_bids else b1
                    if other not in ignore:
                        sd.qpos[qa : qa + 3] = [0, 0, 10]
                        mujoco.mj_forward(m, sd)
                        return False
        sd.qpos[qa : qa + 3] = [0, 0, 10]
        mujoco.mj_forward(m, sd)
        return True

    def _ik_error(self, T, hand):
        if self._ik_error_fn is not None:
            return self._ik_error_fn(T, hand)
        sid = self._sites[hand]
        ik = self._solve_ik(T[:3, 3], T[:3, :3], hand)
        dc = mujoco.MjData(self._model)
        dc.qpos[:] = self._data.qpos
        for jn, v in ik.items():
            if jn in self._qposadr:
                dc.qpos[self._qposadr[jn]] = v
        # _ik_cfg/_ik_fj_qa now live on G1Robot (populated by the _solve_ik
        # call just above, via kinematics()) rather than on this instance.
        robot = getattr(self, "_env", None) and self._env.robot
        if robot is not None and robot._ik_cfg is not None:
            dc.qpos[self._fj_scene + 2] = np.clip(
                robot._ik_cfg.q[robot._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX
            )
        mujoco.mj_forward(self._model, dc)
        return float(np.linalg.norm(dc.site_xpos[sid] - T[:3, 3]))

    def plan(self, info):
        self._phase = PHASE_IDLE
        self._step = 0
        self._hand = "right"
        self._frozen_arm = self._frozen_legs = None
        self._prev_ik = {}
        tp = info.get("target_object_pose")
        vg = info.get("valid_grasps")
        log.info(
            f"[G1_MOLMO_TRACE] plan() entry: tp_is_none={tp is None} "
            f"vg_len={0 if vg is None else len(vg)}"
        )
        if tp is None:
            self._phase = PHASE_DONE
            return
        # _path_is_clear needs this to exclude the target from collision rejection.
        env = getattr(self, "_env", None)
        self._target_body_id = (
            env.task.target.body_id if env is not None and env.task.target is not None else -1
        )

        tp = np.asarray(tp, dtype=np.float64)
        Tw = np.eye(4)
        Tw[:3, :3] = R.from_quat(tp[3:], scalar_first=True).as_matrix()
        Tw[:3, 3] = tp[:3]
        dc = mujoco.MjData(self._model)
        dc.qpos[:] = self._data.qpos
        dc.qvel[:] = self._data.qvel
        dc.ctrl[:] = self._data.ctrl
        mujoco.mj_forward(self._model, dc)

        grasp_T = None
        if vg is not None and len(vg) > 0:
            vg = np.asarray(vg, dtype=np.float64)
            flip = R.from_euler("z", np.pi)
            cands = []
            for hand in ("right",):
                sid = self._sites[hand]
                hr = R.from_matrix(dc.site_xmat[sid].reshape(3, 3))
                for go in vg:
                    c = Tw @ go
                    ro = R.from_matrix(c[:3, :3])
                    rf = ro * flip
                    # Pick the yaw=0 or yaw=pi flip closer to current hand — keeps IK well-conditioned.
                    if (hr.inv() * rf).magnitude() < (hr.inv() * ro).magnitude():
                        c = c.copy()
                        c[:3, :3] = rf.as_matrix()
                    cands.append((c, hand))
            (env.np_random if env is not None else np.random).shuffle(cands)
            # Geometric reach pre-filter: skip candidates whose target is beyond
            # arm reach from the shoulder. Kills 50–80% of cands BEFORE any IK.
            shoulder_world = {}
            for hand in ("right",):
                sb = mujoco.mj_name2id(
                    self._model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{self._prefix}{hand}_shoulder_pitch_link",
                )
                if sb >= 0:
                    shoulder_world[hand] = dc.xpos[sb].copy()
            MAX_REACH = 0.85  # G1 arm fully extended, generous
            MIN_REACH = 0.10
            if shoulder_world:
                cands = [
                    (c, h)
                    for (c, h) in cands
                    if h not in shoulder_world
                    or MIN_REACH <= float(np.linalg.norm(c[:3, 3] - shoulder_world[h])) <= MAX_REACH
                ]
            cached = getattr(self, "_cached", None)
            if cached is not None and cached["hand"] == "right":
                if "grasp_local" in cached:
                    grasp_T_cached = Tw @ cached["grasp_local"]
                else:
                    grasp_T_cached = np.eye(4)
                    grasp_T_cached[:3, 3] = cached["grasp_pos"]
                    grasp_T_cached[:3, :3] = cached["grasp_rot"]
                cands.insert(0, (grasp_T_cached, cached["hand"]))
            self._cached = None
            ik_results = []
            # Cap candidates aggressively. cands is shuffled so 30 is representative;
            # any cached grasp from a prior plan() call is always at index 0.
            try:
                for c, h in cands[:30]:
                    e = self._ik_error(c, h)
                    ik_results.append((e, c, h))
            except KeyboardInterrupt:
                self._phase = PHASE_DONE
                return
            # Sort by IK error and path-check only the top-K. The lowest-error
            # candidate is almost always the right pick — the rest are insurance.
            # Avoids running the expensive _path_is_clear for every passable IK.
            if ik_results:
                ik_results.sort(key=lambda x: x[0])
                # K=1 during precheck (just smell-test the best candidate, cheap),
                # K=5 at runtime once we're committed to actually grasping.
                PATH_CHECK_K = 1 if getattr(self, "_fast_precheck", False) else 5
                log.info(
                    f"[G1_MOLMO_TRACE] plan(): {len(cands)} candidates after reach "
                    f"pre-filter, top errors={[round(e, 4) for e, _, _ in ik_results[:5]]}"
                )
                for e, c, h in ik_results[:PATH_CHECK_K]:
                    if e >= 0.1:
                        break  # even the best is too far — give up
                    if self._path_is_clear(dc, c, h):
                        grasp_T = c
                        self._hand = h
                        break
            else:
                log.info("[G1_MOLMO_TRACE] plan(): 0 candidates survived reach pre-filter")
            if grasp_T is None:
                log.info(
                    "[G1_MOLMO_TRACE] plan(): no grasp_T found (all candidates too far "
                    "or path blocked) -> PHASE_DONE"
                )
                self._phase = PHASE_DONE
                return
        else:
            self._hand = "right"
            sid = self._sites["right"]
            cr = dc.site_xmat[sid].reshape(3, 3)
            fd = cr[:, 0].copy()
            fd[2] = 0
            if np.linalg.norm(fd) < 1e-6:
                fd = np.array([1.0, 0.0, 0.0])
            fd /= np.linalg.norm(fd)
            up = np.array([0.0, 0.0, 1.0])
            rt = np.cross(up, fd)
            rt /= np.linalg.norm(rt) + 1e-8
            fd = np.cross(rt, up)
            grasp_T = np.eye(4)
            grasp_T[:3, :3] = np.column_stack([fd, rt, up])
            grasp_T[:3, 3] = tp[:3]

        off = (env.np_random if env is not None else np.random).uniform(*self.PREGRASP_OFFSET)
        pre = grasp_T.copy()
        pre[:3, 3] -= off * grasp_T[:3, 2]
        self._pregrasp = pre[:3, 3].copy()
        self._grasp_pos = grasp_T[:3, 3].copy()
        self._grasp_rot = grasp_T[:3, :3].copy()
        self._lift_pos = grasp_T[:3, 3].copy()
        self._lift_pos[2] += self.LIFT
        self._grasp_local = np.linalg.inv(Tw) @ grasp_T
        log.info(
            f"[G1_MOLMO_TRACE] plan(): hand={self._hand} grasp_pos={self._grasp_pos.tolist()} "
            f"grasp_rot_euler_xyz={R.from_matrix(self._grasp_rot).as_euler('xyz').tolist()} "
            f"pregrasp_pos={self._pregrasp.tolist()} pregrasp_offset={off:.4f}"
        )
        # In precheck/fast mode, skip the expensive pregrasp IK (slow full-scene
        # mink solve up to 300 iters) and skip _build_collision_limit (O(N_geom)
        # scan over thousands of procthor geoms). Both are only needed for
        # runtime grasp execution — the cache is consumed by _start_grasp's
        # fast-path which is only active when _start_at_pregrasp is True.
        if getattr(self, "_fast_precheck", False):
            self._pregrasp_joints = None
            self._phase = PHASE_APPROACH
            self._step = 0
            return
        if getattr(self, "_skip_pregrasp_ik", False):
            # Consumer (WBC controller) re-solves IK per step; this result is unused.
            self._pregrasp_joints = None
        else:
            try:
                self._pregrasp_joints = self._solve_ik(self._pregrasp, self._grasp_rot, self._hand)
            except Exception:
                self._pregrasp_joints = None
        sid = self._sites[self._hand]
        self._start_pos = dc.site_xpos[sid].copy()
        self._start_rot = dc.site_xmat[sid].reshape(3, 3).copy()
        from scipy.spatial.transform import Slerp

        self._rot_slerp = Slerp(
            [0, 1], R.concatenate([R.from_matrix(self._start_rot), R.from_matrix(self._grasp_rot)])
        )

        self._build_collision_limit(self._grasp_pos)

        self._phase = PHASE_APPROACH
        self._step = 0

    def refresh_grasp_for_current_object(self):
        env = getattr(self, "_env", None)
        if env is None or getattr(self, "_grasp_local", None) is None:
            return False
        tgt = env.target
        if tgt is None:
            return False
        task = getattr(env, "task", None)
        if task is not None and hasattr(task, "grasp_frame_pose"):
            pos, quat = task.grasp_frame_pose(env.scene)
        else:
            pos = tgt.position(env.scene.data)
            quat = tgt.quat(env.scene.data)
        Tw = np.eye(4)
        Tw[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
        Tw[:3, 3] = pos
        grasp_T = Tw @ self._grasp_local
        self._grasp_pos = grasp_T[:3, 3].copy()
        self._grasp_rot = grasp_T[:3, :3].copy()
        self._lift_pos = self._grasp_pos.copy()
        self._lift_pos[2] += self.LIFT
        return True

    def _steps(self):
        return self.STEPS.get(self._phase, 1)

    def _target(self):
        t = min(self._step / max(1, self._steps()), 1.0)
        p = self._phase
        if p == PHASE_APPROACH:
            return (
                self._start_pos * (1 - t) + self._pregrasp * t,
                self._rot_slerp(t).as_matrix(),
                0.0,
            )
        if p == PHASE_DESCEND:
            return self._pregrasp * (1 - t) + self._grasp_pos * t, self._grasp_rot, 0.0
        if p == PHASE_OPEN_HOLD:
            return self._grasp_pos, self._grasp_rot, 0.0
        if p == PHASE_CLOSE:
            return self._grasp_pos, self._grasp_rot, 1.0
        if p == PHASE_POST_CLOSE:
            return self._grasp_pos, self._grasp_rot, 1.0
        if p == PHASE_LIFT:
            return self._grasp_pos * (1 - t) + self._lift_pos * t, self._grasp_rot, 1.0
        return None, None, 0.0

    def _advance(self):
        try:
            idx = _PHASE_ORDER.index(self._phase)
            if self._phase == PHASE_OPEN_HOLD:
                self._frozen_arm = {
                    jn: self._data.qpos[qa]
                    for jn, qa in self._qposadr.items()
                    if "Joint1_1" not in jn and "Joint2_1" not in jn
                }
            self._phase = _PHASE_ORDER[idx + 1]
        except (ValueError, IndexError):
            self._phase = PHASE_DONE
        self._step = 0

    def __call__(self):
        action = np.full(len(_JOINTS), np.nan)
        if self._phase in (PHASE_IDLE, PHASE_DONE):
            return action

        if self._frozen_legs is None:
            self._frozen_legs = {
                jn: self._data.qpos[self._qposadr[jn]] for jn in _LEGS if jn in self._qposadr
            }
        for jn, v in self._frozen_legs.items():
            if jn in _IDX:
                action[_IDX[jn]] = v

        pos, rot, grip = self._target()
        hcfg = _HANDS[self._hand]
        active = set(hcfg["arm"] + _WAIST)

        if self._phase in (PHASE_CLOSE, PHASE_POST_CLOSE) and self._frozen_arm:
            for jn, v in self._frozen_arm.items():
                if jn in _IDX and jn in active:
                    action[_IDX[jn]] = v
        elif pos is not None:
            if self._prev_ik is None:
                self._prev_ik = {}
            ik = self._solve_ik(pos, rot)
            for jn, v in ik.items():
                if jn in _IDX and jn in active:
                    if jn in self._prev_ik:
                        v = self._prev_ik[jn] * (1 - self.IK_SMOOTH) + v * self.IK_SMOOTH
                    self._prev_ik[jn] = v
                    action[_IDX[jn]] = v

        gj = hcfg["gripper"]
        if gj in _IDX:
            action[_IDX[gj]] = self.GRIPPER_OPEN * (1 - grip) + self.GRIPPER_CLOSED * grip

        self._step += 1
        if self._step >= self._steps():
            self._advance()
        return action


class G1Controller:
    WAYPOINT_REACH = 0.10
    FINAL_REACH = 0.05
    SPEED = 0.3
    TURN_KP = 2.0
    MAX_TURN = 1.0
    FACE_TURN = 1.2
    FACE_TOL = 0.1
    FACE_WP_TOL = 0.25
    GRIPPER_OPEN = -0.0222
    GRIPPER_CLOSED = 0.0245

    NEAR_GOAL_DIST = 0.05
    REALIGN_MAX = 0.30
    REALIGN_REACH = 0.020
    REALIGN_YAW_REACH = 0.04
    REALIGN_SPEED_X = 0.25
    REALIGN_SPEED_X_MIN = 0.07
    REALIGN_RAMP_STEPS = 12
    REALIGN_RAMP_DOWN_DIST = 0.08
    REALIGN_INITIAL_HOLD = 20
    REALIGN_SPEED_Y = 0.045
    REALIGN_YAW_SPEED = 0.40
    REALIGN_DRIVE_STEPS = 150
    REALIGN_SETTLE = 80

    def __init__(
        self, walk_timeout_s=20.0, walk_fail_dist=0.2, face_yaw_offset=0.0, grasp_retry_closer=False
    ):
        self._walk_timeout_s = float(walk_timeout_s)
        self._walk_fail_dist = float(walk_fail_dist)
        self._face_yaw_offset_max = float(face_yaw_offset)
        self._grasp_retry_closer = bool(grasp_retry_closer)
        self._env = None
        self._speed = self.SPEED
        self._min_speed = 0.3
        self._grasp_speed_scale = 1.25
        self._nav_noise_scale = 0.0
        self._upper_cmd = None

    @classmethod
    def create(cls, config):
        return cls(
            walk_timeout_s=config.get("walk_timeout_s", 20.0),
            walk_fail_dist=config.get("walk_fail_dist", 0.2),
            face_yaw_offset=config.get("face_yaw_offset", 0.0),
            grasp_retry_closer=config.get("grasp_retry_closer", False),
        )

    @property
    def _low_level(self):
        """The shared WBC PD/ONNX controller (controller_g1ms.G1Controller)
        now lives on env.robot, not on this policy -- G1Robot owns its own
        control stack the same way molmo_spaces' Robot does, rather than the
        attached policy holding it (see robot_g1ms.G1Robot.__init__). This
        policy still reads/writes `_cmd`/`_height_cmd`/`_data`/etc. through
        `self._low_level` exactly as before; only where that object lives
        changed, not how this class uses it.
        """
        return self._env.robot._low_level

    def set_env(self, env):
        self._env = env
        self._grasp_planner._env = env

    def execute_action(self, action):
        """Delegates to env.robot -- tasks/pick_task_sampler_g1ms.py's
        step() now dispatches through env.robot.execute_action directly
        (env.robot always owns a low-level controller, agent or not), but
        this is kept for any caller that still holds the policy and expects
        it to behave like the old inheritance-based G1Controller did.
        """
        return self._env.robot.execute_action(action)

    def _set_groot_defaults(self):
        """Delegates to env.robot -- tasks/pick_task_sampler_g1ms.py pokes
        `env.agent._set_groot_defaults()` directly from several spawn-
        candidate/precheck call sites (all guarded by
        `hasattr(env.agent, '_set_groot_defaults')`, so losing this method
        would silently no-op those resets rather than raising -- this name
        must stay present on the policy itself).
        """
        return self._env.robot._set_groot_defaults()

    def setup(self, model, data, prefix="robot_0/"):
        # env.robot already owns and set up the low-level controller (see
        # robot_g1ms.G1Robot.__init__) -- nothing to delegate here anymore.
        self._sites = {
            h: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{c['site']}")
            for h, c in _HANDS.items()
        }
        r_grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}right_Joint1_1")
        self._right_grip_qa = model.jnt_qposadr[r_grip_jid] if r_grip_jid >= 0 else None
        # _solve_ik_wbc's own mink/standalone-model setup now lives on
        # G1Robot (components/robot_g1ms.py's kinematics_wbc, lazily built
        # there). Only arm_joints survives here -- read by __call__ outside
        # _solve_ik_wbc (PHASE_REALIGN's pregrasp-joint lookup) and cheap
        # enough that duplicating it beats depending on G1Robot's lazy
        # WBC-IK setup having already run by the time that unrelated read
        # happens.
        self._ik_hand_cfg = {
            hand: {"arm_joints": list(_HANDS[hand]["arm"])} for hand in ("right", "left")
        }
        self._active_hand = "right"

        self._waypoints = []
        self._wp_idx = 0
        self._arrived = self._facing = False
        self._target_xy = self._object_xy = None
        self._has_path = False

        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._grasp_pos = None
        self._grasp_rot = None
        self._hand = "right"
        self._grasp_retries_used = 0

        self._grasp_planner = GraspPolicy()
        self._grasp_planner.setup(model, data, prefix)
        self._grasp_planner._use_height = False
        self._grasp_planner._ik_error_fn = self._wbc_ik_error
        # WBC re-solves IK per step and never reads pregrasp_joints; skip the slow
        # full-scene pregrasp IK.
        self._grasp_planner._skip_pregrasp_ik = True

    @property
    def has_path(self):
        return self._has_path

    def in_precision_phase(self):
        return self._grasp_phase in (
            PHASE_REALIGN,
            PHASE_DESCEND,
            PHASE_OPEN_HOLD,
            PHASE_CLOSE,
            PHASE_POST_CLOSE,
            PHASE_LIFT,
        )

    @property
    def done(self):
        return self._grasp_phase == PHASE_DONE

    def precheck_grasp(self, info):
        env = self._env
        if env is None:
            return True
        goal_xy = info.get("goal_xy")
        tgt = info.get("target_object_position")
        if goal_xy is None or tgt is None:
            return False
        goal_yaw = info.get("goal_yaw")
        if goal_yaw is None:
            goal_yaw = float(np.arctan2(tgt[1] - goal_xy[1], tgt[0] - goal_xy[0]))
        saved_qpos = env.scene.data.qpos.copy()
        saved_ctrl = env.scene.data.ctrl.copy()
        try:
            env.robot.set_pose(goal_xy, goal_yaw)
            self._low_level._set_groot_defaults()
            mujoco.mj_forward(env.scene.model, env.scene.data)
            self._grasp_planner._fast_precheck = True
            try:
                self._grasp_planner.plan(info)
            finally:
                self._grasp_planner._fast_precheck = False
            found = self._grasp_planner._phase != PHASE_DONE
            if found:
                p = self._grasp_planner
                self._grasp_planner._cached = dict(
                    grasp_local=p._grasp_local.copy(),
                    grasp_pos=p._grasp_pos.copy(),
                    grasp_rot=p._grasp_rot.copy(),
                    pregrasp=p._pregrasp.copy(),
                    pregrasp_joints=dict(p._pregrasp_joints)
                    if getattr(p, "_pregrasp_joints", None)
                    else None,
                    hand=p._hand,
                )
            return found
        finally:
            env.scene.data.qpos[:] = saved_qpos
            env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(env.scene.model, env.scene.data)

    def _xy(self):
        px, py = self._low_level._data.xpos[self._low_level._pelvis, :2]
        yaw = self._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([px + c * _PELVIS_FWD, py + s * _PELVIS_FWD])

    def _yaw(self):
        m = self._low_level._data.xmat[self._low_level._pelvis].reshape(3, 3)
        return np.arctan2(m[1, 0], m[0, 0])

    def _plan_path(self, info):
        occ = info.get("nav_occupancy_map") or info.get("occupancy_map")
        tgt = info.get("target_object_position")
        if occ is None or tgt is None:
            return
        self._object_xy = np.asarray(tgt[:2], dtype=np.float64)
        goal_xy = info.get("goal_xy")
        self._target_xy = np.asarray(
            goal_xy[:2] if goal_xy is not None else tgt[:2], dtype=np.float64
        )
        path = _astar(
            occ.occupancy, occ._world_to_px(self._xy()), occ._world_to_px(self._target_xy)
        )
        if path:
            path = _simplify_path(path, occ.occupancy)
            self._waypoints = [(occ.map_to_world @ np.array([r, c, 1.0]))[:2] for r, c in path]
            if goal_xy is not None and self._waypoints:
                self._waypoints[-1] = np.asarray(goal_xy[:2], dtype=np.float64)
            self._has_path = True

    def _update_nav_command(self):
        if self._arrived:
            self._low_level._cmd[:] = 0
            return
        if not self._waypoints:
            self._low_level._cmd[:] = 0
            return
        xy, yaw = self._xy(), self._yaw()
        if self._facing:
            face = self._object_xy if self._object_xy is not None else self._target_xy
            if face is not None:
                desired = np.arctan2(face[1] - xy[1], face[0] - xy[0]) + getattr(
                    self, "_face_yaw_offset", 0.0
                )
                ye = (desired - yaw + np.pi) % (2 * np.pi) - np.pi
                if abs(ye) > self.FACE_TOL:
                    self._low_level._cmd[:] = [
                        0,
                        0,
                        np.clip(self.TURN_KP * ye, -self.FACE_TURN, self.FACE_TURN),
                    ]
                    return
            self._arrived = True
            self._low_level._cmd[:] = 0
            return
        wp = self._waypoints[self._wp_idx]
        if (
            np.linalg.norm(xy - wp) < self.WAYPOINT_REACH
            and self._wp_idx < len(self._waypoints) - 1
        ):
            self._wp_idx += 1
        wp = self._waypoints[self._wp_idx]
        delta = wp - xy
        dist = np.linalg.norm(delta)
        final = self._wp_idx >= len(self._waypoints) - 1
        # Smoothstep brake — zero derivative at both ends so we "roll" into a stop
        # instead of stepping. Longer runway (0.70 m) gives ~2.3s of decel at
        # 0.3 m/s — feels natural in real-life walking.
        BRAKE_DIST = 0.70
        STOP_PAD = 0.04  # extra margin to absorb walking inertia
        stop_dist = self.FINAL_REACH + STOP_PAD
        # Arrive at stop_dist (where the brake zeros speed), else robot hangs in the
        # dead zone short of FINAL_REACH.
        if final and dist <= stop_dist:
            self._facing = True
            self._low_level._cmd[:] = 0
            return
        ye = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(ye) > self.FACE_WP_TOL:
            self._low_level._cmd[:] = [
                0,
                0,
                np.clip(self.TURN_KP * ye, -self.MAX_TURN, self.MAX_TURN),
            ]
            return
        if final:
            if dist <= stop_dist:
                spd = 0.0
            elif dist >= BRAKE_DIST:
                spd = self._speed
            else:
                t = (dist - stop_dist) / (BRAKE_DIST - stop_dist)
                spd = self._speed * (3 * t * t - 2 * t * t * t)
        else:
            spd = np.clip(dist, self._min_speed, self._speed)
        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]
        ln = max(np.sqrt(lx**2 + ly**2), 1e-6)
        ang = np.sign(self.TURN_KP * ye) * np.clip(abs(self.TURN_KP * ye), 0.05, self.MAX_TURN)
        self._low_level._cmd[:] = [spd * lx / ln, np.clip(spd * ly / ln, -0.5, 0.5), ang]

    def perturb_action_for_rollout(self, action):
        out = np.asarray(action, dtype=np.float32).copy()
        if self._nav_noise_scale <= 0 or self._arrived or self._grasp_phase != PHASE_IDLE:
            return out
        if not self._waypoints or np.linalg.norm(out[0:3]) < 1e-6:
            return out
        rng = self._env.np_random if getattr(self, "_env", None) is not None else np.random
        out[0:2] += rng.normal(0.0, 0.02 * self._nav_noise_scale, size=2)
        out[2] += rng.normal(0.0, 0.05 * self._nav_noise_scale)
        out[0] = np.clip(out[0], -self._speed, self._speed)
        out[1] = np.clip(out[1], -0.5, 0.5)
        out[2] = np.clip(out[2], -self.MAX_TURN, self.MAX_TURN)
        return out

    def _wbc_ik_error(self, T, hand):
        saved_hand = self._active_hand
        self._active_hand = hand
        _, _, _, err = self._solve_ik_wbc(T[:3, 3], T[:3, :3])
        self._active_hand = saved_hand
        return err

    def _solve_ik_wbc(self, target_pos, target_rot=None, avoid_self_collision=False):
        """Delegates to G1Robot.kinematics_wbc (components/robot_g1ms.py) --
        a verbatim relocation of this method's own former body, onto the
        robot object this controller was set up against. `precision` (the
        tighter max_iters/conv_thresh during PHASE_DESCEND/CLOSE/
        POST_CLOSE/LIFT) stays computed here since it reads controller
        state (self._grasp_phase), not robot-generic state.
        """
        precision = self._grasp_phase in (PHASE_DESCEND, PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT)
        return self._env.robot.kinematics_wbc(
            target_pos,
            target_rot=target_rot,
            hand=self._active_hand,
            avoid_self_collision=avoid_self_collision,
            precision=precision,
        )

    def _start_grasp(self, info):
        self._grasp_planner._env = self._env
        cached = getattr(self._grasp_planner, "_cached", None)
        if self._start_at_pregrasp and cached is not None and "pregrasp" in cached:
            self._grasp_pos = cached["grasp_pos"].copy()
            self._grasp_rot = cached["grasp_rot"].copy()
            self._pregrasp = cached["pregrasp"].copy()
            self._grasp_planner._grasp_local = cached["grasp_local"].copy()
            self._grasp_planner._grasp_pos = cached["grasp_pos"].copy()
            self._grasp_planner._grasp_rot = cached["grasp_rot"].copy()
            self._grasp_planner._pregrasp = cached["pregrasp"].copy()
            self._grasp_planner._hand = cached["hand"]
            self._grasp_planner._phase = PHASE_APPROACH
            self._pregrasp_rot = None
            self._lift_pos = self._grasp_pos.copy()
            self._lift_pos[2] += 0.15
            self._hand = cached["hand"]
            self._active_hand = self._hand
            self._grasp_phase = PHASE_REALIGN
            self._grasp_step = 0
            self._setup_axis_realign()
            return
        self._refresh_info_target_pose(info)
        self._grasp_planner.plan(info)
        if self._grasp_planner._phase == PHASE_DONE:
            # Optional retry: inch closer to the object once and re-walk before giving up.
            if (
                self._grasp_retry_closer
                and self._grasp_retries_used < 1
                and self._install_closer_waypoint(info)
            ):
                self._grasp_retries_used += 1
                self._grasp_planner._phase = PHASE_IDLE
                return
            self._grasp_phase = PHASE_DONE
            return
        self._grasp_pos = self._grasp_planner._grasp_pos.copy()
        self._grasp_rot = self._grasp_planner._grasp_rot.copy()
        self._pregrasp = self._grasp_planner._pregrasp.copy()
        self._pregrasp_rot = None
        offset = info.get("pregrasp_offset")
        if offset is not None:
            xyz_off, rot_off = offset
            self._pregrasp = self._pregrasp + xyz_off
            self._pregrasp_rot = rot_off @ self._grasp_rot
        self._lift_pos = self._grasp_pos.copy()
        self._lift_pos[2] += 0.15
        self._hand = self._grasp_planner._hand
        self._active_hand = self._hand
        self._grasp_phase = PHASE_APPROACH
        self._grasp_step = 0

    def _install_closer_waypoint(self, info):
        """Pick a new waypoint between current xy and object, at standoff uniform in
        [grasp_spawn_radius_min, current_standoff). Resets nav so the controller walks
        there and re-attempts the grasp on arrival. Returns False if no progress is
        possible (already at/under the min standoff)."""
        env = getattr(self, "_env", None)
        if env is None:
            return False
        obj_xy = info.get("target_object_position")
        if obj_xy is None:
            return False
        obj_xy = np.asarray(obj_xy, dtype=np.float64)[:2]
        cur = self._xy().astype(np.float64)
        delta = obj_xy - cur
        cur_standoff = float(np.linalg.norm(delta))
        r_min = float(getattr(env, "_grasp_spawn_radius_min", 0.25))
        if cur_standoff <= r_min + 1e-3:
            return False
        # Sample in the middle 50% of [r_min, cur_standoff] — avoid the bottom 25%
        # (riskiest, closest to object) and the top 25% (no meaningful progress).
        span = cur_standoff - r_min
        lo, hi = r_min + 0.25 * span, r_min + 0.75 * span
        new_standoff = float(env.np_random.uniform(lo, hi))
        direction = delta / cur_standoff
        new_wp = obj_xy - new_standoff * direction
        self._waypoints = [np.asarray(new_wp, dtype=np.float64)]
        self._wp_idx = 0
        self._arrived = False
        self._facing = False
        self._has_path = True
        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._ik_cache = None
        self._upper_cmd = None
        self._cmd_smoothed = None
        return True

    def _setup_axis_realign(self):
        self._realign_target_xy = (
            self._goal_xy_target.copy() if self._goal_xy_target is not None else self._xy()
        )
        self._realign_start_xy = self._xy().copy()
        self._realign_arrived_at = None
        effective_speed = self.REALIGN_SPEED_X * 0.5
        nominal_steps = abs(self._realign_offset) / effective_speed / 0.02
        self._realign_drive_steps = int(np.clip(nominal_steps + 40, 100, 500))

    def _refresh_info_target_pose(self, info):
        """Sync info's target_object_* with live sim pose; object may have shifted during nav."""
        if self._env is None or info is None:
            return
        tgt = self._env.target
        if tgt is None:
            return
        # For tasks like Open where grasps are in a sub-body frame (drawer face,
        # door panel), the task supplies the right Tw for the grasp planner.
        task = getattr(self._env, "task", None)
        try:
            if task is not None and hasattr(task, "grasp_frame_pose"):
                pos, quat = task.grasp_frame_pose(self._env.scene)
            else:
                pos = tgt.position(self._env.scene.data)
                quat = tgt.quat(self._env.scene.data)
        except Exception:
            return
        info["target_object_position"] = np.asarray(pos, dtype=np.float64)
        info["target_object_pose"] = np.concatenate([pos, quat]).astype(np.float64)

    def _grasp_target(self):
        t = min(self._grasp_step / max(1, self._grasp_steps()), 1.0)
        p = self._grasp_phase
        if p == PHASE_APPROACH:
            sid = self._sites[self._hand]
            if self._grasp_step == 0:
                self._approach_start = self._low_level._data.site_xpos[sid].copy()
                target_rot = (
                    self._pregrasp_rot if self._pregrasp_rot is not None else self._grasp_rot
                )
                self._approach_slerp = Slerp(
                    [0, 1],
                    R.concatenate(
                        [
                            R.from_matrix(self._low_level._data.site_xmat[sid].reshape(3, 3)),
                            R.from_matrix(target_rot),
                        ]
                    ),
                )
            pos = self._approach_start * (1 - t) + self._pregrasp * t
            rot = self._approach_slerp(t).as_matrix()
            return pos, rot
        if p == PHASE_DESCEND:
            sid = self._sites[self._hand]
            if self._grasp_step == 0:
                self._descend_start = self._low_level._data.site_xpos[sid].copy()
                start_rot = self._low_level._data.site_xmat[sid].reshape(3, 3).copy()
                self._descend_slerp = Slerp(
                    [0, 1],
                    R.concatenate([R.from_matrix(start_rot), R.from_matrix(self._grasp_rot)]),
                )
            rot = self._descend_slerp(t).as_matrix()
            return self._descend_start * (1 - t) + self._grasp_pos * t, rot
        if p == PHASE_REALIGN:
            return (
                self._pregrasp,
                self._pregrasp_rot if self._pregrasp_rot is not None else self._grasp_rot,
            )
        if p in (PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE):
            return self._grasp_pos, self._grasp_rot
        if p == PHASE_LIFT:
            return self._grasp_pos * (1 - t) + self._lift_pos * t, self._grasp_rot
        return None, None

    def _grasp_steps(self):
        steps = {
            "approach": 700,
            "realign": 500,
            "descend": 600,
            "open_hold": 160,
            "close": 800,
            "post_close": 400,
            "lift": 600,
        }.get(self._grasp_phase, 1)
        if self._grasp_phase in (PHASE_APPROACH, PHASE_DESCEND, PHASE_LIFT):
            steps = int(round(steps / self._grasp_speed_scale))
        return max(1, steps)

    def _grasp_min_steps(self):
        if self._grasp_phase == PHASE_APPROACH and getattr(self, "_quick_approach", False):
            return 5
        return {
            "approach": 40,
            "realign": 20,
            "descend": 40,
            "open_hold": 10,
            "close": 240,
            "post_close": 80,
            "lift": 40,
        }.get(self._grasp_phase, 1)

    def _grasp_phase_goal(self):
        p = self._grasp_phase
        if p in (PHASE_APPROACH, PHASE_REALIGN):
            return self._pregrasp, self._grasp_rot
        if p in (PHASE_DESCEND, PHASE_OPEN_HOLD, PHASE_CLOSE, PHASE_POST_CLOSE):
            return self._grasp_pos, self._grasp_rot
        if p == PHASE_LIFT:
            return self._lift_pos, self._grasp_rot
        return None, None

    def _grasp_phase_done(self):
        if self._grasp_step < self._grasp_min_steps():
            return False
        if self._grasp_phase == PHASE_CLOSE:
            if self._right_grip_qa is None:
                return False
            return self._low_level._data.qpos[self._right_grip_qa] > self.GRIPPER_CLOSED - 0.004
        if self._grasp_phase in (PHASE_OPEN_HOLD, PHASE_POST_CLOSE):
            return True
        if self._grasp_phase == PHASE_REALIGN:
            if self._start_at_pregrasp:
                if self._realign_arrived_at is not None:
                    return (self._grasp_step - self._realign_arrived_at) >= self.REALIGN_SETTLE
                return False
            close = float(np.linalg.norm(self._xy() - self._realign_target_xy)) < self.REALIGN_REACH
            if not close:
                self._realign_arrived_at = None
                return False
            if self._realign_arrived_at is None:
                self._realign_arrived_at = self._grasp_step
            return (self._grasp_step - self._realign_arrived_at) >= self.REALIGN_SETTLE
        goal_pos, goal_rot = self._grasp_phase_goal()
        if goal_pos is None:
            return False
        sid = self._sites[self._hand]
        pos_err = float(np.linalg.norm(self._low_level._data.site_xpos[sid] - goal_pos))
        # angle(R1^T @ R2) = acos((trace(R1^T @ R2) - 1) / 2) — no scipy.Rotation overhead.
        R1 = self._low_level._data.site_xmat[sid].reshape(3, 3)
        rel_trace = float(np.dot(R1.ravel(), goal_rot.ravel()))
        cos = max(-1.0, min(1.0, (rel_trace - 1.0) * 0.5))
        rot_err = float(np.arccos(cos))
        return pos_err < 0.035 and rot_err < 0.45

    def _log_trace(self, new_phase):
        arm = getattr(self, "_last_arm_joints", None)
        waist = getattr(self, "_last_waist_joints", None)
        t = getattr(self._env, "time", None) if getattr(self, "_env", None) is not None else None
        log.info(
            f"[G1_MOLMO_TRACE] phase->{new_phase} t={t} xy={self._xy().tolist()} "
            f"yaw={self._yaw():.4f} height_cmd={self._low_level._height_cmd:.4f} "
            f"arm={None if arm is None else np.round(arm, 4).tolist()} "
            f"waist={None if waist is None else np.round(waist, 4).tolist()}"
        )

    def _advance_grasp(self):
        prev = self._grasp_phase
        if prev == PHASE_REALIGN:
            self._grasp_phase = PHASE_DESCEND
            self._grasp_step = 0
            self._ik_cache = None
            self._upper_cmd = None
            self._realign_done = True
            self._realign_arrived_at = None
            self._quick_approach = True
            if self._grasp_planner.refresh_grasp_for_current_object():
                self._grasp_pos = self._grasp_planner._grasp_pos.copy()
                self._grasp_rot = self._grasp_planner._grasp_rot.copy()
                self._lift_pos = self._grasp_planner._lift_pos.copy()
            self._log_trace(self._grasp_phase)
            return
        try:
            idx = _PHASE_ORDER.index(self._grasp_phase)
            self._grasp_phase = _PHASE_ORDER[idx + 1]
        except (ValueError, IndexError):
            self._grasp_phase = PHASE_DONE
        self._grasp_step = 0
        self._ik_cache = None
        self._upper_cmd = None
        self._quick_approach = False
        if prev == PHASE_APPROACH and self._grasp_phase == PHASE_DESCEND:
            if self._grasp_planner.refresh_grasp_for_current_object():
                self._grasp_pos = self._grasp_planner._grasp_pos.copy()
                self._grasp_rot = self._grasp_planner._grasp_rot.copy()
                self._lift_pos = self._grasp_planner._lift_pos.copy()
        self._log_trace(self._grasp_phase)

    def _start_realign(self) -> bool:
        sid = self._sites[self._hand]
        wrist_xy = self._low_level._data.site_xpos[sid][:2].copy()
        err = self._pregrasp[:2] - wrist_xy
        err_norm = float(np.linalg.norm(err))
        if err_norm > self.REALIGN_MAX:
            return False
        self._realign_target_xy = self._xy() + err
        self._realign_start_xy = self._xy().copy()
        self._grasp_phase = PHASE_REALIGN
        self._grasp_step = 0
        self._ik_cache = None
        self._upper_cmd = None
        return True

    def _realign_cmd(self):
        if self._start_at_pregrasp and self._realign_axis == "x":
            if self._grasp_step < self.REALIGN_INITIAL_HOLD:
                return np.zeros(3, dtype=np.float32)
            delta_world = self._realign_target_xy - self._xy()
            yaw = self._yaw()
            lx = np.cos(yaw) * delta_world[0] + np.sin(yaw) * delta_world[1]
            direction = -float(np.sign(self._realign_offset))
            signed_progress_remaining = direction * lx
            if signed_progress_remaining <= self.REALIGN_REACH:
                self._realign_arrived_at = self._realign_arrived_at or self._grasp_step
                return np.zeros(3, dtype=np.float32)
            spd_up = self.REALIGN_SPEED_X_MIN + (
                self.REALIGN_SPEED_X - self.REALIGN_SPEED_X_MIN
            ) * min(1.0, self._grasp_step / max(1.0, self.REALIGN_RAMP_STEPS))
            decel_t = min(1.0, signed_progress_remaining / max(1e-6, self.REALIGN_RAMP_DOWN_DIST))
            spd_down = (
                self.REALIGN_SPEED_X_MIN
                + (self.REALIGN_SPEED_X - self.REALIGN_SPEED_X_MIN) * decel_t
            )
            speed = min(spd_up, spd_down)
            return np.array([direction * speed, 0.0, 0.0], dtype=np.float32)
        if self._start_at_pregrasp and self._realign_axis == "yaw":
            yaw_err = (self._goal_yaw_target - self._yaw() + np.pi) % (2 * np.pi) - np.pi
            direction = -float(np.sign(self._realign_offset))
            signed_progress_remaining = direction * (-yaw_err)
            if abs(yaw_err) <= self.REALIGN_YAW_REACH or signed_progress_remaining <= 0:
                self._realign_arrived_at = self._realign_arrived_at or self._grasp_step
                return np.zeros(3, dtype=np.float32)
            return np.array([0.0, 0.0, direction * self.REALIGN_YAW_SPEED], dtype=np.float32)
        if self._start_at_pregrasp and self._realign_axis == "y":
            vy = -float(np.sign(self._realign_offset)) * self.REALIGN_SPEED_Y
            return np.array([0.0, vy, 0.0], dtype=np.float32)
        delta_world = self._realign_target_xy - self._xy()
        yaw = self._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        lx = c * delta_world[0] + s * delta_world[1]
        ly = -s * delta_world[0] + c * delta_world[1]
        dist = float(np.linalg.norm(delta_world))
        if dist < self.REALIGN_REACH:
            return np.zeros(3, dtype=np.float32)
        ln = max(np.sqrt(lx * lx + ly * ly), 1e-6)
        speed = min(self.REALIGN_SPEED_X, dist * 2.0)
        return np.array([speed * lx / ln, speed * ly / ln, 0.0], dtype=np.float32)

    def reset(self, info):
        self._arrived = self._facing = False
        self._waypoints = []
        self._wp_idx = 0
        self._has_path = False
        self._grasp_phase = PHASE_IDLE
        self._grasp_step = 0
        self._ik_cache = None
        self._grasp_retries_used = 0
        self._upper_cmd = None
        self._realign_done = False
        self._realign_target_xy = None
        self._realign_arrived_at = None
        self._quick_approach = False
        self._last_arm_joints = None
        self._last_waist_joints = None
        self._cmd_smoothed = None
        self._stall_min_dist = float("inf")
        self._stall_last_progress_t = 0.0
        self._init_arm_at_pregrasp = bool(info and info.get("init_arm_at_pregrasp"))
        self._start_at_pregrasp = bool(info and info.get("start_at_pregrasp"))
        self._realign_axis = info.get("realign_axis") if info else None
        self._realign_offset = float(info.get("realign_offset", 0.0)) if info else 0.0
        self._pregrasp_joints = (
            dict(info["pregrasp_joints"]) if info and info.get("pregrasp_joints") else None
        )
        self._goal_xy_target = (
            np.asarray(info["goal_xy"], dtype=np.float64).copy()
            if info and info.get("goal_xy") is not None
            else None
        )
        self._goal_yaw_target = (
            float(info["goal_yaw"]) if info and info.get("goal_yaw") is not None else None
        )
        self._occ = info.get("occupancy_map") if info else None
        rng = self._env.np_random if getattr(self, "_env", None) is not None else np.random
        self._speed = float(rng.uniform(0.3, 0.5))
        self._min_speed = 0.05
        r = (
            float(info.get("face_yaw_offset_max", self._face_yaw_offset_max))
            if info
            else self._face_yaw_offset_max
        )
        self._face_yaw_offset = float(rng.uniform(-r, r)) if r > 0 else 0.0
        self._grasp_speed_scale = float(rng.uniform(0.85, 1.15))
        self._nav_noise_scale = float(rng.choice([0.0, 1.0], p=[0.7, 0.3]))
        self._low_level._reset_wbc_state()
        self._info = info

        goal_xy = info.get("goal_xy")
        if self._start_at_pregrasp:
            self._arrived = True
            self._has_path = True
            self._start_grasp(info)
        elif (
            goal_xy is not None
            and np.linalg.norm(self._xy() - np.asarray(goal_xy)) < self.NEAR_GOAL_DIST
        ):
            self._arrived = True
            self._has_path = True
            self._start_grasp(info)
        else:
            self._plan_path(info)

        self._low_level._write_default_pose()

        init_height = info.get("init_height") if info else None
        if init_height is not None:
            self._low_level._height_cmd = float(init_height)

        if self._start_at_pregrasp and self._pregrasp_joints is not None and self._env is not None:
            self._env.robot.apply_arm_pose(self._pregrasp_joints)
            self._low_level._target_q[15:] = self._low_level._data.qpos[
                self._low_level._jqpos[15:]
            ].astype(np.float32)
            mujoco.mj_forward(self._low_level._model, self._low_level._data)
            self._init_upper = None
        else:
            init_upper = info.get("init_upper_pose")
            if init_upper is not None and self._env is not None:
                self._env.robot.apply_upper_pose(init_upper)
                self._low_level._target_q[15:] = self._low_level._data.qpos[
                    self._low_level._jqpos[15:]
                ].astype(np.float32)
                mujoco.mj_forward(self._low_level._model, self._low_level._data)
            self._init_upper = (
                np.asarray(init_upper, dtype=np.float32) if init_upper is not None else None
            )

    def set_step_info(self, info):
        self._info = info

    def sample_actions(self, obs):
        self._low_level._step_counter += 1
        # Stall recovery (only for init_arm_at_pregrasp episodes — the awkward
        # pregrasp arm pose can prevent the WBC from walking): if no distance
        # progress for 3s, fast-forward to "arrived" so _start_grasp fires.
        if (
            self._init_arm_at_pregrasp
            and not self._arrived
            and self._grasp_phase == PHASE_IDLE
            and self._waypoints
            and self._env is not None
        ):
            final_wp = self._waypoints[-1]
            cur_dist = float(np.linalg.norm(self._xy() - final_wp))
            min_dist = getattr(self, "_stall_min_dist", float("inf"))
            if cur_dist < min_dist - 0.02:  # 2cm improvement counts as progress
                self._stall_min_dist = cur_dist
                self._stall_last_progress_t = self._env.time
            elif self._env.time - getattr(self, "_stall_last_progress_t", self._env.time) > 0.5:
                self._arrived = True
                self._facing = True
                self._stall_min_dist = float("inf")
                self._stall_last_progress_t = self._env.time
        if (
            not self._arrived
            and self._grasp_phase == PHASE_IDLE
            and self._env is not None
            and self._env.time > self._walk_timeout_s
            and self._target_xy is not None
            and float(np.linalg.norm(self._xy() - self._target_xy)) > self._walk_fail_dist
        ):
            self._grasp_phase = PHASE_DONE
        self._update_nav_command()
        if self._grasp_phase == PHASE_REALIGN:
            self._low_level._cmd[:] = self._realign_cmd()
        else:
            target = self._low_level._cmd.copy()
            prev = getattr(self, "_cmd_smoothed", None)
            if prev is None:
                prev = np.zeros(3, dtype=np.float32)
            if self._arrived or self._facing:
                self._cmd_smoothed = target.astype(np.float32)
            else:
                alpha = 0.15
                self._cmd_smoothed = (1.0 - alpha) * prev + alpha * target
                self._cmd_smoothed[2] = float(target[2])
            self._low_level._cmd[:] = self._cmd_smoothed
        if self._arrived and self._grasp_phase == PHASE_IDLE:
            self._start_grasp(self._info or {})

        pos, rot, grip = None, None, 0.0
        self._gripper_intent = 0.0
        if self._grasp_phase not in (PHASE_IDLE, PHASE_DONE):
            if self._grasp_phase_done():
                self._advance_grasp()
            pos, rot = self._grasp_target()
            if self._grasp_phase in (PHASE_CLOSE, PHASE_POST_CLOSE, PHASE_LIFT):
                grip = 1.0
            # main.py records this binary intent so the saved label flips at PHASE_CLOSE.
            self._gripper_intent = float(grip)
            if grip > 0 and self._grasp_phase == PHASE_CLOSE:
                grip = min(self._grasp_step / max(1, self._grasp_steps()), 1.0)
            self._grasp_step += 1
            if self._grasp_step >= self._grasp_steps():
                self._advance_grasp()

        g_val = self.GRIPPER_OPEN * (1 - grip) + self.GRIPPER_CLOSED * grip

        arm_joints = np.zeros(7, dtype=np.float32)
        waist_joints = np.zeros(3, dtype=np.float32)
        height_cmd = self._low_level._height_cmd
        if (
            self._grasp_phase == PHASE_REALIGN
            and self._start_at_pregrasp
            and self._pregrasp_joints is not None
        ):
            hcfg = self._ik_hand_cfg[self._hand]
            arm_joints = np.array(
                [self._pregrasp_joints.get(jn, 0.0) for jn in hcfg["arm_joints"]], dtype=np.float32
            )
            waist_joints = np.array(
                [self._pregrasp_joints.get(jn, 0.0) for jn in _WAIST], dtype=np.float32
            )
            pos = self._pregrasp
            self._last_arm_joints = arm_joints.copy()
            self._last_waist_joints = waist_joints.copy()
        elif pos is not None:
            IK_DECIM = 3
            cache = getattr(self, "_ik_cache", None)
            if cache is None or (self._low_level._step_counter % IK_DECIM) == 0:
                # Self-collision avoidance on only during grasp execution.
                arm_joints, waist_joints, ik_h, _ = self._solve_ik_wbc(
                    pos, rot, avoid_self_collision=True
                )
                self._ik_cache = (arm_joints, waist_joints, ik_h)
            else:
                arm_joints, waist_joints, ik_h = self._ik_cache
            height_cmd = self._low_level._height_cmd + 0.1 * (ik_h - self._low_level._height_cmd)
            last_arm = getattr(self, "_last_arm_joints", None)
            if last_arm is not None and self._quick_approach and self._grasp_step < 30:
                alpha = float(np.clip(self._grasp_step / 30.0, 0.0, 1.0))
                arm_joints = (1.0 - alpha) * last_arm + alpha * arm_joints
                last_waist = getattr(self, "_last_waist_joints", None)
                if last_waist is not None:
                    waist_joints = (1.0 - alpha) * last_waist + alpha * waist_joints
            self._last_arm_joints = arm_joints.copy()
            self._last_waist_joints = waist_joints.copy()

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[ACT_CMD] = self._low_level._cmd
        action[ACT_HEIGHT] = height_cmd
        action[ACT_WAIST] = waist_joints
        action[ACT_GRIP] = self.GRIPPER_OPEN
        if pos is not None:
            action[ACT_ARM] = arm_joints
            action[ACT_GRIP] = g_val
        elif getattr(self, "_init_upper", None) is not None:
            iu = self._init_upper
            # Only drive the waist when we know _init_upper is a pregrasp pose
            # (init_arm_at_pregrasp). For random arm_init_radius poses the waist
            # noise is tiny and the original behavior (target=0) is fine.
            if self._init_arm_at_pregrasp:
                action[ACT_WAIST] = iu[0:3]
            action[ACT_ARM] = iu[3:10]
            action[ACT_GRIP] = iu[10]
            if self._upper_cmd is None:
                self._upper_cmd = np.concatenate(
                    [
                        obs["base_height"].astype(np.float32),
                        obs["joint_pos"][12:15].astype(np.float32),
                        obs["joint_pos"][22:29].astype(np.float32),
                    ]
                )
            target_upper = action[ACT_UPPER].copy()
            alpha = 0.08 if self._grasp_step < 80 else 0.18
            self._upper_cmd = (1.0 - alpha) * self._upper_cmd + alpha * target_upper
            action[ACT_UPPER] = self._upper_cmd
        return action

    def get_action(self, observation):
        """BasePolicy.get_action's calling convention (molmo_spaces/policy/
        base_policy.py) -- thin wrapper, not a port: returns the same flat
        ACTION_DIM array sample_actions always has, not
        FetchmanPickPlannerPolicy's per-move-group dict (env.step here
        consumes flat arrays, so that representation isn't ported)."""
        return self.sample_actions(observation)

    def get_action_chunk(self, observation):
        """BasePolicy.get_action_chunk's calling convention -- see
        get_action's docstring. Always a single-action chunk (no
        FetchmanPickPlannerPolicy-style RENDER_DECIM batching), so callers
        written against either policy's `get_action_chunk(obs) or
        [get_action(obs)]` fallback pattern work here too."""
        return [self.sample_actions(observation)]

    # molmo_spaces' PlannerPolicy interface (molmo_spaces/policy/base_policy.py)
    # -- planners()/get_phase()/get_all_phases()/retry_count are all abstract
    # there. Added here purely as read-only views onto state sample_actions()
    # already maintains (self._grasp_planner/self._facing/self._arrived/
    # self._grasp_phase/self._grasp_retries_used) -- no existing behavior
    # changes, since nothing above calls these. Shaped after
    # FetchmanPickPlannerPolicy.get_phase/get_all_phases (molmo_spaces/policy/
    # solvers/object_manipulation/fetchman_pick_planner_policy.py), the
    # molmo_spaces analog this file is being reshaped towards.
    @property
    def planners(self):
        return {"grasp": self._grasp_planner}

    @property
    def retry_count(self) -> int:
        return self._grasp_retries_used

    def get_phase(self) -> str:
        if not self._arrived:
            return "facing" if self._facing else "walking"
        return self._grasp_phase

    def get_all_phases(self) -> dict:
        names = ["unknown", "walking", "facing", PHASE_IDLE, *_PHASE_ORDER]
        return {name: i for i, name in enumerate(names)}


def get_config():
    import ml_collections

    return ml_collections.ConfigDict(
        dict(
            agent_name="g1",
            walk_timeout_s=20.0,
            walk_fail_dist=0.2,
            face_yaw_offset=0.0,
            grasp_retry_closer=True,
        )
    )


# ---------------------------------------------------------------------------
# molmo_spaces-native wrapper
#
# G1Controller above is the reference stack's policy: it is constructed bare,
# wired up through set_env/setup, driven by sample_actions() -> flat-15, and
# reads its world through five attributes on the reference G1Env
# (np_random/robot/scene/target/time). molmo_spaces' own policy contract is
# different: policy_factory(config, task), reset(), get_action(obs) -> a
# move-group dict, plus the PlannerPolicy phase accessors.
#
# The two adapters below bridge exactly that gap and nothing else -- the grasp
# logic itself is untouched, so it stays the gold-verified behaviour. This
# mirrors how FetchmanPickPlannerPolicy is structured (a PickPlannerPolicy
# subclass composing an inner grasp planner), which is the shape this file is
# converging on.
# ---------------------------------------------------------------------------


class _NativeTargetView:
    """Presents a molmo_spaces `MlSpacesObject` through the three members the
    reference policy uses on `env.task.target`: `body_id`, and `position()` /
    `quat()` as *methods taking the live MjData*. molmo_spaces exposes those
    two as plain properties, so the data argument is accepted and ignored --
    the property already reads the same live MjData.
    """

    def __init__(self, obj):
        self._obj = obj

    @property
    def body_id(self) -> int:
        return self._obj.body_id

    def position(self, data=None) -> np.ndarray:
        return np.asarray(self._obj.position, dtype=np.float64)

    def quat(self, data=None) -> np.ndarray:
        return np.asarray(self._obj.quat, dtype=np.float64)


class _NativeEnvView:
    """Presents molmo_spaces' `CPUMujocoEnv` + task through the five attributes
    the reference policy reads off its own `G1Env`.

    Explicit named properties, not attribute forwarding: each one states which
    molmo_spaces concept backs it, so a missing piece fails loudly at the right
    place instead of silently resolving somewhere unexpected.
    """

    def __init__(self, task, target_obj, np_random):
        self._task = task
        self._target = _NativeTargetView(target_obj)
        self.np_random = np_random
        # `env.task.target` is read by the reference.
        #
        # Deliberately NO `grasp_frame_pose` attribute: the reference branches
        # on `hasattr(task, "grasp_frame_pose")` and, when present, calls it and
        # unpacks (pos, quat). That is the Open task's sub-body grasp frame
        # (drawer face, door panel). A pick task has none, so the attribute must
        # be *absent* -- that is what selects the target-object pose path.
        self.task = SimpleNamespace(target=self._target)

    @property
    def robot(self):
        return self._task.env.current_robot

    @property
    def scene(self):
        return SimpleNamespace(model=self._task.env.current_model, data=self._task.env.current_data)

    @property
    def target(self):
        return self._target

    @property
    def time(self) -> float:
        return float(self._task.env.current_data.time)


class G1PickPlannerPolicy(PickPlannerPolicy):
    """molmo_spaces planner policy driving the reference G1 pick logic.

    Owns a `G1Controller` (the reference nav + grasp state machine, unmodified)
    and translates at the two boundaries molmo_spaces defines differently:
    construction/reset, and the action shape (flat-15 -> move-group dict).

    Requires G1Config's default WBC-walking mode and the reference G1Robot
    (molmo_spaces/robots/g1.py), whose `_low_level` controller this policy
    writes through -- see G1Controller._low_level.
    """

    def __init__(self, config, task) -> None:
        super().__init__(config, task)
        robot = task.env.current_robot
        if not hasattr(robot, "_low_level"):
            raise TypeError(
                "G1PickPlannerPolicy requires molmo_spaces/robots/g1.py's G1Robot "
                f"(the reference implementation), got {type(robot).__module__}."
                f"{type(robot).__name__}. Point G1Config.robot_factory at "
                "G1Robot.from_mj_data."
            )
        # This policy drives the reference sample_actions(), which advances the
        # WBC gait clock itself -- tell the robot not to advance it too.
        robot._external_gait_clock = True
        self._controller = G1Controller()
        self._controller.setup(
            task.env.current_model,
            task.env.current_data,
            prefix=self.config.robot_config.robot_namespace,
        )

    def _pickup_obj(self):
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        return om.get_object_by_name(self.config.task_config.pickup_obj_name)

    def _build_info(self) -> dict:
        """Construct the reference policy's `info` dict from native state.

        Two shape conversions matter here, and both are the divergences that
        made the independent rewrite behave differently:
          - `target_object_pose` is a 7-vector [pos, quat] (not a 4x4)
          - `valid_grasps` are **object-local** transforms; the reference
            composes them as `Tw @ go`. get_pickup_grasps returns world-frame
            candidates, so they are mapped back into the object frame here.
        """
        obj = self._pickup_obj()
        pose = np.asarray(obj.pose, dtype=np.float64)
        world_grasps = get_pickup_grasps(
            self.task.env, obj, grasp_libraries=self.policy_config.grasp_libraries
        )
        inv_pose = np.linalg.inv(pose)
        local_grasps = [inv_pose @ np.asarray(g, dtype=np.float64) for g in world_grasps]

        # Use the controller's own _xy() (pelvis + PELVIS_FORWARD_OFFSET), not
        # the raw base pose: reset() takes the "already at the goal" branch --
        # skipping nav and starting the grasp immediately -- only when
        # ||_xy() - goal_xy|| < NEAR_GOAL_DIST (0.05m). InteractiveShell drives
        # navigation as its own `nav_to` command, so pick() must always take
        # that branch; anything else falls through to _plan_path, which needs a
        # nav_occupancy_map this info dict does not carry, leaving the policy
        # stuck in the walking phase forever.
        xy = np.asarray(self._controller._xy(), dtype=np.float64)
        to_obj = np.asarray(obj.position, dtype=np.float64)[:2] - xy
        return {
            "target_object_pose": np.concatenate(
                [np.asarray(obj.position, dtype=np.float64), np.asarray(obj.quat, dtype=np.float64)]
            ),
            "target_object_position": np.asarray(obj.position, dtype=np.float64),
            "valid_grasps": np.asarray(local_grasps, dtype=np.float64),
            # InteractiveShell drives navigation as its own `nav_to` command, so
            # the pick skill starts already standing at the object: the goal is
            # where the robot is, facing the target.
            "goal_xy": np.asarray(xy, dtype=np.float64),
            "goal_yaw": float(np.arctan2(to_obj[1], to_obj[0])),
            "pregrasp_joints": None,
        }

    def reset(self, reset_retries: bool = True) -> None:
        target_obj = self._pickup_obj()
        seed = getattr(self.task, "episode_seed", 0) or 0
        self._controller.set_env(_NativeEnvView(self.task, target_obj, np.random.default_rng(seed)))
        info = self._build_info()
        self._controller.reset(info)
        # GraspPoseSensor reads target_poses["grasp"] every tick and requires a
        # 4x4 (utils/pose.py's pose_mat_to_7d asserts the shape) -- info's
        # target_object_pose is the reference's 7-vector, so publish the matrix.
        self._publish_grasp_pose()

    def _publish_grasp_pose(self) -> None:
        """Keep target_poses["grasp"] a live 4x4 for GraspPoseSensor.

        Before the grasp is planned there is no grasp pose yet, so the target
        object's own pose stands in; once the planner has chosen one, publish
        that instead."""
        pos = getattr(self._controller, "_grasp_pos", None)
        rot = getattr(self._controller, "_grasp_rot", None)
        if pos is not None and rot is not None:
            grasp = np.eye(4)
            grasp[:3, :3] = rot
            grasp[:3, 3] = pos
            self.target_poses["grasp"] = grasp
        else:
            self.target_poses["grasp"] = np.asarray(self._pickup_obj().pose, dtype=np.float64)

    def get_action(self, observation):
        flat = self._controller.sample_actions(observation)
        self._publish_grasp_pose()
        return self._flat_to_move_groups(flat)

    def _flat_to_move_groups(self, flat) -> dict:
        """Reference flat-15 -> molmo_spaces move-group dict (see
        controller_g1ms.ACTION_DIM's layout)."""
        flat = np.asarray(flat, dtype=np.float32)
        action = self.robot_view.get_ctrl_dict()
        action["legs_waist"] = np.concatenate(
            [flat[ACT_CMD], [flat[ACT_HEIGHT]], flat[ACT_WAIST]]
        ).astype(np.float32)
        action["right_arm"] = flat[ACT_ARM].astype(np.float32)
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        action[gripper_mg_id] = np.array([flat[ACT_GRIP]], dtype=np.float32)
        if self._controller.done:
            action["done"] = True
        return action

    def get_phase(self) -> str:
        return self._controller.get_phase()

    def get_all_phases(self) -> dict:
        return self._controller.get_all_phases()
