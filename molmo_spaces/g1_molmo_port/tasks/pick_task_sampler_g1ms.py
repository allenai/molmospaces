"""G1TaskSampler: the TaskSampler side of env_g1ms.py's G1Env/G1TaskSampler
split (see env_g1ms.py's own module-level context and G1TaskSampler's
docstring below for the full rationale). Owns reset/sample_task
orchestration, target selection, goal/spawn/placement sampling,
texture/lighting/camera/height randomization, is_terminal/judge_success,
step(), and observation building -- mirroring molmo_spaces/tasks/
task_sampler.py's role, in its own file matching molmo_spaces' layout
(tasks/task_sampler.py, tasks/pick_task_sampler.py,
tasks/pick_g1_task_sampler.py).
"""

import glob
import json as _json
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.g1_molmo_port import ASSETS_DIR
from molmo_spaces.g1_molmo_port.env_g1ms import BASE_MOVE_GROUP, NOISE_MOVE_GROUP, G1Env
from molmo_spaces.g1_molmo_port.sensors_g1ms import OBS_SENSORS, TARGET_POINT_IN_HEAD_SENSOR
from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_IDLE,
    PHASE_LIFT,
    PHASE_OPEN_HOLD,
    PHASE_POST_CLOSE,
    PHASE_REALIGN,
)
from molmo_spaces.robots.g1 import PREFIX

_SENSOR_SUITE = SensorSuite(OBS_SENSORS)


def build_thor_texture_pools():
    """Discover available THOR/repo-local texture files per scene-texture
    category. Pure filesystem/JSON I/O, zero RNG, zero instance state --
    moved here (out of G1Env) since it's "texture randomization stuff", even
    though the actual per-reset *selection* from these pools
    (resolve_texture_pools below is discovery only; G1Env._sample_scene_textures
    is the RNG-dependent selection) has to stay wherever np_random lives
    (G1Env -- construction-order reasons documented on G1TaskSampler's own
    __init__ docstring prevented moving np_random itself in this pass).
    Relocated verbatim from G1Env's old @staticmethod _build_thor_texture_pools.
    """
    from molmo_spaces.g1_molmo_port.components.constants import SCENE_TEXTURE_CATEGORIES

    # repo-local assets/textures/<Category>/ takes precedence over the THOR db
    canonical = set(SCENE_TEXTURE_CATEGORIES.values())
    local_root = ASSETS_DIR / "textures"
    if local_root.is_dir():
        local_pools = {}
        for cat in sorted(canonical):
            files = sorted(str(p) for p in (local_root / cat).glob("*.png"))
            if files:
                local_pools[cat] = files
        if local_pools:
            return local_pools
    db_path = ASSETS_DIR / "objects" / "thor" / "material-database.json"
    mt_path = ASSETS_DIR / "objects" / "thor" / "material_to_textures.json"
    if not db_path.exists() or not mt_path.exists():
        return {}
    with open(db_path) as f:
        cat_db = _json.load(f)
    with open(mt_path) as f:
        mat_tex = _json.load(f)
    canonical = set(SCENE_TEXTURE_CATEGORIES.values())
    canon_by_lc = {c.lower(): c for c in canonical}

    pools: dict[str, set[str]] = {c: set() for c in canonical}
    for db_cat, mats in cat_db.items():
        target = canon_by_lc.get(db_cat.lower())
        if target is None:
            continue
        for m in mats:
            rec = mat_tex.get(m) or {}
            tex = rec.get("_MainTex")
            if not tex:
                continue
            rel = tex.split("Assets/ThorAssets/Textures/", 1)[-1]
            rel = rel.split("Textures/", 1)[-1]
            abs_path = ASSETS_DIR / "objects" / "thor" / "Textures" / rel
            if not abs_path.exists():
                continue
            # MuJoCo's loader only accepts PNG; fall back to sibling .png.
            if abs_path.suffix.lower() != ".png":
                png_alt = abs_path.with_suffix(".png")
                if png_alt.exists():
                    abs_path = png_alt
                else:
                    continue
            pools[target].add(str(abs_path))
    return {c: sorted(v) for c, v in pools.items() if v}


def resolve_texture_pools(randomize_textures, scene_textures_glob):
    """Discovery + fallback-glob + disable-if-empty logic, relocated verbatim
    from G1Env.__init__'s old inline texture-pool setup block. Returns
    (texture_pools, randomize_textures) -- the returned bool may come back
    False even if the input was True, matching gold's own "disable if
    nothing available" fallback (same semantics, just returned instead of
    mutated in place).
    """
    if not randomize_textures:
        return {}, False
    texture_pools = build_thor_texture_pools()
    if not texture_pools:
        # Fallback: legacy flat glob replicated across all categories.
        if Path(scene_textures_glob).is_absolute():
            flat = sorted(glob.glob(scene_textures_glob))
        else:
            flat = sorted(str(p) for p in ASSETS_DIR.glob(scene_textures_glob))
        if flat:
            from molmo_spaces.g1_molmo_port.components.constants import SCENE_TEXTURE_CATEGORIES

            texture_pools = {c: list(flat) for c in set(SCENE_TEXTURE_CATEGORIES.values())}
    if not texture_pools:
        print("[env] texture randomization requested but no textures available; disabling")
        return {}, False
    return texture_pools, True


class G1TaskSampler:
    """Splits G1Env's (env_g1ms.py) TaskSampler-shaped responsibilities (task
    object, reset/sample_task orchestration, randomization, obs/action
    handling, observation building) out of the physics/scene/rendering
    substrate, mirroring molmo_spaces' Env (`molmo_spaces/env/env.py`'s
    `CPUMujocoEnv`) vs. TaskSampler (`molmo_spaces/tasks/task_sampler.py`)
    split -- this file is the TaskSampler side of that split, matching
    molmo_spaces' own file layout (`tasks/task_sampler.py`,
    `tasks/pick_task_sampler.py`, `tasks/pick_g1_task_sampler.py`). G1Env
    (env_g1ms.py) now looks like CPUMujocoEnv: scene/robot/data/rendering/
    occupancy-map substrate only. Everything else (reset/step orchestration,
    target selection, goal/spawn sampling, randomization, obs building)
    lives here. `Sensor` classes and their `OBS_SENSORS` instances live in
    their own `sensors_g1ms.py` module (imported below), matching
    molmo_spaces' own `env/sensors.py`; `MoveGroup` and its
    `BASE_MOVE_GROUP`/`NOISE_MOVE_GROUP` instances stay defined in
    env_g1ms.py (also imported below), matching molmo_spaces' own
    `robots/robot_views/*.py` -- both are env-side, separate from
    task_sampler.py.

    Fully explicit, no proxy/delegation magic: this class holds only
    `self.env` (the wrapped G1Env) plus the relocated methods below, with
    every internal reference to env-owned state or methods spelled out as
    `self.env.X`. Calls to sibling methods that also live on this class
    (e.g. `self._build_obs()`, `self._sample_goal_pose(...)`) stay as plain
    `self.X` -- those genuinely resolve here now.

    `task`, `agent`, `np_random`, and the `target`/`time` properties
    deliberately stay declared on G1Env rather than moving here, for two
    concrete reasons (not just conservatism) -- `viewer_running` has no such
    constraint (turned out unreferenced anywhere) and moved here along with
    `_target_visible_in_head`/`sync_viewer`/`_draw_debug_markers`/
    `_cache_probe_local_geoms` in a follow-up pass, once G1Env's remaining
    method list was re-compared against real CPUMujocoEnv:
    (1) G1Env.__init__ constructs `self.task` and does its own scene-load
        retry loop (calling `_load_scene`, which calls `task.set_objects`)
        BEFORE any G1TaskSampler exists to hold it -- moving `task`
        construction here would mean reordering/duplicating that
        initialization logic, changing gold's exact RNG-consumption order.
    (2) agents/policy_g1ms.py's controller hardcodes `self._env.np_random`,
        `self._env.task`, `self._env.target`, `self._env.time`,
        `self._env.scene`, `self._env.robot`, `self._env._grasp_spawn_radius_min`
        etc. via its own `_env` back-reference -- and `_load_scene` (staying
        on G1Env) is what calls `self.agent.set_env(self)` whenever the scene
        reloads, always passing the raw G1Env. So the agent's `_env` is
        always the raw G1Env regardless of what this class does; those
        attributes have to actually live there for the agent's own
        (unmodified) code to keep working.
    """

    # Defaults for this sampler's own config fields, keyed by the same names
    # config dicts (molmospaces/configs/*.py's get_config()) use. Anything
    # absent from `config` falls back to these -- matching the exact
    # defaults this class's __init__ used to declare as named kwargs, before
    # it took a single `config` object (mirroring molmo_spaces'
    # PickTaskSampler.__init__(self, config)) instead of ~45 individual ones.
    _TASK_SAMPLER_DEFAULTS = {
        "action_noise_std": 0.005,
        "action_noise_stride": 5,
        "arm_init_radius": 0.0,
        "face_yaw_offset": 0.0,
        "goal_offset_xy_noise": 0.0,
        "goal_offset_yaw_noise": 0.0,
        "grasp_spawn_radius_max": 0.80,
        "head_camera_distortion_noise": 0.0,
        "head_camera_fovy_noise": 0.0,
        "head_camera_pos_noise": 0.0,
        "head_camera_rot_noise": 0.0,
        "init_arm_at_pregrasp": False,
        "pregrasp_rot_noise": 0.0,
        "pregrasp_xyz_noise": 0.0,
        "randomize_height": True,
        "randomize_height_favored": 0.95,
        "randomize_height_max": None,
        "randomize_height_min": 0.0,
        "randomize_lighting": False,
        "randomize_lighting_keep_prob": 0.25,
        "randomize_placement": True,
        "randomize_robot_height": False,
        "randomize_robot_height_max": 0.793,
        "randomize_robot_height_min": 0.7,
        "randomize_scene": False,
        "randomize_scene_freq": 1,
        "randomize_textures_keep_prob": 0.25,
        "randomize_textures_solid_color_prob": 0.30,
        "reset_precheck_grasp": True,
        "ring_num_angles": 32,
        "sample_spawn_first": False,
        "skill_profiles": None,
        "spawn_along_line": False,
        "spawn_at_grasp": False,
        "spawn_radius_max": 8.0,
        "spawn_radius_min": 1.0,
        "spawn_reachability_check": True,
        "spawn_visibility_check": False,
        "start_at_pregrasp_joint_noise": 0.0,
        "start_at_pregrasp_xy_noise": 0.0,
        "start_at_pregrasp_yaw_noise": 0.0,
        "terminate_before_grasp_collision": True,
        "terminate_grasp_if_not_visible": True,
        "terminate_on_grasp_collision": True,
        "walk_dist_max": 0.8,
        "walk_dist_min": 0.3,
        "wrist_camera_fovy_noise": 0.0,
        "wrist_camera_pos_noise": 0.0,
        "wrist_camera_rot_noise": 0.0,
    }

    def __init__(self, config):
        """G1TaskSampler now owns constructing its own env (mirroring
        molmo_spaces' real TaskSampler, which also constructs its env in
        __init__ -- task samplers are always single-threaded, so there's no
        reason for the env to be built externally and handed in), and takes
        a single `config` object rather than ~45 named kwargs (mirroring
        molmo_spaces' PickTaskSampler.__init__(self, config: PickBaseConfig)
        -- one config object instead of kwarg soup). `config` is the same
        flat dict/ConfigDict make_env() has always received (every
        molmospaces/configs/*.py's get_config() merged with per-experiment
        overrides); G1Env(**config) picks out its own named parameters and
        ignores the rest via its own `**_unused_task_kwargs`, exactly as
        before -- only the *shape* of what this class receives changed, not
        what ends up on either object.
        """
        self.config = config
        self._env = G1Env(**config)

        d = self._TASK_SAMPLER_DEFAULTS

        def get(name):
            return config.get(name, d[name])

        self._action_noise_offset = np.zeros(10, dtype=np.float64)
        self._action_noise_std = float(get("action_noise_std"))
        self._action_noise_step = 0
        self._action_noise_stride = max(1, int(get("action_noise_stride")))
        self._active_profile = None
        self._arm_init_radius = float(get("arm_init_radius"))
        self._face_yaw_offset = float(get("face_yaw_offset"))
        self._frozen_full_state = None
        self._frozen_obj_idx = None
        self._frozen_reset_counter = 0
        self._frozen_rng_state = None
        self._goal_offset_xy_noise = float(get("goal_offset_xy_noise"))
        self._goal_offset_yaw_noise = float(get("goal_offset_yaw_noise"))
        self._grasp_spawn_radius_max = float(get("grasp_spawn_radius_max"))
        self._head_camera_distortion_noise = float(get("head_camera_distortion_noise"))
        self._head_camera_fovy_noise = float(get("head_camera_fovy_noise"))
        self._head_camera_pos_noise = float(get("head_camera_pos_noise"))
        self._head_camera_rot_noise = float(get("head_camera_rot_noise"))
        self._init_arm_at_pregrasp = bool(get("init_arm_at_pregrasp"))
        self._pregrasp_rot_noise = float(get("pregrasp_rot_noise"))
        self._pregrasp_xyz_noise = float(get("pregrasp_xyz_noise"))
        self._prev_grasp_phase = None
        self._randomize_height = bool(get("randomize_height"))
        self._randomize_height_favored = float(get("randomize_height_favored"))
        randomize_height_max = get("randomize_height_max")
        self._randomize_height_max = (
            None if randomize_height_max is None else float(randomize_height_max)
        )
        self._randomize_height_min = float(get("randomize_height_min"))
        self._randomize_lighting = bool(get("randomize_lighting"))
        self._randomize_lighting_keep_prob = float(get("randomize_lighting_keep_prob"))
        self._randomize_placement = get("randomize_placement")
        self._randomize_robot_height = bool(get("randomize_robot_height"))
        self._randomize_robot_height_max = float(get("randomize_robot_height_max"))
        self._randomize_robot_height_min = float(get("randomize_robot_height_min"))
        self._randomize_scene = get("randomize_scene")
        self._randomize_scene_freq = max(1, int(get("randomize_scene_freq")))
        self._randomize_textures_keep_prob = float(get("randomize_textures_keep_prob"))
        self._randomize_textures_solid_color_prob = float(
            get("randomize_textures_solid_color_prob")
        )
        self._reset_counter = 0
        self._reset_precheck_grasp = bool(get("reset_precheck_grasp"))
        self._ring_num_angles = int(get("ring_num_angles"))
        self._sample_spawn_first = bool(get("sample_spawn_first"))
        self._skill_profiles = []
        skill_profiles = get("skill_profiles")
        if skill_profiles:
            for entry in skill_profiles:
                name, weight, profile = entry
                self._skill_profiles.append((str(name), float(weight), dict(profile)))
            wsum = sum(w for _, w, _ in self._skill_profiles)
            assert wsum > 0, "skill_profiles weights must sum > 0"
            self._skill_profiles = [(n, w / wsum, p) for n, w, p in self._skill_profiles]
        self._spawn_along_line = bool(get("spawn_along_line"))
        self._spawn_at_grasp = bool(get("spawn_at_grasp"))
        self._spawn_radius_max = get("spawn_radius_max")
        self._spawn_radius_min = get("spawn_radius_min")
        self._spawn_reachability_check = bool(get("spawn_reachability_check"))
        self._spawn_visibility_check = get("spawn_visibility_check")
        self._start_at_pregrasp_joint_noise = float(get("start_at_pregrasp_joint_noise"))
        self._start_at_pregrasp_xy_noise = float(get("start_at_pregrasp_xy_noise"))
        self._start_at_pregrasp_yaw_noise = float(get("start_at_pregrasp_yaw_noise"))
        self._terminate_before_grasp_collision = bool(get("terminate_before_grasp_collision"))
        self._terminate_grasp_if_not_visible = bool(get("terminate_grasp_if_not_visible"))
        self._terminate_on_grasp_collision = bool(get("terminate_on_grasp_collision"))
        self._walk_dist_max = float(get("walk_dist_max"))
        self._walk_dist_min = float(get("walk_dist_min"))
        self._wrist_camera_fovy_noise = float(get("wrist_camera_fovy_noise"))
        self._wrist_camera_pos_noise = float(get("wrist_camera_pos_noise"))
        self._wrist_camera_rot_noise = float(get("wrist_camera_rot_noise"))

    @property
    def env(self):
        """Read-only: the wrapped G1Env, owned and constructed by this
        sampler (mirrors BaseMujocoTaskSampler.env's own read-only property
        over its lazily-owned env)."""
        return self._env

    @property
    def viewer_running(self):
        return self.env._viewer is not None and self.env._viewer.is_running()

    def _target_visible_in_head(self):
        """True if the target object projects to a pixel inside the head fisheye
        frame (not behind the camera and within [0,W]x[0,H]). Matches the
        target_point obs: not visible == the (-1,-1)/out-of-frame sentinel."""
        pt = TARGET_POINT_IN_HEAD_SENSOR.get_observation(self.env)
        if pt is None:
            return False
        H, W = self.env.camera_manager.size
        u, v = pt
        return 0.0 <= u <= W and 0.0 <= v <= H

    def sync_viewer(self):
        if not self.env._viewer:
            return
        with self.env._viewer.lock():
            if getattr(self.env, "debug", False):
                self._draw_debug_markers()
            self.env._viewer.sync()

    def _draw_debug_markers(self):
        scn = self.env._viewer.user_scn
        scn.ngeom = 0
        if self.env.agent is None:
            return
        wps = getattr(self.env.agent, "_waypoints", None)
        if wps:
            cur_idx = int(getattr(self.env.agent, "_wp_idx", 0))
            eye = np.eye(3).flatten()
            size = np.array([0.20, 0.003, 0.0], dtype=np.float32)
            z = 0.003
            for i, wp in enumerate(wps):
                if scn.ngeom >= len(scn.geoms):
                    break
                done = i < cur_idx
                rgba = (
                    np.array([0.0, 1.0, 0.0, 0.7], dtype=np.float32)
                    if done
                    else np.array([1.0, 0.1, 0.1, 0.7], dtype=np.float32)
                )
                pos = np.array([float(wp[0]), float(wp[1]), z], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    size=size,
                    pos=pos,
                    mat=eye,
                    rgba=rgba,
                )
                scn.ngeom += 1
        # Probe overlay is purely visual; the real probe stays parked at z=10.
        planner = getattr(self.env.agent, "_grasp_planner", None)
        if planner is None:
            return
        grasp_pos = getattr(planner, "_grasp_pos", None)
        grasp_rot = getattr(planner, "_grasp_rot", None)
        if grasp_pos is None or grasp_rot is None:
            return
        if not hasattr(self.env, "_probe_local_geoms"):
            self.env._probe_local_geoms = self._cache_probe_local_geoms()
        grasp_xpos = np.asarray(grasp_pos, dtype=np.float64)
        grasp_xmat = np.asarray(grasp_rot, dtype=np.float64)
        rgba = np.array([0.7, 0.9, 1.0, 0.45], dtype=np.float32)
        for type_int, size, local_pos, local_mat in self.env._probe_local_geoms:
            if scn.ngeom >= len(scn.geoms):
                break
            world_pos = grasp_xpos + grasp_xmat @ local_pos
            world_mat = grasp_xmat @ local_mat
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=type_int,
                size=np.asarray(size, dtype=np.float32),
                pos=world_pos.astype(np.float32),
                mat=world_mat.flatten().astype(np.float32),
                rgba=rgba,
            )
            scn.ngeom += 1
        # Articulate end pose (slide/hinge target) — orange overlay so we can see
        # where the gripper is supposed to end up at the end of the pull/swing.
        end_pos = getattr(self.env.agent, "_articulate_end_pos", None)
        end_rot = getattr(self.env.agent, "_articulate_end_rot", None)
        if end_pos is not None and end_rot is not None:
            end_xpos = np.asarray(end_pos, dtype=np.float64)
            end_xmat = np.asarray(end_rot, dtype=np.float64)
            end_rgba = np.array([1.0, 0.55, 0.1, 0.45], dtype=np.float32)
            for type_int, size, local_pos, local_mat in self.env._probe_local_geoms:
                if scn.ngeom >= len(scn.geoms):
                    break
                world_pos = end_xpos + end_xmat @ local_pos
                world_mat = end_xmat @ local_mat
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=type_int,
                    size=np.asarray(size, dtype=np.float32),
                    pos=world_pos.astype(np.float32),
                    mat=world_mat.flatten().astype(np.float32),
                    rgba=end_rgba,
                )
                scn.ngeom += 1
            # Joint pivot (purple sphere) + joint axis (cyan line) + intermediate
            # waypoints (small white spheres) — exposes the geometry that drives
            # the articulate trajectory so wrong axis/pivot is visible.
            kind = getattr(self.env.agent, "_articulate_kind", None)
            pivot = getattr(self.env.agent, "_articulate_pivot", None)
            axis = getattr(self.env.agent, "_articulate_axis", None)
            if pivot is not None and scn.ngeom < len(scn.geoms):
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
                    size=np.array([0.03, 0.0, 0.0], dtype=np.float32),
                    pos=np.asarray(pivot, dtype=np.float32),
                    mat=np.eye(3, dtype=np.float32).flatten(),
                    rgba=np.array([0.6, 0.0, 0.8, 0.9], dtype=np.float32),
                )
                scn.ngeom += 1
            if axis is not None and scn.ngeom < len(scn.geoms):
                anchor = np.asarray(pivot if pivot is not None else grasp_xpos, dtype=np.float64)
                a = np.asarray(axis, dtype=np.float64)
                a_norm = float(np.linalg.norm(a))
                if a_norm > 1e-6:
                    a = a / a_norm
                    half = 0.25
                    p0, p1 = anchor - a * half, anchor + a * half
                    mid = (p0 + p1) * 0.5
                    z_axis = a
                    tmp = (
                        np.array([1.0, 0.0, 0.0])
                        if abs(z_axis[0]) < 0.9
                        else np.array([0.0, 1.0, 0.0])
                    )
                    x_axis = np.cross(tmp, z_axis)
                    x_axis /= np.linalg.norm(x_axis) + 1e-9
                    y_axis = np.cross(z_axis, x_axis)
                    mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                        size=np.array([0.006, half, 0.0], dtype=np.float32),
                        pos=mid.astype(np.float32),
                        mat=mat.astype(np.float32),
                        rgba=np.array([0.1, 0.9, 1.0, 0.9], dtype=np.float32),
                    )
                    scn.ngeom += 1
            # Trajectory waypoints (10 small white spheres) along the arc/line.
            from scipy.spatial.transform import Rotation as _R

            n_wp = 10
            disp = getattr(self.env.agent, "_articulate_disp", None)
            ang = getattr(self.env.agent, "_articulate_angle", None)
            for i in range(1, n_wp + 1):
                if scn.ngeom >= len(scn.geoms):
                    break
                t = i / n_wp
                if kind == "slide" and disp is not None:
                    wp = grasp_xpos + t * np.asarray(disp, dtype=np.float64)
                elif kind == "hinge" and ang is not None and pivot is not None and axis is not None:
                    R_t = _R.from_rotvec(
                        np.asarray(axis, dtype=np.float64) * (t * float(ang))
                    ).as_matrix()
                    wp = np.asarray(pivot, dtype=np.float64) + R_t @ (
                        grasp_xpos - np.asarray(pivot, dtype=np.float64)
                    )
                else:
                    continue
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom],
                    type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
                    size=np.array([0.008, 0.0, 0.0], dtype=np.float32),
                    pos=wp.astype(np.float32),
                    mat=np.eye(3, dtype=np.float32).flatten(),
                    rgba=np.array([1.0, 1.0, 1.0, 0.85], dtype=np.float32),
                )
                scn.ngeom += 1

    def _cache_probe_local_geoms(self):
        """Snapshot probe geoms in 'fingers-open' pose in root-body frame so debug viz
        can overlay them at any grasp pose without touching physics."""
        m = self.env.scene.model
        d = self.env.scene.data
        gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_probe_joint")
        if gj < 0:
            return []
        saved = d.qpos.copy()
        qa = int(m.jnt_qposadr[gj])
        d.qpos[qa : qa + 3] = [0, 0, 0]
        d.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
        for jname, val in (("gripper_probe_joint_a", 0.04), ("gripper_probe_joint_b", -0.04)):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = val
        mujoco.mj_forward(m, d)
        probe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_probe")
        descendants = self.env.scene.get_body_descendants(probe_bid) if probe_bid >= 0 else set()
        out = []
        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) not in descendants:
                continue
            out.append(
                (
                    int(m.geom_type[gid]),
                    np.array(m.geom_size[gid], dtype=np.float64).copy(),
                    np.array(d.geom_xpos[gid], dtype=np.float64).copy(),
                    np.array(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3).copy(),
                )
            )
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return out

    def set_agent(self, agent):
        self.env.agent = agent

    def place_base(self, xy, yaw):
        self.env.robot.place(xy, yaw)

    def _build_obs(self):
        return _SENSOR_SUITE.get_observations(self.env, self.env.task)

    def _robot_touches_world(self):
        """Returns True if any robot geom is in contact with a non-robot, non-floor,
        non-target geom. Pure read of MuJoCo's already-computed contact array —
        microseconds. The target object is excluded so policies that brush against
        the bowl during PHASE_REALIGN don't get pre-crash terminated."""
        m, d = self.env.scene.model, self.env.scene.data
        rset = self.env._robot_body_set
        if not rset:
            return False
        # Target body set (for compound objects). May be absent on bare/open tasks.
        tset = getattr(self.env.task, "_target_body_set", None) or set()
        PLANE = mujoco.mjtGeom.mjGEOM_PLANE
        for i in range(d.ncon):
            c = d.contact[i]
            bid1 = int(m.geom_bodyid[c.geom1])
            bid2 = int(m.geom_bodyid[c.geom2])
            b1 = bid1 in rset
            b2 = bid2 in rset
            if b1 == b2:  # both robot (self) OR both non-robot (world-world)
                continue
            # Floor / world plane: don't count.
            if m.geom_type[c.geom1] == PLANE or m.geom_type[c.geom2] == PLANE:
                continue
            # Target object (bowl/etc): don't count — contact with the goal is fine.
            other = bid2 if b1 else bid1
            if other in tset:
                continue
            return True
        return False

    def _robot_has_scene_collision(self):
        m, d = self.env.scene.model, self.env.scene.data
        mujoco.mj_collision(m, d)
        robot_bids = set()
        for bid in range(m.nbody):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(PREFIX):
                robot_bids.add(bid)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            b1_robot = b1 in robot_bids
            b2_robot = b2 in robot_bids
            if b1_robot != b2_robot:
                other = b2 if b1_robot else b1
                bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or ""
                if (
                    m.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE
                    or m.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE
                ):
                    continue
                return True
        return False

    def _sample_goal_pose(self, target_pos, obj=None, attempts=25, sampled_upper=None):
        # fast-fail: no free standoff cell around the target -> unservable
        if not self.env.occ.any_free_in_annulus(
            target_pos[:2], self.env._grasp_spawn_radius_min, self._grasp_spawn_radius_max
        ):
            return None, None
        saved_qpos = self.env.scene.data.qpos.copy()
        saved_ctrl = self.env.scene.data.ctrl.copy()
        # Optional task-driven angle preference (e.g. slider drawers want the robot
        # lined up with the slide axis). Returns list of 2D unit vectors; empty = no
        # preference (PickTask / hinge joints).
        pref_dirs = []
        if hasattr(self.env.task, "preferred_goal_directions"):
            pref_dirs = list(self.env.task.preferred_goal_directions(self.env.scene))
        cone = 0.5  # rad, ~28° half-angle around the preferred direction
        try:
            for _ in range(attempts):
                if pref_dirs:
                    direction = pref_dirs[int(self.env.np_random.integers(len(pref_dirs)))]
                    base_ang = float(np.arctan2(direction[1], direction[0]))
                    ang = base_ang + float(self.env.np_random.uniform(-cone, cone))
                    r = float(
                        self.env.np_random.uniform(
                            self.env._grasp_spawn_radius_min, self._grasp_spawn_radius_max
                        )
                    )
                    xy = np.array(
                        [target_pos[0] + r * np.cos(ang), target_pos[1] + r * np.sin(ang)],
                        dtype=np.float64,
                    )
                    if not self.env.occ.is_free(xy):
                        continue
                else:
                    xy = self.env.occ.sample_near(
                        target_pos[:2],
                        radius_min=self.env._grasp_spawn_radius_min,
                        radius_max=self._grasp_spawn_radius_max,
                        np_random=self.env.np_random,
                    )
                if xy is None:
                    continue
                xy = np.asarray(xy, dtype=np.float64)
                yaw = float(np.arctan2(target_pos[1] - xy[1], target_pos[0] - xy[0]))
                if obj is None:
                    return xy, yaw
                self.env.robot.set_pose(xy, yaw)
                self.env.robot.set_defaults()
                if self.env.agent is not None and hasattr(self.env.agent, "_set_groot_defaults"):
                    self.env.agent._set_groot_defaults()
                if sampled_upper is not None:
                    self.env.robot.apply_upper_pose(sampled_upper)
                mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
                # Gate visibility check on _reset_precheck_grasp (it's a GPU
                # render, so non-deterministic across worker EGL contexts).
                if self._reset_precheck_grasp and not self.env.robot.check_object_visibility(
                    obj.body_id
                ):
                    continue
                if self._robot_has_scene_collision():
                    continue
                return xy, yaw
        finally:
            self.env.scene.data.qpos[:] = saved_qpos
            self.env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
        return None, None

    def sample_goal_pose_for_current_target(self):
        if not self.env.target:
            return None, None
        return self._sample_goal_pose(
            self.env.target.position(self.env.scene.data), self.env.target
        )

    def _sample_spawn_then_goal(self, target_pos, obj, sampled_upper=None, spawn_attempts=50):
        """sample_spawn_first path: pick spawn first (visibility-checked), then pick goal
        on a thin ring around the object, ordering ring candidates by closeness to spawn
        so the resulting A* path is short.
        Returns (spawn_xy, spawn_yaw, goal_xy, goal_yaw) or all-None on failure."""
        saved_qpos = self.env.scene.data.qpos.copy()
        saved_ctrl = self.env.scene.data.ctrl.copy()
        try:
            for _ in range(spawn_attempts):
                spawn_xy = self.env.occ.sample_near(
                    target_pos[:2],
                    radius_min=self._spawn_radius_min,
                    radius_max=self._spawn_radius_max,
                    np_random=self.env.np_random,
                )
                if spawn_xy is None:
                    continue
                spawn_xy = np.asarray(spawn_xy, dtype=np.float64)
                bearing = float(
                    np.arctan2(target_pos[1] - spawn_xy[1], target_pos[0] - spawn_xy[0])
                )
                spawn_yaw = bearing + self.env.np_random.uniform(-0.5, 0.5)
                if not self._place_robot_and_check(
                    spawn_xy, spawn_yaw, obj, sampled_upper, visibility=self._spawn_visibility_check
                ):
                    continue
                if self._spawn_reachability_check and not self.env.occ_safe.same_free_component(
                    spawn_xy, target_pos[:2]
                ):
                    continue
                goal_xy, goal_yaw = self._sample_goal_on_ring(
                    spawn_xy, target_pos, obj, sampled_upper
                )
                if goal_xy is None:
                    continue
                return spawn_xy, spawn_yaw, goal_xy, goal_yaw
        finally:
            self.env.scene.data.qpos[:] = saved_qpos
            self.env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
        return None, None, None, None

    def _sample_goal_on_ring(self, spawn_xy, target_pos, obj, sampled_upper=None):
        """Pick a goal on a single ring (one sampled standoff radius) around the object,
        iterating ring candidates in order of Euclidean closeness to spawn. Runs the
        same visibility + collision checks as _sample_goal_pose."""
        r = float(
            self.env.np_random.uniform(
                self.env._grasp_spawn_radius_min, self._grasp_spawn_radius_max
            )
        )
        thetas = np.linspace(0.0, 2 * np.pi, self._ring_num_angles, endpoint=False)
        cands = np.stack(
            [target_pos[0] + r * np.cos(thetas), target_pos[1] + r * np.sin(thetas)], axis=1
        )
        order = np.argsort(np.linalg.norm(cands - spawn_xy, axis=1))
        for idx in order:
            xy = cands[idx]
            if not self.env.occ.is_free(xy):
                continue
            yaw = float(np.arctan2(target_pos[1] - xy[1], target_pos[0] - xy[0]))
            # Goal-pose visibility check uses MuJoCo segmentation render, which
            # is per-EGL-context (each worker on a different render GPU). Two
            # mates with identical sim state can get different visibility
            # answers → different retry counts → different _obj_idx → different
            # target. Gating on _reset_precheck_grasp lets GRPO turn off this
            # entire "reset feasibility filtering" pathway.
            if self._place_robot_and_check(
                xy, yaw, obj, sampled_upper, visibility=self._reset_precheck_grasp
            ):
                return xy, yaw
        return None, None

    def _sample_spawn_along_line(self, goal_xy, target_pos, obj, sampled_upper, march_step=0.05):
        """spawn_along_line path: place spawn on the straight line extending OUT from
        the goal (away from the object), at a uniformly-sampled clear distance.
        Guarantees the spawn-to-goal segment is collision-free by construction —
        no rejection / retry needed. Returns (spawn_xy, spawn_yaw) or (None, None)."""
        obj_xy = np.asarray(target_pos[:2], dtype=np.float64)
        gxy = np.asarray(goal_xy, dtype=np.float64)
        delta = gxy - obj_xy
        d_goal = float(np.linalg.norm(delta))
        if d_goal < 1e-6:
            return None, None
        direction = delta / d_goal  # away from object, through goal, outward
        # March outward from goal until occ_safe says blocked or we hit the spawn cap.
        # Cap above spawn_radius_max so the upper sampling bound is reachable.
        # Walk length is measured from the GOAL (not from the object) so it's
        # independent of how far the goal happens to land. Decouples the spawn
        # sampler from grasp_spawn_radius and makes the walk length predictable.
        walk_min = self._walk_dist_min
        walk_max = self._walk_dist_max
        cap = walk_max + 0.1
        r_max = 0.0
        # Use occ (15cm robot radius), same as goal sampling. occ_safe (extra 12.5cm
        # dilation for A*) would reject too many goals that landed in tight corners.
        while r_max + march_step <= cap:
            probe = gxy + (r_max + march_step) * direction
            if not self.env.occ.is_free(probe):
                break
            r_max += march_step
        # Offset of spawn behind goal, along the outward direction.
        min_offset = walk_min
        max_offset = min(r_max, walk_max)
        if max_offset < min_offset + 1e-3:
            return None, None
        offset = float(self.env.np_random.uniform(min_offset, max_offset))
        spawn_xy = gxy + offset * direction
        bearing = float(np.arctan2(target_pos[1] - spawn_xy[1], target_pos[0] - spawn_xy[0]))
        spawn_yaw = bearing + float(self.env.np_random.uniform(-0.8, 0.8))
        saved_qpos = self.env.scene.data.qpos.copy()
        saved_ctrl = self.env.scene.data.ctrl.copy()
        try:
            if self._place_robot_and_check(
                spawn_xy, spawn_yaw, obj, sampled_upper, visibility=self._spawn_visibility_check
            ):
                if not self._spawn_reachability_check or self.env.occ_safe.same_free_component(
                    spawn_xy, target_pos[:2]
                ):
                    return spawn_xy, spawn_yaw
        finally:
            self.env.scene.data.qpos[:] = saved_qpos
            self.env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
        return None, None

    def _place_robot_and_check(self, xy, yaw, obj, sampled_upper, visibility):
        """Set robot pose + run mj_forward, then collision/visibility checks. Returns bool."""
        self.env.robot.set_pose(xy, yaw)
        self.env.robot.set_defaults()
        if self.env.agent is not None and hasattr(self.env.agent, "_set_groot_defaults"):
            self.env.agent._set_groot_defaults()
        if sampled_upper is not None:
            self.env.robot.apply_upper_pose(sampled_upper)
        mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
        if self._robot_has_scene_collision():
            return False
        return not (visibility and not self.env.robot.check_object_visibility(obj.body_id))

    def _compute_pregrasp_upper(self, goal_xy, goal_yaw):
        """Run the grasp planner at (goal_xy, goal_yaw) and convert the resulting
        IK pregrasp joints into an 11-element upper-body pose vector matching
        Robot._UPPER_RAND_IDX: [waist_yaw, waist_roll, waist_pitch, right_arm_7, grip]."""
        if self.env.agent is None:
            return None
        info_preview = self.env.task.make_info(self.env.scene, self.env.np_random)
        info_preview["goal_xy"] = goal_xy
        info_preview["goal_yaw"] = goal_yaw
        saved_qpos = self.env.scene.data.qpos.copy()
        saved_ctrl = self.env.scene.data.ctrl.copy()
        try:
            self.env.robot.set_pose(goal_xy, goal_yaw)
            if hasattr(self.env.agent, "_set_groot_defaults"):
                self.env.agent._set_groot_defaults()
            mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
            self.env.agent._grasp_planner.plan(info_preview)
            pregrasp_joints = getattr(self.env.agent._grasp_planner, "_pregrasp_joints", None)
        finally:
            self.env.scene.data.qpos[:] = saved_qpos
            self.env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
        if not pregrasp_joints:
            return None
        arr = np.zeros(11, dtype=np.float32)
        upper_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        for i, jn in enumerate(upper_names):
            if jn in pregrasp_joints:
                arr[i] = float(pregrasp_joints[jn])
        arr[10] = -0.0222  # GRIPPER_OPEN
        return arr

    def _sample_pregrasp_offset(self):
        if self._pregrasp_xyz_noise <= 0 and self._pregrasp_rot_noise <= 0:
            return None
        xyz = (
            self.env.np_random.uniform(-self._pregrasp_xyz_noise, self._pregrasp_xyz_noise, size=3)
            if self._pregrasp_xyz_noise > 0
            else np.zeros(3)
        )
        if self._pregrasp_rot_noise > 0:
            from scipy.spatial.transform import Rotation as _R

            axis = self.env.np_random.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-9)
            angle = self.env.np_random.uniform(-self._pregrasp_rot_noise, self._pregrasp_rot_noise)
            rot = _R.from_rotvec(axis * angle).as_matrix()
        else:
            rot = np.eye(3)
        return (xyz, rot)

    def reset(self, *, seed=None, options=None, freeze=False):
        # freeze=True replays the most recent non-frozen reset's task (same
        # scene, object, placement, spawn). Used for Flow-GRPO style groups
        # where M envs must share an initial state.
        #
        # Implementation: at end of a non-frozen reset we snapshot the entire
        # sim state (qpos/qvel/ctrl/mocap/eq_active/model mutated fields) AND
        # the task's tracking state AND the returned info. On freeze=True we
        # blit that snapshot back and SKIP the entire sample_task recompute,
        # which sidesteps every np_random consumption divergence between the
        # initial and replayed reset. This is the only way to guarantee
        # bit-identical cycle 0 vs cycle 1 within a frozen-task rollout.
        if freeze:
            if self._frozen_full_state is None:
                raise RuntimeError(
                    "reset(freeze=True) called before any non-frozen reset; nothing to repeat"
                )
            fs = self._frozen_full_state
            m, d = self.env.scene.model, self.env.scene.data
            d.qpos[:] = fs["qpos"]
            d.qvel[:] = fs["qvel"]
            if fs["act"] is not None and d.act.size:
                d.act[:] = fs["act"]
            d.ctrl[:] = fs["ctrl"]
            d.mocap_pos[:] = fs["mocap_pos"]
            d.mocap_quat[:] = fs["mocap_quat"]
            d.eq_active[:] = fs["eq_active"]
            m.body_pos[:] = fs["body_pos"]
            m.body_simple[:] = fs["body_simple"]
            m.body_sameframe[:] = fs["body_sameframe"]
            m.geom_matid[:] = fs["geom_matid"]
            m.mat_rgba[:] = fs["mat_rgba"]
            if m.nlight:
                m.light_pos[:] = fs["light_pos"]
                m.light_dir[:] = fs["light_dir"]
                m.light_specular[:] = fs["light_specular"]
                m.light_ambient[:] = fs["light_ambient"]
                m.light_diffuse[:] = fs["light_diffuse"]
                m.light_active[:] = fs["light_active"]
            m.cam_pos[:] = fs["cam_pos"]
            m.cam_quat[:] = fs["cam_quat"]
            m.cam_fovy[:] = fs["cam_fovy"]
            self.env.np_random.bit_generator.state = fs["np_random"]
            self._reset_counter = fs["reset_counter"]
            if hasattr(self.env.task, "_obj_idx") and fs["obj_idx"] is not None:
                self.env.task._obj_idx = fs["obj_idx"]
            self.env.task.target = fs["target"]
            self.env.task._target_z0 = fs["target_z0"]
            self.env.task._target_body_set = fs["target_body_set"]
            self.env._sim_time = 0.0
            self._action_noise_step = 0
            self.env._gripper_precrash = False
            self._prev_grasp_phase = None
            self._action_noise_offset.fill(0.0)
            mujoco.mj_forward(m, d)
            self.env._skip_episode = False
            self.env._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
            return self._build_obs(), dict(fs["info"])
        # Non-frozen path: snapshot RNG bits for old code paths that still read
        # them, then run the full sample_task compute.
        self._frozen_rng_state = self.env.np_random.bit_generator.state
        self._frozen_reset_counter = self._reset_counter
        self._frozen_obj_idx = getattr(self.env.task, "_obj_idx", None)
        self.env.reset(seed=seed)
        self.env._skip_episode = False
        self._reset_counter += 1
        self.env._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
        return self.sample_task()

    def sample_task(self):
        """Retry-until-valid entry point, matching molmo_spaces' TaskSampler.
        sample_task() naming: retries _sample_task() (the per-attempt hook --
        molmo_spaces' equivalent abstract hook is also named _sample_task(env))
        up to 12 times, snapshotting full sim state on success for future
        freeze=True replays. Relocated verbatim from reset()'s own retry loop,
        not rewritten -- same statements, same order, just behind a method
        boundary instead of inlined.

        NOT called from make_env()/__init__: `_sample_task()`'s placement
        logic conditionally checks `self.env.agent is not None`
        (`_reset_precheck_grasp`'s reachability precheck) -- every caller
        constructs/attaches the agent *after* make_env() returns, so
        sampling this early would silently skip that precheck and accept
        placements gold's own (agent-attached) first reset() would reject.
        Deferring the first real sample to the caller's own reset() (as
        before) keeps this bit-exact with gold.
        """
        for _ in range(12):
            try:
                r = self._sample_task(self.env)
            except mujoco.FatalError as e:
                print(f"[env] _sample_task MuJoCo error, retrying: {e}")
                r = None
            if r is not None:
                # Snapshot full sim state for future freeze=True replays.
                m, d = self.env.scene.model, self.env.scene.data
                self._frozen_full_state = {
                    "qpos": d.qpos.copy(),
                    "qvel": d.qvel.copy(),
                    "act": d.act.copy() if d.act.size else None,
                    "ctrl": d.ctrl.copy(),
                    "mocap_pos": d.mocap_pos.copy(),
                    "mocap_quat": d.mocap_quat.copy(),
                    "eq_active": d.eq_active.copy(),
                    "body_pos": m.body_pos.copy(),
                    "body_simple": m.body_simple.copy(),
                    "body_sameframe": m.body_sameframe.copy(),
                    "geom_matid": m.geom_matid.copy(),
                    "mat_rgba": m.mat_rgba.copy(),
                    "light_pos": m.light_pos.copy() if m.nlight else None,
                    "light_dir": m.light_dir.copy() if m.nlight else None,
                    "light_specular": m.light_specular.copy() if m.nlight else None,
                    "light_ambient": m.light_ambient.copy() if m.nlight else None,
                    "light_diffuse": m.light_diffuse.copy() if m.nlight else None,
                    "light_active": m.light_active.copy() if m.nlight else None,
                    "cam_pos": m.cam_pos.copy(),
                    "cam_quat": m.cam_quat.copy(),
                    "cam_fovy": m.cam_fovy.copy(),
                    "np_random": self.env.np_random.bit_generator.state,
                    "reset_counter": self._reset_counter,
                    "obj_idx": getattr(self.env.task, "_obj_idx", None),
                    "target": self.env.task.target,
                    "target_z0": float(getattr(self.env.task, "_target_z0", 0.0)),
                    "target_body_set": set(getattr(self.env.task, "_target_body_set", set())),
                    "info": dict(r[1]) if isinstance(r, tuple) and len(r) >= 2 else {},
                }
                return r
        raise RuntimeError("Could not find valid placement after 12 retries")

    # ---- explicit episode export/restore (fixed eval sets) ----

    def export_reset_state(self, info=None):
        """JSON-able dict capturing everything reset randomized, for exact replay."""
        m, d = self.env.scene.model, self.env.scene.data

        def rel(p):
            p = str(p)
            for base in (ASSETS_DIR, ASSETS_DIR.resolve()):
                try:
                    return str(Path(p).relative_to(base))
                except ValueError:
                    pass
            try:
                return str(Path(p).resolve().relative_to(ASSETS_DIR.resolve()))
            except ValueError:
                return p

        st = {
            "scene": rel(self.env._current_scene_path),
            "scene_textures": {
                c: [rel(p) for p in v]
                for c, v in (self.env.scene._scene_texture_paths or {}).items()
            },
            "target_name": self.env.task.target.name,
            "target_z0": float(self.env.task._target_z0),
            "robot_xy": [float(x) for x in self.env.robot.get_xy()],
            "robot_yaw": float(self.env.robot.get_yaw()),
            "object_name": (info or {}).get("object_name", ""),
            "prompt": (info or {}).get("prompt", ""),
            "init_height": (info or {}).get("init_height"),
            # Structural: the regex that decides which bodies get freejoints,
            # i.e. the qpos layout. Restore re-applies it so the eval env config
            # (--env.objects) doesn't have to match the generation config.
            "object_regex": self.env._object_regex,
            "articulated_regex": self.env._articulated_regex,
            "qpos": d.qpos.tolist(),
            "qvel": d.qvel.tolist(),
            "ctrl": d.ctrl.tolist(),
            "mocap_pos": d.mocap_pos.tolist(),
            "mocap_quat": d.mocap_quat.tolist(),
            "eq_active": d.eq_active.tolist(),
            "body_pos": m.body_pos.tolist(),
            "body_simple": m.body_simple.tolist(),
            "body_sameframe": m.body_sameframe.tolist(),
            "geom_matid": m.geom_matid.tolist(),
            "mat_rgba": m.mat_rgba.tolist(),
            "cam_pos": m.cam_pos.tolist(),
            "cam_quat": m.cam_quat.tolist(),
            "cam_fovy": m.cam_fovy.tolist(),
            "headlight": [
                m.vis.headlight.ambient.tolist(),
                m.vis.headlight.diffuse.tolist(),
                m.vis.headlight.specular.tolist(),
            ],
        }
        if m.nlight:
            st.update(
                light_pos=m.light_pos.tolist(),
                light_dir=m.light_dir.tolist(),
                light_specular=m.light_specular.tolist(),
                light_ambient=m.light_ambient.tolist(),
                light_diffuse=m.light_diffuse.tolist(),
                light_active=m.light_active.tolist(),
            )
            if hasattr(m, "light_castshadow"):
                st["light_castshadow"] = m.light_castshadow.tolist()
        if self.env.camera_manager.fisheye is not None:
            st["fisheye_K"] = self.env.camera_manager.fisheye.K.tolist()
            st["fisheye_D"] = self.env.camera_manager.fisheye.D.tolist()
        return st

    def restore_reset_state(self, st):
        """Exact replay of an export_reset_state() dict. Returns (obs, info)."""
        textures = {
            c: [str(ASSETS_DIR / p) if not Path(p).is_absolute() else p for p in v]
            for c, v in st.get("scene_textures", {}).items()
        }
        # The freejoint regex sets the qpos layout, so the saved state only fits a
        # scene compiled with the same regex. Adopt the generation-time regex
        # (forcing a recompile if it differs) so eval works regardless of the env
        # config's --env.objects.
        regex_changed = False
        if "object_regex" in st and st["object_regex"] != self.env._object_regex:
            self.env._object_regex = st["object_regex"]
            self.env.task._object_regex = st["object_regex"]
            regex_changed = True
        if "articulated_regex" in st and st["articulated_regex"] != self.env._articulated_regex:
            self.env._articulated_regex = st["articulated_regex"]
            regex_changed = True

        def _abs_scene(p):
            p = Path(p)
            return (p if p.is_absolute() else ASSETS_DIR / p).resolve()

        same_scene = (
            not regex_changed
            and self.env._current_scene_path is not None
            and _abs_scene(st["scene"]) == _abs_scene(self.env._current_scene_path)
        )
        same_tex = (
            textures
            == {
                c: [str(p) for p in v]
                for c, v in (self.env.scene._scene_texture_paths or {}).items()
            }
            if self.env.scene
            else False
        )
        if not (same_scene and same_tex):
            self.env._current_scene_path = None
            self.env._load_scene(st["scene"], texture_override=textures)
        m, d = self.env.scene.model, self.env.scene.data
        d.qpos[:] = st["qpos"]
        d.qvel[:] = st["qvel"]
        d.ctrl[:] = st["ctrl"]
        if d.mocap_pos.size:
            d.mocap_pos[:] = st["mocap_pos"]
            d.mocap_quat[:] = st["mocap_quat"]
        if d.eq_active.size:
            d.eq_active[:] = st["eq_active"]
        m.body_pos[:] = st["body_pos"]
        m.body_simple[:] = st["body_simple"]
        m.body_sameframe[:] = st["body_sameframe"]
        m.geom_matid[:] = st["geom_matid"]
        m.mat_rgba[:] = st["mat_rgba"]
        m.cam_pos[:] = st["cam_pos"]
        m.cam_quat[:] = st["cam_quat"]
        m.cam_fovy[:] = st["cam_fovy"]
        hl = m.vis.headlight
        hl.ambient[:] = st["headlight"][0]
        hl.diffuse[:] = st["headlight"][1]
        hl.specular[:] = st["headlight"][2]
        if m.nlight and "light_pos" in st:
            m.light_pos[:] = st["light_pos"]
            m.light_dir[:] = st["light_dir"]
            m.light_specular[:] = st["light_specular"]
            m.light_ambient[:] = st["light_ambient"]
            m.light_diffuse[:] = st["light_diffuse"]
            m.light_active[:] = st["light_active"]
            if "light_castshadow" in st and hasattr(m, "light_castshadow"):
                m.light_castshadow[:] = st["light_castshadow"]
        if "fisheye_K" in st:
            if self.env.camera_manager.fisheye is None:
                self.env._ensure_fisheye(*self.env.camera_manager.size)
            self.env.camera_manager.fisheye.set_intrinsics(
                np.asarray(st["fisheye_K"]), np.asarray(st["fisheye_D"])
            )
        target = next((o for o in self.env.scene.pickable if o.name == st["target_name"]), None)
        if target is None:
            raise RuntimeError(f"restore: target {st['target_name']!r} not in scene {st['scene']}")
        self.env.task.target = target
        self.env.task._target_z0 = float(st["target_z0"])
        self.env.task._target_body_set = self.env.scene.get_body_descendants(target.body_id)
        self.env.task._target_grasps = self.env.task._load_grasps(getattr(target, "asset_id", ""))
        self.env._sim_time = 0.0
        self._action_noise_step = 0
        self.env._gripper_precrash = False
        self._prev_grasp_phase = None
        self._action_noise_offset.fill(0.0)
        # Exact replay is noise-free by definition: without this, a benchmark
        # episode inherits whatever action_noise_std the previous (training)
        # episode's skill profile left behind. Training is unaffected — every
        # env.reset() re-applies the profile's own value.
        self._action_noise_std = 0.0
        self.env._skip_episode = False
        self.env._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
        # _load_scene (robot.set_defaults / agent.setup) forwards the data at the
        # default body_pos, caching frames that a later mj_forward will not fully
        # recompute after body_pos is overwritten -> restored supports (e.g. a
        # lowered table) stay at default height. Reset the data to a blank slate
        # AFTER body_pos is set, re-apply the dynamic state, then forward.
        mujoco.mj_resetData(m, d)
        d.qpos[:] = st["qpos"]
        d.qvel[:] = st["qvel"]
        d.ctrl[:] = st["ctrl"]
        if d.mocap_pos.size:
            d.mocap_pos[:] = st["mocap_pos"]
            d.mocap_quat[:] = st["mocap_quat"]
        if d.eq_active.size:
            d.eq_active[:] = st["eq_active"]
        mujoco.mj_forward(m, d)
        info = self.env.task.make_info(self.env.scene, self.env.np_random)
        info.update(scene=st["scene"], object_name=st.get("object_name", ""))
        if st.get("prompt"):
            info["prompt"] = st["prompt"]
        # The exported qpos was captured AFTER agent.reset applied its sampled
        # upper-body pose. Hand that pose back so agent.reset re-applies the
        # same values (no-op on qpos) instead of clobbering with defaults.
        idx = self.env.robot._UPPER_RAND_IDX
        info["init_upper_pose"] = d.qpos[self.env.robot._qpos_ids[idx]].copy()
        if st.get("init_height") is not None:
            info["init_height"] = float(st["init_height"])
        return self._build_obs(), info

    def _randomize_lights(self):
        m = self.env.scene.model
        rng = self.env.np_random
        n = m.nlight
        m.light_pos[:] = self.env.scene._init_light_pos + rng.uniform(-1.0, 1.0, (n, 3))
        for i in range(n):
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-0.8, 0.8))
            cos, sin = float(np.cos(ang)), float(np.sin(ang))
            d = self.env.scene._init_light_dir[i]
            new_d = d * cos + np.cross(axis, d) * sin + axis * float(np.dot(axis, d)) * (1.0 - cos)
            norm = float(np.linalg.norm(new_d))
            if norm > 1e-6:
                m.light_dir[i] = new_d / norm * float(np.linalg.norm(d))
        scene_bright = float(rng.uniform(0.25, 1.7))
        bright = scene_bright * rng.uniform(0.5, 1.4, (n, 1))
        WARM = np.array([1.0, 0.78, 0.55])
        COOL = np.array([0.65, 0.82, 1.0])
        scene_t = float(rng.uniform(-1.0, 1.0))
        ts = np.clip(scene_t + rng.uniform(-0.35, 0.35, n), -1.0, 1.0)
        tint = np.where(
            ts[:, None] >= 0,
            1.0 + ts[:, None] * (WARM[None] - 1.0),
            1.0 + (-ts[:, None]) * (COOL[None] - 1.0),
        )
        m.light_specular[:] = np.clip(
            self.env.scene._init_light_specular + rng.uniform(-0.3, 0.3, (n, 3)), 0, 1
        )
        m.light_ambient[:] = np.clip(
            (self.env.scene._init_light_ambient + rng.uniform(-0.2, 0.2, (n, 3))) * bright * tint,
            0,
            1,
        )
        m.light_diffuse[:] = np.clip(
            (self.env.scene._init_light_diffuse + rng.uniform(-0.2, 0.2, (n, 3))) * bright * tint,
            0,
            1,
        )
        p_on = float(rng.uniform(0.25, 0.95))
        m.light_active[:] = (rng.uniform(0, 1, n) < p_on).astype(m.light_active.dtype)
        if not m.light_active.any():
            m.light_active[0] = 1
        if hasattr(m, "light_castshadow"):
            m.light_castshadow[:] = (rng.uniform(0, 1, n) < 0.5).astype(m.light_castshadow.dtype)
        if getattr(self.env, "_init_headlight", None) is not None:
            hl = m.vis.headlight
            hl_t = np.clip(scene_t + float(rng.uniform(-0.2, 0.2)), -1, 1)
            hl_tint = (1.0 + hl_t * (WARM - 1.0)) if hl_t >= 0 else (1.0 + (-hl_t) * (COOL - 1.0))
            hl_scale = float(np.clip(scene_bright * rng.uniform(0.6, 1.3), 0.3, 1.6))
            amb0, dif0, spec0 = self.env._init_headlight
            hl.ambient[:] = np.clip(amb0 * hl_scale * hl_tint, 0, 1)
            hl.diffuse[:] = np.clip(dif0 * hl_scale * hl_tint, 0, 1)
            hl.specular[:] = np.clip(spec0 * float(rng.uniform(0.5, 1.2)), 0, 1)
        if self.env._robot_white_mid >= 0:
            rgba = self.env._robot_white_rgba0.copy()
            rgba[:3] = np.clip(rgba[:3] + rng.uniform(-0.1, 0.1, 3).astype(rgba.dtype), 0, 1)
            m.mat_rgba[self.env._robot_white_mid] = rgba

    def _perturb_camera(self, cam_id, pos0, quat0, fovy0, pn, rn, fn):
        if cam_id < 0 or pos0 is None:
            return
        if pn <= 0 and rn <= 0 and fn <= 0:
            return
        m = self.env.scene.model
        rng = self.env.np_random
        m.cam_pos[cam_id] = pos0 + rng.uniform(-pn, pn, 3)
        if rn > 0:
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-rn, rn))
            half = 0.5 * ang
            dq = np.array([float(np.cos(half)), *(axis * float(np.sin(half)))])
            w0, x0, y0, z0 = quat0
            w1, x1, y1, z1 = dq
            m.cam_quat[cam_id] = [
                w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
                w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
            ]
        else:
            m.cam_quat[cam_id] = quat0
        if fn > 0:
            m.cam_fovy[cam_id] = fovy0 + float(rng.uniform(-fn, fn))
        else:
            m.cam_fovy[cam_id] = fovy0

    def _randomize_wrist_camera(self):
        self._perturb_camera(
            self.env.camera_manager.ids.get("wrist_image", -1),
            self.env.camera_manager.wrist_cam_pos0,
            self.env.camera_manager.wrist_cam_quat0,
            self.env.camera_manager.wrist_cam_fovy0,
            self._wrist_camera_pos_noise,
            self._wrist_camera_rot_noise,
            self._wrist_camera_fovy_noise,
        )

    def _randomize_head_camera(self):
        # Head = fisheye. pos/rot perturb all 5 tile cameras (+ head_pov) with
        # the SAME delta so they stay co-located. fovy scales the fisheye K
        # (fx, fy) — this triggers a LUT rebuild via set_intrinsics, ~50ms.
        if not self.env.camera_manager.head_cam_ids_all:
            return
        pn = self._head_camera_pos_noise
        rn = self._head_camera_rot_noise
        fn = self._head_camera_fovy_noise
        dn = self._head_camera_distortion_noise
        if pn <= 0 and rn <= 0 and fn <= 0 and dn <= 0:
            return
        m = self.env.scene.model
        rng = self.env.np_random
        dp = rng.uniform(-pn, pn, 3) if pn > 0 else np.zeros(3)
        if rn > 0:
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-rn, rn))
            half = 0.5 * ang
            dq = np.array([float(np.cos(half)), *(axis * float(np.sin(half)))])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        for cid, pos0, quat0 in zip(
            self.env.camera_manager.head_cam_ids_all,
            self.env.camera_manager.head_cam_pos0_all,
            self.env.camera_manager.head_cam_quat0_all,
        ):
            m.cam_pos[cid] = pos0 + dp
            w0, x0, y0, z0 = quat0
            w1, x1, y1, z1 = dq
            m.cam_quat[cid] = [
                w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
                w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
            ]
        if (fn > 0 or dn > 0) and self.env.camera_manager.fisheye is not None:
            # Perturb around the *calibrated* lens from config, not around the
            # renderer's current K/D -- those already carry the previous
            # episode's noise, which would random-walk across resets.
            cfg = self.env.camera_manager.fisheye_config
            K = np.asarray(cfg.fisheye_K, dtype=float)
            D = np.asarray(cfg.fisheye_D, dtype=float)
            if fn > 0:
                # fovy noise (degrees) → focal-length scale around the calibrated f.
                scale = 1.0 + float(rng.uniform(-fn, fn)) / 90.0
                K[0, 0] *= scale
                K[1, 1] *= scale
            if dn > 0:
                # Proportional noise: each coef shifts by ±dn fraction of its own
                # magnitude. Keeps the distortion model self-consistent — k1
                # changes ~20× more than k4 in absolute terms, same as the
                # calibrated ratio. Avoids edge scrambling from over-perturbed k3/k4.
                D = D * (1.0 + rng.uniform(-dn, dn, 4))
            self.env.camera_manager.fisheye.set_intrinsics(K=K, D=D)

    def _is_descendant(self, child, ancestor):
        cur = int(child)
        while cur > 0:
            if cur == ancestor:
                return True
            cur = int(self.env.scene.model.body_parentid[cur])
        return False

    def _has_freejoint(self, bid):
        if bid <= 0:
            return False
        m = self.env.scene.model
        jadr = int(m.body(bid).jntadr[0])
        return jadr >= 0 and m.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE

    def _support_root(self, body_id):
        m = self.env.scene.model
        bid = int(body_id)
        while bid > 0 and int(m.body_parentid[bid]) != 0:
            bid = int(m.body_parentid[bid])
        return bid

    def _raycast_down_skip_self(self, from_bid):
        """Returns (hit_top_z, hit_body_id), (None, None) on miss, or (None, 'floor')."""
        m, d = self.env.scene.model, self.env.scene.data
        pnt = d.xpos[from_bid].copy().astype(np.float64)
        pnt[2] += 1e-3
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        for _ in range(20):
            gid_arr = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, gid_arr)
            gid = int(gid_arr[0])
            if dist < 0 or gid < 0:
                return None, None
            hit_bid = int(m.geom_bodyid[gid])
            if self._is_descendant(hit_bid, from_bid):
                pnt[2] = pnt[2] - dist - 1e-3
                continue
            if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                return None, "floor"
            return float(pnt[2] - dist), hit_bid
        return None, None

    def _trace_support_chain(self, body_id, max_depth=10):
        current = int(body_id)
        chain = []
        for _ in range(max_depth):
            top_z, hit_bid = self._raycast_down_skip_self(current)
            if hit_bid is None or hit_bid == "floor":
                return 0, None, chain
            node = hit_bid
            while node > 0:
                if self._has_freejoint(node):
                    chain.append(node)
                    current = node
                    break
                if int(self.env.scene.model.body_parentid[node]) == 0:
                    return node, top_z, chain
                node = int(self.env.scene.model.body_parentid[node])
            else:
                return 0, None, chain
        return 0, None, chain

    def _support_group_via_contacts(self, support_bid):
        m, d = self.env.scene.model, self.env.scene.data

        def owner(bid):
            cur = int(bid)
            while cur > 0:
                if self._has_freejoint(cur):
                    return cur
                if int(m.body_parentid[cur]) == 0:
                    return cur
                cur = int(m.body_parentid[cur])
            return 0

        adj = {}
        for i in range(d.ncon):
            c = d.contact[i]
            o1 = owner(int(m.geom_bodyid[int(c.geom1)]))
            o2 = owner(int(m.geom_bodyid[int(c.geom2)]))
            if o1 == o2 or o1 <= 0 or o2 <= 0:
                continue
            adj.setdefault(o1, set()).add(o2)
            adj.setdefault(o2, set()).add(o1)
        visited = {support_bid}
        queue = [support_bid]
        while queue:
            cur = queue.pop()
            for nb in adj.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return [b for b in visited if self._has_freejoint(b)]

    def _randomize_target_support_height(self, target_obj):
        if not self._randomize_height or target_obj is None:
            return
        # Articulated targets sit on the floor and have no movable support; height
        # randomization would also throw off our hard-coded closed-joint=0 reward.
        if getattr(target_obj, "is_articulated", False):
            return
        m, d = self.env.scene.model, self.env.scene.data
        target_bid = int(target_obj.body_id)

        sup_root, sup_top_z, target_chain = self._trace_support_chain(target_bid)
        if sup_root == 0 or sup_top_z is None:
            return
        upper = sup_top_z
        if self._randomize_height_max is not None:
            upper = min(upper, self._randomize_height_max)
        if upper <= self._randomize_height_min:
            return

        mode = float(np.clip(self._randomize_height_favored, self._randomize_height_min, upper))
        new_top = float(self.env.np_random.triangular(self._randomize_height_min, mode, upper))
        dz = new_top - sup_top_z
        if abs(dz) < 1e-3:
            return

        grouped_bids = set()
        grouped_bids.add(target_bid)
        grouped_bids.update(target_chain)
        for bid in range(m.nbody):
            if bid in grouped_bids or bid == sup_root or bid == 0:
                continue
            if int(m.body_parentid[bid]) != 0:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(self.env.scene._robot_prefix) or name in (
                "grasp_probe",
                "gripper_probe",
            ):
                continue
            sr, _, chain = self._trace_support_chain(bid)
            if sr == sup_root:
                grouped_bids.add(bid)
                grouped_bids.update(chain)
        # Contact-graph union catches edge-perched objects the raycast misses.
        for bid in self._support_group_via_contacts(sup_root):
            if bid in grouped_bids or bid == sup_root or bid == 0:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(self.env.scene._robot_prefix) or name in (
                "grasp_probe",
                "gripper_probe",
            ):
                continue
            grouped_bids.add(bid)

        # when lowering, furniture under the support rides down with it
        if dz < 0:
            sup_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == sup_root]
            if sup_geoms:
                sup_xy = d.geom_xpos[sup_geoms][:, :2]
                lo = sup_xy.min(axis=0) - 0.05
                hi = sup_xy.max(axis=0) + 0.05
                _SKIP = ("wall", "floor", "ceiling", "door", "window")
                under_bids = set()
                for bid in range(1, m.nbody):
                    if bid in grouped_bids or bid == sup_root or int(m.body_parentid[bid]) != 0:
                        continue
                    c = d.xipos[bid]
                    if not (lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and c[2] < sup_top_z):
                        continue
                    name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
                    if (
                        name.startswith(self.env.scene._robot_prefix.lower())
                        or "probe" in name
                        or any(k in name for k in _SKIP)
                    ):
                        continue
                    under_bids.add(bid)
                if under_bids:
                    for j in range(m.njnt):
                        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                            continue
                        bid = int(m.jnt_bodyid[j])
                        if (
                            bid in grouped_bids
                            or bid in under_bids
                            or int(m.body_parentid[bid]) != 0
                        ):
                            continue
                        sr, _, chain = self._trace_support_chain(bid)
                        if sr in under_bids:
                            under_bids.add(bid)
                            under_bids.update(chain)
                grouped_bids.update(under_bids)

        # Freejoints shift via qpos; static bodies via body_pos (clear simple/sameframe).
        grouped_qadrs = []
        static_bids = []
        for b in grouped_bids:
            jadr = int(m.body(b).jntadr[0])
            if jadr >= 0 and m.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                grouped_qadrs.append(int(m.jnt_qposadr[jadr]))
            elif int(m.body_parentid[b]) == 0:
                static_bids.append(int(b))

        m.body_simple[sup_root] = 0
        m.body_sameframe[sup_root] = 0
        m.body_pos[sup_root, 2] += dz

        for b in static_bids:
            m.body_simple[b] = 0
            m.body_sameframe[b] = 0
            m.body_pos[b, 2] += dz

        for qa in grouped_qadrs:
            d.qpos[qa + 2] += dz

        # Sleep keeps asleep bodies frozen — toggle off so kinematics refresh.
        sleep_bit = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        sleep_was_on = bool(m.opt.enableflags & sleep_bit)
        m.opt.enableflags &= ~sleep_bit
        mujoco.mj_forward(m, d)
        if sleep_was_on:
            m.opt.enableflags |= sleep_bit

    def _sample_skill_profile(self):
        if not self._skill_profiles:
            self._active_profile = None
            return
        names = [n for n, _, _ in self._skill_profiles]
        weights = [w for _, w, _ in self._skill_profiles]
        idx = int(self.env.np_random.choice(len(names), p=weights))
        name, _, profile = self._skill_profiles[idx]
        self._active_profile = name
        if "spawn_at_grasp" in profile:
            self._spawn_at_grasp = bool(profile["spawn_at_grasp"])
        if "sample_spawn_first" in profile:
            self._sample_spawn_first = bool(profile["sample_spawn_first"])
        if "ring_num_angles" in profile:
            self._ring_num_angles = int(profile["ring_num_angles"])
        if "spawn_along_line" in profile:
            self._spawn_along_line = bool(profile["spawn_along_line"])
        if "spawn_reachability_check" in profile:
            self._spawn_reachability_check = bool(profile["spawn_reachability_check"])
        if "walk_dist_min" in profile:
            self._walk_dist_min = float(profile["walk_dist_min"])
        if "walk_dist_max" in profile:
            self._walk_dist_max = float(profile["walk_dist_max"])
        if "spawn_radius_min" in profile:
            self._spawn_radius_min = float(profile["spawn_radius_min"])
        if "spawn_radius_max" in profile:
            self._spawn_radius_max = float(profile["spawn_radius_max"])
        if "spawn_visibility_check" in profile:
            self._spawn_visibility_check = bool(profile["spawn_visibility_check"])
        if "arm_init_radius" in profile:
            self._arm_init_radius = float(profile["arm_init_radius"])
        if "pregrasp_xyz_noise" in profile:
            self._pregrasp_xyz_noise = float(profile["pregrasp_xyz_noise"])
        if "pregrasp_rot_noise" in profile:
            self._pregrasp_rot_noise = float(profile["pregrasp_rot_noise"])
        if "action_noise_std" in profile:
            self._action_noise_std = float(profile["action_noise_std"])
        if "face_yaw_offset" in profile:
            self._face_yaw_offset = float(profile["face_yaw_offset"])
        if "randomize_height" in profile:
            self._randomize_height = bool(profile["randomize_height"])
        if "randomize_height_min" in profile:
            self._randomize_height_min = float(profile["randomize_height_min"])
        if "randomize_height_favored" in profile:
            self._randomize_height_favored = float(profile["randomize_height_favored"])
        if "randomize_height_max" in profile:
            self._randomize_height_max = (
                None
                if profile["randomize_height_max"] is None
                else float(profile["randomize_height_max"])
            )
        if "start_at_pregrasp_xy_noise" in profile:
            self._start_at_pregrasp_xy_noise = float(profile["start_at_pregrasp_xy_noise"])
        if "start_at_pregrasp_yaw_noise" in profile:
            self._start_at_pregrasp_yaw_noise = float(profile["start_at_pregrasp_yaw_noise"])
        if "start_at_pregrasp_joint_noise" in profile:
            self._start_at_pregrasp_joint_noise = float(profile["start_at_pregrasp_joint_noise"])
        if "init_arm_at_pregrasp" in profile:
            self._init_arm_at_pregrasp = bool(profile["init_arm_at_pregrasp"])

    def init_scene(self, env):
        """New-scene setup: (re)load the MJCF, matching PickTaskSampler.
        init_scene's per-house-boundary role in the base contract --
        G1TaskSampler's "house" boundary is a scene (re)load rather than a
        house-index advance, gated by the same condition
        _setup_scene_and_robot always checked before this split existed
        (no scene picked yet, or randomize_scene due this reset). Returns
        True on success, False if scene loading failed (caller retries via
        _setup_scene_and_robot's own (False, None) return).
        """
        _need_objects = not getattr(env.task, "objects", None)
        if not (
            len(env._scene_paths) > 1
            and (
                _need_objects
                or (
                    self._randomize_scene
                    and (self._reset_counter % self._randomize_scene_freq) == 0
                )
            )
        ):
            return True
        try:
            env._load_scene(env._scene_paths[env.np_random.integers(len(env._scene_paths))])
        except (ValueError, IndexError, OSError) as e:
            print(f"[env] skipping scene (load failed: {type(e).__name__}: {e})")
            return False
        except Exception as e:
            # Catch-all: occupancy_map's PIL.Image.open can raise
            # UnidentifiedImageError when concurrent workers race on the
            # _thormap.png cache; scene XML compile can throw assorted
            # MuJoCo errors. Killing a worker here cascades to the whole
            # vec_env via EOFError on master pipes, so swallow + retry.
            print(f"[env] skipping scene (load error: {type(e).__name__}: {e})")
            return False
        return True

    def randomize_scene(self, env, robot_view):
        """Per-reset randomization: texture/lighting/camera perturbation and
        robot defaults/pose -- matching PickTaskSampler.randomize_scene's
        per-reset (not per-house) contract, run every reset regardless of
        whether init_scene reloaded the MJCF this time. `robot_view` is
        `env.robot` (G1Env has no separate move-group-view abstraction, so
        the two are the same object here). Returns sampled_upper (the
        randomized arm pose, if any, already applied to the robot).
        """
        env.scene.reset()
        # Must run after scene.reset() — reset restores geom_matid to defaults.
        if (
            env._randomize_textures
            and env.scene.scene_matids
            and env.scene.scene_geom_ids
            and env.np_random.random() >= self._randomize_textures_keep_prob
        ):
            color_mids = env.scene.scene_color_matids
            if color_mids:
                rand_rgba = env.np_random.uniform(0.0, 1.0, size=(len(color_mids), 4)).astype(
                    np.float32
                )
                rand_rgba[:, 3] = 1.0
                for i, mid in enumerate(color_mids):
                    env.scene.model.mat_rgba[mid] = rand_rgba[i]
            scp = self._randomize_textures_solid_color_prob
            for cat, gids in env.scene.scene_geom_ids.items():
                if not gids:
                    continue
                tex_mids = env.scene.scene_matids.get(cat)
                if not tex_mids:
                    continue
                tex_picks = env.np_random.choice(tex_mids, size=len(gids))
                use_solid = (env.np_random.random(size=len(gids)) < scp) if color_mids else None
                solid_picks = (
                    env.np_random.choice(color_mids, size=len(gids)) if color_mids else None
                )
                for k, gid in enumerate(gids):
                    if use_solid is not None and use_solid[k]:
                        env.scene.model.geom_matid[gid] = int(solid_picks[k])
                    else:
                        env.scene.model.geom_matid[gid] = int(tex_picks[k])
        if (
            self._randomize_lighting
            and env.scene.model.nlight > 0
            and env.np_random.random() >= self._randomize_lighting_keep_prob
        ):
            self._randomize_lights()
        self._randomize_wrist_camera()
        self._randomize_head_camera()
        robot_view.zero_velocities()
        robot_view.set_defaults()

        sampled_upper = (
            robot_view.sample_upper_pose(env.np_random, self._arm_init_radius)
            if self._arm_init_radius > 0
            else None
        )
        if sampled_upper is not None:
            robot_view.apply_upper_pose(sampled_upper)
        return sampled_upper

    def _setup_scene_and_robot(self, env):
        """Scene (re)load + texture/lighting/camera randomization + robot
        defaults -- the self-contained first phase of _sample_task(), before
        target selection begins. Its only "output" other than mutated
        env.scene/env.robot state is sampled_upper (the randomized arm
        pose, if any, already applied to the robot here). Returns (True,
        sampled_upper) on success, or (False, None) if scene loading failed
        (caller treats this the same as _sample_task's own retry-worthy
        `return None`). Thin orchestrator over init_scene()/randomize_scene()
        now -- same statements, same order, just behind two named hooks
        matching PickTaskSampler's own init_scene/randomize_scene contract.
        """
        env._sim_time = 0.0
        self._action_noise_step = 0
        env._gripper_precrash = False
        self._action_noise_offset.fill(0.0)
        self._sample_skill_profile()
        if not self.init_scene(env):
            return False, None
        sampled_upper = self.randomize_scene(env, env.robot)
        return True, sampled_upper

    def _sample_task(self, env):
        """Per-attempt sample: scene/target/placement selection, all the way
        to a valid (obs, info) or None (caller retries). This is the abstract
        hook molmo_spaces' TaskSampler contract calls _sample_task(env),
        taking env explicitly now (previously read implicitly off
        self.env) -- renamed from _try_reset, body split into
        _setup_scene_and_robot() (its self-contained first phase) +
        _select_target_and_place() (the rest), neither individually
        rewritten.
        """
        ok, sampled_upper = self._setup_scene_and_robot(env)
        if not ok:
            return None
        return self._select_target_and_place(env, sampled_upper)

    def _select_target_and_place(self, env, sampled_upper):
        """Target selection, goal/spawn sampling, realign/pregrasp handling,
        and final robot placement -- the rest of _sample_task() after scene/
        robot setup. Kept as ONE method rather than split further: unlike
        _setup_scene_and_robot() (one early-return, one output), this block
        has many retry-worthy early-return points and heavy cross-variable
        threading (sampled_upper gets *reassigned* when init_arm_at_pregrasp
        fires; goal_xy/goal_yaw computed then perturbed; obj/tgt/realign_info/
        _spawn_first_xy/_used_pregrasp_upper all feed into placement below) --
        splitting THIS further would risk a dropped early-return or
        misthreaded variable the way a first attempt at Stage 4's
        MoveGroup split did. Extracting it whole (same statements, same
        order, same every-early-return) carries none of that risk; only
        the interface (one input, one None-or-(obs,info) output, identical
        to what _sample_task already returned) is new.
        """
        obj = env.task.select_target(env.np_random)
        # Jitter object xy (per-task) before reading pos so goal sampling sees the perturbed location.
        env.task.perturb_objects(env.scene, env.np_random)
        # Must run before goal sampling so the goal lands at the new z.
        self._randomize_target_support_height(obj)
        # Run init_target_tracking BEFORE goal sampling so the close task's drawer
        # (which gets opened by init_target_tracking) is at its actual starting
        # pose when we sample around it. Open task / pick task are unaffected
        # since their init_target_tracking doesn't move the target.
        env.task.init_target_tracking(env.scene)
        # Use the task's grasp frame (the *moving* body, e.g. drawer) for goal sampling
        # when available — otherwise fall back to the object root.
        if hasattr(env.task, "grasp_frame_pose"):
            tgt = env.task.grasp_frame_pose(env.scene)[0]
        else:
            tgt = obj.position(env.scene.data)

        # When sample_spawn_first is on (and we're not in spawn_at_grasp short-circuit),
        # pick spawn first then derive goal on the closest ring point.
        _spawn_first_xy = _spawn_first_yaw = None
        if self._sample_spawn_first and not self._spawn_at_grasp:
            sxy, syaw, goal_xy, goal_yaw = self._sample_spawn_then_goal(
                tgt, obj, sampled_upper=sampled_upper
            )
            if goal_xy is None:
                env.task.target = None
                return None
            _spawn_first_xy, _spawn_first_yaw = sxy, syaw
        else:
            goal_xy, goal_yaw = self._sample_goal_pose(tgt, obj, sampled_upper=sampled_upper)
            if goal_xy is None:
                env.task.target = None
                return None
        # init_arm_at_pregrasp: now that we have goal_xy, compute pregrasp_joints
        # and use them as sampled_upper for the rest of the pipeline. Spawn checks,
        # the final apply_upper_pose, and info["init_upper_pose"] all flow from
        # sampled_upper — overwriting it here = the arm is at pregrasp from the
        # moment the robot is placed, not after a switch later.
        # Track whether sampled_upper is the pregrasp pose (vs random sample), so
        # the agent can apply pregrasp-only behaviors (waist holding, stall recovery)
        # without changing baseline configs.
        _used_pregrasp_upper = False
        if self._init_arm_at_pregrasp and env.agent is not None:
            pregrasp_upper = self._compute_pregrasp_upper(goal_xy, goal_yaw)
            if pregrasp_upper is not None:
                sampled_upper = pregrasp_upper
                _used_pregrasp_upper = True
        # Misalign spawn (when spawn_at_grasp) so controller must realign — base-motion-during-grasp demos.
        if self._goal_offset_xy_noise > 0:
            goal_xy = goal_xy + env.np_random.uniform(
                -self._goal_offset_xy_noise, self._goal_offset_xy_noise, size=2
            )
        if self._goal_offset_yaw_noise > 0:
            goal_yaw = float(
                goal_yaw
                + env.np_random.uniform(-self._goal_offset_yaw_noise, self._goal_offset_yaw_noise)
            )

        if env.agent is not None and self._reset_precheck_grasp:
            info_preview = env.task.make_info(env.scene, env.np_random)
            info_preview["goal_xy"] = goal_xy
            info_preview["goal_yaw"] = goal_yaw
            if not env.agent.precheck_grasp(info_preview):
                env.task.target = None
                return None

        realign_info = None
        spp_xy_noise = self._start_at_pregrasp_xy_noise
        spp_yaw_noise = self._start_at_pregrasp_yaw_noise
        if (spp_xy_noise > 0 or spp_yaw_noise > 0) and env.agent is not None:
            pregrasp_joints = getattr(env.agent._grasp_planner, "_pregrasp_joints", None)
            if pregrasp_joints:
                # WBC walks forward well, sideways/yaw poorly — only spawn behind goal in body-x.
                axis = "x"
                offset = -float(env.np_random.uniform(spp_xy_noise * 0.5, spp_xy_noise))
                cy, sy = float(np.cos(goal_yaw)), float(np.sin(goal_yaw))
                spawn_xy = np.array(goal_xy, dtype=np.float64).copy()
                spawn_yaw = float(goal_yaw)
                if axis == "x":
                    spawn_xy[0] += offset * cy
                    spawn_xy[1] += offset * sy
                elif axis == "y":
                    spawn_xy[0] += -offset * sy
                    spawn_xy[1] += offset * cy
                else:
                    spawn_yaw = float(goal_yaw + offset)
                pj = dict(pregrasp_joints)
                jn = self._start_at_pregrasp_joint_noise
                if jn > 0:
                    perturb = {
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                        "right_wrist_roll_joint",
                        "right_wrist_pitch_joint",
                        "right_wrist_yaw_joint",
                        "waist_yaw_joint",
                        "waist_roll_joint",
                        "waist_pitch_joint",
                    }
                    m = env.scene.model
                    for name in list(pj.keys()):
                        if name not in perturb:
                            continue
                        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{PREFIX}{name}")
                        if jid < 0:
                            continue
                        lo, hi = m.jnt_range[jid]
                        val = pj[name] + float(env.np_random.uniform(-jn, jn))
                        pj[name] = float(np.clip(val, lo, hi))
                realign_info = dict(
                    start_at_pregrasp=True,
                    realign_axis=axis,
                    realign_offset=offset,
                    pregrasp_joints=pj,
                    spawn_xy=spawn_xy,
                    spawn_yaw=spawn_yaw,
                )

        if realign_info is not None:
            xy, yaw = realign_info["spawn_xy"], realign_info["spawn_yaw"]
        elif _spawn_first_xy is not None:
            xy, yaw = _spawn_first_xy, _spawn_first_yaw
        elif self._spawn_at_grasp:
            xy, yaw = goal_xy, goal_yaw
        elif self._spawn_along_line:
            xy, yaw = self._sample_spawn_along_line(goal_xy, tgt, obj, sampled_upper)
            if xy is None:
                return None
        elif self._randomize_placement:
            saved_qpos = env.scene.data.qpos.copy()
            saved_ctrl = env.scene.data.ctrl.copy()
            xy, yaw = None, None
            for _ in range(50):
                cand_xy = env.occ.sample_near(
                    tgt[:2],
                    radius_min=self._spawn_radius_min,
                    radius_max=self._spawn_radius_max,
                    np_random=env.np_random,
                )
                if cand_xy is None:
                    continue
                cand_xy = np.asarray(cand_xy, dtype=np.float64)
                if self._spawn_visibility_check:
                    bearing = float(np.arctan2(tgt[1] - cand_xy[1], tgt[0] - cand_xy[0]))
                    cand_yaw = bearing + env.np_random.uniform(-0.5, 0.5)
                else:
                    cand_yaw = env.np_random.uniform(-np.pi, np.pi)
                env.robot.set_pose(cand_xy, cand_yaw)
                env.robot.set_defaults()
                if env.agent is not None and hasattr(env.agent, "_set_groot_defaults"):
                    env.agent._set_groot_defaults()
                if sampled_upper is not None:
                    env.robot.apply_upper_pose(sampled_upper)
                mujoco.mj_forward(env.scene.model, env.scene.data)
                if self._robot_has_scene_collision():
                    continue
                if self._spawn_visibility_check and not env.robot.check_object_visibility(
                    obj.body_id
                ):
                    continue
                if self._spawn_reachability_check and not env.occ_safe.same_free_component(
                    cand_xy, tgt[:2]
                ):
                    continue
                xy, yaw = cand_xy, cand_yaw
                break
            env.scene.data.qpos[:] = saved_qpos
            env.scene.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(env.scene.model, env.scene.data)
            if xy is None:
                return None
        else:
            xy, yaw = env.robot.get_xy(), env.robot.get_yaw()

        env.robot.set_pose(xy, yaw)
        env.robot.set_defaults()
        if realign_info is not None:
            env.robot.apply_arm_pose(realign_info["pregrasp_joints"])
        elif sampled_upper is not None:
            env.robot.apply_upper_pose(sampled_upper)
        env.robot.zero_velocities()
        mujoco.mj_forward(env.scene.model, env.scene.data)

        # robot must end up on mapped floor (catches un-placed robot at origin)
        if not env.occ.is_free(env.robot.get_xy()):
            return None

        # nav spawns must be navigable to the target on occ_safe
        if (
            self._spawn_reachability_check
            and realign_info is None
            and not self._spawn_at_grasp
            and not env.occ_safe.same_free_component(env.robot.get_xy(), tgt[:2])
        ):
            return None

        weld = mujoco.mj_name2id(env.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, "pelvis_weld")
        if weld >= 0:
            env.scene.data.eq_active[weld] = 0
        # init_target_tracking was already called earlier (before goal sampling),
        # so we don't need to call it again here.

        pregrasp_offset = self._sample_pregrasp_offset()

        info = env.task.make_info(env.scene, env.np_random)
        info.update(
            distance=float(np.linalg.norm(env.robot.get_xy() - tgt[:2])),
            occupancy_map=env.occ,
            nav_occupancy_map=env.occ_safe,
            goal_xy=goal_xy,
            goal_yaw=goal_yaw,
            init_upper_pose=sampled_upper,
            pregrasp_offset=pregrasp_offset,
            init_arm_at_pregrasp=_used_pregrasp_upper,
            skill_profile=(self._active_profile or "default"),
            face_yaw_offset_max=self._face_yaw_offset,
            scene=env._scene_name(),
        )
        if realign_info is not None:
            info.update(
                start_at_pregrasp=True,
                realign_axis=realign_info["realign_axis"],
                realign_offset=realign_info["realign_offset"],
                pregrasp_joints=realign_info["pregrasp_joints"],
            )
        if self._randomize_robot_height and (realign_info is not None or self._spawn_at_grasp):
            info["init_height"] = float(
                env.np_random.uniform(
                    self._randomize_robot_height_min, self._randomize_robot_height_max
                )
            )
        obs = self._build_obs()
        return obs, info

    def is_terminal(self, reward):
        """Task-shaped terminal check, mirroring molmo_spaces' BaseMujocoTask.
        is_terminal() naming: delegates to self.env.task.is_terminated() when the
        task defines one (e.g. OpenTask), else falls back to G1Env's own
        reward-threshold rule -- PickTask itself defines neither
        is_terminated nor is_success (see judge_success below), relying
        entirely on this fallback. Relocated verbatim from step(), not
        rewritten.
        """
        if hasattr(self.env.task, "is_terminated"):
            return bool(self.env.task.is_terminated(self.env.scene))
        return reward > 0.1

    def judge_success(self, reward):
        """Task-shaped success check, mirroring molmo_spaces' BaseMujocoTask.
        judge_success() naming: delegates to self.env.task.is_success() when
        defined, else falls back to G1Env's own reward-threshold rule.
        Relocated verbatim from step(), not rewritten.
        """
        if hasattr(self.env.task, "is_success"):
            return bool(self.env.task.is_success(self.env.scene))
        return reward > 0.04

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        # Record pre-noise base velocity command for next-step observation.
        if len(action) >= 3:
            self.env._last_base_vel_cmd = (
                BASE_MOVE_GROUP.action_view(action).astype(np.float32).copy()
            )
        if self._action_noise_std > 0 and len(action) == 15:
            in_precision = (
                self.env.agent is not None
                and hasattr(self.env.agent, "in_precision_phase")
                and self.env.agent.in_precision_phase()
            )
            if not in_precision:
                # Resample noise once per record-stride env.steps so saved frames stay consistent.
                if self._action_noise_step % self._action_noise_stride == 0:
                    self._action_noise_offset = self.env.np_random.normal(
                        0.0, self._action_noise_std, size=10
                    )
                self._action_noise_step += 1
                action = action.copy()
                action[NOISE_MOVE_GROUP.action_slice] += self._action_noise_offset
            else:
                self._action_noise_offset.fill(0.0)
                self._action_noise_step = 0
        # Waist envelope is enforced upstream by the WBC IK joint limits (policy.py).
        terminated, sim_error = False, None
        lock = self.env._viewer.lock() if self.env._viewer else nullcontext()
        try:
            with lock:
                if (
                    self.env.robot is not None
                    and hasattr(self.env.robot, "execute_action")
                    and len(action) == 15
                ):
                    self.env.robot.execute_action(action)
                else:
                    mask = ~np.isnan(action)
                    if np.any(mask):
                        self.env.scene.data.ctrl[self.env.robot.act_ids[mask]] = action[mask]
                mujoco.mj_step(
                    self.env.scene.model, self.env.scene.data, nstep=self.env.robot.n_substeps
                )
            if not self.env.robot.state_is_finite():
                terminated, sim_error = True, "non-finite state"
            elif self.env.robot.pelvis_height() < 0.15:
                terminated, sim_error = True, f"fell (z={self.env.robot.pelvis_height():.3f})"
            else:
                # Pre-grasp gripper-collision check: active while walking/realigning,
                # off once controller starts reaching for the object (APPROACH onward).
                # Set terminate_before_grasp_collision=False in env cfg to skip this entirely
                # (e.g. during RL fine-tuning where collisions are part of exploration).
                if self._terminate_before_grasp_collision:
                    phase = (
                        getattr(self.env.agent, "_grasp_phase", None)
                        if self.env.agent is not None
                        else None
                    )
                    pre_grasp = phase in (None, PHASE_IDLE, PHASE_REALIGN)
                    if pre_grasp and self._robot_touches_world():
                        self.env._gripper_precrash = True
                        terminated, sim_error = True, "robot hit world before grasp"
                # During-grasp collision check: terminate when the robot
                # contacts any non-floor, non-target geom while in the actual
                # grasping phases (arm reaching / closing / lifting). The
                # target body set is already excluded inside _robot_touches_world,
                # so contact with the bowl during CLOSE/LIFT doesn't count.
                if not terminated and self._terminate_on_grasp_collision:
                    phase = (
                        getattr(self.env.agent, "_grasp_phase", None)
                        if self.env.agent is not None
                        else None
                    )
                    in_grasp = phase in (
                        PHASE_APPROACH,
                        PHASE_DESCEND,
                        PHASE_OPEN_HOLD,
                        PHASE_CLOSE,
                        PHASE_POST_CLOSE,
                        PHASE_LIFT,
                    )
                    if in_grasp and self._robot_touches_world():
                        terminated, sim_error = True, "robot hit world during grasp"
                # Visibility check: at the single step the controller transitions
                # into reaching (enters APPROACH), the object must be visible in
                # the head camera. Otherwise the demo would teach a grasp of
                # something the policy can't see, so terminate (failure -> not
                # saved). Checked once at the transition, not every step.
                if not terminated and self._terminate_grasp_if_not_visible:
                    phase = (
                        getattr(self.env.agent, "_grasp_phase", None)
                        if self.env.agent is not None
                        else None
                    )
                    starts_reaching = (
                        phase == PHASE_APPROACH and self._prev_grasp_phase != PHASE_APPROACH
                    )
                    if starts_reaching and not self._target_visible_in_head():
                        terminated, sim_error = True, "target not visible at grasp"
                    self._prev_grasp_phase = phase
        except mujoco.FatalError as e:
            terminated, sim_error = True, str(e)

        dt = self.env.robot.n_substeps * self.env.scene.model.opt.timestep
        self.env._sim_time += dt
        self.env._capture_frame()
        if self.env._viewer:
            self.sync_viewer()

        obs = self._build_obs()
        tgt = self.env.target
        dist = float(
            np.linalg.norm(self.env.robot.get_xy() - tgt.position(self.env.scene.data)[:2])
        )
        reward = self.env.task.compute_reward(self.env.scene)
        if self.is_terminal(reward):
            terminated = True
        success = self.judge_success(reward)
        info = self.env.task.step_info()
        info.update(
            target_object_position=tgt.position(self.env.scene.data),
            target_object_pose=np.concatenate(
                [tgt.position(self.env.scene.data), tgt.quat(self.env.scene.data)]
            ),
            distance=dist,
            success=success,
            sim_error=sim_error,
        )
        self.env.task.attach_grasps(info)
        return obs, reward, terminated, False, info

    def consume_skip_episode(self):
        s = self.env._skip_episode
        self.env._skip_episode = False
        return s
