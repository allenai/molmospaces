"""G1TaskSampler: the task-sampler side of the port (G1CPUMujocoEnv is the
substrate, G1Task the rollout). Scene selection and loading, target selection,
goal/spawn/placement sampling, and texture/lighting/camera/height randomization.
"""

import glob
import json as _json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np

from fetchman.configs.g1_port_configs import G1ExpConfig
from fetchman.env_g1ms import G1CPUMujocoEnv
from fetchman.tasks.open_g1ms import OpenTask
from fetchman.tasks.pick_g1ms import PickTask
from molmo_spaces.env.arena import scene_spec_ops
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.g1 import PREFIX
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler, MetadataAdder
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

log = logging.getLogger(__name__)


def build_thor_texture_pools():
    """Scene-texture files per category (filesystem only, no RNG). Gold's
    precedence: the curated fetchman pack under textures/fetchman/, else THOR's
    material database, which yields different pools and so different renders.
    """
    from molmo_spaces.env.arena.randomization.texture import SCENE_TEXTURE_CATEGORIES

    canonical = set(SCENE_TEXTURE_CATEGORIES.values())
    local_root = ASSETS_DIR / "textures" / "fetchman"
    local_pools = {}
    if local_root.is_dir():
        for cat in sorted(canonical):
            files = sorted(str(p) for p in (local_root / cat).glob("*.png"))
            if files:
                local_pools[cat] = files
    if local_pools:
        return local_pools

    log.warning(
        f"[env] JORDI-TODO: the fetchman scene-texture pack is missing from {local_root}, "
        f"falling back to pools derived from THOR's material-database.json. Renders will "
        f"NOT match gold (different, larger, differently categorized pools) -- any "
        f"gold-vs-ported pixel comparison run in this state is invalid. Install the pack "
        f"with the asset manager (molmospaces_resources) into "
        f"{ASSETS_DIR}/textures/fetchman/<Category>/*.png; it is not yet a registered "
        f"source in DATA_TYPE_TO_SOURCE_TO_VERSION, so until it is, unzip the "
        f"textures.zip pack into {ASSETS_DIR}."
    )

    db_path = ASSETS_DIR / "objects" / "thor" / "material-database.json"
    mt_path = ASSETS_DIR / "objects" / "thor" / "material_to_textures.json"
    if not db_path.exists() or not mt_path.exists():
        raise FileNotFoundError(
            f"JORDI-TODO: no scene textures at all -- the fetchman pack is missing from "
            f"{local_root} (see the warning above for how to install it) and so is THOR's "
            f"{db_path.name}/{mt_path.name} fallback pair under {db_path.parent}. Install "
            f"the objects/thor source with the asset manager, or turn off "
            f"randomize_textures."
        )
    with open(db_path) as f:
        cat_db = _json.load(f)
    with open(mt_path) as f:
        mat_tex = _json.load(f)
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
    db_pools = {c: sorted(v) for c, v in pools.items() if v}
    if not db_pools:
        raise FileNotFoundError(
            f"JORDI-TODO: no scene textures at all -- the fetchman pack is missing from "
            f"{local_root} (see the warning above for how to install it) and THOR's "
            f"material database under {db_path.parent} yielded no usable PNGs for any of "
            f"{', '.join(sorted(canonical))}."
        )
    log.info(
        "[env] THOR-db texture pools: "
        + ", ".join(f"{c}={len(v)}" for c, v in sorted(db_pools.items()))
    )
    return db_pools


def resolve_texture_pools(randomize_textures, scene_textures_glob):
    """Returns (texture_pools, randomize_textures). `scene_textures_glob` is
    kept for signature parity with gold and no longer consulted."""
    if not randomize_textures:
        return {}, False
    return build_thor_texture_pools(), True


def _resolve_scene_paths(pattern):
    cand = Path(pattern)
    repo_root = ASSETS_DIR.parent.parent
    for base in (None, repo_root, ASSETS_DIR):
        full = cand if cand.is_absolute() else (base / cand if base is not None else None)
        if full is None or not full.is_file():
            continue
        if full.suffix == ".txt":
            out = []
            for line in full.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lp = Path(line) if Path(line).is_absolute() else ASSETS_DIR / line
                if lp.is_file() and not lp.name.endswith("_ceiling.xml"):
                    out.append(lp)
            return out
        break
    p = cand if cand.is_absolute() else ASSETS_DIR / cand
    return [Path(m) for m in sorted(f for f in glob.glob(str(p)) if not f.endswith("_ceiling.xml"))]


class G1TaskSampler(BaseMujocoTaskSampler):
    """Chooses the episodes: scene and textures, target object, robot spawn,
    and every per-reset randomization. Constructs the env and the task and
    owns `np_random`, the one Generator all three layers draw from in a fixed
    order. The env holds no reference to the task or policy.
    """

    def __init__(self, exp_config: G1ExpConfig):
        """Config reading first, the scene load last: the load is the first
        thing to consume RNG, and the texture pools it draws from must exist.

        A real ``BaseMujocoTaskSampler``, but one that does not run the base
        ``__init__``: that resolves ``house_inds`` against a dataset index this
        sampler does not use (it draws scenes from its own ``scene`` glob) and
        seeds the global ``random``/``np.random``, which nothing here reads --
        every draw in this file comes from ``self.np_random``. The base
        attributes its inherited methods do read are set below.
        """
        self.config = exp_config
        self._env = None
        self._datagen_profiler = None
        self._metadata_adder = MetadataAdder()
        self.current_seed = exp_config.seed
        self.lighting_randomizer = None
        self.texture_randomizer = None
        self.dynamics_randomizer = None
        self._dataset_index_map = None
        self._house_inds = exp_config.task_sampler_config.house_inds or []
        self._house_iterator_index = -1
        self._samples_per_current_house = 0
        self._max_tasks = getattr(exp_config.task_sampler_config, "max_tasks", None) or math.inf
        self._current_tasks_left = self._max_tasks
        self._last_loaded_house_index = None
        self.object_synset_counter = Counter()
        self.used_robot_positions = defaultdict(list)
        self._asset_failure_counts = Counter()
        self._dynamic_blacklist = set()
        self._max_asset_failures = getattr(exp_config.task_sampler_config, "max_asset_failures", 10)
        ts = exp_config.task_sampler_config

        self._active_profile = None
        self._arm_init_radius = float(ts.arm_init_radius)
        self._face_yaw_offset = float(ts.face_yaw_offset)
        self._frozen_full_state = None
        self._frozen_obj_idx = None
        self._frozen_reset_counter = 0
        self._frozen_rng_state = None
        self._goal_offset_xy_noise = float(ts.goal_offset_xy_noise)
        self._goal_offset_yaw_noise = float(ts.goal_offset_yaw_noise)
        self._grasp_spawn_radius_max = float(ts.grasp_spawn_radius_max)
        self._head_camera_distortion_noise = float(ts.head_camera_distortion_noise)
        self._head_camera_fovy_noise = float(ts.head_camera_fovy_noise)
        self._head_camera_pos_noise = float(ts.head_camera_pos_noise)
        self._head_camera_rot_noise = float(ts.head_camera_rot_noise)
        self._init_arm_at_pregrasp = bool(ts.init_arm_at_pregrasp)
        self._pregrasp_rot_noise = float(ts.pregrasp_rot_noise)
        self._pregrasp_xyz_noise = float(ts.pregrasp_xyz_noise)
        self._randomize_height = bool(ts.randomize_height)
        self._randomize_height_favored = float(ts.randomize_height_favored)
        randomize_height_max = ts.randomize_height_max
        self._randomize_height_max = (
            None if randomize_height_max is None else float(randomize_height_max)
        )
        self._randomize_height_min = float(ts.randomize_height_min)
        self._randomize_lighting = bool(ts.randomize_lighting)
        self._randomize_lighting_keep_prob = float(ts.randomize_lighting_keep_prob)
        self._randomize_placement = ts.randomize_placement
        self._randomize_robot_height = bool(ts.randomize_robot_height)
        self._randomize_robot_height_max = float(ts.randomize_robot_height_max)
        self._randomize_robot_height_min = float(ts.randomize_robot_height_min)
        self._randomize_scene = ts.randomize_scene
        self._randomize_scene_freq = max(1, int(ts.randomize_scene_freq))
        self._randomize_textures_keep_prob = float(ts.randomize_textures_keep_prob)
        self._randomize_textures_solid_color_prob = float(ts.randomize_textures_solid_color_prob)
        self._reset_counter = 0
        self._reset_precheck_grasp = bool(ts.reset_precheck_grasp)
        self._ring_num_angles = int(ts.ring_num_angles)
        self._sample_spawn_first = bool(ts.sample_spawn_first)
        self._skill_profiles = []
        skill_profiles = ts.skill_profiles
        if skill_profiles:
            for entry in skill_profiles:
                name, weight, profile = entry
                self._skill_profiles.append((str(name), float(weight), dict(profile)))
            wsum = sum(w for _, w, _ in self._skill_profiles)
            assert wsum > 0, "skill_profiles weights must sum > 0"
            self._skill_profiles = [(n, w / wsum, p) for n, w, p in self._skill_profiles]
        self._spawn_along_line = bool(ts.spawn_along_line)
        self._spawn_at_grasp = bool(ts.spawn_at_grasp)
        self._spawn_radius_max = ts.spawn_radius_max
        self._spawn_radius_min = ts.spawn_radius_min
        self._spawn_reachability_check = bool(ts.spawn_reachability_check)
        self._spawn_visibility_check = ts.spawn_visibility_check
        self._start_at_pregrasp_joint_noise = float(ts.start_at_pregrasp_joint_noise)
        self._start_at_pregrasp_xy_noise = float(ts.start_at_pregrasp_xy_noise)
        self._start_at_pregrasp_yaw_noise = float(ts.start_at_pregrasp_yaw_noise)
        self._walk_dist_max = float(ts.walk_dist_max)
        self._walk_dist_min = float(ts.walk_dist_min)
        self._wrist_camera_fovy_noise = float(ts.wrist_camera_fovy_noise)
        self._wrist_camera_pos_noise = float(ts.wrist_camera_pos_noise)
        self._wrist_camera_rot_noise = float(ts.wrist_camera_rot_noise)

        # ---- scene selection, textures and the shared RNG ----
        # One Generator for the whole stack: the env, the task and the policy
        # all hold this same object, so there is a single draw order.
        self.np_random = np.random.default_rng(exp_config.seed)
        self._scene_paths = _resolve_scene_paths(ts.scene)
        self._grasp_spawn_radius_min = float(ts.grasp_spawn_radius_min)
        self._deterministic_scene_textures = bool(ts.deterministic_scene_textures)
        # max_textures is interpreted per category (5 walls, 5 floors, etc.).
        # Discovery (which texture files exist) is resolve_texture_pools above;
        # the RNG-dependent per-reset selection from those pools is
        # _sample_scene_textures below.
        self._texture_pools, self._randomize_textures = resolve_texture_pools(
            bool(ts.randomize_textures), ts.scene_textures_glob
        )
        self._max_textures = int(ts.max_textures) if ts.max_textures else 0

        # ---- substrate, task, first scene ----
        self._env = G1CPUMujocoEnv(
            exp_config, np_random=self.np_random, launch_viewer=exp_config.use_passive_viewer
        )
        # task_cls mirrors molmo_spaces' own PickTaskSampler._task_cls hook,
        # as a config field rather than a method; task_type picks the default.
        task_cls = exp_config.task_config.task_cls
        if task_cls is None:
            task_type = str(exp_config.task_config.task_type).lower()
            task_cls = OpenTask if task_type in ("open", "close") else PickTask
        self.task = task_cls(self._env, exp_config)
        # Read by the policy's grasp-retry off its own `_task` reference.
        self.task._grasp_spawn_radius_min = self._grasp_spawn_radius_min

        # Retry init scene load: some scenes in a curated list can still lack
        # pickable target objects (missing grasp files, naming mismatch, etc)
        # and raise ValueError. Without this loop, a single worker drawing a
        # bad scene kills the whole vec_env.
        _init_loaded = False
        for _init_try in range(50):
            try:
                self._load_scene(self._scene_paths[self.np_random.integers(len(self._scene_paths))])
                _init_loaded = True
                break
            except (HouseInvalidForTask, ValueError, IndexError, OSError) as e:
                print(f"[env] init: skipping scene (load failed: {type(e).__name__}: {e})")
            except Exception as e:
                print(f"[env] init: skipping scene (load error: {type(e).__name__}: {e})")
        if not _init_loaded:
            raise RuntimeError(
                f"Could not load any valid initial scene from {len(self._scene_paths)} candidates after 50 tries"
            )

    def _sample_scene_textures(self):
        if not self._randomize_textures or not self._texture_pools:
            return {}
        # GRPO needs identical Scene contents across group-mate workers and
        # identical np_random consumption regardless of whether load_scene
        # short-circuits. Both require: pool subsampling uses a deterministic
        # local RNG, not self.np_random.
        if self._deterministic_scene_textures:
            rng = np.random.RandomState(12345)
            choice = rng.choice
        else:
            choice = self.np_random.choice
        out: dict[str, list[str]] = {}
        # Iterate in SORTED category order so deterministic-RNG consumption is
        # identical across workers (the dict's natural order comes from a set
        # iteration upstream, which is per-process random via PYTHONHASHSEED).
        for cat in sorted(self._texture_pools):
            pool = self._texture_pools[cat]
            if self._max_textures and self._max_textures < len(pool):
                idx = choice(len(pool), size=self._max_textures, replace=False)
                out[cat] = [pool[int(i)] for i in idx]
            else:
                out[cat] = list(pool)
        return out

    def _load_scene(self, xml_path, texture_override=None):
        """Load a scene and check it has valid targets (else HouseInvalidForTask,
        with the env's loaded-scene cache rolled back). `texture_override`
        replays an exact texture assignment without consuming RNG."""
        self.env.load_scene(
            xml_path,
            texture_override if texture_override is not None else self._sample_scene_textures(),
            force=texture_override is not None,
        )
        try:
            self.task.set_objects(self.env.scene)
        except Exception as e:
            self.env._current_scene_path = None
            raise HouseInvalidForTask(
                house_info=str(xml_path), reason="no valid task objects", error=e
            ) from e
        # The scene was rebuilt, so the policy's name->id caches are stale.
        if self.task.agent is not None:
            self.task.agent.setup(self.env.scene.model, self.env.scene.data)

    @property
    def env(self):
        """Read-only: the G1CPUMujocoEnv owned and constructed by this sampler
        (mirrors BaseMujocoTaskSampler.env's own read-only property)."""
        return self._env

    def reset(self) -> None:
        """Start episode sampling over, as BaseMujocoTaskSampler.reset does.
        This sampler is unbounded (no max_tasks), so there is nothing to
        replenish -- only the per-episode counter that gates randomize_scene_freq.
        """
        self._reset_counter = 0

    def close(self) -> None:
        """Clean up this sampler and its env, as BaseMujocoTaskSampler.close does."""
        import gc

        if getattr(self, "_env", None) is not None:
            self._env.close()
            self._env = None
        gc.collect()

    def seed_task_sampling(self, seed) -> None:
        """Reseed episode sampling, matching BaseMujocoTaskSampler.seed_task_sampling
        -- except this reseeds only this sampler's own Generator (shared with the
        env, task and policy), not the process-global RNGs, so it cannot disturb a
        caller's own random state."""
        self.np_random = np.random.default_rng(seed)
        self.env.np_random = self.np_random

    def set_agent(self, agent):
        """Kept as the sampler-level name callers have always used; the task is
        what owns the policy now (see G1Task.register_policy, which also hands
        the policy its `set_task` reference)."""
        agent.setup(self.env.scene.model, self.env.scene.data)
        self.task.register_policy(agent)

    def place_base(self, xy, yaw):
        self.env.robot.place(xy, yaw)

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
            target_pos[:2], self._grasp_spawn_radius_min, self._grasp_spawn_radius_max
        ):
            return None, None
        saved_qpos = self.env.scene.data.qpos.copy()
        saved_ctrl = self.env.scene.data.ctrl.copy()
        # Optional task-driven angle preference (e.g. slider drawers want the robot
        # lined up with the slide axis). Returns list of 2D unit vectors; empty = no
        # preference (PickTask / hinge joints).
        pref_dirs = []
        if hasattr(self.task, "preferred_goal_directions"):
            pref_dirs = list(self.task.preferred_goal_directions(self.env.scene))
        cone = 0.5  # rad, ~28° half-angle around the preferred direction
        try:
            for _ in range(attempts):
                if pref_dirs:
                    direction = pref_dirs[int(self.np_random.integers(len(pref_dirs)))]
                    base_ang = float(np.arctan2(direction[1], direction[0]))
                    ang = base_ang + float(self.np_random.uniform(-cone, cone))
                    r = float(
                        self.np_random.uniform(
                            self._grasp_spawn_radius_min, self._grasp_spawn_radius_max
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
                        radius_min=self._grasp_spawn_radius_min,
                        radius_max=self._grasp_spawn_radius_max,
                        np_random=self.np_random,
                    )
                if xy is None:
                    continue
                xy = np.asarray(xy, dtype=np.float64)
                yaw = float(np.arctan2(target_pos[1] - xy[1], target_pos[0] - xy[0]))
                if obj is None:
                    return xy, yaw
                self.env.robot.set_pose(xy, yaw)
                self.env.robot.set_defaults()
                if self.task.agent is not None and hasattr(self.task.agent, "_set_groot_defaults"):
                    self.task.agent._set_groot_defaults()
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
        if not self.task.target:
            return None, None
        return self._sample_goal_pose(
            self.task.target.position(self.env.scene.data), self.task.target
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
                    np_random=self.np_random,
                )
                if spawn_xy is None:
                    continue
                spawn_xy = np.asarray(spawn_xy, dtype=np.float64)
                bearing = float(
                    np.arctan2(target_pos[1] - spawn_xy[1], target_pos[0] - spawn_xy[0])
                )
                spawn_yaw = bearing + self.np_random.uniform(-0.5, 0.5)
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
            self.np_random.uniform(self._grasp_spawn_radius_min, self._grasp_spawn_radius_max)
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
        offset = float(self.np_random.uniform(min_offset, max_offset))
        spawn_xy = gxy + offset * direction
        bearing = float(np.arctan2(target_pos[1] - spawn_xy[1], target_pos[0] - spawn_xy[0]))
        spawn_yaw = bearing + float(self.np_random.uniform(-0.8, 0.8))
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
        if self.task.agent is not None and hasattr(self.task.agent, "_set_groot_defaults"):
            self.task.agent._set_groot_defaults()
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
        if self.task.agent is None:
            return None
        info_preview = self.task.make_info(self.env.scene, self.np_random)
        info_preview["goal_xy"] = goal_xy
        info_preview["goal_yaw"] = goal_yaw
        saved_qpos = self.env.scene.data.qpos.copy()
        saved_ctrl = self.env.scene.data.ctrl.copy()
        try:
            self.env.robot.set_pose(goal_xy, goal_yaw)
            if hasattr(self.task.agent, "_set_groot_defaults"):
                self.task.agent._set_groot_defaults()
            mujoco.mj_forward(self.env.scene.model, self.env.scene.data)
            self.task.agent._grasp_planner.plan(info_preview)
            pregrasp_joints = getattr(self.task.agent._grasp_planner, "_pregrasp_joints", None)
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
            self.np_random.uniform(-self._pregrasp_xyz_noise, self._pregrasp_xyz_noise, size=3)
            if self._pregrasp_xyz_noise > 0
            else np.zeros(3)
        )
        if self._pregrasp_rot_noise > 0:
            from scipy.spatial.transform import Rotation as _R

            axis = self.np_random.normal(size=3)
            axis /= max(np.linalg.norm(axis), 1e-9)
            angle = self.np_random.uniform(-self._pregrasp_rot_noise, self._pregrasp_rot_noise)
            rot = _R.from_rotvec(axis * angle).as_matrix()
        else:
            rot = np.eye(3)
        return (xyz, rot)

    def set_datagen_profiler(self, profiler) -> None:
        """Accepted for interface compatibility with molmo_spaces' pipeline;
        this sampler has no sub-timing hooks to attach a profiler to."""
        self._datagen_profiler = profiler

    def sample_task(self, *, seed=None, freeze=False, house_index=None, force_advance_scene=False):
        """Sample one episode and return the (single, reused) task; `task.reset()`
        yields its first observation. Retries `_sample_task()` up to 12 times
        and snapshots the sim state for `freeze=True` replays.

        `house_index` addresses this sampler's own scene list -- the glob or
        .txt in `task_sampler_config.scene` -- so molmo_spaces' pipeline, which
        iterates houses by index, can drive it. Omit it and the sampler advances
        scenes itself, as gold does. `force_advance_scene` is what
        `randomize_scene` already controls.
        """
        if house_index is not None:
            if not self._scene_paths:
                raise ValueError("no scenes matched task_sampler_config.scene")
            self._load_scene(self._scene_paths[house_index % len(self._scene_paths)])
        # freeze=True replays the last non-frozen reset bit-identically (GRPO
        # groups sharing an initial state): restore the full sim + task
        # snapshot and skip the sample_task recompute and its RNG draws.
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
            self.np_random.bit_generator.state = fs["np_random"]
            self._reset_counter = fs["reset_counter"]
            if hasattr(self.task, "_obj_idx") and fs["obj_idx"] is not None:
                self.task._obj_idx = fs["obj_idx"]
            self.task.target = fs["target"]
            self.task._target_z0 = fs["target_z0"]
            self.task._target_body_set = fs["target_body_set"]
            self.env._sim_time = 0.0
            self.task._action_noise_step = 0
            self.env._gripper_precrash = False
            self.task._prev_grasp_phase = None
            self.task._action_noise_offset.fill(0.0)
            mujoco.mj_forward(m, d)
            self.env._skip_episode = False
            self.env._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
            self.task._episode_info = dict(fs["info"])
            return self.task
        # Non-frozen path: snapshot RNG bits for old code paths that still read
        # them, then run the full sample_task compute.
        self._frozen_rng_state = self.np_random.bit_generator.state
        self._frozen_reset_counter = self._reset_counter
        self._frozen_obj_idx = getattr(self.task, "_obj_idx", None)
        if seed is not None:
            self.seed_task_sampling(seed)
        self.env._skip_episode = False
        self._reset_counter += 1
        self.env._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
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
                    "np_random": self.np_random.bit_generator.state,
                    "reset_counter": self._reset_counter,
                    "obj_idx": getattr(self.task, "_obj_idx", None),
                    "target": self.task.target,
                    "target_z0": float(getattr(self.task, "_target_z0", 0.0)),
                    "target_body_set": set(getattr(self.task, "_target_body_set", set())),
                    "info": dict(r),
                }
                # The task carries its episode's info; task.reset() returns it
                # together with the first observation.
                self.task._episode_info = dict(r)
                return self.task
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
            "target_name": self.task.target.name,
            "target_z0": float(self.task._target_z0),
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
        if self.env.fisheye is not None:
            st["fisheye_K"] = self.env.fisheye.K.tolist()
            st["fisheye_D"] = self.env.fisheye.D.tolist()
        return st

    def restore_reset_state(self, st):
        """Exact replay of an export_reset_state() dict. Returns the task, like
        sample_task(); task.reset() then yields its first observation."""
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
            self.task._object_regex = st["object_regex"]
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
            self._load_scene(st["scene"], texture_override=textures)
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
            self.env.ensure_fisheye().set_intrinsics(
                np.asarray(st["fisheye_K"]), np.asarray(st["fisheye_D"])
            )
        target = next((o for o in self.env.scene.pickable if o.name == st["target_name"]), None)
        if target is None:
            raise RuntimeError(f"restore: target {st['target_name']!r} not in scene {st['scene']}")
        self.task.target = target
        self.task._target_z0 = float(st["target_z0"])
        self.task._target_body_set = self.env.scene.get_body_descendants(target.body_id)
        self.task._target_grasps = self.task._load_grasps(getattr(target, "asset_id", ""))
        self.env._sim_time = 0.0
        self.task._action_noise_step = 0
        self.env._gripper_precrash = False
        self.task._prev_grasp_phase = None
        self.task._action_noise_offset.fill(0.0)
        # Exact replay is noise-free by definition: without this, a benchmark
        # episode inherits whatever action_noise_std the previous (training)
        # episode's skill profile left behind. Training is unaffected — every
        # sample_task() re-applies the profile's own value.
        self.task._action_noise_std = 0.0
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
        info = self.task.make_info(self.env.scene, self.np_random)
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
        self.task._episode_info = dict(info)
        return self.task

    def _randomize_lights(self):
        m = self.env.scene.model
        rng = self.np_random
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
        rng = self.np_random
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
        cam = self.env.camera("wrist_image")
        mjcf = cam.mjcf if cam is not None else None
        self._perturb_camera(
            mjcf.camera_id if mjcf is not None else -1,
            mjcf.base_pos if mjcf is not None else None,
            mjcf.base_quat if mjcf is not None else None,
            mjcf.base_fovy if mjcf is not None else None,
            self._wrist_camera_pos_noise,
            self._wrist_camera_rot_noise,
            self._wrist_camera_fovy_noise,
        )

    def _randomize_head_camera(self):
        # Head = fisheye. pos/rot perturb all 5 tile cameras (+ head_pov) with
        # the SAME delta so they stay co-located. fovy scales the fisheye K
        # (fx, fy) — this triggers a LUT rebuild via set_intrinsics, ~50ms.
        head = self.env.camera("head_image")
        if head is None or head.mjcf is None or not head.mjcf.is_fisheye:
            return
        pn = self._head_camera_pos_noise
        rn = self._head_camera_rot_noise
        fn = self._head_camera_fovy_noise
        dn = self._head_camera_distortion_noise
        if pn <= 0 and rn <= 0 and fn <= 0 and dn <= 0:
            return
        m = self.env.scene.model
        rng = self.np_random
        dp = rng.uniform(-pn, pn, 3) if pn > 0 else np.zeros(3)
        if rn > 0:
            axis = rng.uniform(-1, 1, 3)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            ang = float(rng.uniform(-rn, rn))
            half = 0.5 * ang
            dq = np.array([float(np.cos(half)), *(axis * float(np.sin(half)))])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        # head_pov itself plus its five fisheye tiles, all co-located.
        for cid, pos0, quat0 in zip(
            head.mjcf.rig_camera_ids, head.mjcf.rig_base_pos, head.mjcf.rig_base_quat
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
        if (fn > 0 or dn > 0) and self.env.fisheye is not None:
            # Perturb around the *calibrated* lens from config, not around the
            # renderer's current K/D -- those already carry the previous
            # episode's noise, which would random-walk across resets.
            cfg = head.mjcf.config
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
            self.env.fisheye.set_intrinsics(K=K, D=D)

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
        new_top = float(self.np_random.triangular(self._randomize_height_min, mode, upper))
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
                scene_spec_ops.grasp_probe_body_name(0),
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
                scene_spec_ops.grasp_probe_body_name(0),
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
        idx = int(self.np_random.choice(len(names), p=weights))
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
            self.task._action_noise_std = float(profile["action_noise_std"])
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
        """(Re)load the MJCF when no scene is picked yet or randomize_scene is
        due -- PickTaskSampler.init_scene's role. False if loading failed."""
        _need_objects = not getattr(self.task, "objects", None)
        if not (
            len(self._scene_paths) > 1
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
            env._load_scene(self._scene_paths[self.np_random.integers(len(self._scene_paths))])
        except (HouseInvalidForTask, ValueError, IndexError, OSError) as e:
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
        """Per-reset texture/lighting/camera randomization and robot defaults
        (PickTaskSampler.randomize_scene's role). Returns the randomized arm
        pose, if any, already applied."""
        env.scene.reset()
        # Must run after scene.reset() — reset restores geom_matid to defaults.
        if (
            self._randomize_textures
            and env.scene.scene_matids
            and env.scene.scene_geom_ids
            and self.np_random.random() >= self._randomize_textures_keep_prob
        ):
            color_mids = env.scene.scene_color_matids
            if color_mids:
                rand_rgba = self.np_random.uniform(0.0, 1.0, size=(len(color_mids), 4)).astype(
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
                tex_picks = self.np_random.choice(tex_mids, size=len(gids))
                use_solid = (self.np_random.random(size=len(gids)) < scp) if color_mids else None
                solid_picks = (
                    self.np_random.choice(color_mids, size=len(gids)) if color_mids else None
                )
                for k, gid in enumerate(gids):
                    if use_solid is not None and use_solid[k]:
                        env.scene.model.geom_matid[gid] = int(solid_picks[k])
                    else:
                        env.scene.model.geom_matid[gid] = int(tex_picks[k])
        if (
            self._randomize_lighting
            and env.scene.model.nlight > 0
            and self.np_random.random() >= self._randomize_lighting_keep_prob
        ):
            self._randomize_lights()
        self._randomize_wrist_camera()
        self._randomize_head_camera()
        robot_view.zero_velocities()
        robot_view.set_defaults()

        sampled_upper = (
            robot_view.sample_upper_pose(self.np_random, self._arm_init_radius)
            if self._arm_init_radius > 0
            else None
        )
        if sampled_upper is not None:
            robot_view.apply_upper_pose(sampled_upper)
        return sampled_upper

    def _setup_scene_and_robot(self, env):
        """init_scene() then randomize_scene(): the first phase of
        _sample_task(). Returns (True, sampled_upper) or (False, None) if the
        scene failed to load."""
        env._sim_time = 0.0
        self.task._action_noise_step = 0
        env._gripper_precrash = False
        self.task._action_noise_offset.fill(0.0)
        self._sample_skill_profile()
        if not self.init_scene(env):
            return False, None
        sampled_upper = self.randomize_scene(env, env.robot)
        return True, sampled_upper

    def _sample_task(self, env):
        """One sampling attempt: scene, target and placement, to a valid info
        dict or None (the caller retries). molmo_spaces' `_sample_task(env)` hook."""
        ok, sampled_upper = self._setup_scene_and_robot(env)
        if not ok:
            return None
        return self._select_target_and_place(env, sampled_upper)

    def _select_target_and_place(self, env, sampled_upper):
        """Target selection, goal/spawn sampling, realign/pregrasp handling and
        robot placement. One method on purpose: many retry early-returns and
        variables threaded across phases (sampled_upper is reassigned when
        init_arm_at_pregrasp fires), so it is gold's block kept whole.
        """
        obj = self.task.select_target(self.np_random)
        # Jitter object xy (per-task) before reading pos so goal sampling sees the perturbed location.
        self.task.perturb_objects(env.scene, self.np_random)
        # Must run before goal sampling so the goal lands at the new z.
        self._randomize_target_support_height(obj)
        # Run init_target_tracking BEFORE goal sampling so the close task's drawer
        # (which gets opened by init_target_tracking) is at its actual starting
        # pose when we sample around it. Open task / pick task are unaffected
        # since their init_target_tracking doesn't move the target.
        self.task.init_target_tracking(env.scene)
        # Use the task's grasp frame (the *moving* body, e.g. drawer) for goal sampling
        # when available — otherwise fall back to the object root.
        if hasattr(self.task, "grasp_frame_pose"):
            tgt = self.task.grasp_frame_pose(env.scene)[0]
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
                self.task.target = None
                return None
            _spawn_first_xy, _spawn_first_yaw = sxy, syaw
        else:
            goal_xy, goal_yaw = self._sample_goal_pose(tgt, obj, sampled_upper=sampled_upper)
            if goal_xy is None:
                self.task.target = None
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
        if self._init_arm_at_pregrasp and self.task.agent is not None:
            pregrasp_upper = self._compute_pregrasp_upper(goal_xy, goal_yaw)
            if pregrasp_upper is not None:
                sampled_upper = pregrasp_upper
                _used_pregrasp_upper = True
        # Misalign spawn (when spawn_at_grasp) so controller must realign — base-motion-during-grasp demos.
        if self._goal_offset_xy_noise > 0:
            goal_xy = goal_xy + self.np_random.uniform(
                -self._goal_offset_xy_noise, self._goal_offset_xy_noise, size=2
            )
        if self._goal_offset_yaw_noise > 0:
            goal_yaw = float(
                goal_yaw
                + self.np_random.uniform(-self._goal_offset_yaw_noise, self._goal_offset_yaw_noise)
            )

        if self.task.agent is not None and self._reset_precheck_grasp:
            info_preview = self.task.make_info(env.scene, self.np_random)
            info_preview["goal_xy"] = goal_xy
            info_preview["goal_yaw"] = goal_yaw
            if not self.task.agent.precheck_grasp(info_preview):
                self.task.target = None
                return None

        realign_info = None
        spp_xy_noise = self._start_at_pregrasp_xy_noise
        spp_yaw_noise = self._start_at_pregrasp_yaw_noise
        if (spp_xy_noise > 0 or spp_yaw_noise > 0) and self.task.agent is not None:
            pregrasp_joints = getattr(self.task.agent._grasp_planner, "_pregrasp_joints", None)
            if pregrasp_joints:
                # WBC walks forward well, sideways/yaw poorly — only spawn behind goal in body-x.
                axis = "x"
                offset = -float(self.np_random.uniform(spp_xy_noise * 0.5, spp_xy_noise))
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
                        val = pj[name] + float(self.np_random.uniform(-jn, jn))
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
                    np_random=self.np_random,
                )
                if cand_xy is None:
                    continue
                cand_xy = np.asarray(cand_xy, dtype=np.float64)
                if self._spawn_visibility_check:
                    bearing = float(np.arctan2(tgt[1] - cand_xy[1], tgt[0] - cand_xy[0]))
                    cand_yaw = bearing + self.np_random.uniform(-0.5, 0.5)
                else:
                    cand_yaw = self.np_random.uniform(-np.pi, np.pi)
                env.robot.set_pose(cand_xy, cand_yaw)
                env.robot.set_defaults()
                if self.task.agent is not None and hasattr(self.task.agent, "_set_groot_defaults"):
                    self.task.agent._set_groot_defaults()
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

        info = self.task.make_info(env.scene, self.np_random)
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
                self.np_random.uniform(
                    self._randomize_robot_height_min, self._randomize_robot_height_max
                )
            )
        return info


def make_task_sampler(exp_config: G1ExpConfig) -> G1TaskSampler:
    """Build the sampler, env and task from an experiment config. Does not
    sample: the reachability precheck needs the policy, attached by the caller."""
    return G1TaskSampler(exp_config)
