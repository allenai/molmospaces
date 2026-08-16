import contextlib
import gc
import glob
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from molmo_spaces.configs.camera_configs import G1CameraSystem
from molmo_spaces.env.env import BaseMujocoEnv, CPUMujocoEnv
from molmo_spaces.env.object_manager import ObjectManager
from molmo_spaces.g1_molmo_port.components import Scene
from molmo_spaces.g1_molmo_port.tasks.open import OpenTask
from molmo_spaces.g1_molmo_port.tasks.open import get_config as get_open_task_config
from molmo_spaces.g1_molmo_port.tasks.pick_g1ms import PickTask
from molmo_spaces.g1_molmo_port.tasks.pick_g1ms import get_config as get_task_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.g1 import JOINT_NAMES, PREFIX, XML_PATH, G1Robot

ROBOT_PREFIX = PREFIX


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


def _mj_id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, f"{PREFIX}{name}")


class MoveGroup:
    """Minimal MoveGroup abstraction mirroring molmo_spaces/robots/
    robot_views/abstract.py's shape: a named slice of the flat 15-dim
    high-level action array (cmd[0:3], height[3], waist[4:7], right_arm
    [7:14], right_gripper[14]), self-contained here (same shape only).

    The actual WBC/PD control law that turns these slices into qpos/ctrl
    targets lives in components/controller.py's G1Controller.execute_action
    and agents/policy.py -- both shared, unforked dependencies (only env.py
    itself was forked into env_g1ms.py, see Stage 0), so porting that control
    law itself is out of scope for this single-file env-side port. What DOES
    live in this file is: recording the pre-noise base command for next-
    step's `last_base_vel_cmd` obs, and injecting action noise onto
    waist+arm -- both currently bare magic-number slicing in step(). This
    class only names those slices; it doesn't change what happens to them.
    """

    def __init__(self, name, action_slice):
        self.name = name
        self.action_slice = action_slice

    def action_view(self, action):
        return action[self.action_slice]


BASE_MOVE_GROUP = MoveGroup("base", slice(0, 3))
HEIGHT_MOVE_GROUP = MoveGroup("height", slice(3, 4))
WAIST_MOVE_GROUP = MoveGroup("waist", slice(4, 7))
ARM_MOVE_GROUP = MoveGroup("right_arm", slice(7, 14))
GRIPPER_MOVE_GROUP = MoveGroup("right_gripper", slice(14, 15))
# env.py's own action-noise injection treats waist+arm as one contiguous
# block (indices 4:14) rather than injecting into each separately.
NOISE_MOVE_GROUP = MoveGroup("waist_arm", slice(4, 14))


class CameraManager:
    """Groups G1Env's camera-related state -- previously scattered across
    ~10 separate G1Env attributes (_cam_size, _cam_ids, _renderer,
    _renderers, _fisheye, _wrist_cam_pos0/_quat0/_fovy0,
    _head_cam_ids_all/_pos0_all/_quat0_all) -- into one cohesive object,
    named/owned (env.camera_manager) to match molmo_spaces' own
    molmo_spaces/env/camera_manager.py::CameraManager attribute. Unlike that
    reference (a dynamic registry of Camera objects with per-step pose
    tracking for robot-mounted/randomized cameras), G1's cameras are baked
    into the MJCF and only reset at scene load, so this only needs to be a
    plain state container -- same shape only, not the full CameraManager
    contract. Pure attribute-organization refactor, not an ownership change:
    this is still genuinely physics/rendering substrate state, so G1Env owns
    one instance (self.camera_manager), constructed once and populated by
    _load_scene on every (re)load same as before.
    """

    # Which obs-dict camera keys map to which MJCF camera name.
    NAMES = {"head_image": "head_pov", "wrist_image": "right_wrist_camera"}

    def __init__(self, size):
        self.size = tuple(size)
        self.renderer = None
        self.renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.ids = {}
        self.fisheye = None
        # Every parameter of the head fisheye -- tile camera names, tile size and
        # FOV, blend exponent, lens calibration -- lives on this config rather
        # than being restated at the FisheyeRenderer call sites. See
        # configs/camera_configs.py::FisheyeMjcfCameraConfig.
        self.fisheye_config = G1CameraSystem().get_camera_by_name("head_camera")
        self.wrist_cam_pos0 = None
        self.wrist_cam_quat0 = None
        self.wrist_cam_fovy0 = None
        self.head_cam_ids_all = []
        self.head_cam_pos0_all = []
        self.head_cam_quat0_all = []


class G1Env(gym.Env, CPUMujocoEnv):
    """Inherits molmo_spaces' own BaseMujocoEnv/CPUMujocoEnv ABC (env/env.py)
    for interface compatibility -- current_data/current_model/current_robot/
    current_model_path/is_loaded()/rgb_frame etc. all become available via
    the real BaseMujocoEnv property chain once mj_datas/n_batch/robots are
    overridden below (see those properties' own docstrings).

    Deliberately does NOT call CPUMujocoEnv.__init__ (that constructs N
    batched MjData via its own mujoco.mj_forward + sim_settle_timesteps
    mj_step loop, a real GPU renderer via MjOpenGLRenderer/MjFilamentRenderer,
    a ThreadPoolExecutor, and one ObjectManager per batch item) -- none of
    that matches gold's own settle/reset sequencing, which
    G1TaskSampler's reset()/sample_task() already reproduces bit-exactly
    (verified via the g1_molmo_comparison harness). Running CPUMujocoEnv's
    own settle loop on top would inject extra, different physics steps and
    RNG consumption gold's own reference never does. Calls
    BaseMujocoEnv.__init__ directly instead, for its safe, side-effect-free
    placeholder state (self.config, self._mj_model, rendering placeholders)
    -- G1Env's own already-verified `_load_scene` stays the only thing that
    actually builds/reloads the scene.

    `object_managers` stays an empty list (BaseMujocoEnv's own type-hinted
    attribute, not populated) and CPUMujocoEnv's own higher-level methods
    (get_thormap, place_robot_near, check_visibility,
    get_segmentation_mask_of_object, ...) are NOT safely callable on a G1Env
    instance -- all of them assume the real renderer/object_managers/batch
    construction this class deliberately skips. That's real, deferred
    Scene/Object -> real ObjectManager work (see components/scene.py's own
    docstring), not part of this inheritance step.
    """

    # Matches gold's own env.py class attribute exactly (same keys/values as
    # CameraManager.NAMES above) -- LeRobotRecorder (dataset/lerobot_recorder.py)
    # reads `env.cameras` directly, ported verbatim from gold's own usage.
    cameras = CameraManager.NAMES

    def __init__(
        self,
        scene,
        objects=".*",
        seed=None,
        randomize_object=False,
        grasp_spawn_radius_min=0.25,
        randomize_textures=False,
        scene_textures_glob="textures/randomization/*.png",
        max_textures=5,
        deterministic_scene_textures=False,
        task_type="pick",  # "pick" (default) | "open" | "close"
        articulated_regex=None,  # only used when task_type in {"open","close"}
        object_noise=0.0,  # pick task: gaussian xy jitter on object spawn (m)
        open_success_threshold=0.5,
        open_terminate_threshold=1.0,
        open_init_percent=0.9,  # only for task_type="close"
        open_require_joint_grasp=False,
        launch_viewer=False,
        camera_size=(224, 224),
        **_unused_task_kwargs,
    ):
        """`**_unused_task_kwargs` swallows the config keys G1TaskSampler's
        constructor needs (action_noise_std, spawn_radius_min,
        randomize_height, terminate_*, ...) -- make_env() passes the full
        config dict to both constructors; each only names the parameters it
        actually stores. See G1TaskSampler.__init__'s matching docstring.
        """
        # Explicit by-name calls, not super().__init__() -- gym.Env and
        # BaseMujocoEnv are unrelated hierarchies (neither cooperatively
        # chains via super() into the other), so a single super().__init__()
        # call would only ever reach gym.Env's. BaseMujocoEnv.__init__ is the
        # safe, side-effect-free one (self.config, self._mj_model, rendering
        # placeholders) -- see the class docstring for why CPUMujocoEnv.
        # __init__ itself is deliberately never called.
        gym.Env.__init__(self)
        BaseMujocoEnv.__init__(self, exp_config=None, mj_model=None)
        # ObjectManager.is_excluded reads env.config.robot_config.robot_namespace
        # as a substring check against body names -- give it the same shape
        # CPUMujocoEnv's real exp_config would have, without pulling in the rest
        # of MlSpacesExpConfig.
        self.config = SimpleNamespace(robot_config=SimpleNamespace(robot_namespace=ROBOT_PREFIX))
        self.object_managers = []
        self._viewer = None
        self._launch_viewer = launch_viewer
        self._scene_paths = _resolve_scene_paths(scene)
        self._object_regex = objects or ".*"
        self.np_random = np.random.default_rng(seed)
        self._current_scene_path = None
        self._grasp_spawn_radius_min = float(grasp_spawn_radius_min)
        self._deterministic_scene_textures = bool(deterministic_scene_textures)
        # max_textures is interpreted per category (5 walls, 5 floors, etc.).
        # Discovery (which texture files exist) + the disable-if-empty
        # fallback now live in tasks/pick_task_sampler_g1ms.py's
        # resolve_texture_pools -- "texture randomization stuff", moved out
        # of G1Env per that request; deferred import to avoid a circular
        # module-level import (pick_task_sampler_g1ms.py imports several
        # names from this module already). _texture_pools/_randomize_textures
        # still live here because _sample_scene_textures (the RNG-dependent
        # per-reset *selection* from these pools) needs self.np_random, which
        # stays on G1Env for construction-order reasons.
        from molmo_spaces.g1_molmo_port.tasks.pick_task_sampler_g1ms import resolve_texture_pools

        self._texture_pools, self._randomize_textures = resolve_texture_pools(
            bool(randomize_textures), scene_textures_glob
        )
        self._max_textures = int(max_textures) if max_textures else 0
        self._scene_texture_paths: dict[str, list[str]] = {}
        self._skip_episode = False
        self._sim_time = 0.0
        # Read directly by LastBaseVelCmdSensor (which receives the raw
        # G1Env, not G1TaskSampler) -- stays here rather than moving to
        # G1TaskSampler for the same reason target/time do.
        self._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
        self.robot = None
        self.agent = None
        self.camera_manager = CameraManager(camera_size)

        self._task_type = str(task_type).lower()
        self._articulated_regex = articulated_regex
        if self._task_type in ("open", "close"):
            cfg = get_open_task_config()
            cfg.randomize_object = randomize_object
            cfg.task_type = self._task_type
            cfg.success_threshold = float(open_success_threshold)
            cfg.terminate_threshold = float(open_terminate_threshold)
            cfg.init_open_percent = float(open_init_percent)
            cfg.require_joint_grasp = bool(open_require_joint_grasp)
            self.task = OpenTask(config=cfg, object_regex=self._object_regex)
        else:
            cfg = get_task_config()
            cfg.randomize_object = randomize_object
            cfg.object_noise = float(object_noise)
            self.task = PickTask(config=cfg, object_regex=self._object_regex)

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
            except (ValueError, IndexError, OSError) as e:
                print(f"[env] init: skipping scene (load failed: {type(e).__name__}: {e})")
            except Exception as e:
                print(f"[env] init: skipping scene (load error: {type(e).__name__}: {e})")
        if not _init_loaded:
            raise RuntimeError(
                f"Could not load any valid initial scene from {len(self._scene_paths)} candidates after 50 tries"
            )
        n_joints = len(JOINT_NAMES)
        # joint_pos layout (30): legs[0:12] waist[12:15] left_arm[15:22] right_arm[22:29] right_grip[29].
        n_upper_joints = 11

        self.observation_space = spaces.Dict(
            {
                "base_position": spaces.Box(-np.inf, np.inf, shape=(2,)),
                "base_yaw": spaces.Box(-np.pi, np.pi, shape=(1,)),
                "base_rpy": spaces.Box(-np.pi, np.pi, shape=(3,)),
                "base_rp": spaces.Box(-np.pi, np.pi, shape=(2,)),
                "last_base_vel_cmd": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "base_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
                "base_velocity": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "base_angular_velocity": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "joint_pos": spaces.Box(-np.inf, np.inf, shape=(n_joints,)),
                "upper_joint_pos": spaces.Box(-np.inf, np.inf, shape=(n_upper_joints,)),
                "right_hand_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "right_gripper_pos": spaces.Box(-0.0222, 0.0245, shape=(1,)),
                "target_object_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "target_point": spaces.Box(-np.inf, np.inf, shape=(2,)),
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(15,))

    @property
    def camera_shape(self):
        return (*self.camera_manager.size, 3)

    def _scene_name(self):
        if self._current_scene_path is None:
            return ""
        p = Path(self._current_scene_path)
        try:
            return str(p.resolve().relative_to(ASSETS_DIR.resolve()))
        except Exception:
            return str(p)

    def _sample_scene_textures(self):
        if not self._randomize_textures or not self._texture_pools:
            return {}
        # GRPO needs identical Scene contents across group-mate workers and
        # identical np_random consumption regardless of whether _load_scene
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
        xml_path = Path(xml_path)
        if xml_path == self._current_scene_path and texture_override is None:
            return
        # Invalidate the cache up front: self.scene is about to be replaced, so if
        # any step below raises (e.g. set_objects finds no targets) self.scene and
        # the per-scene indices (_robot_white_mid, _cam_ids, ...) are left half-built.
        # Clearing to None guarantees the next _load_scene does a full rebuild instead
        # of early-returning into that half-built scene — which previously desynced
        # _robot_white_mid from self.scene.model and crashed _randomize_lights.
        self._current_scene_path = None
        # Free GPU/EGL resources tied to the OUTGOING scene before we drop the
        # references to it. The env renderers are closed further below; the
        # robot owns a separate visibility renderer that must be closed here, or
        # its framebuffer leaks on the render GPU every reload (VRAM -> OOM).
        # Carry the low-level WBC controller forward across reload -- it isn't
        # scene-specific (only its qpos/qdof index arrays are, rebuilt by
        # G1Robot.__init__'s low_level.setup() call below) and reloading it
        # would re-load its groot_balance/groot_walk ONNX sessions for nothing.
        _prev_low_level = getattr(self.robot, "_low_level", None)
        if self.robot is not None and hasattr(self.robot, "close"):
            self.robot.close()
        for r in self.camera_manager.renderers.values():
            with contextlib.suppress(Exception):
                r.close()
        self.camera_manager.renderers.clear()
        self.camera_manager.renderer = None
        self.camera_manager.fisheye = None
        self._scene_texture_paths = (
            texture_override if texture_override is not None else self._sample_scene_textures()
        )
        self.scene = Scene(
            xml_path,
            robot_xml=XML_PATH,
            mobile_regex=self._object_regex,
            scene_textures=self._scene_texture_paths,
            articulated_regex=self._articulated_regex,
        )
        # Keep BaseMujocoEnv's own model/path/metadata state in sync with the
        # scene this class actually (re)builds -- current_model/is_loaded()/
        # current_model_path (inherited, unmodified) all read these.
        self._mj_model = self.scene.model
        self._mj_base_scene_path = str(xml_path)
        self._scene_metadata = self.scene.metadata
        # Fresh per scene (re)load -- Scene rebuilds its own MjModel/MjData each
        # time, so a stale ObjectManager would otherwise reference dead arrays.
        self.object_manager = ObjectManager(self, batch_idx=0)
        self.object_managers = [self.object_manager]
        self.scene.object_manager = self.object_manager
        hl = self.scene.model.vis.headlight
        self._init_headlight = (hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy())
        self.occ = self.get_occupancy_map(agent_radius=0.15)
        # Extra-inflated for A* path planning; goal sampling still uses self.occ.
        self.occ_safe = self.occ.dilated(0.125)
        self.robot = G1Robot(self.scene.model, self.scene.data, env=self, low_level=_prev_low_level)
        self.robot.set_defaults()
        self.task.set_objects(self.scene)

        m = self.scene.model
        # 16k default shadow map dominates render time; clamp before renderers exist.
        if m.vis.quality.shadowsize > 4096:
            m.vis.quality.shadowsize = 4096
        JOINT, SITE, BODY, CAMERA = (
            mujoco.mjtObj.mjOBJ_JOINT,
            mujoco.mjtObj.mjOBJ_SITE,
            mujoco.mjtObj.mjOBJ_BODY,
            mujoco.mjtObj.mjOBJ_CAMERA,
        )
        self._obs_qpos_ids = np.array([m.jnt_qposadr[_mj_id(m, JOINT, n)] for n in JOINT_NAMES])
        self._obs_dof_ids = np.array([m.jnt_dofadr[_mj_id(m, JOINT, n)] for n in JOINT_NAMES])
        self._obs_fj_dadr = m.jnt_dofadr[_mj_id(m, JOINT, "floating_base_joint")]
        self._obs_r_sid = _mj_id(m, SITE, "right_grasp")
        self._obs_r_grip_qa = m.jnt_qposadr[_mj_id(m, JOINT, "right_Joint1_1")]
        self._obs_pelvis_bid = _mj_id(m, BODY, "pelvis")
        # All robot body IDs — used by the pre-grasp robot-vs-world contact check.
        self._robot_body_set = {
            bid
            for bid in range(m.nbody)
            if (mujoco.mj_id2name(m, BODY, bid) or "").startswith(PREFIX)
        }
        # Robot 'white' material — base rgba hardcoded so per-reset jitter never drifts.
        self._robot_white_mid = int(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{PREFIX}white")
        )
        self._robot_white_rgba0 = np.array([0.75, 0.75, 0.78, 1.0], dtype=np.float32)
        self.camera_manager.ids = {
            name: _mj_id(m, CAMERA, mj_name) for name, mj_name in CameraManager.NAMES.items()
        }

        wrist_id = self.camera_manager.ids.get("wrist_image", -1)
        if wrist_id >= 0:
            self.camera_manager.wrist_cam_pos0 = m.cam_pos[wrist_id].copy()
            self.camera_manager.wrist_cam_quat0 = m.cam_quat[wrist_id].copy()
            self.camera_manager.wrist_cam_fovy0 = float(m.cam_fovy[wrist_id])
        else:
            self.camera_manager.wrist_cam_pos0 = None
            self.camera_manager.wrist_cam_quat0 = None
            self.camera_manager.wrist_cam_fovy0 = None

        # Head camera = fisheye, which composites 5 perspective "tile" cameras
        # sharing the same optical center. Randomize all 5 (plus head_pov) so
        # pos/rot noise on the rig propagates through the fisheye output.
        head_tile_names = [
            f"{PREFIX}head_pov{suf}"
            for suf in ("", "_tile_center", "_tile_up", "_tile_down", "_tile_left", "_tile_right")
        ]
        self.camera_manager.head_cam_ids_all = []
        self.camera_manager.head_cam_pos0_all = []
        self.camera_manager.head_cam_quat0_all = []
        for nm in head_tile_names:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, nm)
            if cid >= 0:
                self.camera_manager.head_cam_ids_all.append(cid)
                self.camera_manager.head_cam_pos0_all.append(m.cam_pos[cid].copy())
                self.camera_manager.head_cam_quat0_all.append(m.cam_quat[cid].copy())

        # (Renderers were already closed at the top of _load_scene, before the
        # outgoing scene's model was dropped.)
        # viewer.launch_passive binds to a specific MjModel/MjData. When the
        # scene model changes (randomize_scene), the old viewer keeps showing
        # the old model — close + relaunch so it tracks the new scene.
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.close()
            self._viewer = None
        if self._launch_viewer:
            self._viewer = mujoco.viewer.launch_passive(
                self.scene.model,
                self.scene.data,
                key_callback=lambda k: setattr(self, "_skip_episode", True) if k == 32 else None,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._viewer.opt.geomgroup[5] = 1

        self.robot.set_env(self)
        if getattr(self, "agent", None) is not None:
            self.agent.setup(self.scene.model, self.scene.data)
            self.agent.set_env(self)

        # The outgoing scene's MjModel/MjData/robot are now unreferenced. The
        # explicit renderer/robot close() above frees the GPU side; a periodic
        # full gc sweep mops up any pybind C buffers whose refcount-zero free
        # lagged. Throttled (not every reload) so the sweep cost doesn't tax
        # steady-state collection under randomize_scene_freq=1.
        self._reload_count = getattr(self, "_reload_count", 0) + 1
        if self._reload_count % 16 == 0:
            gc.collect()

        # Mark the scene fully loaded ONLY after every step above succeeded. The
        # cache was cleared to None at the top, so a mid-load failure leaves it None
        # and the next _load_scene rebuilds from scratch rather than early-returning
        # into a half-built scene.
        self._current_scene_path = xml_path

    @property
    def target(self):
        return self.task.target

    @property
    def time(self):
        return self._sim_time

    # BaseMujocoEnv abstract properties, overridden as single-element
    # wrappers around G1Env's own single-scene/single-robot state --
    # this class is genuinely single-batch (n_batch always 1), unlike
    # CPUMujocoEnv's real multi-batch construction (deliberately not used,
    # see the class docstring). Overriding these three is what makes
    # current_data/current_model/current_robot/is_loaded() (all inherited,
    # unmodified from BaseMujocoEnv) resolve correctly against this class's
    # own state instead of raising AttributeError on CPUMujocoEnv's own
    # (never-populated) _mj_datas/_n_batch/_robots.
    @property
    def mj_datas(self):
        return [self.scene.data]

    @property
    def n_batch(self) -> int:
        return 1

    @property
    def robots(self):
        return (self.robot,)

    # This env is FetchMan's own, so it always serves FetchMan's own map (see
    # utils/scene_maps.OCCUPANCY_MAP_IMPLS); the native envs default to "thor".
    occupancy_map_impl = "aabb"

    def get_occupancy_map(self, agent_radius: float = 0.15):
        """Same name/shape as CPUMujocoEnv.get_occupancy_map -- callers written
        against either env don't need to know which one they have. Goes through
        Scene.occupancy_map (utils/aabb_map.AABBMap, via AABBMap.from_scene's own
        cache semantics), not CPUMujocoEnv.get_thormap's ProcTHORMap/
        from_mj_model_path pipeline, which assumes the real batched renderer
        this class deliberately skips (see the class docstring)."""
        return self.scene.occupancy_map(agent_radius=agent_radius)

    def _ensure_renderer(self, h, w):
        key = (int(h), int(w))
        r = self.camera_manager.renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.scene.model, h, w)
            self.camera_manager.renderers[key] = r
        self.camera_manager.renderer = r
        return r

    def _default_render_opt(self):
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[5] = 1  # robot head/logo + wrist_mount — hidden on wrist cam only
        return opt

    def render_cameras(self):
        H, W = self.camera_manager.size
        out = {}
        for name, mj_name in CameraManager.NAMES.items():
            if mj_name == "head_pov":
                out[name] = self.render_fisheye(output_h=H, output_w=W).copy()
            else:
                r = self._ensure_renderer(H, W)
                opt = self._default_render_opt()
                if name == "wrist_image":
                    opt.geomgroup[1] = 0  # hide wrist_mount + camera body from own wrist obs
                r.update_scene(self.scene.data, self.camera_manager.ids[name], opt)
                out[name] = r.render().copy()
        return out

    def render_fisheye(self, output_h=224, output_w=224):
        self._ensure_fisheye(output_h, output_w)
        r = self._ensure_renderer(
            self.camera_manager.fisheye.tile_size, self.camera_manager.fisheye.tile_size
        )
        opt = self._default_render_opt()
        opt.geomgroup[5] = 0
        return self.camera_manager.fisheye.render(self.scene.data, r, scene_option=opt)

    def render_fisheye_robot_mask(self, output_h=224, output_w=224):
        """Binary mask (uint8, 0/255) of the robot through the same fisheye projection."""
        self._ensure_fisheye(output_h, output_w)
        m = self.scene.model
        robot_geom_ids = [
            gid
            for gid in range(m.ngeom)
            if (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid])) or ""
            ).startswith(PREFIX)
        ]
        r = self._ensure_renderer(
            self.camera_manager.fisheye.tile_size, self.camera_manager.fisheye.tile_size
        )
        return self.camera_manager.fisheye.render_mask(self.scene.data, r, robot_geom_ids)

    def _ensure_fisheye(self, output_h, output_w):
        """(Re)build the head FisheyeRenderer at this output size. Which camera,
        which tiles, and every lens parameter come from
        camera_manager.fisheye_config -- output size is the only thing this
        call decides."""
        if (
            self.camera_manager.fisheye is None
            or self.camera_manager.fisheye.output_h != output_h
            or self.camera_manager.fisheye.output_w != output_w
        ):
            from molmo_spaces.utils.fisheye_cubemap import FisheyeRenderer

            self.camera_manager.fisheye = FisheyeRenderer(
                self.scene.model,
                **self.camera_manager.fisheye_config.cubemap_renderer_kwargs(output_h, output_w),
            )

    def render_debug_panel(self, height=224, width=224):
        imgs = self.render_cameras()
        if not imgs:
            return None
        panels = []
        for name, img in imgs.items():
            img = img.copy()
            try:
                import cv2 as _cv2

                _cv2.putText(
                    img, name, (8, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, _cv2.LINE_AA
                )
                _cv2.putText(
                    img,
                    name,
                    (8, 22),
                    _cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    _cv2.LINE_AA,
                )
            except ImportError:
                pass
            panels.append(img)
        return np.concatenate(panels, axis=1)

    def start_recording(self, width=960, height=720, cameras=("side_right", "third_person")):
        m = self.scene.model
        self._rec_cams = [_mj_id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in cameras]
        self._rec_cams = [c for c in self._rec_cams if c >= 0]
        self._rec_renderer = mujoco.Renderer(m, height, width)
        self._rec_frames = []

    def _capture_frame(self):
        if not getattr(self, "_rec_renderer", None):
            return
        panels = []
        for cid in self._rec_cams:
            self._rec_renderer.update_scene(self.scene.data, cid)
            panels.append(self._rec_renderer.render().copy())
        if not panels:
            return
        frame = np.concatenate(panels, axis=1)
        prompt = getattr(self.task, "_prompt", "")
        if prompt:
            frame = _overlay_prompt(frame, prompt)
        self._rec_frames.append(frame)

    def save_recording(self, path):
        if not self._rec_frames:
            return
        import imageio

        fps = 1.0 / (self.robot.n_substeps * self.scene.model.opt.timestep)
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", pixelformat="yuv420p", output_params=["-crf", "18"]
        )
        for f in self._rec_frames:
            writer.append_data(f)
        writer.close()

    def discard_recording(self):
        self._rec_frames = []

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None


def _overlay_prompt(frame, text):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 36)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = img.width - tw - 20, img.height - th - 20
    for dx, dy, color in ((-1, -1, (0, 0, 0)), (1, 1, (0, 0, 0)), (0, 0, (255, 255, 0))):
        draw.text((x + dx, y + dy), text, font=font, fill=color)
    return np.array(img)


def get_config():
    import ml_collections

    return ml_collections.ConfigDict(
        dict(
            scene="",
            objects=".*",
            seed=0,
            randomize_scene=False,
            randomize_scene_freq=1,
            randomize_object=False,
            randomize_placement=True,
            object_noise=0.0,
            spawn_radius_min=1.0,
            spawn_radius_max=8.0,
            grasp_spawn_radius_min=0.25,
            grasp_spawn_radius_max=0.80,
            spawn_visibility_check=False,
            arm_init_radius=0.0,
            pregrasp_xyz_noise=0.0,
            pregrasp_rot_noise=0.0,
            action_noise_std=0.005,
            action_noise_stride=5,
            spawn_at_grasp=False,
            sample_spawn_first=False,
            ring_num_angles=32,
            spawn_along_line=False,
            walk_dist_min=0.3,
            walk_dist_max=0.8,
            randomize_textures=False,
            scene_textures_glob="textures/randomization/*.png",
            max_textures=5,
            randomize_lighting=False,
            face_yaw_offset=0.0,
            randomize_height=True,
            randomize_height_min=0.0,
            randomize_height_favored=0.95,
            randomize_height_max=None,
            randomize_robot_height=False,
            randomize_robot_height_min=0.7,
            randomize_robot_height_max=0.793,
            wrist_camera_pos_noise=0.0,
            wrist_camera_rot_noise=0.0,
            wrist_camera_fovy_noise=0.0,
            head_camera_pos_noise=0.0,
            head_camera_rot_noise=0.0,
            head_camera_fovy_noise=0.0,
            head_camera_distortion_noise=0.0,
            goal_offset_xy_noise=0.0,
            goal_offset_yaw_noise=0.0,
            start_at_pregrasp_xy_noise=0.0,
            start_at_pregrasp_yaw_noise=0.0,
            start_at_pregrasp_joint_noise=0.0,
            init_arm_at_pregrasp=False,
            launch_viewer=False,
            camera_size=(224, 224),
        )
    )


def make_env(config):
    # Local import: pick_task_sampler_g1ms.py imports BASE_MOVE_GROUP/
    # NOISE_MOVE_GROUP from this module (OBS_SENSORS/sensor classes now live
    # in molmo_spaces/env/g1_sensors.py instead), so importing it back at
    # module level here would be circular. By the time make_env() actually
    # runs, this module has finished loading (those names already exist), so a
    # deferred import inside the function resolves cleanly.
    from molmo_spaces.g1_molmo_port.tasks.pick_task_sampler_g1ms import G1TaskSampler

    # G1TaskSampler now constructs its own G1Env internally (see its
    # __init__'s docstring) -- just return it. NOT task_sampler.sample_task():
    # that runs the real placement-sampling RNG draws, and every caller
    # constructs/attaches its agent only after make_env() returns, so
    # sampling this early would silently skip _sample_task()'s agent-gated
    # precheck_grasp rejection and diverge from gold (see sample_task()'s
    # own docstring).
    return G1TaskSampler(config)
