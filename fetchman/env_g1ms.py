"""G1CPUMujocoEnv: the g1_molmo port's simulation substrate, a CPUMujocoEnv.

Construction and every scene (re)load go through CPUMujocoEnv, so model,
MjData, robots, ObjectManagers, CameraManager and the occupancy-map cache are
the standard ones. Two things stay this class's own to keep the rollout
bit-identical to the reference (fetchman/scripts/
check_gold_parity.py): the scene is compiled by Scene, whose MjData is adopted
without a settle (G1TaskSampler owns that sequencing), and occupancy maps come
from Scene.occupancy_map with gold's cache semantics. Rendering is
CPUMujocoEnv.render_rgb_frame's, through the MJCF camera ids the CameraManager
records (camera_manager.MjcfCameraInfo, MjcfCameraConfig.render_via_mjcf); all
this class adds is the observation image keys those cameras are known by
(`cameras`) and the reference stack's image size (`_render_size`).
"""

import contextlib
import gc
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from fetchman.scene_g1ms import Scene
from molmo_spaces.configs.task_sampler_configs import OccupancyMapImpl
from molmo_spaces.env.camera_manager import CameraManager
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.scene import spec_ops
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.g1 import JOINT_NAMES, PREFIX, XML_PATH, G1Robot

ROBOT_PREFIX = PREFIX


def _mj_id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, f"{PREFIX}{name}")


class G1CPUMujocoEnv(CPUMujocoEnv):
    """See the module docstring."""

    # Observation-dict image key -> camera name in the exp config's
    # G1CameraSystem. The keys are what the reference stack's recordings use
    # (LeRobotRecorder reads `env.cameras`); the values are molmo_spaces'
    # camera names, so anything written against CameraManager finds them too.
    cameras = {"head_image": "head_camera", "wrist_image": "wrist_camera"}

    def __init__(self, exp_config, np_random, launch_viewer=False):
        """Substrate only -- which scene to load, with which textures, and
        which task runs on it are G1TaskSampler's, which constructs this and
        owns `np_random` (handed in as the SAME Generator object so every layer
        draws from one stream). `exp_config` is the port's G1ExpConfig, an
        MlSpacesExpConfig: its robot_config/camera_config/task_sampler_config
        sections are what CPUMujocoEnv and ObjectManager read.
        """
        ts = exp_config.task_sampler_config
        # No scene yet: load_scene installs one via _initialize_with_model.
        super().__init__(
            exp_config,
            robot_factory=self._make_robot,
            mj_model=None,
            mj_base_scene_path=None,
            parallelize=False,
        )
        self._viewer = None
        self._launch_viewer = launch_viewer
        self._object_regex = ts.objects or ".*"
        self._articulated_regex = ts.articulated_regex
        self.np_random = np_random
        self.scene = None
        self._current_scene_path = None
        self._scene_texture_paths: dict[str, list[str]] = {}
        self._skip_episode = False
        self._sim_time = 0.0
        # Read directly by LastBaseVelCmdSensor, which receives this env rather
        # than the task -- stays here for the same reason the sim clock does.
        self._last_base_vel_cmd = np.zeros(3, dtype=np.float32)
        # (height, width) of every rendered observation image -- the reference
        # stack's own camera_size, not camera_config.img_resolution, which is
        # (width, height) and sized for molmo_spaces' recordings.
        self._camera_size = tuple(int(v) for v in ts.camera_size)

    # ---- CPUMujocoEnv hooks ----

    def _make_robot(self, mj_data):
        """CPUMujocoEnv's robot_factory. Carries the low-level WBC controller
        forward across scene reloads -- it isn't scene-specific (only its
        qpos/qdof index arrays are, rebuilt by G1Robot.__init__'s
        low_level.setup() call) and reloading it would re-load its
        groot_balance/groot_walk ONNX sessions for nothing."""
        return G1Robot(self.scene.model, mj_data, env=self, low_level=self._prev_low_level)

    def _create_renderer(self, width, height):
        # Rendering goes through the mujoco.Renderer pool and the fisheye
        # composite below, not CPUMujocoEnv._render_frame.
        return None

    def get_occupancy_map(self, agent_radius: float = 0.15, px_per_m: int = 200, impl=None):
        """Same name/shape as CPUMujocoEnv.get_occupancy_map -- callers written
        against either env don't need to know which one they have. Always the
        FetchMan map (OccupancyMapImpl.AABB) through Scene.occupancy_map
        (AABBMap.from_scene, its own `<scene>_thormap.png` cache), not
        AABBMap.from_model_path: from_scene is the call the bit-exact gold
        rollout goes through and keeps gold's exact cache semantics."""
        if impl is not None and OccupancyMapImpl(impl) != OccupancyMapImpl.AABB:
            raise ValueError(f"G1CPUMujocoEnv serves only {OccupancyMapImpl.AABB!r} maps")
        return self.scene.occupancy_map(agent_radius=agent_radius)

    # ---- state the port reads by these names ----

    @property
    def robot(self) -> G1Robot | None:
        """The one robot (n_batch is always 1 here), by the name the port reads."""
        return self._robots[0] if self._robots else None

    @property
    def camera_size(self) -> tuple[int, int]:
        """(height, width) of the rendered observation images."""
        return self._camera_size

    @property
    def camera_shape(self):
        return (*self._camera_size, 3)

    @property
    def time(self):
        return self._sim_time

    @property
    def viewer_running(self):
        return self._viewer is not None and self._viewer.is_running()

    def _scene_name(self):
        if self._current_scene_path is None:
            return ""
        p = Path(self._current_scene_path)
        try:
            return str(p.resolve().relative_to(ASSETS_DIR.resolve()))
        except Exception:
            return str(p)

    # ---- scene loading ----

    def load_scene(self, xml_path, scene_textures, force=False):
        """Compile and install a scene. Purely substrate: whether the scene is
        usable for the task is the sampler's judgement, made right after this
        returns (see G1TaskSampler._load_scene).

        `force` re-compiles even when this scene is already loaded -- used by
        restore_reset_state, which has to re-apply an exact texture assignment
        onto a scene whose path has not changed."""
        xml_path = Path(xml_path)
        if xml_path == self._current_scene_path and not force:
            return
        # Invalidate the cache up front: self.scene is about to be replaced, so if
        # any step below raises (e.g. the MJCF fails to compile) self.scene and
        # the per-scene indices (_robot_white_mid, ...) are left half-built.
        # Clearing to None guarantees the next load_scene does a full rebuild instead
        # of early-returning into that half-built scene.
        self._current_scene_path = None
        # Free GPU/EGL resources tied to the OUTGOING scene before we drop the
        # references to it. The env renderers are closed here; the robot owns a
        # separate visibility renderer that must be closed too, or its
        # framebuffer leaks on the render GPU every reload (VRAM -> OOM).
        self._prev_low_level = getattr(self.robot, "_low_level", None)
        if self.robot is not None and hasattr(self.robot, "close"):
            self.robot.close()
        self._close_renderers()
        self._scene_texture_paths = scene_textures
        self.scene = Scene(
            xml_path,
            robot_xml=XML_PATH,
            mobile_regex=self._object_regex,
            scene_textures=self._scene_texture_paths,
            articulated_regex=self._articulated_regex,
        )
        # CPUMujocoEnv's own installation of a model: adopts Scene's MjData as
        # the single batch element (no forward, no settle -- see the module
        # docstring), builds the robot through _make_robot, and rebuilds the
        # ObjectManagers and occupancy-map cache.
        self._initialize_with_model(
            self.scene.model,
            str(xml_path),
            mj_datas=[self.scene.data],
            scene_metadata=self.scene.metadata,
        )
        self.object_manager = self.object_managers[0]
        self.scene.object_manager = self.object_manager
        hl = self.scene.model.vis.headlight
        self._init_headlight = (hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy())
        self.occ = self.get_occupancy_map(agent_radius=0.15)
        # Extra-inflated for A* path planning; goal sampling still uses self.occ.
        self.occ_safe = self.occ.dilated(0.125)
        self.robot.set_defaults()

        m = self.scene.model
        # 16k default shadow map dominates render time; clamp before renderers exist.
        if m.vis.quality.shadowsize > 4096:
            m.vis.quality.shadowsize = 4096
        JOINT, SITE, BODY = (
            mujoco.mjtObj.mjOBJ_JOINT,
            mujoco.mjtObj.mjOBJ_SITE,
            mujoco.mjtObj.mjOBJ_BODY,
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

        # Cameras: molmo_spaces' own manager, from the exp config's
        # G1CameraSystem (head fisheye + right wrist). The MJCF facts each
        # Camera records (id, base mounting, fisheye tiles) are what the
        # render_* methods and G1TaskSampler's per-episode camera noise read.
        # Noise is NOT applied at setup: the sampler redraws it every episode
        # from the shared RNG, in the order the gold parity gate compares.
        self.camera_manager = CameraManager()
        self.camera_manager.setup_cameras(self, self.config.camera_config, apply_mjcf_noise=False)

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
        # The policy is the task's, and is re-setup() against this rebuilt
        # scene by G1TaskSampler._load_scene right after this returns.

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
        # and the next load_scene rebuilds from scratch rather than early-returning
        # into a half-built scene.
        self._current_scene_path = xml_path

    # ---- cameras / rendering ----

    def camera(self, obs_key: str):
        """The registered Camera behind an observation image key (see
        `cameras`), or None if the exp config's camera was not in the model."""
        return self.camera_manager.registry.cameras.get(self.cameras[obs_key])

    def mjcf_camera_id(self, obs_key: str) -> int:
        """The MJCF camera id behind an observation image key, or -1."""
        cam = self.camera(obs_key)
        return cam.mjcf.camera_id if cam is not None and cam.mjcf is not None else -1

    @property
    def fisheye(self):
        """The head camera's FisheyeRenderer if one has been built (None until
        the first head render or `ensure_fisheye`)."""
        cam = self.camera("head_image")
        return cam.mjcf.fisheye if cam is not None and cam.mjcf is not None else None

    def ensure_fisheye(self, output_h=None, output_w=None):
        """The head camera's FisheyeRenderer, at the observation image size
        unless another is given -- `fisheye_renderer` by the name and the
        obs-key-free signature the port's callers use."""
        return self.fisheye_renderer(self.cameras["head_image"], output_h, output_w)

    def _render_size(self) -> tuple[int, int]:
        """The reference stack's own camera_size, not
        camera_config.img_resolution."""
        return self._camera_size

    def render_cameras(self):
        """Every observation image, keyed as `cameras` keys them. Each one is
        rendered by CPUMujocoEnv.render_rgb_frame through its MJCF camera (the
        head as its fisheye composite) -- see MjcfCameraConfig.render_via_mjcf
        and the geomgroup_overrides on G1CameraSystem."""
        out = {}
        for obs_key, camera_name in self.cameras.items():
            cam = self.camera(obs_key)
            if cam is None or cam.mjcf is None:
                continue
            out[obs_key] = super().render_rgb_frame(camera_name)
        return out

    def render_rgb_frame(self, camera_name: str, height=None, width=None) -> np.ndarray:
        """CPUMujocoEnv.render_rgb_frame, additionally accepting an observation
        image key (see `cameras`) wherever a camera name is expected."""
        return super().render_rgb_frame(self.cameras.get(camera_name, camera_name), height, width)

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

    # ---- debug ----

    def _cache_probe_local_geoms(self):
        """Snapshot probe geoms in 'fingers-open' pose in root-body frame so debug viz
        can overlay them at any grasp pose without touching physics."""
        m = self.scene.model
        d = self.scene.data
        gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, spec_ops.GRIPPER_PROBE_JOINT_NAME)
        if gj < 0:
            return []
        saved = d.qpos.copy()
        qa = int(m.jnt_qposadr[gj])
        d.qpos[qa : qa + 3] = [0, 0, 0]
        d.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
        for jname, val in zip(
            spec_ops.GRIPPER_PROBE_FINGER_JOINT_NAMES, spec_ops.GRIPPER_PROBE_FINGER_OPEN_QPOS
        ):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = val
        mujoco.mj_forward(m, d)
        probe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec_ops.GRIPPER_PROBE_BODY_NAME)
        descendants = self.scene.get_body_descendants(probe_bid) if probe_bid >= 0 else set()
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

    def close(self):
        if getattr(self, "_viewer", None):
            self._viewer.close()
            self._viewer = None
        super().close()
