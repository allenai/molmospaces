"""Typed configuration for the FetchMan port: G1TaskSamplerConfig (scene,
textures, spawning), G1TaskConfig (reward, termination, action noise) and
G1ExpConfig (robot, cameras, timing). Field names are gold's flat-config
names, which the skill-profile overrides are keyed by.
"""

from __future__ import annotations

from pathlib import Path

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import G1CameraSystem
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.policy_configs_fetchman_pick import (
    FetchmanPickPlannerPolicyConfig,
)
from molmo_spaces.configs.robot_configs import BaseRobotConfig, G1Config
from molmo_spaces.configs.task_configs import BaseMujocoTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    OccupancyMapImpl,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR


class G1TaskConfig(BaseMujocoTaskConfig):
    """What G1Task and its subclasses read: reward/termination rules, action
    noise, and the pick/open specifics."""

    task_cls: type | None = None  # PickTask | OpenTask, set by the exp config

    # "pick" (default) | "open" | "close". Selects PickTask vs OpenTask.
    task_type: str = "pick"

    # --- action noise (G1Task.step) ---
    action_noise_std: float = 0.005
    # Resample noise once per this many steps, so recorded frames stay consistent.
    action_noise_stride: int = 5

    # --- termination rules (G1Task.step) ---
    terminate_before_grasp_collision: bool = True
    terminate_grasp_if_not_visible: bool = True
    terminate_on_grasp_collision: bool = True

    # --- pick ---
    randomize_object: bool = False
    object_noise: float = 0.0  # gaussian xy jitter on object spawn (m)

    # --- open / close (only read when task_type is "open" or "close") ---
    open_success_threshold: float = 0.5  # fraction of joint range counting as success
    open_terminate_threshold: float = 1.0  # fraction of joint range ending the episode
    open_init_percent: float = 0.9  # only for task_type="close" (start mostly open)
    open_require_joint_grasp: bool = False  # only accept objects with a joint-grasp .npz


class G1TaskSamplerConfig(BaseMujocoTaskSamplerConfig):
    """Scene selection, textures and per-episode randomization. The inherited
    house-iteration fields get defaults because this sampler draws scenes from
    `scene` (a glob, a path, or a scene-list .txt) itself."""

    house_inds: list[int] | None = None
    samples_per_house: int | None = None
    task_batch_size: int = 1
    max_tasks: int | None = None

    # This sampler serves FetchMan's own AABB grid, not ProcTHOR's map.
    occupancy_map_impl: OccupancyMapImpl = OccupancyMapImpl.AABB

    # Filled in by the concrete config's get_config(), not here: the sampler
    # module imports this one for its type annotations, so naming the class at
    # class-definition time is a circular import.
    task_sampler_class: type | None = None

    # --- scene selection ---
    # A concrete .xml path, a glob, or a .txt listing scenes one per line.
    scene: str = ""
    objects: str = ".*"  # regex over pickable object names
    articulated_regex: str | None = None  # only used for open/close tasks
    randomize_scene: bool = False
    randomize_scene_freq: int = 1
    camera_size: tuple[int, int] = (224, 224)

    # --- texture / lighting randomization ---
    randomize_textures: bool = False
    scene_textures_glob: str = "textures/randomization/*.png"
    max_textures: int = 5  # per category (5 walls, 5 floors, ...)
    # Subsample the pools with a fixed local RNG, so GRPO group-mates that must
    # share a scene consume np_random identically regardless of scene reuse.
    deterministic_scene_textures: bool = False
    randomize_textures_keep_prob: float = 0.25
    randomize_textures_solid_color_prob: float = 0.30
    randomize_lighting: bool = False
    randomize_lighting_keep_prob: float = 0.25

    # --- camera perturbation ---
    head_camera_distortion_noise: float = 0.0
    head_camera_fovy_noise: float = 0.0
    head_camera_pos_noise: float = 0.0
    head_camera_rot_noise: float = 0.0
    wrist_camera_fovy_noise: float = 0.0
    wrist_camera_pos_noise: float = 0.0
    wrist_camera_rot_noise: float = 0.0

    # --- robot placement / goal sampling ---
    randomize_placement: bool = True
    spawn_radius_min: float = 1.0
    spawn_radius_max: float = 8.0
    grasp_spawn_radius_min: float = 0.25
    grasp_spawn_radius_max: float = 0.80
    spawn_at_grasp: bool = False
    sample_spawn_first: bool = False
    spawn_along_line: bool = False
    spawn_reachability_check: bool = True
    spawn_visibility_check: bool = False
    ring_num_angles: int = 32
    walk_dist_min: float = 0.3
    walk_dist_max: float = 0.8
    face_yaw_offset: float = 0.0
    goal_offset_xy_noise: float = 0.0
    goal_offset_yaw_noise: float = 0.0
    # Reject an (object, placement) sample whose best grasp is not even
    # plausibly IK-reachable, rather than committing to a guaranteed-fail run.
    reset_precheck_grasp: bool = True

    # --- arm / pregrasp initialization ---
    arm_init_radius: float = 0.0
    init_arm_at_pregrasp: bool = False
    pregrasp_rot_noise: float = 0.0
    pregrasp_xyz_noise: float = 0.0
    start_at_pregrasp_joint_noise: float = 0.0
    start_at_pregrasp_xy_noise: float = 0.0
    start_at_pregrasp_yaw_noise: float = 0.0

    # --- support / robot height randomization ---
    randomize_height: bool = True
    randomize_height_favored: float = 0.95
    randomize_height_max: float | None = None
    randomize_height_min: float = 0.0
    randomize_robot_height: bool = False
    randomize_robot_height_max: float = 0.793
    randomize_robot_height_min: float = 0.7

    # --- skill profiles ---
    # (name, weight, {field_name: value}) drawn per episode; the fields override
    # this config's own values on the sampler/task for that episode.
    skill_profiles: list[tuple[str, float, dict]] | None = None


class G1ExpConfig(MlSpacesExpConfig):
    """Gold's timing/robot/camera setup.

    The reference sampler and task implement molmo_spaces' abstractions
    (`BaseMujocoTaskSampler` / `BaseMujocoTask`), so run_pipeline.py can drive
    this config directly -- `task_sampler_config.task_sampler_class` is
    `G1TaskSampler`, and `sample_task(house_index=N)` indexes its own scene
    glob. `make_task_sampler` is the equivalent one-liner."""

    num_envs: int = 1
    task_type: str = "pick"
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = {
        "distance": 5.0,
        "azimuth": 45.0,
        "elevation": -30.0,
        "lookat": [0.0, 0.0, 0.5],
    }

    # gold's own controller runs one policy tick every physics step, at
    # model.opt.timestep=0.005 (n_substeps=1) -- a real 5ms policy tick.
    sim_dt_ms: float = 5.0
    ctrl_dt_ms: float = 5.0
    policy_dt_ms: float = 5.0
    # gold gives each episode a fixed 60s sim-time budget; 60s / 0.005s = 12000.
    task_horizon: int | None = 12000
    seed: int | None = 0

    scene_dataset: str = "procthor-10k"
    data_split: str = "val"

    robot_config: BaseRobotConfig = G1Config()
    camera_config: G1CameraSystem = G1CameraSystem()
    policy_config: BasePolicyConfig = FetchmanPickPlannerPolicyConfig()

    task_sampler_config: G1TaskSamplerConfig = G1TaskSamplerConfig()
    task_config: G1TaskConfig = G1TaskConfig()

    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "g1_port_v1"

    @property
    def tag(self) -> str:
        return "g1_port_datagen"
