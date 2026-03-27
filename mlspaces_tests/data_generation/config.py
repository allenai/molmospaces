"""
Test-specific configuration for Franka pick data generation.
Disables all randomization for deterministic regression testing.
"""

from pydantic import BaseModel

from molmo_spaces.configs.base_open_task_configs import OpeningBaseConfig
from molmo_spaces.configs.camera_configs import (
    FrankaGoProD405D455CameraSystem,
    FrankaRandomizedD405D455CameraSystem,
)
from molmo_spaces.configs.policy_configs import OpenClosePlannerPolicyConfig
from molmo_spaces.configs.robot_configs import FloatingRUMRobotConfig
from molmo_spaces.configs.task_sampler_configs import OpenTaskSamplerConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickAndPlaceDroidDataGenConfig,
    FrankaPickDroidDataGenConfig,
    FrankaPickRandomizedDataGenConfig,
    RUMPickDataGenConfig,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import PickPlannerPolicy
from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler


class RandomizationTestConfig(BaseModel):
    house_type: str = "ithor"
    house_index: int = 8
    seed: int = 3
    enable_texture_randomization: bool = True
    enable_dynamics_randomization: bool = True
    enable_lighting_randomization: bool = True


class THORMapConfig(BaseModel):
    house_type: str
    house_index: int
    agent_radius: float = 0.25
    px_per_m: int = 100


class THORMapTestConfig(BaseModel):
    thormap_configs: list[THORMapConfig] = [
        THORMapConfig(house_type="procthor-10k", house_index=8),
        THORMapConfig(house_type="ithor", house_index=8),
    ]


class FrankaPickDroidTestConfig(FrankaPickDroidDataGenConfig):
    """Test configuration for Franka pick with DROID cameras - disables randomization for deterministic tests."""

    def model_post_init(self, __context) -> None:
        """Override to apply test-specific settings after initialization."""
        super().model_post_init(__context)

        self.policy_config.policy_cls = PickPlannerPolicy

        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10
        self.task_sampler_config.pickup_types = ["TissueBox"]

        # Set fixed seed and task horizon for deterministic tests
        self.seed = 3
        self.task_horizon = 6

        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing

        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False

        # Robot joint position noise
        self.robot_config.init_qpos_noise_range = {"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        # Disable action noise
        self.robot_config.action_noise_config.enabled = False


class FrankaPickRandomizedTestConfig(FrankaPickRandomizedDataGenConfig):
    """Test configuration for Franka pick with randomized cameras - disables randomization for deterministic tests."""

    def model_post_init(self, __context) -> None:
        """Override to apply test-specific settings after initialization."""
        super().model_post_init(__context)

        self.policy_config.policy_cls = PickPlannerPolicy

        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10
        self.task_sampler_config.pickup_types = ["TissueBox"]
        self.task_sampler_config.robot_object_z_offset_random_min = 0.0
        self.task_sampler_config.robot_object_z_offset_random_max = 0.0

        # Set fixed seed and task horizon for deterministic tests
        self.seed = 3
        self.task_horizon = 6

        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing

        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False

        # Robot joint position noise
        self.robot_config.init_qpos_noise_range = {"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        # Disable action noise
        self.robot_config.action_noise_config.enabled = False


class FrankaPickAndPlaceDroidTestConfig(FrankaPickAndPlaceDroidDataGenConfig):
    def model_post_init(self, __context) -> None:
        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10
        self.task_sampler_config.pickup_types = ["TissueBox"]
        self.task_sampler_config.robot_object_z_offset_random_min = 0.0
        self.task_sampler_config.robot_object_z_offset_random_max = 0.0

        self.policy_config.place_z_offset = 0.02  # lower the lift pose so its not out of reach

        # Set fixed seed and task horizon for deterministic tests
        self.seed = 1
        self.task_horizon = 6

        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing

        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False

        # Robot joint position noise
        self.robot_config.init_qpos_noise_range = {"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        # Disable action noise
        self.robot_config.action_noise_config.enabled = False


class FrankaPickAndPlaceGoProD405D455TestConfig(FrankaPickAndPlaceDroidDataGenConfig):
    """Test configuration for Franka pick and place with GoPro/D405/D455 cameras - disables randomization for deterministic tests."""

    camera_config: FrankaGoProD405D455CameraSystem = FrankaGoProD405D455CameraSystem()

    def model_post_init(self, __context) -> None:
        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10
        self.task_sampler_config.pickup_types = ["TissueBox"]
        self.task_sampler_config.robot_object_z_offset_random_min = 0.0
        self.task_sampler_config.robot_object_z_offset_random_max = 0.0

        self.policy_config.place_z_offset = 0.02  # lower the lift pose so its not out of reach

        # Set fixed seed and task horizon for deterministic tests
        self.seed = 14
        self.task_horizon = 6

        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing

        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False

        # Robot joint position noise
        self.robot_config.init_qpos_noise_range = {"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        # Disable camera noise for deterministic tests
        for camera in self.camera_config.cameras:
            if hasattr(camera, "pos_noise_range"):
                camera.pos_noise_range = (0.0, 0.0)
            if hasattr(camera, "orientation_noise_degrees"):
                camera.orientation_noise_degrees = 0.0
            if hasattr(camera, "lookat_noise_range"):
                camera.lookat_noise_range = (0.0, 0.0)

        # Disable action noise
        self.robot_config.action_noise_config.enabled = False


class RUMPickTestConfig(RUMPickDataGenConfig):
    """Test configuration for RUM pick task - disables randomization for deterministic tests."""

    def model_post_init(self, __context) -> None:
        """Override to apply test-specific settings after initialization."""
        super().model_post_init(__context)

        self.scene_dataset = "procthor-10k"
        # Set fixed seed and task horizon for deterministic tests
        self.seed = 0
        self.task_horizon = 6
        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing
        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10
        self.task_sampler_config.pickup_types = ["TissueBox"]
        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False
        self.camera_config.img_resolution = (624, 352)  # reset to test-data defaults

        # Robot joint position noise (RUM uses gripper, not arm)
        self.robot_config.init_qpos_noise_range = {}


class RUMOpenTestConfig(OpeningBaseConfig):
    """Test configuration for RUM open task - disables randomization for deterministic tests."""

    task_type: str = "open"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler
    )
    policy_config: OpenClosePlannerPolicyConfig = OpenClosePlannerPolicyConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    def model_post_init(self, __context) -> None:
        """Override to apply test-specific settings after initialization."""
        super().model_post_init(__context)

        # Set task type to open
        self.task_type = "open"
        self.task_sampler_config.target_initial_state_open_percentage = 0.0

        # Use iTHOR for testing instead of ProcTHOR
        self.scene_dataset = "ithor"

        # Set target type to drawer
        self.task_sampler_config.pickup_types = ["drawer"]

        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10

        # Set fixed seed and task horizon for deterministic tests
        self.seed = 0
        self.task_horizon = 6

        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing
        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False

        # Robot joint position noise (RUM uses gripper, not arm)
        self.robot_config.init_qpos_noise_range = {}

        # RUM-specific task sampler settings
        self.task_sampler_config.robot_object_z_offset = 0
        self.task_sampler_config.base_pose_sampling_radius_range = (0, 0.8)
        self.task_sampler_config.robot_safety_radius = 0.2

        # Open/close specific policy settings (similar to OpenClosePlannerPolicyConfig)
        self.policy_config.grasp_pos_cost_weight = 1.0
        self.policy_config.grasp_rot_cost_weight = 0.0
        self.policy_config.grasp_vertical_cost_weight = 0.0
        self.policy_config.grasp_com_dist_cost_weight = 0.0

        # Open/close specific policy settings (similar to OpenClosePlannerPolicyConfig)
        self.policy_config.grasp_pos_cost_weight = 1.0
        self.policy_config.grasp_rot_cost_weight = 0.0
        self.policy_config.grasp_vertical_cost_weight = 0.0
        self.policy_config.grasp_com_dist_cost_weight = 0.0


class RUMCloseTestConfig(OpeningBaseConfig):
    """Test configuration for RUM close task - disables randomization for deterministic tests."""

    task_type: str = "close"
    robot_config: FloatingRUMRobotConfig = FloatingRUMRobotConfig()
    task_sampler_config: OpenTaskSamplerConfig = OpenTaskSamplerConfig(
        task_sampler_class=OpenTaskSampler
    )
    policy_config: OpenClosePlannerPolicyConfig = OpenClosePlannerPolicyConfig()
    camera_config: FrankaRandomizedD405D455CameraSystem = FrankaRandomizedD405D455CameraSystem()

    def model_post_init(self, __context) -> None:
        """Override to apply test-specific settings after initialization."""
        super().model_post_init(__context)

        # Set task type to close
        self.task_type = "close"
        self.task_sampler_config.target_initial_state_open_percentage = 0.5

        # Use iTHOR for testing instead of ProcTHOR
        self.scene_dataset = "ithor"

        # Set target type to drawer
        self.task_sampler_config.pickup_types = ["drawer"]

        # Override house to use for testing (use house 8, the default)
        self.task_sampler_config.house_inds = [8]
        self.task_sampler_config.samples_per_house = 1
        # Allow multiple task sampling attempts for retries (max_tasks limits total attempts, not successes)
        self.task_sampler_config.max_tasks = 10

        # Set fixed seed and task horizon for deterministic tests
        self.seed = 0
        self.task_horizon = 6

        self.filter_for_successful_trajectories = False

        # Disable all randomization for deterministic testing
        # Texture randomization
        self.task_sampler_config.enable_texture_randomization = False

        # Robot joint position noise (RUM uses gripper, not arm)
        self.robot_config.init_qpos_noise_range = {}

        # RUM-specific task sampler settings
        self.task_sampler_config.robot_object_z_offset = 0
        self.task_sampler_config.base_pose_sampling_radius_range = (0, 0.8)
        self.task_sampler_config.robot_safety_radius = 0.2

        # Open/close specific policy settings (similar to OpenClosePlannerPolicyConfig)
        self.policy_config.grasp_pos_cost_weight = 1.0
        self.policy_config.grasp_rot_cost_weight = 0.0
        self.policy_config.grasp_vertical_cost_weight = 0.0
        self.policy_config.grasp_com_dist_cost_weight = 0.0
