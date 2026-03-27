# MolmoSpaces

A modular robotics simulation framework built on MuJoCo for data generation and robot control experiments.

## Code Structure

```
molmo_spaces/
├── configs/                         # Configuration management
│   ├── abstract_config.py          # Base configuration class
│   ├── abstract_exp_config.py      # Experiment configuration interface
│   ├── base_nav_to_obj_config.py   # Navigation task base config
│   ├── base_open_task_configs.py   # Open/close task base configs
│   ├── base_packing_configs.py     # Packing task base config
│   ├── base_pick_and_place_color_configs.py
│   ├── base_pick_and_place_configs.py
│   ├── base_pick_and_place_next_to_configs.py
│   ├── base_pick_config.py         # Pick task base config
│   ├── camera_configs.py           # Camera system configurations
│   ├── policy_configs.py           # Planner policy configurations
│   ├── policy_configs_baselines.py # Learned/teleop policy configurations
│   ├── robot_configs.py            # Robot-specific configurations
│   ├── task_configs.py             # Task parameter configurations
│   └── task_sampler_configs.py     # Task sampler configurations
├── controllers/                     # Robot control interfaces
│   ├── abstract.py                 # Base controller class
│   ├── base_pose.py                # Base pose controller
│   ├── joint_pos.py                # Joint position controller
│   ├── joint_rel_pos.py            # Joint relative position controller
│   ├── joint_vel.py                # Joint velocity controller
│   └── torso_height.py             # Torso height controller
├── data_generation/                 # Data generation pipeline
│   ├── config/                     # Data generation configs
│   │   ├── benchmarks_datagen_configs.py
│   │   ├── door_opening_configs.py
│   │   ├── nav_to_obj_configs.py
│   │   └── object_manipulation_datagen_configs.py
│   ├── config_registry.py          # Config auto-discovery registry
│   ├── main.py                     # Entry point for data generation
│   └── pipeline.py                 # Parallel rollout runner
├── env/                            # Environment abstractions
│   ├── arena/                      # Scene construction and randomization
│   │   ├── randomization/          # Domain randomization (lighting, texture, dynamics)
│   │   ├── arena_utils.py
│   │   ├── bathroom.py
│   │   ├── cabinet.py
│   │   ├── drawer.py
│   │   ├── kitchen.py
│   │   ├── procthor_types.py
│   │   └── scene_tweaks.py
│   ├── abstract_sensors.py         # Base sensor interface
│   ├── camera_manager.py           # Camera lifecycle management
│   ├── data_views.py               # Data view utilities
│   ├── env.py                      # MuJoCo environment wrapper
│   ├── mj_extensions.py            # MuJoCo API extensions
│   ├── object_manager.py           # Scene object management
│   ├── sensors.py                  # Sensor implementations
│   └── sensors_cameras.py          # Camera sensor implementations
├── evaluation/                     # Benchmark evaluation
│   ├── configs/                    # Evaluation configurations
│   ├── benchmark_schema.py         # Benchmark JSON schema
│   ├── eval_main.py               # Evaluation entry point
│   ├── json_eval_runner.py        # JSON benchmark runner
│   ├── policy_server.py           # Policy serving for evaluation
│   └── robot_eval_overrides.py    # Robot-specific eval overrides
├── grasp_generation/               # Grasp generation pipeline
│   ├── pipeline/                  # Mesh processing and grasp generation
│   ├── robotiq_gripper.py         # Robotiq gripper utilities
│   ├── run_articulable.py         # Articulated object grasps
│   └── run_rigid.py               # Rigid object grasps
├── housegen/                       # House/scene generation from JSON
│   ├── builder.py                 # Scene builder
│   ├── constants.py               # Scene generation constants
│   ├── exporter.py                # MJCF exporter
│   └── utils.py                   # Scene generation utilities
├── kinematics/                     # Kinematic solvers
│   ├── parallel/                  # GPU-parallel kinematics
│   ├── bimanual_yam_kinematics.py
│   ├── floating_rum_kinematics.py
│   ├── franka_kinematics.py
│   ├── i2rt_yam_kinematics.py
│   ├── mujoco_kinematics.py       # MuJoCo-based kinematics
│   ├── rby1_kinematics.py         # RBY1-specific kinematics
│   └── stretch_kinematics.py
├── planner/                        # Motion planning
│   ├── robot_specific/            # Robot-specific planners
│   ├── abstract.py                # Base planner interface
│   ├── astar_planner.py           # A* path planner (navigation)
│   ├── curobo_planner.py          # CuRobo integration
│   ├── curobo_planner_client.py   # CuRobo gRPC client
│   └── curobo_planner_server.py   # CuRobo gRPC server
├── policy/                         # Policy implementations
│   ├── learned_policy/            # Learned policies (Pi, CAP, etc.)
│   ├── solvers/                   # Planning-based policies
│   │   ├── navigation/           # Navigation planner policies
│   │   └── object_manipulation/  # Pick, place, open/close planner policies
│   ├── base_policy.py            # Base policy interface
│   ├── dummy_policy.py           # No-op policy for testing
│   └── random_policy.py          # Random action policy
├── renderer/                       # Rendering backends
│   ├── offline_renderers/         # Domain randomization and Omniverse renderers
│   ├── abstract_renderer.py       # Base renderer interface
│   ├── filament_rendering.py      # Filament renderer
│   ├── opengl_context.py          # OpenGL context management
│   └── opengl_rendering.py        # OpenGL renderer
├── robots/                         # Robot implementations
│   ├── robot_views/               # Robot observation/action views
│   ├── abstract.py                # Base robot interface
│   ├── bimanual_yam.py
│   ├── floating_robotiq.py
│   ├── floating_rum.py
│   ├── franka.py
│   ├── i2rt_yam.py
│   └── rby1.py
├── tasks/                          # Task definitions
│   ├── util_samplers/             # Grasp and navgoal samplers
│   ├── task.py                    # Base task interface
│   ├── task_sampler.py            # Task sampling interface
│   ├── pick_task.py               # Pick task
│   ├── pick_and_place_task.py     # Pick-and-place task
│   ├── pick_and_place_color_task.py
│   ├── pick_and_place_next_to_task.py
│   ├── opening_tasks.py           # Door/drawer open/close tasks
│   ├── packing_task.py            # Packing task
│   ├── nav_task.py                # Navigation task
│   ├── multi_task.py              # Multi-task wrapper
│   ├── *_task_sampler.py          # Per-task sampler implementations
│   └── scene_xml_utils.py         # Scene XML construction
├── utils/                          # Shared utilities
│   ├── constants/                 # Domain constants (cameras, objects, rooms, etc.)
│   ├── devices/                   # Input devices (keyboard, spacemouse)
│   ├── save_utils.py             # Trajectory saving (HDF5, video)
│   ├── profiler_utils.py         # Profiling and timing
│   ├── mp_logging.py             # Multiprocessing-safe logging
│   ├── camera_utils.py           # Camera projection utilities
│   ├── linalg_utils.py           # Linear algebra helpers
│   ├── pose.py                   # Pose representations
│   └── ...                       # Many more domain-specific utilities
└── molmo_spaces_constants.py       # Global paths and constants
```

## Information Flow

### Data Generation Pipeline

1. **Entry Point**: `main.py` loads an experiment configuration and initializes the pipeline.
2. **Configuration**: Experiment configs inherit from `MlSpacesExpConfig` and define:
    - Task sampler configuration
    - Robot configuration
    - Policy configuration
    - Camera configuration
    - Fixed task parameters
3. **Pipeline**: `ParallelRolloutRunner` manages parallel execution:
    - Creates task samplers for each worker process
    - Samples tasks with fixed parameters
    - Initializes policies and environments
    - Runs rollouts and collects data
4. **Task Execution**: Each episode follows:
    - Task sampling → Policy initialization → Episode rollout → Success evaluation

### Core Components

- **Tasks**: Define robot objectives and success criteria
- **Policies**: Generate actions from observations (planner, teleop, learned)
- **Robots**: Interface with MuJoCo simulation and provide control
- **Environments**: Manage MuJoCo models and scene construction
- **Controllers**: Handle low-level robot control commands
- **Renderer**: Rendering backends (OpenGL, Filament) for camera observations

## Configuration Hierarchy

The framework uses a hierarchical configuration system:

1. **Experiment Config** (`MlSpacesExpConfig`): Top-level experiment parameters
2. **Task Sampler Config**: Defines task sampling ranges and constraints
3. **Task Config**: Fixed parameters for specific task instances
4. **Robot Config**: Robot-specific settings and control modes
5. **Camera Config**: Camera system and sensor layout
6. **Policy Config**: Policy type and parameters

## Example Running Command

```bash
# Set environment variables (macOS)
export PYTHONPATH="${PYTHONPATH}:."
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Run door opening data generation
python -m molmo_spaces.data_generation.main DoorOpeningDataGenConfig
```

## Key Features

- **Modular Design**: Clean separation between tasks, policies, and robots
- **Parallel Execution**: Multi-process data generation with multiprocessing
- **Flexible Configuration**: Hierarchical config system with auto-discovery registry
- **Multiple Policy Types**: Support for planning, teleoperation, and learned policies
- **Robot Agnostic**: Abstract interfaces allow easy robot integration
- **Domain Randomization**: Lighting, texture, and dynamics randomization
- **Multiple Renderers**: OpenGL and Filament rendering backends
