"""`CuroboPlannerConfig`, split out from `curobo_planner.py` so that referencing the
config does not drag in the planner's dependencies.

`curobo_planner.py` imports torch and curobo at module level. It is
`configs/policy_configs.py` that needs this config class -- to annotate the
Curobo*PolicyConfig fields -- and that module is imported by essentially every
config in the repo. Importing the planner just to name its config therefore cost
every config import ~0.5s and ~1000 modules for torch, and on a machine without
curobo the curobo import then failed anyway, so the cost bought nothing but a
stub class.

The config itself is declarative: plain scalars plus one curobo-typed field
(see `world_config`). Keeping it here, free of torch/curobo, means the heavy
`curobo_planner` module is imported only where a planner is actually
constructed. `curobo_planner` re-exports this name, so existing
`from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig` imports
keep working.
"""

from typing import TYPE_CHECKING, Any

from molmo_spaces.configs.abstract_config import Config

if TYPE_CHECKING:
    # Type checkers and IDEs resolve the real curobo type; at runtime the alias below
    # keeps this module curobo-free. A bare string annotation (`world_config:
    # "WorldConfig"`) does NOT work here -- pydantic cannot resolve the forward ref
    # without curobo installed and raises PydanticUserError ("is not fully defined")
    # on the first instantiation.
    from curobo.geom.types import WorldConfig
else:
    WorldConfig = Any


class CuroboPlannerConfig(Config):
    # --- Curobo setup parameters ---
    curobo_robot_config_path: str
    # curobo.geom.types.WorldConfig under a type checker, Any at runtime (see the
    # TYPE_CHECKING block above), so this module -- and so every config that
    # references it -- stays importable without curobo installed. Only ever assigned
    # at runtime (curobo_planner_server) by code that has curobo.
    world_config: WorldConfig = None
    kinematics_config: dict = None
    lock_joints: dict | None = None  # Override locked joint values: {joint_name: value}

    # --- Robot asset paths (optional, defaults to rby1 for backward compatibility) ---
    urdf_path: str | None = None
    asset_root_path: str | None = None
    usd_robot_root: str | None = None
    collision_spheres_path: str | None = None

    # --- Motion planner parameters ---
    trajopt_tsteps: int = 20
    interpolation_dt: float = 0.02  # Match control_dt for smooth execution in MuJoCo
    time_dilation_factor: float = 1.0
    collision_activation_distance: float = (
        0.2  # collision cost calculated within this distance (metres)
    )
    num_ik_seeds: int = 64
    num_trajopt_seeds: int = 4
    fixed_iters_trajopt: bool = True
    maximum_trajectory_dt: float = 0.5
    max_attempts: int = 5
    collision_cache: dict = {"mesh": 3, "obb": 80}
    check_start_validity: bool = True
    enable_finetune_trajopt: bool = True
