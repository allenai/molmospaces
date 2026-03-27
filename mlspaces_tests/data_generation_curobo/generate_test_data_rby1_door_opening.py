"""
Helper script to generate test data files for RBY1 Door Opening tests.

This script runs a single episode with a fixed seed and saves the observations
to numpy files that can be used for regression testing.

Usage:
    python generate_test_data_rby1_door_opening.py
"""

import os
from pathlib import Path

import numpy as np

from molmo_spaces.utils.test_utils import run_task_for_steps_with_observations

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data" / "rby1_door_opening"


def generate_test_data_for_rby1():
    """Generate test data files for RBY1 door opening configuration.

    This generates:
    - Visual observations from all cameras at task reset (get_observations)
    - Visual observations from all cameras after running policy steps
    - Final joint positions after policy execution
    """
    print(f"\n{'=' * 60}")
    print("Generating test data for RBY1 DOOR OPENING configuration")
    print(f"{'=' * 60}")

    from molmo_spaces.data_generation.config.door_opening_configs import DoorOpeningDataGenConfig

    config = DoorOpeningDataGenConfig()
    config.seed = 0
    config.task_horizon = 6
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    config.task_sampler_config.samples_per_house = 1
    config.task_sampler_config.house_inds = [22]
    config.num_workers = 1
    config.filter_for_successful_trajectories = False

    # Mesh collision cache must be > 0 when using CollisionCheckerType.MESH with
    # WorldConfig.create_collision_support_world (which converts primitives to meshes)
    if config.policy_config is not None:
        for planner_cfg in [
            config.policy_config.left_curobo_planner_config,
            config.policy_config.right_curobo_planner_config,
        ]:
            if planner_cfg is not None:
                planner_cfg.collision_cache = {"mesh": 500, "obb": 80}

    task_sampler_class = config.task_sampler_config.task_sampler_class
    task_sampler = task_sampler_class(config)
    task_sampler.reset()

    # Sample task with fixed seed
    task = task_sampler.sample_task()

    # Save initial observations using get_observations (matching existing test pattern)
    print("\n=== Saving initial observations (using get_observations) ===")
    obs = task.get_observations()
    obs_dict = obs[0]  # Extract first environment's observations

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rby1_door_opening_obs_{sensor}.npy"
                np.save(output_path, obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    # Run policy and capture observations after steps
    print("\n=== Running policy and capturing observations after steps ===")
    policy_config = config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(config, task)
    policy.reset()

    # Run policy for 10 steps using shared utility and capture observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(task, policy, num_steps=10, profiler=config.profiler)
    )
    print(f"Initial qpos: {initial_qpos}")
    print(f"Final qpos: {final_qpos}")

    # Save final qpos
    qpos_path = TEST_DATA_DIR / "rby1_door_opening_policy_final_qpos.npy"
    np.save(qpos_path, final_qpos)
    print(f"✓ Saved final qpos -> {qpos_path.name}")

    # Save observations after running policy steps
    print("\n=== Saving observations after running policy steps ===")
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if final_obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rby1_door_opening_after_steps_{sensor}.npy"
                np.save(output_path, final_obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    print(f"✓ Joints moved: {np.max(np.abs(final_qpos - initial_qpos)):.4f}")
    print("\n=== Done with RBY1 DOOR OPENING ===")


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_test_data_for_rby1()

    print("\n" + "=" * 60)
    print("RBY1 DOOR OPENING TEST DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
