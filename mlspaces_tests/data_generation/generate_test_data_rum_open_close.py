"""
Helper script to generate test data files for RUM Open and Close tests.

This script runs single episodes with fixed seeds and saves the observations
to numpy files that can be used for regression testing.

Usage:
    python generate_test_data_rum_open_close.py
"""

import argparse
import os
from pathlib import Path

import numpy as np

from mlspaces_tests.data_generation.config import RUMCloseTestConfig, RUMOpenTestConfig
from molmo_spaces.utils.test_utils import run_task_for_steps_with_observations

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data" / "rum_open_close"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_data_for_rum_open():
    """Generate test data files for RUM open configuration.

    This generates:
    - Visual observations from ego-centric camera (wrist_camera)
    - Final joint positions after policy execution
    """
    print(f"\n{'=' * 60}")
    print("Generating test data for RUM OPEN configuration")
    print(f"{'=' * 60}")

    config = RUMOpenTestConfig()
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False

    task_sampler_config = config.task_sampler_config
    task_sampler_class = task_sampler_config.task_sampler_class

    task_sampler = task_sampler_class(config)
    task_sampler.reset()

    # Sample task with fixed seed (seed and task_horizon are set in config)
    task = task_sampler.sample_task()

    # Save observations (vectorized format)
    obs = task.reset()
    obs_list, info_dict = obs
    obs_dict = obs_list[0]  # Extract first (and only) environment's observations

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    print("\n=== Saving visual observations (at task reset) ===")
    # Save exo camera observations (just like RUM pick)
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rum_open_obs_{sensor}.npy"
                np.save(output_path, obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    # Run policy - follow pipeline.py pattern
    print("\n=== Running policy and capturing observations after steps ===")
    policy_config = config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(config, task)  # Policy takes (config, task)
    policy.reset()

    # Run policy for 10 steps using shared utility and capture observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(task, policy, num_steps=10, profiler=config.profiler)
    )
    print(f"Initial qpos: {initial_qpos}")
    print(f"Final qpos: {final_qpos}")

    qpos_path = TEST_DATA_DIR / "rum_open_policy_final_qpos.npy"
    np.save(qpos_path, final_qpos)
    print(f"✓ Saved final qpos -> {qpos_path.name}")

    # Save observations after running policy steps
    print("\n=== Saving observations after running policy steps ===")
    # Save exo camera observations (just like RUM pick)
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if final_obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rum_open_after_steps_{sensor}.npy"
                np.save(output_path, final_obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    print(f"✓ Joints moved: {np.max(np.abs(final_qpos - initial_qpos)):.4f}")
    print("\n=== Done with RUM OPEN ===")


def generate_test_data_for_rum_close():
    """Generate test data files for RUM close configuration.

    This generates:
    - Visual observations from ego-centric camera (wrist_camera)
    - Final joint positions after policy execution
    """
    print(f"\n{'=' * 60}")
    print("Generating test data for RUM CLOSE configuration")
    print(f"{'=' * 60}")

    config = RUMCloseTestConfig()
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False

    task_sampler_config = config.task_sampler_config
    task_sampler_class = task_sampler_config.task_sampler_class

    task_sampler = task_sampler_class(config)
    task_sampler.reset()

    # Sample task with fixed seed (seed and task_horizon are set in config)
    task = task_sampler.sample_task()

    # Save observations (vectorized format)
    obs = task.reset()
    obs_list, info_dict = obs
    obs_dict = obs_list[0]  # Extract first (and only) environment's observations

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    print("\n=== Saving visual observations (at task reset) ===")
    # Save exo camera observations (just like RUM pick)
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rum_close_obs_{sensor}.npy"
                np.save(output_path, obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    # Run policy - follow pipeline.py pattern
    print("\n=== Running policy and capturing observations after steps ===")
    policy_config = config.policy_config
    policy_cls = policy_config.policy_cls
    policy = policy_cls(config, task)  # Policy takes (config, task)
    policy.reset()

    # Run policy for 10 steps using shared utility and capture observations
    initial_qpos, final_qpos, initial_obs_dict, final_obs_dict = (
        run_task_for_steps_with_observations(task, policy, num_steps=10, profiler=config.profiler)
    )
    print(f"Initial qpos: {initial_qpos}")
    print(f"Final qpos: {final_qpos}")

    qpos_path = TEST_DATA_DIR / "rum_close_policy_final_qpos.npy"
    np.save(qpos_path, final_qpos)
    print(f"✓ Saved final qpos -> {qpos_path.name}")

    # Save observations after running policy steps
    print("\n=== Saving observations after running policy steps ===")
    # Save exo camera observations (just like RUM pick)
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if final_obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rum_close_after_steps_{sensor}.npy"
                np.save(output_path, final_obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    print(f"✓ Joints moved: {np.max(np.abs(final_qpos - initial_qpos)):.4f}")
    print("\n=== Done with RUM CLOSE ===")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for RUM Open and Close tests")
    parser.add_argument(
        "--task",
        type=str,
        choices=["open", "close", "all"],
        default="all",
        help="Which task to generate test data for (default: all)",
    )
    args = parser.parse_args()

    if args.task in ["open", "all"]:
        generate_test_data_for_rum_open()
    if args.task in ["close", "all"]:
        generate_test_data_for_rum_close()

    print("\n" + "=" * 60)
    print("RUM OPEN/CLOSE TEST DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
