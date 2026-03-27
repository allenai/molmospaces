"""
Helper script to generate test data files for RUM Pick tests.

This script runs a single episode with a fixed seed and saves the observations
to numpy files that can be used for regression testing.

Usage:
    python generate_test_data_rum_pick.py
"""

import os
from pathlib import Path

import numpy as np

import molmo_spaces.utils.patch_renderer_flags as patch_renderer_flags
from mlspaces_tests.data_generation.config import RUMPickTestConfig
from molmo_spaces.utils.test_utils import run_task_for_steps_with_observations

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data" / "rum_pick"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_data_for_rum():
    patch_renderer_flags()

    """Generate test data files for RUM pick configuration.

    This generates:
    - Visual observations from all cameras (2 exo cameras only, no wrist camera)
    - Final joint positions after policy execution
    """
    print(f"\n{'=' * 60}")
    print("Generating test data for RUM PICK configuration")
    print(f"{'=' * 60}")

    config = RUMPickTestConfig()
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
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rum_pick_obs_{sensor}.npy"
                np.save(output_path, obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    # Run policy - follow pipeline.py pattern (line 295)
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

    qpos_path = TEST_DATA_DIR / "rum_pick_policy_final_qpos.npy"
    np.save(qpos_path, final_qpos)
    print(f"✓ Saved final qpos -> {qpos_path.name}")

    # Save observations after running policy steps
    print("\n=== Saving observations after running policy steps ===")
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if final_obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"rum_pick_after_steps_{sensor}.npy"
                np.save(output_path, final_obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")

    print(f"✓ Joints moved: {np.max(np.abs(final_qpos - initial_qpos)):.4f}")
    print("\n=== Done with RUM PICK ===")


def main():
    generate_test_data_for_rum()

    print("\n" + "=" * 60)
    print("RUM PICK TEST DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
