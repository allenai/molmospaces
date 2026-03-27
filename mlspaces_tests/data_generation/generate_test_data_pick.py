"""
Helper script to generate test data files for Franka Pick Place tests.

This script runs a single episode with a fixed seed and saves the observations
to numpy files that can be used for regression testing.

Usage:
    python generate_test_data_franka_pick_place.py [--config droid|randomized|all]
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from mlspaces_tests.data_generation.config import (
    FrankaPickDroidTestConfig,
    FrankaPickRandomizedTestConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.utils.test_utils import run_task_for_steps_with_observations

log = logging.getLogger(__name__)


TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data" / "franka_pick"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_test_data_for_config(config_orig, config_name: str):
    """Generate test data files for a specific config.

    Args:
        config: Test configuration instance
        config_name: Name identifier for the config (e.g., 'droid', 'randomized')
    """
    config = config_orig.model_copy(deep=True)
    print(f"\n{'=' * 60}")
    print(f"Generating test data for {config_name.upper()} configuration")
    print(f"{'=' * 60}")

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

    print("\n=== Saving visual observations (at task reset) ===")
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"franka_pick_{config_name}_obs_{sensor}.npy"
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

    qpos_path = TEST_DATA_DIR / f"franka_pick_{config_name}_policy_final_qpos.npy"
    np.save(qpos_path, final_qpos)
    print(f"✓ Saved final qpos -> {qpos_path.name}")

    # Save observations after running policy steps
    print("\n=== Saving observations after running policy steps ===")
    for sensor in task.sensor_suite.sensors:
        if "camera" in sensor and "sensor_param" not in sensor:
            if final_obs_dict[sensor] is not None:
                output_path = TEST_DATA_DIR / f"franka_pick_{config_name}_after_steps_{sensor}.npy"
                np.save(output_path, final_obs_dict[sensor])
                print(f"Saved test data for {sensor} -> {output_path.name}")
                # from PIL import Image
                # Image.fromarray(final_obs_dict[sensor]).save(output_path.with_suffix(".png"))

    print(f"✓ Joints moved: {np.max(np.abs(final_qpos - initial_qpos)):.4f}")

    config = config_orig.model_copy(deep=True)
    runner = ParallelRolloutRunner(config)
    config.output_dir = TEST_DATA_DIR / f"franka_pick_{config_name}"
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)

    runner.run()

    print(f"\n=== Done with {config_name.upper()} ===")


def generate_droid_test_data():
    """Generate test data for DROID camera configuration."""
    config = FrankaPickDroidTestConfig()
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    generate_test_data_for_config(config, "droid")


def generate_randomized_test_data():
    """Generate test data for randomized camera configuration."""
    config = FrankaPickRandomizedTestConfig()
    config.use_passive_viewer = False
    config.profile = True
    config.use_wandb = False
    generate_test_data_for_config(config, "randomized")


def main():
    parser = argparse.ArgumentParser(description="Generate test data for Franka pick place tests")
    parser.add_argument(
        "--config",
        type=str,
        choices=["droid", "randomized", "all"],
        default="all",
        help="Which configuration to generate test data for (default: all)",
    )
    args = parser.parse_args()

    if args.config == "droid":
        generate_droid_test_data()
    elif args.config == "randomized":
        generate_randomized_test_data()
    elif args.config == "all":
        generate_droid_test_data()
        generate_randomized_test_data()

    print("\n" + "=" * 60)
    print("ALL TEST DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
