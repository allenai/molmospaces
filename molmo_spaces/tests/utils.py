import datetime
from pathlib import Path

import h5py
import numpy as np
import wandb

WANDB_PROJECT_NAME = "molmospaces-ci"
DATETIME_NOW = datetime.datetime.now().strftime("%m%d%y-%H:%M:%S")


def get_success_counts_from_run(results_path: Path, house_id: int) -> tuple[int, int]:
    n_success, n_total = (0, 0)
    if not results_path.is_dir():
        print(f"[ERROR]: the given results folder @ {results_path} doesn't exist")
        return n_success, n_total

    traj_filepath = results_path / f"house_{house_id}" / "trajectories_batch_1_of_1.h5"
    with h5py.File(traj_filepath) as fhandle:
        for traj in fhandle.values():
            success = np.any(traj["success"][:])
            n_total += 1
            n_success += 1 if success else 0

    return n_success, n_total


def push_datagen_results_to_wandb(results_path: Path, house_indices: list[int]) -> list[float]:
    success_rates: list[float] = []
    if not results_path.is_dir():
        print(f"[ERROR]: the given results folder @ {results_path} doesn't exist")
        return success_rates

    run_name = f"datagen-ci-{DATETIME_NOW}-{results_path.stem.replace('_', '-')}"
    with wandb.init(project=WANDB_PROJECT_NAME, name=run_name) as run:
        for house_idx in house_indices:
            table = wandb.Table(
                columns=["house-id", "episode-id", "video-exo-cam", "video-wrist-cam", "tag"]
            )
            results_house_folder = results_path / f"house_{house_idx}"
            traj_filepath = results_house_folder / "trajectories_batch_1_of_1.h5"
            with h5py.File(traj_filepath) as fhandle:
                n_success, n_total = 0, len(fhandle)
                for i_traj, key_traj in enumerate(fhandle.keys()):
                    success = np.any(fhandle[key_traj]["success"][:])
                    n_success += 1 if success else 0
                    video_path_exocam = (
                        results_house_folder
                        / f"episode_0000000{i_traj}_exo_camera_1_batch_1_of_1.mp4"
                    )
                    video_path_wristcam = (
                        results_house_folder
                        / f"episode_0000000{i_traj}_wrist_camera_batch_1_of_1.mp4"
                    )
                    table.add_data(
                        f"procthor-10k-train-{house_idx}",
                        i_traj + 1,
                        "success" if success else "failure",
                        wandb.Video(video_path_exocam),
                        wandb.Video(video_path_wristcam),
                    )
                success_rates.append(n_success / n_total)
            run.log({f"house-{house_idx}": table})

    return success_rates


def main() -> int:
    from dataclasses import dataclass
    from typing import Literal

    import tyro

    @dataclass
    class Args:
        mode: Literal["datagen", "renderer", "camera", "robots"] = "datagen"

    args = tyro.cli(Args)

    match args.mode:
        case "datagen":
            from .test_datagen_franka_pick import (
                TEST_HOUSES_INDICES as TEST_FRANKA_PICK_HOUSES_INDICES,
            )
            from .test_datagen_franka_pick import TEST_OUTPUT_DIR as TEST_FRANKA_PICK_OUTPUT_DIR

            push_datagen_results_to_wandb(
                TEST_FRANKA_PICK_OUTPUT_DIR, TEST_FRANKA_PICK_HOUSES_INDICES
            )

            from .test_datagen_franka_pick_and_place import (
                TEST_HOUSES_INDICES as TEST_FRANKA_PICK_AND_PLACE_HOUSES_INDICES,
            )
            from .test_datagen_franka_pick_and_place import (
                TEST_OUTPUT_DIR as TEST_FRANKA_PICK_AND_PLACE_OUTPUT_DIR,
            )

            push_datagen_results_to_wandb(
                TEST_FRANKA_PICK_AND_PLACE_OUTPUT_DIR, TEST_FRANKA_PICK_AND_PLACE_HOUSES_INDICES
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
