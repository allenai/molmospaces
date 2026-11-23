from pathlib import Path

import h5py
import numpy as np
import wandb

WANDB_PROJECT_NAME = "molmospaces-ci"


def push_datagen_results_to_wandb(results_path: Path, house_indices: list[int]) -> None:
    run_name = f"datagen-ci-{results_path.stem.replace('_', '-')}"
    with wandb.init(project=WANDB_PROJECT_NAME, name=run_name) as run:
        for house_idx in house_indices:
            table = wandb.Table(columns=["episode-id", "video-exo-cam", "video-wrist-cam", "tag"])
            results_house_folder = results_path / f"house_{house_idx}"
            traj_filepath = results_house_folder / "trajectories_batch_1_of_1.h5"
            with h5py.File(traj_filepath) as fhandle:
                for i_traj, key_traj in enumerate(fhandle.keys()):
                    success = np.any(fhandle[key_traj]["success"][:])
                    video_path_exocam = (
                        results_house_folder
                        / f"episode_0000000{i_traj}_exo_camera_1_batch_1_of_1.mp4"
                    )
                    video_path_wristcam = (
                        results_house_folder
                        / f"episode_0000000{i_traj}_wrist_camera_batch_1_of_1.mp4"
                    )
                    table.add_data(
                        i_traj + 1,
                        "success" if success else "failure",
                        wandb.Video(video_path_exocam),
                        wandb.Video(video_path_wristcam),
                    )
            run.log({f"house-{house_idx}": table})
