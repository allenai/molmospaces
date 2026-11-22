from pathlib import Path

import h5py
import numpy as np
import wandb

WANDB_PROJECT_NAME = "molmospaces-ci"


def push_datagen_results_to_wandb(results_path: Path, house_indices: list[int]) -> None:
    run_name = f"datagen-ci-{results_path.stem.replace('_', '-')}"
    with wandb.init(project=WANDB_PROJECT_NAME, name=run_name) as run:
        for house_idx in house_indices:
            results_house_folder = results_path / f"house_{house_idx}"
        pass
