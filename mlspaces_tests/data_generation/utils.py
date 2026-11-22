from pathlib import Path

import wandb

WANDB_PROJECT_NAME = "molmospaces-ci"


def push_datagen_results_to_wandb(results_path: Path) -> None:
    run_name = f"datagen-ci-{results_path.stem.replace('_', '-')}"
    with wandb.init(project=WANDB_PROJECT_NAME, name=run_name) as run:
        pass
