import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from molmospaces_resources import HFRemoteStorage, R2RemoteStorage, ResourceManager

logger = logging.getLogger("molmospaces_resources")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

DEFAULT_CACHE_DIR = Path.home() / ".molmospaces"

SOURCE_TO_VERSION = {
    "objects": {
        "mjcf": {
            "thor": "20251117",
            "objaverse": "20260131",
        },
        "usd": {
            "thor": "20260128",
            "objaverse": "20260128",
        },
    },
    "scenes": {
        "mjcf": {
            "ithor": "20251217",
            "procthor-10k-train": "20251122",
            "procthor-10k-val": "20251217",
            "procthor-10k-test": "20251121",
            "holodeck-objaverse-train": "20251217",
            "holodeck-objaverse-val": "20251217",
            "procthor-objaverse-train": "20251205",
            "procthor-objaverse-val": "20251205",
        },
        "usd": {
            "ithor": "20260121",
            "procthor-10k-train": "20260128",
            "procthor-10k-val": "20260128",
            "procthor-10k-test": "20260128",
            "procthor-objaverse-train": "20260128",
            "procthor-objaverse-val": "20260128",
            "holodeck-objaverse-train": "20260128",
            "holodeck-objaverse-val": "20260128",
        },
    },
}

TYPE_TO_PREFIX: dict[str, str] = {
    "mjcf": "mujoco",
    "usd": "isaac",
}


@dataclass
class DownloadArgs:
    # `mjcf` for MuJoCo or ManiSkill, `usd` for Isaac
    type: Literal["mjcf", "usd"]

    # Path to symlink extracted data from the cache_dir
    install_dir: Path

    assets: list[Literal["thor", "objaverse"]] = field(default_factory=lambda: ["thor"])

    scenes: list[
        Literal[
            "ithor",
            "procthor-10k-train",
            "procthor-10k-val",
            "procthor-10k-test",
            "procthor-objaverse-train",
            "procthor-objaverse-val",
            "holodeck-objaverse-train",
            "holodeck-objaverse-val",
        ]
    ] = field(default_factory=list)

    # Path to extract (versioned) downloaded data
    cache_dir: Path = DEFAULT_CACHE_DIR

    # If not provided, uses HF_TOKEN from environment
    hf_token: str | None = None

    # Use R2 remote storage (HuggingFace by default)
    use_r2: bool = False


def main() -> int:
    args = tyro.cli(DownloadArgs)

    args.install_dir.mkdir(parents=True, exist_ok=True)

    assert args.type in TYPE_TO_PREFIX, (
        f"Something went wrong, must only use {set(TYPE_TO_PREFIX.keys())}, but got '{args.type}'"
    )

    logger.info(f"Symlinking from directory '{args.install_dir}'")
    logger.info(f"Downloading '{args.type}' version of the assets")

    sources_to_version = dict(objects=dict(), scenes=dict())
    sources_to_version["objects"]["thor"] = SOURCE_TO_VERSION["objects"][args.type]["thor"]
    for dataset_id in args.assets:
        sources_to_version["objects"][dataset_id] = SOURCE_TO_VERSION["objects"][args.type][
            dataset_id
        ]

    for dataset_id in args.scenes:
        sources_to_version["scenes"][dataset_id] = SOURCE_TO_VERSION["scenes"][args.type][
            dataset_id
        ]

    manager = ResourceManager(
        remote_storage=R2RemoteStorage(f"{TYPE_TO_PREFIX[args.type]}-thor-resources")
        if args.use_r2
        else HFRemoteStorage(
            repo_id="allenai/molmospaces",
            repo_prefix=TYPE_TO_PREFIX[args.type],
            token=args.hf_token or os.getenv("HF_TOKEN"),
        ),
        data_type_to_source_to_version=sources_to_version,
        symlink_dir=args.install_dir,
        cache_dir=args.cache_dir / args.type,
        force_install=True,
    )
    manager.setup()
    enforce_scenes = len(args.scenes) > 0
    enforce_objects = len(args.assets) > 1

    if enforce_scenes:
        logger.info("Installing scenes...")
        manager.install_all_for_data_type("scenes", skip_linking=False)

    if enforce_objects:
        logger.info("Installing objects...")
        manager.install_all_for_data_type("objects", skip_linking=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
