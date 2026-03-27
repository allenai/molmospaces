from molmospaces_resources import setup_resource_manager, ResourceManager, R2RemoteStorage
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR

MUJOCO_TRAINING_DATA_SYMLINK_DIR = ASSETS_DIR.parent / "mujoco-training-data"
MUJOCO_TRAINING_DATA_CACHE_DIR = DATA_CACHE_DIR.parent / "mujoco-training-data"

assert MUJOCO_TRAINING_DATA_CACHE_DIR.resolve() != MUJOCO_TRAINING_DATA_SYMLINK_DIR.resolve(), (
    f"Training data paths\n"
    f"{MUJOCO_TRAINING_DATA_SYMLINK_DIR=}\n"
    f"and\n"
    f"{MUJOCO_TRAINING_DATA_CACHE_DIR=}\n"
    f"automatically derived from\n"
    f"{ASSETS_DIR=}\n"
    f"and\n"
    f"{DATA_CACHE_DIR=}\n"
    f"(defined in molmo_spaces.molmo_spaces_constants) should not resolve to the same path."
)

MUJOCO_TRAINING_DATA_TYPE_TO_SOURCE_TO_VERSION = {
    "franka-rby1-training-data": {
        "FrankaPickAndPlaceOmniCamConfig": "20260210",
    }
}


_RESOURCE_MANAGER: ResourceManager | None = None


def get_resource_manager(force_post_setop: bool = False) -> ResourceManager:
    global _RESOURCE_MANAGER
    if _RESOURCE_MANAGER is None:

        def post_setup(manager: ResourceManager) -> None:
            for (
                data_type,
                source_to_version,
            ) in MUJOCO_TRAINING_DATA_TYPE_TO_SOURCE_TO_VERSION.items():
                source_to_archives = {}
                for source in source_to_version:
                    source_to_archives[source] = [
                        archive
                        for archive in manager.tries(data_type, source).keys()
                        if "_house_" not in archive and "FloorPlan" not in archive
                    ]
                manager.install_packages(data_type, source_to_archives)

        _RESOURCE_MANAGER = setup_resource_manager(
            R2RemoteStorage("mujoco-thor-training-data"),
            symlink_dir=MUJOCO_TRAINING_DATA_SYMLINK_DIR,
            versions=MUJOCO_TRAINING_DATA_TYPE_TO_SOURCE_TO_VERSION,
            cache_dir=MUJOCO_TRAINING_DATA_CACHE_DIR,
            env_prefix="MLSPACES",
            post_setup=post_setup,
            force_post_setup=force_post_setop,
        )

    return _RESOURCE_MANAGER


def install_scene_from_source_index(data_type: str, source: str, idx: int) -> dict[str, list[str]]:
    archives = list(get_resource_manager().index_lookup(data_type, source, str(idx)))

    if len(archives) == 0:
        raise ValueError(f"Unable to find archive for {source=} for {idx=}")
    elif len(archives) > 1:
        raise RuntimeError(
            f"Bug in function to search for archives ({len(archives)} values returned)."
        )

    source_to_archives = {source: archives}
    get_resource_manager().install_packages(data_type, source_to_archives)
    return source_to_archives


if __name__ == "__main__":

    def main():
        import json

        print("Setting up asset dirs and installing unindexed packages...")
        get_resource_manager(force_post_setop=True)

        samples = [
            ("franka-rby1-training-data", "FrankaPickAndPlaceOmniCamConfig", 1),
        ]

        for data_type, source, idx in samples:
            print(f"{data_type} {source} {idx}...")
            data_type_to_source_to_archives = install_scene_from_source_index(
                data_type, source, idx
            )
            print(json.dumps(data_type_to_source_to_archives, indent=2))

    main()
    print("DONE")
