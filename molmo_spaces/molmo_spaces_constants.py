"""
Constants and paths for the MolmoSpaces project. Paths should be provided as Path objects.

Overwrite in the environment with e.g.:
    MLSPACES_ASSETS_DIR=/Users/username/mlspaces_resources mjpython scripts/...
"""

import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import compress_json

from molmo_spaces.utils.resource_manager_setup_utils import setup_resource_manager, str2bool


def single_thread_environment():
    print(f"Setting single thread environment for proc {os.getpid()}")
    try:
        import torch

        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

        if os.environ.get("OMP_NUM_THREADS") != "1":
            os.environ["OMP_NUM_THREADS"] = "1"
        if os.environ.get("MKL_NUM_THREADS") != "1":
            os.environ["MKL_NUM_THREADS"] = "1"
    except Exception:
        pass


if str2bool(os.environ.get("MLSPACES_SINGLE_THREAD_PROCS", "False")):
    single_thread_environment()

ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR = Path(__file__).resolve().parent.parent

_DATA_CACHE_DEFAULT = Path("~/.cache/molmo-spaces-resources").expanduser()
DATA_CACHE_DIR = Path(os.environ.get("MLSPACES_CACHE_DIR", _DATA_CACHE_DEFAULT))

ASSETS_DIR = Path(
    os.environ.get("MLSPACES_ASSETS_DIR", ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "assets")
)
ROBOTS_DIR = ASSETS_DIR / "robots"
OBJAVERSE_ASSETS_DIR = Path(
    os.environ.get("MLSPACES_OBJAVERSE_ASSETS_DIR", ASSETS_DIR / "objects" / "objaverse")
)

PINNED_ASSETS_FILE = (
    Path(os.environ["MLSPACES_PINNED_ASSETS_FILE"])
    if "MLSPACES_PINNED_ASSETS_FILE" in os.environ
    else None
)

DATA_TYPE_TO_SOURCE_TO_VERSION = dict(
    robots={
        "rby1": "20251224",
        "rby1m": "20251224",
        "franka_droid": "20260127",
        "franka_cap": "20260213",
        "floating_rum": "20251110",
        "floating_robotiq": "20260208_retry4",
    },
    scenes={
        "ithor": "20251217",
        "refs": "20250923",
        "procthor-10k-train": "20251122",
        "procthor-10k-val": "20251217",
        "procthor-10k-test": "20251121",
        "holodeck-objaverse-train": "20251217",
        "holodeck-objaverse-val": "20251217",
        "procthor-objaverse-train": "20251205",
        "procthor-objaverse-val": "20251205",
    },
    objects={
        "thor": "20251117",
        "objaverse": "20260131",
        "objathor_metadata": "20260129",
    },
    grasps={
        "droid": "20251116",
        "droid_objaverse": "20251218",
    },
    test_data={
        "franka_pick": "20260202",
        "franka_pick_and_place": "20260202",
        "rby1_door_opening": "20250107",
        "rum_open_close": "20260202",
        "rum_pick": "20260202",
        "test_randomized_data": "20251209",
        "thormap": "20251209",
    },
    benchmarks={
        "molmospaces-bench-v1": "20260210",
    },
)

_RESOURCE_MANAGER = None


def get_resource_manager():
    global _RESOURCE_MANAGER
    if _RESOURCE_MANAGER is None:
        _RESOURCE_MANAGER = setup_resource_manager(
            "mujoco-thor-resources",
            ASSETS_DIR,
            DATA_TYPE_TO_SOURCE_TO_VERSION,
            DATA_CACHE_DIR,
        )
    return _RESOURCE_MANAGER


def _merge_dicts(dict1: dict, dict2: dict):
    """
    Merges dict2 into dict1, only overwriting leaf values.
    """
    for key, value in dict2.items():
        if isinstance(value, dict):
            if key not in dict1:
                dict1[key] = {}
            _merge_dicts(dict1[key], value)
        else:
            dict1[key] = value


if PINNED_ASSETS_FILE:
    assert PINNED_ASSETS_FILE.is_file(), f"Could not find pinned assets file: {PINNED_ASSETS_FILE}"
    with open(PINNED_ASSETS_FILE, "r") as f:
        pinned_assets = json.load(f)
        print(f"Pinning assets from {PINNED_ASSETS_FILE}:\n{json.dumps(pinned_assets, indent=2)}")
        _merge_dicts(DATA_TYPE_TO_SOURCE_TO_VERSION, pinned_assets)


# ------------------------------
# Scene dataset helpers
# ------------------------------

# Simple in-memory cache for dataset to split to index maps
_DATASET_INDEX_CACHE: dict[str, dict[str, dict]] = {}


_SCENES_ROOT = None

# Determine a root for scene resources:
# Prefer the scenes dir under assets_dir if it exists (should always exist once installed);
# otherwise use env var;
# otherwise use builtin assets/scenes.


def get_scenes_root():
    global _SCENES_ROOT
    if _SCENES_ROOT is None:
        # Ensure scenes dir under asset root exists
        get_resource_manager()

        if (ASSETS_DIR / "scenes").exists():
            _SCENES_ROOT = ASSETS_DIR / "scenes"

        else:
            _SCENES_ROOT = Path(
                os.environ.get(
                    "MLSPACES_SCENES_ROOT",
                    ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "assets" / "scenes",
                )
            )
        print(f"Using SCENES_ROOT: {_SCENES_ROOT}")

    return _SCENES_ROOT


_ASSET_ID_TO_OBJECT_TYPE = None


def get_asset_id_to_object_type():
    global _ASSET_ID_TO_OBJECT_TYPE
    if _ASSET_ID_TO_OBJECT_TYPE is None:
        ref_file = get_scenes_root() / "refs" / "asset_id_to_object_type.json"
        try:
            _ASSET_ID_TO_OBJECT_TYPE = compress_json.load(str(ref_file))
        except Exception as e:
            print(f"Warning: Failed to load asset_id_to_object_type.json: {e}")
            _ASSET_ID_TO_OBJECT_TYPE = {}

    return _ASSET_ID_TO_OBJECT_TYPE


_OBJECT_TYPE_TO_ASSET_IDS = None


def get_object_type_to_asset_ids():
    global _OBJECT_TYPE_TO_ASSET_IDS
    if _OBJECT_TYPE_TO_ASSET_IDS is None:
        # Group asset IDs by object type
        _OBJECT_TYPE_TO_ASSET_IDS = defaultdict(list)
        for asset_id, obj_type in get_asset_id_to_object_type().items():
            _OBJECT_TYPE_TO_ASSET_IDS[obj_type].append(asset_id)
        _OBJECT_TYPE_TO_ASSET_IDS = dict(_OBJECT_TYPE_TO_ASSET_IDS)  # Convert back to regular dict

    return _OBJECT_TYPE_TO_ASSET_IDS


def _build_scene_index_map_procthor(dataset_root: Path, split: str) -> dict:
    """Build mapping of available scene files under dataset_root.

    Returns:
      {"train": {idx: {variant: path_or_None}}, "val": {idx: {variant: path_or_None}}}
    Looks for files matching patterns:
    - "<split>_<index>.xml" (base variant, no suffix)
    - "<split>_<index>_ceiling.xml" (ceiling variant)
    - "<split>_<index>_map.png" (map variant, PNG file)
    Missing indices up to the maximum discovered index are included with value None for each variant.
    """
    index_map: dict[str, dict[int, dict[str, str | None]]] = {"train": {}, "val": {}, "test": {}}

    # Known variants to track
    known_variants = {"ceiling", "map", "base"}

    prefix = f"{split}_"
    present_indices: set[int] = set()
    variant_files: dict[tuple[int, str], str] = {}

    try:
        entries = itertools.chain.from_iterable(
            iter(
                get_resource_manager()
                .scene_root_and_archive_paths(dataset_root.name)["archive_to_relative_paths"]
                .values()
            )
        )
    except FileNotFoundError:
        return index_map

    # Collect present files matching variant patterns
    for fn in entries:
        fn = str(fn)
        if not fn.startswith(prefix):
            continue

        # Handle different file types
        if fn.endswith(".xml"):
            # Extract the part between prefix and .xml
            stem = fn[len(prefix) : -len(".xml")]

            # Check for ceiling pattern: train_0_ceiling.xml
            if stem.endswith("_ceiling"):
                index_str = stem[: -len("_ceiling")]
                if index_str.isdigit():
                    idx = int(index_str)
                    present_indices.add(idx)
                    variant_files[(idx, "ceiling")] = str(dataset_root / fn)
                continue

            # Check for base pattern: train_0.xml (no suffix after index)
            if stem.isdigit():
                idx = int(stem)
                present_indices.add(idx)
                variant_files[(idx, "base")] = str(dataset_root / fn)
                continue

        elif fn.endswith(".png"):
            # Handle map files: train_0_map.png
            stem = fn[len(prefix) : -len(".png")]
            if stem.endswith("_map"):
                index_str = stem[: -len("_map")]
                if index_str.isdigit():
                    idx = int(index_str)
                    present_indices.add(idx)
                    variant_files[(idx, "map")] = str(dataset_root / fn)
                continue

    # Build the nested structure
    if present_indices:
        max_idx = max(present_indices)
        for i in range(max_idx + 1):
            index_map[split][i] = {}
            for variant in known_variants:
                index_map[split][i][variant] = variant_files.get((i, variant))

    # Sort keys deterministically
    index_map[split] = {k: index_map[split][k] for k in sorted(index_map[split].keys())}

    return index_map


def _build_scene_index_map_ithor(dataset_root: Path) -> dict:
    """Build mapping of available scene files under dataset_root.

    Returns:
      {"train": {idx: path_or_None}, "val": {idx: path_or_None}}
    Missing indices up to the maximum discovered index are included with value None.
    """
    index_map: dict[str, dict[int, str | None]] = {"train": {}, "val": {}, "test": {}}

    try:
        entries = itertools.chain.from_iterable(
            iter(
                get_resource_manager()
                .scene_root_and_archive_paths(dataset_root.name)["archive_to_relative_paths"]
                .values()
            )
        )
    except FileNotFoundError:
        return index_map

    prefix = "FloorPlan"
    suffix = "_physics.xml"
    indices = {}

    # Collect present files
    for fn in entries:
        fn_str = str(fn.name)
        if not fn_str.startswith(prefix) or not fn_str.endswith(suffix):
            continue
        stem = fn_str[len(prefix) : -len(suffix)]
        # Accept only pure numeric stem (filters out other variants)
        if not stem.isdigit():
            continue
        idx = int(stem)
        indices[idx] = dataset_root / fn_str

    # Fill missing indices with None up to max
    if indices:
        max_idx = max(indices.keys())
        for i in range(max_idx + 1):
            index_in_scene_type = i % 100
            index_map["train"][i] = None
            index_map["val"][i] = None
            index_map["test"][i] = None
            if i in indices:
                if index_in_scene_type <= 12:
                    index_map["train"][i] = indices[i]
                elif index_in_scene_type <= 24:
                    index_map["val"][i] = indices[i]
                elif index_in_scene_type <= 30:
                    index_map["test"][i] = indices[i]
                else:
                    raise ValueError(
                        f"Unknown index type for ithor scenes: {index_in_scene_type} from index {i}"
                    )
    # Sort keys deterministically
    for split in ("train", "val", "test"):
        index_map[split] = {k: index_map[split][k] for k in sorted(index_map[split].keys())}

    return index_map


def get_scenes(
    dataset_name: str, split: str = "train", return_version: bool = False
) -> dict | tuple[dict, str | None]:
    names2functions = {
        "ithor": get_ithor_houses,
        "procthor-10k": get_procthor_10k_houses,
        "procthor-100k-debug": get_procthor_objaverse_houses,
        "procthor-objaverse-debug": get_procthor_objaverse_debug_houses,
        "procthor-objaverse": get_procthor_objaverse_houses,
        "holodeck-objaverse": get_holodeck_objaverse_houses,
    }
    if dataset_name not in names2functions:
        raise ValueError(
            f"dataset_name was {dataset_name}, must be one of {names2functions.keys()}"
        )
    index_map = names2functions[dataset_name](split=split)

    if not return_version:
        return index_map
    else:
        if dataset_name.startswith("procthor-10k"):
            version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][f"procthor-10k-{split}"]
        elif (
            dataset_name.startswith("procthor-objaverse")
            and dataset_name != "procthor-objaverse-debug"
        ):
            version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][f"procthor-objaverse-{split}"]
        elif dataset_name.startswith("holodeck-objaverse"):
            version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][f"holodeck-objaverse-{split}"]

        elif dataset_name not in DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"]:
            print(f"WARNING: Missing source for {dataset_name}")
            version = None
        else:
            version = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"][dataset_name]
        return index_map, version


def check_in_cache(cache_key, split):
    return cache_key in _DATASET_INDEX_CACHE and split in _DATASET_INDEX_CACHE[cache_key]


def populate_cache(cache_key, split, index_map):
    if cache_key not in _DATASET_INDEX_CACHE:
        _DATASET_INDEX_CACHE[cache_key] = {}
    # If we got val and test (we should), add those to the cache
    for rsplit in ["train", "val", "test"]:
        num_houses = len([idx for idx in index_map[rsplit] if index_map[rsplit][idx] is not None])
        if num_houses > 0:
            _DATASET_INDEX_CACHE[cache_key][rsplit] = index_map[rsplit]

    # If no houses available for split, make sure to still set it up in the cache
    _DATASET_INDEX_CACHE[cache_key][split] = index_map[split]


def get_ithor_houses(split) -> dict:
    """Return {split: {index: xml_path_or_None}} for iTHOR houses."""
    cache_key = "ithor"

    if check_in_cache(cache_key, split):
        return _DATASET_INDEX_CACHE[cache_key]

    houses_dir = get_scenes_root() / "ithor"
    index_map = _build_scene_index_map_ithor(houses_dir)

    populate_cache(cache_key, split, index_map)

    return index_map


def get_procthor_10k_houses(split) -> dict:
    """Return {split: {index: xml_path_or_None}} for ProcTHOR-10k houses."""
    cache_key = "procthor-10k"
    if split == "train":
        location = "procthor-10k-train"
    elif split == "val":
        location = "procthor-10k-val"
    elif split == "test":
        location = "procthor-10k-test"
    else:
        raise ValueError

    if check_in_cache(cache_key, split):
        return _DATASET_INDEX_CACHE[cache_key]

    houses_dir = get_scenes_root() / location
    index_map = _build_scene_index_map_procthor(houses_dir, split)

    populate_cache(cache_key, split, index_map)

    return index_map


def get_procthor_objaverse_debug_houses(split) -> dict:
    """Return {split: {index: xml_path_or_None}} for ProcTHOR Objaverse houses."""
    cache_key = "procthor-objaverse-debug"

    if check_in_cache(cache_key, split):
        return _DATASET_INDEX_CACHE[cache_key]

    houses_dir = get_scenes_root() / "procthor-objaverse-debug"
    index_map = _build_scene_index_map_procthor(houses_dir, split)

    populate_cache(cache_key, split, index_map)

    return index_map


def get_procthor_objaverse_houses(split) -> dict:
    """Return {split: {index: xml_path_or_None}} for ProcTHOR Objaverse houses."""
    cache_key = "procthor-objaverse"

    if check_in_cache(cache_key, split):
        return _DATASET_INDEX_CACHE[cache_key]

    houses_dir = get_scenes_root() / f"procthor-objaverse-{split}"
    index_map = _build_scene_index_map_procthor(houses_dir, split)

    populate_cache(cache_key, split, index_map)

    return index_map


def get_holodeck_objaverse_houses(split) -> dict:
    """Return {split: {index: xml_path_or_None}} for ProcTHOR Objaverse houses."""
    cache_key = "holodeck-objaverse"

    if check_in_cache(cache_key, split):
        return _DATASET_INDEX_CACHE[cache_key]

    houses_dir = get_scenes_root() / f"holodeck-objaverse-{split}"
    index_map = _build_scene_index_map_procthor(houses_dir, split)

    populate_cache(cache_key, split, index_map)

    return index_map


def get_robot_paths() -> dict[str, Path]:
    """Return {robot_name: Path} for all available robot files."""
    robot_paths = {}
    for robot_name in os.listdir(ROBOTS_DIR):
        robot_paths[robot_name] = ROBOTS_DIR / robot_name
    return robot_paths


def get_robot_path(robot_name) -> Path:
    return ROBOTS_DIR / robot_name


def print_license_info(data_type, data_source, asset_or_tar_id):
    from molmo_spaces.utils.license_utils import resolve_license

    def get_identifiers():
        archives = get_resource_manager().tries(data_source, data_type).keys()
        return [
            archive.replace(f"{data_source}_", "").replace(".tar.zst", "") for archive in archives
        ]

    if asset_or_tar_id == "--list_all":
        print(f"Possible identifiers: {sorted(get_identifiers())}")
        return

    try:
        license_info = resolve_license(data_type, data_source, asset_or_tar_id)
        print(json.dumps(license_info, indent=2))
    except ValueError as e:
        import random

        archives = get_identifiers()
        formatted = "\n".join(sorted(random.choices(archives, k=min(len(archives), 10))))
        print(e)
        print(f"Possible identifiers:\n{formatted}{'...' if len(archives) > 10 else ''}")


if __name__ == "__main__":

    def setup_assets():
        print("Setting up assets for MuJoCo...")
        get_resource_manager()

        print("DONE")

    setup_assets()
