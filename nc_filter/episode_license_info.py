import io
import json
import tarfile
from contextlib import contextmanager
import pickle
import base64
import re
import gzip
import threading

try:
    import h5py
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import zstandard as zstd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url, get_token
except ImportError as e:
    print(
        f"Please setup your environment with e.g. `pip install h5py requests zstandard datasets huggingface_hub`"
    )
    raise

TOKEN = get_token()
REPO = "allenai/molmobot-data"
TASK_CONFIGS = [
    "DoorOpeningDataGenConfig",
    "FrankaPickAndPlaceColorOmniCamConfig",
    "FrankaPickAndPlaceNextToOmniCamConfig",
    "FrankaPickAndPlaceOmniCamConfig",
    "FrankaPickOmniCamConfig",
    "RBY1OpenDataGenConfig",
    "RBY1PickAndPlaceDataGenConfig",
    "RBY1PickDataGenConfig",
    "FrankaPickAndPlaceOmniCamConfig_ObjectBackfill",
]


def _http_session() -> requests.Session:
    """Shared session with retries for transient DNS / connection failures."""
    retry = Retry(
        total=8,
        connect=8,
        read=4,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_thread_local = threading.local()


def _get_http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _http_session()
        _thread_local.session = session
    return session


@contextmanager
def stream_pkg(
    entry: dict,
    split: str | None,
    config_name: str,
    buffer_size: int = 8192,
    repo_id: str = REPO,
    connect_timeout: float = 30.0,
    read_timeout: float = 600.0,
):
    """
    Streams a single compressed archive (tar.zst) from within a shard using
    an HTTP Range request. Each shard contains multiple archives packed
    contiguously; the entry's offset and size identify the byte range for
    one archive. This context manager exposes an open tarfile.
    """
    if split is None:
        split = ""
    else:
        split = f"{split}_"

    url = hf_hub_url(
        repo_id=repo_id,
        filename=f"{config_name}/{split}shards/{entry['shard_id']:05d}.tar",
        repo_type="dataset",
        revision="main",
    )

    start = entry["offset"]
    end = start + entry["size"] - 1
    headers = {"Range": f"bytes={start}-{end}"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    with _get_http_session().get(
        url,
        headers=headers,
        stream=True,
        timeout=(connect_timeout, read_timeout),
    ) as response:
        response.raise_for_status()

        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(response.raw) as reader:
            buffered = io.BufferedReader(reader, buffer_size=buffer_size)

            with tarfile.open(fileobj=buffered, mode="r|") as tar:
                yield tar


def _extract_h5_buffers(entry: dict, split: str, config_name: str):
    """
    Streams the archive for the given entry and extracts only
    the h5 buffers, keyed by batch id. Also returns the scene_info string.
    """
    scene_info = None
    batch_to_h5 = {}

    with stream_pkg(entry, split, config_name) as tar:
        for member in tar:
            if not member.name.endswith(".h5"):
                continue

            batch = member.name.split("/")[1].split(".")[0].split("_batch_")[1]

            if scene_info is None:
                scene_info = f"part{entry['part']}_{member.name.split('/')[0]}"

            batch_to_h5[batch] = tar.extractfile(member).read()

    return scene_info, batch_to_h5


class Config:
    """Generic placeholder for unpickling config classes."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        self.__dict__ = state["__dict__"]

    def __repr__(self):
        return f"{self.__dict__}"


class ConfigUnpickler(pickle.Unpickler):
    """Unpickler that resolves numpy/pathlib classes normally and stubs everything else."""

    def find_class(self, module, name):
        if module.startswith(("numpy", "pathlib")):
            import importlib

            loaded = importlib.import_module(module)
            return getattr(loaded, name)

        return Config


def safe_load_config(encoded_frozen_config):
    """
    Deserializes a base64-encoded pickled config, replacing unknown
    classes with a generic Config placeholder. Returns None on failure.
    """
    try:
        return ConfigUnpickler(
            io.BytesIO(base64.b64decode(encoded_frozen_config))
        ).load()
    except Exception as e:
        print(f"Warning: config pickle could not be fully loaded: {e}")
        return None


def iterate_episode_info(entry: dict, split: str, config_name: str):
    """
    Yields per-episode obs_scene metadata (config, scene_id, traj_id, etc.)
    by streaming only the h5 data from the archive.
    """
    scene_info, batch_to_h5 = _extract_h5_buffers(entry, split, config_name)

    for batch, h5_bytes in batch_to_h5.items():
        with h5py.File(io.BytesIO(h5_bytes), "r") as f:
            if "valid_traj_mask" in f.keys():
                valid_traj_mask = f["valid_traj_mask"][()]
            else:
                traj_keys = {
                    int(key.split("traj_")[-1])
                    for key in f.keys()
                    if key.startswith("traj_")
                }
                valid_traj_mask = [
                    idx in traj_keys for idx in range(max(traj_keys) + 1)
                ]

            for eid, valid in enumerate(valid_traj_mask):
                if not valid:
                    continue

                traj = f[f"traj_{eid}"]
                obs_scene = json.loads(traj["obs_scene"][()].decode())
                obs_scene["config"] = safe_load_config(obs_scene.pop("frozen_config"))
                obs_scene["scene_id"] = scene_info
                obs_scene["traj_id"] = f"{batch}_ep{eid}"

                yield obs_scene


_CACHED_IDX_TO_ENTRY = {}
_CACHED_IDX_TO_OBJECTS = {}


def extract_number_substring(input_str: str):
    """Returns the first integer found in the string, or None."""
    match = re.search(r"\d+", input_str)
    if match:
        return int(match.group())
    return None


def resolve_scene_source(scene_idx, split: str, scene_objects):
    """
    Finds which molmospaces scene source best matches the given scene
    by comparing object sets. Results are cached across calls.
    """
    repo_id = "allenai/molmospaces"
    scene_sources = [
        "mujoco__scenes__procthor-objaverse-train__20251205",
        "mujoco__scenes__ithor__20251217",
        "mujoco__scenes__procthor-objaverse-val__20251205",
        "mujoco__scenes__procthor-10k-train__20251122",
        "mujoco__scenes__holodeck-objaverse-train__20251217",
        "mujoco__scenes__procthor-10k-val__20251217",
        "mujoco__scenes__holodeck-objaverse-val__20251217",
        "mujoco__scenes__procthor-10k-test__20251121",
    ]

    scene_objects = set(scene_objects)

    matches = []
    best_match = 0
    best_source = None

    for source in scene_sources:
        if source not in _CACHED_IDX_TO_ENTRY:
            source_name = source.split("__")[-2]
            ds = load_dataset(repo_id, name=source, split="pkgs")
            _CACHED_IDX_TO_ENTRY[source] = {}
            for entry in ds:
                stem = entry["path"].split("/")[-1].replace(source_name, "")
                if (
                    "FloorPlan" in stem
                    or "house" in stem
                    or "train" in stem
                    or "val" in stem
                ) and not ("log" in stem):
                    idx = extract_number_substring(stem)
                    if idx is not None:
                        _CACHED_IDX_TO_ENTRY[source][idx] = entry

        if scene_idx not in _CACHED_IDX_TO_ENTRY[source]:
            continue

        if source not in _CACHED_IDX_TO_OBJECTS:
            _CACHED_IDX_TO_OBJECTS[source] = {}

        if scene_idx not in _CACHED_IDX_TO_OBJECTS[source]:
            with stream_pkg(
                _CACHED_IDX_TO_ENTRY[source][scene_idx],
                None,
                source.replace("__", "/"),
                repo_id=repo_id,
            ) as tar:
                for member in tar:
                    if member.name.endswith("_metadata.json"):
                        meta_encoded = tar.extractfile(member).read()
                        meta = json.loads(meta_encoded.decode("utf-8"))
                        if "objects" in meta:
                            _CACHED_IDX_TO_OBJECTS[source][scene_idx] = meta["objects"]
                        break

        cur_objects = set(_CACHED_IDX_TO_OBJECTS[source][scene_idx].keys())
        matches.append(len(scene_objects & cur_objects) / len(scene_objects))
        if matches[-1] > best_match:
            best_match = matches[-1]
            best_source = source
            if best_match == 1.0:
                return best_source

    return best_source


_OBJECT_METADATA = None


DEFAULT_LICENSE = {
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "creator_name": "Allen Institute for AI (Ai2)",
}

ATTRIBUTION_TEMPLATE = (
    "{assets}" + f" by the {DEFAULT_LICENSE['creator_name']},"
    f" licensed under {DEFAULT_LICENSE['license'].replace('-', ' ')}."
)


def resolve_object_license(anno):
    """Builds a license dict for a single object from its annotation."""
    if anno["isObjaverse"]:
        lic = anno["license_info"]

        assert (
            "sketchfab" in lic["creator_profile_url"]
        ), f"Only sketchfab assets expected, got {lic['creator_profile_url']}"

        cur_license = {
            "asset_id": anno["assetId"],
            "source": "Sketchfab",
            "modifications": "The model has been significantly modified to reduce memory and processing requirements,"
            " including mesh decimation, convex collider extraction, and baking of visual effects via Blender scripts."
            " The provided quality may not reflect the original model.",
        }

        if lic["license"] == "by":
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY 4.0."
            )
        elif lic["license"] == "by-sa":
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY-SA 4.0."
            )
        elif lic["license"] == "cc0":
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC0-1.0."
            )
        elif lic["license"] == "by-nc":
            cur_license["commercial_use"] = False
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY-NC 4.0."
                f" Non-commercial use only."
            )
        elif lic["license"] == "by-nc-sa":
            cur_license["commercial_use"] = False
            cur_license["attribution"] = (
                f"Model by {lic['creator_display_name']} ({lic['creator_username']}), licensed under CC BY-NC-SA"
                f" 4.0. Non-commercial use only."
            )
        else:
            raise NotImplementedError(f"Got unsupported license {lic['license']}")

    else:
        cur_license = {
            "asset_id_or_archive_name": anno["assetId"],
            "source": "In-house (Ai2)",
            "attribution": ATTRIBUTION_TEMPLATE.format(assets="Model(s)"),
        }

    return cur_license


def make_episode_id(config, batch_info, episode_info):
    """Constructs a unique episode identifier."""
    return f"{config}__batch_{episode_info}__{batch_info}"


def resolve_episode_licenses(episode_id, scene_source, scene_idx, added_objects):
    """
    Builds a full license dict for an episode, combining the scene-level
    license with per-object licenses from the object metadata catalog.
    """
    repo_id = "allenai/molmospaces"
    meta_source = "mujoco__objects__objathor_metadata__20260129"

    global _OBJECT_METADATA
    if _OBJECT_METADATA is None:
        ds = load_dataset(repo_id, name=meta_source, split="pkgs")
        with stream_pkg(
            ds[0],
            None,
            meta_source.replace("__", "/"),
            repo_id=repo_id,
        ) as tar:
            for member in tar:
                if member.name.endswith(".json.gz"):
                    meta_bytes = tar.extractfile(member).read()
                    with gzip.open(io.BytesIO(meta_bytes), "rt") as f:
                        _OBJECT_METADATA = json.load(f)
                    break

    scene_objects = _CACHED_IDX_TO_OBJECTS[scene_source][scene_idx]
    includes = []
    for obj_name, obj_info in scene_objects.items():
        asset_id = obj_info["asset_id"]
        if asset_id not in _OBJECT_METADATA:
            continue
        meta = _OBJECT_METADATA[asset_id]
        obj_lic = resolve_object_license(meta)
        obj_lic["id_in_scene"] = obj_name
        includes.append(obj_lic)

    for obj_name, path in added_objects.items():
        asset_id = path.name.rsplit(".", 1)[0]
        meta = _OBJECT_METADATA[asset_id]
        obj_lic = resolve_object_license(meta)
        obj_lic["id_in_scene"] = obj_name
        includes.append(obj_lic)

    scene_license = {
        "episode_id": episode_id,
        **DEFAULT_LICENSE,
        "attribution": ATTRIBUTION_TEMPLATE.format(assets="Scene"),
        "scope": "Scene composition, layout, non-object-specific textures, and metadata.",
        "relationship_to_assets": "collection",
        "asset_licenses": "Assets are independently licensed; see assets info below for details.",
        "license_determination": "Scenes are collections referencing independently licensed assets;"
        f" {DEFAULT_LICENSE['license']} applies only to scene composition, layout, and metadata.",
    }
    if includes:
        scene_license["assets"] = includes
    return scene_license


def list_configs(split: str):
    """Print every config name in the dataset together with its entry count."""
    for config in TASK_CONFIGS:
        ds = load_dataset(REPO, name=config, split=f"{split}_pkgs")
        print(f"{config}: {len(ds)} entries")
    print("\nUse --config CONFIG --index INDEX to show licenses for a specific entry.")


def licenses_for_entry(config_name: str, split: str, index: int):
    """
    Print license information for every episode contained in the
    dataset entry at the given index within the given config.
    """
    ds = load_dataset(REPO, name=config_name, split=f"{split}_pkgs")
    if index < 0 or index >= len(ds):
        raise IndexError(
            f"Index {index} out of range for config '{config_name}' "
            f"(has {len(ds)} entries, valid range 0..{len(ds) - 1})"
        )
    entry = ds[index]

    print(f"Config: {config_name} {split}")
    print(f"Entry index: {index}")
    print(f"Path: {entry['path']}")
    print(
        f"Shard: {entry['shard_id']}, offset: {entry['offset']}, size: {entry['size']}"
    )
    print()

    episode_count = 0
    for obs_scene in iterate_episode_info(entry, split, config_name):
        scene_id = obs_scene["scene_id"]
        scene_idx = extract_number_substring(scene_id.split("_")[-1])
        traj_id = obs_scene["traj_id"]
        added_objects = obs_scene["config"].task_config.added_objects
        scene_objects = sorted(
            set(obs_scene["config"].task_config.object_poses.keys())
            - set(added_objects.keys())
        )

        scene_source = resolve_scene_source(scene_idx, split, scene_objects)
        episode_id = make_episode_id(config_name, scene_id, traj_id)
        license_info = resolve_episode_licenses(
            episode_id, scene_source, scene_idx, added_objects
        )

        print(f"{json.dumps(license_info, indent=2)}\n")
        episode_count += 1

    print(f"Total episodes in entry: {episode_count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="License information for allenai/molmobot-data episodes."
        " With no arguments, lists all configs and their entry counts."
    )
    parser.add_argument(
        "--config",
        choices=TASK_CONFIGS,
        help="Config name (task configuration).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split (one of `train`or `val`)",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Dataset entry index within the config.",
    )

    args = parser.parse_args()

    if args.config is not None and args.index is not None:
        licenses_for_entry(args.config, args.split, args.index)
    elif args.config is not None or args.index is not None:
        parser.error("Both --config and --index are required together.")
    else:
        list_configs(args.split)


if __name__ == "__main__":
    main()
