import os
from pathlib import Path
from typing import Iterator, Collection, Callable

from pxr import Sdf, Usd, Ar

from molmospaces_resources import setup_resource_manager, R2RemoteStorage
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR
from molmospaces_resources import ResourceManager, SourceInfo

ISAAC_ASSETS_DIR = ASSETS_DIR.parent / "isaac-assets"
ISAAC_DATA_CACHE_DIR = DATA_CACHE_DIR.parent / "isaac-thor-resources"

ISAAC_DATA_TYPE_TO_SOURCE_TO_VERSION = {
    "objects": {
        "thor": "20260128",
        "objaverse": "20260128",
    },
    "scenes": {
        "ithor": "20260121",
        "procthor-10k-train": "20260128",
        "procthor-10k-val": "20260128",
        "procthor-10k-test": "20260128",
        "procthor-objaverse-train": "20260128",
        "procthor-objaverse-val": "20260128",
        "holodeck-objaverse-train": "20260128",
        "holodeck-objaverse-val": "20260128",
    },
}


_RESOURCE_MANAGER: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    global _RESOURCE_MANAGER
    if _RESOURCE_MANAGER is None:
        _RESOURCE_MANAGER = setup_resource_manager(
            R2RemoteStorage("isaac-thor-resources"),
            symlink_dir=ISAAC_ASSETS_DIR,
            versions=ISAAC_DATA_TYPE_TO_SOURCE_TO_VERSION,
            cache_dir=ISAAC_DATA_CACHE_DIR,
        )
    return _RESOURCE_MANAGER


def archive_resolver(
    source: str, identifier: str, fn: Callable[[str, str, str], set[str]], data_type: str = "scenes"
) -> set[str]:
    archives = fn(source, identifier, data_type)

    if len(archives) == 0:
        raise ValueError(f"Unable to find archive for {source=} with {identifier=}")
    elif len(archives) > 1:
        raise RuntimeError(
            f"Bug in function to search for archives with numbers ({len(archives)} values returned)."
        )

    return archives


def archives_for_source_index(source: str, idx: int, data_type: str = "scenes") -> set[str]:
    fn = get_resource_manager().archives_with_number
    return archive_resolver(source, str(idx), fn, data_type)


def install_scene_from_source_index(source: str, idx: int) -> dict[str, set[str]]:
    source_to_archives = {source: archives_for_source_index(source, idx)}
    get_resource_manager().install_scenes(source_to_archives)
    return source_to_archives


def scene_path_resolve(
    source: str, idx: int, source_to_archives: dict[str, Collection[str]], variant: str = ""
) -> Path:
    if "procthor" in source or "holodeck" in source:
        fn = procthor_resolver
    elif "ithor" in source:
        fn = ithor_resolver
    else:
        raise NotImplementedError(f"Missing implementation for {source}")

    archive = list(source_to_archives[source])[0]
    scene_info = get_resource_manager().scene_root_and_archive_paths(source)
    modalities = scene_info["archive_to_relative_paths"][archive]
    return fn(source, idx, scene_info, modalities, variant)


def ithor_resolver(
    source: str, idx: int, scene_info: SourceInfo, modalities: list[Path], variant: str = ""
) -> Path:
    return scene_info["root_dir"] / modalities[0]


def procthor_resolver(
    source: str, idx: int, scene_info: SourceInfo, modalities: list[Path], variant: str = ""
) -> Path:
    split = source.split("-")[-1]
    target = f"{split}_{idx}" + variant
    assert Path(target) in modalities, f"Missing {target=} with available modalities {modalities}"
    return scene_info["root_dir"] / target


def find_object_paths(
    source: str,
    idx: int,
    source_to_archives: dict[str, set[str]],
    variant: str = "",
    exclude_thor=True,
) -> Iterator[tuple[str, Path]]:
    scene_path = scene_path_resolve(source, idx, source_to_archives, variant)

    layer = Sdf.Layer.FindOrOpen(str(scene_path / "Payload" / "Geometry.usda"))

    external_assets = sorted(set(layer.externalReferences))
    if exclude_thor:
        external_assets = [asset for asset in external_assets if "/objects/thor/" not in asset]

    external_assets = [asset for asset in external_assets if "../objects/" in asset]

    for file_path in external_assets:
        # Objects are globally linked from the cache
        full_path = (scene_path / "Payload" / file_path).resolve()
        source = (full_path.relative_to(get_resource_manager().cache_dir / "objects")).parts[0]
        rel_asset = full_path.relative_to(get_resource_manager().object_dirs[source])
        yield source, rel_asset


def install_objects_for_scene(
    source: str,
    idx: int,
    scene_source_to_archive: dict[str, set[str]],
    variant: str = "",
    exclude_thor: bool = True,
) -> dict[str, list[str]]:
    if "objaverse" not in ISAAC_DATA_TYPE_TO_SOURCE_TO_VERSION["objects"]:
        return {}

    source_to_archives = {}

    for source, rel_asset in find_object_paths(
        source, idx, scene_source_to_archive, variant, exclude_thor=exclude_thor
    ):
        archives = get_resource_manager().archives_for_paths(
            source, [rel_asset], data_type="objects"
        )
        if source not in source_to_archives:
            source_to_archives[source] = archives
        else:
            source_to_archives[source].extend(archives)

    source_to_archives: dict[str, list[str]] = {
        source: list(set(archives)) for source, archives in source_to_archives.items()
    }

    get_resource_manager().install_objects(source_to_archives)

    return source_to_archives


def flatten_usd_for_blender(
    input_usd_path: Path | str,
    output_usd_path: Path | str,
    resolver_search_paths: list[str | Path] | None = None,
    strip_physics: bool = True,
    verbose: bool = False,
):
    """
    Flattens a USD scene into a single self-contained USD suitable for Blender.

    Parameters:
        input_usd_path: path to the scene.usda/.usdc
        output_usd_path: path to write the flattened USD
        resolver_search_paths: list of directories where USD should look for external assets
        strip_physics: remove physics/collision prims (optional)
    """

    # Configure the resolver before loading anything
    if resolver_search_paths:
        resolver = Ar.GetResolver()
        current_paths = list(resolver.GetSearchPaths())
        # Insert your asset paths at highest priority
        for path in resolver_search_paths:
            path = os.path.abspath(str(path))
            if path not in current_paths:
                current_paths.insert(0, path)
        resolver.SetSearchPaths(current_paths)
        if verbose:
            print("USD resolver search paths:", resolver.GetSearchPaths())

    # Open the stage
    stage = Usd.Stage.Open(str(input_usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {input_usd_path}")

    # Load all payloads so all assets are visible
    stage.Load()

    # Optionally strip physics / collision prims
    if strip_physics:
        to_remove = []
        for prim in stage.Traverse():
            if prim.GetTypeName().startswith("Physics") or "collision" in prim.GetName().lower():
                to_remove.append(prim.GetPath())
        for path in to_remove:
            stage.RemovePrim(path)

    # Flatten the stage into a single layer
    flattened_layer = stage.Flatten()

    # Export the flattened layer
    Path(output_usd_path).parent.mkdir(parents=True, exist_ok=True)
    flattened_layer.Export(str(output_usd_path))
    if verbose:
        print(f"Flattened USD written to: {output_usd_path}")


def install_scene_with_objects(
    source: str, idx: int, variant: str = "", exclude_thor: bool = True
) -> dict[str, dict[str, Collection[str]]]:
    scene_source_to_archive = install_scene_from_source_index(source, idx)
    objects_source_to_archives = install_objects_for_scene(
        source, idx, scene_source_to_archive, variant, exclude_thor
    )
    return {"scenes": scene_source_to_archive, "objects": objects_source_to_archives}


def flatten_from_install_info(
    source: str,
    idx: int,
    data_type_to_source_to_archives: dict[str, dict[str, Collection[str]]],
    output_path: str | Path,
    variant: str = "",
):
    scene_dir = scene_path_resolve(source, idx, data_type_to_source_to_archives["scenes"], variant)
    flatten_usd_for_blender(
        str(scene_dir / "scene.usda"),
        str(output_path),
    )


def archives_for_object_uid(source: str, uid: str, data_type: str = "objects") -> set[str]:
    fn = get_resource_manager().archives_with_substring
    return archive_resolver(source, uid, fn, data_type)


def install_objaverse_uid(uid: str, source: str = "objaverse") -> dict[str, set[str]]:
    source_to_archives = {source: archives_for_object_uid(source, uid)}
    get_resource_manager().install_objects(source_to_archives)
    return source_to_archives


if __name__ == "__main__":
    samples = [
        ("ithor", 1, ""),
        ("procthor-10k-train", 0, ""),
        ("procthor-10k-val", 0, ""),
        ("procthor-10k-test", 0, ""),
        ("procthor-objaverse-train", 0, ""),
        ("procthor-objaverse-val", 0, ""),
        ("holodeck-objaverse-train", 0, ""),
        ("holodeck-objaverse-val", 0, ""),
    ]

    for source, idx, variant in samples:
        print(f"{source} {idx}{variant}...")
        output_path = Path(f"~/Desktop/flat_scenes/{source}_{idx}{variant}.usdc").expanduser()
        data_type_to_source_to_archives = install_scene_with_objects(source, idx, variant)
        flatten_from_install_info(
            source, idx, data_type_to_source_to_archives, output_path, variant
        )

    print("Install single objaverse object")
    source_to_archives = install_objaverse_uid("1d702f4c5ec44f37bfa6ebafca3275a5")

    print("DONE")
