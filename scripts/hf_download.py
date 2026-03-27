"""Download molmospaces data.
The repository contains 2GB tar shards of ``.tar.zst`` archives,
with each archive's shard id / byte offset / size recorded in a
parquet arrow table. The shards and two metadata files
(a manifest JSON and an archive tries compressed JSON) are also stored
under a `data_source_dir` such as ``mujoco/objects/thor/20251117``.
This script auto-discovers every available
source (by scanning for ``pkgs`` parquet files), downloads
the metadata and shard tars, and decompresses + extracts each inner
``.tar.zst`` on the fly into a local directory tree.
"""

import io
import os
import tarfile
from pathlib import Path
import json

import datasets
import zstandard as zstd
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()

HF_REPO_ID = "allenai/molmospaces"
HF_REPO_TYPE = "dataset"

MANIFEST_NAME = "mjthor_resource_file_to_size_mb.json"
TRIES_NAME = "mjthor_resources_combined_meta.json.gz"
EXTRACTED_MANIFEST_NAME = "mjthor_data_type_to_source_to_versions.json"

VALID_SOURCES = ("mujoco", "isaac", "all")

# ------------------------------------------------------------------
# Meta files
# ------------------------------------------------------------------


def download_meta(
    data_source_dir: str,
    target_dir: str,
    revision: str = "main",
) -> dict[str, str]:
    """Download the manifest and tries files.
    The files are fetched via ``hf_hub_download`` (which caches them
    locally) and then copied into ``<target_dir>`` so the output
    directory is self-contained.
    Parameters
    ----------
    data_source_dir:
        Repository-relative directory, e.g.
        ``"mujoco/objects/thor/20251117"``.
    target_dir:
        Local directory; meta files are placed in ``<target_dir>/``.
    revision:
        HF branch or revision.
    Returns
    -------
    dict[str, str]
        Mapping of filename to local path for each successfully
        downloaded meta file.  Files that do not exist in the repo are
        silently skipped.
    """
    downloaded: dict[str, str] = {}
    meta_dir = target_dir
    os.makedirs(meta_dir, exist_ok=True)

    for name in [MANIFEST_NAME, TRIES_NAME]:
        path_in_repo = f"{data_source_dir}/{name}"
        local_path = os.path.join(meta_dir, name)
        try:
            cached = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=path_in_repo,
                repo_type=HF_REPO_TYPE,
                revision=revision,
            )
            # Copy from HF cache into target_dir so the user has a
            # self-contained output directory.
            _copy_file(cached, local_path)
            downloaded[name] = local_path
            print(f"Downloaded meta: {name}")
        except Exception as e:
            print(f"Skipped meta (not found): {name}  ({e})")

    return downloaded


# ------------------------------------------------------------------
# Arrow table
# ------------------------------------------------------------------


def load_arrow_table(data_source_dir: str, revision: str = "main") -> Dataset:
    """Load the arrow table that describes the contents of each shard.
    Each row in the returned Dataset contains:
    - **path** (str): relative path of the ``.tar.zst`` member inside
      the shard tar.
    - **shard_id** (int): which shard tar the member lives in.
    - **offset** (int): byte offset of the member's data within the
      shard tar.
    - **size** (int): size in bytes of the ``.tar.zst`` payload.
    Parameters
    ----------
    data_source_dir:
        Repository-relative directory, e.g.
        ``"mujoco/objects/thor/20251117"``.
    revision:
        HF branch or revision.
    Returns
    -------
    Dataset
        Arrow table with one row per ``.tar.zst`` archive.
    Raises
    ------
    FileNotFoundError
        If no ``pkgs`` parquet files are found under *data_source_dir*.
    """
    api = HfApi()

    # Parquet configs live under a __-joined path.
    parquet_dir = data_source_dir.replace("/", "__")
    repo_items = api.list_repo_tree(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo=parquet_dir,
        revision=revision,
    )
    parquet_paths = sorted(
        item.path
        for item in repo_items
        if getattr(item, "path", "").endswith(".parquet") and "pkgs" in os.path.basename(item.path)
    )

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for split 'pkgs' under {data_source_dir}")

    # Download each parquet shard from the HF cache.
    local_paths = [
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=pf,
            repo_type=HF_REPO_TYPE,
            revision=revision,
        )
        for pf in parquet_paths
    ]

    ds = Dataset.from_parquet(local_paths if len(local_paths) > 1 else local_paths[0])
    return ds


# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------


def _copy_file(src: str, dst: str) -> None:
    """Copy *src* to *dst*, creating parent dirs as needed."""
    import shutil

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _extract_tar_zst_bytes(data: bytes, output_dir: str) -> None:
    """Decompress a .tar.zst payload in-memory and stream-extract to *output_dir*."""
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(data)) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(path=output_dir)


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024**4:
        return f"{size_bytes / 1024**4:.2f} TiB"
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def estimate_sizes(data_source_dir: str, revision: str = "main") -> tuple[int, int | None]:
    """Return (download_bytes, extracted_bytes_or_None) for a source.
    *download_bytes* is the sum of compressed .tar.zst sizes (which equals
    both the network transfer and the HF cache disk usage).
    *extracted_bytes* is the sum of ``inflated_size`` if available in the
    arrow table, otherwise ``None``.
    """
    ds = load_arrow_table(data_source_dir, revision=revision)
    rows = ds.to_list()
    download = sum(r["size"] for r in rows)
    if rows and "inflated_size" in rows[0]:
        extracted = sum(r["inflated_size"] for r in rows)
    else:
        extracted = None
    return download, extracted


# ------------------------------------------------------------------
# Main download-and-extract
# ------------------------------------------------------------------


def download_and_extract(
    data_source_dir: str,
    target_dir: str,
    revision: str = "main",
    versioned: bool = True,
    confirm: bool = True,
) -> None:
    """Download and extract a single data source.
    This is the main entry point for downloading one ``data_source_dir``.  It
    performs three steps:
    1. **Metadata** -- download the manifest JSON and tries JSON file
       into ``<target_dir>/``.
    2. **Arrow table** -- load the parquet table to discover which
       shards exist and how many archives they contain.
    3. **Shards** -- for each shard tar, download it (via
       ``hf_hub_download``, which caches locally), iterate through its
       members, decompress each ``.tar.zst`` payload with zstandard, and
       stream-extract the inner tar into *target_dir*.
    Parameters
    ----------
    data_source_dir:
        Repository-relative directory, e.g.
        ``"mujoco/objects/thor/20251117"``.
    target_dir:
        Local directory to extract all data into.
    revision:
        HF branch or revision.
    versioned:
        Whether to include the version string in the extracted data paths
    """
    os.makedirs(target_dir, exist_ok=True)

    bucket, data_type, data_source, version = Path(data_source_dir).parts
    base_dir = Path().joinpath(*Path(target_dir).parts[:-4])
    extracted_manifest_path = base_dir / bucket / EXTRACTED_MANIFEST_NAME
    print("Using", extracted_manifest_path, "for", data_source_dir)

    manifest = {}
    if os.path.isfile(extracted_manifest_path):
        with open(extracted_manifest_path, "r") as f:
            manifest = json.load(f)
    if (
        data_type in manifest
        and data_source in manifest[data_type]
        and version in manifest[data_type][data_source]
    ):
        print(f"\nPre-extracted archives for {target_dir}")
        return

    if not versioned:
        target_dir = Path(target_dir).parent

    # 1. Meta files ---------------------------------------------------
    print("=== Downloading metadata ===")
    download_meta(data_source_dir, target_dir, revision=revision)

    # 2. Arrow table --------------------------------------------------
    print("\n=== Loading arrow table ===")
    ds = load_arrow_table(data_source_dir, revision=revision)
    rows = ds.to_list()
    print(f"Arrow table has {len(rows)} entries")

    # Group entries by shard so we know which shards to fetch.
    shard_ids: set[int] = set()
    for row in rows:
        shard_ids.add(row["shard_id"])

    num_shards = len(shard_ids)
    total_entries = len(rows)

    # Size estimates
    download_bytes = sum(r["size"] for r in rows)
    has_inflated = rows and "inflated_size" in rows[0]
    extracted_bytes = sum(r["inflated_size"] for r in rows) if has_inflated else None

    print(f"{total_entries} entries spread across {num_shards} shard(s)")
    print(f"  Download size       : {_format_size(download_bytes)}")
    if extracted_bytes is not None:
        print(f"  Extracted on disk   : {_format_size(extracted_bytes)}")
    else:
        print(f"  Extracted on disk   : (unknown — inflated_size not yet backfilled)")

    if confirm:
        answer = input("\nProceed with download? [Y/n] ").strip().lower()
        if answer and answer not in ("y", "yes"):
            print("Skipped.")
            return

    # Resume support: track completed shards in a local file so that
    # interrupted downloads can skip already-extracted shards.
    progress_file = Path(target_dir) / f".progress_{data_source_dir.replace('/', '_')}.json"
    completed_shards: set[int] = set()
    if progress_file.is_file():
        with open(progress_file) as f:
            completed_shards = set(json.load(f))

    remaining_shards = sorted(shard_ids - completed_shards)
    if completed_shards:
        print(
            f"Resuming: {len(completed_shards)} shard(s) already extracted, "
            f"{len(remaining_shards)} remaining"
        )

    # 3. Download & extract shards ------------------------------------
    print(f"\n=== Downloading & extracting {len(remaining_shards)} shard(s) ===")

    extracted = 0
    for shard_id in tqdm(remaining_shards, desc="Shards"):
        shard_filename = f"{data_source_dir}/shards/{shard_id:05d}.tar"

        # hf_hub_download caches the file; repeated runs skip the download.
        shard_local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=shard_filename,
            repo_type=HF_REPO_TYPE,
            revision=revision,
        )

        # Stream through the shard tar and extract each .tar.zst member.
        with tarfile.open(shard_local, "r:") as shard_tar:
            members = [m for m in shard_tar.getmembers() if m.isfile()]
            for member in tqdm(members, desc=f"  Shard {shard_id}", leave=False):
                fobj = shard_tar.extractfile(member)
                if fobj is None:
                    continue

                data = fobj.read()
                _extract_tar_zst_bytes(data, target_dir)
                extracted += 1

        # Free the cached shard to avoid accumulating disk usage.
        try:
            os.remove(shard_local)
        except OSError:
            pass

        completed_shards.add(shard_id)
        with open(progress_file, "w") as f:
            json.dump(sorted(completed_shards), f)

    if data_type not in manifest:
        manifest[data_type] = {}
    if data_source not in manifest[data_type]:
        manifest[data_type][data_source] = []
    manifest[data_type][data_source].append(version)

    with open(extracted_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean up progress file now that the source is fully extracted
    if progress_file.is_file():
        progress_file.unlink()

    print(f"\nDone. Extracted {extracted} archives into: {target_dir}")


# ------------------------------------------------------------------
# Discover & download all sources
# ------------------------------------------------------------------


def discover_sources(
    revision: str = "main",
    source: str = "all",
) -> list[str]:
    """Auto-discover every available source in the HF repo.
    Scans the full file listing of the repository and collects the
    parent directories of all parquet files whose name contains
    ``pkgs`` (the split name used by ``mirror_shard.py``).  Each such
    directory corresponds to one ``data_source_dir`` / config that was
    uploaded.
    Parameters
    ----------
    revision:
        HF branch or revision.
    source:
        Which top-level prefix to include.  One of `VALID_SOURCES`.
    Returns
    -------
    list[str]
        Sorted list of ``data_source_dir`` paths, e.g.::
            [
                "mujoco/benchmarks/molmospaces-bench-v1/20260210",
                "mujoco/objects/thor/20251117",
                ...
            ]
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source {source!r}, must be one of {VALID_SOURCES}")

    api = HfApi()
    all_files = api.list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        revision=revision,
    )

    data_source_dirs: set[str] = set()
    for path in all_files:
        if path.endswith(".parquet") and "pkgs" in os.path.basename(path):
            parent = os.path.dirname(path)
            # Parquet configs use "__" as separator; convert back to "/"
            if "__" in parent:
                parent = parent.replace("__", "/")
            else:
                continue
            if source == "all" or parent.startswith(source + "/"):
                data_source_dirs.add(parent)

    sources = sorted(data_source_dirs)
    return sources


def download_all(
    base_dir: str,
    revision: str = "main",
    source: str = "all",
    versioned: bool = True,
) -> None:
    """Discover and download every source in the HF repo.
    Calls :func:`discover_sources` to list all available ``data_source_dir``
    entries, then iterates through them calling
    :func:`download_and_extract` for each one.  Each source's data is
    extracted into ``<base_dir>/<prefix>/<data_type>/<name>`` (the
    version segment is stripped -- see :func:`_target_dir_for_data_source_dir`).
    Sources that fail are logged and skipped so that a single failure
    does not abort the entire batch.
    Parameters
    ----------
    base_dir:
        Root directory under which all sources are extracted.
    revision:
        HF branch or revision.
    source:
        Which top-level prefix to include.  One of `VALID_SOURCES`.
    versioned:
        Whether to include the version string in the extracted data paths
    """
    print("=== Discovering sources from HF repo ===")
    sources = discover_sources(revision=revision, source=source)
    print(f"Found {len(sources)} source(s):\n")

    total_download = 0
    total_extracted = 0
    has_all_inflated = True
    for s in sources:
        dl, ex = estimate_sizes(s, revision=revision)
        total_download += dl
        if ex is not None:
            total_extracted += ex
        else:
            has_all_inflated = False
        ex_str = _format_size(ex) if ex is not None else "unknown"
        print(f"  {s:60s}  dl: {_format_size(dl):>10s}  extracted: {ex_str:>10s}")

    print(f"\n  {'Total':60s}  dl: {_format_size(total_download):>10s}", end="")
    if has_all_inflated:
        print(f"  extracted: {_format_size(total_extracted):>10s}")
        print(
            f"\n  Disk usage: {_format_size(total_extracted)} (data shards are deleted from cache after extraction)"
        )
    else:
        print(f"  extracted: {'(partial)':>10s}")
        print(f"\n  Some sources lack inflated_size — total extracted size is incomplete.")
    print()

    answer = input("Proceed with download? [Y/n] ").strip().lower()
    if answer and answer not in ("y", "yes"):
        print("Aborted.")
        return

    completed: list[str] = []
    failed: list[str] = []

    for data_source_dir in sources:
        target_dir = str(Path(base_dir) / data_source_dir)

        print(f"\n{'=' * 60}")
        print(f"  HF dir    : {data_source_dir}")
        print(f"  Target dir: {target_dir}")
        print(f"{'=' * 60}")

        try:
            download_and_extract(
                data_source_dir,
                target_dir,
                revision=revision,
                versioned=versioned,
                confirm=False,
            )
            completed.append(data_source_dir)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"FAILED: {data_source_dir}  ({e})")
            failed.append(data_source_dir)

    print(f"\nCompleted: {len(completed)}/{len(sources)}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download molmospaces",
    )
    parser.add_argument(
        "target_dir",
        help="Local root directory to extract into",
    )
    parser.add_argument(
        "--data_source_dir",
        default=None,
        help=(
            "Download a single source by its directory path, e.g."
            " mujoco/objects/thor/20251117. When omitted, all sources"
            " are auto-discovered from the repo and downloaded. Use --list"
            " to find valid directory paths"
        ),
    )
    parser.add_argument(
        "--source",
        choices=VALID_SOURCES,
        default="all",
        help=(
            f"Which data source prefix to include among {VALID_SOURCES}, with "
            "'all' the default.  Only affects --list and bulk download; "
            "ignored when --data_source_dir is given."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Only list discovered sources, don't download anything",
    )
    parser.add_argument(
        "--versioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downloading as versioned enables the extracted data to be directly used"
        " by the MolmoSpaces codebase as extracted cache data, e.g. by exporting"
        " MLSPACES_CACHE_DIR=<target_dir>/mujoco and optionally exporting a different"
        " MLSPACES_ASSETS_DIR (where the cached contents will be symlinked)."
        " Disabling this flag enables directly visualizing scenes or using the"
        " data with external codebases, as the data versions do not get in the"
        " way of the expected relative paths between objects and scenes.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    args = parser.parse_args()

    if args.list_only:
        sources = discover_sources(revision="main", source=args.source)
        print(f"Found {len(sources)} source(s):")
        for s in sources:
            print(f"  {s}")
    elif args.data_source_dir:
        target = Path(args.target_dir) / args.data_source_dir
        download_and_extract(
            args.data_source_dir,
            str(target),
            revision="main",
            versioned=args.versioned,
            confirm=not args.yes,
        )
    else:
        download_all(
            args.target_dir,
            revision="main",
            source=args.source,
            versioned=args.versioned,
        )


if __name__ == "__main__":
    main()
