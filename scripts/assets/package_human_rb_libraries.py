"""Pack the generated Rocketbox avatar libraries into one archive for transfer.

scripts/assets/convert_human_rb.py writes the three builds into
ROBOTS_DIR as `humans_rocketbox_static` / `humans_rocketbox_articulated` /
`humans_rocketbox_skinned` (see HumanRBVariant). They are gitignored like every other
locally-cached asset, so moving them to another machine -- or up to the shared
asset store -- means shipping the directories themselves. This packs them into
a single zstd tarball that unpacks straight back into a ROBOTS_DIR.

Usage:
    # all three variants -> ~/Downloads/humans_rocketbox_all_variants.tar.zst
    python scripts/assets/package_human_rb_libraries.py

    # one variant, somewhere else
    python scripts/assets/package_human_rb_libraries.py \
        --variants humans_rocketbox_skinned --out /tmp/skinned.tar.zst

To unpack on the far side:
    tar --use-compress-program=unzstd -xf humans_rocketbox_all_variants.tar.zst -C "$ROBOTS_DIR"

Requires the `zstd` binary on PATH (brew install zstd), matching the .tar.zst
packaging the shared asset store already uses.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ARCHIVE_NAME = "humans_rocketbox_all_variants.tar.zst"

# Finder droppings and AppleDouble sidecars. They serve no purpose in the
# archive and quietly inflate the file listing on every machine that unpacks it.
EXCLUDE_PATTERNS = (".DS_Store", "._*")


def default_variants() -> list[str]:
    from molmo_spaces.configs.task_sampler_configs import HumanRBVariant

    return [str(v) for v in HumanRBVariant]


def robots_dir() -> Path:
    from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR

    return ROBOTS_DIR


def check_library(library_dir: Path) -> int:
    """Fail before packing rather than after uploading.

    A library is one directory per avatar plus the assets_index.json that
    lazy_loading_utils.install_uid resolves uids through -- an index listing a
    uid whose directory never got written (a conversion that died partway,
    which the converter's per-avatar subprocess isolation makes survivable and
    therefore easy to miss) produces an archive that looks complete and fails
    on use.
    """
    if not library_dir.is_dir():
        raise SystemExit(
            f"No avatar library at {library_dir}. Generate it first with "
            "scripts/assets/convert_human_rb.py (see its docstring)."
        )
    index_path = library_dir / "assets_index.json"
    if not index_path.exists():
        raise SystemExit(f"{library_dir} has no assets_index.json -- it is not a usable library.")

    indexed = set(json.loads(index_path.read_text()))
    present = {p.name for p in library_dir.iterdir() if p.is_dir()}
    if missing := sorted(indexed - present):
        raise SystemExit(f"{library_dir.name}: indexed but not on disk: {missing}")
    if extra := sorted(present - indexed):
        print(f"  warning: {library_dir.name} has unindexed directories: {extra}")
    return len(indexed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Library directory names under ROBOTS_DIR (default: all three HumanRBVariant values).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "Downloads" / DEFAULT_ARCHIVE_NAME,
        help=f"Output archive path (default: ~/Downloads/{DEFAULT_ARCHIVE_NAME}).",
    )
    parser.add_argument(
        "--compression-level", type=int, default=10, help="zstd level, 1-19 (default: 10)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the output archive if it already exists."
    )
    args = parser.parse_args()

    if shutil.which("zstd") is None:
        raise SystemExit("zstd not found on PATH (brew install zstd).")
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists (pass --force to overwrite).")

    source_dir = robots_dir()
    variants = args.variants or default_variants()

    total = 0
    for variant in variants:
        count = check_library(source_dir / variant)
        print(f"  {variant}: {count} avatars")
        total += count

    args.out.parent.mkdir(parents=True, exist_ok=True)
    excludes = [f"--exclude={pattern}" for pattern in EXCLUDE_PATTERNS]
    # Packed relative to ROBOTS_DIR so the archive's top-level entries are the
    # library names themselves, and unpacking is `tar -C "$ROBOTS_DIR"`.
    subprocess.run(
        [
            "tar",
            # Follow symlinks. A library under ROBOTS_DIR is often a symlink to
            # a shared copy elsewhere (several molmospaces checkouts each get
            # their own ASSETS_DIR, keyed by install path, so pointing them all
            # at one generated set is the normal way to avoid regenerating
            # 1.6GB per checkout). Without this, tar dutifully archives the
            # symlink itself and produces a 455-byte "complete" archive.
            "-h",
            *excludes,
            "--use-compress-program",
            f"zstd -{args.compression_level} -T0",
            "-cf",
            str(args.out),
            *variants,
        ],
        cwd=source_dir,
        check=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({size_mb:.0f} MB, {total} avatars across {len(variants)} variants)")


if __name__ == "__main__":
    sys.exit(main())
