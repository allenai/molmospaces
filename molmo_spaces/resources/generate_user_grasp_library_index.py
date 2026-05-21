"""
Script to generate a grasp index for a user-provided grasp library.
"""

import argparse
from pathlib import Path

from molmo_spaces.utils.lazy_loading_utils import UserGraspLibraryIndex


def build_grasp_index(source_dir: Path) -> UserGraspLibraryIndex:
    grasp_paths: dict[str, dict[str, Path]] = {}

    for grasp_file in sorted(source_dir.rglob("grasps_*.npz")):
        uid = grasp_file.parent.name
        robot = grasp_file.stem.removeprefix("grasps_")
        if not robot:
            continue

        rel_path = grasp_file.relative_to(source_dir)
        grasp_paths.setdefault(robot, {})[uid] = rel_path

    # Keep output stable for easier diffs/review.
    sorted_grasp_paths = {
        robot: dict(sorted(uid_to_path.items()))
        for robot, uid_to_path in sorted(grasp_paths.items())
    }
    # TODO: add articulated_grasp_paths
    return UserGraspLibraryIndex(grasp_paths=sorted_grasp_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Grasp library directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for grasps_index.json. Defaults to <source_dir>/grasps_index.json.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if it exists.")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else source_dir / "grasps_index.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to overwrite.")

    grasp_index = build_grasp_index(source_dir=source_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(grasp_index.model_dump_json(indent=4))

    num_robots = len(grasp_index.grasp_paths)
    num_uids = sum(len(uid_to_path) for uid_to_path in grasp_index.grasp_paths.values())
    print(f"Wrote {output_path} with {num_robots} robots / {num_uids} grasp files.")


if __name__ == "__main__":
    main()
