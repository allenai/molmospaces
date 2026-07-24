"""
Build procthor_licenses.json: per-scene sets of object license keys for
procthor-objaverse train and val scenes.

Reads scene metadata directly from the resource-manager install cache and
maps object asset IDs to licenses via license_to_asset_id.json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from molmo_spaces.molmo_spaces_constants import get_resource_manager

SPLIT_RANGES: dict[str, int] = {
    "train": 100_000,
    "val": 10_000,
}

NON_COMMERCIAL_LICENSES = frozenset({"by-nc", "by-nc-sa"})
DEFAULT_LICENSE = "by"
DEFAULT_DASHBOARD_INTERVAL = 10
DEFAULT_DASHBOARD_SECONDS = 2.0


def load_asset_to_license(license_to_asset_path: Path) -> dict[str, str]:
    """Invert license_to_asset_id.json.gz into asset_id -> license."""
    with gzip.open(license_to_asset_path, "rt") as f:
        license_to_assets: dict[str, list[str]] = json.load(f)

    asset_to_license: dict[str, str] = {}
    for license_key, asset_ids in license_to_assets.items():
        for asset_id in asset_ids:
            asset_to_license[asset_id] = license_key
    return asset_to_license


def scene_install_dir(split: str) -> Path:
    """Return the on-disk install path for a procthor-objaverse split."""
    data_source = f"procthor-objaverse-{split}"
    rm = get_resource_manager()
    version = rm.versions["scenes"][data_source]
    return rm.cache_dir / "scenes" / data_source / version


def object_licenses_for_metadata(
    metadata: dict, asset_to_license: dict[str, str]
) -> list[str]:
    """Return one license key per object instance in a scene."""
    licenses: list[str] = []
    for obj in metadata.get("objects", {}).values():
        asset_id = obj.get("asset_id")
        if not asset_id:
            continue
        licenses.append(asset_to_license.get(asset_id, DEFAULT_LICENSE))
    return licenses


def licenses_for_metadata(metadata: dict, asset_to_license: dict[str, str]) -> set[str]:
    """Return the set of license keys used by objects in a scene metadata dict."""
    return set(object_licenses_for_metadata(metadata, asset_to_license))


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * part / total, 2)


def non_commercial_scene_pct(total_scenes: int, non_commercial_scenes: int) -> float:
    return pct(non_commercial_scenes, total_scenes)


def avg_objects_per_house(total_objects: int, total_scenes: int) -> float:
    if total_scenes == 0:
        return 0.0
    return round(total_objects / total_scenes, 2)


def count_nc_objects(object_licenses: list[str]) -> int:
    return sum(1 for lic in object_licenses if lic in NON_COMMERCIAL_LICENSES)


def nc_house_stats(nc_objects_per_nc_house: Counter[int]) -> dict:
    """Summarize NC object counts across houses that have at least one NC object."""
    nc_houses = sum(nc_objects_per_nc_house.values())
    total_nc_objects = sum(
        count * houses for count, houses in nc_objects_per_nc_house.items()
    )
    max_nc_objects = max(nc_objects_per_nc_house) if nc_objects_per_nc_house else 0
    return {
        "nc_houses": nc_houses,
        "total_nc_objects_in_nc_houses": total_nc_objects,
        "avg_nc_objects_per_nc_house": avg_objects_per_house(total_nc_objects, nc_houses),
        "max_nc_objects_in_one_house": max_nc_objects,
        "nc_objects_per_nc_house_histogram": {
            str(count): houses
            for count, houses in sorted(nc_objects_per_nc_house.items())
        },
    }


@dataclass
class SplitStats:
    indices_total: int
    indices_scanned: int = 0
    total_scenes: int = 0
    non_commercial_scenes: int = 0
    object_license_counts: Counter[str] = field(default_factory=Counter)
    nc_objects_per_nc_house: Counter[int] = field(default_factory=Counter)

    def record_scene(self, object_licenses: list[str]) -> None:
        self.total_scenes += 1
        nc_count = count_nc_objects(object_licenses)
        if nc_count > 0:
            self.non_commercial_scenes += 1
            self.nc_objects_per_nc_house[nc_count] += 1
        self.object_license_counts.update(object_licenses)

    def to_dashboard(self) -> dict:
        total_objects = sum(self.object_license_counts.values())
        commercial_scenes = self.total_scenes - self.non_commercial_scenes
        return {
            "indices_scanned": self.indices_scanned,
            "indices_total": self.indices_total,
            "indices_remaining": self.indices_total - self.indices_scanned,
            "completion_pct": pct(self.indices_scanned, self.indices_total),
            "scenes_found": self.total_scenes,
            "total_objects": total_objects,
            "avg_objects_per_house": avg_objects_per_house(
                total_objects, self.total_scenes
            ),
            "scenes_with_nc_pct": pct(self.non_commercial_scenes, self.total_scenes),
            "scenes_without_nc_pct": pct(commercial_scenes, self.total_scenes),
            "object_license_pcts": {
                license_key: pct(count, total_objects)
                for license_key, count in sorted(self.object_license_counts.items())
            },
            "nc_house_stats": nc_house_stats(self.nc_objects_per_nc_house),
        }


def build_dashboard_payload(
    split_stats: dict[str, SplitStats],
    *,
    status: str,
    current_split: str | None,
    start_time: float,
    completed_indices: int,
    total_indices: int,
) -> dict:
    elapsed = time.monotonic() - start_time
    indices_scanned = completed_indices + (
        split_stats[current_split].indices_scanned if current_split in split_stats else 0
    )
    indices_remaining = total_indices - indices_scanned
    scenes_found = sum(stats.total_scenes for stats in split_stats.values())
    total_objects = sum(
        sum(stats.object_license_counts.values()) for stats in split_stats.values()
    )
    nc_objects_per_nc_house: Counter[int] = Counter()
    for stats in split_stats.values():
        nc_objects_per_nc_house.update(stats.nc_objects_per_nc_house)

    if indices_scanned > 0 and indices_remaining > 0:
        eta_seconds = elapsed / indices_scanned * indices_remaining
    else:
        eta_seconds = 0.0 if indices_remaining == 0 else None

    return {
        "progress": {
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": None if eta_seconds is None else round(eta_seconds, 1),
            "indices_scanned": indices_scanned,
            "indices_total": total_indices,
            "indices_remaining": indices_remaining,
            "completion_pct": pct(indices_scanned, total_indices),
            "scenes_found": scenes_found,
            "total_objects": total_objects,
            "avg_objects_per_house": avg_objects_per_house(total_objects, scenes_found),
            "nc_house_stats": nc_house_stats(nc_objects_per_nc_house),
        },
        "splits": {split: stats.to_dashboard() for split, stats in split_stats.items()},
        "status": {
            "state": status,
            "current_split": current_split,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


class DashboardWriter:
    """Write a compact live dashboard JSON file as the scan progresses."""

    def __init__(
        self,
        path: Path,
        split_ranges: dict[str, int],
        *,
        dashboard_interval: int,
        dashboard_seconds: float,
    ):
        self.path = path
        self.split_ranges = split_ranges
        self.dashboard_interval = dashboard_interval
        self.dashboard_seconds = dashboard_seconds
        self.total_indices = sum(split_ranges.values())
        self.completed_indices = 0
        self.start_time = time.monotonic()
        self.last_write_time = 0.0
        self.split_stats: dict[str, SplitStats] = {}
        self.current_split: str | None = None
        self.current_end_idx = 0

    def start_split(self, split: str, end_idx: int) -> None:
        self.current_split = split
        self.current_end_idx = end_idx
        self.split_stats[split] = SplitStats(indices_total=self.split_ranges[split])
        self._write(status="scanning", force=True)

    def finish_split(self, split: str) -> None:
        self.completed_indices += self.split_stats[split].indices_scanned
        self.current_split = None
        self.current_end_idx = 0

    def _should_write(self, scene_idx: int, *, force: bool = False) -> bool:
        if force:
            return True
        if (scene_idx + 1) % self.dashboard_interval == 0:
            return True
        if scene_idx + 1 == self.current_end_idx:
            return True
        return time.monotonic() - self.last_write_time >= self.dashboard_seconds

    def update_progress(
        self,
        split: str,
        scene_idx: int,
        *,
        force: bool = False,
    ) -> None:
        self.split_stats[split].indices_scanned = scene_idx + 1
        if self._should_write(scene_idx, force=force):
            self._write(status="scanning")

    def record_scene(
        self,
        split: str,
        scene_idx: int,
        object_licenses: list[str],
    ) -> None:
        stats = self.split_stats[split]
        stats.record_scene(object_licenses)
        stats.indices_scanned = scene_idx + 1
        if self._should_write(scene_idx):
            self._write(status="scanning")

    def finish(self) -> None:
        self._write(status="complete", force=True)

    def _write(self, *, status: str, force: bool = False) -> None:
        del force
        payload = build_dashboard_payload(
            self.split_stats,
            status=status,
            current_split=self.current_split,
            start_time=self.start_time,
            completed_indices=self.completed_indices,
            total_indices=self.total_indices,
        )

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp_path.replace(self.path)
        self.last_write_time = time.monotonic()


class SceneJsonlWriter:
    """Append one JSON object per scene for live tailing during long scans."""

    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "w")

    def write_scene(
        self,
        split: str,
        scene_idx: int,
        licenses: set[str],
        object_licenses: list[str],
    ) -> None:
        record = {
            "split": split,
            "scene_idx": scene_idx,
            "licenses": sorted(licenses),
            "object_count": len(object_licenses),
            "nc_object_count": count_nc_objects(object_licenses),
        }
        self._file.write(json.dumps(record, sort_keys=True))
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def scan_split(
    split: str,
    install_dir: Path,
    asset_to_license: dict[str, str],
    end_idx: int,
    dashboard: DashboardWriter | None = None,
    scenes_jsonl: SceneJsonlWriter | None = None,
) -> tuple[dict[str, list[str]], dict[str, int], dict]:
    """Scan scene indices and return per-scene licenses plus summary counts."""
    if dashboard is not None:
        dashboard.start_split(split, end_idx)

    scene_licenses: dict[str, list[str]] = {}
    nc_scenes: dict[str, int] = {}
    object_license_counts: Counter[str] = Counter()
    nc_objects_per_nc_house: Counter[int] = Counter()
    non_commercial_scenes = 0

    for scene_idx in range(end_idx):
        metadata_path = install_dir / f"{split}_{scene_idx}_metadata.json"
        if metadata_path.is_file():
            with open(metadata_path) as f:
                metadata = json.load(f)

            object_licenses = object_licenses_for_metadata(metadata, asset_to_license)
            licenses = set(object_licenses)
            scene_licenses[str(scene_idx)] = sorted(licenses)
            object_license_counts.update(object_licenses)
            nc_count = count_nc_objects(object_licenses)
            if nc_count > 0:
                non_commercial_scenes += 1
                nc_scenes[str(scene_idx)] = nc_count
                nc_objects_per_nc_house[nc_count] += 1

            if scenes_jsonl is not None:
                scenes_jsonl.write_scene(split, scene_idx, licenses, object_licenses)

            if dashboard is not None:
                dashboard.record_scene(split, scene_idx, object_licenses)
            continue

        if dashboard is not None:
            dashboard.update_progress(split, scene_idx)

    if dashboard is not None:
        dashboard.split_stats[split].indices_scanned = end_idx
        dashboard.finish_split(split)
        dashboard._write(status="scanning", force=True)

    summary = {
        "total_scenes": len(scene_licenses),
        "non_commercial_scenes": non_commercial_scenes,
        "non_commercial_scene_pct": non_commercial_scene_pct(
            len(scene_licenses), non_commercial_scenes
        ),
        "indices_scanned": end_idx,
        "object_license_counts": dict(sorted(object_license_counts.items())),
        "nc_house_stats": nc_house_stats(nc_objects_per_nc_house),
    }
    return scene_licenses, nc_scenes, summary


def generate_procthor_licenses(
    license_to_asset_path: Path,
    splits: list[str] | None = None,
    ranges: dict[str, int] | None = None,
    dashboard: DashboardWriter | None = None,
    scenes_jsonl: SceneJsonlWriter | None = None,
) -> dict:
    """Scan installed procthor scenes and return licenses plus summary stats."""
    splits = splits or list(SPLIT_RANGES)
    ranges = ranges or SPLIT_RANGES

    asset_to_license = load_asset_to_license(license_to_asset_path)

    result: dict[str, dict[str, list[str]]] = {}
    nc_scenes: dict[str, dict[str, int]] = {}
    summary: dict[str, dict] = {}

    for split in splits:
        if split not in ranges:
            raise ValueError(f"No index range configured for split {split!r}")

        install_dir = scene_install_dir(split)
        if not install_dir.is_dir():
            raise FileNotFoundError(f"Scene install dir not found: {install_dir}")

        scene_licenses, split_nc_scenes, split_summary = scan_split(
            split,
            install_dir,
            asset_to_license,
            ranges[split],
            dashboard=dashboard,
            scenes_jsonl=scenes_jsonl,
        )
        result[split] = scene_licenses
        nc_scenes[split] = split_nc_scenes
        summary[split] = split_summary

        nc_stats = split_summary["nc_house_stats"]
        print(
            f"{split}: {split_summary['total_scenes']} valid scenes "
            f"({split_summary['non_commercial_scenes']} with non-commercial licenses, "
            f"{split_summary['non_commercial_scene_pct']}% of scenes) "
            f"out of {split_summary['indices_scanned']} indices; "
            f"avg {nc_stats['avg_nc_objects_per_nc_house']} NC objects/NC house"
        )

    if dashboard is not None:
        dashboard.finish()

    return {
        "scenes": result,
        "nc_scenes": nc_scenes,
        "non_commercial_scene_counts": {
            split: summary[split]["non_commercial_scenes"] for split in splits
        },
        "non_commercial_scene_pcts": {
            split: summary[split]["non_commercial_scene_pct"] for split in splits
        },
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate procthor_licenses.json from installed procthor-objaverse scenes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("procthor_licenses.json"),
        help="Output JSON path. Default: procthor_licenses.json",
    )
    parser.add_argument(
        "--scenes-jsonl",
        type=Path,
        default=None,
        help=(
            "Append per-scene results as JSONL while scanning. "
            "Default: <output-stem>.jsonl beside --output."
        ),
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=None,
        help=(
            "Live-updating dashboard JSON path. "
            "Default: <output-stem>.dashboard.json beside --output."
        ),
    )
    parser.add_argument(
        "--dashboard-interval",
        type=int,
        default=DEFAULT_DASHBOARD_INTERVAL,
        help=(
            "Also update dashboard every N checked indices. "
            f"Default: {DEFAULT_DASHBOARD_INTERVAL}."
        ),
    )
    parser.add_argument(
        "--dashboard-seconds",
        type=float,
        default=DEFAULT_DASHBOARD_SECONDS,
        help=(
            "Update dashboard at least every N seconds. "
            f"Default: {DEFAULT_DASHBOARD_SECONDS}."
        ),
    )
    parser.add_argument(
        "--license-to-asset",
        type=Path,
        default=Path("license_to_asset_id.json.gz"),
        help="Path to license_to_asset_id.json.gz. Default: license_to_asset_id.json.gz",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Split(s) to scan. Default: train val.",
    )
    parser.add_argument(
        "--train-end",
        type=int,
        default=SPLIT_RANGES["train"],
        help=f"Exclusive end index for train. Default: {SPLIT_RANGES['train']}.",
    )
    parser.add_argument(
        "--val-end",
        type=int,
        default=SPLIT_RANGES["val"],
        help=f"Exclusive end index for val. Default: {SPLIT_RANGES['val']}.",
    )
    args = parser.parse_args()

    ranges = {split: args.train_end if split == "train" else args.val_end for split in args.split}
    dashboard_path = args.dashboard or args.output.with_suffix(".dashboard.json")
    scenes_jsonl_path = args.scenes_jsonl or args.output.with_suffix(".jsonl")

    print("Setting up resource manager...")
    get_resource_manager()

    dashboard = DashboardWriter(
        dashboard_path,
        split_ranges=ranges,
        dashboard_interval=args.dashboard_interval,
        dashboard_seconds=args.dashboard_seconds,
    )
    scenes_jsonl = SceneJsonlWriter(scenes_jsonl_path)
    print(f"Writing live dashboard to {dashboard_path}")
    print(f"Streaming scenes to {scenes_jsonl_path}")

    try:
        output = generate_procthor_licenses(
            license_to_asset_path=args.license_to_asset,
            splits=args.split,
            ranges=ranges,
            dashboard=dashboard,
            scenes_jsonl=scenes_jsonl,
        )
    finally:
        scenes_jsonl.close()

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {args.output}")
    print(f"Scenes JSONL: {scenes_jsonl_path}")
    print(f"Dashboard complete: {dashboard_path}")


if __name__ == "__main__":
    main()
