"""
Build a parquet index of commercial-use-safe houses from allenai/molmobot-data.

Keeps a house when it is either:
  - iTHOR (all CC-BY), or
  - procthor-objaverse with no NC object instances in the scene (from
    procthor_licenses.jsonl).

Uses only pkgs metadata (path, shard offset, scene_idx parsed from path). Does not
stream tar/H5 from HuggingFace. Episode-level filtering (e.g. NC in added_objects)
belongs in a separate downstream script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_split_names, load_dataset, load_dataset_builder

from episode_license_info import REPO, TASK_CONFIGS

HOUSE_PATH_RE = re.compile(r"_house_(\d+)\.tar\.zst$")

PARQUET_SCHEMA = pa.schema(
    [
        ("config", pa.string()),
        ("split", pa.string()),
        ("entry_index", pa.int64()),
        ("part", pa.int64()),
        ("path", pa.string()),
        ("shard_id", pa.int64()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
        ("scene_id", pa.string()),
        ("scene_idx", pa.int64()),
        ("scene_family", pa.string()),
    ]
)


@dataclass(frozen=True)
class ProcthorLicenseSets:
    commercial: set[tuple[str, int]]
    nc: set[tuple[str, int]]
    scanned: set[tuple[str, int]]


def empty_split_stats() -> dict[str, int]:
    return {
        "pkgs_rows_scanned": 0,
        "house_entries_scanned": 0,
        "houses_ready_for_analysis": 0,
        "houses_kept": 0,
        "houses_kept_ithor": 0,
        "houses_kept_procthor": 0,
        "skipped_non_house_path": 0,
        "skipped_procthor_nc": 0,
        "skipped_procthor_missing_jsonl": 0,
        "skipped_unsupported_scene_family": 0,
    }


def split_stats_for(stats: dict, config_name: str, split: str) -> dict[str, int]:
    by_config = stats.setdefault("by_config_split", {})
    by_split = by_config.setdefault(config_name, {})
    if split not in by_split:
        by_split[split] = empty_split_stats()
    return by_split[split]


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def is_ready_for_analysis(skip_reason: str | None) -> bool:
    """iTHOR always; procthor only when present in the license JSONL."""
    return skip_reason in (None, "procthor_nc")


def format_kept_summary(kept: int, ready: int, total_houses: int) -> str:
    if ready:
        return (
            f"{kept}/{ready} ({pct(kept, ready)}% of ready, "
            f"{total_houses} total houses)"
        )
    return f"{kept}/0 (no houses ready, {total_houses} total houses)"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def build_work_plan(configs: list[str], splits: list[str]) -> list[tuple[str, str, int]]:
    """Return (config, split, pkgs_row_count) using dataset metadata only."""
    plan: list[tuple[str, str, int]] = []
    for config_name in configs:
        try:
            builder = load_dataset_builder(REPO, name=config_name)
        except Exception:
            continue
        available = set(builder.info.splits or {})
        for split in splits:
            split_name = f"{split}_pkgs"
            if split_name not in available:
                try:
                    hub_splits = get_dataset_split_names(REPO, config_name)
                except Exception:
                    hub_splits = []
                if split_name not in hub_splits:
                    print(
                        f"Skipping {config_name} split={split}: not in {sorted(available)}",
                        file=sys.stderr,
                    )
                    continue
            num_rows = builder.info.splits[split_name].num_examples
            plan.append((config_name, split, num_rows))
    return plan


class ProgressReporter:
    def __init__(
        self,
        *,
        total_rows: int,
        progress_interval: int,
        progress_seconds: float,
    ) -> None:
        self.total_rows = total_rows
        self.progress_interval = progress_interval
        self.progress_seconds = progress_seconds
        self.start_time = time.monotonic()
        self.last_report_time = self.start_time
        self.rows_scanned = 0
        self.current_config = ""
        self.current_split = ""
        self.current_split_total = 0
        self.current_split_rows = 0

    def start_split(self, config_name: str, split: str, split_total: int) -> None:
        self.current_config = config_name
        self.current_split = split
        self.current_split_total = split_total
        self.current_split_rows = 0
        self.report({}, prefix="start")

    def finish_split(self, stats: dict) -> None:
        split_stats = (
            stats.get("by_config_split", {})
            .get(self.current_config, {})
            .get(self.current_split, {})
        )
        kept = split_stats.get("houses_kept", 0)
        ready = split_stats.get("houses_ready_for_analysis", 0)
        houses = split_stats.get("house_entries_scanned", 0)
        print(
            f"done [{self.current_config} {self.current_split}] "
            f"kept {format_kept_summary(kept, ready, houses)}",
            file=sys.stderr,
            flush=True,
        )
        self.report(stats, prefix="done")
        print(file=sys.stderr)

    def tick(self, stats: dict) -> None:
        self.rows_scanned += 1
        self.current_split_rows += 1
        due_interval = self.rows_scanned % self.progress_interval == 0
        due_time = time.monotonic() - self.last_report_time >= self.progress_seconds
        if due_interval or due_time:
            self.report(stats)

    def eta_seconds(self) -> float | None:
        if self.rows_scanned <= 0:
            return None
        remaining = self.total_rows - self.rows_scanned
        if remaining <= 0:
            return 0.0
        elapsed = time.monotonic() - self.start_time
        return elapsed / self.rows_scanned * remaining

    def report(self, stats: dict, *, prefix: str = "") -> None:
        now = time.monotonic()
        self.last_report_time = now

        elapsed = now - self.start_time
        eta = self.eta_seconds()
        split_pct = pct(self.current_split_rows, self.current_split_total)
        global_pct = pct(self.rows_scanned, self.total_rows)
        tag = f"[{self.current_config} {self.current_split}]"
        if prefix:
            tag = f"{prefix} {tag}"

        line = (
            f"{tag} pkgs {self.current_split_rows}/{self.current_split_total} "
            f"({split_pct}%) | global {self.rows_scanned}/{self.total_rows} "
            f"({global_pct}%) | houses {stats.get('house_entries_scanned', 0)} | "
            f"kept {stats.get('houses_kept', 0)} | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
        )
        print(line, file=sys.stderr, flush=True)


def load_procthor_license_sets(jsonl_path: Path) -> ProcthorLicenseSets:
    """Load commercial, NC, and all scanned procthor houses from JSONL."""
    commercial: set[tuple[str, int]] = set()
    nc: set[tuple[str, int]] = set()
    scanned: set[tuple[str, int]] = set()
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["split"], int(record["scene_idx"]))
            scanned.add(key)
            if record.get("nc_object_count", 0) > 0:
                nc.add(key)
            else:
                commercial.add(key)
    return ProcthorLicenseSets(commercial=commercial, nc=nc, scanned=scanned)


def format_stats(stats: dict) -> str:
    procthor_classified = stats["skipped_procthor_nc"] + stats["houses_kept_procthor"]
    ready = stats["houses_ready_for_analysis"]
    lines = [
        f"pkgs rows scanned: {stats['pkgs_rows_scanned']}",
        f"house entries scanned: {stats['house_entries_scanned']}",
        f"houses ready for analysis: {ready} "
        "(all iTHOR + procthor present in JSONL)",
        f"houses kept: {format_kept_summary(stats['houses_kept'], ready, stats['house_entries_scanned'])}",
        "discarded house entries by reason:",
        f"  procthor NC house: {stats['skipped_procthor_nc']}",
        f"  procthor missing from JSONL: {stats['skipped_procthor_missing_jsonl']}",
        f"  unsupported scene family: {stats['skipped_unsupported_scene_family']}",
        f"pkgs rows skipped (non-house paths): {stats['skipped_non_house_path']}",
        "kept houses by scene family:",
        f"  ithor: {stats['houses_kept_ithor']}",
        f"  procthor-objaverse (NC-free): {stats['houses_kept_procthor']}",
    ]
    if procthor_classified:
        lines.append(
            "among procthor houses present in JSONL "
            f"({procthor_classified} entries): "
            f"{pct(stats['skipped_procthor_nc'], procthor_classified)}% NC, "
            f"{pct(stats['houses_kept_procthor'], procthor_classified)}% kept"
        )
    if stats["skipped_procthor_missing_jsonl"]:
        lines.append(
            f"procthor houses not yet in JSONL: {stats['skipped_procthor_missing_jsonl']} "
            "(re-run after procthor_licenses.jsonl scan completes)"
        )
    by_config_split = stats.get("by_config_split")
    if by_config_split:
        lines.append(
            "kept houses by config and split (denominator = ready for analysis):"
        )
        for config_name in sorted(by_config_split):
            for split in sorted(by_config_split[config_name]):
                split_stats = by_config_split[config_name][split]
                kept = split_stats["houses_kept"]
                ready_split = split_stats["houses_ready_for_analysis"]
                houses = split_stats["house_entries_scanned"]
                lines.append(
                    f"  {config_name} [{split}]: "
                    f"{format_kept_summary(kept, ready_split, houses)}"
                )
    return "\n".join(lines)


def load_scene_source_map(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def scene_family(source: str) -> str:
    if "ithor" in source:
        return "ithor"
    if "procthor-objaverse" in source:
        return "procthor-objaverse"
    raise ValueError(f"Unsupported scene source: {source}")


def part_source(scene_source_map: dict, config: str, split: str, part: int) -> str:
    try:
        return scene_source_map[config][split][str(part)]
    except KeyError as e:
        raise KeyError(
            f"No scene source for config={config!r} split={split!r} part={part!r}"
        ) from e


def scene_idx_from_path(path: str) -> int | None:
    """Parse scene index from e.g. FrankaPickOmniCamConfig_house_42.tar.zst."""
    match = HOUSE_PATH_RE.search(path)
    if not match:
        return None
    return int(match.group(1))


def filter_house_entry(
    split: str,
    part: int,
    scene_idx: int,
    scene_source_map: dict,
    config_name: str,
    procthor_licenses: ProcthorLicenseSets,
) -> tuple[str | None, str]:
    """
    Keep/discard using path-derived scene_idx only (no H5).
    Returns (skip_reason, scene_family). skip_reason is None if kept.
    """
    source = part_source(scene_source_map, config_name, split, part)
    family = scene_family(source)

    if family == "ithor":
        return None, family

    if family == "procthor-objaverse":
        key = (split, scene_idx)
        if key not in procthor_licenses.scanned:
            return "procthor_missing_jsonl", family
        if key in procthor_licenses.nc:
            return "procthor_nc", family
        if key not in procthor_licenses.commercial:
            return "procthor_missing_jsonl", family
        return None, family

    return "unsupported_scene_family", family


def house_row(
    config_name: str,
    split: str,
    entry_index: int,
    entry: dict,
    part: int,
    scene_idx: int,
    family: str,
) -> dict:
    return {
        "config": config_name,
        "split": split,
        "entry_index": entry_index,
        "part": part,
        "path": entry["path"],
        "shard_id": int(entry["shard_id"]),
        "offset": int(entry["offset"]),
        "size": int(entry["size"]),
        "scene_id": f"part{part}_house_{scene_idx}",
        "scene_idx": scene_idx,
        "scene_family": family,
    }


def generate_commercial_houses_parquet(
    output_path: Path,
    scene_source_map: dict,
    procthor_licenses: ProcthorLicenseSets,
    configs: list[str],
    splits: list[str],
    *,
    flush_every: int = 500,
    max_entries: int | None = None,
    progress_interval: int = 1000,
    progress_seconds: float = 5.0,
) -> dict:
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    stats: dict = {
        "pkgs_rows_scanned": 0,
        "house_entries_scanned": 0,
        "houses_ready_for_analysis": 0,
        "houses_kept": 0,
        "houses_kept_ithor": 0,
        "houses_kept_procthor": 0,
        "skipped_non_house_path": 0,
        "skipped_procthor_nc": 0,
        "skipped_procthor_missing_jsonl": 0,
        "skipped_unsupported_scene_family": 0,
        "by_config_split": {},
    }

    def flush() -> None:
        nonlocal writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=PARQUET_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA)
        writer.write_table(table)
        buffer.clear()

    work_plan = build_work_plan(configs, splits)
    if max_entries is not None:
        total_rows = sum(min(size, max_entries) for _, _, size in work_plan)
    else:
        total_rows = sum(size for _, _, size in work_plan)
    print(
        f"Workload: {len(work_plan)} config/split(s), {total_rows} pkgs rows to scan",
        file=sys.stderr,
    )
    progress = ProgressReporter(
        total_rows=total_rows,
        progress_interval=progress_interval,
        progress_seconds=progress_seconds,
    )

    try:
        for config_name, split, split_size in work_plan:
            effective_split_size = (
                min(split_size, max_entries) if max_entries is not None else split_size
            )
            progress.start_split(config_name, split, effective_split_size)

            ds = load_dataset(REPO, name=config_name, split=f"{split}_pkgs")

            split_stats = split_stats_for(stats, config_name, split)

            for entry_index, entry in enumerate(ds):
                if max_entries is not None and entry_index >= max_entries:
                    break
                stats["pkgs_rows_scanned"] += 1
                split_stats["pkgs_rows_scanned"] += 1
                progress.tick(stats)

                scene_idx = scene_idx_from_path(entry["path"])
                if scene_idx is None:
                    stats["skipped_non_house_path"] += 1
                    split_stats["skipped_non_house_path"] += 1
                    continue

                stats["house_entries_scanned"] += 1
                split_stats["house_entries_scanned"] += 1
                part = int(entry["part"])
                skip_reason, family = filter_house_entry(
                    split,
                    part,
                    scene_idx,
                    scene_source_map,
                    config_name,
                    procthor_licenses,
                )
                if skip_reason is not None:
                    if skip_reason == "procthor_nc":
                        stats["skipped_procthor_nc"] += 1
                        split_stats["skipped_procthor_nc"] += 1
                    elif skip_reason == "procthor_missing_jsonl":
                        stats["skipped_procthor_missing_jsonl"] += 1
                        split_stats["skipped_procthor_missing_jsonl"] += 1
                    elif skip_reason == "unsupported_scene_family":
                        stats["skipped_unsupported_scene_family"] += 1
                        split_stats["skipped_unsupported_scene_family"] += 1
                    if is_ready_for_analysis(skip_reason):
                        stats["houses_ready_for_analysis"] += 1
                        split_stats["houses_ready_for_analysis"] += 1
                    continue

                stats["houses_ready_for_analysis"] += 1
                split_stats["houses_ready_for_analysis"] += 1

                stats["houses_kept"] += 1
                split_stats["houses_kept"] += 1
                if family == "ithor":
                    stats["houses_kept_ithor"] += 1
                    split_stats["houses_kept_ithor"] += 1
                else:
                    stats["houses_kept_procthor"] += 1
                    split_stats["houses_kept_procthor"] += 1
                buffer.append(
                    house_row(
                        config_name,
                        split,
                        entry_index,
                        entry,
                        part,
                        scene_idx,
                        family,
                    )
                )
                if len(buffer) >= flush_every:
                    flush()

            progress.finish_split(stats)

    finally:
        flush()
        if writer is not None:
            writer.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a parquet index of commercial-use-safe molmobot-data houses."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("commercial_houses.parquet"),
        help="Output parquet path. Default: commercial_houses.parquet",
    )
    parser.add_argument(
        "--procthor-licenses-jsonl",
        type=Path,
        default=Path("procthor_licenses.jsonl"),
        help="Scene license JSONL from generate_procthor_licenses.py.",
    )
    parser.add_argument(
        "--scene-source-json",
        type=Path,
        default=Path("data_to_scene_source.json"),
        help="Config/part to scene source mapping.",
    )
    parser.add_argument(
        "--config",
        choices=TASK_CONFIGS,
        help="Run for a single config. Default: all TASK_CONFIGS.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Split(s) to process. Default: train val.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=500,
        help="Flush buffered rows to parquet every N kept houses. Default: 500.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap on entries scanned per config/split (for testing).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N pkgs rows. Default: 1000.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=5.0,
        help="Print progress at least every N seconds. Default: 5.",
    )
    args = parser.parse_args()

    configs = [args.config] if args.config else list(TASK_CONFIGS)
    scene_source_map = load_scene_source_map(args.scene_source_json)
    procthor_licenses = load_procthor_license_sets(args.procthor_licenses_jsonl)
    print(
        f"Loaded procthor JSONL: {len(procthor_licenses.scanned)} scanned houses "
        f"({len(procthor_licenses.commercial)} commercial, "
        f"{len(procthor_licenses.nc)} NC) from {args.procthor_licenses_jsonl}",
        flush=True,
    )

    stats = generate_commercial_houses_parquet(
        args.output,
        scene_source_map,
        procthor_licenses,
        configs,
        args.split,
        flush_every=args.flush_every,
        max_entries=args.max_entries,
        progress_interval=args.progress_interval,
        progress_seconds=args.progress_seconds,
    )

    print(f"\nWrote {args.output}")
    print(format_stats(stats))
    print()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
