"""
Build a commercial episode index from commercial_houses.parquet.

One row per house. Configs that place added_objects at episode time are streamed
from tar/H5; valid episode ordinals (0..N-1 in archive iteration order) with no
NC added_objects are stored as a comma-separated list in valid_episodes_string.

All other configs use valid_episodes_string='*' (all episodes kept; no H5 read).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from episode_license_info import iterate_episode_info
from run_dashboard import RunDashboard, section

NON_COMMERCIAL_LICENSES = frozenset({"by-nc", "by-nc-sa"})
DEFAULT_LICENSE = "by"
WILDCARD_EPISODES = "*"

# From probe_added_objects.py sampling (configs WITH added_objects in samples).
DEFAULT_CONFIGS_REQUIRING_EPISODE_CHECK = frozenset(
    {
        "FrankaPickAndPlaceColorOmniCamConfig",
        "FrankaPickAndPlaceOmniCamConfig",
        "FrankaPickAndPlaceOmniCamConfig_ObjectBackfill",
        "RBY1PickAndPlaceDataGenConfig",
    }
)

PARQUET_SCHEMA = pa.schema(
    [
        ("config", pa.string()),
        ("split", pa.string()),
        ("part", pa.int64()),
        ("entry_index", pa.int64()),
        ("path", pa.string()),
        ("shard_id", pa.int64()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
        ("scene_id", pa.string()),
        ("scene_idx", pa.int64()),
        ("scene_family", pa.string()),
        ("valid_episodes_string", pa.string()),
        ("episodes_total", pa.int64()),
        ("episodes_discarded_nc", pa.int64()),
    ]
)


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


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


def load_asset_to_license(license_to_asset_path: Path) -> dict[str, str]:
    """Invert license_to_asset_id.json.gz into asset_id -> license key."""
    with gzip.open(license_to_asset_path, "rt") as f:
        license_to_assets: dict[str, list[str]] = json.load(f)

    asset_to_license: dict[str, str] = {}
    for license_key, asset_ids in license_to_assets.items():
        for asset_id in asset_ids:
            asset_to_license[asset_id] = license_key
    return asset_to_license


def load_episode_check_configs(
    probe_json: Path | None,
    configs: list[str] | None,
) -> frozenset[str]:
    if configs is not None:
        return frozenset(configs)
    if probe_json is not None:
        data = json.loads(probe_json.read_text())
        return frozenset(data["configs_with_added_objects"])
    return DEFAULT_CONFIGS_REQUIRING_EPISODE_CHECK


def empty_part_stats() -> dict[str, int]:
    return {
        "houses_processed": 0,
        "houses_wildcard": 0,
        "houses_checked": 0,
        "episodes_total": 0,
        "episodes_kept": 0,
        "episodes_discarded_nc_added": 0,
        "episode_total_unknown_houses": 0,
    }


def part_stats_for(stats: dict, config_name: str, split: str, part: int) -> dict[str, int]:
    by_config = stats.setdefault("by_config_split_part", {})
    by_split = by_config.setdefault(config_name, {})
    by_part = by_split.setdefault(split, {})
    part_key = str(part)
    if part_key not in by_part:
        by_part[part_key] = empty_part_stats()
    return by_part[part_key]


def asset_id_from_object_path(path) -> str:
    name = path.name if hasattr(path, "name") else str(path).split("/")[-1]
    return name.rsplit(".", 1)[0]


def get_added_objects(obs_scene: dict) -> dict:
    config = obs_scene.get("config")
    if config is None:
        return {}
    task_config = getattr(config, "task_config", None)
    if task_config is None:
        return {}
    return getattr(task_config, "added_objects", None) or {}


def added_objects_have_nc_license(
    added_objects: dict, asset_to_license: dict[str, str]
) -> bool:
    for path in added_objects.values():
        asset_id = asset_id_from_object_path(path)
        license_key = asset_to_license.get(asset_id, DEFAULT_LICENSE)
        if license_key in NON_COMMERCIAL_LICENSES:
            return True
    return False


def valid_episodes_string_for_house(
    house: dict,
    asset_to_license: dict[str, str],
) -> tuple[str, int, int, int]:
    """
    Return (valid_episodes_string, episodes_total, episodes_kept, episodes_discarded_nc).

    Episode ordinals are 0..N-1 in iterate_episode_info() order within the archive.
    """
    config_name = house["config"]
    split = house["split"]
    entry = {
        "shard_id": house["shard_id"],
        "offset": house["offset"],
        "size": house["size"],
        "part": house["part"],
        "path": house["path"],
    }

    valid_indices: list[int] = []
    episodes_total = 0
    episodes_discarded_nc = 0

    for obs_scene in iterate_episode_info(entry, split, config_name):
        episode_ordinal = episodes_total
        episodes_total += 1
        added_objects = get_added_objects(obs_scene)
        if added_objects_have_nc_license(added_objects, asset_to_license):
            episodes_discarded_nc += 1
            continue
        valid_indices.append(episode_ordinal)

    episodes_kept = len(valid_indices)
    if episodes_kept == 0:
        return "", episodes_total, episodes_kept, episodes_discarded_nc
    return (
        ",".join(str(i) for i in valid_indices),
        episodes_total,
        episodes_kept,
        episodes_discarded_nc,
    )


def episodes_kept_from_valid_string(valid_episodes_string: str) -> int:
    if not valid_episodes_string or valid_episodes_string == WILDCARD_EPISODES:
        return 0
    return len(valid_episodes_string.split(","))


def format_episode_part_summary(part_stats: dict[str, int]) -> str:
    kept = part_stats["episodes_kept"]
    total = part_stats["episodes_total"]
    if part_stats["episode_total_unknown_houses"] == 0 and total > 0:
        return f"{kept}/{total} ep ({pct(kept, total)}% kept)"
    if kept > 0:
        return f"{kept} ep kept"
    if part_stats["houses_checked"]:
        return f"{part_stats['houses_checked']} houses checked"
    return "0/0 ep"


def normalize_episode_row(row: dict) -> dict:
    """Ensure episode stat columns exist (legacy parquet compatibility)."""
    normalized = dict(row)
    valid = normalized["valid_episodes_string"]
    if "episodes_total" in normalized and "episodes_discarded_nc" in normalized:
        return normalized
    if valid == WILDCARD_EPISODES:
        normalized["episodes_total"] = -1
        normalized["episodes_discarded_nc"] = 0
    elif valid == "":
        normalized["episodes_total"] = 0
        normalized["episodes_discarded_nc"] = 0
    else:
        normalized["episodes_total"] = -1
        normalized["episodes_discarded_nc"] = -1
    return normalized


def house_key(house: dict) -> tuple[str, str, str]:
    return house["config"], house["split"], house["path"]


def load_existing_episode_rows(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    return pq.read_table(output_path).to_pylist()


def seed_stats_from_existing_rows(stats: dict, existing_rows: list[dict]) -> None:
    for row in existing_rows:
        config_name = row["config"]
        split = row["split"]
        part = int(row["part"])
        part_stats = part_stats_for(stats, config_name, split, part)
        stats["houses_processed"] += 1
        part_stats["houses_processed"] += 1
        if row["valid_episodes_string"] == WILDCARD_EPISODES:
            stats["houses_wildcard"] += 1
            part_stats["houses_wildcard"] += 1
        else:
            stats["houses_checked"] += 1
            part_stats["houses_checked"] += 1
            row = normalize_episode_row(row)
            kept = episodes_kept_from_valid_string(row["valid_episodes_string"])
            stats["episodes_kept"] += kept
            part_stats["episodes_kept"] += kept
            total = int(row["episodes_total"])
            discarded = int(row["episodes_discarded_nc"])
            if total >= 0:
                stats["episodes_total"] += total
                part_stats["episodes_total"] += total
                if discarded >= 0:
                    stats["episodes_discarded_nc_added"] += discarded
                    part_stats["episodes_discarded_nc_added"] += discarded
            else:
                stats["episode_total_unknown_houses"] += 1
                part_stats["episode_total_unknown_houses"] += 1


def house_part_key(house: dict) -> tuple[str, str, int]:
    return house["config"], house["split"], int(house["part"])


def part_stats_lookup(
    stats: dict, config_name: str, split: str, part: int
) -> dict[str, int]:
    return (
        stats.get("by_config_split_part", {})
        .get(config_name, {})
        .get(split, {})
        .get(str(part), empty_part_stats())
    )


def adjust_in_flight(
    stats: dict,
    lock: threading.Lock,
    house: dict,
    delta: int,
) -> None:
    key = house_part_key(house)
    with lock:
        in_flight = stats.setdefault("in_flight_by_part", {})
        new_count = in_flight.get(key, 0) + delta
        if new_count <= 0:
            in_flight.pop(key, None)
        else:
            in_flight[key] = new_count


def parallel_checked_house_worker(
    house: dict,
    asset_to_license: dict[str, str],
    stats: dict,
    in_flight_lock: threading.Lock,
) -> dict:
    adjust_in_flight(stats, in_flight_lock, house, 1)
    try:
        return process_checked_house(house, asset_to_license)
    finally:
        adjust_in_flight(stats, in_flight_lock, house, -1)


def process_checked_house(
    house: dict,
    asset_to_license: dict[str, str],
) -> dict:
    """Stream one house archive and return a parquet row (worker-safe)."""
    valid_eps, ep_total, ep_kept, ep_discarded = valid_episodes_string_for_house(
        house, asset_to_license
    )
    return house_row(
        house,
        valid_eps,
        episodes_total=ep_total,
        episodes_discarded_nc=ep_discarded,
    )


def record_completed_row(stats: dict, house: dict, row: dict) -> None:
    """Update global and per-part stats after one house row is produced."""
    config_name = house["config"]
    split = house["split"]
    part = int(house["part"])
    if stats.get("workers", 1) <= 1:
        stats["current_part"] = (config_name, split, part)
    part_stats = part_stats_for(stats, config_name, split, part)

    if row["valid_episodes_string"] == WILDCARD_EPISODES:
        stats["houses_wildcard"] += 1
        part_stats["houses_wildcard"] += 1
    else:
        stats["houses_checked"] += 1
        part_stats["houses_checked"] += 1
        ep_total = int(row["episodes_total"])
        ep_discarded = int(row["episodes_discarded_nc"])
        ep_kept = episodes_kept_from_valid_string(row["valid_episodes_string"])
        stats["episodes_total"] += ep_total
        stats["episodes_kept"] += ep_kept
        stats["episodes_discarded_nc_added"] += ep_discarded
        part_stats["episodes_total"] += ep_total
        part_stats["episodes_kept"] += ep_kept
        part_stats["episodes_discarded_nc_added"] += ep_discarded

    stats["houses_processed"] += 1
    part_stats["houses_processed"] += 1


def house_row(
    house: dict,
    valid_episodes_string: str,
    *,
    episodes_total: int = -1,
    episodes_discarded_nc: int = 0,
) -> dict:
    return {
        "config": house["config"],
        "split": house["split"],
        "part": int(house["part"]),
        "entry_index": int(house["entry_index"]),
        "path": house["path"],
        "shard_id": int(house["shard_id"]),
        "offset": int(house["offset"]),
        "size": int(house["size"]),
        "scene_id": house["scene_id"],
        "scene_idx": int(house["scene_idx"]),
        "scene_family": house["scene_family"],
        "valid_episodes_string": valid_episodes_string,
        "episodes_total": episodes_total,
        "episodes_discarded_nc": episodes_discarded_nc,
    }


def count_houses_per_part(houses: list[dict]) -> dict[tuple[str, str, int], int]:
    counts: dict[tuple[str, str, int], int] = {}
    for house in houses:
        key = (house["config"], house["split"], int(house["part"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _current_part_progress_text(stats: dict) -> str:
    workers = stats.get("workers", 1)
    if workers > 1:
        in_flight = stats.get("in_flight_by_part") or {}
        active = sum(in_flight.values())
        return (
            f"parallel {workers}w | {active} in flight | "
            f"{len(in_flight)} parts active | "
        )

    current = stats.get("current_part")
    part_totals = stats.get("part_totals") or {}
    if not current:
        return ""
    config_name, split, part = current
    part_stats = part_stats_lookup(stats, config_name, split, part)
    total = part_totals.get((config_name, split, part), 0)
    processed = part_stats["houses_processed"]
    text = f"current {config_name} [{split}] p{part} {processed}/{total} | "
    if part_stats["houses_checked"]:
        text += f"{format_episode_part_summary(part_stats)} | "
    return text


def format_serial_current_section(
    stats: dict,
    part_totals: dict[tuple[str, str, int], int],
) -> dict | None:
    current = stats.get("current_part")
    if not current:
        return None
    config_name, split, part = current
    part_stats = part_stats_lookup(stats, config_name, split, part)
    total = part_totals.get((config_name, split, part), 0)
    processed = part_stats["houses_processed"]
    rows: list[tuple[str, str]] = [
        ("houses", f"{processed}/{total} ({pct(processed, total)}%)"),
    ]
    if part_stats["houses_checked"]:
        rows.append(("episodes", format_episode_part_summary(part_stats)))
    elif part_stats["houses_wildcard"]:
        rows.append(("mode", "wildcard (*)"))
    return section(f"Current: {config_name} [{split}] p{part}", rows)


def format_parallel_workers_section(
    stats: dict,
    part_totals: dict[tuple[str, str, int], int],
) -> dict | None:
    workers = stats.get("workers", 1)
    if workers <= 1:
        return format_serial_current_section(stats, part_totals)

    in_flight = stats.get("in_flight_by_part") or {}
    total_in_flight = sum(in_flight.values())
    rows: list[tuple[str, str]] = [
        ("workers", str(workers)),
        (
            "in flight",
            f"{total_in_flight} houses across {len(in_flight)} part(s)",
        ),
    ]

    active_parts = sorted(
        in_flight.items(),
        key=lambda item: (-item[1], item[0]),
    )
    for (config_name, split, part), count in active_parts[:8]:
        part_stats = part_stats_lookup(stats, config_name, split, part)
        total = part_totals.get((config_name, split, part), 0)
        processed = part_stats["houses_processed"]
        detail = f"{count} in flight | {processed}/{total} houses done"
        if part_stats["houses_checked"]:
            detail += f" | {format_episode_part_summary(part_stats)}"
        rows.append((f"{config_name} [{split}] p{part}", detail))

    if not active_parts:
        rows.append(("status", "waiting for workers"))
    return section("Parallel episode check", rows)


def _part_row_label(
    stats: dict,
    config_name: str,
    split: str,
    part_num: int,
) -> str:
    label = f"{config_name} [{split}] p{part_num}"
    if stats.get("workers", 1) > 1:
        in_flight = (stats.get("in_flight_by_part") or {}).get(
            (config_name, split, part_num), 0
        )
        if in_flight > 0:
            label = f"→ {label} ({in_flight} in flight)"
    elif stats.get("current_part") == (config_name, split, part_num):
        label = f"→ {label}"
    return label


def format_current_part_section(
    stats: dict,
    part_totals: dict[tuple[str, str, int], int],
) -> dict | None:
    return format_parallel_workers_section(stats, part_totals)


def format_stats_sections(stats: dict) -> list[dict]:
    sections: list[dict] = []
    part_totals = stats.get("part_totals") or {}
    current_section = format_current_part_section(stats, part_totals)
    if current_section is not None:
        sections.append(current_section)
    by_config_split_part = stats.get("by_config_split_part")
    if not by_config_split_part:
        return sections
    rows: list[tuple[str, str]] = []
    for config_name in sorted(by_config_split_part):
        for split in sorted(by_config_split_part[config_name]):
            for part in sorted(
                by_config_split_part[config_name][split],
                key=lambda p: int(p),
            ):
                part_stats = by_config_split_part[config_name][split][part]
                part_num = int(part)
                if part_stats["houses_checked"]:
                    value = format_episode_part_summary(part_stats)
                else:
                    value = f"{part_stats['houses_wildcard']} wildcard"
                total = part_totals.get((config_name, split, part_num), 0)
                if total > 0:
                    value = (
                        f"{part_stats['houses_processed']}/{total} houses | {value}"
                    )
                label = _part_row_label(stats, config_name, split, part_num)
                rows.append((label, value))
    if rows:
        sections.append(section("By config / split / part", rows))
    return sections


def format_stats(stats: dict) -> str:
    lines = [
        f"houses processed: {stats['houses_processed']}",
        f"houses wildcard ('*', no H5): {stats['houses_wildcard']}",
        f"houses episode-checked (H5): {stats['houses_checked']}",
        f"episodes total (checked configs only): {stats['episodes_total']}",
        f"episodes kept: {stats['episodes_kept']} "
        f"({pct(stats['episodes_kept'], stats['episodes_total'])}%)",
        f"episodes discarded (NC in added_objects): {stats['episodes_discarded_nc_added']}",
        f"configs requiring episode check: {sorted(stats['configs_requiring_episode_check'])}",
    ]
    by_config_split_part = stats.get("by_config_split_part")
    if by_config_split_part:
        lines.append("by config, split, and part:")
        for config_name in sorted(by_config_split_part):
            for split in sorted(by_config_split_part[config_name]):
                for part in sorted(
                    by_config_split_part[config_name][split],
                    key=lambda p: int(p),
                ):
                    part_stats = by_config_split_part[config_name][split][part]
                    if part_stats["houses_checked"]:
                        detail = (
                            f"{format_episode_part_summary(part_stats)}, "
                            f"{part_stats['houses_checked']} houses checked"
                        )
                    else:
                        detail = (
                            f"{part_stats['houses_wildcard']} houses wildcard (*)"
                        )
                    lines.append(f"  {config_name} [{split}] part {part}: {detail}")
    return "\n".join(lines)


def load_houses(
    houses_path: Path,
    *,
    config: str | None = None,
    split: str | None = None,
    part: int | None = None,
) -> list[dict]:
    table = pq.read_table(houses_path)
    houses = table.to_pylist()
    if config is not None:
        houses = [h for h in houses if h["config"] == config]
    if split is not None:
        houses = [h for h in houses if h["split"] == split]
    if part is not None:
        houses = [h for h in houses if int(h["part"]) == part]
    return houses


def limit_houses_per_group(houses: list[dict], max_per_group: int) -> list[dict]:
    """Keep at most max_per_group houses per (config, split, part)."""
    counts: dict[tuple[str, str, int], int] = {}
    limited: list[dict] = []
    for house in houses:
        key = (house["config"], house["split"], int(house["part"]))
        if counts.get(key, 0) >= max_per_group:
            continue
        counts[key] = counts.get(key, 0) + 1
        limited.append(house)
    return limited


def generate_commercial_episodes_parquet(
    output_path: Path,
    asset_to_license: dict[str, str],
    houses: list[dict],
    configs_requiring_episode_check: frozenset[str],
    *,
    existing_rows: list[dict] | None = None,
    part_totals: dict[tuple[str, str, int], int] | None = None,
    dashboard: RunDashboard | None = None,
    flush_every: int = 500,
    progress_interval: int = 10,
    progress_seconds: float = 5.0,
    workers: int = 1,
) -> dict:
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    stats: dict = {
        "houses_processed": 0,
        "houses_wildcard": 0,
        "houses_checked": 0,
        "episodes_total": 0,
        "episodes_kept": 0,
        "episodes_discarded_nc_added": 0,
        "episode_total_unknown_houses": 0,
        "configs_requiring_episode_check": sorted(configs_requiring_episode_check),
        "by_config_split_part": {},
        "part_totals": part_totals or {},
        "current_part": None,
        "workers": workers,
        "in_flight_by_part": {},
    }
    existing_rows = existing_rows or []
    seed_stats_from_existing_rows(stats, existing_rows)
    baseline_houses = len(existing_rows)
    total_houses = baseline_houses + len(houses)
    start_time = time.monotonic()
    last_report_time = start_time

    def restart_progress_timing() -> None:
        nonlocal start_time, last_report_time
        start_time = time.monotonic()
        last_report_time = start_time
        if dashboard is not None:
            dashboard.mark_progress_baseline()

    def flush() -> None:
        nonlocal writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=PARQUET_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(output_path, PARQUET_SCHEMA)
        writer.write_table(table)
        buffer.clear()

    if existing_rows:
        for offset in range(0, len(existing_rows), flush_every):
            buffer.extend(
                normalize_episode_row(row)
                for row in existing_rows[offset : offset + flush_every]
            )
            flush()
        if dashboard is not None:
            dashboard.update(
                current=stats["houses_processed"],
                metrics={
                    "houses_wildcard": stats["houses_wildcard"],
                    "houses_checked": stats["houses_checked"],
                    "episodes_total": stats["episodes_total"],
                    "episodes_kept": stats["episodes_kept"],
                    "episodes_discarded_nc_added": stats[
                        "episodes_discarded_nc_added"
                    ],
                },
                sections=format_stats_sections(stats),
                force=True,
            )
        restart_progress_timing()

    def maybe_report(force: bool = False) -> None:
        nonlocal last_report_time
        now = time.monotonic()
        if not force:
            if stats["houses_processed"] % progress_interval != 0:
                if now - last_report_time < progress_seconds:
                    return
        last_report_time = now
        elapsed = now - start_time
        remaining = total_houses - stats["houses_processed"]
        done_this_run = stats["houses_processed"] - baseline_houses
        if done_this_run > 0 and remaining > 0:
            eta = elapsed / done_this_run * remaining
        else:
            eta = 0.0 if remaining <= 0 else None
        print(
            f"houses {stats['houses_processed']}/{total_houses} "
            f"({pct(stats['houses_processed'], total_houses)}%) | "
            f"wildcard {stats['houses_wildcard']} | "
            f"episodes {stats['episodes_kept']}/{stats['episodes_total']} kept | "
            f"{_current_part_progress_text(stats)}"
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
            file=sys.stderr,
            flush=True,
        )

    def dashboard_metrics() -> dict:
        metrics = {
            "houses_wildcard": stats["houses_wildcard"],
            "houses_checked": stats["houses_checked"],
            "episodes_total": stats["episodes_total"],
            "episodes_kept": stats["episodes_kept"],
            "episodes_discarded_nc_added": stats["episodes_discarded_nc_added"],
        }
        if workers > 1:
            in_flight = stats.get("in_flight_by_part") or {}
            metrics["workers"] = workers
            metrics["houses_in_flight"] = sum(in_flight.values())
            metrics["parts_in_flight"] = len(in_flight)
        return metrics

    def after_row(row: dict, house: dict) -> None:
        record_completed_row(stats, house, row)
        if len(buffer) >= flush_every:
            flush()
        if dashboard is not None:
            dashboard.update(
                current=stats["houses_processed"],
                metrics=dashboard_metrics(),
                sections=format_stats_sections(stats),
            )
        maybe_report()

    wildcard_houses = [
        h for h in houses if h["config"] not in configs_requiring_episode_check
    ]
    check_houses = [
        h for h in houses if h["config"] in configs_requiring_episode_check
    ]

    error: BaseException | None = None
    try:
        for house in wildcard_houses:
            row = house_row(
                house,
                WILDCARD_EPISODES,
                episodes_total=-1,
                episodes_discarded_nc=0,
            )
            buffer.append(row)
            after_row(row, house)

        if workers <= 1:
            for house in check_houses:
                row = process_checked_house(house, asset_to_license)
                buffer.append(row)
                after_row(row, house)
        else:
            in_flight_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        parallel_checked_house_worker,
                        house,
                        asset_to_license,
                        stats,
                        in_flight_lock,
                    ): house
                    for house in check_houses
                }
                for future in as_completed(futures):
                    house = futures[future]
                    row = future.result()
                    buffer.append(row)
                    after_row(row, house)

    except BaseException as exc:
        error = exc
        raise
    finally:
        flush()
        if writer is not None:
            writer.close()
        maybe_report(force=True)
        if dashboard is not None:
            finished_metrics = dashboard_metrics()
            finished_sections = format_stats_sections(stats)
            if stats["houses_processed"] >= total_houses:
                dashboard.finish(
                    state="complete",
                    metrics=finished_metrics,
                    sections=finished_sections,
                )
            elif isinstance(error, KeyboardInterrupt):
                dashboard.finish(
                    state="interrupted",
                    message=(
                        f"stopped at {stats['houses_processed']}/{total_houses} houses"
                    ),
                    metrics=finished_metrics,
                    sections=finished_sections,
                )
            elif error is not None:
                dashboard.finish(
                    state="failed",
                    message=f"{type(error).__name__}: {error}",
                    metrics=finished_metrics,
                    sections=finished_sections,
                )
            else:
                dashboard.finish(
                    state="interrupted",
                    message=(
                        f"stopped at {stats['houses_processed']}/{total_houses} houses"
                    ),
                    metrics=finished_metrics,
                    sections=finished_sections,
                )

    return stats


def stats_for_json(stats: dict) -> dict:
    """Return a JSON-serializable copy of run stats (tuple keys → strings)."""

    def key_str(key: tuple | str | int) -> str:
        if isinstance(key, tuple):
            return "|".join(str(part) for part in key)
        return str(key)

    encoded: dict = {}
    for name, value in stats.items():
        if name in ("part_totals", "in_flight_by_part") and isinstance(value, dict):
            encoded[name] = {key_str(k): v for k, v in value.items()}
        elif name == "current_part" and value is not None:
            encoded[name] = key_str(value)
        else:
            encoded[name] = value
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a per-house commercial episode index with valid_episodes_string "
            "('*' or comma-separated episode ordinals)."
        )
    )
    parser.add_argument(
        "--houses-parquet",
        type=Path,
        default=Path("commercial_houses.parquet"),
        help="Input house index from generate_commercial_houses_parquet.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("commercial_episodes.parquet"),
        help="Output parquet path. Default: commercial_episodes.parquet",
    )
    parser.add_argument(
        "--license-to-asset",
        type=Path,
        default=Path("license_to_asset_id.json.gz"),
        help="Asset license map from get_asset_per_license.py.",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=None,
        help=(
            "Optional probe_added_objects.py JSON output; uses "
            "configs_with_added_objects as the episode-check set."
        ),
    )
    parser.add_argument(
        "--episode-check-config",
        action="append",
        dest="episode_check_configs",
        metavar="CONFIG",
        help=(
            "Config that requires per-episode added_objects NC checks. "
            "May be passed multiple times; overrides defaults."
        ),
    )
    parser.add_argument(
        "--config",
        help="Optional filter: process only this config.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        help="Optional filter: process only this split.",
    )
    parser.add_argument(
        "--part",
        type=int,
        help="Optional filter: process only this part.",
    )
    parser.add_argument(
        "--max-houses",
        type=int,
        default=None,
        help="Optional global cap on houses processed (after other filters).",
    )
    parser.add_argument(
        "--max-houses-per-part",
        type=int,
        default=None,
        metavar="N",
        help="For testing: at most N houses per (config, split, part).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=500,
        help="Flush buffered rows every N houses. Default: 500.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N houses. Default: 10.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=5.0,
        help="Print progress at least every N seconds. Default: 5.",
    )
    parser.add_argument(
        "--dashboard",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Write a live dashboard JSON while running. "
            "Default path: <output>.dashboard.json"
        ),
    )
    parser.add_argument(
        "--dashboard-interval",
        type=int,
        default=10,
        help="Update dashboard every N houses. Default: 10.",
    )
    parser.add_argument(
        "--dashboard-seconds",
        type=float,
        default=2.0,
        help="Update dashboard at least every N seconds. Default: 2.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Thread pool size for episode-checked houses (HTTP range downloads). "
            "Wildcard configs always run on the main thread. Default: 1."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip houses already present in --output and append remaining rows. "
            "Use after a network failure or interrupt."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    configs_requiring_episode_check = load_episode_check_configs(
        args.probe_json,
        args.episode_check_configs,
    )
    asset_to_license = load_asset_to_license(args.license_to_asset)
    print(
        f"Loaded {len(asset_to_license)} asset licenses from {args.license_to_asset}",
        flush=True,
    )
    print(
        f"Episode NC check enabled for {len(configs_requiring_episode_check)} configs; "
        f"all others get valid_episodes_string='*'",
        flush=True,
    )
    if args.workers > 1:
        print(f"Episode check workers: {args.workers}", flush=True)

    houses = load_houses(
        args.houses_parquet,
        config=args.config,
        split=args.split,
        part=args.part,
    )
    if args.max_houses_per_part is not None:
        houses = limit_houses_per_group(houses, args.max_houses_per_part)
    if args.max_houses is not None:
        houses = houses[: args.max_houses]

    part_totals = count_houses_per_part(houses)

    existing_rows: list[dict] = []
    if args.resume:
        existing_rows = load_existing_episode_rows(args.output)
        if existing_rows:
            processed_keys = {house_key(row) for row in existing_rows}
            before = len(houses)
            houses = [house for house in houses if house_key(house) not in processed_keys]
            print(
                f"Resume: {len(existing_rows)} houses already in {args.output}; "
                f"skipping {before - len(houses)}, {len(houses)} remaining",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"Resume: no existing rows in {args.output}; starting fresh",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"Processing {len(houses)} houses from {args.houses_parquet}"
        + (
            f" (max {args.max_houses_per_part} per config/split/part)"
            if args.max_houses_per_part is not None
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )

    dashboard: RunDashboard | None = None
    if args.dashboard is not None:
        dashboard_path = (
            args.output.with_suffix(".dashboard.json")
            if args.dashboard == ""
            else Path(args.dashboard)
        )
        dashboard = RunDashboard(
            dashboard_path,
            name="commercial_episodes",
            title="Commercial episode index",
            total=len(existing_rows) + len(houses),
            unit="houses",
            baseline_current=len(existing_rows),
            interval=args.dashboard_interval,
            seconds=args.dashboard_seconds,
        )
        print(f"Writing live dashboard to {dashboard_path}", flush=True)

    stats = generate_commercial_episodes_parquet(
        args.output,
        asset_to_license,
        houses,
        configs_requiring_episode_check,
        existing_rows=existing_rows,
        part_totals=part_totals,
        dashboard=dashboard,
        flush_every=args.flush_every,
        progress_interval=args.progress_interval,
        progress_seconds=args.progress_seconds,
        workers=args.workers,
    )

    print(f"\nWrote {args.output}")
    print(format_stats(stats))
    print()
    print(json.dumps(stats_for_json(stats), indent=2))


if __name__ == "__main__":
    main()
