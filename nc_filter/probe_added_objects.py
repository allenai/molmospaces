"""
Probe whether episodes use task_config.added_objects by sampling houses per group.

Draws N random house archives per (config, split, part), streams each tar/H5,
and reports whether any episode has non-empty added_objects. Use this to see
which configs need episode-level NC filtering in generate_commercial_episodes_parquet.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from datasets import get_dataset_split_names, load_dataset

from episode_license_info import REPO, TASK_CONFIGS, iterate_episode_info
from generate_commercial_episodes_parquet import get_added_objects

HOUSE_PATH_RE = re.compile(r"_house_(\d+)\.tar\.zst$")


def pkgs_entry_to_house(
    config_name: str, split: str, entry_index: int, entry: dict
) -> dict:
    part = int(entry["part"])
    scene_idx_match = HOUSE_PATH_RE.search(entry["path"])
    scene_idx = int(scene_idx_match.group(1)) if scene_idx_match else -1
    return {
        "config": config_name,
        "split": split,
        "part": part,
        "entry_index": entry_index,
        "path": entry["path"],
        "shard_id": int(entry["shard_id"]),
        "offset": int(entry["offset"]),
        "size": int(entry["size"]),
        "scene_id": f"part{part}_house_{scene_idx}",
        "scene_idx": scene_idx,
    }


def load_house_groups_from_parquet(
    houses_path: Path,
    *,
    config: str | None = None,
    splits: list[str] | None = None,
) -> dict[tuple[str, str, int], list[dict]]:
    houses = pq.read_table(houses_path).to_pylist()
    if config is not None:
        houses = [h for h in houses if h["config"] == config]
    if splits is not None:
        split_set = set(splits)
        houses = [h for h in houses if h["split"] in split_set]

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for house in houses:
        key = (house["config"], house["split"], int(house["part"]))
        groups[key].append(house)
    return dict(groups)


def load_house_groups_from_pkgs(
    configs: list[str],
    splits: list[str],
) -> dict[tuple[str, str, int], list[dict]]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for config_name in configs:
        for split in splits:
            try:
                available = get_dataset_split_names(REPO, config_name)
            except Exception:
                continue
            if f"{split}_pkgs" not in available:
                print(
                    f"Skipping {config_name} [{split}]: not in {available}",
                    file=sys.stderr,
                )
                continue

            print(f"Indexing {config_name} [{split}]...", file=sys.stderr, flush=True)
            ds = load_dataset(REPO, name=config_name, split=f"{split}_pkgs")
            for entry_index, entry in enumerate(ds):
                if not HOUSE_PATH_RE.search(entry["path"]):
                    continue
                part = int(entry["part"])
                key = (config_name, split, part)
                groups[key].append(
                    pkgs_entry_to_house(config_name, split, entry_index, entry)
                )
    return dict(groups)


def sample_groups(
    groups: dict[tuple[str, str, int], list[dict]],
    samples_per_group: int,
    rng: random.Random,
) -> dict[tuple[str, str, int], list[dict]]:
    sampled: dict[tuple[str, str, int], list[dict]] = {}
    for key in sorted(groups):
        houses = groups[key]
        n = min(samples_per_group, len(houses))
        if n == 0:
            continue
        sampled[key] = rng.sample(houses, n)
    return sampled


def probe_house(house: dict) -> dict:
    config_name = house["config"]
    split = house["split"]
    entry = {
        "shard_id": house["shard_id"],
        "offset": house["offset"],
        "size": house["size"],
        "part": house["part"],
        "path": house["path"],
    }

    episodes_total = 0
    episodes_with_added_objects = 0
    added_object_names: set[str] = set()

    for obs_scene in iterate_episode_info(entry, split, config_name):
        episodes_total += 1
        added_objects = get_added_objects(obs_scene)
        if added_objects:
            episodes_with_added_objects += 1
            added_object_names.update(added_objects.keys())

    return {
        "path": house["path"],
        "entry_index": int(house["entry_index"]),
        "episodes_total": episodes_total,
        "episodes_with_added_objects": episodes_with_added_objects,
        "has_added_objects": episodes_with_added_objects > 0,
        "added_object_names": sorted(added_object_names),
    }


def empty_group_result() -> dict:
    return {
        "houses_available": 0,
        "houses_sampled": 0,
        "houses_with_added_objects": 0,
        "episodes_total": 0,
        "episodes_with_added_objects": 0,
        "any_added_objects": False,
        "house_samples": [],
    }


def probe_sampled_groups(
    sampled: dict[tuple[str, str, int], list[dict]],
    groups: dict[tuple[str, str, int], list[dict]],
) -> dict:
    by_config_split_part: dict[str, dict[str, dict[str, dict]]] = {}
    by_config: dict[str, dict] = {}

    for (config_name, split, part), houses in sorted(sampled.items()):
        group_result = empty_group_result()
        group_result["houses_available"] = len(groups[(config_name, split, part)])
        group_result["houses_sampled"] = len(houses)

        for house in houses:
            print(
                f"  probing {config_name} [{split}] part {part}: {house['path']}",
                file=sys.stderr,
                flush=True,
            )
            house_result = probe_house(house)
            group_result["house_samples"].append(house_result)
            group_result["episodes_total"] += house_result["episodes_total"]
            group_result["episodes_with_added_objects"] += house_result[
                "episodes_with_added_objects"
            ]
            if house_result["has_added_objects"]:
                group_result["houses_with_added_objects"] += 1
                group_result["any_added_objects"] = True

        by_config_split_part.setdefault(config_name, {}).setdefault(split, {})[
            str(part)
        ] = group_result

        cfg = by_config.setdefault(
            config_name,
            {
                "groups_sampled": 0,
                "groups_with_added_objects": 0,
                "houses_sampled": 0,
                "houses_with_added_objects": 0,
                "episodes_total": 0,
                "episodes_with_added_objects": 0,
                "any_added_objects": False,
            },
        )
        cfg["groups_sampled"] += 1
        if group_result["any_added_objects"]:
            cfg["groups_with_added_objects"] += 1
            cfg["any_added_objects"] = True
        cfg["houses_sampled"] += group_result["houses_sampled"]
        cfg["houses_with_added_objects"] += group_result["houses_with_added_objects"]
        cfg["episodes_total"] += group_result["episodes_total"]
        cfg["episodes_with_added_objects"] += group_result["episodes_with_added_objects"]

    configs_with_added = sorted(c for c, s in by_config.items() if s["any_added_objects"])
    configs_without_added = sorted(
        c for c, s in by_config.items() if not s["any_added_objects"]
    )

    return {
        "by_config_split_part": by_config_split_part,
        "by_config": by_config,
        "configs_with_added_objects": configs_with_added,
        "configs_without_added_objects": configs_without_added,
    }


def format_report(results: dict, samples_per_group: int) -> str:
    lines = [
        f"Sampled up to {samples_per_group} houses per (config, split, part).",
        "",
        "Configs WITH added_objects in samples:",
    ]
    if results["configs_with_added_objects"]:
        for config_name in results["configs_with_added_objects"]:
            cfg = results["by_config"][config_name]
            lines.append(
                f"  {config_name}: "
                f"{cfg['groups_with_added_objects']}/{cfg['groups_sampled']} groups, "
                f"{cfg['episodes_with_added_objects']}/{cfg['episodes_total']} episodes"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Configs WITHOUT added_objects in samples:")
    if results["configs_without_added_objects"]:
        for config_name in results["configs_without_added_objects"]:
            cfg = results["by_config"][config_name]
            lines.append(
                f"  {config_name}: "
                f"{cfg['houses_sampled']} houses, "
                f"{cfg['episodes_total']} episodes checked"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Per config / split / part:")
    for config_name in sorted(results["by_config_split_part"]):
        for split in sorted(results["by_config_split_part"][config_name]):
            for part in sorted(
                results["by_config_split_part"][config_name][split],
                key=int,
            ):
                g = results["by_config_split_part"][config_name][split][part]
                flag = "yes" if g["any_added_objects"] else "no"
                lines.append(
                    f"  {config_name} [{split}] part {part}: added_objects={flag} "
                    f"({g['houses_with_added_objects']}/{g['houses_sampled']} houses, "
                    f"{g['episodes_with_added_objects']}/{g['episodes_total']} episodes)"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random houses per (config, split, part) and check whether "
            "any episode has non-empty task_config.added_objects."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--houses-parquet",
        type=Path,
        default=None,
        help="Sample from an existing house parquet (e.g. commercial_houses.parquet).",
    )
    source.add_argument(
        "--from-pkgs",
        action="store_true",
        help="Sample house entries directly from allenai/molmobot-data pkgs (default).",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=3,
        help="Random houses to probe per (config, split, part). Default: 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling. Default: 42.",
    )
    parser.add_argument(
        "--config",
        choices=TASK_CONFIGS,
        help="Optional: only this config.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Split(s) to include. Default: train val.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON results.",
    )
    args = parser.parse_args()

    configs = [args.config] if args.config else list(TASK_CONFIGS)
    rng = random.Random(args.seed)

    if args.houses_parquet is not None:
        print(f"Loading houses from {args.houses_parquet}", file=sys.stderr, flush=True)
        groups = load_house_groups_from_parquet(
            args.houses_parquet,
            config=args.config,
            splits=args.split,
        )
    else:
        groups = load_house_groups_from_pkgs(configs, args.split)

    sampled = sample_groups(groups, args.samples_per_group, rng)
    total_samples = sum(len(h) for h in sampled.values())
    print(
        f"Probing {total_samples} houses across {len(sampled)} "
        f"(config, split, part) groups...",
        file=sys.stderr,
        flush=True,
    )

    results = probe_sampled_groups(sampled, groups)
    results["samples_per_group"] = args.samples_per_group
    results["seed"] = args.seed
    results["source"] = (
        str(args.houses_parquet) if args.houses_parquet is not None else "pkgs"
    )

    print()
    print(format_report(results, args.samples_per_group))
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
