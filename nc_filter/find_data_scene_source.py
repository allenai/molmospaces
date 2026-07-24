"""
Sample entries per (config, part) and identify the consensual molmospaces
scene source, without resolving licenses for every episode.

For each task config and each part, samples N dataset entry indices, resolves
the best-matching scene source for one episode in each sample, then reports
the majority-vote consensus source for that part.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from datasets import get_dataset_split_names, load_dataset

from episode_license_info import (
    REPO,
    TASK_CONFIGS,
    extract_number_substring,
    iterate_episode_info,
    resolve_scene_source,
)


def scene_source_for_entry(config_name: str, split: str, entry: dict, index: int):
    """
    Resolve the best-matching scene source for the first episode in an entry.
    Returns (scene_id, scene_idx, scene_source) or None if no valid episode.
    """
    for obs_scene in iterate_episode_info(entry, split, config_name):
        scene_id = obs_scene["scene_id"]
        scene_idx = extract_number_substring(scene_id.split("_")[-1])
        added_objects = obs_scene["config"].task_config.added_objects
        scene_objects = sorted(
            set(obs_scene["config"].task_config.object_poses.keys())
            - set(added_objects.keys())
        )
        scene_source = resolve_scene_source(scene_idx, split, scene_objects)
        return scene_id, scene_idx, scene_source, index

    return None


def sample_indices(indices: list[int], n: int, rng: random.Random) -> list[int]:
    """Sample up to n indices without replacement, sorted for stable output."""
    if len(indices) <= n:
        return sorted(indices)
    return sorted(rng.sample(indices, n))


def consensus_source(sources: list[str | None]) -> tuple[str | None, int, int]:
    """
    Majority-vote consensus over resolved sources (ignoring None).
    Returns (winner, winner_count, total_non_none).
    """
    votes = [s for s in sources if s is not None]
    if not votes:
        return None, 0, 0
    winner, count = Counter(votes).most_common(1)[0]
    return winner, count, len(votes)


def available_splits(config_name: str) -> set[str]:
    """Return split names without the '_pkgs' suffix (e.g. 'train', 'val')."""
    return {
        name.removesuffix("_pkgs")
        for name in get_dataset_split_names(REPO, config_name)
    }


def find_scene_sources_for_config(
    config_name: str,
    split: str,
    samples_per_part: int,
    rng: random.Random,
):
    """Sample and resolve scene sources for every part in a config."""
    ds = load_dataset(REPO, name=config_name, split=f"{split}_pkgs")

    part_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, entry in enumerate(ds):
        part_to_indices[entry["part"]].append(i)

    print(f"\n{'=' * 72}")
    print(f"Config: {config_name}  split={split}  entries={len(ds)}")
    print(f"Parts: {sorted(part_to_indices)}")
    print(f"{'=' * 72}")

    part_consensus: dict[int, str | None] = {}

    for part in sorted(part_to_indices):
        indices = part_to_indices[part]
        sampled = sample_indices(indices, samples_per_part, rng)

        print(f"\n--- part {part} ({len(indices)} entries, sampling {len(sampled)}) ---")

        sources: list[str | None] = []
        for index in sampled:
            entry = ds[index]
            result = scene_source_for_entry(config_name, split, entry, index)
            if result is None:
                print(f"  index={index:>6d}  (no valid episodes)")
                sources.append(None)
                continue

            scene_id, scene_idx, scene_source, _ = result
            sources.append(scene_source)
            print(
                f"  index={index:>6d}  scene_id={scene_id}  "
                f"scene_idx={scene_idx}  best_source={scene_source}"
            )

        winner, count, total = consensus_source(sources)
        part_consensus[part] = winner
        if winner is None:
            print(f"  CONSENSUS part {part}: NONE (no successful resolutions)")
        elif count == total:
            print(f"  CONSENSUS part {part}: {winner} (unanimous, {count}/{total})")
        else:
            print(f"  CONSENSUS part {part}: {winner} ({count}/{total} votes)")

    return part_consensus


def main():
    parser = argparse.ArgumentParser(
        description="Sample entries per (config, part) and find the consensual "
        "molmospaces scene source for each part."
    )
    parser.add_argument(
        "--config",
        choices=TASK_CONFIGS,
        help="Run for a single config. Default: all TASK_CONFIGS.",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Split(s) to process. Default: train val.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of entry indices to sample per part. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for sampling. Default: 0.",
    )
    args = parser.parse_args()

    configs = [args.config] if args.config else list(TASK_CONFIGS)
    rng = random.Random(args.seed)

    # (config, split) -> {part: consensus_source}
    all_consensus: dict[tuple[str, str], dict[int, str | None]] = {}
    for config_name in configs:
        config_splits = available_splits(config_name)
        for split in args.split:
            if split not in config_splits:
                print(
                    f"\nSkipping {config_name} split={split}: "
                    f"not in {sorted(config_splits)}"
                )
                continue
            all_consensus[(config_name, split)] = find_scene_sources_for_config(
                config_name, split, args.samples, rng
            )

    print(f"\n{'=' * 72}")
    print("SUMMARY: consensual scene source per (config, split, part)")
    print(f"{'=' * 72}")
    for (config_name, split), part_map in all_consensus.items():
        print(f"\n{config_name} [{split}]:")
        for part, source in sorted(part_map.items()):
            print(f"  part {part}: {source}")


if __name__ == "__main__":
    main()
