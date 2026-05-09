"""
Postprocess a rollout annotation JSONL (e.g. rollout_eval.jsonl) and produce
histograms of the outcome distribution split by nearby-graspable density:
  - episodes with num_nearby_graspable <= threshold
  - episodes with num_nearby_graspable >  threshold

The "nearby graspable" metric matches the one used in create_json_benchmark.py:
3D Euclidean distance from the pickup object's start pose, with categories
filtered against THOR_PICKUP_OBJECTS_LOWERCASE.

USAGE:
    python scripts/benchmarks/analyze_outcomes_by_density.py \\
        --annotation_file rollout_eval.jsonl \\
        --threshold 2 \\
        --output_dir /weka/prior/aguru/datasets/rollout_analysis
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the helpers from create_json_benchmark so the density definition stays
# in lock-step with the benchmark creation pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_json_benchmark import (
    compute_num_nearby_graspable,
    extract_frozen_config,
    parse_obs_scene,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logging.getLogger("h5py").setLevel(logging.WARNING)


def remap_video_path(video_path: str, prefix_remaps: list[tuple[str, str]]) -> str:
    """Apply path prefix rewrites (e.g. Mac client path -> Linux server path)."""
    for old, new in prefix_remaps:
        if video_path.startswith(old):
            return new + video_path[len(old):]
    return video_path


def get_outcome(rec: dict) -> str | None:
    """Extract outcome label from either schema:
    - human-annotated: top-level "annotation"
    - VLM-output:      "result.outcome"
    """
    if "annotation" in rec:
        return rec["annotation"]
    if isinstance(rec.get("result"), dict):
        return rec["result"].get("outcome")
    return None


def locate_h5_for_record(video_path: str) -> Path | None:
    """Map a rollout video_path
        .../house_X/episode_NNNNNNNN_<cam>_batch_K_of_M.mp4
    to its trajectories H5
        .../house_X/trajectories_batch_K_of_M.h5
    Returns None if the batch suffix can't be parsed.
    """
    p = Path(video_path)
    m = re.search(r"_(batch_\d+_of_\d+)\.mp4$", p.name)
    if not m:
        return None
    return p.parent / f"trajectories_{m.group(1)}.h5"


def episode_to_traj_key(episode_str: str) -> str:
    """'episode_00000000' -> 'traj_0'."""
    digits = episode_str.replace("episode_", "")
    return f"traj_{int(digits)}"


def annotate_with_density(
    annotation_file: Path,
    radius_m: float,
    prefix_remaps: list[tuple[str, str]],
) -> list[dict]:
    """Read JSONL, attach num_nearby_graspable per record. Caches open H5s."""
    enriched: list[dict] = []
    h5_cache: dict[Path, h5py.File] = {}
    n_total = 0
    n_no_h5 = 0
    n_no_traj = 0
    n_compute_fail = 0

    try:
        with open(annotation_file) as fjsonl:
            for line in fjsonl:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_total += 1

                video = rec.get("video_path")
                episode = rec.get("episode")
                rec["num_nearby_graspable"] = None

                if not video or not episode:
                    n_no_h5 += 1
                    enriched.append(rec)
                    continue

                video = remap_video_path(video, prefix_remaps)
                h5_path = locate_h5_for_record(video)
                if h5_path is None or not h5_path.exists():
                    n_no_h5 += 1
                    enriched.append(rec)
                    continue

                if h5_path not in h5_cache:
                    try:
                        h5_cache[h5_path] = h5py.File(h5_path, "r")
                    except OSError as e:
                        log.warning(f"Failed to open {h5_path}: {e}")
                        n_no_h5 += 1
                        enriched.append(rec)
                        continue

                f5 = h5_cache[h5_path]
                tk = episode_to_traj_key(episode)
                if tk not in f5:
                    n_no_traj += 1
                    enriched.append(rec)
                    continue

                try:
                    obs_scene = parse_obs_scene(f5[tk]["obs_scene"][()])
                    fc = extract_frozen_config(obs_scene)
                    rec["num_nearby_graspable"] = compute_num_nearby_graspable(
                        fc, radius_m
                    )
                except Exception as e:
                    n_compute_fail += 1
                    log.debug(f"density compute failed for {h5_path}/{tk}: {e}")

                enriched.append(rec)
    finally:
        for f5 in h5_cache.values():
            try:
                f5.close()
            except Exception:
                pass

    log.info(
        f"Processed {n_total} records: "
        f"no_h5={n_no_h5}, no_traj={n_no_traj}, compute_fail={n_compute_fail}"
    )
    return enriched


# Order outcomes from "best" to "worst" so the bar chart reads naturally.
OUTCOME_ORDER = [
    "grasp_right_object_success",
    "grasp_right_object_fail_idiosyncratic",
    "grasp_right_object_fail_clutter",
    "grasp_wrong_object",
    "no_grasp_attempt",
]


def _ordered_outcomes(records: list[dict]) -> list[str]:
    """All outcomes seen, with known ones in OUTCOME_ORDER and the rest appended."""
    seen = {o for r in records if (o := get_outcome(r)) is not None}
    ordered = [o for o in OUTCOME_ORDER if o in seen]
    ordered += sorted(o for o in seen if o not in OUTCOME_ORDER)
    return ordered


def _bar_panel(ax, records: list[dict], title: str, outcomes: list[str]) -> None:
    counts = Counter(o for r in records if (o := get_outcome(r)) is not None)
    n = sum(counts.values())
    pcts = [100.0 * counts.get(o, 0) / n if n else 0.0 for o in outcomes]
    bars = ax.bar(range(len(outcomes)), pcts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcomes, rotation=30, ha="right")
    ax.set_ylabel("% of episodes in group")
    ax.set_title(f"{title} (n={n})")
    ax.set_ylim(0, max(pcts + [1.0]) * 1.2)
    for bar, o, pct in zip(bars, outcomes, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{counts.get(o, 0)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_split_histograms(
    enriched: list[dict],
    threshold: int,
    radius_m: float,
    output_dir: Path,
) -> dict:
    """Plot two side-by-side outcome histograms (<= vs > threshold) and a grouped
    comparison plot. Returns a dict of summary counts.
    """
    valid = [r for r in enriched if r.get("num_nearby_graspable") is not None]
    low = [r for r in valid if r["num_nearby_graspable"] <= threshold]
    high = [r for r in valid if r["num_nearby_graspable"] > threshold]

    log.info(f"Total annotated records: {len(enriched)}")
    log.info(f"Records with computable density: {len(valid)}")
    log.info(f"<= {threshold} nearby: {len(low)}")
    log.info(f">  {threshold} nearby: {len(high)}")

    outcomes = _ordered_outcomes(valid)

    # Side-by-side panels (one histogram per density group).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    _bar_panel(axes[0], low, f"<= {threshold} nearby graspable", outcomes)
    _bar_panel(axes[1], high, f">  {threshold} nearby graspable", outcomes)
    fig.suptitle(
        f"Outcome distribution by nearby-graspable density "
        f"(radius={radius_m:.3f} m, threshold={threshold})",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    side_path = output_dir / f"outcomes_split_threshold{threshold}.png"
    plt.savefig(side_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved: {side_path}")

    # Grouped comparison (both groups overlaid for easy comparison).
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(outcomes))
    width = 0.4
    n_low = max(1, len(low))
    n_high = max(1, len(high))
    low_counts = Counter(o for r in low if (o := get_outcome(r)) is not None)
    high_counts = Counter(o for r in high if (o := get_outcome(r)) is not None)
    low_pcts = [100.0 * low_counts.get(o, 0) / n_low for o in outcomes]
    high_pcts = [100.0 * high_counts.get(o, 0) / n_high for o in outcomes]
    bars_low = ax.bar(
        x - width / 2,
        low_pcts,
        width,
        color="steelblue",
        edgecolor="black",
        label=f"<= {threshold} nearby (n={len(low)})",
    )
    bars_high = ax.bar(
        x + width / 2,
        high_pcts,
        width,
        color="indianred",
        edgecolor="black",
        label=f">  {threshold} nearby (n={len(high)})",
    )
    for bar, pct, c in zip(bars_low, low_pcts, [low_counts.get(o, 0) for o in outcomes]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{c}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, pct, c in zip(bars_high, high_pcts, [high_counts.get(o, 0) for o in outcomes]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{c}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=30, ha="right")
    ax.set_ylabel("% of episodes in group")
    ax.set_title(
        f"Outcome distribution: <= {threshold} vs > {threshold} nearby "
        f"(radius={radius_m:.3f} m)"
    )
    ax.legend()
    plt.tight_layout()
    cmp_path = output_dir / f"outcomes_compare_threshold{threshold}.png"
    plt.savefig(cmp_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved: {cmp_path}")

    summary = {
        "radius_m": radius_m,
        "threshold": threshold,
        "n_total_records": len(enriched),
        "n_with_density": len(valid),
        f"n_lte_{threshold}": len(low),
        f"n_gt_{threshold}": len(high),
        "outcomes": outcomes,
        f"counts_lte_{threshold}": {o: int(low_counts.get(o, 0)) for o in outcomes},
        f"counts_gt_{threshold}": {o: int(high_counts.get(o, 0)) for o in outcomes},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split rollout annotations by nearby-graspable density and "
        "plot outcome histograms."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to rollout_eval.jsonl (or any JSONL with the same schema).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Density threshold: episodes with <= threshold go to one bucket, "
        "the rest to the other. Default: 2",
    )
    parser.add_argument(
        "--nearby_radius_m",
        type=float,
        default=0.12,
        help="Radius (3D Euclidean, meters) for the nearby-graspable count. "
        "Default: 0.12 (matches create_json_benchmark default).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for histogram PNGs and the enriched JSONL. "
        "Default: alongside annotation_file.",
    )
    parser.add_argument(
        "--video_path_prefix_remap",
        action="append",
        default=[],
        help=(
            "Rewrite video_path prefixes when locating H5 files, e.g. "
            "'/Users/arjunguru/weka/=/weka/prior/aguru/' to translate Mac "
            "client paths to Linux server paths. May be passed multiple times."
        ),
    )

    args = parser.parse_args()

    prefix_remaps: list[tuple[str, str]] = []
    for spec in args.video_path_prefix_remap:
        if "=" not in spec:
            raise ValueError(
                f"--video_path_prefix_remap must be 'OLD=NEW', got: {spec!r}"
            )
        old, new = spec.split("=", 1)
        prefix_remaps.append((old, new))

    annotation_file = Path(args.annotation_file).resolve()
    if not annotation_file.exists():
        raise FileNotFoundError(annotation_file)

    output_dir = (
        Path(args.output_dir) if args.output_dir else annotation_file.parent
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Annotation file: {annotation_file}")
    log.info(f"Output dir:      {output_dir}")
    log.info(f"Radius:          {args.nearby_radius_m:.3f} m")
    log.info(f"Threshold:       {args.threshold}")

    enriched = annotate_with_density(
        annotation_file, args.nearby_radius_m, prefix_remaps=prefix_remaps
    )

    enriched_path = output_dir / f"{annotation_file.stem}_with_density.jsonl"
    with open(enriched_path, "w") as f:
        for r in enriched:
            f.write(json.dumps(r) + "\n")
    log.info(f"Saved enriched annotations: {enriched_path}")

    summary = plot_split_histograms(
        enriched,
        threshold=args.threshold,
        radius_m=args.nearby_radius_m,
        output_dir=output_dir,
    )

    summary_path = (
        output_dir / f"outcomes_summary_threshold{args.threshold}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary: {summary_path}")

    # Final at-a-glance log.
    print()
    log.info("=" * 60)
    log.info("Outcome counts:")
    for o in summary["outcomes"]:
        lo = summary[f"counts_lte_{args.threshold}"][o]
        hi = summary[f"counts_gt_{args.threshold}"][o]
        log.info(f"  {o:<42s}  <={args.threshold}: {lo:>4d}   >{args.threshold}: {hi:>4d}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
