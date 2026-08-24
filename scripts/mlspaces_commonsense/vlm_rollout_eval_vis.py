"""Sample rollout videos per VLM-classified outcome for visual inspection.

Reads a ``rollout_eval.jsonl`` produced by ``vlm_rollout_eval.py`` and, for
each outcome class the VLM emits (e.g. ``no_grasp_attempt``,
``grasp_right_object_success``, ``grasp_right_object_fail_clutter``, ...),
copies up to ``--max-per-class`` randomly sampled rollout videos into a
per-class subfolder so you can eyeball whether the VLM is classifying
correctly. Classes with fewer videos than the cap get all of them.

Usage:
    python scripts/vlm_rollout_eval_vis.py \\
        --jsonl rollout_eval.jsonl \\
        --out vlm_eval_vis \\
        --max-per-class 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("vlm_rollout_eval_vis")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"


def load_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d in %s: %s", lineno, jsonl_path, e)
    return records


def group_by_outcome(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if "error" in rec:
            groups["_error"].append(rec)
            continue
        outcome = rec.get("result", {}).get("outcome")
        if not outcome:
            groups["_missing_outcome"].append(rec)
            continue
        groups[outcome].append(rec)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=Path("rollout_eval.jsonl"),
                        help="Path to the rollout_eval.jsonl produced by vlm_rollout_eval.py.")
    parser.add_argument("--out", type=Path, default=Path("vlm_eval_vis"),
                        help="Output directory to create with per-class subfolders.")
    parser.add_argument("--max-per-class", type=int, default=10,
                        help="Max videos to randomly sample per outcome class. Classes with "
                             "fewer videos get all of them. (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible sampling (default: %(default)s).")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink videos instead of copying (faster, but breaks if source moves).")
    parser.add_argument("--include-errored", action="store_true",
                        help="Also sample records whose VLM call errored, under _error/.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.jsonl.is_file():
        print(f"ERROR: jsonl not found: {args.jsonl}", file=sys.stderr)
        return 2

    records = load_records(args.jsonl)
    log.info("Loaded %d records from %s", len(records), args.jsonl)

    groups = group_by_outcome(records)
    rng = random.Random(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    total_written = 0
    for outcome in sorted(groups.keys()):
        recs = groups[outcome]
        if outcome == "_error" and not args.include_errored:
            log.info("Skipping %d errored records (use --include-errored to include).", len(recs))
            summary_lines.append(f"{outcome}: {len(recs)} total, 0 sampled (skipped)")
            continue

        class_dir = args.out / _sanitize(outcome)
        class_dir.mkdir(parents=True, exist_ok=True)

        n_sample = min(args.max_per_class, len(recs))
        sampled = rng.sample(recs, n_sample) if n_sample < len(recs) else list(recs)
        log.info("Class %-40s %d total, sampling %d -> %s",
                 outcome, len(recs), n_sample, class_dir)

        per_class_manifest: list[dict] = []
        for rec in sampled:
            video_path = Path(rec.get("video_path", ""))
            if not video_path.is_file():
                log.warning("Missing video (skipped): %s", video_path)
                continue
            house = rec.get("house", "unknown")
            episode = rec.get("episode", "unknown")
            stem = f"{house}__{episode}"
            dst = class_dir / f"{stem}{video_path.suffix}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.symlink:
                dst.symlink_to(video_path.resolve())
            else:
                shutil.copy2(video_path, dst)

            # Companion target PNG lives alongside the video:
            # {house_dir}/{episode}_target.png
            target_src = video_path.parent / f"{episode}_target.png"
            target_dst_name: str | None = None
            if target_src.is_file():
                target_dst = class_dir / f"{stem}_target.png"
                if target_dst.exists() or target_dst.is_symlink():
                    target_dst.unlink()
                if args.symlink:
                    target_dst.symlink_to(target_src.resolve())
                else:
                    shutil.copy2(target_src, target_dst)
                target_dst_name = target_dst.name
            else:
                log.warning("Missing target png for %s/%s: %s", house, episode, target_src)

            per_class_manifest.append({
                "house": house,
                "episode": episode,
                "video_path": str(video_path),
                "target_image_path": str(target_src) if target_src.is_file() else None,
                "copied_to": dst.name,
                "target_copied_to": target_dst_name,
                "result": rec.get("result"),
                "error": rec.get("error"),
            })
            total_written += 1

        manifest_path = class_dir / "manifest.jsonl"
        with manifest_path.open("w") as mf:
            for entry in per_class_manifest:
                mf.write(json.dumps(entry) + "\n")

        summary_lines.append(
            f"{outcome}: {len(recs)} total, {len(per_class_manifest)} sampled -> {class_dir}"
        )

    summary_path = args.out / "summary.txt"
    summary_path.write_text(
        f"Source: {args.jsonl}\nTotal records: {len(records)}\nMax per class: {args.max_per_class}\n"
        f"Seed: {args.seed}\nTransfer mode: {'symlink' if args.symlink else 'copy'}\n\n"
        + "\n".join(summary_lines) + "\n"
    )
    log.info("Wrote %d videos across %d classes. Summary: %s",
             total_written, len([k for k in groups if not (k == '_error' and not args.include_errored)]),
             summary_path)
    for line in summary_lines:
        log.info("  %s", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
