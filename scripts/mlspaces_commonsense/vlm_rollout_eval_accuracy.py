"""Quantitative VLM classification accuracy vs human annotations.

Loads a human-annotated JSONL (from ``vlm_rollout_annotate.py``), runs the
Gemini VLM on each annotated video (reuses ``process_job`` from
``vlm_rollout_eval.py``), then prints a confusion matrix and a per-class
classification accuracy table.

Usage:
    export GEMINI_API_KEY=...
    # One-shot: run inference in-memory and score.
    python scripts/mlspaces_commonsense/vlm_rollout_eval_accuracy.py \\
        --annotations human_annotations.jsonl \\
        --report vlm_accuracy_report.txt

    # Persistent / resumable: append predictions to a file so re-runs skip
    # already-predicted episodes.
    python scripts/mlspaces_commonsense/vlm_rollout_eval_accuracy.py \\
        --annotations human_annotations.jsonl \\
        --predictions vlm_predictions.jsonl

Pass ``--skip-vlm`` (requires ``--predictions``) to reuse an existing
predictions file without hitting the API again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_rollout_eval import (  # noqa: E402
    RESPONSE_SCHEMA,
    VideoJob,
    process_job,
)

log = logging.getLogger("vlm_rollout_eval_accuracy")

CLASSES: list[str] = list(RESPONSE_SCHEMA["properties"]["outcome"]["enum"])


def load_latest_per_key(
    path: Path, *, drop_errors: bool = True,
) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return latest
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if drop_errors and rec.get("error"):
                continue
            key = (rec.get("house", ""), rec.get("episode", ""))
            latest[key] = rec
    return latest


def build_jobs_from_annotations(annotations: dict[tuple[str, str], dict]) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for rec in annotations.values():
        video_path = Path(rec.get("video_path", ""))
        target_image_path = Path(rec.get("target_image_path", ""))
        if not video_path.is_file():
            log.warning("Skipping (video missing): %s", video_path)
            continue
        if not target_image_path.is_file():
            log.warning("Skipping (target missing): %s", target_image_path)
            continue
        jobs.append(VideoJob(
            house=rec.get("house", ""),
            episode=rec.get("episode", ""),
            video_path=video_path,
            target_image_path=target_image_path,
        ))
    return jobs


def run_vlm(
    jobs: list[VideoJob],
    predictions_path: Path | None,
    model: str,
    concurrency: int,
    debug_image_dir: Path | None,
) -> dict[tuple[str, str], dict]:
    """Run the VLM on ``jobs`` and return predictions keyed by (house, episode).

    If ``predictions_path`` is provided, records are appended to it (resumable:
    already-predicted jobs are skipped). If ``None``, predictions are collected
    in-memory only.
    """
    from google import genai  # lazy import so --skip-vlm doesn't need the dep

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set GEMINI_API_KEY in the environment.")
    client = genai.Client(api_key=api_key)

    if predictions_path is not None:
        prior = load_latest_per_key(predictions_path)
        done = set(prior.keys())
    else:
        prior = {}
        done = set()
    pending = [j for j in jobs if (j.house, j.episode) not in done]
    log.info("VLM: %d annotated videos, %d already predicted, %d pending.",
             len(jobs), len(done), len(pending))

    results: dict[tuple[str, str], dict] = dict(prior)
    if not pending:
        return results

    out_f = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = predictions_path.open("a")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(process_job, j, client, model, debug_image_dir): j
                for j in pending
            }
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                job = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "house": job.house, "episode": job.episode,
                        "video_path": str(job.video_path), "model": model,
                        "error": f"unhandled: {e}",
                    }
                if out_f is not None:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                if not rec.get("error"):
                    results[(job.house, job.episode)] = rec
                status = rec.get("result", {}).get("outcome") or rec.get("error", "?")
                log.info("[%d/%d] %s %s -> %s (%.1fs)",
                         i, len(pending), job.house, job.episode, status,
                         rec.get("elapsed_s", 0.0))
    finally:
        if out_f is not None:
            out_f.close()
    return results


def score(
    annotations: dict[tuple[str, str], dict],
    predictions: dict[tuple[str, str], dict],
    canonical_labels: list[str],
) -> tuple[
    dict[str, Counter],            # confusion[true][pred] = count
    Counter,                       # per_class_total (true label)
    Counter,                       # per_class_correct (true == pred)
    int,                           # matched pair count
    list[tuple[str, str]],         # unmatched annotation keys (no prediction)
    list[str],                     # ordered label universe
]:
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_total: Counter = Counter()
    per_correct: Counter = Counter()
    matched = 0
    unmatched: list[tuple[str, str]] = []
    seen_labels: set[str] = set(canonical_labels)

    for key, ann in annotations.items():
        pred_rec = predictions.get(key)
        if pred_rec is None:
            unmatched.append(key)
            continue
        true_label = ann.get("annotation")
        pred_label = (pred_rec.get("result") or {}).get("outcome")
        if not true_label or not pred_label:
            continue
        seen_labels.update([true_label, pred_label])
        confusion[true_label][pred_label] += 1
        per_total[true_label] += 1
        if true_label == pred_label:
            per_correct[true_label] += 1
        matched += 1

    extras = sorted(seen_labels - set(canonical_labels))
    labels = canonical_labels + extras
    return confusion, per_total, per_correct, matched, unmatched, labels


def format_confusion_matrix(
    confusion: dict[str, Counter], labels: list[str],
) -> str:
    """Render with letter column headers + a legend, since class names are long."""
    if not labels:
        return "(no labels)"
    letters = list(string.ascii_uppercase)
    if len(labels) > len(letters):
        # Degrade gracefully for unexpectedly many labels.
        letters = [f"L{i}" for i in range(len(labels))]
    key = {labels[i]: letters[i] for i in range(len(labels))}

    label_w = max(len(label) for label in labels) + 2
    col_w = 6
    corner = "true \\ pred"
    header = (
        f"{corner:<{label_w + 4}}"
        + "".join(f"{key[label]:>{col_w}}" for label in labels)
        + f"{'total':>{col_w + 1}}"
    )
    lines = ["Legend:"]
    for label in labels:
        lines.append(f"  {key[label]} = {label}")
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for true_label in labels:
        row = confusion.get(true_label, Counter())
        total = sum(row.values())
        cells = "".join(f"{row.get(p, 0):>{col_w}}" for p in labels)
        prefix = f"{key[true_label]} {true_label:<{label_w}}  "
        lines.append(f"{prefix}{cells}{total:>{col_w + 1}}")
    return "\n".join(lines)


def format_accuracy_table(
    per_total: Counter, per_correct: Counter, labels: list[str],
) -> str:
    label_w = max(len(label) for label in labels) + 2
    header = f"{'class':<{label_w}} {'correct':>8} {'total':>7} {'accuracy':>10}"
    lines = [header, "-" * len(header)]
    for label in labels:
        tot = per_total.get(label, 0)
        cor = per_correct.get(label, 0)
        if tot == 0:
            acc = "n/a"
        else:
            acc = f"{100.0 * cor / tot:8.2f}%"
        lines.append(f"{label:<{label_w}} {cor:>8} {tot:>7} {acc:>10}")
    total_cor = sum(per_correct.values())
    total_tot = sum(per_total.values())
    overall = "n/a" if total_tot == 0 else f"{100.0 * total_cor / total_tot:8.2f}%"
    lines.append("-" * len(header))
    lines.append(f"{'overall':<{label_w}} {total_cor:>8} {total_tot:>7} {overall:>10}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="JSONL from vlm_rollout_annotate.py.")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Optional JSONL of VLM predictions. If given, appended to (resumable); "
                             "pass an existing file plus --skip-vlm to score without re-running the "
                             "API. If omitted, inference runs fully in-memory for a one-shot eval.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write the confusion matrix + accuracy table to.")
    parser.add_argument("--model", default="gemini-robotics-er-1.6-preview",
                        help="Gemini model id (default: %(default)s).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent Gemini requests (default: %(default)s).")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Do not call the VLM; score using existing --predictions as-is.")
    parser.add_argument("--debug-image-dir", type=Path, default=None,
                        help="Forwarded to vlm_rollout_eval.process_job; dumps sent images per query.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.annotations.is_file():
        print(f"ERROR: annotations not found: {args.annotations}", file=sys.stderr)
        return 2

    annotations = load_latest_per_key(args.annotations, drop_errors=False)
    log.info("Loaded %d annotated videos from %s", len(annotations), args.annotations)
    if not annotations:
        print("ERROR: no annotations found.", file=sys.stderr)
        return 2

    if args.skip_vlm and args.predictions is None:
        print("ERROR: --skip-vlm requires --predictions.", file=sys.stderr)
        return 2

    if args.skip_vlm:
        predictions = load_latest_per_key(args.predictions)
        log.info("Loaded %d successful VLM predictions from %s",
                 len(predictions), args.predictions)
    else:
        jobs = build_jobs_from_annotations(annotations)
        predictions = run_vlm(
            jobs=jobs,
            predictions_path=args.predictions,
            model=args.model,
            concurrency=args.concurrency,
            debug_image_dir=args.debug_image_dir,
        )
        if args.predictions is not None:
            log.info("Wrote predictions to %s (%d kept for scoring).",
                     args.predictions, len(predictions))
        else:
            log.info("Scored %d in-memory predictions (no --predictions path given).",
                     len(predictions))

    confusion, per_total, per_correct, matched, unmatched, labels = score(
        annotations, predictions, CLASSES,
    )

    report_lines: list[str] = []
    report_lines.append(f"Annotations: {args.annotations} ({len(annotations)} unique videos)")
    pred_source = str(args.predictions) if args.predictions is not None else "(in-memory)"
    report_lines.append(f"Predictions: {pred_source} ({len(predictions)} successful)")
    report_lines.append(f"Matched pairs: {matched}")
    if unmatched:
        report_lines.append(
            f"Unmatched annotations (no successful prediction): {len(unmatched)}"
        )
        for house, episode in unmatched[:10]:
            report_lines.append(f"  - {house} {episode}")
        if len(unmatched) > 10:
            report_lines.append(f"  ... and {len(unmatched) - 10} more")
    report_lines.append("")
    report_lines.append("Confusion matrix (rows = human label, cols = VLM prediction):")
    report_lines.append(format_confusion_matrix(confusion, labels))
    report_lines.append("")
    report_lines.append("Per-class classification accuracy:")
    report_lines.append(format_accuracy_table(per_total, per_correct, labels))

    report = "\n".join(report_lines)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n")
        log.info("Wrote report to %s", args.report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
