"""Human annotation tool for robot policy rollout videos.

Discovers ``*exo*.mp4`` videos under an eval-output directory (same layout as
``vlm_rollout_eval.py`` expects), tiles each rollout next to its target PNG,
and lets a human classify the outcome with a single keypress. Each annotation
is appended to a JSONL file immediately (and the file is closed) so a crash
never loses completed work.

The classification space matches the enum in ``vlm_rollout_eval.py``.

Usage:
    python scripts/vlm_rollout_annotate.py \\
        --eval-dir /weka/prior/aguru/mujoco-thor/eval_output/.../20260320_022322 \\
        --output human_annotations.jsonl

Requires a display (OpenCV GUI). For SSH sessions, X11 forwarding must be set
up.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path



import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("vlm_rollout_annotate")

# Kept in sync with RESPONSE_SCHEMA["properties"]["outcome"]["enum"] in vlm_rollout_eval.py.
CLASSES: list[str] = [
    "no_grasp_attempt",
    "grasp_wrong_object",
    "grasp_right_object_success",
    "grasp_right_object_fail_clutter",
    "grasp_right_object_fail_idiosyncratic",
]


class VideoJob:
    __slots__ = ("house", "episode", "video_path", "target_image_path")

    def __init__(self, house: str, episode: str, video_path: Path, target_image_path: Path):
        self.house = house
        self.episode = episode
        self.video_path = video_path
        self.target_image_path = target_image_path


def discover_jobs(eval_dir: Path) -> list[VideoJob]:
    """Mirror of vlm_rollout_eval.discover_jobs, inlined to avoid importing genai."""
    jobs: list[VideoJob] = []
    for house_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("house_")):
        for video in sorted(house_dir.glob("*exo*.mp4")):
            parts = video.stem.split("_")
            episode = "_".join(parts[:2]) if len(parts) >= 2 and parts[0] == "episode" else video.stem
            target_image_path = house_dir / f"{episode}_target.png"
            jobs.append(VideoJob(
                house=house_dir.name,
                episode=episode,
                video_path=video,
                target_image_path=target_image_path,
            ))
    return jobs

# '1'..'N' map to the classes above, in the order defined by the VLM schema.
KEY_TO_CLASS = {ord(str(i + 1)): c for i, c in enumerate(CLASSES)}

KEY_REPLAY = ord("r")
KEY_SKIP = ord("s")
KEY_BACK = ord("b")
KEY_QUIT = ord("q")
ESC = 27

WINDOW_NAME = "vlm_rollout_annotate"


def load_existing(path: Path) -> dict[tuple[str, str], dict]:
    """Return {(house, episode): last_annotation_record}. Later lines win."""
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
            key = (rec.get("house", ""), rec.get("episode", ""))
            latest[key] = rec
    return latest


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: list[np.ndarray] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def load_target_bgr(target_path: Path) -> np.ndarray:
    img = Image.open(target_path).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_composite(
    video_frame_bgr: np.ndarray,
    target_bgr: np.ndarray,
    legend_lines: list[str],
    status_line: str,
    panel_h: int = 240,
) -> np.ndarray:
    """Tile target | video on top, legend text in a panel below."""
    vh, vw = video_frame_bgr.shape[:2]
    th, tw = target_bgr.shape[:2]
    target_w = max(1, int(tw * vh / th))
    target_resized = cv2.resize(target_bgr, (target_w, vh))

    top = np.hstack([target_resized, video_frame_bgr])
    total_w = top.shape[1]

    panel = np.zeros((panel_h, total_w, 3), dtype=np.uint8)

    cv2.putText(
        panel, status_line, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    for i, line in enumerate(legend_lines):
        y = 62 + 24 * i
        cv2.putText(
            panel, line, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA,
        )
    return np.vstack([top, panel])


def wait_for_annotation(
    video_frames: list[np.ndarray],
    fps: float,
    target_bgr: np.ndarray,
    legend_lines: list[str],
    status_line: str,
) -> int:
    """Loop video playback until the user presses a recognized key. Returns the key code.

    'r' restarts the current video from the first frame; other unrecognized keys
    are ignored so stray keypresses don't exit the loop.
    """
    frame_delay_ms = max(1, int(1000.0 / max(fps, 1.0)))
    end_linger_iters = max(1, int(1000 / 100))  # ~1s of hold on last frame

    while True:
        interrupted = False
        for frame in video_frames:
            comp = make_composite(frame, target_bgr, legend_lines, status_line)
            cv2.imshow(WINDOW_NAME, comp)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == 0xFF:
                continue
            if key == KEY_REPLAY:
                interrupted = True
                break
            if key == ESC:
                return KEY_QUIT
            if key in KEY_TO_CLASS or key in (KEY_SKIP, KEY_BACK, KEY_QUIT):
                return key
        if interrupted:
            continue

        # Linger on the final frame so slow humans have a chance to classify.
        for _ in range(end_linger_iters):
            comp = make_composite(
                video_frames[-1], target_bgr, legend_lines,
                status_line + "  [end; press key to classify, r to replay]",
            )
            cv2.imshow(WINDOW_NAME, comp)
            key = cv2.waitKey(100) & 0xFF
            if key == 0xFF:
                continue
            if key == KEY_REPLAY:
                break
            if key == ESC:
                return KEY_QUIT
            if key in KEY_TO_CLASS or key in (KEY_SKIP, KEY_BACK, KEY_QUIT):
                return key


def append_record(path: Path, rec: dict) -> None:
    """Open-append-close so a crash can't lose the line."""
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def build_legend() -> list[str]:
    lines = ["Classify outcome (rows=human label):"]
    for i, c in enumerate(CLASSES, start=1):
        lines.append(f"  [{i}] {c}")
    lines.append("")
    lines.append("Controls: [r] replay  [s] skip  [b] back  [q]/ESC quit")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Root evaluation directory containing house_* subdirectories.")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL file to append annotations to (created if absent).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle pending videos before presenting (useful for unbiased sampling).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --shuffle (default: %(default)s).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max videos to present this session (after resume filter).")
    parser.add_argument("--reannotate", action="store_true",
                        help="Present already-annotated videos too; a new annotation overrides the prior one.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.eval_dir.is_dir():
        print(f"ERROR: --eval-dir does not exist: {args.eval_dir}", file=sys.stderr)
        return 2

    jobs = discover_jobs(args.eval_dir)
    log.info("Discovered %d exo videos across %d houses.",
             len(jobs), len({j.house for j in jobs}))

    existing = load_existing(args.output)
    if existing:
        log.info("%d prior annotations loaded from %s.", len(existing), args.output)

    if args.reannotate:
        pending = list(jobs)
    else:
        pending = [j for j in jobs if (j.house, j.episode) not in existing]

    if args.shuffle:
        random.Random(args.seed).shuffle(pending)

    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        log.info("Nothing to annotate. Done.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    legend_lines = build_legend()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print(f"ERROR: cv2 GUI unavailable ({e}). Need a display / X11 forwarding.",
              file=sys.stderr)
        return 2

    annotated = 0
    skipped = 0
    idx = 0
    try:
        while 0 <= idx < len(pending):
            job = pending[idx]
            try:
                frames, fps = load_video_frames(job.video_path)
                target_bgr = load_target_bgr(job.target_image_path)
            except Exception as e:  # noqa: BLE001
                log.warning("Load failed for %s (%s); auto-skipping.", job.video_path, e)
                idx += 1
                continue

            prior = existing.get((job.house, job.episode))
            status = f"[{idx + 1}/{len(pending)}] {job.house} {job.episode}"
            if prior is not None:
                status += f"  (prior: {prior.get('annotation', '?')})"

            key = wait_for_annotation(frames, fps, target_bgr, legend_lines, status)

            if key == KEY_QUIT:
                log.info("Quit requested.")
                break
            if key == KEY_SKIP:
                skipped += 1
                log.info("[%d/%d] %s %s skipped", idx + 1, len(pending), job.house, job.episode)
                idx += 1
                continue
            if key == KEY_BACK:
                idx = max(0, idx - 1)
                continue
            if key in KEY_TO_CLASS:
                cls = KEY_TO_CLASS[key]
                rec = {
                    "house": job.house,
                    "episode": job.episode,
                    "video_path": str(job.video_path),
                    "target_image_path": str(job.target_image_path),
                    "annotation": cls,
                    "annotated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                append_record(args.output, rec)
                existing[(job.house, job.episode)] = rec
                annotated += 1
                log.info("[%d/%d] %s %s -> %s",
                         idx + 1, len(pending), job.house, job.episode, cls)
                idx += 1
                continue
            # Unrecognized (shouldn't happen given wait_for_annotation's filter)
            log.debug("Ignoring unrecognized key %d.", key)
    finally:
        cv2.destroyAllWindows()

    log.info("Session complete: %d annotated, %d skipped. Annotations at %s.",
             annotated, skipped, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
