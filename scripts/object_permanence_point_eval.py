"""Object-permanence pointing eval: which mug is the ball hidden under?

Each ``mug_ball_pick`` episode starts with two upside-down mugs frozen in
mid-air above a table and a small yellow ball resting on the table directly
beneath exactly one of them. The mugs then fall straight down: one lands on
top of the ball and hides it completely, the other lands on bare table.

This script samples frames from the exo-camera rollout video -- always
including frame 0, where the mugs are still in mid-air and the ball is
visible -- and asks a Gemini model to point, *in the last frame provided*, at
the mug the ball is now hidden under. A correct answer therefore requires
remembering where the ball went after it stopped being visible.

Ground truth comes from the episode HDF5: the task sampler sets the pickup
object to the correct mug, so ``obs/extra/object_image_points/pickup_obj`` is
the correct mug's projection and ``.../added_1`` is the distractor's. A
prediction is scored correct when it is closer to the correct mug's centroid
than to the distractor's, in the same frame the model was asked about.

Only the first ``--end-frame`` frames (default 12, i.e. the first ~6 seconds at
2 fps) are ever shown. The robot is still far from the mugs that early in the
rollout, so the answer is never given away by the manipulation itself.

Usage:
    export GEMINI_API_KEY=...
    python scripts/object_permanence_point_eval.py \\
        --eval-dir eval_output/objpermanence_cap/CAPPolicyEvalConfig/20260807_150608 \\
        --output op_point_eval.jsonl --debug-image-dir /tmp/op_frames

    # control: hide the mid-air frame, so the model can only guess
    python scripts/object_permanence_point_eval.py ... --frame-mode drop-first

    # re-print the summary for an existing run
    python scripts/object_permanence_point_eval.py --summarize op_point_eval.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from google import genai
from google.genai import types
from PIL import Image

log = logging.getLogger("object_permanence_point_eval")

CORRECT_KEY = "pickup_obj"  # the mug covering the ball (task pickup object)
DISTRACTOR_KEY = "added_1"  # the other mug

DEFAULT_NUM_FRAMES = 8
DEFAULT_END_FRAME = 12  # only the start of the rollout is shown; the robot is nowhere near the mugs yet
DEFAULT_SETTLE_FRAME = 4  # mugs have landed by here (video is 2 fps)


SCENE_DESCRIPTION = """You are looking at {n} frames, in chronological order, from a fixed camera watching a tabletop scene in a simulated house. Two mugs are placed upside-down on the table, and a small yellow ball is on the table under one of them.
"""

MIDAIR_DESCRIPTION = """In the FIRST frame the two mugs are frozen in mid-air above the table, and the small yellow ball is visible on the table surface below them. The ball lies directly beneath exactly one of the two mugs.

Over the following frames both mugs fall straight down and land on the table. Neither mug moves sideways while falling. One mug lands on top of the yellow ball and hides it completely; the other lands on bare table and hides nothing. After landing, neither mug is moved.
"""

TASK_INSTRUCTION = """Your task: in the LAST frame provided (frame {n} of {n}), point to the mug that the yellow ball is hidden under.

Reason about it like this: find the ball in the first frame, work out which of the two mugs comes down on top of it, then track that same mug through the remaining frames. The two mugs look very similar, so identify the correct one by its position, not by its appearance.

Report a single point that lands on the correct mug AS IT APPEARS IN THE LAST FRAME, in [y, x] format normalized to 0-1000 (y measured from the top of the image, x from the left)."""

GUESS_INSTRUCTION = """Your task: in the LAST frame provided (frame {n} of {n}), point to the mug that the yellow ball is hidden under.

The ball is never visible in any frame you are given -- it is already hidden under one of the mugs. Make your best judgement and commit to one of the two mugs.

Report a single point that lands on the chosen mug AS IT APPEARS IN THE LAST FRAME, in [y, x] format normalized to 0-1000 (y measured from the top of the image, x from the left)."""


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": (
                "Where was the ball in the first frame, which mug fell onto it, and where "
                "is that mug in the last frame? Two or three sentences."
            ),
        },
        "mug_description": {
            "type": "STRING",
            "description": "Short description of the chosen mug in the last frame, e.g. 'the left mug, closer to the camera'.",
        },
        "point": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
            "min_items": 2,
            "max_items": 2,
            "description": "[y, x] of the chosen mug in the LAST frame, normalized 0-1000.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in [0, 1] that the ball is under the mug pointed at.",
        },
    },
    "property_ordering": ["reasoning", "mug_description", "point", "confidence"],
    "required": ["reasoning", "mug_description", "point", "confidence"],
}


@dataclass(frozen=True)
class Episode:
    house: str
    episode: str
    traj_key: str
    video_path: Path
    h5_path: Path
    camera: str


def discover_episodes(eval_dir: Path, camera_hint: str = "exo") -> list[Episode]:
    """Find one job per exo video, pairing it with its trajectory in the house HDF5."""
    episodes: list[Episode] = []
    house_dirs = sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("house_"))
    for house_dir in house_dirs:
        h5_files = sorted(house_dir.glob("*.h5"))
        if not h5_files:
            log.warning("No .h5 in %s, skipping.", house_dir)
            continue
        h5_path = h5_files[0]
        videos = [
            v for v in sorted(house_dir.glob(f"*{camera_hint}*.mp4")) if "depth" not in v.name
        ]
        for video in videos:
            m = re.match(r"episode_(\d+)", video.stem)
            idx = int(m.group(1)) if m else 0
            camera = "exo_camera_1"
            cam_match = re.search(r"(exo_camera_\d+|wrist_camera)", video.stem)
            if cam_match:
                camera = cam_match.group(1)
            episodes.append(
                Episode(
                    house=house_dir.name,
                    episode=video.stem.split("_batch")[0],
                    traj_key=f"traj_{idx}",
                    video_path=video,
                    h5_path=h5_path,
                    camera=camera,
                )
            )
    return episodes


def load_mug_tracks(episode: Episode) -> dict[str, np.ndarray]:
    """Per-frame image-space centroids of both mugs, shape (T, 2), normalized [0, 1].

    Frames where an object is fully occluded (no projected points) come back as NaN.
    """
    tracks: dict[str, np.ndarray] = {}
    with h5py.File(episode.h5_path, "r") as f:
        if episode.traj_key not in f:
            raise KeyError(f"{episode.traj_key} not in {episode.h5_path} (has {list(f.keys())[:5]})")
        group = f[f"{episode.traj_key}/obs/extra/object_image_points"]
        for key in (CORRECT_KEY, DISTRACTOR_KEY):
            if key not in group:
                raise KeyError(
                    f"'{key}' missing from object_image_points in {episode.h5_path} "
                    f"(has {list(group.keys())})"
                )
            cam_group = group[f"{key}/{episode.camera}"]
            points = cam_group["points"][:]  # (T, P, 2)
            counts = cam_group["num_points"][:].reshape(-1)  # (T,)
            centroids = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
            for t, n in enumerate(counts):
                if n <= 0:
                    continue
                visible = points[t, : int(n)]
                visible = visible[~np.isnan(visible).any(axis=1)]
                if len(visible):
                    centroids[t] = visible.mean(axis=0)
            tracks[key] = centroids
    return tracks


def centroid_at(track: np.ndarray, frame: int) -> np.ndarray | None:
    """Centroid at ``frame``, falling back to the nearest earlier visible frame."""
    for t in range(min(frame, len(track) - 1), -1, -1):
        if not np.isnan(track[t]).any():
            return track[t]
    return None


def static_centroid(track: np.ndarray, start: int, end: int) -> np.ndarray | None:
    """Median centroid over [start, end].

    The mugs do not move between landing and the end frame, so pooling frames
    beats any single frame: each frame only carries 10 randomly sampled mask
    pixels, which puts roughly +/-0.02 of noise on a per-frame centroid.
    """
    window = track[max(start, 0) : end + 1]
    window = window[~np.isnan(window).any(axis=1)]
    if not len(window):
        return centroid_at(track, end)
    return np.median(window, axis=0)


def select_frame_indices(
    end_frame: int,
    num_frames: int,
    frame_mode: str,
    settle_frame: int = DEFAULT_SETTLE_FRAME,
) -> list[int]:
    """Frame indices to show, always ending on ``end_frame``.

    full        frame 0 (mugs mid-air, ball visible) plus evenly spaced frames after it
    drop-first  same span but starting after the mugs have landed (control condition)
    last-only   just the final frame (control condition)
    """
    if frame_mode == "last-only":
        return [end_frame]
    start = 0 if frame_mode == "full" else settle_frame
    start = min(start, end_frame)
    span = end_frame - start
    if span <= 0:
        return [end_frame]
    count = min(num_frames, span + 1)
    indices = [start + round(i * span / (count - 1)) for i in range(count)] if count > 1 else [end_frame]
    # Guarantee the mid-air frame survives rounding, and dedupe while keeping order.
    indices[0], indices[-1] = start, end_frame
    seen: set[int] = set()
    return [i for i in indices if not (i in seen or seen.add(i))]


def read_frames(video_path: Path, indices: list[int]) -> list[Image.Image]:
    """Decode only the requested frames (sequential read; these videos are short)."""
    wanted = set(indices)
    last = max(indices)
    grabbed: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    try:
        t = 0
        while t <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if t in wanted:
                grabbed[t] = frame
            t += 1
    finally:
        cap.release()
    missing = wanted - set(grabbed)
    if missing:
        raise RuntimeError(f"Could not decode frames {sorted(missing)} from {video_path}")
    return [Image.fromarray(cv2.cvtColor(grabbed[i], cv2.COLOR_BGR2RGB)) for i in indices]


def build_prompt(num_shown: int, frame_mode: str) -> str:
    scene = SCENE_DESCRIPTION.format(n=num_shown)
    if frame_mode == "full":
        return scene + "\n" + MIDAIR_DESCRIPTION + "\n" + TASK_INSTRUCTION.format(n=num_shown)
    return scene + "\n" + GUESS_INSTRUCTION.format(n=num_shown)


def frame_caption(position: int, total: int, index: int, frame_mode: str) -> str:
    tags = []
    if position == 0 and frame_mode == "full":
        tags.append("mugs in mid-air, ball visible")
    if position == total - 1:
        tags.append("LAST FRAME - give your answer in this image")
    suffix = f" ({'; '.join(tags)})" if tags else ""
    return f"Frame {position + 1} of {total} [video frame {index}]{suffix}:"


def pil_to_part(img: Image.Image, jpeg_quality: int = 92) -> types.Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def parse_point(raw: object) -> tuple[float, float]:
    """Gemini's [y, x] in 0-1000 -> (x, y) normalized to [0, 1]."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"Unparseable point: {raw!r}")
    y, x = float(raw[0]), float(raw[1])
    scale = 1.0 if max(abs(x), abs(y)) <= 1.5 else 1000.0  # tolerate 0-1 fractions
    return x / scale, y / scale


def query_model(
    client: genai.Client,
    model: str,
    frames: list[Image.Image],
    frame_indices: list[int],
    frame_mode: str,
    temperature: float,
    max_retries: int = 3,
) -> dict:
    parts: list = []
    for position, (img, index) in enumerate(zip(frames, frame_indices, strict=True)):
        parts.append(types.Part.from_text(text=frame_caption(position, len(frames), index, frame_mode)))
        parts.append(pil_to_part(img))
    prompt = build_prompt(len(frames), frame_mode)
    parts.append(types.Part.from_text(text=prompt))

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=parts, config=config)
            return json.loads((response.text or "").strip())
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            msg = str(e)
            is_client_error = (isinstance(status, int) and 400 <= status < 500) or any(
                tok in msg
                for tok in ("404", "400", "401", "403", "PERMISSION_DENIED", "INVALID_ARGUMENT", "NOT_FOUND")
            )
            if is_client_error:
                raise RuntimeError(f"Gemini client error (not retrying): {e}") from e
            last_err = e
            backoff = 2**attempt
            log.warning("Gemini call failed (attempt %d/%d): %s; retrying in %ds",
                        attempt + 1, max_retries, e, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"Gemini failed after {max_retries} attempts: {last_err}") from last_err


def dump_debug_images(
    debug_dir: Path,
    frames: list[Image.Image],
    frame_indices: list[int],
    prompt: str,
    gt: dict[str, np.ndarray | None],
    predicted: tuple[float, float] | None,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for position, (img, index) in enumerate(zip(frames, frame_indices, strict=True)):
        img.convert("RGB").save(debug_dir / f"{position:02d}_frame{index:03d}.png")
    (debug_dir / "prompt.txt").write_text(prompt)

    annotated = cv2.cvtColor(np.array(frames[-1].convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = annotated.shape[:2]
    overlays = [
        (gt.get(CORRECT_KEY), (0, 255, 0), "GT correct mug"),
        (gt.get(DISTRACTOR_KEY), (0, 165, 255), "distractor mug"),
    ]
    for point, color, label in overlays:
        if point is None:
            continue
        px = (int(point[0] * width), int(point[1] * height))
        cv2.drawMarker(annotated, px, color, cv2.MARKER_CROSS, 26, 2)
        cv2.putText(annotated, label, (px[0] + 8, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if predicted is not None:
        px = (int(predicted[0] * width), int(predicted[1] * height))
        cv2.circle(annotated, px, 9, (0, 0, 255), 2)
        cv2.putText(annotated, "prediction", (px[0] + 10, px[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(debug_dir / "last_frame_annotated.png"), annotated)


def process_episode(
    episode: Episode,
    client: genai.Client,
    args: argparse.Namespace,
) -> dict:
    started = time.time()
    record: dict = {
        "house": episode.house,
        "episode": episode.episode,
        "video_path": str(episode.video_path),
        "model": args.model,
        "frame_mode": args.frame_mode,
    }
    try:
        tracks = load_mug_tracks(episode)
        end_frame = min(args.end_frame, min(len(t) for t in tracks.values()) - 1)
        frame_indices = select_frame_indices(end_frame, args.num_frames, args.frame_mode, args.settle_frame)
        frames = read_frames(episode.video_path, frame_indices)

        # The mugs are static from landing to end_frame, so pool that window.
        gt = {
            CORRECT_KEY: static_centroid(tracks[CORRECT_KEY], args.settle_frame, end_frame),
            DISTRACTOR_KEY: static_centroid(tracks[DISTRACTOR_KEY], args.settle_frame, end_frame),
        }
        if gt[CORRECT_KEY] is None or gt[DISTRACTOR_KEY] is None:
            raise RuntimeError(f"No visible ground-truth centroid at frame {end_frame}")

        record.update({
            "frame_indices": frame_indices,
            "end_frame": end_frame,
            "gt_correct_mug_xy": [round(float(v), 4) for v in gt[CORRECT_KEY]],
            "gt_distractor_mug_xy": [round(float(v), 4) for v in gt[DISTRACTOR_KEY]],
            "gt_correct_is_left": bool(gt[CORRECT_KEY][0] < gt[DISTRACTOR_KEY][0]),
            "mug_separation": round(float(np.linalg.norm(gt[CORRECT_KEY] - gt[DISTRACTOR_KEY])), 4),
        })

        response = query_model(
            client, args.model, frames, frame_indices, args.frame_mode, args.temperature
        )
        pred = parse_point(response.get("point"))
        dist_correct = float(np.linalg.norm(np.array(pred) - gt[CORRECT_KEY]))
        dist_distractor = float(np.linalg.norm(np.array(pred) - gt[DISTRACTOR_KEY]))

        record.update({
            "prediction_xy": [round(v, 4) for v in pred],
            "dist_to_correct": round(dist_correct, 4),
            "dist_to_distractor": round(dist_distractor, 4),
            "correct": bool(dist_correct < dist_distractor),
            "predicted_left": bool(pred[0] < (gt[CORRECT_KEY][0] + gt[DISTRACTOR_KEY][0]) / 2),
            "confidence": response.get("confidence"),
            "reasoning": response.get("reasoning"),
            "mug_description": response.get("mug_description"),
            "elapsed_s": round(time.time() - started, 2),
        })

        if args.debug_image_dir is not None:
            dump_debug_images(
                args.debug_image_dir / args.frame_mode / episode.house / episode.episode,
                frames,
                frame_indices,
                build_prompt(len(frames), args.frame_mode),
                gt,
                pred,
            )
        return record
    except Exception as e:  # noqa: BLE001
        log.error("Failed on %s/%s: %s", episode.house, episode.episode, e)
        record["error"] = str(e)
        record["elapsed_s"] = round(time.time() - started, 2)
        return record


def summarize(records: list[dict]) -> str:
    scored = [r for r in records if "correct" in r]
    errors = [r for r in records if "error" in r]
    if not scored:
        return f"No scored episodes ({len(errors)} errored)."

    correct = sum(r["correct"] for r in scored)
    n = len(scored)
    accuracy = correct / n
    stderr = math.sqrt(accuracy * (1 - accuracy) / n)
    chose_left = sum(r.get("predicted_left", False) for r in scored)
    gt_left = sum(r.get("gt_correct_is_left", False) for r in scored)
    confident = [r for r in scored if isinstance(r.get("confidence"), (int, float))]
    mean_conf = sum(r["confidence"] for r in confident) / len(confident) if confident else float("nan")
    conf_when_right = [r["confidence"] for r in confident if r["correct"]]
    conf_when_wrong = [r["confidence"] for r in confident if not r["correct"]]

    lines = [
        "",
        "=" * 62,
        f"Episodes scored : {n}  (errored: {len(errors)})",
        f"Accuracy        : {correct}/{n} = {accuracy:.1%}  (+/- {stderr:.1%} s.e., chance = 50%)",
        f"Model chose left: {chose_left}/{n} = {chose_left / n:.1%}   "
        f"(ground truth left: {gt_left}/{n} = {gt_left / n:.1%})",
        f"Mean confidence : {mean_conf:.2f}"
        + (f"  (correct: {sum(conf_when_right) / len(conf_when_right):.2f}" if conf_when_right else "  (correct: n/a")
        + (f", wrong: {sum(conf_when_wrong) / len(conf_when_wrong):.2f})" if conf_when_wrong else ", wrong: n/a)"),
    ]
    by_mode: dict[str, list[dict]] = {}
    for r in scored:
        by_mode.setdefault(r.get("frame_mode", "?"), []).append(r)
    if len(by_mode) > 1:
        lines.append("-" * 62)
        for mode, rows in sorted(by_mode.items()):
            hits = sum(r["correct"] for r in rows)
            lines.append(f"  frame_mode={mode:<10} {hits}/{len(rows)} = {hits / len(rows):.1%}")
    lines.append("-" * 62)
    for r in sorted(scored, key=lambda r: (r["house"], r["episode"])):
        mark = "OK  " if r["correct"] else "MISS"
        lines.append(
            f"  {mark} {r['house']:<9} frames={str(r.get('frame_indices', []))[:34]:<34} "
            f"d_correct={r['dist_to_correct']:.3f} d_distractor={r['dist_to_distractor']:.3f} "
            f"conf={r.get('confidence')}"
        )
    lines.append("=" * 62)
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="Eval output root containing house_* subdirectories.")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL output path. Appended to; already-scored episodes are skipped.")
    parser.add_argument("--model", default="gemini-robotics-er-1.6-preview",
                        help="Gemini model id (default: %(default)s).")
    parser.add_argument("--frame-mode", default="full", choices=("full", "drop-first", "last-only"),
                        help="full: include the mid-air frame. drop-first / last-only: controls where "
                             "the ball is never visible, so accuracy should fall to chance (default: %(default)s).")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Frames shown per episode (default: %(default)s).")
    parser.add_argument("--end-frame", type=int, default=DEFAULT_END_FRAME,
                        help="Last video frame the model may see; frames are sampled from "
                             "[0, end-frame] (default: %(default)s).")
    parser.add_argument("--settle-frame", type=int, default=DEFAULT_SETTLE_FRAME,
                        help="Frame by which the mugs have landed; ground-truth mug positions are "
                             "pooled over [settle-frame, end-frame] (default: %(default)s).")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Max episodes to run (after resume filter).")
    parser.add_argument("--debug-image-dir", type=Path, default=None,
                        help="Dump the exact frames sent, the prompt, and an annotated last frame "
                             "(ground truth + prediction) per episode.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do frame selection and ground truth only; no API calls.")
    parser.add_argument("--summarize", type=Path, default=None,
                        help="Print the summary for an existing JSONL and exit.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.summarize is not None:
        print(summarize(read_jsonl(args.summarize)))
        return 0

    if args.eval_dir is None or not args.eval_dir.is_dir():
        print("ERROR: --eval-dir must point at an existing eval output directory.", file=sys.stderr)
        return 2

    episodes = discover_episodes(args.eval_dir)
    log.info("Discovered %d episodes across %d houses.", len(episodes), len({e.house for e in episodes}))

    if args.dry_run:
        for episode in episodes[: args.limit or len(episodes)]:
            try:
                tracks = load_mug_tracks(episode)
                end_frame = min(args.end_frame, min(len(t) for t in tracks.values()) - 1)
                indices = select_frame_indices(end_frame, args.num_frames, args.frame_mode, args.settle_frame)
                correct = static_centroid(tracks[CORRECT_KEY], args.settle_frame, end_frame)
                distractor = static_centroid(tracks[DISTRACTOR_KEY], args.settle_frame, end_frame)
                separation = (
                    float(np.linalg.norm(correct - distractor))
                    if correct is not None and distractor is not None
                    else float("nan")
                )
                log.info("%s: end=%d frames=%s correct=%s distractor=%s sep=%.3f",
                         episode.house, end_frame, indices,
                         np.round(correct, 3), np.round(distractor, 3), separation)
            except Exception as e:  # noqa: BLE001
                log.error("%s: %s", episode.house, e)
        return 0

    if args.output is None:
        print("ERROR: --output is required (unless --dry-run or --summarize).", file=sys.stderr)
        return 2

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY in the environment.", file=sys.stderr)
        return 2
    client = genai.Client(api_key=api_key)

    existing = read_jsonl(args.output)
    done = {
        (r.get("house"), r.get("episode"), r.get("frame_mode"), r.get("model"))
        for r in existing
        if "error" not in r
    }
    pending = [
        e for e in episodes if (e.house, e.episode, args.frame_mode, args.model) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    log.info("Running %d episodes (%d already done) with concurrency=%d, frame_mode=%s, model=%s.",
             len(pending), len(episodes) - len(pending), args.concurrency, args.frame_mode, args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_records: list[dict] = []
    with args.output.open("a") as out_f, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {pool.submit(process_episode, e, client, args): e for e in pending}
        for future in concurrent.futures.as_completed(futures):
            episode = futures[future]
            try:
                record = future.result()
            except Exception as e:  # noqa: BLE001
                record = {"house": episode.house, "episode": episode.episode,
                          "model": args.model, "frame_mode": args.frame_mode,
                          "error": f"unhandled: {e}"}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            new_records.append(record)
            status = record.get("error") or ("correct" if record.get("correct") else "wrong")
            log.info("[%d/%d] %s %s -> %s (%.1fs)", len(new_records), len(pending),
                     record["house"], record["episode"], status, record.get("elapsed_s", 0.0))

    relevant = [
        r for r in read_jsonl(args.output)
        if r.get("model") == args.model and r.get("frame_mode") == args.frame_mode
    ]
    print(summarize(relevant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
