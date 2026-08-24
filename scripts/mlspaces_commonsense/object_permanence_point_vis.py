"""Visualize object_permanence_point_eval.py results.

For each episode, writes one PNG: the frames the model was given, tiled in
order, with the model's predicted point drawn on the last tile (the frame it
was asked to answer in). The ground-truth mug centre is drawn there too, so the
tile can be read on its own.

Usage:
    python scripts/object_permanence_point_vis.py \\
        --results op_point_eval.jsonl --output-dir op_point_vis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("object_permanence_point_vis")

GT_COLOR = (95, 190, 60)  # BGR, mug the ball is actually under
PREDICTION_COLOR = (46, 68, 220)  # BGR, model's point
TILE_WIDTH = 480
COLUMNS = 4


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_frames(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted, last = set(indices), max(indices)
    frames: dict[int, np.ndarray] = {}
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
                frames[t] = frame
            t += 1
    finally:
        cap.release()
    return frames


def draw_marker(image: np.ndarray, point_xy: list[float], color, label: str, cross: bool) -> None:
    height, width = image.shape[:2]
    px = (int(point_xy[0] * width), int(point_xy[1] * height))
    radius = max(width // 40, 9)
    if cross:
        cv2.drawMarker(image, px, (20, 20, 20), cv2.MARKER_TILTED_CROSS, radius * 2, 6)
        cv2.drawMarker(image, px, color, cv2.MARKER_TILTED_CROSS, radius * 2, 2)
    else:
        cv2.circle(image, px, radius, (20, 20, 20), 5)
        cv2.circle(image, px, radius, color, 2)
    origin = (px[0] + radius + 5, px[1] - radius - 3)
    for thickness, tone in ((4, (20, 20, 20)), (1, color)):
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_DUPLEX, 0.6, tone, thickness, cv2.LINE_AA)


def caption(image: np.ndarray, text: str) -> None:
    for thickness, tone in ((4, (20, 20, 20)), (1, (255, 255, 255))):
        cv2.putText(image, text, (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, tone, thickness, cv2.LINE_AA)


def build_tiles(record: dict, tile_width: int, columns: int) -> np.ndarray:
    indices = record["frame_indices"]
    frames = read_frames(Path(record["video_path"]), indices)
    missing = [i for i in indices if i not in frames]
    if missing:
        raise RuntimeError(f"missing frames {missing}")

    tiles = []
    for position, index in enumerate(indices):
        frame = frames[index]
        height = round(frame.shape[0] * tile_width / frame.shape[1])
        tile = cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA)
        label = f"frame {index}"
        if position == 0:
            label += "  (mugs in mid-air)"
        if position == len(indices) - 1:
            label += "  <- model answers here"
            draw_marker(tile, record["gt_correct_mug_xy"], GT_COLOR, "ball is here", cross=False)
            draw_marker(tile, record["prediction_xy"], PREDICTION_COLOR, "model", cross=True)
        caption(tile, label)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (60, 60, 60), 1)
        tiles.append(tile)

    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:  # pad the last row so hstack works
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="JSONL from object_permanence_point_eval.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the per-episode PNGs.")
    parser.add_argument("--model", default=None, help="Only visualize records from this model.")
    parser.add_argument("--frame-mode", default=None, help="Only visualize records from this frame mode.")
    parser.add_argument("--tile-width", type=int, default=TILE_WIDTH)
    parser.add_argument("--columns", type=int, default=COLUMNS)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    records = [r for r in read_jsonl(args.results) if "prediction_xy" in r]
    if args.model:
        records = [r for r in records if r.get("model") == args.model]
    if args.frame_mode:
        records = [r for r in records if r.get("frame_mode") == args.frame_mode]
    if not records:
        log.error("No scored records in %s matching the filters.", args.results)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        try:
            sheet = build_tiles(record, args.tile_width, args.columns)
        except Exception as e:  # noqa: BLE001
            log.error("Skipping %s: %s", record.get("house"), e)
            continue
        verdict = "correct" if record["correct"] else "WRONG"
        # Key on house + episode so multiple episodes per house don't overwrite
        # each other.
        episode = record.get("episode", "episode_0")
        out_path = args.output_dir / f"{record['house']}_{episode}_{verdict}.png"
        cv2.imwrite(str(out_path), sheet)
        log.info("%s -> %s", record["house"], out_path)

    log.info("Wrote %d sheets to %s", len(records), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
