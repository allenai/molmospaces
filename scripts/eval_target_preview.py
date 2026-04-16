"""Generate a per-episode PNG preview of the target object for an eval results folder.

For each `house_*/episode_<N>_exo_camera_1_*.mp4` under a results folder, this script:
  - Loads the first frame of the exo camera video.
  - Looks up the corresponding `traj_<N>` in the batch's h5 file to read the task
    description and the 2D projected points of the pickup object.
  - Saves `<results>/<house>/episode_<N>_target.png` (alongside the eval videos)
    containing the full exo frame with the target object boxed, a zoomed-in crop
    of the target, and the task description / object name as overlaid text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPISODE_RE = re.compile(r"episode_(\d+)_exo_camera_1_batch_(\d+)_of_(\d+)\.mp4$")


def _parse_obs_scene(raw: bytes) -> dict:
    """obs_scene is a JSON blob with a pickled `frozen_config` string — that string
    can contain unescaped control bytes that break json.loads. Strip it before
    parsing so we can still read task_description / referral_expressions."""
    text = raw.decode("utf-8", errors="replace")
    # Drop the frozen_config entry (it's opaque base64-ish pickled data).
    text = re.sub(r',\s*"frozen_config"\s*:\s*"[^"]*"', "", text)
    return json.loads(text)


def _target_name(scene: dict) -> str:
    refs = scene.get("referral_expressions", {}).get("pickup_obj_name")
    if refs:
        return refs[0][0]
    return scene.get("task_description", "unknown target")


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _points_bbox(points: np.ndarray, num_valid: int, w: int, h: int) -> tuple[int, int, int, int] | None:
    if num_valid <= 0:
        return None
    pts = points[:num_valid]
    xs = pts[:, 0] * w
    ys = pts[:, 1] * h
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    pad = max(20.0, 0.05 * max(w, h))
    x0 = max(0, int(x0 - pad)); y0 = max(0, int(y0 - pad))
    x1 = min(w, int(x1 + pad)); y1 = min(h, int(y1 + pad))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _render_preview(frame: np.ndarray, bbox, task_desc: str, target_name: str) -> Image.Image:
    h, w = frame.shape[:2]
    base = Image.fromarray(frame).convert("RGB")

    annotated = base.copy()
    draw = ImageDraw.Draw(annotated)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 64, 64), width=4)

    crop_w = w // 2
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        crop = base.crop(bbox).resize((crop_w, crop_w), Image.LANCZOS)
    else:
        crop = Image.new("RGB", (crop_w, crop_w), (30, 30, 30))
        ImageDraw.Draw(crop).text((10, crop_w // 2), "no target points", fill=(255, 255, 255))

    header_h = 110
    total_w = w + crop_w + 30
    total_h = max(h, crop_w) + header_h + 20
    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))

    title_font = _load_font(28)
    body_font = _load_font(22)
    cdraw = ImageDraw.Draw(canvas)
    cdraw.text((20, 15), f"target: {target_name}", fill=(255, 220, 120), font=title_font)
    cdraw.text((20, 55), task_desc, fill=(230, 230, 230), font=body_font)

    canvas.paste(annotated, (10, header_h))
    canvas.paste(crop, (w + 20, header_h))
    cdraw.text((w + 20, header_h + crop_w + 4), "target crop", fill=(200, 200, 200), font=body_font)
    return canvas


def _pick_step(num_ds: np.ndarray, preferred: int) -> int:
    """Return `preferred` if its num_points > 0, else the nearest valid step."""
    T = num_ds.shape[0]
    preferred = max(0, min(preferred, T - 1))
    if int(num_ds[preferred][0]) > 0:
        return preferred
    for offset in range(1, T):
        for k in (preferred - offset, preferred + offset):
            if 0 <= k < T and int(num_ds[k][0]) > 0:
                return k
    return preferred


def _process_house(house_dir: Path, frame_index: int, collected: list[tuple[Path, Path]], resume: bool) -> int:
    h5_files = sorted(house_dir.glob("trajectories_batch_*.h5"))
    if not h5_files:
        return 0
    written = 0
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            # Pair traj_<i> with episode_<N>_exo_camera_1_<same batch>.mp4.
            batch_tag = h5_path.stem.replace("trajectories_", "")  # e.g. batch_1_of_1
            mp4s = sorted(house_dir.glob(f"episode_*_exo_camera_1_{batch_tag}.mp4"))
            for traj_key, mp4_path in zip(traj_keys, mp4s):
                m = EPISODE_RE.search(mp4_path.name)
                episode_id = m.group(1) if m else traj_key
                out_path = house_dir / f"episode_{episode_id}_target.png"

                if resume and out_path.exists():
                    collected.append((mp4_path, out_path))
                    print(f"[skip] {out_path} (exists)")
                    continue

                try:
                    scene = _parse_obs_scene(bytes(f[f"{traj_key}/obs_scene"][()]))
                except Exception as e:
                    print(f"[warn] {house_dir.name}/{traj_key}: scene parse failed: {e}")
                    continue
                task_desc = scene.get("task_description", "")
                target_name = _target_name(scene)

                pts_ds = f.get(f"{traj_key}/obs/extra/object_image_points/pickup_obj/exo_camera_1/points")
                num_ds = f.get(f"{traj_key}/obs/extra/object_image_points/pickup_obj/exo_camera_1/num_points")

                step = frame_index
                if num_ds is not None:
                    step = _pick_step(np.asarray(num_ds), frame_index)

                try:
                    frame = iio.imread(mp4_path, index=step)
                except Exception as e:
                    print(f"[warn] {mp4_path}: frame {step} read failed: {e}")
                    continue
                h, w = frame.shape[:2]

                bbox = None
                if pts_ds is not None and num_ds is not None:
                    bbox = _points_bbox(np.asarray(pts_ds[step]), int(num_ds[step][0]), w, h)

                preview = _render_preview(frame, bbox, task_desc, target_name)
                preview.save(out_path)
                collected.append((mp4_path, out_path))
                written += 1
                print(f"[ok] {out_path}")
    return written


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(1, int(round(w * target_h / h)))
    return np.asarray(Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS))


def _stitch_video(pairs: list[tuple[Path, Path]], out_path: Path, k: int) -> None:
    sampled = pairs[::k]
    if not sampled:
        print("[stitch] no episodes to sample, skipping")
        return

    import imageio.v2 as iio2  # FFMPEG reader/writer lives cleanly under v2.

    fps = 30.0
    try:
        reader0 = iio2.get_reader(str(sampled[0][0]), "ffmpeg")
        meta = reader0.get_meta_data()
        fps = float(meta.get("fps", fps))
        reader0.close()
    except Exception:
        pass

    first_frame = iio.imread(sampled[0][0], index=0, plugin="FFMPEG")
    panel_h = first_frame.shape[0]

    writer = iio2.get_writer(
        str(out_path),
        format="ffmpeg",
        fps=fps * 3.0,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    try:
        for idx, (mp4_path, png_path) in enumerate(sampled):
            target_img = np.asarray(Image.open(png_path).convert("RGB"))
            target_resized = _resize_to_height(target_img, panel_h)
            print(f"[stitch] {idx + 1}/{len(sampled)}: {mp4_path.parent.name}/{mp4_path.name}")
            reader = iio2.get_reader(str(mp4_path), "ffmpeg")
            try:
                for frame in reader:
                    frame_resized = _resize_to_height(np.asarray(frame), panel_h)
                    combined = np.concatenate([frame_resized, target_resized], axis=1)
                    hh, ww = combined.shape[:2]
                    pad_h = hh % 2
                    pad_w = ww % 2
                    if pad_h or pad_w:
                        combined = np.pad(combined, ((0, pad_h), (0, pad_w), (0, 0)))
                    writer.append_data(combined)
            finally:
                reader.close()
    finally:
        writer.close()
    print(f"[stitch] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="Eval results folder (contains house_* subfolders)")
    ap.add_argument("--frame-index", type=int, default=0,
                    help="Trajectory step to sample for both the video frame and the projected object points (default: 0)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip episodes whose target png already exists (still includes them in stitching)")
    ap.add_argument("--stitch-every", type=int, default=0,
                    help="If > 0, also write a stitched mp4 sampling every k-th episode with the target preview side-by-side")
    ap.add_argument("--stitch-out", type=Path, default=None,
                    help="Output path for the stitched video (default: <results_dir>/stitched_every_<k>.mp4)")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"not a directory: {results_dir}")
    total = 0
    collected: list[tuple[Path, Path]] = []
    for house_dir in sorted(results_dir.glob("house_*")):
        if house_dir.is_dir():
            total += _process_house(house_dir, args.frame_index, collected, args.resume)
    print(f"wrote {total} previews under {results_dir}")

    if args.stitch_every and args.stitch_every > 0:
        out_path = args.stitch_out or (results_dir / f"stitched_every_{args.stitch_every}.mp4")
        _stitch_video(collected, out_path, args.stitch_every)


if __name__ == "__main__":
    main()
