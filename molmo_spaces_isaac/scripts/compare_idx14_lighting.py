#!/usr/bin/env python3
"""Compare idx14 MuJoCo and Arena lighting at the OpenPI image boundary.

This diagnostic intentionally works from saved policy traces/videos so it can be
rerun without launching Isaac Sim.  It answers two questions:

1. What lights are present in the MuJoCo base XML and converted Arena USD?
2. How much of the policy image mismatch looks like a brightness/exposure
   mismatch versus geometry/camera/material mismatch?
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MUJOCO_SHOULDER = Path(
    "/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/"
    "mujoco_slice/PiRemotePolicyEvalConfig/20260505_231442/house_20/"
    "episode_00000001_droid_shoulder_light_randomization_batch_1_of_1.mp4"
)
DEFAULT_MUJOCO_WRIST = Path(
    "/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/"
    "mujoco_slice/PiRemotePolicyEvalConfig/20260505_231442/house_20/"
    "episode_00000001_wrist_camera_zed_mini_batch_1_of_1.mp4"
)
DEFAULT_ARENA_TRACE = Path(
    "/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/arena_trace_run/pi_trace"
)
DEFAULT_ARENA_USD_CONTENTS = Path(
    "/home/horde/.molmospaces/usd/scenes/ithor/20260121/"
    "FloorPlan20_physics/Payload/Contents.usda"
)
DEFAULT_MUJOCO_BASE_XML = Path("molmo_spaces/resources/base_scene.xml")
DEFAULT_OUT_DIR = Path("/home/horde/molmo-proj/diagnostics/idx14_lighting_compare")


@dataclass
class ImagePair:
    name: str
    mujoco: np.ndarray
    arena: np.ndarray


def _resize_with_pad(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    img = np.asarray(img[:, :, :3], dtype=np.uint8)
    h, w = img.shape[:2]
    if h == height and w == width:
        return img
    ratio = max(w / width, h / height)
    new_w, new_h = int(w / ratio), int(h / ratio)
    resized = Image.fromarray(img).resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    out = Image.new("RGB", (width, height), 0)
    out.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(out, dtype=np.uint8)


def _read_video_frame(path: Path, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_index = max(0, min(frame_index, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_index} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _luma(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _image_stats(img: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(img, dtype=np.float64)
    lum = _luma(img)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
        lum = lum[np.asarray(mask, dtype=bool)]
        if arr.size == 0:
            arr = np.asarray(img, dtype=np.float64).reshape(-1, 3)
            lum = _luma(img).reshape(-1)
    else:
        arr = arr.reshape(-1, 3)
    return {
        "rgb_mean": arr.reshape(-1, 3).mean(axis=0).round(4).tolist(),
        "rgb_std": arr.reshape(-1, 3).std(axis=0).round(4).tolist(),
        "luma_mean": float(lum.mean()),
        "luma_std": float(lum.std()),
        "luma_p01": float(np.percentile(lum, 1)),
        "luma_p05": float(np.percentile(lum, 5)),
        "luma_p50": float(np.percentile(lum, 50)),
        "luma_p95": float(np.percentile(lum, 95)),
        "luma_p99": float(np.percentile(lum, 99)),
        "dark_pct_luma_lt_5": float(np.mean(lum < 5.0) * 100.0),
        "bright_pct_luma_gt_250": float(np.mean(lum > 250.0) * 100.0),
        "saturated_channel_pct_gt_250": float(np.mean(arr > 250.0) * 100.0),
    }


def _diff_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    la = _luma(a)
    lb = _luma(b)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        aa = aa[m]
        bb = bb[m]
        la = la[m]
        lb = lb[m]
        if aa.size == 0:
            aa = np.asarray(a, dtype=np.float64)
            bb = np.asarray(b, dtype=np.float64)
            la = _luma(a)
            lb = _luma(b)
    diff = aa - bb
    ad = np.abs(diff)
    return {
        "rgb_mean_abs": float(ad.mean()),
        "rgb_rms": float(math.sqrt(np.mean(diff**2))),
        "rgb_max_abs": float(ad.max()),
        "luma_mean_abs": float(np.abs(la - lb).mean()),
        "luma_rms": float(math.sqrt(np.mean((la - lb) ** 2))),
    }


def _fit_gain(src: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(src, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(x, y) / denom)


def _fit_gain_bias(src: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = np.asarray(src, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    A = np.stack([x, np.ones_like(x)], axis=1)
    gain, bias = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(gain), float(bias)


def _apply_gain_bias(img: np.ndarray, gain: float, bias: float = 0.0) -> np.ndarray:
    out = np.asarray(img, dtype=np.float64) * gain + bias
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _content_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask out OpenPI resize padding; keep real rendered content from either image."""
    return (_luma(a) > 5.0) | (_luma(b) > 5.0)


def _candidate_exposure_search(arena: np.ndarray, mujoco: np.ndarray) -> list[dict[str, float]]:
    candidates = []
    for gain in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10]:
        corrected = _apply_gain_bias(arena, gain)
        stats = _diff_stats(mujoco, corrected)
        candidates.append(
            {
                "gain": float(gain),
                "bias": 0.0,
                "rgb_mean_abs": stats["rgb_mean_abs"],
                "luma_mean_abs": stats["luma_mean_abs"],
            }
        )
    candidates.sort(key=lambda x: (x["luma_mean_abs"], x["rgb_mean_abs"]))
    return candidates


def _parse_mujoco_xml_lights(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    visual = root.find("visual")
    headlight = {}
    quality = {}
    if visual is not None:
        head = visual.find("headlight")
        if head is not None:
            headlight = dict(head.attrib)
        q = visual.find("quality")
        if q is not None:
            quality = dict(q.attrib)
    lights = [dict(el.attrib) for el in root.findall(".//light")]
    return {
        "source": str(path),
        "visual_headlight": headlight,
        "visual_quality": quality,
        "explicit_lights": lights,
        "explicit_light_count": len(lights),
    }


def _parse_usda_lights(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="ignore")
    lights: list[dict[str, Any]] = []
    pattern = re.compile(r'def\s+(\w*Light)\s+"([^"]+)"\s*\{(.*?)\n\s*\}', re.S)
    attr_re = re.compile(r"(?:color3f|float|token|asset)\s+([\w:]+)\s*=\s*(.+)")
    for match in pattern.finditer(text):
        light_type, name, body = match.groups()
        attrs: dict[str, str] = {}
        for line in body.splitlines():
            line = line.strip()
            attr_match = attr_re.match(line)
            if attr_match:
                attrs[attr_match.group(1)] = attr_match.group(2).strip()
        lights.append({"type": light_type, "name": name, "attributes": attrs})
    return lights


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(np.asarray(arr[:, :, :3], dtype=np.uint8)).save(path)


def _save_montage(path: Path, tiles: list[tuple[str, np.ndarray]], cols: int = 3) -> None:
    tile_w, tile_h = 224, 224
    label_h = 28
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + label_h)
        canvas.paste(Image.fromarray(np.asarray(arr, dtype=np.uint8)), (x, y + label_h))
        draw.rectangle((x, y, x + tile_w, y + label_h), fill=(245, 245, 245))
        draw.text((x + 7, y + 8), label[:38], fill=(0, 0, 0))
    canvas.save(path)


def _save_hist_plot(path: Path, pair: ImagePair, corrected: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    bins = np.linspace(0, 255, 65)
    ax.hist(_luma(pair.mujoco).reshape(-1), bins=bins, alpha=0.45, density=True, label="MuJoCo")
    ax.hist(_luma(pair.arena).reshape(-1), bins=bins, alpha=0.45, density=True, label="Arena raw")
    for label, img in corrected.items():
        ax.hist(_luma(img).reshape(-1), bins=bins, alpha=0.28, density=True, label=label)
    ax.set_title(f"{pair.name}: luma distribution")
    ax.set_xlabel("luma")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _build_pair_metrics(pair: ImagePair, out_dir: Path) -> dict[str, Any]:
    pair_dir = out_dir / pair.name
    pair_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(pair_dir / "mujoco.png", pair.mujoco)
    _save_rgb(pair_dir / "arena_raw.png", pair.arena)

    content_mask = _content_mask(pair.mujoco, pair.arena)
    gain = _fit_gain(pair.arena, pair.mujoco)
    gain_img = _apply_gain_bias(pair.arena, gain)
    gb_gain, gb_bias = _fit_gain_bias(pair.arena, pair.mujoco)
    gb_img = _apply_gain_bias(pair.arena, gb_gain, gb_bias)
    luma_gain = _fit_gain(_luma(pair.arena), _luma(pair.mujoco))
    luma_gain_img = _apply_gain_bias(pair.arena, luma_gain)
    content_luma_gain = _fit_gain(_luma(pair.arena)[content_mask], _luma(pair.mujoco)[content_mask])
    content_luma_gain_img = _apply_gain_bias(pair.arena, content_luma_gain)

    corrected = {
        f"gain {gain:.3f}": gain_img,
        f"gain+bias {gb_gain:.3f},{gb_bias:.1f}": gb_img,
        f"luma gain {luma_gain:.3f}": luma_gain_img,
        f"content luma gain {content_luma_gain:.3f}": content_luma_gain_img,
    }
    for label, img in corrected.items():
        safe = label.replace(" ", "_").replace(",", "_").replace("+", "plus")
        _save_rgb(pair_dir / f"arena_{safe}.png", img)
    _save_montage(
        pair_dir / "montage.png",
        [
            ("MuJoCo", pair.mujoco),
            ("Arena raw", pair.arena),
            (f"Arena gain {gain:.3f}", gain_img),
            (f"Arena gain+bias {gb_gain:.3f},{gb_bias:.1f}", gb_img),
            (f"Arena luma gain {luma_gain:.3f}", luma_gain_img),
            (f"Arena content gain {content_luma_gain:.3f}", content_luma_gain_img),
        ],
        cols=3,
    )
    _save_hist_plot(pair_dir / "luma_hist.png", pair, corrected)

    return {
        "content_mask_pixel_count": int(content_mask.sum()),
        "content_mask_pixel_pct": float(content_mask.mean() * 100.0),
        "mujoco": _image_stats(pair.mujoco),
        "arena_raw": _image_stats(pair.arena),
        "mujoco_content": _image_stats(pair.mujoco, content_mask),
        "arena_raw_content": _image_stats(pair.arena, content_mask),
        "raw_diff": _diff_stats(pair.mujoco, pair.arena),
        "raw_diff_content": _diff_stats(pair.mujoco, pair.arena, content_mask),
        "best_gain": {
            "gain": gain,
            "diff": _diff_stats(pair.mujoco, gain_img),
        },
        "best_gain_bias": {
            "gain": gb_gain,
            "bias": gb_bias,
            "diff": _diff_stats(pair.mujoco, gb_img),
        },
        "best_luma_gain": {
            "gain": luma_gain,
            "diff": _diff_stats(pair.mujoco, luma_gain_img),
            "diff_content": _diff_stats(pair.mujoco, luma_gain_img, content_mask),
        },
        "best_content_luma_gain": {
            "gain": content_luma_gain,
            "diff": _diff_stats(pair.mujoco, content_luma_gain_img),
            "diff_content": _diff_stats(pair.mujoco, content_luma_gain_img, content_mask),
        },
        "gain_grid_best": _candidate_exposure_search(pair.arena, pair.mujoco)[:5],
        "artifacts": {
            "montage": str(pair_dir / "montage.png"),
            "hist": str(pair_dir / "luma_hist.png"),
            "mujoco": str(pair_dir / "mujoco.png"),
            "arena_raw": str(pair_dir / "arena_raw.png"),
        },
    }


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>idx14 lighting comparison</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:28px auto;line-height:1.4}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px 8px}"
        "code{background:#f4f4f4;padding:1px 4px}img{max-width:100%;border:1px solid #ddd}</style>",
        "</head><body>",
        "<h1>idx14 MuJoCo vs Arena Lighting Comparison</h1>",
        "<h2>Light Inventory</h2>",
        "<h3>MuJoCo base XML</h3>",
        f"<pre>{html.escape(json.dumps(summary['lights']['mujoco'], indent=2))}</pre>",
        "<h3>Arena USD lights</h3>",
        f"<pre>{html.escape(json.dumps(summary['lights']['arena_usd'], indent=2))}</pre>",
        "<h2>Policy Image Metrics</h2>",
        "<table><tr><th>camera</th><th>MuJoCo luma mean</th><th>Arena luma mean</th>"
        "<th>MuJoCo content luma</th><th>Arena content luma</th>"
        "<th>raw RGB MAE</th><th>best gain</th><th>MAE after gain</th>"
        "<th>best gain+bias</th><th>MAE after gain+bias</th></tr>",
    ]
    for name, metrics in summary["pairs"].items():
        lines.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics['mujoco']['luma_mean']:.2f}</td>"
            f"<td>{metrics['arena_raw']['luma_mean']:.2f}</td>"
            f"<td>{metrics['mujoco_content']['luma_mean']:.2f}</td>"
            f"<td>{metrics['arena_raw_content']['luma_mean']:.2f}</td>"
            f"<td>{metrics['raw_diff']['rgb_mean_abs']:.2f}</td>"
            f"<td>{metrics['best_gain']['gain']:.3f}</td>"
            f"<td>{metrics['best_gain']['diff']['rgb_mean_abs']:.2f}</td>"
            f"<td>{metrics['best_gain_bias']['gain']:.3f}, {metrics['best_gain_bias']['bias']:.1f}</td>"
            f"<td>{metrics['best_gain_bias']['diff']['rgb_mean_abs']:.2f}</td>"
            "</tr>"
        )
    lines.extend(["</table>"])
    lines.append("<h2>Interpretation</h2>")
    for item in summary["interpretation"]:
        lines.append(f"<p>{html.escape(item)}</p>")
    for name, metrics in summary["pairs"].items():
        lines.append(f"<h2>{html.escape(name.title())}</h2>")
        lines.append(f"<img src='{_rel(Path(metrics['artifacts']['montage']), out_dir)}'>")
        lines.append(f"<img src='{_rel(Path(metrics['artifacts']['hist']), out_dir)}'>")
    lines.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(lines))

    md = [
        "# idx14 MuJoCo vs Arena Lighting Comparison",
        "",
        "## Findings",
        "",
        *[f"- {item}" for item in summary["interpretation"]],
        "",
        "## Artifacts",
        "",
        f"- HTML report: `{out_dir / 'report.html'}`",
        f"- Summary JSON: `{out_dir / 'summary.json'}`",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco_shoulder_video", type=Path, default=DEFAULT_MUJOCO_SHOULDER)
    parser.add_argument("--mujoco_wrist_video", type=Path, default=DEFAULT_MUJOCO_WRIST)
    parser.add_argument("--arena_trace_dir", type=Path, default=DEFAULT_ARENA_TRACE)
    parser.add_argument("--arena_chunk_index", type=int, default=0)
    parser.add_argument("--mujoco_video_frame_index", type=int, default=0)
    parser.add_argument("--arena_usd_contents", type=Path, default=DEFAULT_ARENA_USD_CONTENTS)
    parser.add_argument("--mujoco_base_xml", type=Path, default=DEFAULT_MUJOCO_BASE_XML)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_dir = args.arena_trace_dir / f"chunk_{args.arena_chunk_index:04d}"
    pairs = [
        ImagePair(
            name="exterior",
            mujoco=_resize_with_pad(_read_video_frame(args.mujoco_shoulder_video, args.mujoco_video_frame_index)),
            arena=_resize_with_pad(_read_image(chunk_dir / "exterior_image_1_left.png")),
        ),
        ImagePair(
            name="wrist",
            mujoco=_resize_with_pad(_read_video_frame(args.mujoco_wrist_video, args.mujoco_video_frame_index)),
            arena=_resize_with_pad(_read_image(chunk_dir / "wrist_image_left.png")),
        ),
    ]

    pair_metrics = {pair.name: _build_pair_metrics(pair, out_dir) for pair in pairs}
    combined_arena_luma_gain = float(
        np.mean([pair_metrics[name]["best_luma_gain"]["gain"] for name in pair_metrics])
    )
    combined_content_luma_gain = float(
        np.mean([pair_metrics[name]["best_content_luma_gain"]["gain"] for name in pair_metrics])
    )
    combined_gain_bias_gain = float(
        np.mean([pair_metrics[name]["best_gain_bias"]["gain"] for name in pair_metrics])
    )

    exterior = pair_metrics["exterior"]
    wrist = pair_metrics["wrist"]
    interpretation = [
        (
            "The converted Arena FloorPlan20 USD has a DistantLight plus a warm DomeLight; "
            "the MuJoCo base XML uses a headlight/ambient-diffuse setup. These are not a matched lighting rig."
        ),
        (
            f"Exterior luma mean is MuJoCo {exterior['mujoco']['luma_mean']:.1f} vs Arena "
            f"{exterior['arena_raw']['luma_mean']:.1f}; wrist luma mean is MuJoCo "
            f"{wrist['mujoco']['luma_mean']:.1f} vs Arena {wrist['arena_raw']['luma_mean']:.1f}."
        ),
        (
            f"Content-only luma means are exterior MuJoCo {exterior['mujoco_content']['luma_mean']:.1f} "
            f"vs Arena {exterior['arena_raw_content']['luma_mean']:.1f}, and wrist MuJoCo "
            f"{wrist['mujoco_content']['luma_mean']:.1f} vs Arena {wrist['arena_raw_content']['luma_mean']:.1f}."
        ),
        (
            f"A simple full-image luma gain averages about {combined_arena_luma_gain:.3f}; "
            f"content-only luma gain averages about {combined_content_luma_gain:.3f}; "
            f"gain+bias averages gain {combined_gain_bias_gain:.3f}. This means lighting/exposure "
            "is a real contributor, but brightness alone does not remove most of the image error."
        ),
        (
            "Recommended fix path: expose runtime Arena scene-light scaling, then test dimming the "
            "USD DistantLight/DomeLight instead of changing policy thresholds. Start with a light "
            f"scale near {combined_content_luma_gain:.2f}, then rerender and rerun idx14 policy I/O."
        ),
    ]

    summary: dict[str, Any] = {
        "inputs": {
            "mujoco_shoulder_video": str(args.mujoco_shoulder_video),
            "mujoco_wrist_video": str(args.mujoco_wrist_video),
            "arena_trace_dir": str(args.arena_trace_dir),
            "arena_chunk_index": int(args.arena_chunk_index),
            "mujoco_video_frame_index": int(args.mujoco_video_frame_index),
            "arena_usd_contents": str(args.arena_usd_contents),
            "mujoco_base_xml": str(args.mujoco_base_xml),
        },
        "lights": {
            "mujoco": _parse_mujoco_xml_lights(args.mujoco_base_xml),
            "arena_usd": {
                "source": str(args.arena_usd_contents),
                "lights": _parse_usda_lights(args.arena_usd_contents),
            },
        },
        "pairs": pair_metrics,
        "recommended_initial_light_scale": combined_arena_luma_gain,
        "recommended_initial_content_light_scale": combined_content_luma_gain,
        "interpretation": interpretation,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(out_dir, summary)
    print(f"[lighting_compare] wrote {out_dir / 'report.html'}", flush=True)
    print(f"[lighting_compare] wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
