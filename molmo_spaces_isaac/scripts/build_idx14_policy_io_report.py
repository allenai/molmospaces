#!/usr/bin/env python3
"""Build a MuJoCo-vs-Arena policy I/O report for iTHOR pick episode idx 14."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_MUJOCO_H5 = Path(
    "/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/"
    "mujoco_slice/PiRemotePolicyEvalConfig/20260505_231442/house_20/"
    "trajectories_batch_1_of_1.h5"
)
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
    "/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/"
    "arena_trace_run/pi_trace"
)
DEFAULT_ARENA_RESULT = Path(
    "/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/"
    "arena_trace_run/result_subset10_idx14.json"
)
DEFAULT_ARENA_EXTERNAL_VIDEO = Path(
    "/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/"
    "videos/arena/arena_ep0010_external_camera_rgb.mp4"
)
DEFAULT_ARENA_WRIST_VIDEO = Path(
    "/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff/"
    "videos/arena/arena_ep0010_wrist_camera_rgb.mp4"
)
DEFAULT_SUMMARY = Path(
    "/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/"
    "mujoco_vs_arena_3eps_x10_summary.json"
)
DEFAULT_MUJOCO_LOG = Path(
    "/home/horde/molmo-proj/diagnostics/episode_sweep_3x10_20260505/"
    "mujoco_slice/PiRemotePolicyEvalConfig/20260505_231442/running_log.log"
)
DEFAULT_OUT_DIR = Path("/home/horde/molmo-proj/diagnostics/idx14_policy_io_diff")

JOINT_LABELS = [f"j{i}" for i in range(1, 8)]


def _decode_h5_json_row(row: Any) -> dict[str, Any]:
    arr = np.asarray(row)
    data = bytes(arr[arr != 0]).decode("utf-8")
    return json.loads(data) if data else {}


def _read_json_series(dataset: h5py.Dataset) -> list[dict[str, Any]]:
    return [_decode_h5_json_row(dataset[i]) for i in range(len(dataset))]


def _series_to_array(
    rows: list[dict[str, Any]],
    key: str,
    width: int,
    fill: float = np.nan,
) -> np.ndarray:
    out = np.full((len(rows), width), fill, dtype=float)
    for idx, row in enumerate(rows):
        if key not in row:
            continue
        val = np.asarray(row[key], dtype=float).reshape(-1)
        out[idx, : min(width, len(val))] = val[:width]
    return out


def _gripper_scalar(raw_gripper: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_gripper, dtype=float)
    if raw.ndim == 2:
        raw = raw[:, 0]
    return np.clip(raw / 0.824033, 0.0, 1.0)


def _resize_with_pad(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    img = np.asarray(img[:, :, :3], dtype=np.uint8)
    h, w = img.shape[:2]
    if h == height and w == width:
        return img
    ratio = max(w / width, h / height)
    new_w, new_h = int(w / ratio), int(h / ratio)
    resized = Image.fromarray(img).resize((new_w, new_h), resample=Image.BILINEAR)
    out = Image.new("RGB", (width, height), 0)
    out.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(out, dtype=np.uint8)


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_index = max(0, min(frame_index, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _video_frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _save_labeled_grid(
    out_path: Path,
    tiles: list[tuple[str, np.ndarray]],
    cols: int = 2,
    tile_size: tuple[int, int] = (224, 224),
) -> None:
    label_h = 30
    tile_w, tile_h = tile_size
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + label_h)
        img = Image.fromarray(np.asarray(arr[:, :, :3], dtype=np.uint8)).resize(
            (tile_w, tile_h), resample=Image.BILINEAR
        )
        canvas.paste(img, (x, y + label_h))
        draw.rectangle((x, y, x + tile_w, y + label_h), fill=(245, 245, 245))
        draw.text((x + 8, y + 9), label[:42], fill=(0, 0, 0))
    canvas.save(out_path)


def _save_absdiff(out_path: Path, a: np.ndarray, b: np.ndarray, title: str) -> None:
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean(axis=2)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    im = ax.imshow(diff, cmap="magma", vmin=0, vmax=max(5.0, float(diff.max())))
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean abs pixel")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _image_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    diff = np.abs(aa - bb)
    return {
        "mean_abs": float(diff.mean()),
        "rms": float(np.sqrt(np.mean((aa - bb) ** 2))),
        "max_abs": float(diff.max()),
    }


def _safe_rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _extract_mujoco_prompt(
    log_path: Path,
    fallback: str,
    house_index: int | None = None,
    rollout_index: int | None = None,
) -> str:
    if not log_path.is_file():
        return fallback
    pattern = re.compile(r"Current prompt:\s*(.*?)\s*\[")
    if house_index is not None and rollout_index is not None:
        complete_pattern = re.compile(
            rf"house\s+{house_index}\s+episode\s+{rollout_index}\s+object\s+.*completed",
        )
        lines = log_path.read_text(errors="ignore").splitlines()
        for line_idx, line in enumerate(lines):
            if not complete_pattern.search(line):
                continue
            for follow in lines[line_idx : min(len(lines), line_idx + 12)]:
                match = pattern.search(follow)
                if match:
                    return match.group(1).strip()
    last = None
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            last = match.group(1).strip()
    return last or fallback


def _read_policy_config_from_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(errors="ignore") if log_path.is_file() else ""
    out: dict[str, Any] = {}
    for key in ("chunk_size", "grasping_threshold"):
        match = re.search(rf"'{key}':\s*([0-9.]+)", text)
        if match:
            value = match.group(1)
            out[key] = float(value) if "." in value else int(value)
    return out


def _load_mujoco_trace(args: argparse.Namespace) -> dict[str, Any]:
    with h5py.File(args.mujoco_h5, "r") as h5:
        root = h5[args.mujoco_traj]
        qpos_rows = _read_json_series(root["obs/agent/qpos"])
        cmd_rows = _read_json_series(root["actions/commanded_action"])
        joint_pos_rows = _read_json_series(root["actions/joint_pos"])
        qpos_arm = _series_to_array(qpos_rows, "arm", 7)
        qpos_gripper_raw = _series_to_array(qpos_rows, "gripper", 2)
        cmd_arm = _series_to_array(cmd_rows, "arm", 7)
        cmd_gripper = _series_to_array(cmd_rows, "gripper", 1)
        joint_pos_arm = _series_to_array(joint_pos_rows, "arm", 7)
        tcp_pose = np.asarray(root["obs/extra/tcp_pose"][:], dtype=float)
        robot_base_pose = np.asarray(root["obs/extra/robot_base_pose"][:], dtype=float)
        obj_start = np.asarray(root["obs/extra/obj_start"][:], dtype=float)
        obj_end = np.asarray(root["obs/extra/obj_end"][:], dtype=float)
        success = np.asarray(root["success"][:], dtype=bool)

    rollout_match = re.search(r"(\d+)$", args.mujoco_traj)
    rollout_index = int(rollout_match.group(1)) if rollout_match else None
    mujoco_prompt_human = _extract_mujoco_prompt(
        args.mujoco_log,
        args.mujoco_prompt,
        house_index=args.mujoco_house_index,
        rollout_index=rollout_index,
    )
    mujoco_prompt_policy = mujoco_prompt_human.strip().lower()
    reset_shoulder = _resize_with_pad(_read_video_frame(args.mujoco_shoulder_video, 0))
    reset_wrist = _resize_with_pad(_read_video_frame(args.mujoco_wrist_video, 0))
    first_valid = np.where(np.isfinite(cmd_arm).all(axis=1))[0]
    first_action_index = int(first_valid[0]) if len(first_valid) else None

    return {
        "qpos_arm": qpos_arm,
        "qpos_gripper_raw": qpos_gripper_raw,
        "qpos_gripper_scalar": _gripper_scalar(qpos_gripper_raw),
        "cmd_arm": cmd_arm,
        "cmd_gripper": cmd_gripper.reshape(-1),
        "joint_pos_arm": joint_pos_arm,
        "tcp_pose": tcp_pose,
        "robot_base_pose": robot_base_pose,
        "obj_start": obj_start,
        "obj_end": obj_end,
        "success": success,
        "success_indices": np.where(success)[0].astype(int).tolist(),
        "first_action_index": first_action_index,
        "prompt_human": mujoco_prompt_human,
        "prompt_policy": mujoco_prompt_policy,
        "reset_shoulder": reset_shoulder,
        "reset_wrist": reset_wrist,
        "policy_config": _read_policy_config_from_log(args.mujoco_log),
    }


def _load_arena_trace(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = args.arena_trace_dir
    chunk_dirs = sorted(
        p for p in trace_dir.glob("chunk_*") if p.is_dir() and (p / "metadata.json").is_file()
    )
    if not chunk_dirs:
        raise FileNotFoundError(f"No Arena chunk traces found under {trace_dir}")

    chunks = []
    for chunk_dir in chunk_dirs:
        with open(chunk_dir / "metadata.json") as f:
            meta = json.load(f)
        chunks.append(
            {
                "chunk_index": int(meta.get("chunk_index", len(chunks))),
                "prompt": meta.get("prompt", ""),
                "qpos_arm": np.asarray(np.load(chunk_dir / "joint_position.npy"), dtype=float).reshape(7),
                "gripper_scalar": float(
                    np.asarray(np.load(chunk_dir / "gripper_position.npy"), dtype=float).reshape(1)[0]
                ),
                "actions": np.asarray(np.load(chunk_dir / "actions.npy"), dtype=float),
                "shoulder": _read_image(chunk_dir / "exterior_image_1_left.png"),
                "wrist": _read_image(chunk_dir / "wrist_image_left.png"),
            }
        )

    calls = []
    for call_path in sorted(trace_dir.glob("call_*.json")):
        with open(call_path) as f:
            call = json.load(f)
        calls.append(
            {
                "call_index": int(call["call_index"]),
                "chunk_index": int(call["chunk_index"]),
                "buffer_index": int(call["buffer_index"]),
                "model_output": np.asarray(call["model_output"], dtype=float).reshape(-1),
                "arm": np.asarray(call["arm"], dtype=float).reshape(7),
                "gripper": float(np.asarray(call.get("gripper", [np.nan]), dtype=float).reshape(-1)[0]),
            }
        )
    if not calls:
        raise FileNotFoundError(f"No Arena call traces found under {trace_dir}")

    chunk_call_x = []
    for chunk in chunks:
        matching = [c["call_index"] for c in calls if c["chunk_index"] == chunk["chunk_index"]]
        chunk_call_x.append(min(matching) if matching else chunk["chunk_index"] * args.pi_chunk_size)

    result = {}
    if args.arena_result_json.is_file():
        with open(args.arena_result_json) as f:
            result = json.load(f)

    return {
        "chunks": chunks,
        "calls": calls,
        "chunk_x": np.asarray(chunk_call_x, dtype=int),
        "prompt_policy": str(chunks[0]["prompt"]),
        "reset_shoulder": chunks[0]["shoulder"],
        "reset_wrist": chunks[0]["wrist"],
        "result": result,
    }


def _plot_reset_state(mujoco: dict[str, Any], arena: dict[str, Any], out_path: Path) -> None:
    m = np.concatenate([mujoco["qpos_arm"][0], [mujoco["qpos_gripper_scalar"][0]]])
    a = np.concatenate([arena["chunks"][0]["qpos_arm"], [arena["chunks"][0]["gripper_scalar"]]])
    labels = JOINT_LABELS + ["grip"]
    x = np.arange(len(labels))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), dpi=140, sharex=True)
    width = 0.36
    ax0.bar(x - width / 2, m, width, label="MuJoCo")
    ax0.bar(x + width / 2, a, width, label="Arena")
    ax0.set_ylabel("value")
    ax0.set_title("Reset policy input: joint positions and gripper scalar")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend()
    ax1.bar(x, np.abs(m - a), 0.6, color="#b45309")
    ax1.set_ylabel("abs diff")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_qpos_traces(mujoco: dict[str, Any], arena: dict[str, Any], out_path: Path) -> None:
    q_m = mujoco["qpos_arm"]
    g_m = mujoco["qpos_gripper_scalar"]
    x_m = np.arange(len(q_m))
    x_a = arena["chunk_x"]
    q_a = np.stack([c["qpos_arm"] for c in arena["chunks"]])
    g_a = np.asarray([c["gripper_scalar"] for c in arena["chunks"]])
    x_limit = max(10, min(len(q_m) - 1, 120))

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), dpi=140, sharex=True)
    axes = axes.reshape(-1)
    for i in range(7):
        ax = axes[i]
        ax.plot(x_m, q_m[:, i], label="MuJoCo", linewidth=1.6)
        mask = x_a <= x_limit
        ax.plot(x_a[mask], q_a[mask, i], "o-", label="Arena chunks", markersize=3, linewidth=1.0)
        ax.set_title(JOINT_LABELS[i])
        ax.grid(alpha=0.25)
    ax = axes[7]
    ax.plot(x_m, g_m, label="MuJoCo", linewidth=1.6)
    mask = x_a <= x_limit
    ax.plot(x_a[mask], g_a[mask], "o-", label="Arena chunks", markersize=3, linewidth=1.0)
    ax.set_title("gripper scalar")
    ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xlim(0, x_limit)
    axes[0].legend(loc="best")
    fig.supxlabel("MuJoCo row / Arena policy call index")
    fig.supylabel("policy input value")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_action_traces(mujoco: dict[str, Any], arena: dict[str, Any], out_path: Path) -> None:
    m_arm = mujoco["cmd_arm"]
    m_grip = mujoco["cmd_gripper"]
    calls = arena["calls"]
    a_x = np.asarray([c["call_index"] for c in calls], dtype=int)
    a_arm = np.stack([c["arm"] for c in calls])
    a_grip = np.asarray([c["gripper"] for c in calls], dtype=float)
    m_x = np.arange(len(m_arm))
    x_limit = max(10, min(len(m_arm) - 1, 120))
    mask_a = a_x <= x_limit

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), dpi=140, sharex=True)
    axes = axes.reshape(-1)
    for i in range(7):
        ax = axes[i]
        ax.plot(m_x, m_arm[:, i], label="MuJoCo commanded", linewidth=1.5)
        ax.plot(a_x[mask_a], a_arm[mask_a, i], label="Arena decoded", linewidth=1.0)
        ax.set_title(JOINT_LABELS[i])
        ax.grid(alpha=0.25)
    ax = axes[7]
    ax.plot(m_x, m_grip, label="MuJoCo commanded", linewidth=1.5)
    ax.plot(a_x[mask_a], a_grip[mask_a], label="Arena decoded", linewidth=1.0)
    ax.set_title("decoded gripper command")
    ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xlim(0, x_limit)
    axes[0].legend(loc="best")
    fig.supxlabel("MuJoCo row / Arena policy call index")
    fig.supylabel("commanded joint target")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_first_action(mujoco: dict[str, Any], arena: dict[str, Any], out_path: Path) -> None:
    idx = mujoco["first_action_index"]
    if idx is None:
        return
    m = np.concatenate([mujoco["cmd_arm"][idx], [mujoco["cmd_gripper"][idx]]])
    first_call = arena["calls"][0]
    a = np.concatenate([first_call["arm"], [first_call["gripper"]]])
    labels = JOINT_LABELS + ["grip"]
    x = np.arange(len(labels))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), dpi=140, sharex=True)
    width = 0.36
    ax0.bar(x - width / 2, m, width, label=f"MuJoCo row {idx}")
    ax0.bar(x + width / 2, a, width, label="Arena call 0")
    ax0.set_title("First decoded policy action")
    ax0.set_ylabel("target")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend()
    ax1.bar(x, np.abs(m - a), 0.6, color="#7c3aed")
    ax1.set_ylabel("abs diff")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_raw_gripper_score(arena: dict[str, Any], out_path: Path, threshold: float) -> None:
    calls = arena["calls"]
    x = np.asarray([c["call_index"] for c in calls], dtype=int)
    raw = np.asarray([c["model_output"][7] if len(c["model_output"]) > 7 else np.nan for c in calls])
    decoded = np.asarray([c["gripper"] for c in calls], dtype=float)
    x_limit = max(10, min(int(x.max()), 160))
    mask = x <= x_limit
    fig, ax0 = plt.subplots(figsize=(11, 4.5), dpi=140)
    ax0.plot(x[mask], raw[mask], label="Arena raw model_output[7]", color="#2563eb")
    ax0.axhline(threshold, color="#dc2626", linestyle="--", label=f"threshold {threshold:g}")
    ax0.set_ylabel("raw gripper score")
    ax0.set_xlim(0, x_limit)
    ax0.grid(alpha=0.25)
    ax1 = ax0.twinx()
    ax1.step(x[mask], decoded[mask], label="decoded gripper", color="#059669", alpha=0.65)
    ax1.set_ylabel("decoded gripper command")
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc="best")
    ax0.set_title("Arena raw gripper score and decoded command")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_mujoco_tcp_context(mujoco: dict[str, Any], out_path: Path) -> None:
    tcp = mujoco["tcp_pose"]
    obj_start = mujoco["obj_start"][0, :3]
    obj_end = mujoco["obj_end"][0, :3]
    x = np.arange(len(tcp))
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    for i, axis in enumerate("xyz"):
        ax.plot(x, tcp[:, i], label=f"tcp {axis}")
        ax.axhline(obj_start[i], color=f"C{i}", linestyle="--", alpha=0.45, label=f"obj start {axis}")
        ax.axhline(obj_end[i], color=f"C{i}", linestyle=":", alpha=0.45, label=f"obj end {axis}")
    ax.set_title("MuJoCo TCP pose against saved object start/end pose")
    ax.set_xlabel("MuJoCo row")
    ax.set_ylabel("world coordinate")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_video_snapshot_strip(
    out_path: Path,
    videos: list[tuple[str, Path]],
    fractions: list[float],
    tile_size: tuple[int, int] = (160, 90),
) -> None:
    label_w = 120
    header_h = 28
    tile_w, tile_h = tile_size
    canvas = Image.new("RGB", (label_w + tile_w * len(fractions), header_h + tile_h * len(videos)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, frac in enumerate(fractions):
        label = "end" if frac >= 1.0 else f"{int(frac * 100)}%"
        draw.text((label_w + col * tile_w + 8, 8), label, fill=(0, 0, 0))
    for row, (label, video) in enumerate(videos):
        y = header_h + row * tile_h
        draw.rectangle((0, y, label_w, y + tile_h), fill=(245, 245, 245))
        draw.text((8, y + tile_h // 2 - 6), label[:18], fill=(0, 0, 0))
        count = max(1, _video_frame_count(video))
        for col, frac in enumerate(fractions):
            idx = int(round((count - 1) * min(max(frac, 0.0), 1.0)))
            try:
                frame = _read_video_frame(video, idx)
                img = Image.fromarray(frame[:, :, :3]).resize(tile_size, resample=Image.BILINEAR)
            except Exception:
                img = Image.new("RGB", tile_size, (40, 40, 40))
            canvas.paste(img, (label_w + col * tile_w, y))
    canvas.save(out_path)


def _make_link(path: Path, text: str | None = None, base: Path | None = None) -> str:
    label = html.escape(text or path.name)
    href = html.escape(_safe_rel(path, base) if base else str(path))
    return f'<a href="{href}">{label}</a>'


def _write_report(
    args: argparse.Namespace,
    mujoco: dict[str, Any],
    arena: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Path],
) -> None:
    out_dir = args.out_dir
    rel = lambda p: html.escape(_safe_rel(p, out_dir))

    summary = metrics["benchmark_summary"]
    arena_result = arena.get("result", {})
    arena_first = (arena_result.get("results") or [{}])[0]
    prompt_exact = metrics["prompt_exact_match"]
    prompt_text = "MATCH" if prompt_exact else "MISMATCH"
    success_line = (
        f"MuJoCo idx14 slice: {summary['mujoco_successes']}/{summary['mujoco_total']} "
        f"({summary['mujoco_success_rate']:.0%}); "
        f"Arena idx14 slice: {summary['arena_successes']}/{summary['arena_total']} "
        f"({summary['arena_success_rate']:.0%})."
    )

    key_findings = [
        "Reset arm qpos is byte-level aligned at the policy boundary; max abs diff "
        f"{metrics['reset_joint_max_abs_diff']:.6g} rad.",
        "The gripper policy scalar is close but not identical at reset; abs diff "
        f"{metrics['reset_gripper_abs_diff']:.6g}.",
        "The camera inputs are visually comparable, but not pixel-identical; exterior mean abs "
        f"{metrics['image_metrics']['exterior']['mean_abs']:.2f}, wrist mean abs "
        f"{metrics['image_metrics']['wrist']['mean_abs']:.2f}.",
        "Prompt strings do not exactly match at the traced policy boundary: "
        f"MuJoCo sends {mujoco['prompt_policy']!r}, Arena sends {arena['prompt_policy']!r}.",
        "Arena action outputs diverge from the saved MuJoCo successful rollout immediately; "
        "the report plots this as decoded joint targets rather than raw MuJoCo OpenPI chunks "
        "because the old MuJoCo H5 does not contain raw model chunks.",
    ]

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>idx14 MuJoCo vs Arena policy I/O diff</title>",
        "<style>",
        "body{font-family:Inter,Arial,sans-serif;margin:32px;line-height:1.45;color:#172033;}",
        "h1,h2{margin:0.8em 0 0.35em;} h1{font-size:28px;} h2{font-size:21px;}",
        ".note{background:#f8fafc;border-left:4px solid #64748b;padding:10px 14px;margin:14px 0;}",
        ".warn{background:#fff7ed;border-left:4px solid #ea580c;padding:10px 14px;margin:14px 0;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin:16px 0;}",
        ".card{border:1px solid #d8dee9;border-radius:6px;padding:12px;background:white;}",
        "img{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:4px;}",
        "table{border-collapse:collapse;margin:12px 0;width:100%;font-size:14px;}",
        "td,th{border:1px solid #d8dee9;padding:7px;text-align:left;vertical-align:top;}",
        "th{background:#f1f5f9;} code{background:#f1f5f9;padding:1px 4px;border-radius:3px;}",
        "li{margin:6px 0;} .small{font-size:13px;color:#475569;}",
        "</style></head><body>",
        "<h1>idx14 MuJoCo vs Arena Policy I/O Diff</h1>",
        f"<p><b>Episode:</b> original idx 14, subset idx 10, house 20, task "
        f"<code>{html.escape(arena_first.get('task_description', 'Pick up the smooth gray bowl'))}</code>.</p>",
        f"<p><b>Outcome context:</b> {html.escape(success_line)} This report compares a successful MuJoCo "
        "rollout (<code>traj_1</code>, success at row 88) to the traced Arena rollout "
        f"({arena_first.get('step_count', 'unknown')} steps, success={arena_first.get('success', False)}).</p>",
        "<div class='warn'><b>Current blocker signal:</b> "
        f"prompt exact match is <b>{prompt_text}</b>; qpos parity is good; camera/action traces still need inspection.</div>",
        "<h2>Key Findings</h2>",
        "<ul>",
        *[f"<li>{html.escape(item)}</li>" for item in key_findings],
        "</ul>",
        "<h2>Boundary Settings</h2>",
        "<table><tr><th>Field</th><th>MuJoCo</th><th>Arena</th></tr>",
        f"<tr><td>Policy checkpoint</td><td colspan='2'><code>pi05_droid_jointpos</code></td></tr>",
        f"<tr><td>Prompt sent to policy</td><td><code>{html.escape(mujoco['prompt_policy'])}</code></td>"
        f"<td><code>{html.escape(arena['prompt_policy'])}</code></td></tr>",
        f"<tr><td>Chunk size</td><td>{mujoco['policy_config'].get('chunk_size', 'unknown')}</td>"
        f"<td>{args.pi_chunk_size}</td></tr>",
        f"<tr><td>Grasping threshold</td><td>{mujoco['policy_config'].get('grasping_threshold', 'unknown')}</td>"
        f"<td>{args.pi_grasping_threshold:g}</td></tr>",
        f"<tr><td>Arena action repeat</td><td>n/a</td><td>{args.pi_action_repeat}</td></tr>",
        "</table>",
        "<h2>Visual Inputs</h2>",
        "<div class='grid'>",
        f"<div class='card'><h3>Reset policy input cameras</h3><img src='{rel(artifacts['camera_tile'])}'></div>",
        f"<div class='card'><h3>Exterior pixel diff</h3><img src='{rel(artifacts['exterior_diff'])}'></div>",
        f"<div class='card'><h3>Wrist pixel diff</h3><img src='{rel(artifacts['wrist_diff'])}'></div>",
        f"<div class='card'><h3>Exterior rollout snapshots</h3><img src='{rel(artifacts['exterior_snapshots'])}'></div>",
        f"<div class='card'><h3>Wrist rollout snapshots</h3><img src='{rel(artifacts['wrist_snapshots'])}'></div>",
        "</div>",
        "<h2>Policy Inputs</h2>",
        "<div class='grid'>",
        f"<div class='card'><h3>Reset proprioception</h3><img src='{rel(artifacts['reset_state_plot'])}'></div>",
        f"<div class='card'><h3>Proprioception traces</h3><img src='{rel(artifacts['qpos_trace_plot'])}'></div>",
        "</div>",
        "<h2>Policy Outputs</h2>",
        "<div class='grid'>",
        f"<div class='card'><h3>First decoded action</h3><img src='{rel(artifacts['first_action_plot'])}'></div>",
        f"<div class='card'><h3>Decoded action traces</h3><img src='{rel(artifacts['action_trace_plot'])}'></div>",
        f"<div class='card'><h3>Arena gripper raw score</h3><img src='{rel(artifacts['arena_gripper_plot'])}'></div>",
        "</div>",
        "<h2>Extra Context</h2>",
        "<div class='grid'>",
        f"<div class='card'><h3>MuJoCo TCP/object context</h3><img src='{rel(artifacts['tcp_context_plot'])}'></div>",
        "</div>",
        "<h2>Video Links</h2>",
        "<ul>",
        f"<li>{_make_link(artifacts['mujoco_shoulder_video'], 'MuJoCo shoulder video', out_dir)}</li>",
        f"<li>{_make_link(artifacts['mujoco_wrist_video'], 'MuJoCo wrist video', out_dir)}</li>",
        f"<li>{_make_link(args.arena_external_video, 'Arena external camera video', out_dir)}</li>",
        f"<li>{_make_link(args.arena_wrist_video, 'Arena wrist camera video', out_dir)}</li>",
        "</ul>",
        "<h2>Numerical Summary</h2>",
        "<pre>",
        html.escape(json.dumps(metrics, indent=2)),
        "</pre>",
        "<p class='small'>Caveat: this is a retrospective diff from the saved MuJoCo H5/video and the new Arena pi trace. "
        "The MuJoCo file stores decoded/applied action fields, not raw OpenPI action chunks. A strict raw-chunk diff "
        "requires rerunning MuJoCo with the same trace hooks now used in Arena.</p>",
        "</body></html>",
    ]
    (out_dir / "report.html").write_text("\n".join(html_lines))

    md_lines = [
        "# idx14 MuJoCo vs Arena Policy I/O Diff",
        "",
        f"Episode: original idx 14, subset idx 10, house 20, task `{arena_first.get('task_description', 'Pick up the smooth gray bowl')}`.",
        "",
        f"Outcome context: {success_line}",
        "",
        "## Key findings",
        "",
        *[f"- {item}" for item in key_findings],
        "",
        "## Main artifacts",
        "",
        f"- HTML report: `{out_dir / 'report.html'}`",
        f"- Reset camera tile: `{artifacts['camera_tile']}`",
        f"- Reset proprioception plot: `{artifacts['reset_state_plot']}`",
        f"- Action trace plot: `{artifacts['action_trace_plot']}`",
        f"- Summary JSON: `{out_dir / 'summary.json'}`",
        "",
        "## Caveat",
        "",
        "The MuJoCo H5 stores decoded/applied action fields, not raw OpenPI chunks. "
        "A strict raw-chunk diff requires rerunning MuJoCo with matching trace hooks.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(md_lines))


def _make_video_link(link_path: Path, target: Path) -> Path:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path


def _build_metrics(
    args: argparse.Namespace,
    mujoco: dict[str, Any],
    arena: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    with open(summary_path) as f:
        bench_summary = json.load(f)["episodes"]["14"]
    reset_joint_diff = np.abs(mujoco["qpos_arm"][0] - arena["chunks"][0]["qpos_arm"])
    reset_gripper_diff = abs(float(mujoco["qpos_gripper_scalar"][0]) - float(arena["chunks"][0]["gripper_scalar"]))

    chunk_x = arena["chunk_x"]
    arena_qpos = np.stack([c["qpos_arm"] for c in arena["chunks"]])
    valid = chunk_x < len(mujoco["qpos_arm"])
    qpos_trace_metrics: dict[str, float | None] = {"mean_abs": None, "max_abs": None}
    if np.any(valid):
        qdiff = np.abs(arena_qpos[valid] - mujoco["qpos_arm"][chunk_x[valid]])
        qpos_trace_metrics = {"mean_abs": float(qdiff.mean()), "max_abs": float(qdiff.max())}

    first_idx = mujoco["first_action_index"]
    first_action_diff = None
    if first_idx is not None:
        m = np.concatenate([mujoco["cmd_arm"][first_idx], [mujoco["cmd_gripper"][first_idx]]])
        first_call = arena["calls"][0]
        a = np.concatenate([first_call["arm"], [first_call["gripper"]]])
        first_action_diff = {
            "mujoco_index": int(first_idx),
            "mean_abs": float(np.abs(m - a).mean()),
            "max_abs": float(np.abs(m - a).max()),
            "per_dim_abs": np.abs(m - a).astype(float).tolist(),
        }

    action_trace_diff: dict[str, float | None] = {"mean_abs": None, "max_abs": None}
    n = min(len(mujoco["cmd_arm"]), len(arena["calls"]))
    if n > 1:
        a_arm = np.stack([c["arm"] for c in arena["calls"][:n]])
        mask = np.isfinite(mujoco["cmd_arm"][:n]).all(axis=1)
        if np.any(mask):
            adiff = np.abs(a_arm[mask] - mujoco["cmd_arm"][:n][mask])
            action_trace_diff = {"mean_abs": float(adiff.mean()), "max_abs": float(adiff.max())}

    return {
        "episode": {
            "original_idx": 14,
            "subset_idx": 10,
            "house_index": 20,
            "task_description": bench_summary["task_description"],
            "mujoco_traj": args.mujoco_traj,
        },
        "benchmark_summary": {
            "mujoco_successes": int(bench_summary["mujoco"]["successes"]),
            "mujoco_total": int(bench_summary["mujoco"]["total"]),
            "mujoco_success_rate": float(bench_summary["mujoco"]["success_rate"]),
            "arena_successes": int(bench_summary["arena"]["successes"]),
            "arena_total": int(bench_summary["arena"]["total"]),
            "arena_success_rate": float(bench_summary["arena"]["success_rate"]),
        },
        "mujoco_success_indices": mujoco["success_indices"],
        "prompt": {
            "mujoco_policy_prompt": mujoco["prompt_policy"],
            "arena_policy_prompt": arena["prompt_policy"],
            "exact_match": mujoco["prompt_policy"] == arena["prompt_policy"],
            "match_after_terminal_punctuation_strip": mujoco["prompt_policy"].rstrip(".!?")
            == arena["prompt_policy"].rstrip(".!?"),
        },
        "prompt_exact_match": mujoco["prompt_policy"] == arena["prompt_policy"],
        "settings": {
            "mujoco_policy_config_from_log": mujoco["policy_config"],
            "arena_pi_chunk_size": int(args.pi_chunk_size),
            "arena_pi_grasping_threshold": float(args.pi_grasping_threshold),
            "arena_pi_action_repeat": int(args.pi_action_repeat),
        },
        "reset_joint_max_abs_diff": float(reset_joint_diff.max()),
        "reset_joint_mean_abs_diff": float(reset_joint_diff.mean()),
        "reset_joint_abs_diff": reset_joint_diff.astype(float).tolist(),
        "reset_gripper_abs_diff": float(reset_gripper_diff),
        "qpos_trace_at_arena_chunks": qpos_trace_metrics,
        "first_decoded_action_diff": first_action_diff,
        "decoded_action_trace_diff": action_trace_diff,
        "image_metrics": {
            "exterior": _image_metrics(mujoco["reset_shoulder"], arena["reset_shoulder"]),
            "wrist": _image_metrics(mujoco["reset_wrist"], arena["reset_wrist"]),
        },
        "paths": {
            "mujoco_h5": str(args.mujoco_h5),
            "arena_trace_dir": str(args.arena_trace_dir),
            "arena_result_json": str(args.arena_result_json),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco_h5", type=Path, default=DEFAULT_MUJOCO_H5)
    parser.add_argument("--mujoco_shoulder_video", type=Path, default=DEFAULT_MUJOCO_SHOULDER)
    parser.add_argument("--mujoco_wrist_video", type=Path, default=DEFAULT_MUJOCO_WRIST)
    parser.add_argument("--mujoco_log", type=Path, default=DEFAULT_MUJOCO_LOG)
    parser.add_argument("--mujoco_traj", default="traj_1")
    parser.add_argument("--mujoco_house_index", type=int, default=20)
    parser.add_argument("--mujoco_prompt", default="Pick up the smooth gray bowl")
    parser.add_argument("--arena_trace_dir", type=Path, default=DEFAULT_ARENA_TRACE)
    parser.add_argument("--arena_result_json", type=Path, default=DEFAULT_ARENA_RESULT)
    parser.add_argument("--arena_external_video", type=Path, default=DEFAULT_ARENA_EXTERNAL_VIDEO)
    parser.add_argument("--arena_wrist_video", type=Path, default=DEFAULT_ARENA_WRIST_VIDEO)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pi_chunk_size", type=int, default=8)
    parser.add_argument("--pi_grasping_threshold", type=float, default=0.5)
    parser.add_argument("--pi_action_repeat", type=int, default=3)
    args = parser.parse_args()

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    plots_dir = args.out_dir / "plots"
    videos_dir = args.out_dir / "videos"
    frames_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    args.mujoco_h5 = args.mujoco_h5.expanduser().resolve()
    args.mujoco_shoulder_video = args.mujoco_shoulder_video.expanduser().resolve()
    args.mujoco_wrist_video = args.mujoco_wrist_video.expanduser().resolve()
    args.mujoco_log = args.mujoco_log.expanduser().resolve()
    args.arena_trace_dir = args.arena_trace_dir.expanduser().resolve()
    args.arena_result_json = args.arena_result_json.expanduser().resolve()
    args.arena_external_video = args.arena_external_video.expanduser().resolve()
    args.arena_wrist_video = args.arena_wrist_video.expanduser().resolve()
    args.summary_json = args.summary_json.expanduser().resolve()

    mujoco = _load_mujoco_trace(args)
    arena = _load_arena_trace(args)

    artifacts: dict[str, Path] = {
        "camera_tile": frames_dir / "reset_policy_input_cameras.png",
        "exterior_diff": frames_dir / "reset_exterior_absdiff.png",
        "wrist_diff": frames_dir / "reset_wrist_absdiff.png",
        "exterior_snapshots": frames_dir / "rollout_snapshots_exterior.png",
        "wrist_snapshots": frames_dir / "rollout_snapshots_wrist.png",
        "reset_state_plot": plots_dir / "reset_policy_input_proprio.png",
        "qpos_trace_plot": plots_dir / "policy_input_qpos_traces.png",
        "first_action_plot": plots_dir / "first_decoded_action_diff.png",
        "action_trace_plot": plots_dir / "decoded_action_traces.png",
        "arena_gripper_plot": plots_dir / "arena_raw_gripper_score.png",
        "tcp_context_plot": plots_dir / "mujoco_tcp_object_context.png",
        "mujoco_shoulder_video": videos_dir / "mujoco_shoulder_traj1.mp4",
        "mujoco_wrist_video": videos_dir / "mujoco_wrist_traj1.mp4",
    }

    _save_labeled_grid(
        artifacts["camera_tile"],
        [
            ("MuJoCo shoulder/exterior", mujoco["reset_shoulder"]),
            ("Arena exterior", arena["reset_shoulder"]),
            ("MuJoCo wrist", mujoco["reset_wrist"]),
            ("Arena wrist", arena["reset_wrist"]),
        ],
    )
    _save_absdiff(
        artifacts["exterior_diff"],
        mujoco["reset_shoulder"],
        arena["reset_shoulder"],
        "Reset exterior image abs diff",
    )
    _save_absdiff(
        artifacts["wrist_diff"],
        mujoco["reset_wrist"],
        arena["reset_wrist"],
        "Reset wrist image abs diff",
    )
    _save_video_snapshot_strip(
        artifacts["exterior_snapshots"],
        [
            ("MuJoCo", args.mujoco_shoulder_video),
            ("Arena", args.arena_external_video),
        ],
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    _save_video_snapshot_strip(
        artifacts["wrist_snapshots"],
        [
            ("MuJoCo", args.mujoco_wrist_video),
            ("Arena", args.arena_wrist_video),
        ],
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )

    _plot_reset_state(mujoco, arena, artifacts["reset_state_plot"])
    _plot_qpos_traces(mujoco, arena, artifacts["qpos_trace_plot"])
    _plot_first_action(mujoco, arena, artifacts["first_action_plot"])
    _plot_action_traces(mujoco, arena, artifacts["action_trace_plot"])
    _plot_raw_gripper_score(arena, artifacts["arena_gripper_plot"], args.pi_grasping_threshold)
    _plot_mujoco_tcp_context(mujoco, artifacts["tcp_context_plot"])

    _make_video_link(artifacts["mujoco_shoulder_video"], args.mujoco_shoulder_video)
    _make_video_link(artifacts["mujoco_wrist_video"], args.mujoco_wrist_video)

    metrics = _build_metrics(args, mujoco, arena, args.summary_json)
    (args.out_dir / "summary.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "report_html": str(args.out_dir / "report.html"),
                "report_md": str(args.out_dir / "report.md"),
                "summary_json": str(args.out_dir / "summary.json"),
                "artifacts": {key: str(value) for key, value in artifacts.items()},
            },
            indent=2,
        )
    )
    np.savez_compressed(
        args.out_dir / "trace_arrays.npz",
        mujoco_qpos_arm=mujoco["qpos_arm"],
        mujoco_gripper_scalar=mujoco["qpos_gripper_scalar"],
        mujoco_cmd_arm=mujoco["cmd_arm"],
        mujoco_cmd_gripper=mujoco["cmd_gripper"],
        arena_chunk_x=arena["chunk_x"],
        arena_chunk_qpos_arm=np.stack([c["qpos_arm"] for c in arena["chunks"]]),
        arena_chunk_gripper_scalar=np.asarray([c["gripper_scalar"] for c in arena["chunks"]]),
        arena_call_x=np.asarray([c["call_index"] for c in arena["calls"]]),
        arena_call_arm=np.stack([c["arm"] for c in arena["calls"]]),
        arena_call_gripper=np.asarray([c["gripper"] for c in arena["calls"]]),
        arena_call_model_output=np.stack([c["model_output"] for c in arena["calls"]]),
    )
    _write_report(args, mujoco, arena, metrics, artifacts)
    print(f"[idx14_report] wrote {args.out_dir / 'report.html'}")
    print(f"[idx14_report] wrote {args.out_dir / 'report.md'}")
    print(f"[idx14_report] wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
