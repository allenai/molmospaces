#!/usr/bin/env python3
"""Replay a MuJoCo HDF5 trajectory in Arena and build paired camera videos.

This is a thin orchestration layer around run_arena_benchmark_episode.py's
``--policy_type h5_replay`` mode. It chooses a successful MuJoCo trajectory,
launches Arena replay for the matching benchmark episode, then creates a
2-by-2 comparison video:

    MuJoCo external | Arena external
    MuJoCo wrist    | Arena wrist
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MOLMOSPACES_ROOT = Path(__file__).resolve().parents[2]
ISAAC_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("MOLMO_PROJ_ROOT", MOLMOSPACES_ROOT.parent)).expanduser().resolve()


@dataclass
class TrajectoryInfo:
    traj: str
    length: int
    success_indices: list[int]
    first_valid_action_index: int


def _default_arena_root() -> Path:
    lab23 = WORKSPACE_ROOT / "IsaacLab-Arena-working-lab2.3"
    if lab23.is_dir():
        return lab23
    return WORKSPACE_ROOT / "IsaacLab-Arena"


def _default_ffmpeg() -> Path | None:
    try:
        import imageio_ffmpeg

        path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if path.is_file():
            return path
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    return Path(ffmpeg) if ffmpeg else None


def _decode_h5_json_row(row: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(row)
    data = bytes(arr[arr != 0]).decode("utf-8")
    return json.loads(data) if data else {}


def _traj_index(traj: str) -> int:
    match = re.search(r"(\d+)$", str(traj))
    if not match:
        raise ValueError(f"Could not infer numeric trajectory index from {traj!r}")
    return int(match.group(1))


def _trajectory_info(h5_path: Path, traj: str) -> TrajectoryInfo:
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as h5:
        if traj not in h5:
            raise KeyError(f"{traj!r} not found in {h5_path}; available: {sorted(h5.keys())}")
        root = h5[traj]
        success = np.asarray(root["success"][:], dtype=bool)
        success_indices = np.where(success)[0].astype(int).tolist()
        actions = root["actions/commanded_action"]
        first_valid = 0
        for idx, row in enumerate(actions):
            decoded = _decode_h5_json_row(row)
            if "arm" in decoded:
                first_valid = idx
                break
    return TrajectoryInfo(
        traj=traj,
        length=int(len(success)),
        success_indices=success_indices,
        first_valid_action_index=int(first_valid),
    )


def _choose_success_traj(h5_path: Path) -> TrajectoryInfo:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        trajs = sorted((k for k in h5.keys() if k.startswith("traj_")), key=_traj_index)
    if not trajs:
        raise ValueError(f"No traj_* groups found in {h5_path}")

    infos = [_trajectory_info(h5_path, traj) for traj in trajs]
    for info in infos:
        if info.success_indices:
            return info
    return infos[0]


def _infer_h5_from_summary(episode_idx: int, summary_json: Path) -> Path | None:
    if not summary_json.is_file():
        return None
    with summary_json.open() as f:
        data = json.load(f)
    episodes = data.get("episodes") or {}
    row = episodes.get(str(episode_idx))
    if not row:
        return None
    run_dir = Path(data["mujoco_run_dir"])
    house_index = row.get("house_index")
    if house_index is None:
        return None
    h5_path = run_dir / f"house_{int(house_index)}" / "trajectories_batch_1_of_1.h5"
    return h5_path if h5_path.is_file() else None


def _glob_one(patterns: list[str], base: Path) -> Path | None:
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def _infer_mujoco_video(h5_path: Path, traj: str, camera: str) -> Path | None:
    idx = _traj_index(traj)
    prefix = f"episode_{idx:08d}_"
    if camera == "external":
        patterns = [
            f"{prefix}droid_shoulder*_batch_*.mp4",
            f"{prefix}*shoulder*_batch_*.mp4",
            f"{prefix}*exo*_batch_*.mp4",
            f"{prefix}*zed2_analogue_1*_batch_*.mp4",
        ]
    elif camera == "wrist":
        patterns = [
            f"{prefix}wrist_camera*_batch_*.mp4",
            f"{prefix}*wrist*_batch_*.mp4",
        ]
    else:
        raise ValueError(camera)
    return _glob_one(patterns, h5_path.parent)


def _arena_env(arena_root: Path, imageio_ffmpeg_exe: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(WORKSPACE_ROOT / ".venv")
    env["ISAACLAB_ARENA_PATH"] = str(arena_root)
    env["ACCEPT_EULA"] = "Y"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTHONUNBUFFERED"] = "1"
    if imageio_ffmpeg_exe is not None:
        env["IMAGEIO_FFMPEG_EXE"] = str(imageio_ffmpeg_exe)

    source = arena_root / "submodules/IsaacLab/source"
    source_parts = [
        arena_root,
        source / "isaaclab",
        source / "isaaclab_assets",
        source / "isaaclab_contrib",
        source / "isaaclab_mimic",
        source / "isaaclab_rl",
        source / "isaaclab_tasks",
    ]
    existing = [str(p) for p in source_parts if p.exists()]
    env["PYTHONPATH"] = os.pathsep.join(existing + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _build_arena_command(args: argparse.Namespace, info: TrajectoryInfo, out_dir: Path) -> list[str]:
    arena_root = args.arena_root.resolve()
    steps = int(args.steps)
    if steps <= 0:
        steps = max(1, (info.length - info.first_valid_action_index) * int(args.replay_action_repeat))

    cmd = [
        str(arena_root / "submodules/IsaacLab/isaaclab.sh"),
        "-p",
        str(ISAAC_PACKAGE_ROOT / "scripts/run_arena_benchmark_episode.py"),
        "--arena_spec_manifest",
        str(args.arena_spec_manifest.resolve()),
        "--episode_idx",
        str(args.episode_idx),
        "--results_json",
        str(out_dir / "arena_replay_result.json"),
        "--assets_root",
        str(args.assets_root.resolve()),
        "--scenes_root",
        str(args.scenes_root.resolve()),
        "--policy_type",
        "h5_replay",
        "--replay_h5",
        str(args.mujoco_h5.resolve()),
        "--replay_traj",
        info.traj,
        "--replay_start_index",
        str(info.first_valid_action_index if args.replay_start_index < 0 else args.replay_start_index),
        "--replay_action_repeat",
        str(args.replay_action_repeat),
        "--joint_pos_policy",
        "--steps",
        str(steps),
        "--headless",
        "--experience",
        str(arena_root / "submodules/IsaacLab/apps/isaaclab.python.headless.rendering.kit"),
        "--with_cameras",
        "--record_video_dir",
        str(out_dir / "arena_videos"),
        "--record_video_stride",
        str(args.record_video_stride),
        "--record_video_fps",
        str(args.record_video_fps),
        "--record_video_camera_keys",
        "external_camera_rgb,wrist_camera_rgb",
        "--progress_steps",
        str(args.progress_steps),
    ]
    if args.task_description_map is not None:
        cmd.extend(["--task_description_map", str(args.task_description_map.resolve())])
    if args.replay_init_from_h5_qpos:
        cmd.append("--replay_init_from_h5_qpos")
    return cmd


def _read_frame(cap, frame_index: int, size: tuple[int, int]) -> np.ndarray:
    import cv2
    import numpy as np

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    if not ok or frame is None:
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def _open_video(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return cap, max(1, count), fps


def _label(frame: np.ndarray, text: str, height: int = 30) -> np.ndarray:
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    canvas = Image.new("RGB", (img.width, img.height + height), (245, 245, 245))
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), text[:80], fill=(0, 0, 0))
    return np.asarray(canvas, dtype=np.uint8)


def compose_side_by_side_video(
    *,
    mujoco_external: Path,
    mujoco_wrist: Path,
    arena_external: Path,
    arena_wrist: Path,
    output_path: Path,
    fps: int,
    tile_size: tuple[int, int],
) -> dict[str, Any]:
    import cv2
    import numpy as np

    videos = {
        "mujoco_external": mujoco_external,
        "arena_external": arena_external,
        "mujoco_wrist": mujoco_wrist,
        "arena_wrist": arena_wrist,
    }
    caps: dict[str, Any] = {}
    counts: dict[str, int] = {}
    fps_in: dict[str, float] = {}
    for key, path in videos.items():
        cap, count, in_fps = _open_video(path)
        caps[key] = cap
        counts[key] = count
        fps_in[key] = in_fps

    n_out = max(counts.values())
    tile_w, tile_h = tile_size
    labeled_h = tile_h + 30
    out_w = tile_w * 2
    out_h = labeled_h * 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (out_w, out_h),
    )
    try:
        for out_idx in range(n_out):
            frac = 0.0 if n_out <= 1 else out_idx / float(n_out - 1)

            def get(key: str) -> np.ndarray:
                idx = int(round(frac * float(counts[key] - 1)))
                return _read_frame(caps[key], idx, tile_size)

            row0 = np.concatenate(
                [
                    _label(get("mujoco_external"), "MuJoCo external"),
                    _label(get("arena_external"), "Arena external replay"),
                ],
                axis=1,
            )
            row1 = np.concatenate(
                [
                    _label(get("mujoco_wrist"), "MuJoCo wrist"),
                    _label(get("arena_wrist"), "Arena wrist replay"),
                ],
                axis=1,
            )
            frame = np.concatenate([row0, row1], axis=0)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        for cap in caps.values():
            cap.release()

    return {
        "output_path": str(output_path),
        "output_frames": int(n_out),
        "output_fps": int(fps),
        "input_frames": counts,
        "input_fps": fps_in,
    }


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MuJoCo-Arena Trajectory Replay Parity",
        "",
        f"- Episode index: `{summary['episode_idx']}`",
        f"- MuJoCo H5: `{summary['mujoco_h5']}`",
        f"- MuJoCo trajectory: `{summary['mujoco_traj']}`",
        f"- Arena result: `{summary.get('arena_result_json')}`",
        f"- Side-by-side video: `{summary.get('side_by_side_video')}`",
        "",
        "## Command",
        "",
        "```bash",
        summary["arena_command"],
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--arena_root", type=Path, default=_default_arena_root())
    parser.add_argument(
        "--arena_spec_manifest",
        type=Path,
        required=True,
        help="Arena episode spec manifest produced by export_arena_episode_specs.py.",
    )
    parser.add_argument(
        "--task_description_map",
        type=Path,
        default=None,
        help="Optional JSON task-description map to pass through to the Arena episode runner.",
    )
    parser.add_argument("--assets_root", type=Path, default=Path.home() / ".molmospaces/usd")
    parser.add_argument("--scenes_root", type=Path, default=Path.home() / ".molmospaces/usd/scenes")
    parser.add_argument("--mujoco_h5", type=Path, default=None)
    parser.add_argument(
        "--mujoco_summary_json",
        type=Path,
        default=None,
        help="Optional summary JSON used to infer --mujoco_h5 when --mujoco_h5 is not provided.",
    )
    parser.add_argument("--mujoco_traj", default="auto_success")
    parser.add_argument("--mujoco_external_video", type=Path, default=None)
    parser.add_argument("--mujoco_wrist_video", type=Path, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=WORKSPACE_ROOT / "diagnostics/trajectory_replay_parity",
    )
    parser.add_argument("--replay_action_repeat", type=int, default=3)
    parser.add_argument(
        "--replay_start_index",
        type=int,
        default=-1,
        help="Start action row. -1 means first row with an arm command, usually 1.",
    )
    parser.add_argument("--replay_init_from_h5_qpos", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Arena env steps. 0 means infer from H5 length and replay_action_repeat.",
    )
    parser.add_argument("--record_video_stride", type=int, default=3)
    parser.add_argument("--record_video_fps", type=int, default=15)
    parser.add_argument("--progress_steps", type=int, default=250)
    parser.add_argument("--compose_fps", type=int, default=15)
    parser.add_argument("--tile_width", type=int, default=384)
    parser.add_argument("--tile_height", type=int, default=216)
    parser.add_argument("--imageio_ffmpeg_exe", type=Path, default=_default_ffmpeg())
    parser.add_argument("--dry_run", action="store_true", help="Write manifest/command but do not run Arena.")
    parser.add_argument(
        "--compose_only",
        action="store_true",
        help="Skip Arena run and compose from existing videos in --out_dir.",
    )
    args = parser.parse_args()

    if args.mujoco_h5 is None:
        if args.mujoco_summary_json is None:
            raise SystemExit("Pass --mujoco_h5, or pass --mujoco_summary_json so the HDF5 path can be inferred.")
        inferred = _infer_h5_from_summary(args.episode_idx, args.mujoco_summary_json)
        if inferred is None:
            raise SystemExit("--mujoco_h5 was not provided and could not be inferred from summary.")
        args.mujoco_h5 = inferred
    args.mujoco_h5 = args.mujoco_h5.expanduser().resolve()
    if not args.mujoco_h5.is_file():
        raise SystemExit(f"MuJoCo H5 not found: {args.mujoco_h5}")

    if args.mujoco_traj == "auto_success":
        info = _choose_success_traj(args.mujoco_h5)
    else:
        info = _trajectory_info(args.mujoco_h5, args.mujoco_traj)

    mujoco_external = (
        args.mujoco_external_video.expanduser().resolve()
        if args.mujoco_external_video
        else _infer_mujoco_video(args.mujoco_h5, info.traj, "external")
    )
    mujoco_wrist = (
        args.mujoco_wrist_video.expanduser().resolve()
        if args.mujoco_wrist_video
        else _infer_mujoco_video(args.mujoco_h5, info.traj, "wrist")
    )
    if mujoco_external is None or mujoco_wrist is None:
        raise SystemExit(
            "Could not infer MuJoCo videos. Pass --mujoco_external_video and --mujoco_wrist_video."
        )

    out_dir = args.out_dir.expanduser().resolve() / f"ep{args.episode_idx:04d}_{info.traj}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arena_videos").mkdir(exist_ok=True)
    cmd = _build_arena_command(args, info, out_dir)

    summary: dict[str, Any] = {
        "episode_idx": int(args.episode_idx),
        "arena_root": str(args.arena_root.resolve()),
        "arena_spec_manifest": str(args.arena_spec_manifest.resolve()),
        "mujoco_h5": str(args.mujoco_h5),
        "mujoco_traj": info.traj,
        "trajectory_length": info.length,
        "success_indices": info.success_indices,
        "first_valid_action_index": info.first_valid_action_index,
        "replay_action_repeat": int(args.replay_action_repeat),
        "record_video_stride": int(args.record_video_stride),
        "mujoco_external_video": str(mujoco_external),
        "mujoco_wrist_video": str(mujoco_wrist),
        "arena_command": " ".join(cmd),
        "arena_log": str(out_dir / "arena_replay.log"),
        "arena_result_json": str(out_dir / "arena_replay_result.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(out_dir, summary)

    return_code = None
    if not args.dry_run and not args.compose_only:
        env = _arena_env(args.arena_root.resolve(), args.imageio_ffmpeg_exe)
        with (out_dir / "arena_replay.log").open("w") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(MOLMOSPACES_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = int(proc.returncode)
        summary["arena_return_code"] = return_code

    arena_external = out_dir / "arena_videos" / f"arena_ep{args.episode_idx:04d}_external_camera_rgb.mp4"
    arena_wrist = out_dir / "arena_videos" / f"arena_ep{args.episode_idx:04d}_wrist_camera_rgb.mp4"
    if arena_external.is_file() and arena_wrist.is_file():
        side_by_side = out_dir / "mujoco_arena_replay_side_by_side.mp4"
        compose_info = compose_side_by_side_video(
            mujoco_external=mujoco_external,
            mujoco_wrist=mujoco_wrist,
            arena_external=arena_external,
            arena_wrist=arena_wrist,
            output_path=side_by_side,
            fps=args.compose_fps,
            tile_size=(args.tile_width, args.tile_height),
        )
        summary["arena_external_video"] = str(arena_external)
        summary["arena_wrist_video"] = str(arena_wrist)
        summary["side_by_side_video"] = str(side_by_side)
        summary["compose"] = compose_info
    else:
        summary["compose_skipped_reason"] = (
            f"Missing Arena videos: external={arena_external.is_file()} wrist={arena_wrist.is_file()}"
        )

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(out_dir, summary)
    print(f"Wrote replay parity artifacts to {out_dir}")
    if return_code not in (None, 0):
        print(
            f"Arena replay exited with code {return_code}; videos may still be useful for inspection.",
            file=sys.stderr,
        )
    return 0 if return_code in (None, 0) else return_code


if __name__ == "__main__":
    raise SystemExit(main())
