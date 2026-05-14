#!/usr/bin/env python3
"""Compare MolmoSpaces/MuJoCo and Arena OpenPI inputs for one policy step.

This diagnostic lives outside the Arena runtime on purpose. It compares the
exact OpenPI boundary: prompt, resized camera images, qpos, gripper scalar, and
optionally the first action chunk returned by a running OpenPI server.
"""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw


def _decode_h5_json_row(row) -> dict[str, Any]:
    arr = np.asarray(row)
    data = bytes(arr[arr != 0]).decode("utf-8")
    return json.loads(data) if data else {}


def _normalize_prompt(prompt: str | None) -> str:
    out = (prompt or "pick up the object.").strip().lower()
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _resize_with_pad(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height and w == width:
        return np.asarray(img[:, :, :3], dtype=np.uint8)
    ratio = max(w / width, h / height)
    new_w, new_h = int(w / ratio), int(h / ratio)
    pil = Image.fromarray(np.asarray(img[:, :, :3], dtype=np.uint8))
    resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    out = Image.new("RGB", (width, height), 0)
    out.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(out, dtype=np.uint8)


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _image_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean((a - b) ** 2))),
    }


def _save_montage(
    out_path: Path,
    mujoco_exterior: np.ndarray,
    mujoco_wrist: np.ndarray,
    arena_exterior: np.ndarray,
    arena_wrist: np.ndarray,
) -> None:
    labels = [
        ("MuJoCo exterior", mujoco_exterior),
        ("Arena exterior", arena_exterior),
        ("MuJoCo wrist", mujoco_wrist),
        ("Arena wrist", arena_wrist),
    ]
    tile_w, tile_h = 224, 248
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(labels):
        x = (idx % 2) * tile_w
        y = (idx // 2) * tile_h
        canvas.paste(Image.fromarray(arr[:, :, :3].astype(np.uint8)), (x, y + 24))
        draw.text((x + 8, y + 6), label, fill=(0, 0, 0))
    canvas.save(out_path)


def _load_mujoco_input(
    h5_path: Path,
    traj: str,
    h5_index: int,
    shoulder_video: Path,
    wrist_video: Path,
    video_frame_index: int,
    prompt: str,
) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        root = f[traj]
        qpos = _decode_h5_json_row(root["obs/agent/qpos"][h5_index])
    arm = np.asarray(qpos.get("arm", np.zeros(7)), dtype=np.float64).reshape(7)
    gripper = qpos.get("gripper", np.zeros(2))
    grip = float(np.clip(np.atleast_1d(gripper)[0] / 0.824033, 0, 1))
    return {
        "observation/exterior_image_1_left": _resize_with_pad(_read_video_frame(shoulder_video, video_frame_index)),
        "observation/wrist_image_left": _resize_with_pad(_read_video_frame(wrist_video, video_frame_index)),
        "observation/joint_position": arm,
        "observation/gripper_position": np.asarray([grip], dtype=np.float64),
        "prompt": _normalize_prompt(prompt),
    }


def _load_arena_input(arena_trace_dir: Path, chunk_index: int, prompt: str) -> tuple[dict[str, Any], np.ndarray | None]:
    chunk_dir = arena_trace_dir / f"chunk_{chunk_index:04d}"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Arena trace chunk does not exist: {chunk_dir}")
    actions_path = chunk_dir / "actions.npy"
    actions = np.load(actions_path) if actions_path.is_file() else None
    model_input = {
        "observation/exterior_image_1_left": _resize_with_pad(_read_image(chunk_dir / "exterior_image_1_left.png")),
        "observation/wrist_image_left": _resize_with_pad(_read_image(chunk_dir / "wrist_image_left.png")),
        "observation/joint_position": np.asarray(np.load(chunk_dir / "joint_position.npy"), dtype=np.float64).reshape(7),
        "observation/gripper_position": np.asarray(np.load(chunk_dir / "gripper_position.npy"), dtype=np.float64).reshape(1),
        "prompt": _normalize_prompt(prompt),
    }
    return model_input, actions


def _infer_openpi(model_input: dict[str, Any], host: str, port: int) -> np.ndarray:
    openpi_site = (os.environ.get("MOLMO_OPENPI_VENV_SITE") or "").strip()
    if openpi_site and Path(openpi_site).is_dir():
        site.addsitedir(openpi_site)
        if openpi_site in sys.path:
            sys.path.remove(openpi_site)
        sys.path.insert(0, openpi_site)
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    return np.asarray(client.infer(model_input)["actions"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco_h5", type=Path, required=True)
    parser.add_argument("--mujoco_shoulder_video", type=Path, required=True)
    parser.add_argument("--mujoco_wrist_video", type=Path, required=True)
    parser.add_argument("--arena_trace_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--traj", type=str, default="traj_0")
    parser.add_argument("--h5_index", type=int, default=0)
    parser.add_argument("--video_frame_index", type=int, default=0)
    parser.add_argument("--arena_chunk_index", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="pick up the bowl.")
    parser.add_argument("--run_openpi", action="store_true")
    parser.add_argument("--pi_server_host", type=str, default="localhost")
    parser.add_argument("--pi_server_port", type=int, default=8080)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mujoco_input = _load_mujoco_input(
        args.mujoco_h5.expanduser().resolve(),
        args.traj,
        args.h5_index,
        args.mujoco_shoulder_video.expanduser().resolve(),
        args.mujoco_wrist_video.expanduser().resolve(),
        args.video_frame_index,
        args.prompt,
    )
    arena_input, arena_trace_actions = _load_arena_input(
        args.arena_trace_dir.expanduser().resolve(),
        args.arena_chunk_index,
        args.prompt,
    )

    Image.fromarray(mujoco_input["observation/exterior_image_1_left"]).save(out_dir / "mujoco_exterior_policy_input.png")
    Image.fromarray(mujoco_input["observation/wrist_image_left"]).save(out_dir / "mujoco_wrist_policy_input.png")
    Image.fromarray(arena_input["observation/exterior_image_1_left"]).save(out_dir / "arena_exterior_policy_input.png")
    Image.fromarray(arena_input["observation/wrist_image_left"]).save(out_dir / "arena_wrist_policy_input.png")
    _save_montage(
        out_dir / "policy_input_montage.png",
        mujoco_input["observation/exterior_image_1_left"],
        mujoco_input["observation/wrist_image_left"],
        arena_input["observation/exterior_image_1_left"],
        arena_input["observation/wrist_image_left"],
    )

    summary: dict[str, Any] = {
        "prompt": _normalize_prompt(args.prompt),
        "mujoco_h5_index": int(args.h5_index),
        "mujoco_video_frame_index": int(args.video_frame_index),
        "arena_chunk_index": int(args.arena_chunk_index),
        "joint_position_mujoco": mujoco_input["observation/joint_position"].tolist(),
        "joint_position_arena": arena_input["observation/joint_position"].tolist(),
        "joint_position_abs_diff": np.abs(
            mujoco_input["observation/joint_position"] - arena_input["observation/joint_position"]
        ).tolist(),
        "gripper_position_mujoco": mujoco_input["observation/gripper_position"].tolist(),
        "gripper_position_arena": arena_input["observation/gripper_position"].tolist(),
        "exterior_image_metrics": _image_metrics(
            mujoco_input["observation/exterior_image_1_left"],
            arena_input["observation/exterior_image_1_left"],
        ),
        "wrist_image_metrics": _image_metrics(
            mujoco_input["observation/wrist_image_left"],
            arena_input["observation/wrist_image_left"],
        ),
        "artifacts": {
            "montage": str(out_dir / "policy_input_montage.png"),
            "mujoco_exterior": str(out_dir / "mujoco_exterior_policy_input.png"),
            "mujoco_wrist": str(out_dir / "mujoco_wrist_policy_input.png"),
            "arena_exterior": str(out_dir / "arena_exterior_policy_input.png"),
            "arena_wrist": str(out_dir / "arena_wrist_policy_input.png"),
        },
    }
    if arena_trace_actions is not None:
        summary["arena_trace_first_action"] = np.asarray(arena_trace_actions[0], dtype=float).tolist()

    if args.run_openpi:
        mujoco_actions = _infer_openpi(mujoco_input, args.pi_server_host, args.pi_server_port)
        arena_actions = _infer_openpi(arena_input, args.pi_server_host, args.pi_server_port)
        np.save(out_dir / "mujoco_openpi_actions.npy", mujoco_actions)
        np.save(out_dir / "arena_openpi_actions.npy", arena_actions)
        n = min(len(mujoco_actions), len(arena_actions))
        summary["openpi"] = {
            "mujoco_actions_shape": list(mujoco_actions.shape),
            "arena_actions_shape": list(arena_actions.shape),
            "mujoco_first_action": np.asarray(mujoco_actions[0], dtype=float).tolist(),
            "arena_first_action": np.asarray(arena_actions[0], dtype=float).tolist(),
            "first_action_abs_diff": np.abs(mujoco_actions[0] - arena_actions[0]).tolist(),
            "chunk_mean_abs_diff": float(np.abs(mujoco_actions[:n] - arena_actions[:n]).mean()),
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[compare_pi_policy_inputs] wrote {out_dir / 'summary.json'}", flush=True)
    print(f"[compare_pi_policy_inputs] wrote {out_dir / 'policy_input_montage.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
