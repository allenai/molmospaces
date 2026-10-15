"""LeRobot-format recorder, the sibling of save_utils.save_trajectories.

Both consume the same prepared episodes -- the list of batched dicts
prepare_episode_for_saving() produces -- so an experiment can emit the h5
("molmobot") layout, the LeRobot layout, or both from one rollout, with no
second simulation pass and no re-encoding.

The on-disk layout is written directly rather than through the `lerobot`
package, which is not a dependency here:

    <save_dir>/lerobot/
      meta/info.json          dataset schema + feature dtypes/shapes
      meta/tasks.jsonl        {"task_index", "task"}
      meta/episodes.jsonl     {"episode_index", "tasks", "length"}
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.<camera>/episode_000000.mp4

Videos are *copied*, not re-encoded: prepare_episode_for_saving() already wrote
one mp4 per camera (and stripped the frames from the episode dict to keep peak
memory down), so re-encoding is both impossible here and unnecessary. Copying
also means the two formats reference bit-identical pixels, which is what lets
the recorder test compare them frame count for frame count.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.utils.save_utils import byte_array_to_string

log = logging.getLogger(__name__)

# LeRobot dataset format this writer targets. v2.1 is what `lerobot==0.3.x`
# reads; the string is what its loader validates against.
LEROBOT_CODEBASE_VERSION = "v2.1"

# Episodes per chunk directory. LeRobot's own default; with one chunk the
# {episode_chunk:03d} in the path templates below is always 000.
DEFAULT_CHUNK_SIZE = 1000

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# Which batched key supplies `action`. Episodes carry several action encodings
# side by side (joint_pos, joint_pos_rel, ee_pose, ee_twist, commanded_action);
# joint position is the one every robot here reports and the one the h5's
# actions/ group is keyed on.
DEFAULT_ACTION_KEY = "actions/joint_pos"

# Which batched key supplies `observation.state`.
DEFAULT_STATE_KEY = "qpos"


def _decode_json_rows(tensor) -> list[dict]:
    """Decode a (T, str_max_len) uint8 JSON tensor back to per-timestep dicts.

    Dict-valued sensors (qpos, the action variants) are JSON-encoded into fixed
    width byte rows by batch_observations; see save_utils.dict_to_byte_array.
    """
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return [json.loads(byte_array_to_string(row)) for row in array]


def _flatten_numeric(entry: dict) -> tuple[np.ndarray, list[str]]:
    """Flatten {"arm": [...], "gripper": [...]} to one vector plus its names.

    Keys are visited in sorted order so the layout is stable across episodes and
    runs; the names go into info.json so the packing stays legible downstream.
    """
    values: list[float] = []
    names: list[str] = []
    for key in sorted(entry):
        part = np.asarray(entry[key], dtype=np.float32).ravel()
        values.extend(part.tolist())
        names.extend(f"{key}_{i}" for i in range(part.size))
    return np.asarray(values, dtype=np.float32), names


def _episode_task(episode_data: dict[str, Any]) -> str:
    """The natural-language task string, read from the episode's obs_scene blob."""
    raw = episode_data.get("obs_scene")
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw).get("task_description", "") or ""
    except (json.JSONDecodeError, AttributeError):
        return ""


def _as_1d(episode_data: dict[str, Any], key: str, length: int, dtype) -> np.ndarray:
    """A per-timestep column, zero-filled when the episode never recorded it."""
    value = episode_data.get(key)
    if value is None:
        return np.zeros(length, dtype=dtype)
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return array.astype(dtype).reshape(-1)[:length]


def _episode_videos(save_dir: Path, episode_idx: int, save_file_suffix: str) -> dict[str, Path]:
    """Map camera name -> mp4 that prepare_episode_for_saving already wrote.

    Mirrors save_videos_from_raw_observations' naming
    (episode_{idx:08d}_{camera}{suffix}.mp4), recovering the camera name by
    stripping the prefix and suffix it composes.
    """
    prefix = f"episode_{episode_idx:08d}_"
    videos: dict[str, Path] = {}
    for path in sorted(save_dir.glob(f"{prefix}*{save_file_suffix}.mp4")):
        camera = path.stem[len(prefix) :]
        if save_file_suffix:
            camera = camera[: -len(save_file_suffix)]
        if camera:
            videos[camera] = path
    return videos


def _video_shape(path: Path) -> list[int]:
    """(height, width, channels) of an mp4, or a zeroed shape if unreadable."""
    try:
        import imageio.v3 as iio

        frame = iio.imread(path, index=0)
        return [int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])]
    except Exception as exc:  # noqa: BLE001 - metadata only, never worth failing a save
        log.warning(f"Could not read frame size from {path.name}: {exc}")
        return [0, 0, 0]


def save_trajectories_lerobot(
    episodes_data: list[dict[str, Any]],
    save_dir: str | Path,
    fps: float,
    save_file_suffix: str = "",
    robot_type: str = "unknown",
    state_key: str = DEFAULT_STATE_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Write `episodes_data` as a LeRobot dataset under
    `save_dir/lerobot{save_file_suffix}`.

    Args:
        episodes_data: Prepared episodes, exactly what save_trajectories takes.
        save_dir: House output directory -- the same one holding the h5 and the
            per-camera mp4s this reads back.
        fps: Frame rate of the episode data.
        save_file_suffix: Batch suffix -- names the dataset root and the mp4s read back.
        robot_type: Recorded in info.json; informational.
        state_key: Batched key packed into `observation.state`.
        action_key: Batched key packed into `action`.

    Returns:
        The dataset root, or None if there was nothing to write.
    """
    logger = logger or log
    save_dir = Path(save_dir)

    if not episodes_data:
        logger.warning(f"No episodes to write in LeRobot format for {save_dir.name}")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Batch-suffixed like the h5 side's trajectories{batch_suffix}.h5: every batch
    # of a house is handed the same house_output_dir, so a fixed "lerobot" root
    # made each batch overwrite the previous one's parquet/meta.
    root = save_dir / f"lerobot{save_file_suffix}"
    (root / "meta").mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episode_rows: list[dict] = []
    video_shapes: dict[str, list[int]] = {}
    state_names: list[str] = []
    action_names: list[str] = []
    global_frame = 0
    total_videos = 0

    for episode_idx, episode_data in enumerate(episodes_data):
        if state_key not in episode_data:
            logger.warning(f"Episode {episode_idx} has no {state_key!r}; skipping")
            continue

        states = [_flatten_numeric(entry) for entry in _decode_json_rows(episode_data[state_key])]
        length = len(states)
        if length == 0:
            logger.warning(f"Episode {episode_idx} has no timesteps; skipping")
            continue
        state_matrix = np.stack([vector for vector, _ in states])
        state_names = states[0][1]

        if action_key in episode_data:
            actions = [_flatten_numeric(e) for e in _decode_json_rows(episode_data[action_key])]
            action_matrix = np.stack([vector for vector, _ in actions])[:length]
            action_names = actions[0][1]
        else:
            logger.warning(f"Episode {episode_idx} has no {action_key!r}; writing zeroed actions")
            action_matrix = np.zeros_like(state_matrix)
            action_names = state_names

        task = _episode_task(episode_data)
        task_index = task_to_index.setdefault(task, len(task_to_index))

        # Output index, distinct from episode_idx: episodes skipped above still
        # advance the enumerate counter, so naming files by it leaves gaps while
        # info.json's total_episodes counts only what was written -- LeRobot's
        # loader then looks for an episode_000000.parquet that does not exist.
        # episode_rows gains exactly one entry per written episode, so its
        # current length is the next contiguous index.
        out_idx = len(episode_rows)

        table = pa.table(
            {
                "observation.state": [row.tolist() for row in state_matrix],
                "action": [row.tolist() for row in action_matrix],
                "timestamp": (np.arange(length, dtype=np.float32) / float(fps)).tolist(),
                "frame_index": np.arange(length, dtype=np.int64).tolist(),
                "episode_index": np.full(length, out_idx, dtype=np.int64).tolist(),
                "index": np.arange(global_frame, global_frame + length, dtype=np.int64).tolist(),
                "task_index": np.full(length, task_index, dtype=np.int64).tolist(),
                "next.reward": _as_1d(episode_data, "rewards", length, np.float32).tolist(),
                "next.done": _as_1d(episode_data, "terminateds", length, bool).tolist(),
                "next.success": _as_1d(episode_data, "successes", length, bool).tolist(),
            }
        )

        chunk = out_idx // DEFAULT_CHUNK_SIZE
        data_path = root / DATA_PATH.format(episode_chunk=chunk, episode_index=out_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)

        for camera, source in _episode_videos(save_dir, episode_idx, save_file_suffix).items():
            video_key = f"observation.images.{camera}"
            destination = root / VIDEO_PATH.format(
                episode_chunk=chunk, video_key=video_key, episode_index=out_idx
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            video_shapes.setdefault(video_key, _video_shape(destination))
            total_videos += 1

        episode_rows.append({"episode_index": out_idx, "tasks": [task], "length": length})
        global_frame += length

    if not episode_rows:
        logger.warning(f"No LeRobot episodes written for {save_dir.name}")
        return None

    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        },
        "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "next.success": {"dtype": "bool", "shape": [1], "names": None},
    }
    for video_key, shape in sorted(video_shapes.items()):
        features[video_key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "info": {"video.fps": float(fps), "video.codec": "h264", "video.is_depth_map": False},
        }

    info = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": len(episode_rows),
        "total_frames": global_frame,
        "total_tasks": len(task_to_index),
        "total_videos": total_videos,
        "total_chunks": (len(episode_rows) - 1) // DEFAULT_CHUNK_SIZE + 1,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": float(fps),
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [
            {"task_index": i, "task": t}
            for t, i in sorted(task_to_index.items(), key=lambda p: p[1])
        ],
    )
    _write_jsonl(root / "meta" / "episodes.jsonl", episode_rows)

    logger.info(
        f"Saved LeRobot dataset to {root} "
        f"({len(episode_rows)} episodes, {global_frame} frames, {total_videos} videos)"
    )
    return root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
