from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets import video_utils as _lr_video_utils

# Lerobot hardcodes crf=30 when constructing the StreamingVideoEncoder
# (lerobot_dataset.py:1711). That CRF was tuned for libsvtav1; for h264 CRF=30
# looks visibly worse. Bump h264 to crf=23 to match libsvtav1@30 perceptual
# quality. h264 is still 10× faster than av1, so encode speed is unaffected.
_H264_CRF = 23
_orig_sve_init = _lr_video_utils.StreamingVideoEncoder.__init__

def _patched_sve_init(self, *args, **kwargs):
    vcodec = kwargs.get("vcodec", args[1] if len(args) > 1 else "libsvtav1")
    if "h264" in vcodec and kwargs.get("crf", 30) == 30:
        kwargs["crf"] = _H264_CRF
    return _orig_sve_init(self, *args, **kwargs)

_lr_video_utils.StreamingVideoEncoder.__init__ = _patched_sve_init


# Lerobot's concatenate_video_files uses ffmpeg's concat demuxer via PyAV but
# fails to propagate DTS offsets to remuxed packets. With h264 (denser
# time_base than av1), the new episode's small DTS values land before the
# shard's last DTS, and the mp4 muxer rejects with
# `[Errno 22] Invalid argument: '/tmp/tmpXXX.mp4'`. Replace the helper with a
# direct stream-copy that rewrites DTS/PTS to be monotonically increasing.
def _concat_video_files_fixed(input_video_paths, output_video_path, overwrite=True):
    import shutil
    import tempfile as _tf
    import av as _av
    from pathlib import Path as _Path

    output_video_path = _Path(output_video_path)
    if output_video_path.exists() and not overwrite:
        return
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    with _tf.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_out = f.name

    out = _av.open(tmp_out, mode="w", options={"movflags": "faststart"})
    smap = None  # input_stream_index -> output_stream
    last_dts: dict[int, int] = {}  # by output stream index

    try:
        for path in input_video_paths:
            inp = _av.open(str(path), "r")
            if smap is None:
                smap = {}
                for s_in in inp.streams:
                    if s_in.type in ("video", "audio", "subtitle"):
                        s_out = out.add_stream_from_template(template=s_in, opaque=True)
                        s_out.time_base = s_in.time_base
                        smap[s_in.index] = s_out
                        last_dts[s_out.index] = -1
            dts_offsets: dict[int, int] = {}
            seen_first: dict[int, bool] = {}
            for pkt in inp.demux():
                if pkt.stream.index not in smap or pkt.dts is None:
                    continue
                out_stream = smap[pkt.stream.index]
                if not seen_first.get(out_stream.index, False):
                    dts_offsets[out_stream.index] = (last_dts[out_stream.index] + 1) - pkt.dts
                    seen_first[out_stream.index] = True
                off = dts_offsets[out_stream.index]
                pkt.dts = pkt.dts + off
                if pkt.pts is not None:
                    pkt.pts = pkt.pts + off
                pkt.stream = out_stream
                out.mux(pkt)
                last_dts[out_stream.index] = pkt.dts
            inp.close()
        out.close()
        shutil.move(tmp_out, str(output_video_path))
    except Exception:
        try:
            out.close()
        except Exception:
            pass
        if _Path(tmp_out).exists():
            _Path(tmp_out).unlink()
        raise

# Patch both module-level and the lerobot_dataset.py import.
_lr_video_utils.concatenate_video_files = _concat_video_files_fixed
import lerobot.datasets.lerobot_dataset as _lr_dataset_mod
_lr_dataset_mod.concatenate_video_files = _concat_video_files_fixed


class LeRobotRecorder:
    """Records episodes as a LeRobot v3 dataset.

    Subsamples physics steps down to ``fps``. Proprioceptive obs + action logged
    per step (no videos). Call ``begin_episode`` on reset, ``add_step`` each
    env step, ``save_episode`` on success, ``discard_episode`` on failure, and
    ``close`` at shutdown.
    """

    def __init__(self, repo_id, env, action_dim, root=None, fps=10, physics_fps=200, robot_type="g1",
                 data_files_size_in_mb=0, video_files_size_in_mb=0):
        self._env = env
        # `physics_fps` here means the rate at which `add_step` is called by the
        # caller (typically the env.step rate, NOT the underlying mj_step rate).
        # Caller should pass `1 / (mj_timestep * n_substeps)`.
        self._stride = max(1, int(round(physics_fps / fps)))
        self._step_in_episode = 0
        self._task = ""
        self._object_name = ""
        self._object_id = ""
        self._skill_profile = "default"
        self._scene = ""
        root = Path(root) if root is not None else None

        features = {
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"a{i}" for i in range(action_dim)],
            },
            "object_name": {"dtype": "string", "shape": (1,)},
            "object_id": {"dtype": "string", "shape": (1,)},
            "skill_profile": {"dtype": "string", "shape": (1,)},
            "scene": {"dtype": "string", "shape": (1,)},
        }
        for key, space in env.observation_space.spaces.items():
            shape = tuple(int(s) for s in space.shape)
            n = int(np.prod(shape))
            features[f"observation.{key}"] = {
                "dtype": "float32",
                "shape": shape,
                "names": [f"{key}_{i}" for i in range(n)],
            }
        self._cameras = list(env.cameras)
        for name in self._cameras:
            features[f"observation.{name}"] = {
                "dtype": "video",
                "shape": env.camera_shape,
                "names": ["height", "width", "channels"],
            }

        if root is not None and root.exists():
            self._dataset = LeRobotDataset(repo_id, root=root)
            self._validate_existing_dataset(features, fps)
        else:
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=root,
                fps=fps,
                features=features,
                robot_type=robot_type,
                use_videos=bool(self._cameras),
                vcodec="h264",
                streaming_encoding=bool(self._cameras),
            )
            # Size caps of 0: every episode is finalized into its OWN data
            # parquet + per-camera mp4 (instead of being remux-concatenated into a
            # growing chunk file). A sudden death (Ctrl+C, kill, power loss) then
            # loses at most the single in-flight episode — all completed episodes
            # are self-contained, intact files. They get re-concatenated into
            # chunks at merge time. chunks_size keeps per-directory file counts
            # bounded (default 1000 files per chunk-XXX dir).
            self._dataset.meta.info["data_files_size_in_mb"] = int(data_files_size_in_mb)
            self._dataset.meta.info["video_files_size_in_mb"] = int(video_files_size_in_mb)
            self._dataset.meta.info["chunks_size"] = int(self._dataset.meta.info.get("chunks_size", 1000) or 1000)
            # Flush episode metadata every episode (default buffers 10). Otherwise
            # a sudden death loses the metadata for up to the last 10 episodes even
            # though their per-episode data/video files are intact on disk.
            self._dataset.meta.metadata_buffer_size = 1
            from lerobot.datasets.utils import write_info
            write_info(self._dataset.meta.info, self._dataset.meta.root)

    def _validate_existing_dataset(self, features, fps):
        if int(self._dataset.fps) != int(fps):
            raise ValueError(
                f"Existing dataset fps={self._dataset.fps} does not match requested fps={fps}."
            )
        existing = self._dataset.meta.features
        missing = sorted(set(features) - set(existing))
        extra = sorted(set(existing) - set(features) - {"index", "episode_index", "task_index", "timestamp", "frame_index"})
        if missing or extra:
            raise ValueError(
                f"Existing dataset schema mismatch. missing={missing}, extra={extra}"
            )
        for key, spec in features.items():
            es = existing[key]
            if es.get("dtype") != spec["dtype"] or tuple(es.get("shape", ())) != tuple(spec["shape"]):
                raise ValueError(
                    f"Existing dataset feature {key!r} mismatch: existing={es}, requested={spec}"
                )

    def begin_episode(self, info):
        self._step_in_episode = 0
        self._task = info.get("prompt") or info.get("task") or ""
        scene = info.get("scene") or str(getattr(self._env, "_current_scene_path", "") or "")
        tgt = getattr(getattr(self._env, "task", None), "target", None)
        obj = info.get("object_name") or getattr(tgt, "category", None) or getattr(tgt, "asset_id", "") or ""
        self._object_name = str(obj)
        # Unique asset identity (objaverse UID / THOR asset id) so two different
        # meshes of the same category are distinguishable; "" if unavailable.
        self._object_id = str(info.get("object_id") or getattr(tgt, "asset_id", "") or "")
        self._skill_profile = str(info.get("skill_profile", "default"))
        self._scene = str(scene)

    def add_step(self, obs, action):
        if self._step_in_episode % self._stride != 0:
            self._step_in_episode += 1
            return
        self._step_in_episode += 1
        frame = {
            "action": np.asarray(action, dtype=np.float32).reshape(-1),
            "task": self._task,
            "object_name": self._object_name,
            "object_id": self._object_id,
            "skill_profile": self._skill_profile,
            "scene": self._scene,
        }
        for key, val in obs.items():
            frame[f"observation.{key}"] = np.asarray(val, dtype=np.float32).reshape(-1)
        if self._cameras:
            images = self._env.render_cameras()
            for name in self._cameras:
                frame[f"observation.{name}"] = np.asarray(images[name], dtype=np.uint8)
        self._dataset.add_frame(frame)

    def save_episode(self):
        self._dataset.save_episode()

    def discard_episode(self):
        self._dataset.clear_episode_buffer()

    def close(self):
        self._dataset.finalize()
