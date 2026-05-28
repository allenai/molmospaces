#!/usr/bin/env python3
"""Run MolmoSpaces pick in Isaac Lab Arena (demo-oriented).

Default: one bundled episode JSON (``examples/pick_episode_ithor_thor_only.json``) when present;
otherwise a pick benchmark directory (see ``MOLMO_PICK_BENCHMARK_DIR`` / ``--benchmark_dir``).
Scene vertical placement uses inverse ``robot_base_pose`` plus ``--scene_extra_xyz`` and optional
``MOLMO_ARENA_SCENE_FINE_Z`` only (no automatic iTHOR Z offsets). Pick success mirrors
MolmoSpaces/MuJoCo: lift above threshold and no non-robot scene support.
``--assets_root`` needs ``objects/thor/``. OpenPI: ``--policy_type pi_remote``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # molmo_spaces_isaac
MOLMOSPACES_ROOT = REPO_ROOT.parent  # workspace containing molmo_spaces + molmo_spaces_isaac
SRC_ROOT = REPO_ROOT / "src"


def _default_pick_benchmark_dir() -> Path:
    """Prefer MOLMO_PICK_BENCHMARK_DIR, else bundled simple THOR pick bench, else legacy 10-ep bench, else FrankaPickHardBench path."""
    env = os.environ.get("MOLMO_PICK_BENCHMARK_DIR")
    if env:
        return Path(env).resolve()
    for name in ("benchmark_ithor_pick_hard_simple", "benchmark_ithor_thor_only_10"):
        bundled = REPO_ROOT / "examples" / name
        if bundled.is_dir() and (bundled / "benchmark.json").is_file():
            return bundled.resolve()
    return Path(
        "/home/zryan/molmospaces_bench/mujoco/benchmarks/molmospaces-bench-v1/20260210/ithor/"
        "FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark"
    ).resolve()


_DEFAULT_PICK_BENCHMARK_DIR = _default_pick_benchmark_dir()

# Single-episode demo when neither --episode_json nor --benchmark_dir is passed.
_DEMO_EPISODE_JSON = REPO_ROOT / "examples" / "pick_episode_ithor_thor_only.json"

# Make Isaac Lab Arena importable (set ISAACLAB_ARENA_PATH to your Arena repo root)
_arena_path = os.environ.get("ISAACLAB_ARENA_PATH")
if _arena_path:
    _arena_path = Path(_arena_path).resolve()
    if _arena_path.is_dir() and str(_arena_path) not in sys.path:
        sys.path.insert(0, str(_arena_path))

for p in (SRC_ROOT, REPO_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isaaclab.app import AppLauncher

# Bash one-liner `... -- \ --flag` can produce a token `" --flag"`; argparse won't match `--flag`.
for _i in range(1, len(sys.argv)):
    sys.argv[_i] = sys.argv[_i].lstrip()
# Drop standalone `--` (e.g. `isaaclab.sh -p script.py -- ...`).
while len(sys.argv) > 1 and sys.argv[1] == "--":
    sys.argv.pop(1)

parser = argparse.ArgumentParser(
    description="Run MolmoSpaces pick in Isaac Lab Arena. "
    f"Default: bundled {_DEMO_EPISODE_JSON.name} when present; else a pick benchmark directory."
)
parser.add_argument(
    "--episode_json",
    type=Path,
    default=None,
    help=(
        "Single episode JSON (pick task). If neither this nor --benchmark_dir is set, uses "
        f"{_DEMO_EPISODE_JSON.name} when that file exists."
    ),
)
parser.add_argument(
    "--benchmark_dir",
    type=Path,
    default=None,
    help=(
        "Pick benchmark directory (benchmark.json). Used when no episode JSON is selected and the demo JSON is missing. "
        f"Fallback dir: {_DEFAULT_PICK_BENCHMARK_DIR}."
    ),
)
parser.add_argument(
    "--arena_spec_manifest",
    type=Path,
    default=None,
    help=(
        "Exported Arena episode spec manifest JSON. Uses --episode_idx to select one "
        "already-converted ArenaEpisodeSpec row."
    ),
)
parser.add_argument(
    "--task_description_map",
    type=Path,
    default=None,
    help=(
        "Optional JSON mapping pickup object names to task descriptions when running "
        "from --arena_spec_manifest."
    ),
)
parser.add_argument(
    "--assets_root",
    type=Path,
    default=None,
    help="Root dir for USD assets (e.g. /path/to/molmospaces_isaac). THOR: .../objects/thor/thor/20260128, Objaverse: .../objects/objaverse. Overrides MOLMO_ISAAC_ASSETS_ROOT for this run.",
)
parser.add_argument(
    "--episode_idx",
    type=int,
    default=0,
    help=(
        "0-based row in benchmark.json when using --benchmark_dir. Ignored for --episode_json. "
        "With only --assets_root, non-zero values auto-use bundled benchmark_ithor_pick_hard_simple "
        "(single-file demo ignores episode_idx when episode_idx is 0)."
    ),
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=None,
    metavar="N",
    help="Run N episodes from the benchmark (0 = all). Like MolmoSpaces; default: run single episode at --episode_idx.",
)
parser.add_argument(
    "--episode_indices",
    type=str,
    default=None,
    help=(
        "Comma-separated benchmark episode indices or ranges to run sequentially, "
        "for representative batches (e.g. '0,8,17-20'). Mutually exclusive with "
        "--max_episodes."
    ),
)
parser.add_argument(
    "--results_json",
    type=Path,
    default=None,
    help="Optional JSON output path for episode results and success-rate summaries.",
)
parser.add_argument("--steps", type=int, default=5000, help="Max simulation steps per episode (benchmark runs until done or this limit).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (Isaac Lab vectorises on GPU). E.g. --num_envs 16 runs 16 copies simultaneously.")
parser.add_argument("--env_spacing", type=float, default=None, help="Grid spacing (m) between parallel environments (default: Arena's 30 m). 10 m is sufficient for iTHOR kitchens (~6 m wide).")
parser.add_argument(
    "--progress_steps",
    type=int,
    default=0,
    help="Print progress every N steps (0=off). E.g. 100 to see step count during long runs.",
)
parser.add_argument(
    "--debug_arm_motion",
    type=int,
    default=0,
    help="Print arm state (joint_pos, eef_pos) and last action every N steps (0=off). E.g. 50 to verify the arm is moving.",
)
parser.add_argument(
    "--debug_pick_z",
    type=int,
    default=0,
    help="Print pick object Z, lift height, and tunneling warning every N steps (0=off). E.g. 50 to detect pick-through-scene.",
)
parser.add_argument(
    "--pause_after_reset",
    type=float,
    default=0.0,
    metavar="SECONDS",
    help="Pause N seconds after env.reset() before the sim loop starts (GUI mode: keeps window open for inspection).",
)
parser.add_argument(
    "--policy_type",
    type=str,
    choices=["zero", "random", "pi_remote", "h5_replay"],
    default="zero",
    help=(
        "Action source: 'zero', 'random' (small Gaussian noise), 'pi_remote' "
        "(OpenPI server; start server first), or 'h5_replay' (MolmoSpaces/MuJoCo "
        "commanded actions from eval HDF5)."
    ),
)
parser.add_argument(
    "--with_cameras",
    action="store_true",
    default=False,
    help="Enable camera sensors in the Arena environment (adds wrist + exo cameras to Franka). "
         "Automatically enabled when --policy_type pi_remote. Pass explicitly to enable for other policy types.",
)
parser.add_argument(
    "--joint_pos_policy",
    action="store_true",
    help=(
        "Use joint position control for pi_remote (8D: 7 arm joint angles + 1 gripper). "
        "Use for explicit DROID joint-position checkpoints. "
        "Switches Arena Franka from IK delta-EEF to JointPositionActionCfg."
    ),
)
parser.add_argument(
    "--joint_velocity_policy",
    action="store_true",
    help=(
        "Use DROID joint velocity control for pi_remote (8D: 7 joint velocities + 1 gripper). "
        "This is the default for OpenPI's pi05_droid / pi0_droid checkpoints."
    ),
)
parser.add_argument(
    "--pi_action_repeat",
    type=int,
    default=0,
    help=(
        "Repeat each pi_remote action for N Arena steps before consuming the next chunk action. "
        "0 = auto (25 for MolmoSpaces joint-position PiPolicyEval timing, "
        "3 for stock DROID joint-velocity checkpoints, 1 otherwise)."
    ),
)
parser.add_argument(
    "--pi_grasping_threshold",
    type=float,
    default=float(os.environ.get("MOLMO_PI_GRASPING_THRESHOLD", "0.01")),
    help=(
        "Binary gripper close threshold for pi_remote. The documented "
        "pi05_droid_jointpos checkpoint needs a low threshold (default: 0.01)."
    ),
)
parser.add_argument(
    "--pi_chunk_size",
    type=int,
    default=int(os.environ.get("MOLMO_PI_CHUNK_SIZE", "15")),
    help=(
        "Number of OpenPI chunk actions to consume before requesting a new chunk "
        "(default: 15, matching pi05_droid_jointpos action_horizon)."
    ),
)
parser.add_argument(
    "--pi_server_host",
    type=str,
    default=os.environ.get("MOLMO_PI_SERVER_HOST", "localhost"),
    help="OpenPI server host for --policy_type pi_remote (default: localhost or MOLMO_PI_SERVER_HOST).",
)
parser.add_argument(
    "--pi_server_port",
    type=int,
    default=int(os.environ.get("MOLMO_PI_SERVER_PORT", "8000")),
    help="OpenPI server port for --policy_type pi_remote (default: 8000, or MOLMO_PI_SERVER_PORT).",
)
parser.add_argument(
    "--pi_trace_dir",
    type=Path,
    default=None,
    help=(
        "Optional directory to save OpenPI request traces: resized camera images, qpos, "
        "raw action chunks, and selected actions. Useful for MuJoCo/Arena policy parity debugging."
    ),
)
parser.add_argument(
    "--record_video_dir",
    type=Path,
    default=None,
    help=(
        "Optional directory to save Arena camera rollout MP4s. Records camera_obs frames "
        "from the actual policy rollout."
    ),
)
parser.add_argument(
    "--record_video_stride",
    type=int,
    default=3,
    help="Save one video frame every N Arena env steps when --record_video_dir is set (default: 3).",
)
parser.add_argument(
    "--record_video_fps",
    type=int,
    default=15,
    help="Output FPS for --record_video_dir MP4s (default: 15).",
)
parser.add_argument(
    "--record_video_camera_keys",
    type=str,
    default="external_camera_rgb,wrist_camera_rgb",
    help=(
        "Comma-separated camera_obs keys to record when --record_video_dir is set. "
        "Default: external_camera_rgb,wrist_camera_rgb."
    ),
)
parser.add_argument(
    "--replay_h5",
    type=Path,
    default=None,
    help="MolmoSpaces eval HDF5 to replay when --policy_type h5_replay.",
)
parser.add_argument(
    "--replay_traj",
    type=str,
    default=os.environ.get("MOLMO_ARENA_REPLAY_TRAJ", "traj_0"),
    help="Trajectory group inside --replay_h5 for --policy_type h5_replay (default: traj_0).",
)
parser.add_argument(
    "--replay_action_repeat",
    type=int,
    default=int(os.environ.get("MOLMO_ARENA_REPLAY_ACTION_REPEAT", "25")),
    help="Arena env steps per MuJoCo policy action for --policy_type h5_replay (default: 25).",
)
parser.add_argument(
    "--replay_start_index",
    type=int,
    default=int(os.environ.get("MOLMO_ARENA_REPLAY_START_INDEX", "0")),
    help="Start HDF5 replay from this commanded-action row (default: 0).",
)
parser.add_argument(
    "--replay_init_from_h5_qpos",
    action="store_true",
    help="Before HDF5 replay, write obs/agent/qpos at --replay_start_index directly into the Arena robot.",
)
parser.add_argument(
    "--scenes_root",
    type=Path,
    default=None,
    help=(
        "Root for MolmoSpaces scene USDs (ms-download install dir or MOLMO_SCENES_ROOT). "
        "If omitted, defaults to the same path as --assets_root (or MOLMO_ISAAC_ASSETS_ROOT) so iTHOR "
        "FloorPlans resolve; otherwise the episode loads the Arena kitchen background only."
    ),
)
parser.add_argument(
    "--background",
    type=str,
    default="kitchen",
    help="Arena background when episode scene USD is not found (default: kitchen).",
)
parser.add_argument(
    "--embodiment",
    type=str,
    default=None,
    help=(
        "Arena embodiment key. Default: franka for zero/random runs; droid_abs_joint_pos for "
        "--policy_type pi_remote --joint_pos_policy. Set MOLMO_ARENA_EMBODIMENT to override."
    ),
)
parser.add_argument(
    "--scene_extra_xyz",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=[0.0, 0.0, 0.0],
    help="Extra translation (m) added to the MolmoSpaces scene root when a scene USD is loaded (tune vertical fit with Z).",
)
parser.add_argument(
    "--align_scene_floor_z_zero",
    action="store_true",
    help=(
        "Scene root Z from --scene_extra_xyz Z + MOLMO_ARENA_SCENE_FINE_Z only (skip inverse robot_base_pose Z). "
        "XY and yaw still follow robot_base_pose. Sets MOLMO_ARENA_SCENE_Z_ALIGN_WORLD_ZERO=1."
    ),
)
parser.add_argument(
    "--allow-objaverse",
    action="store_true",
    dest="allow_objaverse",
    help="Allow Objaverse pickup objects (needs objects/objaverse/ USDs). Default is THOR-only.",
)
parser.add_argument(
    "--require_thor_only",
    action="store_true",
    help=argparse.SUPPRESS,
)
# AppLauncher must see --enable_cameras before parse_args to initialize the RTX sensor pipeline.
_argv_needs_cameras = (
    ("--policy_type" in sys.argv and "pi_remote" in sys.argv[sys.argv.index("--policy_type") + 1 :])
    or "--with_cameras" in sys.argv
)
if _argv_needs_cameras and "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if getattr(args, "align_scene_floor_z_zero", False):
    os.environ["MOLMO_ARENA_SCENE_Z_ALIGN_WORLD_ZERO"] = "1"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def load_episode_dict_from_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_input_file(path: Path, *, label: str) -> Path:
    p = Path(path).expanduser()
    candidates = [p]
    if not p.is_absolute():
        candidates.extend([REPO_ROOT / p, MOLMOSPACES_ROOT / p])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"{label} not found: {path}")


def _load_task_description_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    p = _resolve_input_file(path, label="Task description map")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("by_pickup_obj_name", data) if isinstance(data, dict) else {}
    if not isinstance(entries, dict):
        return {}
    out: dict[str, str] = {}
    for name, value in entries.items():
        if isinstance(value, dict):
            text = value.get("task_description")
        else:
            text = value
        if text:
            out[str(name)] = str(text)
    return out


def _arena_manifest_objects_to_tuples(raw_objects) -> list[tuple[str, str, list[float], str]]:
    out: list[tuple[str, str, list[float], str]] = []
    if not isinstance(raw_objects, list):
        raise SystemExit("Arena spec manifest row has no object list")
    for item in raw_objects:
        if isinstance(item, dict):
            name = str(item.get("name") or "")
            asset_id = str(item.get("asset_id") or item.get("scene_prim") or name)
            pose = item.get("pose_7_world") or item.get("pose_7") or item.get("pose")
            source = str(item.get("source") or "thor")
        else:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                raise SystemExit(f"Invalid Arena manifest object entry: {item!r}")
            name = str(item[0])
            asset_id = str(item[1])
            pose = item[2]
            source = str(item[3])
        if not name or not pose or len(pose) < 7:
            raise SystemExit(f"Invalid Arena manifest object entry: {item!r}")
        out.append((name, asset_id, list(pose[:7]), source))
    return out


def _task_description_for_manifest_row(row: dict, pickup_name: str, task_descriptions: dict[str, str]) -> str:
    language = row.get("language") if isinstance(row, dict) else None
    if isinstance(language, dict) and language.get("task_description"):
        return str(language["task_description"])
    if pickup_name in task_descriptions:
        return task_descriptions[pickup_name]
    short_name = (pickup_name or "object").split("_", 1)[0]
    short_name = short_name.replace("Irishpotato", "potato")
    return f"Pick up the {short_name}"


def load_arena_manifest_rows(manifest_path: Path) -> tuple[Path, list[dict]]:
    path = _resolve_input_file(manifest_path, label="Arena spec manifest")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise SystemExit(f"No episodes found in Arena spec manifest: {path}")
    for idx, row in enumerate(episodes):
        if not isinstance(row, dict) or not isinstance(row.get("arena_spec"), dict):
            raise SystemExit(f"Arena spec manifest row {idx} has no arena_spec")
    return path, episodes


def load_arena_spec_from_manifest(
    manifest_path: Path,
    episode_idx: int,
    *,
    scenes_root: Path | None,
    task_descriptions: dict[str, str],
):
    from molmo_spaces_isaac.arena.episode_to_arena import ArenaEpisodeSpec, resolve_episode_scene_usd_path

    path, episodes = load_arena_manifest_rows(manifest_path)
    if episode_idx < 0 or episode_idx >= len(episodes):
        raise SystemExit(
            f"episode_idx {episode_idx} out of range [0, {len(episodes) - 1}] "
            f"(manifest has {len(episodes)} episodes)"
        )

    row = episodes[episode_idx]
    spec_payload = row.get("arena_spec") if isinstance(row, dict) else None
    if not isinstance(spec_payload, dict):
        raise SystemExit(f"Arena spec manifest row {episode_idx} has no arena_spec")
    spec_data = dict(spec_payload)
    spec_data["objects"] = _arena_manifest_objects_to_tuples(spec_data.get("objects"))

    scene_usd_path = spec_data.get("scene_usd_path")
    if scene_usd_path:
        spec_data["scene_usd_path"] = Path(scene_usd_path).expanduser()
    if not spec_data.get("scene_usd_path") or not Path(spec_data["scene_usd_path"]).is_file():
        resolved = resolve_episode_scene_usd_path(row, scenes_root)
        if resolved is not None:
            spec_data["scene_usd_path"] = resolved
        elif spec_data.get("scene_usd_path") and not Path(spec_data["scene_usd_path"]).is_file():
            spec_data["scene_usd_path"] = None

    spec = ArenaEpisodeSpec(**spec_data)
    pickup_name = str(row.get("pickup_obj_name") or spec.pickup_name)
    task_description = _task_description_for_manifest_row(row, pickup_name, task_descriptions)
    episode_dict = {
        "house_index": row.get("house_index"),
        "scene_dataset": row.get("scene_dataset"),
        "data_split": row.get("data_split", "val"),
        "task": {
            "task_type": getattr(spec, "task_type", "pick"),
            "pickup_obj_name": pickup_name,
            "succ_pos_threshold": getattr(spec, "succ_pos_threshold", 0.01),
            "robot_base_pose": list(getattr(spec, "robot_base_pose", []) or []),
        },
        "language": {"task_description": task_description},
        "cameras": list(getattr(spec, "camera_specs", None) or []),
        "img_resolution": list(getattr(spec, "img_resolution", None) or []),
    }
    return spec, episode_dict, path


def _resolve_asset_dirs(assets_root: Path | None):
    """Return (thor_assets_dir, thor_metadata_path, objaverse_assets_dir) from assets_root or (None, None, None)."""
    if assets_root is None:
        return None, None, None
    root = Path(assets_root).resolve()
    if not root.is_dir():
        return None, None, None
    thor_assets_dir = None
    thor_metadata_path = None
    objaverse_assets_dir = None
    from molmo_spaces_isaac.arena.thor_asset import THOR_DEFAULT_VERSION
    versioned = root / "objects" / "thor" / "thor" / THOR_DEFAULT_VERSION
    if versioned.is_dir():
        thor_assets_dir = versioned
        thor_metadata_path = versioned / "usd_assets_metadata.json"
    else:
        flat = root / "objects" / "thor"
        if flat.is_dir():
            thor_assets_dir = flat
            thor_metadata_path = (flat / "usd_assets_metadata.json") if (flat / "usd_assets_metadata.json").is_file() else None
    obja = root / "objects" / "objaverse"
    if not obja.is_dir():
        obja = root / "objaverse"
    if obja.is_dir():
        from molmo_spaces_isaac.arena.objaverse_asset import OBJAVERSE_DEFAULT_VERSION
        obja_versioned = obja / OBJAVERSE_DEFAULT_VERSION
        objaverse_assets_dir = obja_versioned if obja_versioned.is_dir() else obja
    return thor_assets_dir, thor_metadata_path, objaverse_assets_dir


def load_episode_dict_from_benchmark(benchmark_dir: Path, episode_idx: int) -> dict:
    """Load a single episode dict from a benchmark dir (molmo_spaces or benchmark.json)."""
    try:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
        episodes = load_all_episodes(Path(benchmark_dir))
        if not episodes:
            raise SystemExit(f"No episodes found in {benchmark_dir}")
        if episode_idx < 0 or episode_idx >= len(episodes):
            raise SystemExit(
                f"episode_idx {episode_idx} out of range [0, {len(episodes) - 1}] "
                f"(benchmark has {len(episodes)} episodes)"
            )
        return episodes[episode_idx].model_dump()
    except ImportError:
        pass
    # No molmo_spaces (e.g. Isaac Sim env): load from benchmark.json
    bench_file = Path(benchmark_dir) / "benchmark.json"
    if not bench_file.is_file():
        raise SystemExit(
            "Using --benchmark_dir requires molmo_spaces, or a benchmark directory "
            "containing benchmark.json (array of episodes). "
            "Install molmo_spaces from repo root: pip install -e ./molmo_spaces"
        )
    with open(bench_file) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"No episodes in {bench_file}")
    if episode_idx < 0 or episode_idx >= len(data):
        raise SystemExit(
            f"episode_idx {episode_idx} out of range [0, {len(data) - 1}] "
            f"(benchmark has {len(data)} episodes)"
        )
    return data[episode_idx]


def _load_all_episode_dicts(bench_dir: Path) -> list[dict]:
    """All episodes as dicts (molmo_spaces load_all_episodes or benchmark.json array)."""
    try:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
        episodes = load_all_episodes(Path(bench_dir))
        if not episodes:
            return []
        return [ep.model_dump() if hasattr(ep, "model_dump") else ep for ep in episodes]
    except ImportError:
        bench_file = Path(bench_dir) / "benchmark.json"
        if not bench_file.is_file():
            return []
        with open(bench_file) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []


def _parse_episode_indices(raw: str | None, n_total: int) -> list[int] | None:
    """Parse '0,8,17-20' into sorted unique valid episode indices."""
    if raw is None:
        return None
    indices: list[int] = []
    for piece in str(raw).replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            if not start_s or not end_s:
                raise SystemExit(f"Invalid --episode_indices range: {piece!r}")
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(piece))
    if not indices:
        raise SystemExit("--episode_indices did not contain any indices.")
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= n_total:
            raise SystemExit(
                f"--episode_indices contains {idx}, out of range [0, {n_total - 1}] "
                f"(benchmark has {n_total} episodes)"
            )
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def _episode_result_metadata(idx: int, episode_dict: dict, spec=None) -> dict:
    """Small JSON-friendly metadata payload for result tracking."""
    task = episode_dict.get("task") or {}
    scene_usd_path = getattr(spec, "scene_usd_path", None) if spec is not None else None
    return {
        "idx": int(idx),
        "house_index": episode_dict.get("house_index"),
        "scene_dataset": episode_dict.get("scene_dataset"),
        "pickup_obj_name": task.get("pickup_obj_name"),
        "task_description": ((episode_dict.get("language") or {}).get("task_description")),
        "scene_usd_path": str(scene_usd_path) if scene_usd_path else None,
    }


def _write_results_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[molmospaces_arena] wrote results JSON: {out}", flush=True)


def _record_camera_keys(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "record_video_camera_keys", "") or ""
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _obs_camera_frame_to_uint8(value):
    """Return first-env HWC uint8 frame from an Isaac Lab camera observation."""
    import numpy as np

    arr = value
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


class ArenaVideoRecorder:
    """Stream camera observations to MP4 files during an Arena rollout."""

    def __init__(self, output_dir: Path, episode_idx: int, camera_keys: list[str], fps: int) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = int(episode_idx)
        self.camera_keys = camera_keys
        self.fps = max(1, int(fps))
        self._writers = {}
        self._paths: dict[str, str] = {}

    def capture(self, obs: dict, step_count: int) -> None:
        camera_obs = (obs or {}).get("camera_obs") or {}
        for key in self.camera_keys:
            if key not in camera_obs:
                continue
            frame = _obs_camera_frame_to_uint8(camera_obs[key])
            if frame is None:
                continue
            writer = self._writers.get(key)
            if writer is None:
                import imageio.v2 as imageio

                safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)
                path = self.output_dir / f"arena_ep{self.episode_idx:04d}_{safe_key}.mp4"
                writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8, macro_block_size=1)
                self._writers[key] = writer
                self._paths[key] = str(path)
                print(f"[molmospaces_arena] recording {key} video to {path}", flush=True)
            writer.append_data(frame)

    def close(self) -> dict[str, str]:
        for writer in list(self._writers.values()):
            try:
                writer.close()
            except Exception:
                pass
        self._writers.clear()
        return dict(self._paths)


def _maybe_default_scenes_root_from_assets(args: argparse.Namespace, _resolved_bench_dir: Path | None) -> None:
    """If scenes_root is unset, use the same root as assets (typical ms-download: objects/ + scenes/ under one dir).

    Without this, single-episode / demo JSON runs leave scenes_root=None, no FloorPlan USD is resolved, and Arena
    falls back to the default kitchen background.
    """
    if getattr(args, "scenes_root", None) is not None:
        return
    if os.environ.get("MOLMO_SCENES_ROOT"):
        return
    root = args.assets_root or os.environ.get("MOLMO_ISAAC_ASSETS_ROOT")
    if root:
        args.scenes_root = Path(root).resolve()


def _to_list(x):
    """Convert tensor or array to a short list of floats for debug print."""
    if x is None:
        return []
    if hasattr(x, "cpu"):
        x = x.cpu().tolist()
    elif hasattr(x, "tolist"):
        x = x.tolist()
    else:
        x = list(x)
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(v)
        else:
            out.append(v)
    return [round(float(v), 4) for v in out[:12]]


def _decode_h5_json_row(row) -> dict:
    """Decode MolmoSpaces HDF5 fixed-width uint8 JSON rows."""
    import numpy as np

    arr = np.asarray(row)
    data = bytes(arr[arr != 0]).decode("utf-8")
    return json.loads(data) if data else {}


class H5ReplayPolicy:
    """Replay MolmoSpaces/MuJoCo commanded arm/gripper actions into Arena."""

    def __init__(self, h5_path: Path, traj: str = "traj_0", start_index: int = 0) -> None:
        import h5py
        import numpy as np

        self.h5_path = Path(h5_path).expanduser().resolve()
        self.traj = traj
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"--replay_h5 does not exist: {self.h5_path}")
        self.actions: list[np.ndarray] = []
        with h5py.File(self.h5_path, "r") as f:
            dataset = f[f"{traj}/actions/commanded_action"]
            for row in dataset:
                action = _decode_h5_json_row(row)
                arm = np.asarray(action.get("arm", np.zeros(7)), dtype=np.float32)[:7]
                if arm.size < 7:
                    arm = np.pad(arm, (0, 7 - arm.size))
                gripper = float(np.mean(np.atleast_1d(action.get("gripper", [0.0]))))
                gripper_cmd = 1.0 if gripper > 0.5 else 0.0
                self.actions.append(np.concatenate([arm, [gripper_cmd]]).astype(np.float32))
        self.index = max(0, min(int(start_index), max(0, len(self.actions) - 1)))
        print(
            f"[molmospaces_arena] Loaded {len(self.actions)} replay actions from {self.h5_path}; "
            f"start_index={self.index}",
            flush=True,
        )

    def qpos_at(self, index: int) -> dict:
        import h5py

        with h5py.File(self.h5_path, "r") as f:
            dataset = f[f"{self.traj}/obs/agent/qpos"]
            idx = max(0, min(int(index), dataset.shape[0] - 1))
            return _decode_h5_json_row(dataset[idx])

    def next_action(self, device):
        import torch

        if not self.actions:
            return torch.zeros((1, 8), dtype=torch.float32, device=device)
        idx = min(self.index, len(self.actions) - 1)
        self.index += 1
        return torch.from_numpy(self.actions[idx]).unsqueeze(0).to(device)


def _write_arena_robot_qpos_from_mujoco(env, qpos: dict) -> None:
    """Write a MolmoSpaces/MuJoCo qpos dict into the Arena DROID articulation."""
    import math
    import torch

    scene = env.unwrapped.scene
    robot = scene["robot"]
    joint_names = list(getattr(robot.data, "joint_names", []) or [])
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    for i, value in enumerate(list(qpos.get("arm") or [])[:7]):
        name = f"panda_joint{i + 1}"
        if name in name_to_idx:
            joint_pos[0, name_to_idx[name]] = float(value)

    gripper = qpos.get("gripper") or []
    g_raw = float(sum(float(x) for x in gripper) / len(gripper)) if gripper else 0.0
    g = min(max(g_raw, 0.0), math.pi / 4)
    if g_raw > 0.1:
        g = math.pi / 4
    gripper_targets = {
        "finger_joint": g,
        "right_outer_knuckle_joint": g,
        "left_inner_finger_joint": -g,
        "right_inner_finger_joint": g,
        "left_inner_finger_knuckle_joint": -g,
        "right_inner_finger_knuckle_joint": -g,
    }
    for name, value in gripper_targets.items():
        if name in name_to_idx:
            joint_pos[0, name_to_idx[name]] = float(value)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    scene.update(dt=env.unwrapped.physics_dt)
    env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=True)
    print(
        f"[molmospaces_arena] Initialized Arena robot from HDF5 qpos "
        f"(arm={_to_list(joint_pos[0, :7])}, gripper_raw={g_raw:.4f}).",
        flush=True,
    )


def _selected_embodiment_key(args: argparse.Namespace) -> str:
    """Choose the Arena embodiment. OpenPI DROID checkpoints expect DROID observations/actions."""
    explicit = getattr(args, "embodiment", None) or os.environ.get("MOLMO_ARENA_EMBODIMENT")
    if explicit:
        return str(explicit)
    if getattr(args, "joint_pos_policy", False) or (
        getattr(args, "policy_type", "zero") == "pi_remote" and _use_pi_joint_velocity_control(args)
    ):
        return "droid_abs_joint_pos"
    return "franka"


def _use_pi_joint_velocity_control(args: argparse.Namespace) -> bool:
    """OpenPI DROID checkpoints use joint velocity actions unless an explicit joint-pos path is requested."""
    if getattr(args, "policy_type", "zero") != "pi_remote":
        return False
    if getattr(args, "joint_pos_policy", False):
        return False
    explicit = getattr(args, "embodiment", None) or os.environ.get("MOLMO_ARENA_EMBODIMENT")
    if explicit and not str(explicit).startswith("droid") and not getattr(args, "joint_velocity_policy", False):
        return False
    return True


def _pi_action_repeat(args: argparse.Namespace) -> int:
    repeat = int(getattr(args, "pi_action_repeat", 0) or 0)
    if repeat > 0:
        return max(1, repeat)
    if getattr(args, "policy_type", "zero") == "pi_remote" and getattr(args, "joint_pos_policy", False):
        # MolmoSpaces PiPolicyEvalConfig uses policy_dt_ms=500. Arena DROID env step is 0.02 s.
        return 25
    if _use_pi_joint_velocity_control(args):
        return 3
    return 1


def _default_pi_camera_key_map(embodiment_key: str) -> dict[str, str]:
    """Map OpenPI's DROID camera names to Arena observation terms for the active embodiment."""
    if str(embodiment_key).startswith("droid"):
        default_wrist = "camera_obs.wrist_camera_rgb"
        default_exo = "camera_obs.external_camera_rgb"
    else:
        default_wrist = "camera_obs.wrist_cam_rgb"
        default_exo = "camera_obs.exo_cam_rgb"
    return {
        "wrist_camera": os.environ.get("MOLMO_PI_WRIST_CAMERA", default_wrist),
        "exo_camera_1": os.environ.get("MOLMO_PI_EXO_CAMERA", default_exo),
    }


def _print_arm_motion_debug(step_count: int, obs: dict, actions, device) -> None:
    """Print current arm state and last action so user can verify the arm is moving."""
    policy = obs.get("policy") or {}
    joint_pos = policy.get("joint_pos")
    eef_pos = policy.get("eef_pos")
    if joint_pos is not None or eef_pos is not None:
        jp = _to_list(joint_pos) if joint_pos is not None else []
        ep = _to_list(eef_pos) if eef_pos is not None else []
        act = _to_list(actions) if actions is not None else []
        print(
            f"  [arm_motion] step {step_count}: joint_pos={jp} eef_pos={ep} action={act}",
            flush=True,
        )
    else:
        print(f"  [arm_motion] step {step_count}: no policy/joint_pos/eef_pos in obs", flush=True)


def _print_pick_z_debug(step_count: int, env, spec, spawn_z: float, obs: dict | None = None, actions=None) -> None:
    """Print pick object Z/lift plus policy-space distance to the end effector."""
    try:
        base_env = getattr(env, "unwrapped", env)
        scene = getattr(base_env, "scene", None)
        if scene is None:
            print(f"  [pick_z] step {step_count}: no scene", flush=True)
            return
        pick_key = getattr(spec, "pickup_name", None) or ""
        rp = scene[pick_key].data.root_pos_w
        obj_pos = rp[0] if rp.dim() > 1 else rp
        z = float(rp[0, 2].item()) if rp.dim() > 1 else float(rp[2].item())
        lift = z - spawn_z
        dist_text = ""
        grip_text = ""
        if obs:
            policy = obs.get("policy") or {}
            eef_pos = policy.get("eef_pos")
            joint_pos = policy.get("joint_pos")
            gripper_pos = policy.get("gripper_pos")
            if eef_pos is not None:
                eef = eef_pos[0] if getattr(eef_pos, "dim", lambda: 0)() > 1 else eef_pos
                dist = (eef - obj_pos).norm().item()
                dist_text = f"  eef_obj_dist={dist:.4f}"
            if gripper_pos is not None:
                gp = gripper_pos[0] if getattr(gripper_pos, "dim", lambda: 0)() > 1 else gripper_pos
                grip_text = f"  gripper_pos={_to_list(gp)}"
        gripper_geom_text = ""
        try:
            robot = scene["robot"]
            body_names = list(getattr(robot.data, "body_names", []) or [])
            body_pos = getattr(robot.data, "body_pos_w", None)
            finger_names = [
                "left_outer_finger",
                "right_outer_finger",
                "left_inner_finger",
                "right_inner_finger",
            ]
            finger_pos = []
            if body_pos is not None:
                for name in finger_names:
                    if name in body_names:
                        p = body_pos[0, body_names.index(name), :]
                        finger_pos.append(p)
                if finger_pos:
                    import torch

                    fp = torch.stack(finger_pos)
                    finger_center = fp.mean(dim=0)
                    finger_center_dist = (finger_center - obj_pos).norm().item()
                    finger_min_dist = (fp - obj_pos).norm(dim=1).min().item()
                    gripper_geom_text = (
                        f"  finger_center_obj_dist={finger_center_dist:.4f}"
                        f"  finger_min_obj_dist={finger_min_dist:.4f}"
                    )
            joint_names = list(getattr(robot.data, "joint_names", []) or [])
            joint_pos_all = getattr(robot.data, "joint_pos", None)
            if joint_pos_all is not None:
                keep = [
                    (name, joint_pos_all[0, idx].item())
                    for idx, name in enumerate(joint_names)
                    if "finger" in name or "knuckle" in name
                ]
                if keep:
                    short = ",".join(f"{name}={float(val):.3f}" for name, val in keep)
                    gripper_geom_text += f"  gripper_joints={short}"
        except Exception:
            pass
        action_text = ""
        if actions is not None:
            act = actions[0] if getattr(actions, "dim", lambda: 0)() > 1 else actions
            action_text = f"  action={_to_list(act)}"
            if obs:
                jp = (obs.get("policy") or {}).get("joint_pos")
                if jp is not None:
                    jp0 = jp[0] if getattr(jp, "dim", lambda: 0)() > 1 else jp
                    arm_err = (jp0[:7] - act[:7]).norm().item()
                    action_text += f"  arm_cmd_err={arm_err:.4f}"
        warn = "  *** BELOW SPAWN ***" if lift < -0.05 else ""
        print(
            f"  [pick_z] step {step_count}: z={z:.4f}  spawn_z={spawn_z:.4f}  "
            f"lift={lift:+.4f}{dist_text}{grip_text}{gripper_geom_text}{action_text}{warn}",
            flush=True,
        )
    except Exception as e:
        print(f"  [pick_z] step {step_count}: read failed ({e})", flush=True)


def _run_one_episode(
    spec,
    episode_dict: dict,
    args,
    thor_assets_dir,
    thor_metadata_path,
    objaverse_assets_dir,
    cli_args_list: list[str],
    pi_remote_policy=None,
    pi_remote_camera_key_map=None,
    env_name: str = "molospaces_arena_benchmark",
    episode_idx_for_artifacts: int | None = None,
):
    """Build env from spec, run episode, close env. Use unique env_name per episode to avoid gym duplicate registration."""
    import torch
    from molmo_spaces_isaac.arena.build_from_spec import build_arena_env_from_episode_spec

    episode_length_s = args.steps * 0.02
    extra = tuple(getattr(args, "scene_extra_xyz", [0.0, 0.0, 0.0])[:3])
    _enable_cameras = getattr(args, "with_cameras", False) or (getattr(args, "policy_type", "zero") == "pi_remote")
    embodiment_key = _selected_embodiment_key(args)
    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name=env_name,
        embodiment_key=embodiment_key,
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=episode_length_s,
        cli_args_list=cli_args_list,
        scene_extra_translation_xyz=extra,
        use_joint_pos_control=getattr(args, "joint_pos_policy", False),
        use_joint_velocity_control=_use_pi_joint_velocity_control(args),
        num_envs=getattr(args, "num_envs", 1),
        env_spacing=getattr(args, "env_spacing", None),
        enable_cameras=_enable_cameras,
    )
    try:
        obs, _ = env.reset()
    except Exception as e:
        print(f"env.reset() failed: {e}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return False, 0

    # Capture pick spawn Z for per-step tunneling debug (--debug_pick_z).
    spawn_z: float | None = None
    try:
        _scene = getattr(env.unwrapped, "scene", None)
        if _scene is not None:
            _pick_key = getattr(spec, "pickup_name", None) or ""
            _rp = _scene[_pick_key].data.root_pos_w
            spawn_z = float(_rp[0, 2].item()) if _rp.dim() > 1 else float(_rp[2].item())
    except Exception:
        pass

    video_recorder = None
    video_paths: dict[str, str] = {}
    if getattr(args, "record_video_dir", None) is not None:
        artifact_idx = (
            int(episode_idx_for_artifacts)
            if episode_idx_for_artifacts is not None
            else int(getattr(args, "episode_idx", 0))
        )
        video_recorder = ArenaVideoRecorder(
            args.record_video_dir,
            episode_idx=artifact_idx,
            camera_keys=_record_camera_keys(args),
            fps=getattr(args, "record_video_fps", 15),
        )
        video_recorder.capture(obs, step_count=0)

    step_count = 0
    success = False
    device = env.unwrapped.device
    pi_repeat = _pi_action_repeat(args)
    pi_repeat_left = 0
    pi_last_actions = None
    if args.policy_type == "pi_remote":
        print(
            f"[molmospaces_arena] pi_remote action mode: "
            f"{'joint_velocity' if _use_pi_joint_velocity_control(args) else 'joint_position' if getattr(args, 'joint_pos_policy', False) else 'delta_eef'}; "
            f"action_repeat={pi_repeat}.",
            flush=True,
        )
    try:
        for _ in range(args.steps):
            if simulation_app is not None and not simulation_app.is_running():
                break
            with torch.inference_mode():
                if args.policy_type == "pi_remote" and pi_remote_policy is not None:
                    if pi_last_actions is None or pi_repeat_left <= 0:
                        from molmo_spaces_isaac.arena.pi_remote_client import get_pi_remote_action
                        actions = get_pi_remote_action(
                            pi_remote_policy,
                            obs,
                            camera_key_map=pi_remote_camera_key_map or {},
                            default_image_shape=(224, 224, 3),
                            device=device,
                            use_joint_pos_control=getattr(args, "joint_pos_policy", False),
                            use_joint_velocity_control=_use_pi_joint_velocity_control(args),
                        )
                        pi_last_actions = actions
                        pi_repeat_left = pi_repeat
                    else:
                        actions = pi_last_actions
                    pi_repeat_left -= 1
                elif args.policy_type == "random":
                    actions = torch.randn(env.action_space.shape, device=device) * 0.05
                else:
                    actions = torch.zeros(env.action_space.shape, device=device)
                obs, _reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if args.progress_steps > 0 and step_count % args.progress_steps == 0:
                print(f"  step {step_count}/{args.steps}...", flush=True)
            if (
                video_recorder is not None
                and step_count % max(1, int(getattr(args, "record_video_stride", 1))) == 0
            ):
                video_recorder.capture(obs, step_count=step_count)
            if args.debug_arm_motion > 0 and step_count % args.debug_arm_motion == 0:
                _print_arm_motion_debug(step_count, obs, actions, device)
            if args.debug_pick_z > 0 and step_count % args.debug_pick_z == 0 and spawn_z is not None:
                _print_pick_z_debug(step_count, env, spec, spawn_z, obs=obs, actions=actions)
            done = terminated | truncated
            if done.any():
                success = bool(terminated.any().item())
                break
    except Exception as e:
        import traceback
        print(f"Episode loop error after {step_count} steps: {e}", flush=True)
        traceback.print_exc()
        success = False
    finally:
        if video_recorder is not None:
            try:
                if step_count > 0 and step_count % max(1, int(getattr(args, "record_video_stride", 1))) != 0:
                    video_recorder.capture(obs, step_count=step_count)
            except Exception:
                pass
            video_paths = video_recorder.close()

    try:
        env.close()
    except Exception:
        pass
    return success, step_count, video_paths


def main() -> int:
    if getattr(args, "scenes_root", None) is None and os.environ.get("MOLMO_SCENES_ROOT"):
        args.scenes_root = Path(os.environ["MOLMO_SCENES_ROOT"]).resolve()
    source_count = sum(
        1
        for source in (args.episode_json, args.benchmark_dir, args.arena_spec_manifest)
        if source is not None
    )
    if source_count > 1:
        raise SystemExit("Use only one of --episode_json, --benchmark_dir, or --arena_spec_manifest.")
    if getattr(args, "pi_trace_dir", None) is not None:
        os.environ["MOLMO_PI_TRACE_DIR"] = str(Path(args.pi_trace_dir).expanduser().resolve())

    _bundled_simple_bench = REPO_ROOT / "examples" / "benchmark_ithor_pick_hard_simple"

    if args.episode_json is None and args.benchmark_dir is None and args.arena_spec_manifest is None:
        # --episode_idx only selects a row from benchmark.json; it is ignored for a single JSON file.
        if args.episode_idx != 0:
            if (_bundled_simple_bench / "benchmark.json").is_file():
                args.benchmark_dir = _bundled_simple_bench
                print(
                    f"[molmospaces_arena] --episode_idx {args.episode_idx} → loading "
                    f"{args.benchmark_dir}/benchmark.json (not the single-file demo).",
                    flush=True,
                )
            else:
                args.benchmark_dir = _DEFAULT_PICK_BENCHMARK_DIR
                if not args.benchmark_dir.is_dir() or not (Path(args.benchmark_dir) / "benchmark.json").is_file():
                    raise SystemExit(
                        f"--episode_idx {args.episode_idx} needs a benchmark directory with benchmark.json. "
                        f"Bundled bench not found at {_bundled_simple_bench}. Pass --benchmark_dir explicitly."
                    )
                print(
                    f"[molmospaces_arena] --episode_idx {args.episode_idx} → using {args.benchmark_dir}",
                    flush=True,
                )
        elif _DEMO_EPISODE_JSON.is_file():
            args.episode_json = _DEMO_EPISODE_JSON
            print(f"[molmospaces_arena] Using demo episode: {args.episode_json}", flush=True)
        else:
            args.benchmark_dir = _DEFAULT_PICK_BENCHMARK_DIR
            if not args.benchmark_dir.is_dir():
                raise SystemExit(
                    f"No default episode: {_DEMO_EPISODE_JSON} missing and benchmark dir not found: {args.benchmark_dir}. "
                    "Pass --episode_json, --benchmark_dir, or install the bundled examples."
                )

    if args.episode_json is not None and args.episode_idx != 0:
        print(
            f"[molmospaces_arena] NOTE: --episode_idx {args.episode_idx} is ignored when --episode_json is set.",
            flush=True,
        )

    from molmo_spaces_isaac.arena.episode_to_arena import episode_dict_to_arena_spec

    if args.allow_objaverse and args.require_thor_only:
        raise SystemExit("Cannot use both --allow-objaverse and --require_thor_only.")
    require_thor_only_mode = (not args.allow_objaverse) or args.require_thor_only

    resolved_bench_dir: Path | None = None
    resolved_arena_spec_manifest: Path | None = None
    manifest_task_descriptions: dict[str, str] | None = None
    spec = None
    _maybe_default_scenes_root_from_assets(args, resolved_bench_dir)
    if args.arena_spec_manifest is not None:
        task_descriptions = _load_task_description_map(args.task_description_map)
        if args.max_episodes is not None or args.episode_indices is not None:
            resolved_arena_spec_manifest, manifest_rows = load_arena_manifest_rows(args.arena_spec_manifest)
            n_total = len(manifest_rows)
            parsed_indices = _parse_episode_indices(args.episode_indices, n_total)
            if parsed_indices is not None:
                indices_to_run = parsed_indices
            else:
                n_run = n_total if args.max_episodes == 0 else min(args.max_episodes, n_total)
                indices_to_run = list(range(n_run))
            manifest_task_descriptions = task_descriptions
            episode_dict = None
            episode_dicts = None
        else:
            spec, episode_dict, resolved_arena_spec_manifest = load_arena_spec_from_manifest(
                args.arena_spec_manifest,
                args.episode_idx,
                scenes_root=args.scenes_root,
                task_descriptions=task_descriptions,
            )
            episode_dicts = None
            indices_to_run = None
    elif args.benchmark_dir is not None:
        bench_dir = Path(args.benchmark_dir)
        if not bench_dir.is_dir() and not bench_dir.is_absolute():
            bench_dir = REPO_ROOT / bench_dir
        if not bench_dir.is_dir():
            raise SystemExit(f"Benchmark directory not found: {args.benchmark_dir}")
        resolved_bench_dir = bench_dir

        max_ep = getattr(args, "max_episodes", None)
        if max_ep is not None and args.episode_indices is not None:
            raise SystemExit("Use either --max_episodes or --episode_indices, not both.")
        if max_ep is not None or args.episode_indices is not None:
            try:
                from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
                episodes = load_all_episodes(bench_dir)
                episode_dicts = [ep.model_dump() if hasattr(ep, "model_dump") else ep for ep in episodes]
            except ImportError:
                bench_file = bench_dir / "benchmark.json"
                if not bench_file.is_file():
                    raise SystemExit(f"No benchmark.json in {bench_dir}")
                with open(bench_file) as f:
                    episode_dicts = json.load(f)
            if not episode_dicts:
                raise SystemExit("No episodes in benchmark.")
            n_total = len(episode_dicts)
            parsed_indices = _parse_episode_indices(args.episode_indices, n_total)
            if parsed_indices is not None:
                indices_to_run = parsed_indices
            else:
                n_run = n_total if max_ep == 0 else min(max_ep, n_total)
                indices_to_run = list(range(n_run))
            episode_dict = None
        else:
            episode_dict = load_episode_dict_from_benchmark(bench_dir, args.episode_idx)
            episode_dicts = None
            indices_to_run = None
    else:
        path = Path(args.episode_json)
        if not path.is_file() and not path.is_absolute():
            # Try under molmo_spaces_isaac (REPO_ROOT) first, then workspace (MOLMOSPACES_ROOT)
            path = REPO_ROOT / path
            if not path.is_file():
                path = MOLMOSPACES_ROOT / args.episode_json
        if not path.is_file():
            raise SystemExit(f"Episode file not found: {path}")
        episode_dict = load_episode_dict_from_json(path)
        episode_dicts = None
        indices_to_run = None

    from molmo_spaces_isaac.arena.build_from_spec import build_arena_env_from_episode_spec

    _maybe_default_scenes_root_from_assets(args, resolved_bench_dir)

    # Multi-episode run (like MolmoSpaces): run each episode in sequence and print summary.
    if indices_to_run is not None:
        if args.policy_type == "h5_replay":
            raise SystemExit(
                "--policy_type h5_replay is single-episode only for now because each episode "
                "needs its matching MuJoCo HDF5 trajectory. Use --episode_idx instead."
            )
        if args.assets_root is not None and not Path(args.assets_root).resolve().is_dir():
            raise SystemExit(f"Assets root is not a directory: {args.assets_root}")
        if args.assets_root is not None:
            os.environ.setdefault("MOLMO_ISAAC_ASSETS_ROOT", str(Path(args.assets_root).resolve()))
        thor_assets_dir, thor_metadata_path, objaverse_assets_dir = _resolve_asset_dirs(args.assets_root)
        import torch
        sim_device = getattr(args, "device", "cuda:0") or "cuda:0"
        if sim_device == "cuda":
            sim_device = "cuda:0"
        if sim_device.startswith("cuda"):
            try:
                cuda_ok = torch.cuda.is_available()
            except Exception:
                cuda_ok = False
            if not cuda_ok:
                print("Torch not compiled with CUDA; using --device cpu for simulation.", flush=True)
                sim_device = "cpu"
            else:
                try:
                    torch.zeros(1, device=sim_device)
                except AssertionError:
                    print("Torch not compiled with CUDA; using --device cpu for simulation.", flush=True)
                    sim_device = "cpu"
        cli_args_list = ["--device", sim_device, "--num_envs", str(getattr(args, "num_envs", 1))]

        pi_remote_policy = None
        pi_remote_camera_key_map = None
        if args.policy_type == "pi_remote":
            from molmo_spaces_isaac.arena.pi_remote_client import PiRemotePolicy
            try:
                print("Connecting to Pi server...", flush=True)
                pi_remote_policy = PiRemotePolicy(
                    host=args.pi_server_host,
                    port=args.pi_server_port,
                    task_description="pick up the object.",
                    grasping_threshold=args.pi_grasping_threshold,
                    chunk_size=args.pi_chunk_size,
                    connect_timeout_s=10.0,
                    inference_timeout_s=120.0,
                )
                pi_remote_policy.reset()
                pi_remote_camera_key_map = _default_pi_camera_key_map(_selected_embodiment_key(args))
            except Exception as e:
                import traceback
                print(f"Failed to connect to Pi server: {e}", flush=True)
                traceback.print_exc()
                if simulation_app is not None:
                    simulation_app.close()
                return 1

        results: list[dict] = []

        def _write_multi_episode_results() -> None:
            n_ok_so_far = sum(1 for row in results if row.get("success"))
            n_done = len(results)
            _write_results_json(
                args.results_json,
                {
                    "benchmark_dir": str(resolved_bench_dir) if resolved_bench_dir else None,
                    "arena_spec_manifest": str(resolved_arena_spec_manifest) if resolved_arena_spec_manifest else None,
                    "episode_indices": indices_to_run,
                    "completed_count": n_done,
                    "planned_count": len(indices_to_run),
                    "policy_type": args.policy_type,
                    "steps": args.steps,
                    "num_envs": getattr(args, "num_envs", 1),
                    "pi_trace_dir": str(args.pi_trace_dir) if args.pi_trace_dir else None,
                    "success_count": n_ok_so_far,
                    "total_count": n_done,
                    "success_rate": (float(n_ok_so_far) / float(n_done)) if n_done else None,
                    "results": results,
                },
            )

        for idx in indices_to_run:
            if resolved_arena_spec_manifest is not None and manifest_task_descriptions is not None:
                spec, episode_dict, _ = load_arena_spec_from_manifest(
                    resolved_arena_spec_manifest,
                    idx,
                    scenes_root=args.scenes_root,
                    task_descriptions=manifest_task_descriptions,
                )
            else:
                episode_dict = episode_dicts[idx]
                spec = episode_dict_to_arena_spec(
                    episode_dict,
                    require_thor_only=require_thor_only_mode,
                    background_key=args.background,
                    scenes_root=args.scenes_root,
                )
            if spec is None:
                why = (
                    "needs Objaverse or not THOR-only (use --allow-objaverse to run)"
                    if require_thor_only_mode
                    else "not supported for Arena"
                )
                print(f"Episode {idx}: skipped ({why})", flush=True)
                results.append(
                    {
                        **_episode_result_metadata(idx, episode_dict, spec=None),
                        "status": "skipped",
                        "success": False,
                        "step_count": 0,
                        "reason": why,
                    }
                )
                continue
            print(f"Episode {idx}/{len(indices_to_run)}...", flush=True)
            if pi_remote_policy is not None:
                pi_remote_policy.reset()
                task_description = (episode_dict.get("language") or {}).get("task_description")
                if hasattr(pi_remote_policy, "set_task_description"):
                    pi_remote_policy.set_task_description(task_description or "pick up the object.")
            success, step_count, video_paths = _run_one_episode(
                spec, episode_dict, args,
                thor_assets_dir, thor_metadata_path, objaverse_assets_dir,
                cli_args_list, pi_remote_policy, pi_remote_camera_key_map,
                env_name=f"molospaces_arena_benchmark_{idx}",
                episode_idx_for_artifacts=idx,
            )
            results.append(
                {
                    **_episode_result_metadata(idx, episode_dict, spec=spec),
                    "status": "ran",
                    "success": bool(success),
                    "step_count": int(step_count),
                    "reason": None if success else "failed_or_timed_out",
                    "video_paths": video_paths,
                }
            )
            print(f"  Episode {idx}: {'SUCCESS' if success else 'FAIL'} ({step_count} steps)", flush=True)
            _write_multi_episode_results()

        n_ok = sum(1 for row in results if row.get("success"))
        n_total = len(results)
        if n_total == 0:
            print("\nNo episodes run.", flush=True)
        else:
            print(f"\nBenchmark complete: {n_ok}/{n_total} successful ({100.0 * n_ok / n_total:.1f}%)", flush=True)
        _write_multi_episode_results()
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass
        os._exit(0 if n_ok == n_total else 1)

    if spec is None:
        spec = episode_dict_to_arena_spec(
            episode_dict,
            require_thor_only=require_thor_only_mode,
            background_key=args.background,
            scenes_root=args.scenes_root,
        )
    if spec is None and require_thor_only_mode and resolved_bench_dir is not None:
        for j, ep in enumerate(_load_all_episode_dicts(resolved_bench_dir)):
            s = episode_dict_to_arena_spec(
                ep,
                require_thor_only=True,
                background_key=args.background,
                scenes_root=args.scenes_root,
            )
            if s is not None:
                if j != args.episode_idx:
                    print(
                        f"Episode --episode_idx {args.episode_idx} is not THOR-only or unsupported; "
                        f"using index {j} (first THOR-only pick episode in this benchmark).",
                        flush=True,
                    )
                episode_dict = ep
                spec = s
                break
    if spec is None:
        sm = (episode_dict.get("scene_modifications") or {}).get("added_objects") or {}
        any_objaverse = any("objaverse" in (str(v) or "").lower() for v in sm.values())
        task = (episode_dict.get("task") or {})
        pickup_name = task.get("pickup_obj_name")
        pickup_not_in_added = pickup_name and pickup_name not in sm
        if not require_thor_only_mode and (any_objaverse or pickup_not_in_added):
            spec = episode_dict_to_arena_spec(
                episode_dict,
                require_thor_only=False,
                background_key=args.background,
                scenes_root=args.scenes_root,
            )
        if spec is None:
            if require_thor_only_mode and (any_objaverse or pickup_not_in_added):
                raise SystemExit(
                    "This episode needs Objaverse USDs. Add --allow-objaverse and install objects under "
                    "assets_root/objects/objaverse/ (or set MOLMO_OBJAVERSE_USD_DIR), or use a THOR-only "
                    "episode / benchmark (e.g. examples/benchmark_ithor_pick_hard_simple)."
                )
            if not require_thor_only_mode and (any_objaverse or pickup_not_in_added):
                raise SystemExit(
                    "This episode uses Objaverse but could not build spec or find Objaverse USDs. "
                    "Set MOLMO_OBJAVERSE_USD_DIR or use --assets_root with objects/objaverse/ containing obja_<id>/ subdirs."
                )
            raise SystemExit(
                "This episode is not supported for Arena (need pick task with THOR or Objaverse objects)."
            )

    # Resolve asset dirs: --assets_root or MOLMO_ISAAC_ASSETS_ROOT (e.g. /path/to/molmospaces_isaac)
    if args.assets_root is not None and not Path(args.assets_root).resolve().is_dir():
        raise SystemExit(f"Assets root is not a directory: {args.assets_root}")
    if args.assets_root is not None:
        os.environ.setdefault("MOLMO_ISAAC_ASSETS_ROOT", str(Path(args.assets_root).resolve()))
    thor_assets_dir, thor_metadata_path, objaverse_assets_dir = _resolve_asset_dirs(args.assets_root)

    import torch

    # Sim device: prefer GPU; fall back to CPU when PyTorch is not compiled with CUDA (e.g. Isaac Sim aarch64).
    sim_device = getattr(args, "device", "cuda:0") or "cuda:0"
    if sim_device == "cuda":
        sim_device = "cuda:0"
    if sim_device.startswith("cuda"):
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        if not cuda_ok:
            print("Torch not compiled with CUDA; using --device cpu for simulation.", flush=True)
            sim_device = "cpu"
        else:
            try:
                torch.zeros(1, device=sim_device)
            except AssertionError:
                print("Torch not compiled with CUDA; using --device cpu for simulation.", flush=True)
                sim_device = "cpu"
    cli_args_list = ["--device", sim_device]

    hi = episode_dict.get("house_index")
    print(
        f"[molmospaces_arena] Loaded episode house_index={hi} (--episode_idx={args.episode_idx}).",
        flush=True,
    )

    su = getattr(spec, "scene_usd_path", None)
    if su is not None and Path(su).is_file():
        print(f"[molmospaces_arena] MolmoSpaces scene USD: {su}", flush=True)
    else:
        print(
            f"[molmospaces_arena] No FloorPlan scene USD resolved (scenes_root={getattr(args, 'scenes_root', None)}); "
            f"using Arena background '{args.background}'. For iTHOR demo: ms-download --scenes ithor into the same "
            "root as --assets_root, or pass --scenes_root.",
            flush=True,
        )

    # Arena env step is 0.02s; set episode length so timeout matches --steps.
    episode_length_s = args.steps * 0.02
    extra = tuple(getattr(args, "scene_extra_xyz", [0.0, 0.0, 0.0])[:3])
    env_id = f"molmospaces_arena_hi{hi}_ep{args.episode_idx}" if hi is not None else "molmospaces_arena_benchmark"
    _enable_cameras = getattr(args, "with_cameras", False) or (getattr(args, "policy_type", "zero") == "pi_remote")
    embodiment_key = _selected_embodiment_key(args)
    print(f"[molmospaces_arena] Arena embodiment: {embodiment_key}", flush=True)
    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name=env_id,
        embodiment_key=embodiment_key,
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=episode_length_s,
        cli_args_list=cli_args_list,
        scene_extra_translation_xyz=extra,
        use_joint_pos_control=getattr(args, "joint_pos_policy", False),
        use_joint_velocity_control=_use_pi_joint_velocity_control(args),
        num_envs=getattr(args, "num_envs", 1),
        env_spacing=getattr(args, "env_spacing", None),
        enable_cameras=_enable_cameras,
    )
    try:
        obs, _ = env.reset()
    except Exception as e:
        import traceback
        print(f"env.reset() failed: {e}", flush=True)
        traceback.print_exc()
        env.close()
        if simulation_app is not None:
            simulation_app.close()
        return 1

    video_recorder = None
    video_paths: dict[str, str] = {}
    if getattr(args, "record_video_dir", None) is not None:
        video_recorder = ArenaVideoRecorder(
            args.record_video_dir,
            episode_idx=int(getattr(args, "episode_idx", 0)),
            camera_keys=_record_camera_keys(args),
            fps=getattr(args, "record_video_fps", 15),
        )
        video_recorder.capture(obs, step_count=0)

    # Capture pick spawn Z for per-step tunneling debug (--debug_pick_z).
    spawn_z: float | None = None
    try:
        _scene_m = getattr(env.unwrapped, "scene", None)
        if _scene_m is not None:
            _pick_key_m = getattr(spec, "pickup_name", None) or ""
            _rp_m = _scene_m[_pick_key_m].data.root_pos_w
            spawn_z = float(_rp_m[0, 2].item()) if _rp_m.dim() > 1 else float(_rp_m[2].item())
    except Exception:
        pass

    # GUI pause: keep window alive for inspection before physics loop starts.
    pause_s = getattr(args, "pause_after_reset", 0.0)
    if pause_s > 0.0 and simulation_app is not None:
        import time
        print(f"[molmospaces_arena] Pausing {pause_s:.0f}s for inspection (--pause_after_reset)...", flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < pause_s:
            if not simulation_app.is_running():
                break
            simulation_app.update()

    pi_remote_policy = None
    pi_remote_camera_key_map = None
    h5_replay_policy = None
    if args.policy_type == "pi_remote":
        from molmo_spaces_isaac.arena.pi_remote_client import PiRemotePolicy, get_pi_remote_action
        task_description = (episode_dict.get("language") or {}).get("task_description") or "pick up the object."
        try:
            print("Connecting to Pi server (ensure OpenPI server is running)...", flush=True)
            pi_remote_policy = PiRemotePolicy(
                host=args.pi_server_host,
                port=args.pi_server_port,
                task_description=task_description,
                grasping_threshold=args.pi_grasping_threshold,
                chunk_size=args.pi_chunk_size,
                connect_timeout_s=10.0,
                inference_timeout_s=120.0,
            )
            pi_remote_policy.reset()
            # Camera key map: maps pi0 camera names to Arena obs keys (format "group.term").
            # Default keys match the selected Arena embodiment.
            # Override with env vars MOLMO_PI_WRIST_CAMERA / MOLMO_PI_EXO_CAMERA if needed.
            pi_remote_camera_key_map = _default_pi_camera_key_map(embodiment_key)
            print(
                f"[molmospaces_arena] Pi camera keys: wrist_camera={pi_remote_camera_key_map['wrist_camera']} "
                f"exo_camera_1={pi_remote_camera_key_map['exo_camera_1']}",
                flush=True,
            )
            print("Pi remote policy connected; running episode (no molmo_spaces/mujoco in this process).", flush=True)
        except Exception as e:
            import traceback
            print(f"Failed to connect to Pi server: {e}", flush=True)
            traceback.print_exc()
            env.close()
            if simulation_app is not None:
                simulation_app.close()
            return 1
    elif args.policy_type == "h5_replay":
        if args.replay_h5 is None:
            env.close()
            if simulation_app is not None:
                simulation_app.close()
            raise SystemExit("--policy_type h5_replay requires --replay_h5")
        h5_replay_policy = H5ReplayPolicy(
            args.replay_h5,
            traj=args.replay_traj,
            start_index=getattr(args, "replay_start_index", 0),
        )
        if getattr(args, "replay_init_from_h5_qpos", False):
            _write_arena_robot_qpos_from_mujoco(
                env,
                h5_replay_policy.qpos_at(getattr(args, "replay_start_index", 0)),
            )
            obs = env.unwrapped.obs_buf

    step_count = 0
    n_success = 0
    num_envs = getattr(args, "num_envs", 1)
    terminated = truncated = None
    device = env.unwrapped.device
    pi_repeat = _pi_action_repeat(args)
    pi_repeat_left = 0
    pi_last_actions = None
    if args.policy_type == "pi_remote":
        print(
            f"[molmospaces_arena] pi_remote action mode: "
            f"{'joint_velocity' if _use_pi_joint_velocity_control(args) else 'joint_position' if getattr(args, 'joint_pos_policy', False) else 'delta_eef'}; "
            f"action_repeat={pi_repeat}.",
            flush=True,
        )
    elif args.policy_type == "h5_replay":
        print(
            f"[molmospaces_arena] h5_replay action mode: joint_position; "
            f"action_repeat={max(1, args.replay_action_repeat)}.",
            flush=True,
        )
    print(f"Starting benchmark loop ({num_envs} env(s), max {args.steps} steps)...", flush=True)
    try:
        for _ in range(args.steps):
            if simulation_app is not None and not simulation_app.is_running():
                print("Simulation app stopped (e.g. window closed); exiting loop.", flush=True)
                break
            with torch.inference_mode():
                if args.policy_type == "pi_remote" and pi_remote_policy is not None:
                    if pi_last_actions is None or pi_repeat_left <= 0:
                        actions = get_pi_remote_action(
                            pi_remote_policy,
                            obs,
                            camera_key_map=pi_remote_camera_key_map,
                            default_image_shape=(224, 224, 3),
                            device=device,
                            use_joint_pos_control=getattr(args, "joint_pos_policy", False),
                            use_joint_velocity_control=_use_pi_joint_velocity_control(args),
                        )
                        pi_last_actions = actions
                        pi_repeat_left = pi_repeat
                    else:
                        actions = pi_last_actions
                    pi_repeat_left -= 1
                elif args.policy_type == "h5_replay" and h5_replay_policy is not None:
                    if pi_last_actions is None or pi_repeat_left <= 0:
                        actions = h5_replay_policy.next_action(device)
                        pi_last_actions = actions
                        pi_repeat_left = max(1, args.replay_action_repeat)
                    else:
                        actions = pi_last_actions
                    pi_repeat_left -= 1
                elif args.policy_type == "random":
                    actions = torch.randn(env.action_space.shape, device=device) * 0.05
                else:
                    actions = torch.zeros(env.action_space.shape, device=device)
                obs, _reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if (
                video_recorder is not None
                and step_count % max(1, int(getattr(args, "record_video_stride", 1))) == 0
            ):
                video_recorder.capture(obs, step_count=step_count)
            if args.progress_steps > 0 and step_count % args.progress_steps == 0:
                print(f"  step {step_count}/{args.steps}...", flush=True)
            if args.debug_arm_motion > 0 and step_count % args.debug_arm_motion == 0:
                _print_arm_motion_debug(step_count, obs, actions, device)
            if args.debug_pick_z > 0 and step_count % args.debug_pick_z == 0 and spawn_z is not None:
                _print_pick_z_debug(step_count, env, spec, spawn_z, obs=obs, actions=actions)
            done = terminated | truncated
            if done.all():
                n_success = int(terminated.sum().item())
                break
    except Exception as e:
        import traceback
        print(f"Benchmark loop error after {step_count} steps: {e}", flush=True)
        traceback.print_exc()
    finally:
        if video_recorder is not None:
            try:
                if step_count > 0 and step_count % max(1, int(getattr(args, "record_video_stride", 1))) != 0:
                    video_recorder.capture(obs, step_count=step_count)
            except Exception:
                pass
            video_paths = video_recorder.close()

    success = n_success > 0
    # Print result before closing: simulation_app.close() tears down Isaac Sim and exits the process.
    if num_envs > 1:
        print(
            f"Benchmark finished in {step_count} steps. Result: {n_success}/{num_envs} envs succeeded"
            f" ({100.0 * n_success / num_envs:.0f}%) [parallel].",
            flush=True,
        )
    else:
        print(f"Benchmark episode finished in {step_count} steps. Result: {'SUCCESS (object placed)' if success else 'FAIL (timeout or incomplete)'}.", flush=True)
    if not success and step_count >= args.steps:
        print("Tip: task may need more time; try increasing --steps (e.g. --steps 2000).", flush=True)
    exit_code = 0 if success else 1
    _write_results_json(
        args.results_json,
        {
            "benchmark_dir": str(resolved_bench_dir) if resolved_bench_dir else None,
            "arena_spec_manifest": str(resolved_arena_spec_manifest) if resolved_arena_spec_manifest else None,
            "episode_indices": [int(getattr(args, "episode_idx", 0))]
            if (resolved_bench_dir or resolved_arena_spec_manifest)
            else None,
            "policy_type": args.policy_type,
            "steps": args.steps,
            "num_envs": num_envs,
            "pi_trace_dir": str(args.pi_trace_dir) if args.pi_trace_dir else None,
            "video_paths": video_paths,
            "success_count": int(n_success),
            "total_count": int(num_envs),
            "success_rate": (float(n_success) / float(num_envs)) if num_envs else None,
            "results": [
                {
                    **_episode_result_metadata(
                        int(getattr(args, "episode_idx", 0))
                        if (resolved_bench_dir or resolved_arena_spec_manifest)
                        else 0,
                        episode_dict,
                        spec=spec,
                    ),
                    "status": "ran",
                    "success": bool(success),
                    "step_count": int(step_count),
                    "reason": None if success else "failed_or_timed_out",
                    "video_paths": video_paths,
                }
            ],
        },
    )

    try:
        env.close()
    except Exception as e:
        import traceback
        print(f"env.close() failed: {e}", flush=True)
        traceback.print_exc()
    if simulation_app is not None:
        simulation_app.close()  # Shuts down Isaac Sim.
    # Force process exit so the script always terminates (Isaac/Omniverse may leave process alive otherwise).
    os._exit(exit_code)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass
        os._exit(1)
