#!/usr/bin/env python3
"""Run MolmoSpaces pick in Isaac Lab Arena (demo-oriented).

Default: one bundled episode JSON (``examples/pick_episode_ithor_thor_only.json``) when present;
otherwise a pick benchmark directory (see ``MOLMO_PICK_BENCHMARK_DIR`` / ``--benchmark_dir``).
Scene vertical placement uses inverse ``robot_base_pose`` plus ``--scene_extra_xyz`` and optional
``MOLMO_ARENA_SCENE_FINE_Z`` only (no automatic iTHOR Z offsets). Success = lift_height >= threshold.
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
    choices=["zero", "random", "pi_remote"],
    default="zero",
    help="Action source: 'zero', 'random' (small Gaussian noise to verify arm motion), or 'pi_remote' (OpenPI server; start server first).",
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
        "Required for pi0 DROID joint-position policies like pi05_droid_jointpos. "
        "Switches Arena Franka from IK delta-EEF to JointPositionActionCfg."
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


def _print_pick_z_debug(step_count: int, env, spec, spawn_z: float) -> None:
    """Print pick object Z, lift from spawn, and a warning when below spawn (tunneling)."""
    try:
        base_env = getattr(env, "unwrapped", env)
        scene = getattr(base_env, "scene", None)
        if scene is None:
            print(f"  [pick_z] step {step_count}: no scene", flush=True)
            return
        pick_key = getattr(spec, "pickup_name", None) or ""
        rp = scene[pick_key].data.root_pos_w
        z = float(rp[0, 2].item()) if rp.dim() > 1 else float(rp[2].item())
        lift = z - spawn_z
        warn = "  *** BELOW SPAWN ***" if lift < -0.05 else ""
        print(
            f"  [pick_z] step {step_count}: z={z:.4f}  spawn_z={spawn_z:.4f}  lift={lift:+.4f}{warn}",
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
):
    """Build env from spec, run episode, close env. Returns (success, step_count). Use unique env_name per episode to avoid gym duplicate registration."""
    import torch
    from molmo_spaces_isaac.arena.build_from_spec import build_arena_env_from_episode_spec

    episode_length_s = args.steps * 0.02
    extra = tuple(getattr(args, "scene_extra_xyz", [0.0, 0.0, 0.0])[:3])
    _enable_cameras = getattr(args, "with_cameras", False) or (getattr(args, "policy_type", "zero") == "pi_remote")
    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name=env_name,
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=episode_length_s,
        cli_args_list=cli_args_list,
        scene_extra_translation_xyz=extra,
        use_joint_pos_control=getattr(args, "joint_pos_policy", False),
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

    step_count = 0
    success = False
    device = env.unwrapped.device
    try:
        for _ in range(args.steps):
            if simulation_app is not None and not simulation_app.is_running():
                break
            with torch.inference_mode():
                if args.policy_type == "pi_remote" and pi_remote_policy is not None:
                    from molmo_spaces_isaac.arena.pi_remote_client import get_pi_remote_action
                    actions = get_pi_remote_action(
                        pi_remote_policy,
                        obs,
                        camera_key_map=pi_remote_camera_key_map or {},
                        default_image_shape=(224, 224, 3),
                        device=device,
                        use_joint_pos_control=getattr(args, "joint_pos_policy", False),
                    )
                elif args.policy_type == "random":
                    actions = torch.randn(env.action_space.shape, device=device) * 0.05
                else:
                    actions = torch.zeros(env.action_space.shape, device=device)
                obs, _reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if args.progress_steps > 0 and step_count % args.progress_steps == 0:
                print(f"  step {step_count}/{args.steps}...", flush=True)
            if args.debug_arm_motion > 0 and step_count % args.debug_arm_motion == 0:
                _print_arm_motion_debug(step_count, obs, actions, device)
            if args.debug_pick_z > 0 and step_count % args.debug_pick_z == 0 and spawn_z is not None:
                _print_pick_z_debug(step_count, env, spec, spawn_z)
            done = terminated | truncated
            if done.any():
                success = bool(terminated.any().item())
                break
    except Exception as e:
        import traceback
        print(f"Episode loop error after {step_count} steps: {e}", flush=True)
        traceback.print_exc()
        success = False

    try:
        env.close()
    except Exception:
        pass
    return success, step_count


def main() -> int:
    if getattr(args, "scenes_root", None) is None and os.environ.get("MOLMO_SCENES_ROOT"):
        args.scenes_root = Path(os.environ["MOLMO_SCENES_ROOT"]).resolve()
    if args.episode_json is not None and args.benchmark_dir is not None:
        raise SystemExit("Use either --episode_json or --benchmark_dir, not both.")

    _bundled_simple_bench = REPO_ROOT / "examples" / "benchmark_ithor_pick_hard_simple"

    if args.episode_json is None and args.benchmark_dir is None:
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
    if args.benchmark_dir is not None:
        bench_dir = Path(args.benchmark_dir)
        if not bench_dir.is_dir() and not bench_dir.is_absolute():
            bench_dir = REPO_ROOT / bench_dir
        if not bench_dir.is_dir():
            raise SystemExit(f"Benchmark directory not found: {args.benchmark_dir}")
        resolved_bench_dir = bench_dir

        max_ep = getattr(args, "max_episodes", None)
        if max_ep is not None:
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
                    connect_timeout_s=10.0,
                    inference_timeout_s=120.0,
                )
                pi_remote_policy.reset()
                _default_wrist = "camera_obs.wrist_cam_rgb"
                _default_exo = "camera_obs.exo_cam_rgb"
                pi_remote_camera_key_map = {
                    "wrist_camera": os.environ.get("MOLMO_PI_WRIST_CAMERA", _default_wrist),
                    "exo_camera_1": os.environ.get("MOLMO_PI_EXO_CAMERA", _default_exo),
                }
            except Exception as e:
                import traceback
                print(f"Failed to connect to Pi server: {e}", flush=True)
                traceback.print_exc()
                if simulation_app is not None:
                    simulation_app.close()
                return 1

        results = []
        for idx in indices_to_run:
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
                results.append((idx, False, 0))
                continue
            print(f"Episode {idx}/{len(indices_to_run)}...", flush=True)
            success, step_count = _run_one_episode(
                spec, episode_dict, args,
                thor_assets_dir, thor_metadata_path, objaverse_assets_dir,
                cli_args_list, pi_remote_policy, pi_remote_camera_key_map,
                env_name=f"molospaces_arena_benchmark_{idx}",
            )
            results.append((idx, success, step_count))
            print(f"  Episode {idx}: {'SUCCESS' if success else 'FAIL'} ({step_count} steps)", flush=True)

        n_ok = sum(1 for _, s, _ in results if s)
        n_total = len(results)
        if n_total == 0:
            print("\nNo episodes run.", flush=True)
        else:
            print(f"\nBenchmark complete: {n_ok}/{n_total} successful ({100.0 * n_ok / n_total:.1f}%)", flush=True)
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass
        os._exit(0 if n_ok == n_total else 1)

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
    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name=env_id,
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=episode_length_s,
        cli_args_list=cli_args_list,
        scene_extra_translation_xyz=extra,
        use_joint_pos_control=getattr(args, "joint_pos_policy", False),
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
    if args.policy_type == "pi_remote":
        from molmo_spaces_isaac.arena.pi_remote_client import PiRemotePolicy, get_pi_remote_action
        task_description = (episode_dict.get("language") or {}).get("task_description") or "pick up the object."
        try:
            print("Connecting to Pi server (ensure OpenPI server is running)...", flush=True)
            pi_remote_policy = PiRemotePolicy(
                host=args.pi_server_host,
                port=args.pi_server_port,
                task_description=task_description,
                connect_timeout_s=10.0,
                inference_timeout_s=120.0,
            )
            pi_remote_policy.reset()
            # Camera key map: maps pi0 camera names to Arena obs keys (format "group.term").
            # Default keys match _attach_franka_droid_cameras (wrist_cam + exo_cam).
            # Override with env vars MOLMO_PI_WRIST_CAMERA / MOLMO_PI_EXO_CAMERA if needed.
            _default_wrist = "camera_obs.wrist_cam_rgb"
            _default_exo = "camera_obs.exo_cam_rgb"
            pi_remote_camera_key_map = {
                "wrist_camera": os.environ.get("MOLMO_PI_WRIST_CAMERA", _default_wrist),
                "exo_camera_1": os.environ.get("MOLMO_PI_EXO_CAMERA", _default_exo),
            }
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

    step_count = 0
    n_success = 0
    num_envs = getattr(args, "num_envs", 1)
    terminated = truncated = None
    device = env.unwrapped.device
    print(f"Starting benchmark loop ({num_envs} env(s), max {args.steps} steps)...", flush=True)
    try:
        for _ in range(args.steps):
            if simulation_app is not None and not simulation_app.is_running():
                print("Simulation app stopped (e.g. window closed); exiting loop.", flush=True)
                break
            with torch.inference_mode():
                if args.policy_type == "pi_remote" and pi_remote_policy is not None:
                    actions = get_pi_remote_action(
                        pi_remote_policy,
                        obs,
                        camera_key_map=pi_remote_camera_key_map,
                        default_image_shape=(224, 224, 3),
                        device=device,
                        use_joint_pos_control=getattr(args, "joint_pos_policy", False),
                    )
                elif args.policy_type == "random":
                    actions = torch.randn(env.action_space.shape, device=device) * 0.05
                else:
                    actions = torch.zeros(env.action_space.shape, device=device)
                obs, _reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if args.progress_steps > 0 and step_count % args.progress_steps == 0:
                print(f"  step {step_count}/{args.steps}...", flush=True)
            if args.debug_arm_motion > 0 and step_count % args.debug_arm_motion == 0:
                _print_arm_motion_debug(step_count, obs, actions, device)
            if args.debug_pick_z > 0 and step_count % args.debug_pick_z == 0 and spawn_z is not None:
                _print_pick_z_debug(step_count, env, spec, spawn_z)
            done = terminated | truncated
            if done.all():
                n_success = int(terminated.sum().item())
                break
    except Exception as e:
        import traceback
        print(f"Benchmark loop error after {step_count} steps: {e}", flush=True)
        traceback.print_exc()

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
    raise SystemExit(main())
