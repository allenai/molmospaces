#!/usr/bin/env python3
"""Run MolmoSpaces pick/pick_and_place in Isaac Lab Arena. Needs Arena + THOR/Objaverse USDs (--assets_root). Use --benchmark_dir + --episode_idx or --max_episodes; or --episode_json. pi_remote: docs/LEARNED_POLICY_IN_ISAAC.md."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make Isaac Lab Arena importable (set ISAACLAB_ARENA_PATH to your Arena repo root)
_arena_path = os.environ.get("ISAACLAB_ARENA_PATH")
if _arena_path:
    _arena_path = Path(_arena_path).resolve()
    if _arena_path.is_dir() and str(_arena_path) not in sys.path:
        sys.path.insert(0, str(_arena_path))

REPO_ROOT = Path(__file__).resolve().parent.parent  # molmo_spaces_isaac
MOLMOSPACES_ROOT = REPO_ROOT.parent  # workspace containing molmo_spaces + molmo_spaces_isaac
SRC_ROOT = REPO_ROOT / "src"

for p in (SRC_ROOT, REPO_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isaaclab.app import AppLauncher

if len(sys.argv) > 1 and sys.argv[1] == "--":
    sys.argv.pop(1)

parser = argparse.ArgumentParser(
    description="Run a MolmoSpaces benchmark episode in Isaac Lab Arena (pick / pick_and_place)"
)
parser.add_argument(
    "--episode_json",
    type=Path,
    default=None,
    help="Path to a single episode JSON file (self-contained episode spec)",
)
parser.add_argument(
    "--benchmark_dir",
    type=Path,
    default=None,
    help="Path to benchmark directory (benchmark.json or house_*/episode_*.json). E.g. .../molmospaces_bench/mujoco/benchmarks/.../FrankaPickandPlaceHardBench_*_json_benchmark.",
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
    help="Episode index when using --benchmark_dir for a single episode (default: 0). Ignored if --max_episodes is set.",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=None,
    metavar="N",
    help="Run N episodes from the benchmark (0 = all). Like MolmoSpaces; default: run single episode at --episode_idx.",
)
parser.add_argument("--steps", type=int, default=5000, help="Max simulation steps per episode (benchmark runs until done or this limit).")
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
    "--policy_type",
    type=str,
    choices=["zero", "pi_remote"],
    default="zero",
    help="Action source: 'zero' or 'pi_remote' (OpenPI server; start server first).",
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
    help="Root dir for MolmoSpaces scene USDs (e.g. assets/usd/scenes or MOLMO_SCENES_ROOT). When set, the episode's scene is loaded so the same environment as the benchmark is used.",
)
parser.add_argument(
    "--background",
    type=str,
    default="kitchen",
    help="Arena background when episode scene USD is not found (default: kitchen).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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
    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name=env_name,
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=episode_length_s,
        cli_args_list=cli_args_list,
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
                    )
                else:
                    actions = torch.zeros(env.action_space.shape, device=device)
                obs, _reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if args.progress_steps > 0 and step_count % args.progress_steps == 0:
                print(f"  step {step_count}/{args.steps}...", flush=True)
            if args.debug_arm_motion > 0 and step_count % args.debug_arm_motion == 0:
                _print_arm_motion_debug(step_count, obs, actions, device)
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
    if args.episode_json is None and args.benchmark_dir is None:
        raise SystemExit("Provide --episode_json or --benchmark_dir.")

    from molmo_spaces_isaac.arena.episode_to_arena import episode_dict_to_arena_spec

    if args.benchmark_dir is not None:
        bench_dir = Path(args.benchmark_dir)
        if not bench_dir.is_dir() and not bench_dir.is_absolute():
            bench_dir = REPO_ROOT / bench_dir
        if not bench_dir.is_dir():
            raise SystemExit(f"Benchmark directory not found: {args.benchmark_dir}")

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
            path = MOLMOSPACES_ROOT / path
        if not path.is_file():
            raise SystemExit(f"Episode file not found: {path}")
        episode_dict = load_episode_dict_from_json(path)
        episode_dicts = None
        indices_to_run = None

    from molmo_spaces_isaac.arena.build_from_spec import build_arena_env_from_episode_spec

    # Multi-episode run (like MolmoSpaces): run each episode in sequence and print summary.
    if indices_to_run is not None:
        if args.assets_root is not None and not Path(args.assets_root).resolve().is_dir():
            raise SystemExit(f"Assets root is not a directory: {args.assets_root}")
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
        cli_args_list = ["--device", sim_device]

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
                pi_remote_camera_key_map = {}
                if os.environ.get("MOLMO_PI_WRIST_CAMERA"):
                    pi_remote_camera_key_map["wrist_camera"] = os.environ["MOLMO_PI_WRIST_CAMERA"]
                if os.environ.get("MOLMO_PI_EXO_CAMERA"):
                    pi_remote_camera_key_map["exo_camera_1"] = os.environ["MOLMO_PI_EXO_CAMERA"]
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
            spec = episode_dict_to_arena_spec(episode_dict, require_thor_only=False, background_key=args.background, scenes_root=args.scenes_root)
            if spec is None:
                print(f"Episode {idx}: skipped (not supported for Arena)", flush=True)
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

    spec = episode_dict_to_arena_spec(episode_dict, require_thor_only=False, background_key=args.background, scenes_root=args.scenes_root)
    if spec is None:
        sm = (episode_dict.get("scene_modifications") or {}).get("added_objects") or {}
        any_objaverse = any("objaverse" in (str(v) or "").lower() for v in sm.values())
        task = (episode_dict.get("task") or {})
        pickup_name = task.get("pickup_obj_name")
        pickup_not_in_added = pickup_name and pickup_name not in sm
        # Auto-allow Objaverse when episode has Objaverse in added_objects, or pickup has pose but not in added_objects (UUID-from-name fallback)
        if any_objaverse or pickup_not_in_added:
            spec = episode_dict_to_arena_spec(episode_dict, require_thor_only=False, background_key=args.background, scenes_root=args.scenes_root)
        if spec is None:
            if any_objaverse or pickup_not_in_added:
                raise SystemExit(
                    "This episode uses Objaverse but could not build spec. "
                    "Set MOLMO_OBJAVERSE_USD_DIR (or MOLMO_ISAAC_ASSETS_ROOT with objects/objaverse/) to the dir with obja_<id> subdirs."
                )
            raise SystemExit(
                "This episode is not supported for Arena (need pick/pick_and_place with THOR or Objaverse objects)."
            )

    # Resolve asset dirs: --assets_root or MOLMO_ISAAC_ASSETS_ROOT (e.g. /path/to/molmospaces_isaac)
    if args.assets_root is not None and not Path(args.assets_root).resolve().is_dir():
        raise SystemExit(f"Assets root is not a directory: {args.assets_root}")
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

    # Arena env step is 0.02s; set episode length so timeout matches --steps.
    episode_length_s = args.steps * 0.02
    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name="molmospaces_arena_benchmark",
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=episode_length_s,
        cli_args_list=cli_args_list,
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
            pi_remote_camera_key_map = {}
            if os.environ.get("MOLMO_PI_WRIST_CAMERA"):
                pi_remote_camera_key_map["wrist_camera"] = os.environ["MOLMO_PI_WRIST_CAMERA"]
            if os.environ.get("MOLMO_PI_EXO_CAMERA"):
                pi_remote_camera_key_map["exo_camera_1"] = os.environ["MOLMO_PI_EXO_CAMERA"]
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
    success = False
    device = env.unwrapped.device
    print(f"Starting benchmark loop (max {args.steps} steps)...", flush=True)
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
                    )
                else:
                    actions = torch.zeros(env.action_space.shape, device=device)
                obs, _reward, terminated, truncated, info = env.step(actions)
            step_count += 1
            if args.progress_steps > 0 and step_count % args.progress_steps == 0:
                print(f"  step {step_count}/{args.steps}...", flush=True)
            if args.debug_arm_motion > 0 and step_count % args.debug_arm_motion == 0:
                _print_arm_motion_debug(step_count, obs, actions, device)
            done = terminated | truncated
            if done.any():
                success = bool(terminated.any().item())
                break
    except Exception as e:
        import traceback
        print(f"Benchmark loop error after {step_count} steps: {e}", flush=True)
        traceback.print_exc()
        success = False

    # Print result before closing: simulation_app.close() tears down Isaac Sim and exits the process.
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
