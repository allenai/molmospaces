"""
Launcher for benchmark evaluations with mujoco  environment management (for OpenGL/filament renderer).

Usage:
    python scripts/benchmarks/run_benchmarks.py --list
    python scripts/benchmarks/run_benchmarks.py pick_classic --eval_config module.path:EvalConfigClass --checkpoint_path /path/to/ckpt
    python scripts/benchmarks/run_benchmarks.py --set mb_manip --eval_config module.path:EvalConfigClass --num_workers 4
See https://molmospaces.allen.ai/leaderboard for benchmark results.

MS Manip Easy Tasks (MS-Pick, MS-PnP, MS-Open, MS-Close) are not supported
here. They were evaluated with MolmoSpaces <= 0939e18 which used a different
pick success condition, policy_dt=100ms, and single-word prompts.
See https://allenai.org/papers/molmospaces
"""

import argparse
import subprocess
import sys
from pathlib import Path

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR


DEFAULT_BENCH_PARAMS = {
    "sets": [],
    "path": None,
    "use_filament": True,
    "camera_names": None,
    "use_eval_cameras": False,
    "camera_rand_level": None,
}

BENCHMARKS = {
    # ======================================================================
    # MB Manip Tasks  (Paper: https://allenai.github.io/MolmoBot)
    #   Pick: 20s horizon → 303 steps @ 66ms policy_dt
    #   PnP:  40s horizon → 606 steps @ 66ms policy_dt
    # ======================================================================
    "pick_msproc": {
        "sets": ["mb_manip"],
        "description": "Pick-MSProc: THOR objects, MSProc houses, classic mujoco renderer",
        "path": "molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
        "use_filament": False,
    },
    "pick_classic": {
        "sets": ["mb_manip"],
        "description": "Pick-Classic: objaverse objects, classic mujoco renderer",
        "path": "molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
        "use_filament": False,
    },
    "pick": {
        "sets": ["mb_manip"],
        "description": "Pick-Filament: objaverse objects, filament renderer",
        "path": "molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
    },
    "pick_rand": {
        "sets": ["mb_manip"],
        "description": "Pick-RandCam: objaverse objects, filament, randomized camera",
        "path": "molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
        "camera_names": ["randomized_zed2_analogue_1", "wrist_camera"],
    },
    "pnp": {
        "sets": ["mb_manip"],
        "description": "Pick & Place: objaverse objects, filament renderer",
        "path": "molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark",
    },
    "pnp_color": {
        "sets": ["mb_manip"],
        "description": "Pick & Place Color: color-based disambiguation, filament renderer",
        "path": "molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark",
    },
    "pnp_next_to": {
        "sets": ["mb_manip"],
        "description": "Pick & Place Next-To: spatial relation placement, filament renderer",
        "path": "molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark",
    },
}

BENCHMARKS = {name: {**DEFAULT_BENCH_PARAMS, **cfg} for name, cfg in BENCHMARKS.items()}


def get_all_sets() -> list[str]:
    """Return sorted list of unique set names."""
    return sorted({s for cfg in BENCHMARKS.values() for s in cfg["sets"]})


def read_mujoco_spec(use_filament: bool) -> str:
    """Read the mujoco package spec from pyproject.toml extras."""
    import tomllib
    repo_root = Path(__file__).resolve().parent.parent.parent
    with open(repo_root / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    extra_name = "mujoco-filament" if use_filament else "mujoco"
    deps = pyproject["project"]["optional-dependencies"][extra_name]
    return deps[0]


def ensure_mujoco_variant(use_filament: bool, use_uv: bool) -> None:
    """Install the correct mujoco variant if it's not already present."""
    installer = ["uv", "pip"] if use_uv else [sys.executable, "-m", "pip"]
    target_pkg = "mujoco-filament" if use_filament else "mujoco"
    conflicting_pkg = "mujoco" if use_filament else "mujoco-filament"

    check = subprocess.run(
        [*installer, "show", target_pkg], capture_output=True, text=True
    )
    if check.returncode == 0:
        return

    pkg_spec = read_mujoco_spec(use_filament)
    print(f"Installing {pkg_spec} (removing {conflicting_pkg} if present)...")
    subprocess.run(
        [*installer, "uninstall", conflicting_pkg, "-y"],
        capture_output=True,
    )
    subprocess.check_call([*installer, "install", pkg_spec])


def print_benchmark_list():
    """Print all benchmarks grouped by set, with status indicators."""
    all_sets = get_all_sets()
    for s in all_sets:
        print(f"\n{s}")
        print(f"{'─' * 100}")
        for name, cfg in BENCHMARKS.items():
            if s not in cfg["sets"]:
                continue
            print(f"{name:15s}  {cfg['description']}")

    print(f"\nSets: {', '.join(all_sets)}\n")


def run_benchmark(name: str, bench_cfg: dict, args) -> None:
    """Validate, build command, and run a single benchmark."""
    if bench_cfg["path"] is None:
        raise NotImplementedError(
            f"Benchmark '{name}' does not have an installed benchmark path yet."
        )

    use_filament = bench_cfg["use_filament"]
    if use_filament and sys.platform == "darwin":
        raise NotImplementedError(
            f"Benchmark '{name}' requires the filament renderer, "
            f"which is not available on macOS."
        )

    ensure_mujoco_variant(use_filament, args.use_uv)

    benchmark_dir = ASSETS_DIR / "benchmarks" / bench_cfg["path"]
    if not benchmark_dir.exists():
        print(
            f"Error: benchmark not found at {benchmark_dir}\n"
            f"Have you downloaded the benchmark assets?",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent.parent
    eval_main = repo_root / "molmo_spaces" / "evaluation" / "eval_main.py"

    command = [
        sys.executable,
        str(eval_main),
        args.eval_config,
        "--benchmark_dir", str(benchmark_dir),
        "--num_workers", str(args.num_workers),
    ]

    if args.checkpoint_path is not None:
        command.extend(["--checkpoint_path", args.checkpoint_path])

    if use_filament:
        command.append("--use-filament")

    if args.environment_light_intensity is not None:
        command.extend(["--environment-light-intensity", str(args.environment_light_intensity)])

    if args.wandb_project is not None:
        command.extend(["--wandb_project", args.wandb_project])
    else:
        command.append("--no_wandb")

    if args.max_episodes is not None:
        command.extend(["--max_episodes", str(args.max_episodes)])

    if args.output_dir is not None:
        command.extend(["--output_dir", args.output_dir])

    camera_names = bench_cfg["camera_names"]
    if camera_names is not None:
        command.extend(["--camera_names", *camera_names])

    if bench_cfg["use_eval_cameras"]:
        command.append("--use_eval_cameras")
    if bench_cfg["camera_rand_level"] is not None:
        command.extend(["--camera_rand_level", str(bench_cfg["camera_rand_level"])])

    print(f"Benchmark:   {name} — {bench_cfg['description']}")
    print(f"Renderer:    {'filament' if use_filament else 'opengl (classic mujoco)'}")
    if camera_names is not None:
        print(f"Cameras:     {camera_names}")
    if bench_cfg["use_eval_cameras"]:
        print(f"Camera rand: level={bench_cfg['camera_rand_level']}")
    print(f"Eval config: {args.eval_config}")
    print(f"Command:     {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation of '{name}' failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    all_sets = get_all_sets()

    parser = argparse.ArgumentParser(
        description="Run benchmark evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "benchmark",
        type=str,
        nargs="?",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to evaluate.",
    )
    parser.add_argument(
        "--set",
        type=str,
        choices=all_sets,
        default=None,
        help="Run all ready benchmarks in a set (e.g. mb_manip).",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default=None,
        help="Eval config as module:ClassName (e.g. molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available benchmarks and exit.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint. Overrides the checkpoint in the eval config.",
    )
    parser.add_argument(
        "--use_uv",
        action="store_true",
        help="Use uv instead of pip for package management.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name. If not set, wandb is disabled.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--environment-light-intensity",
        type=float,
        default=None,
        help="Filament environment light intensity.",
    )
    args = parser.parse_args()

    if args.list:
        print_benchmark_list()
        return

    if args.benchmark and args.set:
        parser.error("specify either a single benchmark or --set, not both")

    if args.benchmark is None and args.set is None:
        parser.error("benchmark or --set is required (or use --list)")

    if args.eval_config is None:
        parser.error("--eval_config is required (e.g. molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig)")

    if args.benchmark:
        run_benchmark(args.benchmark, BENCHMARKS[args.benchmark], args)
    else:
        targets = [
            (name, cfg) for name, cfg in BENCHMARKS.items()
            if args.set in cfg["sets"] and cfg["path"] is not None
        ]
        if not targets:
            print(f"No ready benchmarks found in set '{args.set}'.", file=sys.stderr)
            sys.exit(1)

        skipped = [
            name for name, cfg in BENCHMARKS.items()
            if args.set in cfg["sets"] and cfg["path"] is None
        ]
        if skipped:
            print(f"Skipping not-ready benchmarks: {', '.join(skipped)}\n")

        for name, cfg in targets:
            print(f"\n{'=' * 70}")
            run_benchmark(name, cfg, args)


if __name__ == "__main__":
    main()
