#!/usr/bin/env python3
"""List pick-benchmark episodes (idx, house_index, iTHOR scene USD path). No Isaac Sim / isaaclab import."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_BUNDLED = ("benchmark_ithor_pick_hard_simple", "benchmark_ithor_thor_only_10")


def _default_benchmark_dir() -> Path:
    env = os.environ.get("MOLMO_PICK_BENCHMARK_DIR")
    if env:
        return Path(env).resolve()
    for name in _BUNDLED:
        p = REPO_ROOT / "examples" / name
        if p.is_dir() and (p / "benchmark.json").is_file():
            return p.resolve()
    return Path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=None,
        help=f"Directory with benchmark.json. Default: bundled simple bench or MOLMO_PICK_BENCHMARK_DIR.",
    )
    parser.add_argument(
        "--assets_root",
        type=Path,
        default=None,
        help="Used with bundled benchmarks to default scenes_root (same as Arena runner).",
    )
    parser.add_argument(
        "--scenes_root",
        type=Path,
        default=None,
        help="Override scene USD search root (else MOLMO_SCENES_ROOT or assets_root for bundled bench).",
    )
    args = parser.parse_args()

    bench = args.benchmark_dir
    if bench is None:
        bench = _default_benchmark_dir()
    bench = Path(bench)
    if not bench.is_dir() and not bench.is_absolute():
        bench = REPO_ROOT / bench
    if not bench.is_dir():
        print(f"Benchmark directory not found: {bench}", file=sys.stderr)
        return 1
    bf = bench / "benchmark.json"
    if not bf.is_file():
        print(f"No benchmark.json in {bench}", file=sys.stderr)
        return 1
    with open(bf) as f:
        episodes = json.load(f)
    if not isinstance(episodes, list) or not episodes:
        print("benchmark.json is empty or not a list.", file=sys.stderr)
        return 1

    scenes_root: Path | None = None
    if args.scenes_root is not None:
        scenes_root = Path(args.scenes_root).resolve()
    elif os.environ.get("MOLMO_SCENES_ROOT"):
        scenes_root = Path(os.environ["MOLMO_SCENES_ROOT"]).resolve()
    elif bench.name in _BUNDLED:
        root = args.assets_root or os.environ.get("MOLMO_ISAAC_ASSETS_ROOT")
        if root:
            scenes_root = Path(root).resolve()

    from molmo_spaces_isaac.arena.episode_to_arena import resolve_episode_scene_usd_path

    print(f"benchmark: {bench}", flush=True)
    print(f"scenes_root: {scenes_root}", flush=True)
    print(f"{'idx':>4}  {'house':>6}  {'usd':^5}  scene path", flush=True)
    for i, ep in enumerate(episodes):
        hi = ep.get("house_index", "?")
        usd = resolve_episode_scene_usd_path(ep, scenes_root) if scenes_root else None
        ok = bool(usd and Path(usd).is_file())
        pstr = str(usd) if usd else ""
        print(f"{i:4}  {hi!s:>6}  {'yes' if ok else 'no':^5}  {pstr}", flush=True)
    print(
        "\nRun another kitchen (Isaac): same command as usual plus  --episode_idx <idx>",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
