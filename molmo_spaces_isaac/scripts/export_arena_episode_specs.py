#!/usr/bin/env python3
"""Export MolmoSpaces benchmark episodes as ArenaEpisodeSpec JSON.

This is an offline migration helper: it does not import Isaac Sim. The output is
intended to be a durable manifest that Arena runners can consume or diff while
we iterate on pose conversion.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _default_benchmark_dir() -> Path:
    env = os.environ.get("MOLMO_PICK_BENCHMARK_DIR")
    if env:
        return Path(env).resolve()
    bundled = REPO_ROOT / "examples" / "benchmark_ithor_pick_hard_simple"
    return bundled.resolve() if bundled.is_dir() else Path()


def _load_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    try:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

        episodes = load_all_episodes(benchmark_dir)
        return [ep.model_dump() if hasattr(ep, "model_dump") else dict(ep) for ep in episodes]
    except Exception:
        bench_file = benchmark_dir / "benchmark.json"
        with open(bench_file) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"{bench_file} is not a JSON list")
        return data


def _spec_to_jsonable(spec) -> dict[str, Any]:
    data = dataclasses.asdict(spec)
    if data.get("scene_usd_path") is not None:
        data["scene_usd_path"] = str(data["scene_usd_path"])
    data["objects"] = [
        {
            "name": name,
            "asset_id": asset_id,
            "pose_7_world": pose,
            "source": source,
        }
        for name, asset_id, pose, source in data.get("objects", [])
    ]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, default=None)
    parser.add_argument("--assets_root", type=Path, default=None)
    parser.add_argument("--scenes_root", type=Path, default=None)
    parser.add_argument("--allow-objaverse", action="store_true", dest="allow_objaverse")
    parser.add_argument("--limit", type=int, default=0, help="Only export first N episodes (0 = all).")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON manifest path.")
    args = parser.parse_args()

    from molmo_spaces_isaac.arena.episode_to_arena import (
        _pose_7_world_to_robot_frame,
        episode_dict_to_arena_spec,
    )

    benchmark_dir = args.benchmark_dir or _default_benchmark_dir()
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.is_dir() and not benchmark_dir.is_absolute():
        benchmark_dir = REPO_ROOT / benchmark_dir
    if not benchmark_dir.is_dir() or not (benchmark_dir / "benchmark.json").is_file():
        raise SystemExit(f"Benchmark directory with benchmark.json not found: {benchmark_dir}")

    scenes_root = args.scenes_root
    if scenes_root is None and os.environ.get("MOLMO_SCENES_ROOT"):
        scenes_root = Path(os.environ["MOLMO_SCENES_ROOT"])
    if scenes_root is None:
        root = args.assets_root or os.environ.get("MOLMO_ISAAC_ASSETS_ROOT")
        if root:
            scenes_root = Path(root)
    scenes_root = scenes_root.resolve() if scenes_root else None

    episodes = _load_episodes(benchmark_dir)
    if args.limit and args.limit > 0:
        episodes = episodes[: args.limit]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, ep in enumerate(episodes):
        spec = episode_dict_to_arena_spec(
            ep,
            background_key="kitchen",
            require_thor_only=not args.allow_objaverse,
            scenes_root=scenes_root,
        )
        task = ep.get("task") or {}
        if spec is None:
            failure = {
                "idx": idx,
                "house_index": ep.get("house_index"),
                "scene_dataset": ep.get("scene_dataset"),
                "pickup_obj_name": task.get("pickup_obj_name"),
                "reason": "episode_to_arena_spec_failed",
            }
            failures.append(failure)
            rows.append({"idx": idx, "status": "failed", **failure})
            continue

        spec_json = _spec_to_jsonable(spec)
        pickup_pose_robot_frame = None
        if spec.objects:
            pickup_pose_robot_frame = _pose_7_world_to_robot_frame(spec.objects[0][2], spec.robot_base_pose)
        rows.append(
            {
                "idx": idx,
                "status": "ready",
                "house_index": ep.get("house_index"),
                "scene_dataset": ep.get("scene_dataset"),
                "pickup_obj_name": task.get("pickup_obj_name"),
                "pickup_pose_robot_frame": pickup_pose_robot_frame,
                "arena_spec": spec_json,
            }
        )

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_dir": str(benchmark_dir.resolve()),
        "scenes_root": str(scenes_root) if scenes_root else None,
        "allow_objaverse": bool(args.allow_objaverse),
        "counts": {
            "ready": sum(1 for row in rows if row["status"] == "ready"),
            "failed": len(failures),
            "total": len(rows),
        },
        "episodes": rows,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {out}")
    print("counts:", payload["counts"])
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
