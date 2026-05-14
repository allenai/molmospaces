#!/usr/bin/env python3
"""Offline preflight for migrating a MolmoSpaces benchmark to Isaac Lab Arena.

This does not import Isaac Sim. It checks episode schema support, scene USD resolution,
pickup asset source, robot init qpos, and the converted robot-frame pickup pose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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


def _task_type(ep: dict[str, Any]) -> str:
    task = ep.get("task") or {}
    task_type = (task.get("task_type") or "").strip().lower()
    if task_type:
        return task_type
    task_cls = task.get("task_cls") or ""
    if "PickTask" in task_cls and "PickAndPlace" not in task_cls:
        return "pick"
    if "PickAndPlace" in task_cls:
        return "pick-and-place"
    return task_cls or "unknown"


def _classify_episode(
    ep: dict[str, Any],
    scenes_root: Path | None,
    allow_objaverse: bool,
    pick_z_extra: float,
) -> tuple[str, str, dict[str, Any]]:
    from molmo_spaces_isaac.arena.episode_to_arena import (
        _pose_7_world_to_robot_frame,
        episode_dict_to_arena_spec,
        resolve_episode_scene_usd_path,
    )

    task = ep.get("task") or {}
    sm = ep.get("scene_modifications") or {}
    added = sm.get("added_objects") or {}
    object_poses = sm.get("object_poses") or {}
    pickup = task.get("pickup_obj_name")
    task_type = _task_type(ep)
    scene_usd = resolve_episode_scene_usd_path(ep, scenes_root) if scenes_root else None
    scene_ok = bool(scene_usd and Path(scene_usd).is_file())

    details: dict[str, Any] = {
        "house_index": ep.get("house_index"),
        "scene_dataset": ep.get("scene_dataset"),
        "task_type": task_type,
        "pickup_obj_name": pickup,
        "pickup_in_added_objects": bool(pickup in added) if pickup else False,
        "scene_usd_path": str(scene_usd) if scene_usd else None,
        "scene_usd_exists": scene_ok,
        "has_robot_init_qpos": bool(((ep.get("robot") or {}).get("init_qpos") or {}).get("arm")),
        "camera_count": len(ep.get("cameras") or task.get("cameras") or []),
    }

    if task_type != "pick":
        return "unsupported", "task_not_pick", details
    if not pickup:
        return "unsupported", "missing_pickup_obj_name", details
    if pickup in added:
        pickup_path = str(added.get(pickup) or "")
        details["pickup_asset_path"] = pickup_path
        if "objaverse" in pickup_path.lower() and not allow_objaverse:
            return "blocked", "objaverse_disabled", details
    elif pickup in object_poses or task.get("pickup_obj_start_pose"):
        details["pickup_asset_path"] = "<existing scene object>"
    else:
        return "unsupported", "pickup_pose_not_found", details

    spec = episode_dict_to_arena_spec(
        ep,
        require_thor_only=not allow_objaverse,
        background_key="kitchen",
        scenes_root=scenes_root,
    )
    if spec is None:
        return "unsupported", "episode_to_arena_spec_failed", details

    if spec.objects:
        if spec.objects[0][3] == "scene" and not scene_ok:
            details["pickup_source"] = "scene"
            details["pickup_asset_id"] = spec.objects[0][1]
            return "blocked", "scene_usd_missing_for_scene_pickup", details
        arena_pose = _pose_7_world_to_robot_frame(spec.objects[0][2], spec.robot_base_pose)
        if spec.objects[0][3] == "thor" and spec.objects[0][0] == spec.pickup_name and pick_z_extra:
            arena_pose = list(arena_pose)
            arena_pose[2] = float(arena_pose[2]) + pick_z_extra
        details["arena_pickup_xyz"] = [round(float(x), 5) for x in arena_pose[:3]]
        details["arena_pickup_xy_distance_m"] = round(math.hypot(float(arena_pose[0]), float(arena_pose[1])), 5)
        details["pickup_source"] = spec.objects[0][3]
        details["pickup_asset_id"] = spec.objects[0][1]
    details["robot_base_pose"] = spec.robot_base_pose
    details["robot_init_joint_pos"] = spec.robot_init_joint_pos
    return "ready", "ok", details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, default=None)
    parser.add_argument("--assets_root", type=Path, default=None)
    parser.add_argument("--scenes_root", type=Path, default=None)
    parser.add_argument("--allow-objaverse", action="store_true", dest="allow_objaverse")
    parser.add_argument(
        "--pick_z_extra",
        type=float,
        default=float((os.environ.get("MOLMO_ARENA_PICK_Z_EXTRA") or "0").strip() or "0"),
        help="Runtime Z lift applied to THOR pickup spawns; defaults to MOLMO_ARENA_PICK_Z_EXTRA.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only inspect first N episodes (0 = all).")
    parser.add_argument("--json_out", type=Path, default=None, help="Optional path for full JSON report.")
    args = parser.parse_args()

    bench = args.benchmark_dir or _default_benchmark_dir()
    bench = Path(bench)
    if not bench.is_dir() and not bench.is_absolute():
        bench = REPO_ROOT / bench
    if not bench.is_dir() or not (bench / "benchmark.json").is_file():
        raise SystemExit(f"Benchmark directory with benchmark.json not found: {bench}")

    scenes_root = args.scenes_root
    if scenes_root is None and os.environ.get("MOLMO_SCENES_ROOT"):
        scenes_root = Path(os.environ["MOLMO_SCENES_ROOT"])
    if scenes_root is None:
        root = args.assets_root or os.environ.get("MOLMO_ISAAC_ASSETS_ROOT")
        if root:
            scenes_root = Path(root)
    scenes_root = scenes_root.resolve() if scenes_root else None

    episodes = _load_episodes(bench)
    if args.limit and args.limit > 0:
        episodes = episodes[: args.limit]

    rows = []
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for idx, ep in enumerate(episodes):
        status, reason, details = _classify_episode(ep, scenes_root, args.allow_objaverse, args.pick_z_extra)
        counts[status] += 1
        reasons[reason] += 1
        row = {"idx": idx, "status": status, "reason": reason, **details}
        rows.append(row)

    print(f"benchmark: {bench}")
    print(f"episodes: {len(rows)}")
    print(f"scenes_root: {scenes_root}")
    print("status:", dict(counts))
    print("reasons:", dict(reasons))
    print()
    print(f"{'idx':>4}  {'status':<11}  {'reason':<36}  {'house':>5}  {'scene':^5}  {'pickup':<28}  {'arena xyz'}")
    for row in rows[:200]:
        xyz = row.get("arena_pickup_xyz")
        xyz_s = "" if xyz is None else ",".join(f"{float(v):.2f}" for v in xyz)
        scene_s = "yes" if row.get("scene_usd_exists") else "no"
        print(
            f"{row['idx']:4}  {row['status']:<11}  {row['reason']:<36}  "
            f"{str(row.get('house_index')):>5}  {scene_s:^5}  {str(row.get('pickup_obj_name')):<28.28}  {xyz_s}"
        )
    if len(rows) > 200:
        print(f"... {len(rows) - 200} more rows omitted from stdout; use --json_out for full details.")

    if args.json_out is not None:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "benchmark_dir": str(bench),
                    "scenes_root": str(scenes_root) if scenes_root else None,
                    "pick_z_extra": args.pick_z_extra,
                    "counts": dict(counts),
                    "reasons": dict(reasons),
                    "episodes": rows,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {out}")
    return 0 if counts.get("ready", 0) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
