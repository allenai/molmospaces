#!/usr/bin/env python3
"""Build a THOR-only iTHOR pick benchmark for Isaac Lab Arena from a published FrankaPickHardBench JSON.

For each distinct house_index (ithor, val split), takes the first episode as a template and copies
robot_base_pose and robot.init_qpos. Replaces the pickup with Bowl_27 and sets world poses from a
fixed offset in the robot frame (default ~0.55 m forward, ~0.42 m up; clears typical Arena Franka
pedestal / counter height better than 0.4/0.35), rotated/translated by the
template robot_base_pose.

See examples/benchmark_ithor_pick_hard_simple/README.md."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # molmo_spaces_isaac
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molmo_spaces_isaac.arena.episode_to_arena import _quat_mul_wxyz, _quat_rotate_vector_wxyz

PICKUP_NAME = "Bowl_27"
THOR_BOWL_PATH = "objects/thor/Bowl_27.xml"
# Slightly past the arm + above typical counter so the bowl is visible in Arena (Isaac Franka base is bulky).
DEFAULT_ROBOT_FRAME_OFFSET_XYZ = (0.55, 0.0, 0.42)


def _default_source_benchmark() -> Path:
    env = os.environ.get("MOLMO_PICK_BENCHMARK_SOURCE") or os.environ.get("MOLMO_PICK_BENCHMARK_DIR")
    if env:
        p = Path(env).resolve()
        if p.is_file():
            return p
        if (p / "benchmark.json").is_file():
            return p / "benchmark.json"
    return Path(
        "/home/zryan/molmospaces_bench/mujoco/benchmarks/molmospaces-bench-v1/20260210/ithor/"
        "FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark/benchmark.json"
    ).resolve()


def _bowl_world_pose_from_robot_base(robot_base_pose_7: list[float], offset_xyz: tuple[float, float, float]) -> list[float]:
    """World pose [x,y,z,qw,qx,qy,qz] for bowl: offset in robot frame, identity orientation in robot frame."""
    if len(robot_base_pose_7) < 7:
        robot_base_pose_7 = list(robot_base_pose_7) + [0.0] * 7
        robot_base_pose_7 = robot_base_pose_7[:7]
    tx, ty, tz = robot_base_pose_7[0], robot_base_pose_7[1], robot_base_pose_7[2]
    q_base = (robot_base_pose_7[3], robot_base_pose_7[4], robot_base_pose_7[5], robot_base_pose_7[6])
    dx, dy, dz = offset_xyz
    rx, ry, rz = _quat_rotate_vector_wxyz(q_base, (dx, dy, dz))
    # Bowl orientation in robot frame = identity -> world = q_base * identity
    q_world = _quat_mul_wxyz(q_base, (1.0, 0.0, 0.0, 0.0))
    return [tx + rx, ty + ry, tz + rz, q_world[0], q_world[1], q_world[2], q_world[3]]


def _build_episode_from_template(template: dict, offset_xyz: tuple[float, float, float]) -> dict:
    task_in = template.get("task") or {}
    rbp = list(task_in.get("robot_base_pose") or [0.0] * 7)[:7]
    if len(rbp) < 7:
        rbp = (rbp + [0.0] * 7)[:7]
    bowl_pose = _bowl_world_pose_from_robot_base(rbp, offset_xyz)

    robot = copy.deepcopy(template.get("robot") or {})
    if not robot.get("robot_name"):
        robot["robot_name"] = "franka_droid"

    task_out = {
        "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
        "task_type": "pick",
        "pickup_obj_name": PICKUP_NAME,
        "pickup_obj_start_pose": bowl_pose,
        "robot_base_pose": rbp,
        "succ_pos_threshold": float(task_in.get("succ_pos_threshold", 0.01)),
    }

    return {
        "house_index": template["house_index"],
        "scene_dataset": template.get("scene_dataset") or "ithor",
        "data_split": template.get("data_split") or "val",
        "robot": robot,
        "img_resolution": list(template.get("img_resolution") or [224, 224]),
        "cameras": [],
        "scene_modifications": {
            "added_objects": {PICKUP_NAME: THOR_BOWL_PATH},
            "object_poses": {PICKUP_NAME: bowl_pose},
            "removed_objects": [],
        },
        "task": task_out,
        "language": {
            "task_description": "Pick up the bowl.",
            "referral_expressions": (template.get("language") or {}).get("referral_expressions") or {},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to FrankaPickHardBench benchmark.json (or its parent dir). "
        "Default: MOLMO_PICK_BENCHMARK_SOURCE / MOLMO_PICK_BENCHMARK_DIR or a fixed dev path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "examples" / "benchmark_ithor_pick_hard_simple" / "benchmark.json",
        help="Output benchmark.json path.",
    )
    parser.add_argument(
        "--scene-dataset",
        type=str,
        default="ithor",
        help="Only use episodes with this scene_dataset (default: ithor).",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="val",
        help="Only use episodes with this data_split (default: val).",
    )
    parser.add_argument(
        "--max-houses",
        type=int,
        default=0,
        metavar="K",
        help="Emit at most K houses (0 = all distinct houses, in ascending house_index order).",
    )
    parser.add_argument(
        "--offset-forward",
        type=float,
        default=DEFAULT_ROBOT_FRAME_OFFSET_XYZ[0],
        help="Bowl offset along robot +X in robot frame (m).",
    )
    parser.add_argument(
        "--offset-lateral",
        type=float,
        default=DEFAULT_ROBOT_FRAME_OFFSET_XYZ[1],
        help="Bowl offset along robot +Y in robot frame (m).",
    )
    parser.add_argument(
        "--offset-up",
        type=float,
        default=DEFAULT_ROBOT_FRAME_OFFSET_XYZ[2],
        help="Bowl offset along robot +Z in robot frame (m).",
    )
    args = parser.parse_args()

    src = args.source
    if src is None:
        src = _default_source_benchmark()
    else:
        src = Path(src).resolve()
        if src.is_dir():
            src = src / "benchmark.json"

    if not src.is_file():
        print(f"Source benchmark not found: {src}", file=sys.stderr)
        return 1

    with open(src) as f:
        episodes = json.load(f)
    if not isinstance(episodes, list):
        print("Source benchmark must be a JSON array of episodes.", file=sys.stderr)
        return 1

    first_by_house: dict[int, dict] = {}
    for ep in episodes:
        if (ep.get("scene_dataset") or "").lower() != args.scene_dataset.lower():
            continue
        if (ep.get("data_split") or "").lower() != args.data_split.lower():
            continue
        h = ep.get("house_index")
        if h is None:
            continue
        try:
            hi = int(h)
        except (TypeError, ValueError):
            continue
        if hi not in first_by_house:
            task = ep.get("task") or {}
            rbp = task.get("robot_base_pose")
            if not rbp or len(rbp) < 7:
                continue
            if not (ep.get("robot") or {}).get("init_qpos"):
                continue
            first_by_house[hi] = ep

    if not first_by_house:
        print("No matching template episodes (check scene_dataset / data_split / robot fields).", file=sys.stderr)
        return 1

    ordered_houses = sorted(first_by_house.keys())
    if args.max_houses and args.max_houses > 0:
        ordered_houses = ordered_houses[: args.max_houses]

    offset = (args.offset_forward, args.offset_lateral, args.offset_up)
    out_eps = [_build_episode_from_template(first_by_house[h], offset) for h in ordered_houses]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_eps, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(out_eps)} episodes to {out_path} (houses {ordered_houses[:5]}{'...' if len(ordered_houses) > 5 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
