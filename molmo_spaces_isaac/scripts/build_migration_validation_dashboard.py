#!/usr/bin/env python3
"""Build a MolmoSpaces -> IsaacLab Arena migration validation dashboard.

This report intentionally treats policy evaluation as an optional overlay.  The
core migration checks are simulator-agnostic: converted spec consistency,
robot/object relative poses, reset-state diagnostics, and availability of
MuJoCo trajectory artifacts for replay.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("episodes", "results", "items", "specs", "rows"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _num(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _vec(values: Any, n: int | None = None) -> list[float] | None:
    if not isinstance(values, (list, tuple)):
        return None
    if n is not None and len(values) < n:
        return None
    try:
        out = [float(v) for v in values[: n or len(values)]]
    except (TypeError, ValueError):
        return None
    return out


def _norm(values: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def _quat_normalize(q: list[float]) -> list[float]:
    n = _norm(q)
    if n <= 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / n for v in q]


def _quat_conj(q: list[float]) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_mul(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    return _quat_mul(_quat_mul(q, [0.0, v[0], v[1], v[2]]), _quat_conj(q))[1:4]


def _pose_world_to_robot(pose_world: list[float], robot_pose_world: list[float]) -> list[float] | None:
    if len(pose_world) < 7 or len(robot_pose_world) < 7:
        return None
    robot_pos = robot_pose_world[:3]
    robot_q = _quat_normalize(robot_pose_world[3:7])
    obj_pos = pose_world[:3]
    obj_q = _quat_normalize(pose_world[3:7])
    inv_q = _quat_conj(robot_q)
    pos_robot = _quat_rotate(inv_q, [obj_pos[i] - robot_pos[i] for i in range(3)])
    q_robot = _quat_mul(inv_q, obj_q)
    return pos_robot + _quat_normalize(q_robot)


def _quat_angle_deg(q1: list[float] | None, q2: list[float] | None) -> float | None:
    if q1 is None or q2 is None:
        return None
    qa = _quat_normalize(q1)
    qb = _quat_normalize(q2)
    dot = abs(sum(qa[i] * qb[i] for i in range(4)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _yaw_deg_from_wxyz(q: list[float] | None) -> float | None:
    if q is None or len(q) < 4:
        return None
    w, x, y, z = _quat_normalize(q[:4])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def _stats(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "max": None, "min": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "max": max(clean),
        "min": min(clean),
    }


def _house_from_scene_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"FloorPlan(\d+)_physics", path)
    return int(match.group(1)) if match else None


def _resolve_scene_path(path: str | None, scene_roots: list[Path]) -> tuple[str | None, bool]:
    if not path:
        return None, False
    original = Path(path)
    if original.is_file():
        return str(original), True
    match = re.search(r"(ithor|procthor|holodeck).*(FloorPlan\d+_physics)/scene\.usd[ac]$", path)
    if match:
        dataset, floorplan = match.group(1), match.group(2)
        for root in scene_roots:
            candidates = [
                root / dataset / floorplan / "scene.usda",
                root / dataset / floorplan / "scene.usdc",
                root / dataset / "20260121" / floorplan / "scene.usda",
                root / dataset / "20260121" / floorplan / "scene.usdc",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate), True
    return str(original), False


def _load_task_descriptions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("by_pickup_obj_name"), dict):
        return {str(k): v for k, v in data["by_pickup_obj_name"].items() if isinstance(v, dict)}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _load_arena_results(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    rows = _as_rows(_load_json(path))
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = row.get("idx")
        if idx is None:
            continue
        out[int(idx)] = row
    return out


def _load_reset_summaries(patterns: list[str]) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    for pattern in patterns:
        for raw_path in glob.glob(pattern):
            path = Path(raw_path)
            try:
                data = _load_json(path)
            except Exception:
                continue
            idx = data.get("episode_idx")
            if idx is None:
                match = re.search(r"ep(\d+)", str(path))
                idx = int(match.group(1)) if match else None
            if idx is None:
                continue
            record = {"path": str(path), "mtime": path.stat().st_mtime, "data": data}
            prev = summaries.get(int(idx))
            if prev is None or float(record["mtime"]) > float(prev["mtime"]):
                summaries[int(idx)] = record
    return summaries


def _object_from_arena_spec(arena_spec: dict[str, Any], pickup_name: str | None) -> dict[str, Any] | None:
    objects = arena_spec.get("objects")
    if not isinstance(objects, list):
        return None
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if pickup_name is None or obj.get("name") == pickup_name:
            return obj
    return objects[0] if objects and isinstance(objects[0], dict) else None


def _joint_error(expected: dict[str, Any], actual_names: list[Any], actual_pos: list[Any]) -> float | None:
    if not expected or not actual_names or not actual_pos:
        return None
    actual = {str(name): _num(value) for name, value in zip(actual_names, actual_pos)}
    errors: list[float] = []
    for key, value in expected.items():
        expected_value = _num(value)
        if expected_value is None:
            continue
        if key == "panda_finger_joint.*":
            actual_value = actual.get("finger_joint")
        else:
            actual_value = actual.get(key)
        if actual_value is not None:
            errors.append(abs(actual_value - expected_value))
    return max(errors) if errors else None


def _reset_metrics(summary_record: dict[str, Any] | None) -> dict[str, Any]:
    if summary_record is None:
        return {}
    data = summary_record["data"]
    snapshots = data.get("snapshots") or []
    reset = next((s for s in snapshots if s.get("label") == "reset"), snapshots[0] if snapshots else {})
    expected_pose = _vec(data.get("expected_arena_object_pose"), 7)
    actual_pos = _vec(reset.get("pickup_root_pos_w"), 3)
    actual_quat = _vec(reset.get("pickup_root_quat_w"), 4)
    expected_pos = expected_pose[:3] if expected_pose else None
    expected_quat = expected_pose[3:7] if expected_pose else None
    object_pos_error = None
    if expected_pos is not None and actual_pos is not None:
        object_pos_error = _norm([actual_pos[i] - expected_pos[i] for i in range(3)])
    object_quat_error = _quat_angle_deg(expected_quat, actual_quat)
    joint_error = _joint_error(
        data.get("episode_robot_init_joint_pos") or {},
        reset.get("robot_joint_names") or [],
        reset.get("robot_joint_pos") or [],
    )
    eef_pos = _vec(reset.get("policy_eef_pos"), 3)
    pickup_to_eef = None
    if actual_pos is not None and eef_pos is not None:
        pickup_to_eef = _norm([actual_pos[i] - eef_pos[i] for i in range(3)])
    images = data.get("images") or {}
    reset_images = images.get("reset") if isinstance(images, dict) else {}
    return {
        "reset_summary_path": summary_record["path"],
        "reset_object_pos_error_m": object_pos_error,
        "reset_object_quat_error_deg": object_quat_error,
        "reset_joint_max_abs_error_rad": joint_error,
        "reset_pickup_to_eef_m": pickup_to_eef,
        "reset_camera_image_count": len(reset_images) if isinstance(reset_images, dict) else 0,
        "reset_external_image": (reset_images or {}).get("external_camera_rgb") if isinstance(reset_images, dict) else None,
        "reset_wrist_image": (reset_images or {}).get("wrist_camera_rgb") if isinstance(reset_images, dict) else None,
    }


def _h5_info(task_info: dict[str, Any]) -> dict[str, Any]:
    source = task_info.get("source")
    house = task_info.get("house_index")
    traj = task_info.get("traj")
    if source is None or house is None:
        return {"mujoco_h5_path": None, "mujoco_h5_exists": False, "mujoco_traj": traj}
    h5_path = Path(str(source)) / f"house_{int(house)}" / "trajectories_batch_1_of_1.h5"
    return {
        "mujoco_h5_path": str(h5_path),
        "mujoco_h5_exists": h5_path.is_file(),
        "mujoco_traj": traj,
    }


def _replay_command(row: dict[str, Any], manifest_path: Path) -> str:
    h5_path = row.get("mujoco_h5_path")
    traj = row.get("mujoco_traj")
    if not h5_path or not traj:
        return ""
    return (
        "python molmo_spaces_isaac/scripts/run_arena_benchmark_episode.py "
        f"--arena_spec_manifest {manifest_path} --episode_idx {row['idx']} "
        "--policy_type h5_replay "
        f"--replay_h5 {h5_path} --replay_traj {traj} "
        "--joint_pos_policy --steps 1500 --replay_action_repeat 3 "
        "--record_video_camera_keys external_camera_rgb,wrist_camera_rgb"
    )


def build_episode_rows(
    *,
    manifest_path: Path,
    task_descriptions: dict[str, dict[str, Any]],
    arena_results: dict[int, dict[str, Any]],
    reset_summaries: dict[int, dict[str, Any]],
    scene_roots: list[Path],
) -> list[dict[str, Any]]:
    data = _load_json(manifest_path)
    rows = _as_rows(data)
    out: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row.get("idx", len(out)))
        arena_spec = row.get("arena_spec") if isinstance(row.get("arena_spec"), dict) else {}
        pickup_name = str(row.get("pickup_obj_name") or arena_spec.get("pickup_name") or "")
        pickup_obj = _object_from_arena_spec(arena_spec, pickup_name) or {}
        pose_world = _vec(pickup_obj.get("pose_7_world"), 7)
        robot_pose = _vec(arena_spec.get("robot_base_pose"), 7)
        pose_robot = _vec(row.get("pickup_pose_robot_frame"), 7)
        computed_pose_robot = _pose_world_to_robot(pose_world, robot_pose) if pose_world and robot_pose else None
        rel_pos_err = None
        rel_quat_err = None
        if pose_robot and computed_pose_robot:
            rel_pos_err = _norm([computed_pose_robot[i] - pose_robot[i] for i in range(3)])
            rel_quat_err = _quat_angle_deg(computed_pose_robot[3:7], pose_robot[3:7])
        scene_path = str(arena_spec.get("scene_usd_path") or "")
        resolved_scene_path, scene_exists = _resolve_scene_path(scene_path, scene_roots)
        scene_house = _house_from_scene_path(scene_path)
        task_info = task_descriptions.get(pickup_name, {})
        h5 = _h5_info(task_info)
        result = arena_results.get(idx, {})
        reset = _reset_metrics(reset_summaries.get(idx))
        out_row: dict[str, Any] = {
            "idx": idx,
            "status": row.get("status"),
            "house_index": row.get("house_index"),
            "scene_dataset": row.get("scene_dataset"),
            "task_type": arena_spec.get("task_type"),
            "pickup_obj_name": pickup_name,
            "task_description": task_info.get("task_description"),
            "pickup_source": pickup_obj.get("source"),
            "pickup_asset_id": pickup_obj.get("asset_id"),
            "object_count_in_spec": len(arena_spec.get("objects") or []),
            "scene_usd_path": scene_path,
            "scene_usd_resolved_path": resolved_scene_path,
            "scene_usd_exists": scene_exists,
            "scene_path_house_index": scene_house,
            "scene_house_matches": scene_house == row.get("house_index"),
            "success_threshold_m": _num(arena_spec.get("succ_pos_threshold")),
            "robot_base_x": robot_pose[0] if robot_pose else None,
            "robot_base_y": robot_pose[1] if robot_pose else None,
            "robot_base_z": robot_pose[2] if robot_pose else None,
            "robot_base_yaw_deg": _yaw_deg_from_wxyz(robot_pose[3:7]) if robot_pose else None,
            "has_robot_init_joint_pos": bool(arena_spec.get("robot_init_joint_pos")),
            "robot_init_joint_count": len(arena_spec.get("robot_init_joint_pos") or {}),
            "pickup_robot_x": pose_robot[0] if pose_robot else None,
            "pickup_robot_y": pose_robot[1] if pose_robot else None,
            "pickup_robot_z": pose_robot[2] if pose_robot else None,
            "pickup_robot_xy_m": _norm(pose_robot[:2]) if pose_robot else None,
            "pickup_world_x": pose_world[0] if pose_world else None,
            "pickup_world_y": pose_world[1] if pose_world else None,
            "pickup_world_z": pose_world[2] if pose_world else None,
            "relative_pose_position_error_m": rel_pos_err,
            "relative_pose_quat_error_deg": rel_quat_err,
            "policy_success": result.get("success") if result else None,
            "policy_step_count": result.get("step_count") or result.get("steps"),
            **h5,
            **reset,
        }
        out_row["h5_replay_command"] = _replay_command(out_row, manifest_path)
        out_row["quality_flags"] = quality_flags(out_row)
        out_row["quality_status"] = "fail" if any(f.startswith("FAIL") for f in out_row["quality_flags"]) else (
            "warn" if out_row["quality_flags"] else "pass"
        )
        out.append(out_row)
    return out


def quality_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if row.get("status") != "ready":
        flags.append("FAIL:not_ready")
    if row.get("task_type") != "pick":
        flags.append("FAIL:not_pick_task")
    if not row.get("scene_house_matches"):
        flags.append("FAIL:scene_house_mismatch")
    if not row.get("has_robot_init_joint_pos"):
        flags.append("FAIL:missing_robot_init")
    if row.get("relative_pose_position_error_m") is not None and row["relative_pose_position_error_m"] > 1e-5:
        flags.append("FAIL:relative_pose_position")
    if row.get("relative_pose_quat_error_deg") is not None and row["relative_pose_quat_error_deg"] > 1e-3:
        flags.append("FAIL:relative_pose_rotation")
    if not row.get("scene_usd_exists"):
        flags.append("WARN:scene_usd_missing_here")
    if not row.get("mujoco_h5_exists"):
        flags.append("WARN:mujoco_h5_missing")
    if row.get("reset_summary_path"):
        if row.get("reset_object_pos_error_m") is not None and row["reset_object_pos_error_m"] > 0.002:
            flags.append("FAIL:reset_object_pos")
        if row.get("reset_joint_max_abs_error_rad") is not None and row["reset_joint_max_abs_error_rad"] > 1e-4:
            flags.append("FAIL:reset_joint_pos")
    else:
        flags.append("WARN:no_reset_summary")
    return flags


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    policy_known = [r for r in rows if r.get("policy_success") is not None]
    return {
        "total_episodes": total,
        "ready_episodes": sum(1 for r in rows if r.get("status") == "ready"),
        "pass_count": sum(1 for r in rows if r.get("quality_status") == "pass"),
        "warn_count": sum(1 for r in rows if r.get("quality_status") == "warn"),
        "fail_count": sum(1 for r in rows if r.get("quality_status") == "fail"),
        "scene_usd_exists_count": sum(1 for r in rows if r.get("scene_usd_exists")),
        "scene_house_match_count": sum(1 for r in rows if r.get("scene_house_matches")),
        "robot_init_present_count": sum(1 for r in rows if r.get("has_robot_init_joint_pos")),
        "mujoco_h5_exists_count": sum(1 for r in rows if r.get("mujoco_h5_exists")),
        "reset_summary_count": sum(1 for r in rows if r.get("reset_summary_path")),
        "policy_result_count": len(policy_known),
        "policy_success_count": sum(1 for r in policy_known if r.get("policy_success")),
        "policy_success_rate": (
            sum(1 for r in policy_known if r.get("policy_success")) / len(policy_known) if policy_known else None
        ),
        "pickup_robot_xy_m": _stats([r["pickup_robot_xy_m"] for r in rows if r.get("pickup_robot_xy_m") is not None]),
        "pickup_robot_z_m": _stats([r["pickup_robot_z"] for r in rows if r.get("pickup_robot_z") is not None]),
        "relative_pose_position_error_m": _stats(
            [r["relative_pose_position_error_m"] for r in rows if r.get("relative_pose_position_error_m") is not None]
        ),
        "relative_pose_quat_error_deg": _stats(
            [r["relative_pose_quat_error_deg"] for r in rows if r.get("relative_pose_quat_error_deg") is not None]
        ),
        "reset_object_pos_error_m": _stats(
            [r["reset_object_pos_error_m"] for r in rows if r.get("reset_object_pos_error_m") is not None]
        ),
        "reset_joint_max_abs_error_rad": _stats(
            [r["reset_joint_max_abs_error_rad"] for r in rows if r.get("reset_joint_max_abs_error_rad") is not None]
        ),
    }


CSV_COLUMNS = [
    "idx",
    "quality_status",
    "quality_flags",
    "status",
    "house_index",
    "scene_dataset",
    "task_type",
    "pickup_obj_name",
    "task_description",
    "pickup_source",
    "scene_usd_exists",
    "scene_usd_resolved_path",
    "scene_house_matches",
    "success_threshold_m",
    "robot_base_x",
    "robot_base_y",
    "robot_base_z",
    "robot_base_yaw_deg",
    "has_robot_init_joint_pos",
    "robot_init_joint_count",
    "pickup_robot_x",
    "pickup_robot_y",
    "pickup_robot_z",
    "pickup_robot_xy_m",
    "pickup_world_x",
    "pickup_world_y",
    "pickup_world_z",
    "relative_pose_position_error_m",
    "relative_pose_quat_error_deg",
    "reset_object_pos_error_m",
    "reset_object_quat_error_deg",
    "reset_joint_max_abs_error_rad",
    "reset_pickup_to_eef_m",
    "mujoco_h5_exists",
    "mujoco_h5_path",
    "mujoco_traj",
    "policy_success",
    "policy_step_count",
    "reset_summary_path",
    "reset_external_image",
    "reset_wrist_image",
    "h5_replay_command",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["quality_flags"] = ";".join(row.get("quality_flags") or [])
            writer.writerow({k: flat.get(k) for k in CSV_COLUMNS})


def _pill(status: Any) -> str:
    cls = str(status or "na").lower()
    label = html.escape(str(status if status is not None else "n/a"))
    return f'<span class="pill {cls}">{label}</span>'


def _link(path: str | None, label: str) -> str:
    if not path:
        return ""
    return f'<a href="{html.escape(path)}">{html.escape(label)}</a>'


def _bar(value: float | None, max_value: float, width: int = 120) -> str:
    if value is None or max_value <= 0:
        return ""
    pct = max(0.0, min(1.0, float(value) / max_value))
    inner = int(width * pct)
    return f'<span class="bar" style="width:{width}px"><span style="width:{inner}px"></span></span>'


def write_html(path: Path, rows: list[dict[str, Any]], agg: dict[str, Any], title: str) -> None:
    max_xy = max([r.get("pickup_robot_xy_m") or 0.0 for r in rows] + [1.0])
    rel_err = agg["relative_pose_position_error_m"]
    reset_err = agg["reset_object_pos_error_m"]
    policy_rate = agg.get("policy_success_rate")
    cards = [
        ("Episodes", agg["total_episodes"], "converted specs"),
        ("Ready", agg["ready_episodes"], "manifest rows"),
        ("Reset Diag", agg["reset_summary_count"], "episodes with Arena reset summaries"),
        ("MuJoCo H5", agg["mujoco_h5_exists_count"], "episodes with replay HDF5 available"),
        ("Policy Overlay", f"{policy_rate:.1%}" if policy_rate is not None else "n/a", "not a migration score"),
        ("Quality", f"{agg['pass_count']} pass / {agg['warn_count']} warn / {agg['fail_count']} fail", "based on structural checks"),
    ]
    card_html = "\n".join(
        f'<section class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(str(value))}</div><div class="card-note">{html.escape(note)}</div></section>'
        for label, value, note in cards
    )
    metric_rows = [
        ("Target distance from robot XY", agg["pickup_robot_xy_m"], "m"),
        ("Target Z in robot frame", agg["pickup_robot_z_m"], "m"),
        ("Spec transform position error", rel_err, "m"),
        ("Spec transform rotation error", agg["relative_pose_quat_error_deg"], "deg"),
        ("Arena reset object position error", reset_err, "m"),
        ("Arena reset joint max error", agg["reset_joint_max_abs_error_rad"], "rad"),
    ]
    metric_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td><td>{stats.get('count', 0)}</td>"
        f"<td>{_fmt(stats.get('mean'))}</td><td>{_fmt(stats.get('median'))}</td>"
        f"<td>{_fmt(stats.get('max'))}</td><td>{html.escape(unit)}</td></tr>"
        for name, stats, unit in metric_rows
    )
    episode_html_parts: list[str] = []
    for row in rows:
        flags = ", ".join(row.get("quality_flags") or [])
        image_links = " ".join(
            x
            for x in (
                _link(row.get("reset_external_image"), "exo"),
                _link(row.get("reset_wrist_image"), "wrist"),
                _link(row.get("reset_summary_path"), "summary"),
            )
            if x
        )
        episode_html_parts.append(
            "<tr>"
            f"<td>{row['idx']}</td>"
            f"<td>{_pill(row.get('quality_status'))}</td>"
            f"<td>{html.escape(str(row.get('house_index') or ''))}</td>"
            f"<td class=\"obj\">{html.escape(row.get('pickup_obj_name') or '')}</td>"
            f"<td>{html.escape(row.get('pickup_source') or '')}</td>"
            f"<td>{_fmt(row.get('pickup_robot_xy_m'))} {_bar(row.get('pickup_robot_xy_m'), max_xy)}</td>"
            f"<td>{_fmt(row.get('pickup_robot_z'))}</td>"
            f"<td>{_fmt(row.get('relative_pose_position_error_m'), 6)}</td>"
            f"<td>{_fmt(row.get('reset_object_pos_error_m'), 6)}</td>"
            f"<td>{_fmt(row.get('reset_joint_max_abs_error_rad'), 6)}</td>"
            f"<td>{_fmt(row.get('mujoco_h5_exists'))}</td>"
            f"<td>{_fmt(row.get('policy_success'))}</td>"
            f"<td class=\"links\">{image_links}</td>"
            f"<td class=\"flags\">{html.escape(flags)}</td>"
            "</tr>"
        )
    episode_html = "\n".join(episode_html_parts)
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172026;
  --muted: #5d6872;
  --line: #d9e0e7;
  --bg: #f6f8fa;
  --panel: #ffffff;
  --pass: #19784a;
  --warn: #a16207;
  --fail: #b42318;
  --accent: #2563eb;
}}
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); background: var(--bg); }}
header {{ padding: 28px 32px 18px; background: var(--panel); border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 720; letter-spacing: 0; }}
p {{ margin: 0 0 10px; color: var(--muted); line-height: 1.45; max-width: 1100px; }}
main {{ padding: 24px 32px 40px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 22px; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
.card-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
.card-value {{ font-size: 24px; margin-top: 6px; font-weight: 720; }}
.card-note {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
section.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; background: #fbfcfd; position: sticky; top: 0; }}
tbody tr:hover {{ background: #f7fbff; }}
.table-wrap {{ max-height: 72vh; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
.pill {{ display: inline-flex; min-width: 42px; justify-content: center; border-radius: 999px; padding: 2px 8px; font-weight: 650; font-size: 12px; }}
.pill.pass {{ color: var(--pass); background: #eaf7ef; }}
.pill.warn {{ color: var(--warn); background: #fff7df; }}
.pill.fail {{ color: var(--fail); background: #fff0ed; }}
.obj {{ max-width: 330px; overflow-wrap: anywhere; }}
.flags {{ max-width: 270px; color: var(--muted); }}
.links a {{ margin-right: 8px; }}
.bar {{ display: inline-block; height: 7px; margin-left: 8px; border-radius: 99px; background: #e7edf3; vertical-align: middle; overflow: hidden; }}
.bar span {{ display: block; height: 100%; border-radius: 99px; background: var(--accent); }}
code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
input {{ width: min(520px, 100%); padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; margin: 0 0 10px; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <p>This dashboard validates automated scene migration without using policy success as the primary score. Policy eval is included only as an overlay because it entangles migration fidelity with rendering, control timing, physics, and policy robustness.</p>
  <p>Recommended interpretation: fix structural/reset failures first, then use MuJoCo HDF5 replay on selected episodes, and only then look at OpenPI transfer performance.</p>
</header>
<main>
  <div class="cards">{card_html}</div>
  <section class="panel">
    <h2>Aggregate Metrics</h2>
    <table>
      <thead><tr><th>Metric</th><th>N</th><th>Mean</th><th>Median</th><th>Max</th><th>Unit</th></tr></thead>
      <tbody>{metric_html}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>Validation Ladder</h2>
    <p><b>Static/spec parity:</b> episode identity, scene/house mapping, robot init qpos, target pose transformed into robot frame, and success threshold.</p>
    <p><b>Reset-state parity:</b> optional Arena diagnostic summaries compare actual reset object pose and robot joints to the converted spec.</p>
    <p><b>Replay parity:</b> rows with MuJoCo HDF5 available include enough metadata to run an HDF5 replay in Arena. Replay outcome should be compared via EE/object trajectories and contact events, not just success.</p>
  </section>
  <section class="panel">
    <h2>Per Episode</h2>
    <input id="filter" placeholder="Filter by episode, object, house, flag, status..." oninput="filterRows()">
    <div class="table-wrap">
      <table id="episodes">
        <thead><tr><th>Idx</th><th>Status</th><th>House</th><th>Object</th><th>Source</th><th>Target XY</th><th>Target Z</th><th>Spec Pos Err</th><th>Reset Obj Err</th><th>Reset Joint Err</th><th>H5</th><th>Policy</th><th>Links</th><th>Flags</th></tr></thead>
        <tbody>{episode_html}</tbody>
      </table>
    </div>
  </section>
</main>
<script>
function filterRows() {{
  const q = document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('#episodes tbody tr').forEach((tr) => {{
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena_spec_manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--task_description_map", type=Path, default=None)
    parser.add_argument("--arena_results_json", type=Path, default=None)
    parser.add_argument(
        "--scenes_root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional converted USD scenes root used to resolve scene paths from another machine, "
            "for example /home/horde/.molmospaces/usd/scenes. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--arena_reset_summary_glob",
        action="append",
        default=[],
        help="Glob for diagnose_arena_episode.py summary.json files. Can be passed multiple times.",
    )
    parser.add_argument("--title", default="MolmoSpaces -> IsaacLab Arena Migration Validation")
    args = parser.parse_args()

    manifest_path = args.arena_spec_manifest.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_descriptions = _load_task_descriptions(args.task_description_map.expanduser().resolve() if args.task_description_map else None)
    arena_results = _load_arena_results(args.arena_results_json.expanduser().resolve() if args.arena_results_json else None)
    reset_summaries = _load_reset_summaries(args.arena_reset_summary_glob)
    scene_roots = [p.expanduser().resolve() for p in args.scenes_root]
    rows = build_episode_rows(
        manifest_path=manifest_path,
        task_descriptions=task_descriptions,
        arena_results=arena_results,
        reset_summaries=reset_summaries,
        scene_roots=scene_roots,
    )
    agg = aggregate(rows)
    summary = {
        "inputs": {
            "arena_spec_manifest": str(manifest_path),
            "task_description_map": str(args.task_description_map) if args.task_description_map else None,
            "arena_results_json": str(args.arena_results_json) if args.arena_results_json else None,
            "arena_reset_summary_glob": args.arena_reset_summary_glob,
            "scenes_root": [str(p) for p in scene_roots],
        },
        "aggregate": agg,
        "episodes": rows,
    }
    summary_path = out_dir / "migration_validation_summary.json"
    csv_path = out_dir / "migration_validation_episodes.csv"
    html_path = out_dir / "index.html"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(csv_path, rows)
    write_html(html_path, rows, agg, args.title)
    print(f"Wrote {html_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(
        "Summary: "
        f"{agg['total_episodes']} episodes, "
        f"{agg['pass_count']} pass, {agg['warn_count']} warn, {agg['fail_count']} fail, "
        f"{agg['reset_summary_count']} reset summaries, {agg['mujoco_h5_exists_count']} MuJoCo H5 files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
