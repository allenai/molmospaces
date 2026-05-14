#!/usr/bin/env python3
"""Diagnose one MolmoSpaces episode after conversion to Isaac Lab Arena.

Outputs a JSON pose summary, camera images, and a top-down plot for a single
ArenaEpisodeSpec. This is meant for debugging robot/object/camera placement
parity before running an expensive policy evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MOLMOSPACES_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

_arena_path = os.environ.get("ISAACLAB_ARENA_PATH")
if _arena_path:
    _arena_path = Path(_arena_path).resolve()
    if _arena_path.is_dir() and str(_arena_path) not in sys.path:
        sys.path.insert(0, str(_arena_path))

for p in (SRC_ROOT, REPO_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isaaclab.app import AppLauncher

for _i in range(1, len(sys.argv)):
    sys.argv[_i] = sys.argv[_i].lstrip()
while len(sys.argv) > 1 and sys.argv[1] == "--":
    sys.argv.pop(1)


def _default_pick_benchmark_dir() -> Path:
    env = os.environ.get("MOLMO_PICK_BENCHMARK_DIR")
    if env:
        return Path(env).resolve()
    return (REPO_ROOT / "examples" / "benchmark_ithor_pick_hard_simple").resolve()


parser = argparse.ArgumentParser(description="Save pose and camera diagnostics for one Arena-converted MolmoSpaces episode.")
parser.add_argument("--episode_json", type=Path, default=None)
parser.add_argument("--benchmark_dir", type=Path, default=None)
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--assets_root", type=Path, default=None)
parser.add_argument("--scenes_root", type=Path, default=None)
parser.add_argument("--background", type=str, default="kitchen")
parser.add_argument("--allow-objaverse", action="store_true", dest="allow_objaverse")
parser.add_argument("--embodiment", type=str, default=None)
parser.add_argument("--joint_pos_policy", action="store_true")
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--out_dir", type=Path, default=Path("diagnostics/arena_episode"))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=None)
parser.add_argument("--scene_extra_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
parser.add_argument("--align_scene_floor_z_zero", action="store_true")
parser.add_argument("--no_cameras", action="store_true", help="Disable camera sensors; pose diagnostics still run.")
parser.add_argument("--replay_h5", type=Path, default=None, help="Optional MolmoSpaces eval HDF5 for Arena FK parity probes.")
parser.add_argument("--replay_traj", type=str, default=os.environ.get("MOLMO_ARENA_REPLAY_TRAJ", "traj_0"))
parser.add_argument(
    "--fk_probe_indices",
    type=int,
    nargs="*",
    default=None,
    help="HDF5 row indices to write into Arena and compare against MuJoCo tcp_pose.",
)

if "--no_cameras" not in sys.argv and "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if getattr(args, "align_scene_floor_z_zero", False):
    os.environ["MOLMO_ARENA_SCENE_Z_ALIGN_WORLD_ZERO"] = "1"

simulation_app = AppLauncher(args).app


def _load_episode_from_benchmark(benchmark_dir: Path, episode_idx: int) -> dict[str, Any]:
    try:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

        episodes = load_all_episodes(Path(benchmark_dir))
        if not episodes:
            raise SystemExit(f"No episodes found in {benchmark_dir}")
        if episode_idx < 0 or episode_idx >= len(episodes):
            raise SystemExit(f"episode_idx {episode_idx} out of range [0, {len(episodes) - 1}]")
        ep = episodes[episode_idx]
        return ep.model_dump() if hasattr(ep, "model_dump") else dict(ep)
    except ImportError:
        bench_file = Path(benchmark_dir) / "benchmark.json"
        if not bench_file.is_file():
            raise SystemExit(f"No benchmark.json in {benchmark_dir}")
        with open(bench_file) as f:
            data = json.load(f)
        if episode_idx < 0 or episode_idx >= len(data):
            raise SystemExit(f"episode_idx {episode_idx} out of range [0, {len(data) - 1}]")
        return data[episode_idx]


def _resolve_asset_dirs(assets_root: Path | None):
    if assets_root is None:
        return None, None, None
    root = Path(assets_root).resolve()
    if not root.is_dir():
        return None, None, None
    from molmo_spaces_isaac.arena.thor_asset import THOR_DEFAULT_VERSION

    thor_assets_dir = None
    thor_metadata_path = None
    versioned = root / "objects" / "thor" / "thor" / THOR_DEFAULT_VERSION
    if versioned.is_dir():
        thor_assets_dir = versioned
        thor_metadata_path = versioned / "usd_assets_metadata.json"
    else:
        flat = root / "objects" / "thor"
        if flat.is_dir():
            thor_assets_dir = flat
            thor_metadata_path = flat / "usd_assets_metadata.json"
            if not thor_metadata_path.is_file():
                thor_metadata_path = None

    objaverse_assets_dir = root / "objects" / "objaverse"
    if not objaverse_assets_dir.is_dir():
        objaverse_assets_dir = root / "objaverse"
    if not objaverse_assets_dir.is_dir():
        objaverse_assets_dir = None
    return thor_assets_dir, thor_metadata_path, objaverse_assets_dir


def _selected_embodiment_key() -> str:
    explicit = args.embodiment or os.environ.get("MOLMO_ARENA_EMBODIMENT")
    if explicit:
        return str(explicit)
    if args.joint_pos_policy:
        return "droid_abs_joint_pos"
    return "franka"


def _tensor_to_list(x: Any, limit: int | None = None) -> list[float]:
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(x, "tolist"):
        x = x.tolist()
    while isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    if not isinstance(x, list):
        try:
            x = list(x)
        except TypeError:
            x = [x]
    out: list[float] = []
    for v in x:
        if isinstance(v, list):
            out.extend(float(y) for y in v)
        else:
            out.append(float(v))
    return out[:limit] if limit is not None else out


def _decode_h5_json_row(row) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(row)
    data = bytes(arr[arr != 0]).decode("utf-8")
    return json.loads(data) if data else {}


def _body_pos(robot, body_name: str):
    body_names = list(getattr(robot.data, "body_names", []) or [])
    body_pos = getattr(robot.data, "body_pos_w", None)
    if body_pos is None or body_name not in body_names:
        return None
    return body_pos[0, body_names.index(body_name), :]


def _frame_pos(scene, frame_name: str):
    try:
        ee_frame = scene["ee_frame"]
        names = list(getattr(ee_frame.data, "target_frame_names", []) or [])
        if frame_name not in names:
            return None
        return ee_frame.data.target_pos_w[0, names.index(frame_name), :]
    except Exception:
        return None


def _set_arena_robot_qpos_from_mujoco(robot, qpos: dict[str, Any]) -> None:
    import math
    import torch

    joint_names = list(getattr(robot.data, "joint_names", []) or [])
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    arm = list(qpos.get("arm") or [])
    for i, value in enumerate(arm[:7]):
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


def _run_fk_probes(env, spec, replay_h5: Path, replay_traj: str, indices: list[int]) -> list[dict[str, Any]]:
    import h5py
    import numpy as np
    import torch

    from molmo_spaces_isaac.arena.episode_to_arena import _pose_7_world_to_robot_frame

    scene = env.unwrapped.scene
    robot = scene["robot"]
    results: list[dict[str, Any]] = []
    with h5py.File(Path(replay_h5).expanduser().resolve(), "r") as f:
        root = f[replay_traj]
        n = int(root["obs/extra/tcp_pose"].shape[0])
        for idx in indices:
            row_idx = max(0, min(int(idx), n - 1))
            qpos = _decode_h5_json_row(root["obs/agent/qpos"][row_idx])
            mujoco_tcp = np.asarray(root["obs/extra/tcp_pose"][row_idx][:3], dtype=np.float64)
            mujoco_obj_world = list(np.asarray(root["obs/extra/obj_start"][row_idx], dtype=np.float64))
            arena_obj = np.asarray(_pose_7_world_to_robot_frame(mujoco_obj_world, spec.robot_base_pose)[:3], dtype=np.float64)

            _set_arena_robot_qpos_from_mujoco(robot, qpos)
            scene.write_data_to_sim()
            env.unwrapped.sim.forward()
            scene.update(dt=env.unwrapped.physics_dt)

            base_link = _body_pos(robot, "base_link")
            left_inner = _body_pos(robot, "left_inner_finger")
            right_inner = _body_pos(robot, "right_inner_finger")
            tool_left = _frame_pos(scene, "tool_leftfinger")
            tool_right = _frame_pos(scene, "tool_rightfinger")
            tensors = [x for x in (tool_left, tool_right) if x is not None]
            tool_center = torch.stack(tensors).mean(dim=0) if tensors else None
            body_tensors = [x for x in (left_inner, right_inner) if x is not None]
            body_center = torch.stack(body_tensors).mean(dim=0) if body_tensors else None

            def dist_to_obj(x):
                if x is None:
                    return None
                arr = np.asarray(_tensor_to_list(x, 3), dtype=np.float64)
                return float(np.linalg.norm(arr - arena_obj))

            res = {
                "index": row_idx,
                "mujoco_success": bool(root["success"][row_idx]) if "success" in root else None,
                "mujoco_tcp_pos": mujoco_tcp.tolist(),
                "arena_object_pos": arena_obj.tolist(),
                "mujoco_tcp_obj_dist": float(np.linalg.norm(mujoco_tcp - arena_obj)),
                "arena_base_link_pos": _tensor_to_list(base_link, 3),
                "arena_base_link_obj_dist": dist_to_obj(base_link),
                "arena_finger_body_center_pos": _tensor_to_list(body_center, 3),
                "arena_finger_body_center_obj_dist": dist_to_obj(body_center),
                "arena_tool_center_pos": _tensor_to_list(tool_center, 3),
                "arena_tool_center_obj_dist": dist_to_obj(tool_center),
                "arena_tool_left_pos": _tensor_to_list(tool_left, 3),
                "arena_tool_right_pos": _tensor_to_list(tool_right, 3),
                "qpos": qpos,
            }
            print(
                "[diagnose] fk_probe "
                f"idx={row_idx} mujoco_tcp_dist={res['mujoco_tcp_obj_dist']:.4f} "
                f"arena_tool_dist={res['arena_tool_center_obj_dist']} "
                f"arena_body_dist={res['arena_finger_body_center_obj_dist']}",
                flush=True,
            )
            results.append(res)
    return results


def _image_array(x: Any):
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_camera_images(obs: dict[str, Any], out_dir: Path, label: str) -> dict[str, str]:
    saved: dict[str, str] = {}
    cam_obs = obs.get("camera_obs") or {}
    if not isinstance(cam_obs, dict):
        return saved
    from PIL import Image

    for key, value in cam_obs.items():
        if not key.endswith("_rgb"):
            continue
        arr = _image_array(value)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            continue
        path = out_dir / f"{label}_{key}.png"
        Image.fromarray(arr[:, :, :3]).save(path)
        saved[key] = str(path)
    return saved


def _usd_camera_poses() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return out
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            path = str(prim.GetPath())
            if "/World/envs/env_0/Robot" not in path:
                continue
            if prim.GetTypeName() != "Camera" and "camera" not in prim.GetName().lower() and "cam" not in prim.GetName().lower():
                continue
            try:
                mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                tr = mat.ExtractTranslation()
                rot = mat.ExtractRotationQuat()
                out[path] = {
                    "position": [float(tr[0]), float(tr[1]), float(tr[2])],
                    "rotation_wxyz": [float(rot.GetReal()), float(rot.GetImaginary()[0]), float(rot.GetImaginary()[1]), float(rot.GetImaginary()[2])],
                }
            except Exception:
                # Some camera prims may not be xformable at the moment they are traversed.
                continue
    except Exception as e:
        out["error"] = {"message": str(e)}
    return out


def _scene_summary(env, obs: dict[str, Any], spec, label: str) -> dict[str, Any]:
    scene = getattr(env.unwrapped, "scene", None)
    policy = obs.get("policy") or {}
    summary: dict[str, Any] = {
        "label": label,
        "policy_joint_pos": _tensor_to_list(policy.get("joint_pos"), 16),
        "policy_gripper_pos": _tensor_to_list(policy.get("gripper_pos"), 8),
        "policy_eef_pos": _tensor_to_list(policy.get("eef_pos"), 3),
        "policy_eef_quat": _tensor_to_list(policy.get("eef_quat"), 4),
        "camera_poses": _usd_camera_poses(),
    }
    if scene is not None:
        sensor_camera_poses: dict[str, dict[str, list[float]]] = {}
        for cam_name in ("external_camera", "external_camera_2", "wrist_camera"):
            try:
                cam = scene[cam_name]
                data = getattr(cam, "data", None)
                if data is None:
                    continue
                sensor_camera_poses[cam_name] = {
                    "pos_w": _tensor_to_list(getattr(data, "pos_w", None), 3),
                    "quat_w_world": _tensor_to_list(getattr(data, "quat_w_world", None), 4),
                }
            except Exception:
                continue
        if sensor_camera_poses:
            summary["sensor_camera_poses"] = sensor_camera_poses
        try:
            pick = scene[spec.pickup_name]
            summary["pickup_root_pos_w"] = _tensor_to_list(pick.data.root_pos_w, 3)
            summary["pickup_root_quat_w"] = _tensor_to_list(pick.data.root_quat_w, 4)
        except Exception as e:
            summary["pickup_error"] = str(e)
        try:
            robot = scene["robot"]
            summary["robot_root_pos_w"] = _tensor_to_list(robot.data.root_pos_w, 3)
            summary["robot_root_quat_w"] = _tensor_to_list(robot.data.root_quat_w, 4)
            summary["robot_joint_names"] = list(getattr(robot.data, "joint_names", []) or [])[:20]
            summary["robot_joint_pos"] = _tensor_to_list(robot.data.joint_pos, 20)
            body_names = list(getattr(robot.data, "body_names", []) or [])
            body_pos = getattr(robot.data, "body_pos_w", None)
            body_quat = getattr(robot.data, "body_quat_w", None)
            interesting = ("panda_link0", "panda_link7", "panda_hand", "base_link", "left_inner_finger", "right_inner_finger")
            robot_bodies: dict[str, dict[str, list[float]]] = {}
            for body_name in body_names:
                if not any(key in body_name for key in interesting):
                    continue
                idx = body_names.index(body_name)
                robot_bodies[body_name] = {
                    "pos_w": _tensor_to_list(body_pos[:, idx, :] if body_pos is not None else None, 3),
                    "quat_w": _tensor_to_list(body_quat[:, idx, :] if body_quat is not None else None, 4),
                }
            summary["robot_body_poses"] = robot_bodies
        except Exception as e:
            summary["robot_error"] = str(e)
    return summary


def _plot_topdown(summary: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[diagnose] matplotlib unavailable, skipping top-down plot: {e}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axhline(0.0, color="#ddd", linewidth=1)
    ax.axvline(0.0, color="#ddd", linewidth=1)
    ax.scatter([0.0], [0.0], marker="s", s=90, label="Arena robot base")

    for snap in summary.get("snapshots", []):
        label = snap.get("label", "snapshot")
        pickup = snap.get("pickup_root_pos_w") or []
        eef = snap.get("policy_eef_pos") or []
        if len(pickup) >= 2:
            ax.scatter([pickup[0]], [pickup[1]], s=80, label=f"{label} pickup")
        if len(eef) >= 2:
            ax.scatter([eef[0]], [eef[1]], marker="x", s=80, label=f"{label} eef")
        for path, cam in (snap.get("camera_poses") or {}).items():
            pos = cam.get("position") if isinstance(cam, dict) else None
            if pos and len(pos) >= 2:
                ax.scatter([pos[0]], [pos[1]], marker="^", s=65, label=f"{label} {Path(path).name}")

    expected = summary.get("expected_arena_object_pose") or []
    if len(expected) >= 2:
        ax.scatter([expected[0]], [expected[1]], marker="o", s=160, facecolors="none", edgecolors="black", label="expected spawn xy")
    ax.set_xlabel("Arena X (m)")
    ax.set_ylabel("Arena Y (m)")
    ax.set_title("MolmoSpaces to Arena geometry")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _settle_action_from_obs(obs: dict[str, Any], env, embodiment_key: str):
    import torch

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    if not args.joint_pos_policy and "abs_joint_pos" not in str(embodiment_key):
        return actions

    policy = obs.get("policy") or {}
    joint_pos = policy.get("joint_pos")
    if joint_pos is None:
        return actions
    if hasattr(joint_pos, "detach"):
        joint_pos = joint_pos.detach()
    if hasattr(joint_pos, "to"):
        joint_pos = joint_pos.to(device=env.unwrapped.device)
    joint_pos = torch.as_tensor(joint_pos, dtype=actions.dtype, device=env.unwrapped.device)
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.unsqueeze(0)
    n = min(actions.shape[-1] - 1, joint_pos.shape[-1])
    actions[..., :n] = joint_pos[..., :n]
    return actions


def main() -> int:
    from molmo_spaces_isaac.arena.build_from_spec import build_arena_env_from_episode_spec
    from molmo_spaces_isaac.arena.episode_to_arena import (
        _pose_7_to_arena_pose,
        _pose_7_world_to_robot_frame,
        episode_dict_to_arena_spec,
    )
    from molmo_spaces_isaac.arena.thor_asset import get_thor_usd_path, should_apply_thor_up_axis_correction

    if args.assets_root is not None:
        os.environ.setdefault("MOLMO_ISAAC_ASSETS_ROOT", str(Path(args.assets_root).resolve()))
    if args.scenes_root is None:
        env_scenes = os.environ.get("MOLMO_SCENES_ROOT")
        if env_scenes:
            args.scenes_root = Path(env_scenes).resolve()
        elif args.assets_root is not None:
            args.scenes_root = Path(args.assets_root).resolve()

    if args.episode_json is not None:
        with open(args.episode_json) as f:
            episode = json.load(f)
        source_label = args.episode_json.stem
    else:
        bench = args.benchmark_dir or _default_pick_benchmark_dir()
        episode = _load_episode_from_benchmark(Path(bench), args.episode_idx)
        source_label = f"{Path(bench).name}_ep{args.episode_idx}"

    require_thor_only = not args.allow_objaverse
    spec = episode_dict_to_arena_spec(
        episode,
        require_thor_only=require_thor_only,
        background_key=args.background,
        scenes_root=args.scenes_root,
    )
    if spec is None:
        raise SystemExit("Episode could not be converted to ArenaEpisodeSpec.")

    out_dir = Path(args.out_dir).resolve() / source_label
    out_dir.mkdir(parents=True, exist_ok=True)

    thor_assets_dir, thor_metadata_path, objaverse_assets_dir = _resolve_asset_dirs(args.assets_root)
    cli_args_list = ["--device", getattr(args, "device", "cuda:0") or "cuda:0", "--num_envs", str(args.num_envs)]
    embodiment_key = _selected_embodiment_key()
    print(f"[diagnose] embodiment={embodiment_key} out_dir={out_dir}", flush=True)

    env, _ = build_arena_env_from_episode_spec(
        spec,
        env_name=f"molmospaces_arena_diag_{args.episode_idx}",
        embodiment_key=embodiment_key,
        enable_cameras=not args.no_cameras,
        thor_assets_dir=thor_assets_dir,
        thor_metadata_path=thor_metadata_path,
        objaverse_assets_dir=objaverse_assets_dir,
        episode_length_s=max(args.settle_steps + 1, 2) * 0.02,
        cli_args_list=cli_args_list,
        scene_extra_translation_xyz=tuple(args.scene_extra_xyz[:3]),
        use_joint_pos_control=args.joint_pos_policy and embodiment_key == "franka",
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
    )

    import torch

    obs, _ = env.reset()
    reset_images = _save_camera_images(obs, out_dir, "reset")
    snapshots = [_scene_summary(env, obs, spec, "reset")]
    if args.settle_steps > 0:
        actions = _settle_action_from_obs(obs, env, embodiment_key)
        for _ in range(args.settle_steps):
            obs, *_ = env.step(actions)
        settle_images = _save_camera_images(obs, out_dir, "settled")
        snapshots.append(_scene_summary(env, obs, spec, "settled"))
    else:
        settle_images = {}

    fk_probes = []
    if args.replay_h5 is not None and args.fk_probe_indices:
        fk_probes = _run_fk_probes(env, spec, args.replay_h5, args.replay_traj, list(args.fk_probe_indices))

    expected = None
    if spec.objects:
        expected = _pose_7_world_to_robot_frame(spec.objects[0][2], spec.robot_base_pose)
        if spec.objects[0][3] == "thor" and spec.objects[0][0] == spec.pickup_name:
            z_extra = float((os.environ.get("MOLMO_ARENA_PICK_Z_EXTRA") or "0").strip() or "0")
            expected = list(expected)
            expected[2] = float(expected[2]) + z_extra
            try:
                usd_p = get_thor_usd_path(spec.objects[0][1], thor_assets_dir)
                apply_up_axis = should_apply_thor_up_axis_correction(usd_p)
            except Exception:
                apply_up_axis = False
            expected_pos, expected_rot = _pose_7_to_arena_pose(expected, apply_thor_up_axis=apply_up_axis)
            expected = [*expected_pos, *expected_rot]

    summary = {
        "episode_idx": args.episode_idx,
        "house_index": episode.get("house_index"),
        "scene_dataset": episode.get("scene_dataset"),
        "scene_usd_path": str(spec.scene_usd_path) if spec.scene_usd_path else None,
        "embodiment": embodiment_key,
        "pickup_name": spec.pickup_name,
        "episode_robot_base_pose": spec.robot_base_pose,
        "episode_robot_init_joint_pos": spec.robot_init_joint_pos,
        "episode_object": spec.objects[0] if spec.objects else None,
        "expected_arena_object_pose": expected,
        "images": {"reset": reset_images, "settled": settle_images},
        "snapshots": snapshots,
        "fk_probes": fk_probes,
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    plot_path = out_dir / "topdown.png"
    _plot_topdown(summary, plot_path)
    print(f"[diagnose] wrote {json_path}", flush=True)
    print(f"[diagnose] wrote {plot_path}", flush=True)

    try:
        env.close()
    finally:
        if simulation_app is not None:
            simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
