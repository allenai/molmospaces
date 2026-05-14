#!/usr/bin/env python3
"""Render a MolmoSpaces benchmark episode in MuJoCo for Arena parity checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _set_default_env() -> None:
    repo = _repo_root()
    os.environ.setdefault("MLSPACES_ASSETS_DIR", str(repo / "assets"))
    os.environ.setdefault("MLSPACES_CACHE_DIR", str(Path.home() / ".cache/molmo-spaces-resources"))
    os.environ.setdefault("MLSPACES_AUTO_INSTALL", "True")
    os.environ.setdefault("MLSPACES_FORCE_INSTALL", "True")
    os.environ.setdefault("MUJOCO_GL", "egl")


def _load_raw_episode(benchmark_dir: Path, episode_index: int) -> dict[str, Any]:
    benchmark_file = benchmark_dir / "benchmark.json"
    if benchmark_file.is_file():
        with benchmark_file.open() as f:
            episodes = json.load(f)
        return episodes[episode_index]

    episode_files = sorted(benchmark_dir.glob("house_*/episode_*.json"))
    if not episode_files:
        raise FileNotFoundError(f"No benchmark.json or house_*/episode_*.json under {benchmark_dir}")
    with episode_files[episode_index].open() as f:
        return json.load(f)


def _restrict_resource_versions(raw_episode: dict[str, Any]) -> None:
    import molmo_spaces.molmo_spaces_constants as constants

    robot_name = raw_episode["robot"]["robot_name"]
    scene_dataset = raw_episode["scene_dataset"]
    data_split = raw_episode.get("data_split", "val")

    scene_versions = constants.DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"]
    robot_versions = constants.DATA_TYPE_TO_SOURCE_TO_VERSION["robots"]
    object_versions = constants.DATA_TYPE_TO_SOURCE_TO_VERSION["objects"]

    if scene_dataset == "ithor":
        scene_sources = {"ithor": scene_versions["ithor"]}
    elif scene_dataset == "procthor-10k":
        key = f"procthor-10k-{data_split}"
        scene_sources = {key: scene_versions[key]}
    elif scene_dataset == "procthor-objaverse":
        key = f"procthor-objaverse-{data_split}"
        scene_sources = {key: scene_versions[key]}
    else:
        raise ValueError(f"Unsupported scene_dataset for this diagnostic: {scene_dataset}")

    constants.DATA_TYPE_TO_SOURCE_TO_VERSION.clear()
    constants.DATA_TYPE_TO_SOURCE_TO_VERSION.update(
        {
            "robots": {robot_name: robot_versions[robot_name]},
            "scenes": scene_sources,
            "objects": {"thor": object_versions["thor"]},
            "grasps": {},
        }
    )
    constants._RESOURCE_MANAGER = None


def _install_lightweight_evaluation_package() -> None:
    """Expose evaluation submodules without running evaluation/__init__.py.

    The MolmoSpaces package currently imports eval_main from evaluation/__init__.py,
    and eval_main asserts that every benchmark data source is pinned to its full
    production version set. This diagnostic intentionally narrows the resource
    manager to one robot, one scene source, and THOR objects so that rendering a
    single episode does not install the entire benchmark corpus.
    """
    import sys
    import types

    package_name = "molmo_spaces.evaluation"
    if package_name in sys.modules:
        return

    package = types.ModuleType(package_name)
    package.__path__ = [str(_repo_root() / "molmo_spaces/evaluation")]
    package.__package__ = package_name
    sys.modules[package_name] = package


def _droid_shoulder_camera_spec() -> dict[str, Any]:
    return {
        "name": "exo_camera_1",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/fr3_link0"],
        "camera_offset": [0.1, 0.57, 0.66],
        "lookat_offset": [0.0, 0.0, 0.08],
        "camera_quaternion": [-0.3633, -0.1241, 0.4263, 0.8191],
        "fov": 71.0,
        "record_depth": False,
    }


def _episode_for_mujoco(raw_episode: dict[str, Any], inject_droid_camera: bool) -> Any:
    import molmo_spaces.molmo_spaces_constants as constants
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    episode = json.loads(json.dumps(raw_episode))

    if inject_droid_camera and not episode.get("cameras"):
        episode["cameras"] = [_droid_shoulder_camera_spec()]
        episode["img_resolution"] = episode.get("img_resolution") or [624, 352]

    scene_modifications = episode.get("scene_modifications", {})
    added_objects = scene_modifications.get("added_objects", {})
    object_poses = scene_modifications.get("object_poses", {})

    for object_name in list(added_objects.keys()):
        if "/" in object_name:
            continue

        namespaced_name = f"added/{object_name}"
        added_objects[namespaced_name] = added_objects.pop(object_name)
        if object_name in object_poses:
            object_poses[namespaced_name] = object_poses.pop(object_name)

        task = episode.get("task", {})
        for key in ("pickup_obj_name", "place_receptacle_name", "receptacle_name"):
            if task.get(key) == object_name:
                task[key] = namespaced_name

    for object_name, rel_path in list(added_objects.items()):
        expected_path = constants.ASSETS_DIR / rel_path
        if expected_path.is_file():
            continue

        installed_path = install_uid(Path(rel_path).stem)
        added_objects[object_name] = str(installed_path.relative_to(constants.ASSETS_DIR))

    return EpisodeSpec.model_validate(episode)


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _body_summary(model: Any, data: Any, substrings: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    needles = [s.lower() for s in substrings]
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if not name:
            continue
        lower = name.lower()
        if any(needle in lower for needle in needles):
            summary[name] = {
                "xpos": data.xpos[body_id].copy(),
                "xquat": data.xquat[body_id].copy(),
            }
    return summary


def _qpos_summary(model: Any, data: Any) -> dict[str, float]:
    qpos = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        if not name:
            continue
        addr = model.jnt_qposadr[joint_id]
        qpos[name] = float(data.qpos[addr])
    return qpos


def _make_render_config(episode: Any, output_dir: Path) -> Any:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.policy_configs import BasePolicyConfig
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.configs.task_configs import BaseMujocoTaskConfig
    from molmo_spaces.configs.task_sampler_configs import BaseMujocoTaskSamplerConfig
    from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler

    render_output_dir = output_dir

    class RenderPolicyConfig(BasePolicyConfig):
        policy_cls: type = None
        policy_type: str = "diagnostic"

    class RenderEvalConfig(MlSpacesExpConfig):
        num_envs: int = 1
        num_workers: int = 1
        task_type: str = "pick"
        use_passive_viewer: bool = False
        viewer_cam_dict: dict = {
            "distance": 5.0,
            "azimuth": 45.0,
            "elevation": -30.0,
            "lookat": [0.0, 0.0, 0.5],
        }
        policy_dt_ms: float = 500.0
        ctrl_dt_ms: float = 2.0
        sim_dt_ms: float = 2.0
        task_horizon: int = 500
        scene_dataset: str = episode.scene_dataset
        data_split: str = episode.data_split
        robot_config: FrankaRobotConfig = FrankaRobotConfig()
        task_sampler_config: BaseMujocoTaskSamplerConfig = BaseMujocoTaskSamplerConfig(
            task_sampler_class=BaseMujocoTaskSampler,
            house_inds=[episode.house_index],
            samples_per_house=1,
            task_batch_size=1,
            max_tasks=1,
            load_robot_from_file=True,
        )
        task_config: BaseMujocoTaskConfig = BaseMujocoTaskConfig(task_cls=None)
        policy_config: RenderPolicyConfig = RenderPolicyConfig()
        output_dir: Path = render_output_dir
        use_wandb: bool = False
        filter_for_successful_trajectories: bool = False
        eval_runtime_params: object = object()

        @property
        def tag(self) -> str:
            return "mujoco_render_diagnostic"

        def model_post_init(self, _context) -> None:
            assert (self.policy_dt_ms / self.ctrl_dt_ms).is_integer()
            assert (self.ctrl_dt_ms / self.sim_dt_ms).is_integer()

    config = RenderEvalConfig()
    config.robot_config.action_noise_config.enabled = False
    config.task_sampler_config.randomize_lighting = False
    config.task_sampler_config.randomize_textures = False
    config.task_sampler_config.randomize_textures_all = False
    config.task_sampler_config.randomize_dynamics = False
    config.task_sampler_config.sim_settle_timesteps = 0
    return config


def render_episode(
    benchmark_dir: Path,
    episode_index: int,
    output_dir: Path,
    inject_droid_camera: bool = True,
) -> None:
    _set_default_env()
    raw_episode = _load_raw_episode(benchmark_dir, episode_index)
    _restrict_resource_versions(raw_episode)
    _install_lightweight_evaluation_package()

    from PIL import Image

    import mujoco
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    episode = _episode_for_mujoco(raw_episode, inject_droid_camera)

    config = _make_render_config(episode, output_dir)

    sampler = JsonEvalTaskSampler(config, episode)
    try:
        task = sampler.sample_task(house_index=episode.house_index, variant="base")
        if task is None:
            raise RuntimeError("JsonEvalTaskSampler returned no task")

        env = sampler.env
        mujoco.mj_forward(env.current_model, env.current_data)
        env.camera_manager.registry.update_all_cameras(env)

        output_dir.mkdir(parents=True, exist_ok=True)
        camera_names = sorted(env.camera_manager.registry.keys())
        for camera_name in camera_names:
            frame = env.render_rgb_frame(camera_name)
            Image.fromarray(frame).save(output_dir / f"mujoco_{camera_name}.png")

        model = env.current_model
        data = env.current_data
        pickup_name = episode.task.get("pickup_obj_name", "")
        body_substrings = [
            pickup_name,
            "bowl",
            "cabinet",
            "counter",
            "fr3_link0",
            "fr3_link",
            "gripper",
            "base_link",
            "finger",
        ]
        summary = {
            "benchmark_dir": str(benchmark_dir),
            "episode_index": episode_index,
            "house_index": episode.house_index,
            "scene_dataset": episode.scene_dataset,
            "data_split": episode.data_split,
            "pickup_obj_name": pickup_name,
            "episode_pickup_pose": episode.task.get("pickup_obj_start_pose"),
            "episode_robot_base_pose": episode.task.get("robot_base_pose"),
            "episode_robot_init_qpos": episode.robot.init_qpos,
            "camera_names": camera_names,
            "camera_parameters": {
                camera_name: env.get_camera_parameters(camera_name) for camera_name in camera_names
            },
            "qpos": _qpos_summary(model, data),
            "bodies": _body_summary(model, data, body_substrings),
            "task_description": task.get_task_description(),
        }
        with (output_dir / "mujoco_summary.json").open("w") as f:
            json.dump(_jsonable(summary), f, indent=2)

        print(f"Wrote MuJoCo render diagnostics to {output_dir}")
    finally:
        sampler.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        default=_repo_root() / "molmo_spaces_isaac/examples/benchmark_ithor_pick_hard_simple",
    )
    parser.add_argument("--episode_index", type=int, default=4)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_repo_root() / "../diagnostics/mujoco_episode_render",
    )
    parser.add_argument("--no_inject_droid_camera", action="store_true")
    args = parser.parse_args()

    render_episode(
        benchmark_dir=args.benchmark_dir,
        episode_index=args.episode_index,
        output_dir=args.output_dir,
        inject_droid_camera=not args.no_inject_droid_camera,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
