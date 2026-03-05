from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing molmo_spaces_isaac when run via isaaclab.sh -p .../viewer.py
_src = Path(__file__).resolve().parent.parent.parent  # viewer -> assets -> molmo_spaces_isaac -> src
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualizer to try out different THOR assets")
parser.add_argument("--mode", type=str, choices=["single", "category", "scene"], required=True)
parser.add_argument("--model", type=str, default="", help="The single model to load")
parser.add_argument("--category", type=str, default="", help="The category of assets to load")
parser.add_argument("--scene", type=str, default="", help="The scene in usd to load")
parser.add_argument("--use-mesh", action="store_true")

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from enum import Enum
from pathlib import Path
from typing import cast

import carb
import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import omni
import torch
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from pxr import Usd
from scipy.spatial.transform import Rotation as R

from molmo_spaces_isaac import MOLMO_SPACES_ISAAC_BASE_DIR
from molmo_spaces_isaac.arena.thor_asset import get_thor_assets_root
from molmo_spaces_isaac.utils.common import PACKAGE_ROOT, load_thor_assets_metadata
from molmo_spaces_isaac.utils.prims import (
    compute_bbox_size,
    get_prim_local_pose,
    get_prim_world_pose,
    set_prim_pose,
)

# Use same THOR root as Arena (MOLMO_ISAAC_ASSETS_ROOT / MOLMO_THOR_USD_DIR, then repo)
DEFAULT_ASSETS_DIR = get_thor_assets_root()
DEFAULT_MODEL_PATH = DEFAULT_ASSETS_DIR / "Apple_1_mesh" / "Apple_1_mesh.usda"

DEFAULT_SCENES_DIR = PACKAGE_ROOT / "assets" / "usd" / "scenes"
DEFAULT_SCENE_PATH = DEFAULT_SCENES_DIR / "ithor" / "FloorPlan1_physics" / "scene.usda"

DEFAULT_RECORD_DIR = PACKAGE_ROOT / "output" / "videos"

# Prefer repo root or package resources (file ships in resources/)
_default_metadata = PACKAGE_ROOT / "usd_assets_metadata.json"
DEFAULT_USD_THOR_METADATA = _default_metadata if _default_metadata.is_file() else MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "usd_assets_metadata.json"

FIX_ASSETS_ROTATION = R.from_rotvec([90, 0, 0], degrees=True)


class eMode(str, Enum):
    SINGLE = "single"
    CATEGORY = "category"
    SCENE = "scene"


def place_object_above_plane(prim: Usd.Prim, world_x: float, world_y: float) -> None:
    position = np.array([world_x, world_y, 0.0], dtype=np.float64)
    if (bbox_size := compute_bbox_size(prim)) is not None:
        local_pos, local_quat = get_prim_local_pose(prim)
        world_pos, world_quat = get_prim_world_pose(prim)
        print(f"bbox: {bbox_size}")
        print(f"local >> pos: {local_pos}, quat: {local_quat}")
        print(f"world >> pos: {world_pos}, quat: {world_quat}")
        position[2] = bbox_size[2] / 2 + 0.001

    set_prim_pose(prim, position, FIX_ASSETS_ROTATION.as_quat(scalar_first=True))


def place_rigid_body_above_plane(rbody: RigidObject, world_x: float, world_y: float) -> None:
    rbody_prim = cast(Usd.Stage, rbody.stage).GetPrimAtPath(rbody.cfg.prim_path)
    if (bbox_size := compute_bbox_size(rbody_prim)) is not None:
        root_state = rbody.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([world_x, world_y, bbox_size[1] / 2 + 0.001])
        root_state[:, 3:7] = torch.tensor(FIX_ASSETS_ROTATION.as_quat(scalar_first=True))
        rbody.write_root_pose_to_sim(root_state[:, :7])
        rbody.reset()
        rbody.write_data_to_sim()


def place_articulation_above_plane(
    articulation: Articulation, world_x: float, world_y: float
) -> None:
    articulation_prim = cast(Usd.Stage, articulation.stage).GetPrimAtPath(
        articulation.cfg.prim_path
    )
    if (bbox_size := compute_bbox_size(articulation_prim)) is not None:
        root_state = articulation.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([world_x, world_y, bbox_size[1] / 2 + 0.001])
        root_state[:, 3:7] = torch.tensor(FIX_ASSETS_ROTATION.as_quat(scalar_first=True))
        articulation.write_root_pose_to_sim(root_state[:, :7])
        articulation.reset()
        articulation.write_data_to_sim()


@configclass
class AssetsViewerEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    action_space: int = 0
    observation_space: int = 0
    episode_length_s: float = 100.0
    state_space = 0

    sim: SimulationCfg = SimulationCfg(device="cuda:0", dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)

    mode: eMode = eMode.SINGLE
    model_path: Path = DEFAULT_MODEL_PATH
    category: str = "fridge"
    scene_path: Path = DEFAULT_SCENE_PATH
    use_mesh: bool = False


class AssetsViewerEnv(DirectRLEnv):
    def __init__(self, cfg: AssetsViewerEnvCfg, render_mode: str | None = None, **kwargs):
        self._usd_thor_metadata = load_thor_assets_metadata(DEFAULT_USD_THOR_METADATA)
        self._objects: list[RigidObject | Articulation] = []
        self._objects_positions: list[np.ndarray] = []
        self._objects_quats: list[np.ndarray] = []

        super().__init__(cfg, render_mode, **kwargs)

        self._is_running = True

        self._setup_init_poses()
        self.setup_keyboard()

    def setup_keyboard(self) -> None:
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "Q":
                print("Pressed Q, requesting application shutdown")
                self._is_running = False

    def is_running(self) -> bool:
        return self._is_running

    def _setup_scene(self) -> None:
        assert isinstance(self.cfg, AssetsViewerEnvCfg)

        if self.cfg.mode in (eMode.SINGLE, eMode.CATEGORY):
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

        match self.cfg.mode:
            case eMode.SINGLE:
                self._setup_single_model(self.cfg.model_path)
            case eMode.CATEGORY:
                self._setup_category_models(self.cfg.category, self.cfg.use_mesh)
            case eMode.SCENE:
                self._setup_thor_scene()

    def _setup_single_model(self, model_path: Path) -> None:
        asset_id = model_path.stem.replace("_prim", "").replace("_mesh", "")
        if asset_id not in self._usd_thor_metadata:
            print(
                f"[ERROR]: the given model '{model_path.stem}' with asset_id='{asset_id}' was not found in the metadata file"
            )
            return

        asset_metadata = self._usd_thor_metadata[asset_id]
        model_prim_path = f"/World/{model_path.stem}"

        spawn_cfg = sim_utils.UsdFileCfg(usd_path=model_path.absolute().as_posix())  # type: ignore
        if asset_metadata.articulated:
            articulation_cfg: ArticulationCfg = ArticulationCfg(
                spawn=spawn_cfg,  # ty:ignore[unknown-argument]
                prim_path=model_prim_path,  # ty:ignore[unknown-argument]
                init_state=ArticulationCfg.InitialStateCfg(),  # ty:ignore[unknown-argument]
                actuators={},  # ty:ignore[unknown-argument]
            )
            articulation = Articulation(cfg=articulation_cfg)
            self._objects.append(articulation)
            self._objects_positions.append(np.array([0.0, 0.0, 0.0]))
            self._objects_quats.append(FIX_ASSETS_ROTATION.as_quat(scalar_first=False))
            # place_articulation_above_plane(articulation, 0.0, 0.0)
            # articulation.update(self.sim.get_physics_dt())
        else:
            rigid_body_cfg = RigidObjectCfg(
                spawn=spawn_cfg,  # ty:ignore[unknown-argument]
                prim_path=model_prim_path,  # ty:ignore[unknown-argument]
                init_state=RigidObjectCfg.InitialStateCfg(),  # ty:ignore[unknown-argument]
            )
            rigid_body = RigidObject(cfg=rigid_body_cfg)
            self._objects.append(rigid_body)
            self._objects_positions.append(np.array([0.0, 0.0, 0.0]))
            self._objects_quats.append(FIX_ASSETS_ROTATION.as_quat(scalar_first=False))
            # place_rigid_body_above_plane(rigid_body, 0.0, 0.0)
            # rigid_body.update(self.sim.get_physics_dt())

    def _setup_init_poses(self) -> None:
        assert isinstance(self.cfg, AssetsViewerEnvCfg)
        if self.cfg.mode == eMode.SINGLE:
            for obj, _, _ in zip(self._objects, self._objects_positions, self._objects_quats):
                if isinstance(obj, RigidObject):
                    place_rigid_body_above_plane(obj, 0.0, 0.0)
                    obj.update(self.sim.get_physics_dt())
                elif isinstance(obj, Articulation):
                    place_articulation_above_plane(obj, 0.0, 0.0)
                    obj.update(self.sim.get_physics_dt())

    def _setup_category_models(self, category: str, use_mesh: bool) -> None:
        pass

    def _setup_thor_scene(self) -> None:
        pass

    def _pre_physics_step(self, actions) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self):
        return {}

    def _get_rewards(self):
        return {}

    def _get_dones(self):
        return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)


def main() -> int:
    model_path: Path | None = None
    if args.model != "" and Path(args.model).is_file():
        model_path = Path(args.model)

    # Prefer GPU; fall back to CPU when PyTorch is not compiled with CUDA (e.g. Isaac Sim aarch64).
    sim_device = getattr(args, "device", "cuda:0") or "cuda:0"
    if sim_device == "cuda":
        sim_device = "cuda:0"
    if sim_device.startswith("cuda"):
        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        if not cuda_ok:
            print("Torch not compiled with CUDA; using device cpu for simulation.", flush=True)
            sim_device = "cpu"
        else:
            try:
                torch.zeros(1, device=sim_device)
            except AssertionError:
                print("Torch not compiled with CUDA; using device cpu for simulation.", flush=True)
                sim_device = "cpu"

    env_cfg = AssetsViewerEnvCfg()
    env_cfg.mode = eMode(args.mode)
    env_cfg.model_path = model_path if model_path is not None else DEFAULT_MODEL_PATH
    env_cfg.sim.device = sim_device
    env_cfg.category = args.category
    env_cfg.use_mesh = args.use_mesh

    viewer_env = AssetsViewerEnv(env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        viewer_env,
        video_folder=DEFAULT_RECORD_DIR.as_posix(),
        disable_logger=True,
    )
    env.reset()

    while simulation_app.is_running() and viewer_env.is_running():
        actions = torch.zeros((viewer_env.num_envs, 0), device=viewer_env.device)
        env.step(actions)

    print("Closing app ...")

    env.close()

    simulation_app.close()

    print("Closing app DONE!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
