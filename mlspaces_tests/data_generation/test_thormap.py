from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlspaces_tests.data_generation.config import THORMapTestConfig
from molmo_spaces.molmo_spaces_constants import get_resource_manager, get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap

TEST_DATA_DIR = get_resource_manager().symlink_dir / "test_data" / "thormap"
DEBUG_IMAGES_DIR = Path(__file__).resolve().parent / "test_debug_images"
OCC_MAP_IOU_THRESHOLD = 0.98


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Set up environment variables for all tests."""
    # TODO: use asset manager to download and save to test_data folder
    # For now, assume the user has defined MLSPACES_ASSETS_DIR in their environment
    yield


@pytest.fixture(scope="session")
def config():
    return THORMapTestConfig()


@pytest.fixture(scope="session")
def install_scenes(config: THORMapTestConfig):
    for thormap_config in config.thormap_configs:
        scene = get_scenes(thormap_config.house_type)["train"][thormap_config.house_index]
        if isinstance(scene, dict):
            scene_path = str(scene["ceiling"])
        else:
            scene_path = str(scene)

        install_scene_with_objects_and_grasps_from_path(scene_path)


@pytest.mark.parametrize(
    "scene_type,dilated",
    [
        ("procthor-10k", False),
        ("procthor-10k", True),
        ("ithor", False),
        ("ithor", True),
    ],
    ids=["procthor_undilated", "procthor_dilated", "ithor_undilated", "ithor_dilated"],
)
def test_thormap(config: THORMapTestConfig, install_scenes, scene_type: str, dilated: bool):
    cls_dict: dict[str, type[ProcTHORMap | iTHORMap]] = {
        "procthor-10k": ProcTHORMap,
        "ithor": iTHORMap,
    }
    cls = cls_dict[scene_type]

    for thormap_config in config.thormap_configs:
        if thormap_config.house_type == scene_type:
            scene = get_scenes(thormap_config.house_type)["train"][thormap_config.house_index]
            if isinstance(scene, dict):
                scene_path = str(scene["ceiling"])
            else:
                scene_path = str(scene)

            thormap = cls.from_mj_model_path(
                scene_path,
                px_per_m=thormap_config.px_per_m,
                agent_radius=thormap_config.agent_radius if dilated else None,
            )
            occupancy = thormap.occupancy

            suffix = "_dilated" if dilated else ""
            expected_thormap = cls.load(
                str(
                    TEST_DATA_DIR / f"thormap_{scene_type}_{thormap_config.house_index}{suffix}.png"
                )
            )
            expected_occupancy = expected_thormap.occupancy

            iou = np.sum(occupancy & expected_occupancy) / np.sum(occupancy | expected_occupancy)
            try:
                assert iou > OCC_MAP_IOU_THRESHOLD, (
                    f"IOU for ithormap {thormap_config.house_index} is {iou}, expected > {OCC_MAP_IOU_THRESHOLD}"
                )
            except AssertionError:
                save_dir = DEBUG_IMAGES_DIR / "thormap"
                save_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{scene_type}_{thormap_config.house_index}{suffix}"
                Image.fromarray(occupancy).save(str(save_dir / f"{prefix}_occupancy.png"))
                Image.fromarray(expected_occupancy).save(
                    str(save_dir / f"{prefix}_expected_occupancy.png")
                )
                Image.fromarray(occupancy ^ expected_occupancy).save(
                    str(save_dir / f"{prefix}_difference.png")
                )
                raise
