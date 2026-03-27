import argparse
from pathlib import Path

from mlspaces_tests.data_generation.config import THORMapConfig, THORMapTestConfig
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.utils.scene_maps import ProcTHORMap, iTHORMap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("house_type", choices=["procthor-10k", "ithor", "all"])
    return parser.parse_args()


def save_thormap(thormap_config: THORMapConfig, out_dir: Path):
    house_type = thormap_config.house_type
    house_index = thormap_config.house_index
    agent_radius = thormap_config.agent_radius
    px_per_m = thormap_config.px_per_m

    cls_dict: dict[str, type[ProcTHORMap | iTHORMap]] = {
        "procthor-10k": ProcTHORMap,
        "ithor": iTHORMap,
    }
    cls = cls_dict[house_type]
    scene = get_scenes(house_type)["train"][house_index]

    if isinstance(scene, dict):
        scene_path = str(scene["ceiling"])
    else:
        scene_path = str(scene)

    install_scene_with_objects_and_grasps_from_path(scene_path)

    thormap = cls.from_mj_model_path(scene_path, px_per_m=px_per_m, agent_radius=None)
    thormap_dilated = cls.from_mj_model_path(
        scene_path, px_per_m=px_per_m, agent_radius=agent_radius
    )
    thormap.save(str(out_dir / f"thormap_{house_type}_{house_index}.png"))
    thormap_dilated.save(str(out_dir / f"thormap_{house_type}_{house_index}_dilated.png"))


def main():
    args = get_args()

    config = THORMapTestConfig()

    out_dir = Path(__file__).resolve().parent / "test_data" / "thormap"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.is_dir(), "Test data directory does not exist"

    for thormap_config in config.thormap_configs:
        if thormap_config.house_type == args.house_type or args.house_type == "all":
            save_thormap(thormap_config, out_dir)


if __name__ == "__main__":
    main()
