"""Prepare simple Unitree G1 tabletop user scenes for MolmoSpaces datagen.

The generated scenes are intentionally small: a floor, a lowered/pelvis-height
table, and a mocap bin that the task sampler can reposition on the tabletop.
Pickup objects are still injected by the task sampler from existing MolmoSpaces
assets so grasp files and episode serialization follow the normal pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from molmo_spaces.molmo_spaces_constants import (
    ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR,
    ASSETS_DIR,
)

SCENE_ASSET_NAME = "unitree_g1_tabletop_v1"
OUTPUT_SCENE_DIR = ASSETS_DIR / "scenes" / SCENE_ASSET_NAME

TABLE_BODY_NAME = "g1_table"
TABLETOP_GEOM_NAME = "g1_tabletop_geom"
TABLETOP_VISUAL_GEOM_NAME = "g1_tabletop_visual"
TABLE_MATERIAL_NAME = "g1_table_mat"
TABLE_RGBA = (0.74, 0.56, 0.36, 1.0)
PLACE_RECEPTACLE_BODY_NAME = "g1_place_bin"
PLACE_RECEPTACLE_SITE_NAME = "g1_place_bin_site"

PELVIS_MINUS_10CM_TABLETOP_HEIGHT_M = 0.693
PELVIS_TABLETOP_HEIGHT_M = 0.793


@dataclass(frozen=True)
class TabletopSceneSpec:
    name: str
    tabletop_height_m: float


SCENE_SPECS = (
    TabletopSceneSpec(
        "unitree_g1_tabletop_pelvis_minus_10cm_v1",
        PELVIS_MINUS_10CM_TABLETOP_HEIGHT_M,
    ),
    TabletopSceneSpec("unitree_g1_tabletop_pelvis_height_v1", PELVIS_TABLETOP_HEIGHT_M),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_SCENE_DIR,
        help="Output scene directory. Defaults to $MLSPACES_ASSETS_DIR/scenes/unitree_g1_tabletop_v1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def _set_attrs(elem: ET.Element, **attrs: object) -> ET.Element:
    for key, value in attrs.items():
        if isinstance(value, (list, tuple)):
            elem.set(key, " ".join(str(x) for x in value))
        else:
            elem.set(key, str(value))
    return elem


def _add_materials(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    _set_attrs(
        ET.SubElement(asset, "material"),
        name=TABLE_MATERIAL_NAME,
        rgba=TABLE_RGBA,
    )
    _set_attrs(
        ET.SubElement(asset, "material"),
        name="g1_bin_mat",
        rgba=[0.12, 0.33, 0.62, 1.0],
    )
    _set_attrs(
        ET.SubElement(asset, "material"),
        name="g1_floor_mat",
        rgba=[0.72, 0.74, 0.70, 1.0],
    )


def _add_visible_collision_geom(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: list[float] | tuple[float, ...],
    size: list[float] | tuple[float, ...],
    material: str,
    visual_name: str | None = None,
) -> None:
    _set_attrs(
        ET.SubElement(parent, "geom"),
        name=name,
        type=geom_type,
        pos=pos,
        size=size,
        material=material,
        **{"class": "__STRUCTURAL_MJT__"},
    )
    _set_attrs(
        ET.SubElement(parent, "geom"),
        name=visual_name or f"{name}_visual",
        type=geom_type,
        pos=pos,
        size=size,
        material=material,
        **{"class": "__VISUAL_MJT__"},
    )


def _add_floor(worldbody: ET.Element) -> None:
    _add_visible_collision_geom(
        worldbody,
        name="g1_tabletop_floor",
        geom_type="plane",
        pos=[0, 0, 0],
        size=[3.0, 3.0, 0.02],
        material="g1_floor_mat",
    )


def _add_table(worldbody: ET.Element, tabletop_height_m: float) -> None:
    table_body = _set_attrs(ET.SubElement(worldbody, "body"), name=TABLE_BODY_NAME)
    tabletop_half_size = (0.35, 0.45, 0.025)
    tabletop_center = (0.55, 0.0, tabletop_height_m - tabletop_half_size[2])
    _add_visible_collision_geom(
        table_body,
        name=TABLETOP_GEOM_NAME,
        geom_type="box",
        pos=tabletop_center,
        size=tabletop_half_size,
        material="g1_table_mat",
        visual_name=TABLETOP_VISUAL_GEOM_NAME,
    )

    leg_half_size = (0.025, 0.025, tabletop_height_m / 2.0)
    leg_z = tabletop_height_m / 2.0
    for i, x in enumerate((0.25, 0.85)):
        for j, y in enumerate((-0.35, 0.35)):
            _add_visible_collision_geom(
                table_body,
                name=f"g1_table_leg_{i}_{j}",
                geom_type="box",
                pos=[x, y, leg_z],
                size=leg_half_size,
                material="g1_table_mat",
            )


def _add_bin(worldbody: ET.Element, tabletop_height_m: float) -> None:
    bin_body = _set_attrs(
        ET.SubElement(worldbody, "body"),
        name=PLACE_RECEPTACLE_BODY_NAME,
        mocap="true",
        pos=[0.43, 0.18, tabletop_height_m + 0.001],
        quat=[1, 0, 0, 0],
    )

    outer_x = 0.14
    outer_y = 0.11
    wall = 0.012
    bottom_h = 0.012
    wall_h = 0.095

    _add_visible_collision_geom(
        bin_body,
        name="g1_place_bin_bottom",
        geom_type="box",
        pos=[0, 0, bottom_h / 2.0],
        size=[outer_x, outer_y, bottom_h / 2.0],
        material="g1_bin_mat",
    )
    for name, pos, size in (
        (
            "g1_place_bin_wall_pos_x",
            [outer_x - wall / 2.0, 0, bottom_h + wall_h / 2.0],
            [wall / 2.0, outer_y, wall_h / 2.0],
        ),
        (
            "g1_place_bin_wall_neg_x",
            [-outer_x + wall / 2.0, 0, bottom_h + wall_h / 2.0],
            [wall / 2.0, outer_y, wall_h / 2.0],
        ),
        (
            "g1_place_bin_wall_pos_y",
            [0, outer_y - wall / 2.0, bottom_h + wall_h / 2.0],
            [outer_x, wall / 2.0, wall_h / 2.0],
        ),
        (
            "g1_place_bin_wall_neg_y",
            [0, -outer_y + wall / 2.0, bottom_h + wall_h / 2.0],
            [outer_x, wall / 2.0, wall_h / 2.0],
        ),
    ):
        _add_visible_collision_geom(
            bin_body,
            name=name,
            geom_type="box",
            pos=pos,
            size=size,
            material="g1_bin_mat",
        )

    _set_attrs(
        ET.SubElement(bin_body, "site"),
        name=PLACE_RECEPTACLE_SITE_NAME,
        pos=[0, 0, bottom_h + wall_h],
        size=[0.015],
        rgba=[0.0, 0.8, 1.0, 0.6],
    )


def _metadata(scene_name: str, tabletop_height_m: float) -> dict:
    return {
        "scene_name": scene_name,
        "objects": {
            TABLE_BODY_NAME: {
                "category": "Table",
                "object_enum": "Table",
                "is_static": True,
                "tabletop_height_m": tabletop_height_m,
            },
            PLACE_RECEPTACLE_BODY_NAME: {
                "category": "Box",
                "object_enum": "Box",
                "is_static": True,
                "boundingBox": {"x": 0.28, "y": 0.22, "z": 0.107},
                "name_map": {"sites": {PLACE_RECEPTACLE_SITE_NAME: PLACE_RECEPTACLE_SITE_NAME}},
            },
        },
    }


def _write_scene(output_dir: Path, scene_spec: TabletopSceneSpec) -> Path:
    base_scene_path = (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR / "molmo_spaces" / "resources" / "base_scene.xml"
    )
    root = ET.parse(base_scene_path).getroot()
    root.set("model", scene_spec.name)
    _add_materials(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    _set_attrs(ET.SubElement(worldbody, "light"), name="key", pos=[0, -1.5, 2.5], dir=[0, 0.5, -1])
    _add_floor(worldbody)
    _add_table(worldbody, scene_spec.tabletop_height_m)
    _add_bin(worldbody, scene_spec.tabletop_height_m)

    xml_path = output_dir / f"{scene_spec.name}.xml"
    metadata_path = output_dir / f"{scene_spec.name}_metadata.json"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    with open(metadata_path, "w") as f:
        json.dump(_metadata(scene_spec.name, scene_spec.tabletop_height_m), f, indent=2)
        f.write("\n")
    mujoco.MjModel.from_xml_path(str(xml_path))
    return xml_path


def prepare_unitree_g1_tabletop(output_dir: Path = OUTPUT_SCENE_DIR, force: bool = False) -> list[Path]:
    output_dir = Path(output_dir)
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [_write_scene(output_dir, scene_spec) for scene_spec in SCENE_SPECS]


def main() -> None:
    args = parse_args()
    paths = prepare_unitree_g1_tabletop(args.output_dir, args.force)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
