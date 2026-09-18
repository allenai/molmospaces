"""Render a gallery grid of object-centered Objaverse asset thumbnails.

Loads houses from the MolmoSpaces objaverse-backed scene sets (holodeck-objaverse,
procthor-objaverse), finds individual object instances inside them, frames each one
in an object-centered free camera, and renders it with the Filament renderer. The
resulting per-object tiles are assembled into a single grid image, similar in spirit
to example Objaverse asset galleries.

Usage:
    python scripts/assets/render_objaverse_gallery.py --out gallery.png

Requires the Filament-enabled mujoco wheel to be importable, e.g.:
    PYTHONPATH=/path/to/filament/mujoco:$PYTHONPATH python scripts/assets/render_objaverse_gallery.py
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_scenes
from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer
from molmo_spaces.tasks.task_sampler import FILAMENT_ATTR_ENV_LIGHT_INTENSITY
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
from molmo_spaces.utils.mj_model_and_data_utils import geom_aabb

# Grasp libraries to check for "pickability", in priority order (matches
# get_valid_pickupable_obja_uids' notion of "has a valid pickup grasp file").
PICKUP_GRASP_LIBRARIES = ("droid_objaverse", "droid")

log = logging.getLogger(__name__)

# Non-object body name prefixes that we never want to treat as "the object".
STRUCTURAL_PREFIXES = (
    "room",
    "wall",
    "floor",
    "ceiling",
    "doorway",
    "window",
    "world",
)

# Categories we prefer, in priority order, to loosely mirror a typical Objaverse asset
# gallery of *pickable* (graspable, tabletop-scale) items. These are drawn from the
# actual ObjectMeta category vocabulary observed among locally pickable objaverse
# assets (small household/decorative items, tools, toys, electronics -- not furniture,
# which is generally too large to be pickable).
PREFERRED_CATEGORIES = [
    "mug",
    "smartphone",
    "headphones",
    "sunglasses",
    "vase",
    "toy car",
    "figurine",
    "fruit",
    "apple",
    "seashell",
    "pen",
    "key",
    "ring",
    "baseball cap",
    "athletic shoe",
    "handheld gaming console",
    "toy",
    "sculpture",
    "decorative object",
    "bird model",
    "beverage can",
    "lantern",
    "mask",
    "toy rocket",
]

NAME_RE = re.compile(r"^(?P<category>[a-zA-Z]+)_(?P<uid>[0-9a-f]{32})_(?P<instance>\d+)_")


@dataclass
class ObjectInstance:
    category: str
    uid: str
    instance_key: str
    body_ids: list[int]


def is_structural(name: str) -> bool:
    prefix = name.split("_")[0].lower()
    return prefix in STRUCTURAL_PREFIXES


def is_pickable(uid: str) -> bool:
    """Whether `uid` has a locally-available pickup grasp file, i.e. is a "pickable"
    object per the same notion used by `synset_utils.get_valid_pickupable_obja_uids`
    (a robot has a scripted grasp for it, as opposed to static/structural props like
    walls, ceilings, or built-in furniture with no grasp annotations)."""
    for library in PICKUP_GRASP_LIBRARIES:
        grasp_file = ASSETS_DIR / "grasps" / library / uid / f"{uid}_grasps_filtered.npz"
        if grasp_file.exists():
            return True
    return False


def semantic_category(uid: str, fallback: str) -> str:
    """Look up the human-curated category for an objaverse uid (e.g. "soap dispenser",
    "music box") from ObjectMeta, falling back to the body-name-derived category. Many
    objaverse assets in these scenes use a generic "obja_<uid>_..." body name (i.e. the
    name-derived category is just the literal, uninformative token "obja"), so relying
    on the body name alone would collapse many genuinely different objects into one
    bucket."""
    try:
        from molmo_spaces.utils.object_metadata import ObjectMeta

        anno = ObjectMeta.annotation(uid)
    except Exception:
        anno = None
    if anno and anno.get("category"):
        return str(anno["category"]).strip().lower()
    return fallback


def collect_object_instances(model: mj.MjModel, pickable_only: bool = True) -> list[ObjectInstance]:
    """Group bodies in the model into per-instance objects, keyed by (category, uid, instance).

    If `pickable_only`, instances whose uid has no local pickup-grasp file are dropped
    (this also naturally drops non-objaverse/unrecognized bodies, whose "uid" is just
    their raw body name and will never match a grasp file).
    """
    groups: dict[tuple[str, str, str], list[int]] = {}
    for body_id in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if not name or is_structural(name):
            continue
        m = NAME_RE.match(name)
        if m is None:
            name_category, uid, instance = "obja", name, "0"
        else:
            name_category, uid, instance = (
                m.group("category"),
                m.group("uid"),
                m.group("instance"),
            )
        key = (name_category, uid, instance)
        groups.setdefault(key, []).append(body_id)

    instances = []
    for (name_category, uid, instance), body_ids in groups.items():
        if pickable_only and not is_pickable(uid):
            continue
        category = semantic_category(uid, fallback=name_category)
        instances.append(
            ObjectInstance(
                category=category,
                uid=uid,
                instance_key=f"{name_category}_{uid}_{instance}",
                body_ids=body_ids,
            )
        )
    return instances


def body_geom_ids(model: mj.MjModel, body_id: int) -> list[int]:
    # Note: object geoms in these scenes are tagged with group=4 (per the
    # __DYNAMIC_MJT__/__STRUCTURAL_MJT__ default classes), so we must not filter by
    # geom group here -- doing so previously excluded almost all real object geometry.
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]


def instance_geom_ids(model: mj.MjModel, instance: ObjectInstance) -> list[int]:
    geom_ids = []
    for body_id in instance.body_ids:
        geom_ids.extend(body_geom_ids(model, body_id))
    return geom_ids


def load_model_for_filament(scene_path: Path, environment_light_intensity: float = 15000.0):
    spec = mj.MjSpec.from_file(str(scene_path))
    spec.add_numeric(
        FILAMENT_ATTR_ENV_LIGHT_INTENSITY,
        [environment_light_intensity],
        1,
        "Filament default env light intensity",
    )
    model = spec.compile()
    return model


def _measure_view(
    renderer: MjFilamentRenderer,
    data: mj.MjData,
    cam: mj.MjvCamera,
    target_body_ids: set[int],
) -> tuple[float, float]:
    """Renders a segmentation pass (at the renderer's native size) and returns
    (fg_fraction, occlusion_penalty). Note: the renderer's offscreen framebuffer is
    fixed at construction time, so this must use the renderer's own width/height
    rather than a separate probe resolution."""
    renderer.update(data, camera=cam)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    body_ids_in_view = seg[..., 2]
    target_mask = np.isin(body_ids_in_view, list(target_body_ids))
    fg_fraction = float(target_mask.mean())
    any_fg_mask = body_ids_in_view >= 0
    occlusion_penalty = 0.0
    if any_fg_mask.any():
        occlusion_penalty = 1.0 - (target_mask.sum() / max(any_fg_mask.sum(), 1))
    return fg_fraction, occlusion_penalty


def render_instance(
    renderer: MjFilamentRenderer,
    model: mj.MjModel,
    data: mj.MjData,
    instance: ObjectInstance,
    n_azimuths: int = 8,
    elevations: tuple[float, ...] = (-15.0, -30.0, -50.0, -70.0),
    distance_scale: float = 3.6,
    target_fill: float = 0.3,
    min_camera_distance_m: float = 0.5,
    max_camera_distance_m: float = 1.5,
) -> np.ndarray | None:
    geom_ids = instance_geom_ids(model, instance)
    if not geom_ids:
        return None
    center, size = geom_aabb(model, data, geom_ids)
    radius = float(np.linalg.norm(size) / 2.0)
    if radius < 1e-4:
        return None
    # Never let the camera get closer than ~1.15x the object's own bounding radius --
    # closer than that risks the near clip plane (or occluding furniture) cutting
    # through the object, producing a degenerate macro/clipped shot -- but also clamp
    # to an absolute [min_camera_distance_m, max_camera_distance_m] range regardless of
    # object size, since the request here is for a consistent "standing back" distance
    # rather than one that's purely relative to each object's own scale.
    min_distance = max(radius * 1.15, 0.12, min_camera_distance_m)
    distance = float(np.clip(radius * distance_scale, min_distance, max_camera_distance_m))

    target_body_ids = set(instance.body_ids)

    # Pass 1: probe azimuth/elevation combos at low resolution, scoring by how
    # close the object's fill fraction is to a target (object-centered, with
    # margin) while penalizing occlusion by other objects.
    best_score = -np.inf
    best_cam_params = (0.0, elevations[0])
    best_fill = target_fill
    for azimuth in np.linspace(0, 360, n_azimuths, endpoint=False):
        for elevation in elevations:
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = center
            cam.distance = distance
            cam.azimuth = float(azimuth)
            cam.elevation = float(elevation)

            fg_fraction, occlusion_penalty = _measure_view(renderer, data, cam, target_body_ids)
            score = -abs(fg_fraction - target_fill) - 0.6 * occlusion_penalty
            if score > best_score:
                best_score = score
                best_cam_params = (float(azimuth), float(elevation))
                best_fill = fg_fraction

    # Pass 2: adjust distance once so the object's fill fraction is close to
    # target_fill (projected area scales roughly as 1/distance^2). The rescale is
    # clamped so a mostly-occluded or edge-on best view (very low measured fill)
    # can't drag the camera in so close that it clips into the object or nearby
    # geometry -- we'd rather keep some margin than produce a degenerate macro shot.
    if best_fill > 1e-4:
        rescale = float(np.clip(np.sqrt(best_fill / target_fill), 0.7, 1.6))
        distance = float(np.clip(distance * rescale, min_distance, max_camera_distance_m))

    azimuth, elevation = best_cam_params
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    renderer.update(data, camera=cam)
    return renderer.render()


def pick_instances(
    all_instances: list[list[ObjectInstance]],
    n_needed: int,
    rng: random.Random,
) -> list[tuple[int, ObjectInstance]]:
    """Pick a diverse set of (scene_index, instance) pairs, preferring PREFERRED_CATEGORIES,
    each category used at most once, falling back to any remaining category to fill more
    slots, and only once every category has been used at least once, allowing repeat
    categories (never repeat *instances*) to reach `n_needed`."""
    by_category: dict[str, list[tuple[int, ObjectInstance]]] = {}
    for scene_idx, instances in enumerate(all_instances):
        for inst in instances:
            by_category.setdefault(inst.category, []).append((scene_idx, inst))

    selected: list[tuple[int, ObjectInstance]] = []
    used_categories: set[str] = set()
    used_instance_keys: set[str] = set()

    def take(scene_idx: int, inst: ObjectInstance) -> None:
        selected.append((scene_idx, inst))
        used_categories.add(inst.category)
        used_instance_keys.add(inst.instance_key)

    for cat in PREFERRED_CATEGORIES:
        if len(selected) >= n_needed:
            return selected[:n_needed]
        candidates = by_category.get(cat)
        if not candidates:
            continue
        take(*rng.choice(candidates))

    remaining_categories = [c for c in by_category if c not in used_categories]
    rng.shuffle(remaining_categories)
    for cat in remaining_categories:
        if len(selected) >= n_needed:
            return selected[:n_needed]
        take(*rng.choice(by_category[cat]))

    # Every available category has now been used at least once. If we still need more
    # tiles than there are distinct categories, allow repeat categories (but never the
    # exact same object instance twice).
    all_candidates = [pair for candidates in by_category.values() for pair in candidates]
    leftover = [
        (scene_idx, inst)
        for scene_idx, inst in all_candidates
        if inst.instance_key not in used_instance_keys
    ]
    rng.shuffle(leftover)
    for scene_idx, inst in leftover:
        if len(selected) >= n_needed:
            break
        take(scene_idx, inst)

    return selected[:n_needed]


def build_grid(tiles: list[np.ndarray], n_cols: int, gutter: int = 6) -> np.ndarray:
    n_rows = (len(tiles) + n_cols - 1) // n_cols
    tile_h, tile_w = tiles[0].shape[:2]
    grid_h = n_rows * tile_h + (n_rows - 1) * gutter
    grid_w = n_cols * tile_w + (n_cols - 1) * gutter
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, n_cols)
        y0 = r * (tile_h + gutter)
        x0 = c * (tile_w + gutter)
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to write an assembled grid image. Omit to skip building the grid "
        "(e.g. when only --tiles-dir is wanted).",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="Directory to write each rendered tile as its own PNG (e.g. for merging "
        "into a custom layout yourself).",
    )
    parser.add_argument("--n-objects", type=int, default=18)
    parser.add_argument("--n-cols", type=int, default=6)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--n-scenes", type=int, default=6)
    parser.add_argument(
        "--scene-sets",
        nargs="+",
        default=["holodeck-objaverse", "procthor-objaverse"],
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-non-pickable",
        action="store_true",
        help="Also sample static/structural objects with no pickup grasp file "
        "(default: only sample pickable objects).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    rng = random.Random(args.seed)

    # Gather candidate scene paths across the requested scene sets.
    scene_paths: list[Path] = []
    for scene_set in args.scene_sets:
        index_map = get_scenes(scene_set, split=args.split)[args.split]
        keys = list(index_map.keys())
        rng.shuffle(keys)
        for key in keys:
            entry = index_map[key]
            base_path = entry["base"] if isinstance(entry, dict) else entry
            scene_paths.append(Path(base_path))
            if len(scene_paths) >= args.n_scenes:
                break

    log.info("Installing and indexing %d candidate scenes", len(scene_paths))

    # First pass: only discover which object instances each scene contains. We avoid
    # holding every scene's (large) compiled model in memory at once here -- loading
    # all houses simultaneously can exhaust memory on a workstation -- so each scene's
    # model/data is dropped again once we've recorded its instance list.
    all_instances: list[list[ObjectInstance]] = []
    for scene_path in scene_paths:
        # This installs (downloads/symlinks) the scene, its objects, and grasps as a
        # side effect; the scene XML itself lives at `scene_path` once installed.
        install_scene_with_objects_and_grasps_from_path(scene_path)
        model = load_model_for_filament(scene_path)
        instances = collect_object_instances(model, pickable_only=not args.include_non_pickable)
        log.info("%s -> %d object instances", scene_path, len(instances))
        all_instances.append(instances)
        del model

    selected = pick_instances(all_instances, args.n_objects, rng)
    log.info(
        "Selected %d objects: %s",
        len(selected),
        [inst.instance_key for _, inst in selected],
    )

    # Group selections by scene so each scene's (large) model/data is only loaded once,
    # even though we still create a fresh, short-lived MjFilamentRenderer per object --
    # reusing one Filament renderer/engine across many renders in-process was observed
    # to eventually crash (accumulated Filament engine/handle-allocator state), so we
    # pay the small per-object renderer-creation cost instead for reliability.
    selections_by_scene: dict[int, list[ObjectInstance]] = {}
    order: list[tuple[int, ObjectInstance]] = []
    for scene_idx, instance in selected:
        selections_by_scene.setdefault(scene_idx, []).append(instance)
        order.append((scene_idx, instance))

    images_by_key: dict[str, np.ndarray] = {}
    for scene_idx, instances in selections_by_scene.items():
        # Second pass: (re-)load only the scenes we actually selected objects from,
        # one at a time.
        model = load_model_for_filament(scene_paths[scene_idx])
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        for instance in instances:
            geom_ids = instance_geom_ids(model, instance)
            center, size = geom_aabb(model, data, geom_ids)
            log.debug(
                "%s: scene=%d ngeoms=%d center=%s size=%s",
                instance.instance_key,
                scene_idx,
                len(geom_ids),
                np.round(center, 3),
                np.round(size, 3),
            )
            renderer = MjFilamentRenderer(model=model, width=args.tile_size, height=args.tile_size)
            image = render_instance(renderer, model, data, instance)
            renderer.close()
            if image is None:
                log.warning("Skipping %s: empty/degenerate AABB", instance.instance_key)
                continue
            images_by_key[instance.instance_key] = image
        del model, data

    rendered = [
        (instance, images_by_key[instance.instance_key])
        for _, instance in order
        if instance.instance_key in images_by_key
    ]
    tiles = [image for _, image in rendered]

    if len(tiles) < args.n_objects:
        log.warning("Only rendered %d/%d requested objects", len(tiles), args.n_objects)

    if args.tiles_dir is not None:
        args.tiles_dir.mkdir(parents=True, exist_ok=True)
        n_digits = max(3, len(str(len(rendered))))
        for idx, (instance, image) in enumerate(rendered, start=1):
            tile_path = (
                args.tiles_dir
                / f"tile_{idx:0{n_digits}d}_{instance.category.replace(' ', '_')}_{instance.uid}.png"
            )
            Image.fromarray(image).save(tile_path)
        log.info("Wrote %d individual tiles to %s", len(rendered), args.tiles_dir)

    if args.out is not None:
        grid = build_grid(tiles, n_cols=args.n_cols)
        Image.fromarray(grid).save(args.out)
        log.info("Wrote gallery grid to %s (%d tiles)", args.out, len(tiles))


if __name__ == "__main__":
    main()
