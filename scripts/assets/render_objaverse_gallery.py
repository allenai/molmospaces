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
import json
import logging
import random
import re
from dataclasses import dataclass, replace
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

# Filament MJCF attribute names (FILAMENT_ATTR_ENV_LIGHT_INTENSITY comes from
# task_sampler; the rest aren't exported there).
FILAMENT_ATTR_HEAD_LIGHT_INTENSITY = "filament.fallback.head_light_intensity"
FILAMENT_ATTR_TONE_MAPPING = "filament.out.tone_mapping"

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


@dataclass
class RenderSettings:
    """Filament render settings, tuned for photographic-looking output.

    Each knob has a `*_center` default (what a single, non-randomized render uses)
    and a `*_range` (inclusive lo/hi) used when randomizing across a dataset.

    Only a subset of MuJoCo's documented ``filament.*`` MJCF attributes actually
    affect this build's render; these were determined empirically (rendering a fixed
    scene and diffing images). Notably inert here: ``filament.out.{exposure,contrast,
    saturation,vibrance,temperature,tint}``, ``filament.msaa.enabled`` and MuJoCo's
    classic ``visual/headlight`` ambient -- so *exposure is controlled solely by the
    environment light intensity*, not by a camera exposure triad.
    """

    # -- Exposure. The env light is the dominant (and only real) exposure control.
    # With ACES, 15k reads as natural indoor daylight; 8k is a dim/evening interior
    # and 26k a bright, sunlit one. ACES holds highlights without clipping across
    # this entire range (measured: 0% clipped pixels even at 40k).
    env_light_intensity_center: float = 15000.0
    env_light_intensity_range: tuple[float, float] = (8000.0, 26000.0)

    # -- Tone mapping. This build accepts only "aces", "filmic" and "linear" (any
    # other string silently falls back to the engine default). ACES is the clear
    # winner: "linear" blows out badly (24% clipped pixels at 25k env) and "filmic"
    # lifts blacks into a washed-out, low-contrast look.
    tone_mapping: str = "aces"

    # -- Camera-mounted headlight. Physically unrealistic (a light that follows the
    # lens casts no shadows and flattens form), so it's off by default; a little of
    # it can rescue objects in very dark corners.
    head_light_intensity_center: float = 0.0
    head_light_intensity_range: tuple[float, float] = (0.0, 6000.0)

    # -- Ambient occlusion: the contact/micro-shadowing that stops objects looking
    # pasted onto the scene. Bent normals + screen-space cone tracing measurably
    # deepen it (image std rises 55.6 -> 63.7 going from AO off to full AO).
    ao_enabled: bool = True
    ao_bent_normals: bool = True
    ao_ssct: bool = True

    shadow_type: int = 0  # 0=PCF; 1..3 (VSM/DPCF/PCSS) look near-identical here

    # -- Camera geometry. Deliberately NOT randomized: the camera pose and FOV are
    # deterministic so repeat runs frame each object identically and only the
    # image-formation side of the render varies. The scene's own FOV is left alone,
    # so framing comes purely from how far back the camera sits.
    #
    # Camera distance scales with the object's bounding radius, then is clamped into
    # an absolute metre band.
    distance_scale: float = 3.6
    distance_m_bounds: tuple[float, float] = (1.0, 3.0)

    # Fraction of the frame the object should cover, used to nudge the distance
    # within the band above.
    target_fill: float = 0.3

    # Viewpoint candidates for the occlusion search.
    elevation_candidates_deg: tuple[float, ...] = (-15.0, -30.0, -50.0, -70.0)
    n_azimuths: int = 8

    def sample(self, rng: random.Random) -> RenderSettings:
        """Draw a randomized variant of the *image-formation* settings only.

        Camera extrinsics and FOV are left untouched; only the lighting/exposure side
        varies. Light intensities are sampled log-uniformly, since perceived
        brightness is roughly logarithmic in them.
        """

        def loguniform(lo: float, hi: float) -> float:
            if lo <= 0.0:
                return rng.uniform(lo, hi)
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

        return replace(
            self,
            env_light_intensity_center=loguniform(*self.env_light_intensity_range),
            head_light_intensity_center=rng.uniform(*self.head_light_intensity_range),
        )

    def as_manifest(self) -> dict:
        return {
            "env_light_intensity": round(self.env_light_intensity_center, 1),
            "tone_mapping": self.tone_mapping,
            "head_light_intensity": round(self.head_light_intensity_center, 1),
            "distance_m_bounds": list(self.distance_m_bounds),
            "ao": [self.ao_enabled, self.ao_bent_normals, self.ao_ssct],
        }


def load_model_for_filament(scene_path: Path, settings: RenderSettings) -> mj.MjModel:
    """Compile a scene with the filament render settings baked in.

    The env/head light intensities are also patched per-object at render time (see
    `set_light_intensities`), which is why they're added as numerics here even when
    left at their defaults.
    """
    spec = mj.MjSpec.from_file(str(scene_path))
    numerics = {
        FILAMENT_ATTR_ENV_LIGHT_INTENSITY: settings.env_light_intensity_center,
        FILAMENT_ATTR_HEAD_LIGHT_INTENSITY: settings.head_light_intensity_center,
        "filament.ao.enabled": float(settings.ao_enabled),
        "filament.ao.bent_normals": float(settings.ao_bent_normals),
        "filament.ao.ssct": float(settings.ao_ssct),
        "filament.shadows.type": float(settings.shadow_type),
    }
    for name, value in numerics.items():
        spec.add_numeric(name, [value], 1, name)
    spec.add_text(FILAMENT_ATTR_TONE_MAPPING, settings.tone_mapping, FILAMENT_ATTR_TONE_MAPPING)
    return spec.compile()


def _numeric_address(model: mj.MjModel, name: str) -> int | None:
    numeric_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id < 0:
        return None
    return int(model.numeric_adr[numeric_id])


def set_light_intensities(model: mj.MjModel, settings: RenderSettings) -> None:
    """Patch the filament light intensities on an already-compiled model.

    The filament context reads these when it is created, and we create a renderer per
    object, so this gives per-object exposure without recompiling the (large) house.
    """
    for name, value in (
        (FILAMENT_ATTR_ENV_LIGHT_INTENSITY, settings.env_light_intensity_center),
        (FILAMENT_ATTR_HEAD_LIGHT_INTENSITY, settings.head_light_intensity_center),
    ):
        adr = _numeric_address(model, name)
        if adr is not None:
            model.numeric_data[adr] = value


def _measure_view(
    renderer: MjFilamentRenderer,
    data: mj.MjData,
    cam: mj.MjvCamera,
    target_body_ids: set[int],
) -> tuple[float, float]:
    """Renders a segmentation pass (at the renderer's native size) and returns
    (fg_fraction, occlusion_penalty).

    Note: the renderer's offscreen framebuffer is fixed at construction time, so this
    must use the renderer's own width/height rather than a separate probe resolution.
    """
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
    settings: RenderSettings,
) -> np.ndarray | None:
    """Render one object-centered tile.

    The scene's own field of view is left untouched, so framing comes purely from the
    camera distance, which scales with the object's size and is then clamped into
    `settings.distance_m_bounds`. Azimuth/elevation are chosen by a segmentation search
    that targets a given on-screen fill while penalizing occlusion by other objects.
    """
    geom_ids = instance_geom_ids(model, instance)
    if not geom_ids:
        return None
    center, size = geom_aabb(model, data, geom_ids)
    radius = float(np.linalg.norm(size) / 2.0)
    if radius < 1e-4:
        return None

    # Never let the camera get closer than ~1.15x the object's own bounding radius --
    # closer than that risks the near clip plane (or occluding furniture) cutting
    # through the object -- and otherwise keep it inside the absolute metre band.
    lo, hi = settings.distance_m_bounds
    min_distance = max(radius * 1.15, lo)
    distance = float(np.clip(radius * settings.distance_scale, min_distance, max(min_distance, hi)))

    target_body_ids = set(instance.body_ids)

    # Pass 1: probe azimuth/elevation combos, scoring by how close the object's fill
    # fraction is to the target (object-centered, with margin) while penalizing
    # occlusion by other objects.
    best_score = -np.inf
    best_cam_params = (0.0, settings.elevation_candidates_deg[0])
    best_fill = settings.target_fill
    for azimuth in np.linspace(0, 360, settings.n_azimuths, endpoint=False):
        for elevation in settings.elevation_candidates_deg:
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = center
            cam.distance = distance
            cam.azimuth = float(azimuth)
            cam.elevation = float(elevation)

            fg_fraction, occlusion_penalty = _measure_view(renderer, data, cam, target_body_ids)
            score = -abs(fg_fraction - settings.target_fill) - 0.6 * occlusion_penalty
            if score > best_score:
                best_score = score
                best_cam_params = (float(azimuth), float(elevation))
                best_fill = fg_fraction

    # Pass 2: nudge the distance once so the object's fill fraction moves toward the
    # target (projected area scales roughly as 1/distance^2), clamped so a
    # mostly-occluded or edge-on best view can't drag the camera into the object.
    if best_fill > 1e-4:
        rescale = float(np.clip(np.sqrt(best_fill / settings.target_fill), 0.7, 1.6))
        distance = float(np.clip(distance * rescale, min_distance, max(min_distance, hi)))

    azimuth, elevation = best_cam_params
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    log.info(
        "%-28s radius=%.3fm distance=%.2fm on-screen=%.1f%%",
        instance.category,
        radius,
        distance,
        best_fill * 100.0,
    )

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
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Sample per-tile image-formation settings (environment/head light "
        "intensity) from RenderSettings' ranges instead of using the tuned centers. "
        "Camera pose and FOV stay deterministic either way.",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Override the camera distance band in metres (default 1.0 3.0).",
    )
    parser.add_argument(
        "--env-light",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Override the environment-light (exposure) randomization range.",
    )
    parser.add_argument(
        "--tone-mapping",
        type=str,
        default=RenderSettings.tone_mapping,
        choices=["aces", "filmic", "linear"],
        help="Filament tone mapper (this MuJoCo build accepts only these three).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    rng = random.Random(args.seed)

    base_settings = RenderSettings(tone_mapping=args.tone_mapping)
    if args.distance_m is not None:
        base_settings = replace(base_settings, distance_m_bounds=tuple(args.distance_m))
    if args.env_light is not None:
        lo, hi = args.env_light
        base_settings = replace(
            base_settings,
            env_light_intensity_range=(lo, hi),
            env_light_intensity_center=float(
                np.clip(base_settings.env_light_intensity_center, lo, hi)
            ),
        )
    log.info(
        "Render settings: tone_mapping=%s env_light=%s distance_m=%s randomize=%s",
        base_settings.tone_mapping,
        base_settings.env_light_intensity_range
        if args.randomize
        else base_settings.env_light_intensity_center,
        base_settings.distance_m_bounds,
        args.randomize,
    )

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
        model = load_model_for_filament(scene_path, base_settings)
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
    settings_by_key: dict[str, dict] = {}
    for scene_idx, instances in selections_by_scene.items():
        # Second pass: (re-)load only the scenes we actually selected objects from,
        # one at a time.
        model = load_model_for_filament(scene_paths[scene_idx], base_settings)
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
            settings = base_settings.sample(rng) if args.randomize else base_settings
            # Must precede renderer construction: the filament context snapshots these
            # light intensities when it is created.
            set_light_intensities(model, settings)
            renderer = MjFilamentRenderer(model=model, width=args.tile_size, height=args.tile_size)
            image = render_instance(renderer, model, data, instance, settings)
            renderer.close()
            if image is None:
                log.warning("Skipping %s: empty/degenerate AABB", instance.instance_key)
                continue
            images_by_key[instance.instance_key] = image
            manifest = settings.as_manifest()
            manifest.update(
                category=instance.category,
                uid=instance.uid,
                scene=scene_paths[scene_idx].name,
            )
            settings_by_key[instance.instance_key] = manifest
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
        manifest = {}
        for idx, (instance, image) in enumerate(rendered, start=1):
            name = (
                f"tile_{idx:0{n_digits}d}_{instance.category.replace(' ', '_')}_{instance.uid}.png"
            )
            Image.fromarray(image).save(args.tiles_dir / name)
            manifest[name] = settings_by_key.get(instance.instance_key, {})
        (args.tiles_dir / "render_settings.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "Wrote %d individual tiles (+ render_settings.json) to %s",
            len(rendered),
            args.tiles_dir,
        )

    if args.out is not None:
        grid = build_grid(tiles, n_cols=args.n_cols)
        Image.fromarray(grid).save(args.out)
        log.info("Wrote gallery grid to %s (%d tiles)", args.out, len(tiles))


if __name__ == "__main__":
    main()
