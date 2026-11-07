"""PBR material overrides for assets that ship without PBR parameters.

Almost none of our assets carry PBR material data. Every objaverse object MJCF
declares exactly one material with nothing but an albedo texture, and the THOR
scene materials are the same story, so ``mat_metallic``/``mat_roughness`` stay at
their "unset" sentinel of -1 and the whole scene is shaded by one undifferentiated
legacy response. This module fills that gap at load time, from the material
composition already recorded in the objathor metadata.

What Filament's PBR shader (``pbr.filamat``; read from the pre-4.0 custom wheel's
copy, MuJoCo 4.0 ships its own in ``mujoco/studio/assets``) actually
does with the numbers we set -- this drives every decision below::

    material.roughness  = RoughnessFactor;                 // <- mat_roughness
    material.roughness *= texture(Roughness, uv).x;        // <- mjTEXROLE_ROUGHNESS
    material.metallic   = MetallicFactor;                  // <- mat_metallic
    material.metallic  *= texture(Metallic, uv).x;         // <- mjTEXROLE_METALLIC
    ...
    diffuseColor = baseColor * (1 - metallic);
    f0           = baseColor * metallic + reflectance * (1 - metallic);
    computeDielectricF0(reflectance) = 0.16 * reflectance * reflectance;

Three consequences worth stating, because they are not what the MuJoCo docs imply:

1. Factors *multiply* their texture (the glTF convention), they do not replace it.
   So the per-class roughness below is a factor and the noise map is authored with
   a mean of 1.0 -- see :func:`roughness_noise_texture`.
2. ``metallic = 1`` zeroes the diffuse term and makes f0 the base colour, which is
   exactly the "baseColor dictates the colour of the metal reflection" rule.
3. There is no ``Reflectance`` uniform in the shipped material -- ``MaterialParams``
   holds only ``BaseColorFactor``, ``MetallicFactor`` and ``RoughnessFactor``. So
   ``reflectance`` keeps Filament's default of 0.5, i.e. an f0 of
   ``0.16 * 0.5^2 = 0.04``, the 4% Fresnel interface we want for dielectrics.
   We get the dielectric reflectance rule for free and cannot change it from MJCF;
   MuJoCo's ``mat_reflectance`` does not reach this shader.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

import mujoco as mj
import numpy as np

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

log = logging.getLogger(__name__)

# The dielectric f0 that the shader's fixed reflectance of 0.5 produces. Kept as a
# named constant so the 4% interface is greppable even though we cannot set it.
DIELECTRIC_F0 = 0.04

# Names of the shared textures this module injects into a spec.
ROUGHNESS_TEXTURE_PREFIX = "mlspaces_pbr_roughness_"
NORMAL_TEXTURE_PREFIX = "mlspaces_pbr_normal_"
AUTHORED_TEXTURE_PREFIX = "mlspaces_pbr_real_"
UNIT_TEXTURE = "mlspaces_pbr_unit"

# Objaverse object materials are named "<32 hex uid>_mat"; everything else in a
# composed scene is a THOR/structural material with a descriptive name.
_UID_MATERIAL_RE = re.compile(r"^(?P<uid>[0-9a-f]{32})_mat$")

# Roughness is clamped into this band: perfectly smooth and perfectly matte
# surfaces do not exist, and the band also keeps factor * noise inside (0, 1).
MIN_ROUGHNESS = 0.06
MAX_ROUGHNESS = 0.92


# Candidate finishes when a class does not name its own. Broad enough that two
# plastic mugs in the same scene do not carry pixel-identical micro-variation.
DEFAULT_FINISHES = ("speckle", "blotch", "smudged")


@dataclass(frozen=True)
class PbrClass:
    """A metallic/roughness pair, plus the surface finishes it may be given.

    ``finishes`` names entries of :data:`ROUGHNESS_FINISHES`; one is sampled per
    material, so surfaces of the same class do not all share one map.
    """

    metallic: float
    roughness: float
    finishes: tuple[str, ...] = DEFAULT_FINISHES

    def __post_init__(self) -> None:
        assert self.metallic in (0.0, 1.0), (
            f"metallic must be strictly 0.0 (dielectric) or 1.0 (metal), got {self.metallic}"
        )
        assert MIN_ROUGHNESS <= self.roughness <= MAX_ROUGHNESS, (
            f"roughness {self.roughness} outside [{MIN_ROUGHNESS}, {MAX_ROUGHNESS}]"
        )
        assert self.finishes, "a class must offer at least one finish"


def _metal(roughness: float, *finishes: str) -> PbrClass:
    # Metal is where the finish reads most strongly: it has no diffuse term, so the
    # reflection is all there is, and roughness alone decides how it breaks up.
    return PbrClass(
        metallic=1.0, roughness=roughness, finishes=finishes or ("brushed", "smudged", "worn")
    )


def _dielectric(roughness: float, *finishes: str) -> PbrClass:
    return PbrClass(metallic=0.0, roughness=roughness, finishes=finishes or DEFAULT_FINISHES)


# Anything we cannot identify is treated as a middling dielectric, which is both the
# safer error (a wrong metal reads far worse than a wrong plastic) and the rule:
# metallic strictly 0, reflectance left at the shader's 4%.
UNKNOWN = _dielectric(0.55)

# Keyed by the material tokens objathor records in each object's "materials" list.
# Frequencies quoted are how often the token is the dominant one across the 131,378
# annotated objects, so the common cases are the ones tuned most carefully.
MATERIAL_CLASSES: dict[str, PbrClass] = {
    # -- metals ------------------------------------------------------------
    "metal": _metal(0.35),  # 32,078 objects
    "steel": _metal(0.30),  # 958
    "stainless steel": _metal(0.25, "brushed", "smudged"),
    "aluminum": _metal(0.35),
    "aluminium": _metal(0.35),
    "iron": _metal(0.45),
    "cast iron": _metal(0.62),
    "copper": _metal(0.30, "smudged", "worn"),
    "brass": _metal(0.30, "smudged", "brushed"),
    "bronze": _metal(0.32),  # 295
    "silver": _metal(0.20, "smudged", "blotch"),
    "gold": _metal(0.20, "smudged", "blotch"),
    "chrome": _metal(0.12, "smudged"),
    "tin": _metal(0.38),
    "lead": _metal(0.55),
    "titanium": _metal(0.35),
    "nickel": _metal(0.28),
    "pewter": _metal(0.45),
    # -- dielectrics -------------------------------------------------------
    "plastic": _dielectric(0.35),  # 46,656
    "wood": _dielectric(0.60, "worn", "blotch"),  # 14,084
    "stone": _dielectric(0.80, "speckle", "worn"),  # 13,551
    "ceramic": _dielectric(0.20, "smudged", "blotch"),  # 3,718
    "porcelain": _dielectric(0.18, "smudged", "blotch"),
    "fabric": _dielectric(0.90, "weave", "worn"),  # 3,571
    "cloth": _dielectric(0.90, "weave", "worn"),
    "canvas": _dielectric(0.92, "weave", "worn"),
    "felt": _dielectric(0.92, "speckle", "worn"),
    "glass": _dielectric(0.08, "flat", "smudged"),  # 1,419
    "clay": _dielectric(0.88, "speckle", "worn"),  # 1,202
    "bone": _dielectric(0.45),  # 971
    "resin": _dielectric(0.40),  # 917
    "concrete": _dielectric(0.85, "speckle", "worn"),  # 769
    "brick": _dielectric(0.88, "speckle", "worn"),  # 754
    "paper": _dielectric(0.85, "speckle", "blotch"),  # 728
    "cardboard": _dielectric(0.88, "speckle", "worn"),  # 463
    "organic material": _dielectric(0.65),  # 697
    "organic matter": _dielectric(0.65),
    "plaster": _dielectric(0.90, "speckle", "blotch"),  # 627
    "calcium carbonate": _dielectric(0.70),  # 587
    "leather": _dielectric(0.60, "worn", "smudged"),  # 550
    "rubber": _dielectric(0.85),  # 509
    "marble": _dielectric(0.25, "blotch", "smudged"),  # 389
    "rock": _dielectric(0.85, "speckle", "worn"),  # 264
    "soil": _dielectric(0.92, "speckle"),
    "moss": _dielectric(0.92),
    "foam": _dielectric(0.92),
    "stuffing": _dielectric(0.92),
    "polymer": _dielectric(0.40),
    "synthetic": _dielectric(0.45),
    "composite": _dielectric(0.50),
    "gemstone": _dielectric(0.10, "flat"),
    "crystal": _dielectric(0.10, "flat", "smudged"),
    "water": _dielectric(0.10, "flat"),
    "ice": _dielectric(0.15, "flat", "blotch"),
    "wax": _dielectric(0.55),
    "ink": _dielectric(0.50),
    "glaze": _dielectric(0.15, "smudged", "blotch"),
    "paint": _dielectric(0.45),
    "enamel": _dielectric(0.25, "smudged", "blotch"),
    "lacquer": _dielectric(0.20, "smudged", "blotch"),
    "varnish": _dielectric(0.25),
    "food": _dielectric(0.65),
    "dough": _dielectric(0.75),
    "flour": _dielectric(0.92),
    "bread": _dielectric(0.80),
    "cotton": _dielectric(0.92, "weave", "worn"),
    "wool": _dielectric(0.92, "weave", "worn"),
    "silk": _dielectric(0.45, "weave", "blotch"),
    "denim": _dielectric(0.92, "weave", "worn"),
    "quartz": _dielectric(0.20, "blotch", "speckle"),
    "granite": _dielectric(0.45, "speckle", "blotch"),
    "terracotta": _dielectric(0.85, "speckle", "worn"),
    "acrylic": _dielectric(0.30),
    "vinyl": _dielectric(0.40),
    "nylon": _dielectric(0.60),
    "polyester": _dielectric(0.75, "weave", "speckle"),
    "latex": _dielectric(0.80),
    "silicone": _dielectric(0.70),
    "chalk": _dielectric(0.92),
    "sand": _dielectric(0.92, "speckle"),
    "gravel": _dielectric(0.90, "speckle"),
    "cork": _dielectric(0.88, "speckle", "worn"),
    "bamboo": _dielectric(0.55, "worn", "blotch"),
    "straw": _dielectric(0.90, "weave", "worn"),
    "wicker": _dielectric(0.85, "weave", "worn"),
    "rattan": _dielectric(0.85, "weave", "worn"),
    "shell": _dielectric(0.35),
    "ivory": _dielectric(0.40),
    "pearl": _dielectric(0.15, "flat", "blotch"),
    "diamond": _dielectric(0.08, "flat"),
    "obsidian": _dielectric(0.12, "flat", "smudged"),
    "slate": _dielectric(0.70, "speckle", "worn"),
    "limestone": _dielectric(0.85, "speckle", "worn"),
    "sandstone": _dielectric(0.90, "speckle", "worn"),
    "tile": _dielectric(0.25, "smudged", "blotch"),
    "tiles": _dielectric(0.25, "smudged", "blotch"),
    "asphalt": _dielectric(0.92, "speckle", "worn"),
    "mud": _dielectric(0.88, "blotch", "speckle"),
    "flesh": _dielectric(0.62),
    "organic": _dielectric(0.65),
    "matter": _dielectric(0.65),
    "material": UNKNOWN,
    "fiber": _dielectric(0.45),
    "chitin": _dielectric(0.40),
    "mineral": _dielectric(0.75),
    "fur": _dielectric(0.92),
    "hair": _dielectric(0.85),
    "snow": _dielectric(0.88),
    "feathers": _dielectric(0.85),
    "feather": _dielectric(0.85),
    "graphite": _dielectric(0.70),
    "carbon": _dielectric(0.45),
    "velvet": _dielectric(0.92, "speckle", "worn"),
    "suede": _dielectric(0.90, "speckle", "worn"),
    "mesh": _dielectric(0.85, "weave", "speckle"),
    "grass": _dielectric(0.90),
    "air": UNKNOWN,
}

# A coat hides whatever it is painted onto: a painted or glazed steel tin is a
# dielectric at the surface, however much metal is under it. So when one of these
# is present in meaningful quantity it wins over a metal verdict, even though it is
# rarely the largest component by volume.
_COATINGS = ("paint", "glaze", "lacquer", "enamel", "varnish")
_COATING_FRACTION = 0.08

# "air" is objathor's filler for hollow volume, never a surface.
_NON_SURFACE = ("air", "water")

# Substring rules for THOR/structural material names, which carry no metadata. Most
# THOR names describe the object ("Fridge_Readout_Mat") rather than the material, so
# these only fire on the minority that name a material outright; the rest fall to
# UNKNOWN. Ordered longest-first so "stainlesssteel" wins over "steel".
_NAME_RULES: tuple[tuple[str, PbrClass], ...] = tuple(
    sorted(
        (
            ("stainlesssteel", MATERIAL_CLASSES["stainless steel"]),
            ("porcelain", MATERIAL_CLASSES["porcelain"]),
            ("aluminum", MATERIAL_CLASSES["aluminum"]),
            ("cardboard", MATERIAL_CLASSES["cardboard"]),
            ("concrete", MATERIAL_CLASSES["concrete"]),
            ("ceramic", MATERIAL_CLASSES["ceramic"]),
            ("plastic", MATERIAL_CLASSES["plastic"]),
            ("leather", MATERIAL_CLASSES["leather"]),
            ("granite", MATERIAL_CLASSES["stone"]),
            ("drywall", MATERIAL_CLASSES["plaster"]),
            ("carpet", MATERIAL_CLASSES["fabric"]),
            ("marble", MATERIAL_CLASSES["marble"]),
            ("fabric", MATERIAL_CLASSES["fabric"]),
            ("rubber", MATERIAL_CLASSES["rubber"]),
            ("chrome", MATERIAL_CLASSES["chrome"]),
            ("bronze", MATERIAL_CLASSES["bronze"]),
            ("copper", MATERIAL_CLASSES["copper"]),
            ("brass", MATERIAL_CLASSES["brass"]),
            ("steel", MATERIAL_CLASSES["steel"]),
            ("glass", MATERIAL_CLASSES["glass"]),
            ("stone", MATERIAL_CLASSES["stone"]),
            ("brick", MATERIAL_CLASSES["brick"]),
            ("cloth", MATERIAL_CLASSES["cloth"]),
            ("paper", MATERIAL_CLASSES["paper"]),
            ("metal", MATERIAL_CLASSES["metal"]),
            ("tiles", MATERIAL_CLASSES["ceramic"]),
            ("wood", MATERIAL_CLASSES["wood"]),
            ("iron", MATERIAL_CLASSES["iron"]),
        ),
        key=lambda rule: -len(rule[0]),
    )
)


def lookup_token(token: str) -> PbrClass | None:
    """Resolve one objathor material token to a PBR class, compounds included.

    The annotation vocabulary has a long tail of compounds -- "synthetic fabric",
    "stainless steel", "organic material" -- so an exact lookup is tried first and
    then the compound is read by its head noun, which in these names is the last
    word that we recognise ("synthetic fabric" is a fabric, not a synthetic).
    """
    token = token.strip().lower()
    if token in MATERIAL_CLASSES:
        return MATERIAL_CLASSES[token]

    for word in reversed(token.split()):
        if word in MATERIAL_CLASSES:
            return MATERIAL_CLASSES[word]
    return None


def classify_composition(materials: list[str], composition: list[float]) -> PbrClass:
    """Pick a PBR class from an objathor ``materials``/``composition`` annotation.

    ``composition`` is a volume fraction, not a surface fraction, so this is a proxy:
    it takes the largest non-filler component as the surface, then lets any
    meaningful coat of paint or glaze override a metal verdict.
    """
    if not materials:
        return UNKNOWN

    tokens = [str(m).strip().lower() for m in materials]
    if not composition or len(composition) != len(tokens):
        # Annotations always agree in the shipped metadata, but do not trust it blindly.
        fractions = [1.0 / len(tokens)] * len(tokens)
    else:
        fractions = [float(c) for c in composition]

    by_token: dict[str, float] = {}
    for token, fraction in zip(tokens, fractions, strict=True):
        by_token[token] = by_token.get(token, 0.0) + fraction

    surface = {t: f for t, f in by_token.items() if t not in _NON_SURFACE}
    if not surface:
        return UNKNOWN

    dominant = max(surface, key=lambda t: surface[t])
    chosen = lookup_token(dominant) or UNKNOWN

    if chosen.metallic == 1.0:
        coat = max(
            (t for t in _COATINGS if surface.get(t, 0.0) >= _COATING_FRACTION),
            key=lambda t: surface[t],
            default=None,
        )
        if coat is not None:
            return MATERIAL_CLASSES[coat]

    return chosen


def classify_material_name(name: str) -> PbrClass | None:
    """Best-effort PBR class for a THOR/structural material, from its name alone.

    Returns None when the name says nothing about the material, so the caller can
    tell "no idea" apart from a confident guess.
    """
    haystack = re.sub(r"[^a-z]", "", name.lower())
    for needle, pbr_class in _NAME_RULES:
        if needle in haystack:
            return pbr_class
    return None


def roughness_factor_gain(amplitude: float) -> float:
    """The factor by which a roughness map is scaled down to fit uint8.

    Multiply it back into the roughness factor so that ``factor * texture`` averages
    out to the roughness the material class actually asked for.
    """
    return 1.0 + amplitude


def _octave_noise(
    resolution: int,
    cells: int,
    rng: np.random.Generator,
    anisotropy: float = 1.0,
) -> np.ndarray:
    """One octave of band-limited noise with roughly `cells` features per edge.

    Low-passed in the frequency domain rather than built by upsampling a coarse grid.
    An upsample-and-blur leaves the coarse grid's cell edges in the field, and while
    that is invisible in a roughness map it is glaring in a normal map, because the
    normal is the *gradient* of the field and a cell edge differentiates into a ridge
    -- the relief comes out looking tiled into squares. A Gaussian low-pass has no
    preferred direction and no grid to leak through.

    ``anisotropy`` > 1 stretches the features along u, which is what a directional
    abrasive leaves behind.
    """
    white = rng.normal(0.0, 1.0, (resolution, resolution)).astype(np.float32)
    spectrum = np.fft.rfft2(white)

    freq_v = np.fft.fftfreq(resolution)[:, None] * resolution
    freq_u = np.fft.rfftfreq(resolution)[None, :] * resolution
    cutoff = max(1.0, float(cells))
    # Features stretch along u by penalising u frequencies, i.e. by band-limiting
    # that axis harder, which leaves variation fast across the streaks and slow
    # along them.
    radius = np.sqrt((freq_u * anisotropy / cutoff) ** 2 + (freq_v / cutoff) ** 2)

    field = np.fft.irfft2(spectrum * np.exp(-(radius**2)), s=(resolution, resolution))
    return field.astype(np.float32)


def _unit(field: np.ndarray) -> np.ndarray:
    """Normalise an octave to unit variance before it is mixed with another.

    The low-pass attenuates an octave by an amount that depends on its cutoff, so
    octaves come out of :func:`_octave_noise` with wildly different variances. Mixing
    them by raw weight then lets whichever happens to be loudest take over -- a
    heavily band-limited octave, such as the streaks in a brushed finish, all but
    vanishes next to a broadband one. Normalising first makes the weights mean what
    they say.
    """
    std = float(field.std())
    return field / std if std > 1e-9 else field


def _finish_flat(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """No variation. For surfaces where micro-detail would be invented, not observed."""
    return np.zeros((resolution, resolution), dtype=np.float32)


def _finish_speckle(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Fine isotropic grain: moulded plastic, matte paint."""
    return _octave_noise(resolution, 256, rng)


def _finish_blotch(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Broad, slow variation in finish: unevenly polished or unevenly worn surfaces."""
    return _octave_noise(resolution, 12, rng)


def _finish_brushed(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Anisotropic streaks along U: brushed and machined metal.

    Made by generating noise that is fine across the surface and smeared along it,
    which is what a directional abrasive leaves behind.
    """
    streaks = _unit(_octave_noise(resolution, 200, rng, anisotropy=60.0))
    fine = _unit(_octave_noise(resolution, 400, rng))
    return 0.85 * streaks + 0.15 * fine


def _finish_smudged(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Soft elliptical blobs over a gentle base: handling marks and fingerprints."""
    return _unit(_octave_noise(resolution, 32, rng)) + _unit(_smudges(resolution, rng))


def _finish_worn(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Broad wear plus grain: old wood, scuffed leather, weathered metal."""
    return _unit(_octave_noise(resolution, 8, rng)) + 0.6 * _unit(
        _octave_noise(resolution, 128, rng)
    )


def _finish_weave(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """A regular cross-hatch: woven textiles, at a random thread pitch and phase."""
    pitch = float(rng.integers(48, 160))
    phase_u, phase_v = rng.uniform(0.0, 2.0 * np.pi, size=2)
    axis = np.arange(resolution, dtype=np.float32) * (2.0 * np.pi / pitch)
    warp = np.sin(axis + phase_u)[None, :]
    weft = np.sin(axis + phase_v)[:, None]
    return _unit(warp + weft) + 0.35 * _unit(_octave_noise(resolution, 256, rng))


#: The template library. Each entry returns a raw, arbitrarily-scaled field; it is
#: normalised and encoded by :func:`roughness_texture`, so a generator only has to
#: describe the *character* of the variation, not its magnitude.
ROUGHNESS_FINISHES: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "flat": _finish_flat,
    "speckle": _finish_speckle,
    "blotch": _finish_blotch,
    "brushed": _finish_brushed,
    "smudged": _finish_smudged,
    "worn": _finish_worn,
    "weave": _finish_weave,
}


@lru_cache(maxsize=64)
def finish_field(finish: str, resolution: int = 1024, seed: int = 0) -> np.ndarray:
    """The normalised height field behind a finish, in [-1, 1] with zero mean.

    Both maps a finish produces come from this one field, so a surface's roughness
    variation and its relief describe the same physical texture: where the weave
    stands proud it is also glossier, because that is the thread the light catches.
    Generating them independently would put the two out of register and read as two
    unrelated patterns laid over each other.
    """
    if finish not in ROUGHNESS_FINISHES:
        raise KeyError(f"unknown finish {finish!r}; known: {sorted(ROUGHNESS_FINISHES)}")

    rng = np.random.default_rng(seed)
    field = ROUGHNESS_FINISHES[finish](resolution, rng)

    # Trim the tail first -- several finishes are skewed rather than symmetric, and
    # clipping a skewed field is not mean-preserving -- and only then centre and
    # normalise. Recentring *after* the clip is what gives the exact zero mean that
    # the roughness multiply depends on; scaling by the peak rather than the standard
    # deviation makes the caller's amplitude the true maximum deviation.
    field = field - field.mean()
    std = float(field.std())
    if std > 1e-6:
        field = field / std
        field = np.clip(field, -3.0, 3.0)
        field = field - field.mean()
        peak = float(np.abs(field).max())
        if peak > 1e-6:
            field = field / peak
    field.flags.writeable = False
    return field


@lru_cache(maxsize=64)
def normal_texture(
    finish: str = "smudged",
    resolution: int = 1024,
    strength: float = 0.10,
    seed: int = 0,
) -> np.ndarray:
    """Build a tangent-space normal map from a finish's height field.

    Roughness and relief are not the same thing. Roughness is a statistic of facets
    below the texel, and only ever blurs or sharpens a reflection -- if the reflected
    environment is dim, as ours are, it barely shows. A normal map instead turns the
    surface to face a different way at every texel, so each bump gets a lit side and a
    shaded side and produces luminance contrast directly. Measured on a macro shot of
    one object, a normal map at strength 0.35 moved 82% of pixels where the roughness
    map at the same framing moved 6%, so this is the knob that carries visible detail
    -- and the reason the default here is deliberately gentle.

    ``strength`` is the RMS surface slope, which keeps it independent of resolution:
    0.10 is a soft texture, 0.35 is aggressively embossed.

    Returns HxWx3 uint8 in the usual encoding, ``rgb = normal * 0.5 + 0.5``, so a flat
    surface is (128, 128, 255). It must be uploaded as linear, not sRGB.
    """
    field = finish_field(finish, resolution, seed)

    # Gradient of the height field; the surface normal is (-dh/du, -dh/dv, 1).
    grad_v, grad_u = np.gradient(field.astype(np.float32))
    rms = float(np.sqrt(np.mean(grad_u**2 + grad_v**2)))
    if rms > 1e-9:
        scale = strength / rms
        grad_u = grad_u * scale
        grad_v = grad_v * scale

    normal = np.stack([-grad_u, -grad_v, np.ones_like(grad_u)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    texture = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    texture.flags.writeable = False
    return texture


@lru_cache(maxsize=64)
def roughness_texture(
    finish: str = "smudged",
    resolution: int = 1024,
    amplitude: float = 0.30,
    seed: int = 0,
) -> np.ndarray:
    """Build a high-resolution roughness map that multiplies the roughness factor.

    A constant roughness gives a reflection that is uniform across a whole surface,
    which is the giveaway of a synthetic render. Each named finish in
    :data:`ROUGHNESS_FINISHES` breaks it up differently -- fine grain, broad wear,
    directional brushing, handling marks, a woven cross-hatch.

    The map multiplies the per-material roughness factor (see the module docstring),
    so conceptually it is centred on 1.0 and varies by +/- ``amplitude``. A uint8
    texture can only encode [0, 1] though, and a map centred on 1.0 would have its
    entire upper half clipped flat, so it is stored scaled down by
    :func:`roughness_factor_gain` -- peak exactly 1.0, mean ``1 / (1 + amplitude)``
    -- and the caller multiplies that gain back into the roughness factor. The
    returned array is a single channel of uint8, to hand to MuJoCo as texture data.

    Generating one costs a few hundred milliseconds and it is identical for a given
    (finish, resolution, amplitude, seed), so results are cached and shared between
    materials and between scenes.
    """
    field = finish_field(finish, resolution, seed)
    # Scale so the peak lands at exactly 1.0, the most a uint8 texture can hold.
    field = (1.0 + amplitude * field) / roughness_factor_gain(amplitude)

    texture = np.clip(field * 255.0, 0.0, 255.0).astype(np.uint8)
    # Cached and shared between scenes, so hand back something nobody can mutate.
    texture.flags.writeable = False
    return texture


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur; enough to band-limit the upsampled noise."""
    if radius < 1:
        return image
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, image)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, blurred)


def _smudges(resolution: int, rng: np.random.Generator, count: int = 24) -> np.ndarray:
    """Soft elliptical blobs: handling marks that locally dull the reflection."""
    ys, xs = np.mgrid[0:resolution, 0:resolution].astype(np.float32)
    out = np.zeros((resolution, resolution), dtype=np.float32)
    for _ in range(count):
        cx, cy = rng.uniform(0, resolution, size=2)
        sx, sy = rng.uniform(0.02, 0.06, size=2) * resolution
        angle = rng.uniform(0.0, np.pi)
        dx, dy = xs - cx, ys - cy
        rx = dx * np.cos(angle) + dy * np.sin(angle)
        ry = -dx * np.sin(angle) + dy * np.cos(angle)
        out += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((rx / sx) ** 2 + (ry / sy) ** 2))
    return out


# --- real, authored maps shipped alongside the THOR textures ---------------------
#
# The THOR texture library ships genuine PBR maps that nothing references: 125 normal
# maps and 60 metallic/smoothness maps, following Unity's naming, where an albedo
# "X_AlbedoTransparency.png" is accompanied by "X_Normal.png" and
# "X_MetallicSmoothness.png". Matched on that stem, 195 of 2,251 distinct scene
# materials (8.7%) can be given a real normal map and 128 (5.7%) real metallic and
# roughness. Authored data beats anything inferred from an annotation, so where a map
# exists it wins and the procedural finish is only the fallback.

#: Suffixes that mark a texture file as the albedo of a PBR set.
_ALBEDO_SUFFIXES = ("_AlbedoTransparency", "_Albedo", "_BaseColor", "_seamless_albedo_S")


def albedo_stem(path: str) -> str:
    """The shared stem of a PBR set, from the filename of its albedo texture."""
    stem = Path(str(path)).stem
    for suffix in _ALBEDO_SUFFIXES:
        if stem.lower().endswith(suffix.lower()):
            return stem[: -len(suffix)].lower()
    return stem.lower()


@cache
def _authored_map_index() -> dict[str, dict[str, Path]]:
    """Index the PBR maps on disk by stem: ``{stem: {"normal": path, "orm": path}}``."""
    index: dict[str, dict[str, Path]] = {}
    root = ASSETS_DIR / "objects" / "thor"
    if not root.exists():
        return index

    patterns = (
        (re.compile(r"^(?P<stem>.*?)_Normal(_resized)?$", re.I), "normal"),
        (re.compile(r"^(?P<stem>.*?)_seamless_normal_S$", re.I), "normal"),
        (re.compile(r"^(?P<stem>.*?)_MetallicSmoothness$", re.I), "metallic_smoothness"),
    )
    for path in root.rglob("*"):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"):
            continue
        for pattern, kind in patterns:
            match = pattern.match(path.stem)
            if match:
                index.setdefault(match.group("stem").lower(), {}).setdefault(kind, path)
                break
    log.debug("indexed %d authored PBR map sets under %s", len(index), root)
    return index


@lru_cache(maxsize=256)
def _load_map(path: Path, resolution: int) -> np.ndarray | None:
    """Read a map file as HxWx4 uint8, or None if it cannot be read.

    Cached: the THOR library is shared, so the same handful of maps are wanted by
    every scene, and decoding plus resampling them dominates the cost otherwise.
    """
    try:
        from PIL import Image

        with Image.open(path) as handle:
            image = handle.convert("RGBA").resize((resolution, resolution), Image.LANCZOS)
        array = np.asarray(image, dtype=np.uint8)
        array.flags.writeable = False  # shared between callers
        return array
    except Exception as error:  # a missing or corrupt map must not fail a whole scene
        log.warning("could not read authored PBR map %s: %s", path, error)
        return None


def split_metallic_smoothness(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a Unity MetallicSmoothness map into separate metallic and roughness maps.

    Unity packs metallic in R and *smoothness* in A. Filament reads ``.x`` for both
    metallic and roughness, so this file cannot simply be bound to both roles -- the
    roughness slot would read the metallic channel, and smoothness is the complement
    of roughness anyway. Hence the repack: metallic is R, roughness is ``1 - A``.
    """
    metallic = image[..., 0]
    roughness = (255 - image[..., 3]).astype(np.uint8)
    return metallic, roughness


@dataclass
class PbrOverrideStats:
    """What :func:`apply_pbr_material_overrides` did, for logging and for tests."""

    total: int = 0
    skipped_authored: int = 0
    authored_normal: int = 0
    authored_metallic_roughness: int = 0
    procedural_normal: int = 0
    skipped_transparent: int = 0
    from_metadata: int = 0
    from_name: int = 0
    fallback: int = 0
    metals: int = 0
    finishes: dict[str, int] = field(default_factory=dict)

    @property
    def overridden(self) -> int:
        return self.from_metadata + self.from_name + self.fallback


# Texture roles that mean "this asset already has real PBR data, leave it alone".
_PBR_ROLES = (
    mj.mjtTextureRole.mjTEXROLE_ROUGHNESS,
    mj.mjtTextureRole.mjTEXROLE_METALLIC,
    mj.mjtTextureRole.mjTEXROLE_ORM,
    mj.mjtTextureRole.mjTEXROLE_NORMAL,
)


def _has_authored_pbr(material) -> bool:
    if material.metallic >= 0.0 or material.roughness >= 0.0:
        return True
    return any(material.textures[int(role)] for role in _PBR_ROLES)


def _is_transparent(material) -> bool:
    """Whether this material relies on blending, and so must stay on the phong path.

    The build ships blended "fade" variants of the phong materials
    (``phong_2d_fade``, ``phong_color_fade``, ...) but there is no faded PBR one --
    ``pbr.filamat`` is opaque, and its ``applyAlphaMask`` is an empty function. Give a
    window or a glass panel a metallic/roughness pair and it stops being see-through:
    measured on a 25%-alpha panel over a green backdrop, the backdrop went from
    [133, 163, 121] to a flat [120, 119, 111]. So translucent materials are left
    alone, and keep the legacy shading they were authored against.
    """
    return float(material.rgba[3]) < 1.0


# The shader reads only the red channel of the roughness and metallic maps, but the
# pre-4.0 custom Filament wheel's texture upload sized its KTX buffer for three
# channels and aborted with "KTX buffer is too small" on a single-channel texture, so
# the value is replicated across RGB. Three channels also work on MuJoCo 4.0 (whether
# one would is unchecked).
_TEXTURE_CHANNELS = 3


def _add_data_texture(spec, name: str, data: np.ndarray):
    """Add a 2D texture to the spec straight from a numpy array, HxW or HxWx3.

    Roughness and metallic are linear quantities, not colours, so the texture is
    tagged linear -- letting it default to sRGB would put a gamma curve through the
    map and shift every roughness value.
    """
    height, width = int(data.shape[0]), int(data.shape[1])
    if data.ndim == 2:
        rgb = np.repeat(data.reshape(height, width, 1), _TEXTURE_CHANNELS, axis=2)
    else:
        assert data.shape[2] == _TEXTURE_CHANNELS, data.shape
        rgb = data

    texture = spec.add_texture()
    texture.name = name
    texture.type = mj.mjtTexture.mjTEXTURE_2D
    texture.colorspace = mj.mjtColorSpace.mjCOLORSPACE_LINEAR
    texture.height, texture.width = height, width
    texture.nchannel = _TEXTURE_CHANNELS
    texture.data = rgb.tobytes()
    return texture


def apply_pbr_material_overrides(
    spec,
    *,
    annotations: dict | None = None,
    noise_resolution: int = 1024,
    noise_amplitude: float = 0.30,
    bump_strength: float = 0.06,
    use_authored_maps: bool = True,
    seed: int = 0,
) -> PbrOverrideStats:
    """Give every material in ``spec`` a metallic/roughness pair before compilation.

    Materials that already carry authored PBR data -- a metallic/roughness value or
    a roughness/metallic/ORM texture -- are left untouched; this only fills in the
    assets that have none, which today is very nearly all of them.

    Args:
        spec: an uncompiled ``mujoco.MjSpec``; mutated in place.
        annotations: objathor metadata keyed by object uid. Defaults to the shipped
            metadata via ``ObjectMeta.annotation()``; pass ``{}`` to classify from
            material names alone and skip loading it.
        noise_resolution: edge length of the shared roughness map, in pixels.
        noise_amplitude: peak relative deviation of the roughness map about 1.0.
        bump_strength: RMS surface slope of the procedural normal map, or 0 to bind
            none. This is by far the strongest visual knob here, so it is deliberately
            gentle: 0.10 across every material was tried and reads as mottled noise
            rather than texture, and past about 0.25 objects look hammered.
        use_authored_maps: prefer the real normal and metallic/smoothness maps that
            ship beside the THOR textures, where a material has them. Set False to
            make every material procedural, which is mainly useful for comparing.
        seed: seed for the roughness map, so a scene renders identically across runs.

    Returns:
        A :class:`PbrOverrideStats` describing what was classified and how.
    """
    materials = list(spec.materials)
    stats = PbrOverrideStats(total=len(materials))

    opaque = [material for material in materials if not _is_transparent(material)]
    stats.skipped_transparent = len(materials) - len(opaque)

    pending = [material for material in opaque if not _has_authored_pbr(material)]
    stats.skipped_authored = len(opaque) - len(pending)
    if not pending:
        return stats

    if annotations is None:
        from molmo_spaces.utils.object_metadata import ObjectMeta

        annotations = ObjectMeta.annotation()

    # The shader multiplies MetallicFactor by the metallic texture unconditionally.
    # We bind an explicit all-ones 1x1 map rather than trusting whatever MuJoCo binds
    # when the role is empty -- if that default were not white, every metal would
    # silently come out as a dielectric.
    _add_data_texture(spec, UNIT_TEXTURE, np.full((1, 1), 255, dtype=np.uint8))

    # Roughness maps live in UV space at full extent so they need no tiling (material
    # texrepeat applies to every layer at once, albedo included, so a tiled detail map
    # is not an option here). One texture is added per finish that actually gets used,
    # and every material that draws that finish shares it.
    added: dict[str, str] = {}

    def texture_for(finish: str) -> str:
        if finish not in added:
            name = f"{ROUGHNESS_TEXTURE_PREFIX}{finish}"
            _add_data_texture(
                spec, name, roughness_texture(finish, noise_resolution, noise_amplitude, seed)
            )
            added[finish] = name
        return added[finish]

    added_normals: dict[str, str] = {}

    # Albedo filenames, so a material can be matched to the authored maps beside it.
    albedo_files = {texture.name: texture.file for texture in spec.textures}
    authored = _authored_map_index() if use_authored_maps else {}
    authored_added: dict[tuple[str, str], str] = {}

    def authored_texture(stem: str, kind: str, data: np.ndarray) -> str:
        key = (stem, kind)
        if key not in authored_added:
            name = f"{AUTHORED_TEXTURE_PREFIX}{kind}_{stem}"[:90]
            _add_data_texture(spec, name, data)
            authored_added[key] = name
        return authored_added[key]

    def maps_for(material) -> dict[str, Path]:
        albedo = (
            material.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)]
            or material.textures[int(mj.mjtTextureRole.mjTEXROLE_RGBA)]
        )
        if not albedo:
            return {}
        source = albedo_files.get(albedo)
        return (authored.get(albedo_stem(source)) or {}) if source else {}

    def normal_for(finish: str) -> str:
        if finish not in added_normals:
            name = f"{NORMAL_TEXTURE_PREFIX}{finish}"
            _add_data_texture(
                spec, name, normal_texture(finish, noise_resolution, bump_strength, seed)
            )
            added_normals[finish] = name
        return added_normals[finish]

    # Seeded separately from the maps themselves, so which finish a material draws is
    # reproducible for a scene but independent of what the maps look like.
    chooser = random.Random(seed)
    gain = roughness_factor_gain(noise_amplitude)
    for material in pending:
        pbr_class, source = _classify(material.name, annotations)
        finish = chooser.choice(pbr_class.finishes)

        real = maps_for(material)

        material.metallic = pbr_class.metallic
        # min() guards the encoding: the gain is undone by the map's mean, but the
        # factor itself still has to be a legal [0, 1] roughness. Where it bites, the
        # effective roughness is compressed to 1/gain instead of the class value --
        # at the default amplitude that is anything above ~0.77, i.e. only the matte
        # end (fabric, plaster, brick), where roughness barely reads anyway. The
        # alternative, a differently-scaled map per roughness value, would multiply
        # the number of 1024-square textures for no visible gain.
        material.roughness = min(1.0, pbr_class.roughness * gain)
        material.textures[int(mj.mjtTextureRole.mjTEXROLE_ROUGHNESS)] = texture_for(finish)
        material.textures[int(mj.mjtTextureRole.mjTEXROLE_METALLIC)] = UNIT_TEXTURE

        # An authored metallic/smoothness map replaces both guesses outright: it says
        # per texel what the surface is, so the factors become plain 1.0 pass-throughs.
        packed = (
            _load_map(real["metallic_smoothness"], noise_resolution)
            if ("metallic_smoothness" in real)
            else None
        )
        if packed is not None:
            stem = albedo_stem(
                albedo_files.get(material.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)], "")
            )
            metallic_map, roughness_map = split_metallic_smoothness(packed)
            material.metallic = 1.0
            material.roughness = 1.0
            material.textures[int(mj.mjtTextureRole.mjTEXROLE_METALLIC)] = authored_texture(
                stem, "metallic", metallic_map
            )
            material.textures[int(mj.mjtTextureRole.mjTEXROLE_ROUGHNESS)] = authored_texture(
                stem, "roughness", roughness_map
            )
            stats.authored_metallic_roughness += 1

        # The procedural relief comes from the same finish as the roughness, so the two
        # stay in register rather than reading as unrelated patterns; an authored normal
        # map, where one exists, is better than either.
        #
        # Procedural relief goes on objects only. A roughness map is invisible on a
        # wall, but relief is not: our maps cover UV [0, 1] once with no tiling, so on
        # a large structural surface the bumps are stretched to the size of the wall
        # and read as blotchy mottling rather than texture. Objects are small and their
        # UV atlases dense, which is the case the maps are sized for. Authored maps are
        # exempt -- they were drawn for the surface they belong to.
        normal = _load_map(real["normal"], noise_resolution) if "normal" in real else None
        if normal is not None:
            stem = albedo_stem(
                albedo_files.get(material.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)], "")
            )
            material.textures[int(mj.mjtTextureRole.mjTEXROLE_NORMAL)] = authored_texture(
                stem, "normal", normal[..., :3]
            )
            stats.authored_normal += 1
        elif bump_strength > 0.0 and finish != "flat" and source == "from_metadata":
            material.textures[int(mj.mjtTextureRole.mjTEXROLE_NORMAL)] = normal_for(finish)
            stats.procedural_normal += 1

        setattr(stats, source, getattr(stats, source) + 1)
        stats.finishes[finish] = stats.finishes.get(finish, 0) + 1
        if pbr_class.metallic == 1.0:
            stats.metals += 1

    log.info(
        "PBR overrides: %d/%d materials (%d from metadata, %d from name, %d fallback, "
        "%d metal; %d real normal maps, %d real metallic/roughness, %d procedural "
        "relief); skipped %d already authored, %d translucent",
        stats.overridden,
        stats.total,
        stats.from_metadata,
        stats.from_name,
        stats.fallback,
        stats.metals,
        stats.authored_normal,
        stats.authored_metallic_roughness,
        stats.procedural_normal,
        stats.skipped_authored,
        stats.skipped_transparent,
    )
    log.debug("PBR finishes drawn: %s", dict(sorted(stats.finishes.items())))
    return stats


def _classify(name: str, annotations: dict) -> tuple[PbrClass, str]:
    """Classify one material by name, returning the class and which rule found it."""
    match = _UID_MATERIAL_RE.match(name or "")
    if match is not None:
        annotation = annotations.get(match.group("uid"))
        if annotation:
            return (
                classify_composition(
                    annotation.get("materials") or [], annotation.get("composition") or []
                ),
                "from_metadata",
            )

    by_name = classify_material_name(name or "")
    if by_name is not None:
        return by_name, "from_name"

    return UNKNOWN, "fallback"
