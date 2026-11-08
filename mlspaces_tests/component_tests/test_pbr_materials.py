"""Tests for the PBR material overrides applied to assets that ship without any."""

import mujoco as mj
import numpy as np
import pytest

from molmo_spaces.renderer.pbr_materials import (
    DEFAULT_FINISHES,
    MATERIAL_CLASSES,
    MAX_ROUGHNESS,
    MIN_ROUGHNESS,
    ROUGHNESS_FINISHES,
    ROUGHNESS_TEXTURE_PREFIX,
    UNIT_TEXTURE,
    UNKNOWN,
    albedo_stem,
    apply_pbr_material_overrides,
    classify_composition,
    classify_material_name,
    lookup_token,
    normal_texture,
    roughness_factor_gain,
    roughness_texture,
    split_metallic_smoothness,
)

_ROUGHNESS_ROLE = int(mj.mjtTextureRole.mjTEXROLE_ROUGHNESS)
_METALLIC_ROLE = int(mj.mjtTextureRole.mjTEXROLE_METALLIC)
_NORMAL_ROLE = int(mj.mjtTextureRole.mjTEXROLE_NORMAL)


def _spec(*material_names: str, authored: dict | None = None) -> mj.MjSpec:
    """A minimal spec with one geom per named material."""
    materials = "".join(f'<material name="{n}"/>' for n in material_names)
    geoms = "".join(
        f'<geom name="g{i}" type="box" size=".1 .1 .1" material="{n}"/>'
        for i, n in enumerate(material_names)
    )
    spec = mj.MjSpec.from_string(
        f"<mujoco><asset>{materials}</asset><worldbody>{geoms}</worldbody></mujoco>"
    )
    for name, values in (authored or {}).items():
        material = next(m for m in spec.materials if m.name == name)
        for key, value in values.items():
            setattr(material, key, value)
    return spec


class TestMaterialTable:
    def test_metallic_is_strictly_zero_or_one(self):
        """The dielectric/metallic rules admit no in-between values."""
        for name, pbr_class in MATERIAL_CLASSES.items():
            assert pbr_class.metallic in (0.0, 1.0), name

    def test_no_material_is_perfectly_smooth_or_perfectly_matte(self):
        for name, pbr_class in MATERIAL_CLASSES.items():
            assert 0.0 < pbr_class.roughness < 1.0, name
            assert MIN_ROUGHNESS <= pbr_class.roughness <= MAX_ROUGHNESS, name

    def test_known_metals_are_metallic(self):
        for name in ("metal", "steel", "brass", "bronze", "copper", "chrome"):
            assert MATERIAL_CLASSES[name].metallic == 1.0, name

    def test_known_dielectrics_are_not_metallic(self):
        for name in ("plastic", "wood", "fabric", "glass", "ceramic", "rubber", "paper"):
            assert MATERIAL_CLASSES[name].metallic == 0.0, name

    def test_fallback_is_a_dielectric(self):
        """Guessing plastic on an unknown surface is far cheaper than guessing metal."""
        assert UNKNOWN.metallic == 0.0


class TestClassification:
    def test_dominant_component_wins(self):
        assert classify_composition(["metal", "plastic"], [0.9, 0.1]).metallic == 1.0
        assert classify_composition(["metal", "plastic"], [0.1, 0.9]).metallic == 0.0

    def test_air_is_not_a_surface(self):
        """objathor uses "air" for hollow volume; it must never decide the shading."""
        assert classify_composition(["air", "metal"], [0.95, 0.05]).metallic == 1.0

    def test_paint_over_metal_reads_as_dielectric(self):
        """A painted tin is a dielectric at the surface, however much metal is under it."""
        painted = classify_composition(["metal", "paint"], [0.85, 0.15])
        assert painted.metallic == 0.0
        assert painted == MATERIAL_CLASSES["paint"]

    def test_a_trace_of_paint_does_not_hide_the_metal(self):
        assert classify_composition(["metal", "paint"], [0.98, 0.02]).metallic == 1.0

    def test_paint_over_plastic_is_left_alone(self):
        """The coat rule exists to veto metal; it should not restyle other dielectrics."""
        assert classify_composition(["plastic", "paint"], [0.8, 0.2]) == MATERIAL_CLASSES["plastic"]

    def test_compound_tokens_resolve_by_head_noun(self):
        assert lookup_token("synthetic fabric") == MATERIAL_CLASSES["fabric"]
        assert lookup_token("synthetic leather") == MATERIAL_CLASSES["leather"]
        assert lookup_token("stainless steel") == MATERIAL_CLASSES["stainless steel"]

    def test_unknown_token_is_unresolved(self):
        assert lookup_token("unobtainium") is None

    def test_missing_annotation_falls_back(self):
        assert classify_composition([], []) == UNKNOWN
        assert classify_composition(["air"], [1.0]) == UNKNOWN

    def test_mismatched_composition_is_tolerated(self):
        """Never trust the annotation's shape; a bad length must not raise."""
        assert classify_composition(["metal", "plastic"], [1.0]).metallic in (0.0, 1.0)

    def test_thor_material_names(self):
        assert classify_material_name("material_GenericStainlessSteel").metallic == 1.0
        assert classify_material_name("material_WhitePlastic") == MATERIAL_CLASSES["plastic"]
        assert classify_material_name("DarkWood2") == MATERIAL_CLASSES["wood"]
        assert classify_material_name("Porcelain_Black_Mat") == MATERIAL_CLASSES["porcelain"]

    def test_longer_material_names_win(self):
        """ "stainlesssteel" must not be decided by the "steel" rule."""
        assert classify_material_name("StainlessSteel") == MATERIAL_CLASSES["stainless steel"]

    def test_uninformative_names_are_unclassified(self):
        assert classify_material_name("material_Fridge_Readout_Mat") is None
        assert classify_material_name("PureWhite") is None


FINISH = "smudged"


class TestRoughnessNoise:
    def test_is_centred_on_one_once_the_gain_is_undone(self):
        """The shader multiplies this into the roughness factor, so it must average 1."""
        for amplitude in (0.04, 0.08, 0.15):
            texture = roughness_texture(FINISH, 256, amplitude=amplitude, seed=0) / 255.0
            mean = np.mean(texture) * roughness_factor_gain(amplitude)
            assert mean == pytest.approx(1.0, abs=0.01), amplitude

    def test_fits_in_the_uint8_range(self):
        """A map conceptually centred on 1.0 would lose its whole upper half to uint8."""
        for amplitude in (0.04, 0.08, 0.15):
            texture = roughness_texture(FINISH, 256, amplitude=amplitude, seed=0)
            assert texture.max() == 255, amplitude
            # A small plateau at the peak is just uint8 quantisation of a smooth
            # field, not clipping; what matters is that it is nowhere near flat.
            assert (texture == 255).mean() < 0.05, amplitude

    def test_respects_the_amplitude(self):
        amplitude = 0.08
        texture = roughness_texture(FINISH, 256, amplitude=amplitude, seed=0) / 255.0
        effective = texture * roughness_factor_gain(amplitude)
        assert effective.min() >= 1.0 - amplitude - 1e-2
        assert effective.max() <= 1.0 + amplitude + 1e-2

    def test_actually_varies(self):
        """A constant map would defeat the entire point of having one."""
        assert np.std(roughness_texture(FINISH, 256, seed=0) / 255.0) > 0.005

    def test_is_deterministic_for_a_seed(self):
        """Repeat renders of a scene must be identical."""
        assert np.array_equal(
            roughness_texture(FINISH, 256, seed=3), roughness_texture(FINISH, 256, seed=3)
        )
        assert not np.array_equal(
            roughness_texture(FINISH, 256, seed=3), roughness_texture(FINISH, 256, seed=4)
        )

    def test_product_stays_in_range(self):
        """factor * noise must stay a valid roughness for every class in the table."""
        amplitude = 0.08
        texture = roughness_texture(FINISH, 256, amplitude=amplitude, seed=0) / 255.0
        for name, pbr_class in MATERIAL_CLASSES.items():
            factor = min(1.0, pbr_class.roughness * roughness_factor_gain(amplitude))
            product = factor * texture
            assert product.min() > 0.0, name
            assert product.max() < 1.0, name
            assert product.mean() == pytest.approx(pbr_class.roughness, abs=0.02), name

    def test_cached_map_cannot_be_mutated(self):
        with pytest.raises(ValueError):
            roughness_texture(FINISH, 256, seed=0)[0, 0] = 7


class TestApplyOverrides:
    def test_sets_metallic_and_roughness_on_every_bare_material(self):
        spec = _spec("material_Metal", "material_WhitePlastic", "PureWhite")
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.total == 3
        assert stats.overridden == 3

        model = spec.compile()
        assert not (model.mat_metallic < 0).any()
        assert not (model.mat_roughness < 0).any()

    def test_binds_both_pbr_texture_roles(self):
        """The shader multiplies each factor by its texture, so both must be bound."""
        spec = _spec("material_Metal")
        apply_pbr_material_overrides(spec, annotations={})
        model = spec.compile()
        assert model.mat_texid[0, _ROUGHNESS_ROLE] >= 0
        assert model.mat_texid[0, _METALLIC_ROLE] >= 0

    def test_metallic_map_is_all_ones(self):
        """Metal must survive the multiply whatever MuJoCo binds by default."""
        spec = _spec("material_Metal")
        apply_pbr_material_overrides(spec, annotations={})
        model = spec.compile()
        texture_id = model.mat_texid[0, _METALLIC_ROLE]
        start = model.tex_adr[texture_id]
        size = (
            model.tex_height[texture_id]
            * model.tex_width[texture_id]
            * model.tex_nchannel[texture_id]
        )
        assert (model.tex_data[start : start + size] == 255).all()

    def test_scalar_maps_are_single_channel(self):
        """The shader reads only the red channel, so roughness/metallic need no RGB."""
        spec = _spec("material_Metal")
        apply_pbr_material_overrides(spec, annotations={})
        model = spec.compile()
        for role in (_ROUGHNESS_ROLE, _METALLIC_ROLE):
            assert model.tex_nchannel[model.mat_texid[0, role]] == 1

    def test_maps_are_linear_not_srgb(self):
        """Roughness is not a colour; an sRGB tag would bend every value."""
        spec = _spec("material_Metal")
        apply_pbr_material_overrides(spec, annotations={})
        texture = next(t for t in spec.textures if t.name.startswith(ROUGHNESS_TEXTURE_PREFIX))
        assert texture.colorspace == mj.mjtColorSpace.mjCOLORSPACE_LINEAR

    def test_metal_is_classified_from_the_name(self):
        spec = _spec("material_GenericStainlessSteel", "material_WhitePlastic")
        stats = apply_pbr_material_overrides(spec, annotations={})
        model = spec.compile()
        assert stats.metals == 1
        assert sorted(model.mat_metallic) == [0.0, 1.0]

    def test_uid_materials_use_the_annotation(self):
        uid = "0" * 32
        spec = _spec(f"{uid}_mat")
        annotations = {uid: {"materials": ["metal", "air"], "composition": [0.8, 0.2]}}
        stats = apply_pbr_material_overrides(spec, annotations=annotations)
        assert stats.from_metadata == 1
        assert spec.compile().mat_metallic[0] == 1.0

    def test_a_uid_missing_from_the_annotations_still_gets_a_material(self):
        spec = _spec(f"{'a' * 32}_mat")
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.fallback == 1
        assert spec.compile().mat_metallic[0] == UNKNOWN.metallic

    def test_authored_pbr_is_left_alone(self):
        """Assets that do carry real PBR data must not be overwritten by our guesses."""
        spec = _spec("good", "bare", authored={"good": {"metallic": 1.0, "roughness": 0.2}})
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.skipped_authored == 1
        assert stats.overridden == 1

        model = spec.compile()
        good = next(
            i
            for i in range(model.nmat)
            if mj.mj_id2name(model, mj.mjtObj.mjOBJ_MATERIAL, i) == "good"
        )
        assert model.mat_metallic[good] == 1.0
        assert model.mat_roughness[good] == pytest.approx(0.2)
        assert model.mat_texid[good, _ROUGHNESS_ROLE] == -1

    def test_textures_are_shared_not_duplicated_per_material(self):
        spec = _spec(*[f"m{i}" for i in range(20)])
        apply_pbr_material_overrides(spec, annotations={})
        names = [t.name for t in spec.textures]
        assert len(names) == len(set(names)), "a finish must not be added twice"
        assert names.count(UNIT_TEXTURE) == 1
        rough = [n for n in names if n.startswith(ROUGHNESS_TEXTURE_PREFIX)]
        assert 0 < len(rough) <= len(ROUGHNESS_FINISHES)

    def test_transparent_materials_are_left_on_the_legacy_path(self):
        """There is no faded PBR material, so overriding a window makes it opaque."""
        spec = _spec("glass", "wall", authored={"glass": {"rgba": [0.8, 0.9, 1.0, 0.25]}})
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.skipped_transparent == 1
        assert stats.overridden == 1

        model = spec.compile()
        glass = next(
            i
            for i in range(model.nmat)
            if mj.mj_id2name(model, mj.mjtObj.mjOBJ_MATERIAL, i) == "glass"
        )
        assert model.mat_metallic[glass] == -1.0
        assert model.mat_texid[glass, _ROUGHNESS_ROLE] == -1

    def test_fully_opaque_materials_are_not_treated_as_transparent(self):
        spec = _spec("wall", authored={"wall": {"rgba": [0.5, 0.5, 0.5, 1.0]}})
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.skipped_transparent == 0
        assert stats.overridden == 1

    def test_a_spec_with_nothing_to_do_adds_no_textures(self):
        spec = _spec("good", authored={"good": {"metallic": 0.0, "roughness": 0.5}})
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.overridden == 0
        assert [t.name for t in spec.textures] == []


class TestFinishLibrary:
    def test_every_class_names_known_finishes(self):
        for name, pbr_class in MATERIAL_CLASSES.items():
            assert pbr_class.finishes, name
            for finish in pbr_class.finishes:
                assert finish in ROUGHNESS_FINISHES, f"{name} -> {finish}"

    def test_default_finishes_are_known(self):
        for finish in DEFAULT_FINISHES:
            assert finish in ROUGHNESS_FINISHES

    @pytest.mark.parametrize("finish", sorted(ROUGHNESS_FINISHES))
    def test_each_finish_is_a_valid_multiplier(self, finish):
        """Whatever its character, every map must keep the mean-1.0 contract."""
        amplitude = 0.08
        texture = roughness_texture(finish, 256, amplitude, 0) / 255.0
        assert np.mean(texture) * roughness_factor_gain(amplitude) == pytest.approx(1.0, abs=0.01)
        assert texture.max() <= 1.0
        assert texture.min() > 0.0

    @pytest.mark.parametrize("finish", sorted(set(ROUGHNESS_FINISHES) - {"flat"}))
    def test_each_finish_actually_varies(self, finish):
        assert np.std(roughness_texture(finish, 256, 0.08, 0) / 255.0) > 0.005

    def test_flat_is_flat(self):
        texture = roughness_texture("flat", 64, 0.08, 0)
        assert texture.min() == texture.max()

    def test_finishes_differ_from_one_another(self):
        maps = {f: roughness_texture(f, 128, 0.08, 0) for f in ROUGHNESS_FINISHES}
        for a in maps:
            for b in maps:
                if a < b:
                    assert not np.array_equal(maps[a], maps[b]), f"{a} == {b}"

    def test_brushed_is_directional(self):
        """A brushed finish must vary across the streaks far more than along them."""
        texture = roughness_texture("brushed", 256, 0.08, 0).astype(float)
        along = np.abs(np.diff(texture, axis=1)).mean()
        across = np.abs(np.diff(texture, axis=0)).mean()
        assert across > 2 * along, (across, along)

    def test_unknown_finish_is_rejected(self):
        with pytest.raises(KeyError):
            roughness_texture("holographic", 64)

    def test_materials_draw_a_mix_of_finishes(self):
        spec = _spec(*[f"m{i}" for i in range(60)])
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.overridden == 60
        assert sum(stats.finishes.values()) == 60
        assert len(stats.finishes) > 1, "every material drew the same finish"

    def test_finish_choice_is_reproducible(self):
        def run():
            spec = _spec(*[f"m{i}" for i in range(30)])
            apply_pbr_material_overrides(spec, annotations={}, seed=7)
            model = spec.compile()
            return model.mat_texid[:, _ROUGHNESS_ROLE].tolist()

        assert run() == run()

    def test_seed_changes_the_draw(self):
        def run(seed):
            spec = _spec(*[f"m{i}" for i in range(40)])
            apply_pbr_material_overrides(spec, annotations={}, seed=seed)
            return [t.name for t in spec.textures]

        assert run(0) != run(3)

    def test_a_class_only_draws_its_own_finishes(self):
        """Glass must never come back brushed."""
        spec = _spec(*[f"material_Glass_{i}" for i in range(20)])
        stats = apply_pbr_material_overrides(spec, annotations={}, seed=1)
        assert set(stats.finishes) <= set(MATERIAL_CLASSES["glass"].finishes)


class TestEffectiveRoughness:
    """The factor times the map's mean is what the shader actually shades with."""

    AMPLITUDE = 0.30

    def _effective(self, pbr_class):
        gain = roughness_factor_gain(self.AMPLITUDE)
        factor = min(1.0, pbr_class.roughness * gain)
        texture = roughness_texture(pbr_class.finishes[0], 256, self.AMPLITUDE, 0) / 255.0
        return factor * float(np.mean(texture))

    def test_shiny_materials_keep_their_roughness(self):
        """Below the clamp the class value must survive the encoding intact."""
        for name, pbr_class in MATERIAL_CLASSES.items():
            if pbr_class.roughness > 0.75:
                continue
            assert self._effective(pbr_class) == pytest.approx(pbr_class.roughness, abs=0.02), name

    def test_matte_materials_are_compressed_but_stay_matte(self):
        """Above the clamp roughness is pulled down to 1/gain -- documented, bounded."""
        floor = 1.0 / roughness_factor_gain(self.AMPLITUDE)
        for name, pbr_class in MATERIAL_CLASSES.items():
            if pbr_class.roughness <= 0.75:
                continue
            effective = self._effective(pbr_class)
            assert effective == pytest.approx(floor, abs=0.02), name
            assert effective > 0.7, name

    def test_roughness_never_leaves_the_legal_range(self):
        texture = roughness_texture("speckle", 256, self.AMPLITUDE, 0) / 255.0
        gain = roughness_factor_gain(self.AMPLITUDE)
        for name, pbr_class in MATERIAL_CLASSES.items():
            product = min(1.0, pbr_class.roughness * gain) * texture
            assert product.min() > 0.0, name
            assert product.max() <= 1.0, name


class TestNormalMaps:
    @pytest.mark.parametrize("finish", sorted(ROUGHNESS_FINISHES))
    def test_encoding_is_a_unit_normal(self, finish):
        """Every texel must decode to a unit vector pointing out of the surface."""
        texture = normal_texture(finish, 128, 0.10, 0)
        assert texture.shape == (128, 128, 3)
        vec = texture.astype(float) / 255.0 * 2.0 - 1.0
        assert np.allclose(np.linalg.norm(vec, axis=-1), 1.0, atol=0.02)
        assert (vec[..., 2] > 0).all(), "normals must not point into the surface"

    def test_strength_is_the_rms_slope_and_resolution_independent(self):
        for resolution in (128, 256):
            for strength in (0.05, 0.10, 0.30):
                vec = normal_texture("speckle", resolution, strength, 0).astype(float)
                vec = vec / 255.0 * 2.0 - 1.0
                slope = np.sqrt(
                    np.mean((vec[..., 0] / vec[..., 2]) ** 2 + (vec[..., 1] / vec[..., 2]) ** 2)
                )
                assert slope == pytest.approx(strength, rel=0.15), (resolution, strength)

    def test_flat_finish_has_no_relief(self):
        texture = normal_texture("flat", 64, 0.10, 0)
        assert (texture[..., 2] == 255).all()

    def test_relief_shares_the_finish_field_with_roughness(self):
        """The two maps must describe one texture, so they have to stay in register."""
        rough = roughness_texture("weave", 128, 0.30, 0).astype(float)
        normal = normal_texture("weave", 128, 0.10, 0).astype(float)
        # where the height field is steepest the roughness map is changing fastest
        rough_grad = np.abs(np.gradient(rough, axis=0)) + np.abs(np.gradient(rough, axis=1))
        tilt = np.abs(normal[..., 0] - 127.5) + np.abs(normal[..., 1] - 127.5)
        corr = np.corrcoef(rough_grad.ravel(), tilt.ravel())[0, 1]
        assert corr > 0.5, corr

    def test_bound_to_object_materials(self):
        uids = [f"{i:032x}" for i in range(6)]
        spec = _spec(*[f"{uid}_mat" for uid in uids])
        annotations = {uid: {"materials": ["plastic"], "composition": [1.0]} for uid in uids}
        stats = apply_pbr_material_overrides(spec, annotations=annotations, use_authored_maps=False)
        assert stats.procedural_normal == 6
        assert (spec.compile().mat_texid[:, _NORMAL_ROLE] >= 0).all()

    def test_not_bound_to_structural_materials(self):
        """Our maps span UV [0, 1] once, so on a wall the bumps stretch to wall size."""
        spec = _spec("material_WhitePlastic", "DarkWood2", "PureWhite")
        stats = apply_pbr_material_overrides(spec, annotations={}, use_authored_maps=False)
        assert stats.overridden == 3
        assert stats.procedural_normal == 0
        assert (spec.compile().mat_texid[:, _NORMAL_ROLE] == -1).all()

    def test_can_be_switched_off(self):
        uid = "f" * 32
        spec = _spec(f"{uid}_mat")
        annotations = {uid: {"materials": ["plastic"], "composition": [1.0]}}
        apply_pbr_material_overrides(
            spec, annotations=annotations, bump_strength=0.0, use_authored_maps=False
        )
        assert (spec.compile().mat_texid[:, _NORMAL_ROLE] == -1).all()

    def test_authored_normal_map_is_left_alone(self):
        """A material that already has relief must not have ours layered over it."""
        spec = _spec("m")
        spec.materials[0].textures[_NORMAL_ROLE] = "whatever"
        stats = apply_pbr_material_overrides(spec, annotations={})
        assert stats.skipped_authored == 1


class TestAuthoredMaps:
    def test_albedo_stem_strips_the_known_suffixes(self):
        assert albedo_stem("BrushedIron_AlbedoTransparency.png") == "brushediron"
        assert albedo_stem("a/b/Book_Pages_Albedo.png") == "book_pages"
        assert albedo_stem("TexturesCom_Wood0044_seamless_albedo_S.png") == "texturescom_wood0044"

    def test_albedo_stem_leaves_plain_names_alone(self):
        assert albedo_stem("Carpet1.png") == "carpet1"

    def test_metallic_smoothness_unpacks_to_metallic_and_roughness(self):
        """Unity packs metallic in R and smoothness -- not roughness -- in A."""
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        image[..., 0] = 255  # fully metallic
        image[..., 3] = 200  # very smooth
        metallic, roughness = split_metallic_smoothness(image)
        assert (metallic == 255).all()
        assert (roughness == 55).all()

    def test_smooth_becomes_low_roughness(self):
        image = np.zeros((1, 1, 4), dtype=np.uint8)
        image[..., 3] = 255
        _, roughness = split_metallic_smoothness(image)
        assert roughness[0, 0] == 0
