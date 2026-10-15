"""Unit tests for the pure geometry/interpolation helpers in
utils/eval_camera_randomization_utils.py.

These map a randomization "level" in [0, 100] to concrete camera parameter
values and perturb a reference camera pose in spherical coordinates. None of
the functions tested here touch MuJoCo or a live env -- `apply_camera_perturbation`
only needs a camera config object (a SimpleNamespace stands in for the real
pydantic EvalExocentricCameraConfig, since the function only reads attributes
off it) and a seeded RandomState.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from molmo_spaces.utils.eval_camera_randomization_utils import (
    _decompose_to_spherical,
    _flatten_value,
    _reshape_to_original,
    apply_camera_perturbation,
    derive_episode_camera_seed,
    piecewise_linear,
)

# --- _flatten_value -----------------------------------------------------


def test_flatten_value_scalar():
    assert _flatten_value(3.0) == [3.0]
    assert _flatten_value(2) == [2.0]


def test_flatten_value_flat_tuple():
    assert _flatten_value((1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]


def test_flatten_value_nested_tuple():
    # e.g. pos_noise_range = ((-0.015, -0.005, -0.02), (0.015, 0.005, 0.02))
    nested = ((-0.015, -0.005, -0.02), (0.015, 0.005, 0.02))
    assert _flatten_value(nested) == [-0.015, -0.005, -0.02, 0.015, 0.005, 0.02]


# --- _reshape_to_original -------------------------------------------------


def test_reshape_to_original_passes_through_scalars():
    assert _reshape_to_original(3.0, 5.0) == 3.0


def test_reshape_to_original_passes_through_plain_tuple():
    # original_val is a flat tuple (not a pair-of-tuples) -> no reshape needed.
    flat = (1.0, 2.0, 3.0)
    assert _reshape_to_original(flat, (0.0, 0.0, 0.0)) == flat


def test_reshape_to_original_splits_paired_ranges():
    original = ((-0.015, -0.005, -0.02), (0.015, 0.005, 0.02))
    flat = (-0.01, -0.004, -0.01, 0.01, 0.004, 0.01)
    lo, hi = _reshape_to_original(flat, original)
    assert lo == (-0.01, -0.004, -0.01)
    assert hi == (0.01, 0.004, 0.01)


# --- piecewise_linear -----------------------------------------------------


def test_piecewise_linear_clamps_below_first_breakpoint():
    assert piecewise_linear(-10, [0, 50, 100], [1.0, 2.0, 3.0]) == 1.0


def test_piecewise_linear_clamps_above_last_breakpoint():
    assert piecewise_linear(200, [0, 50, 100], [1.0, 2.0, 3.0]) == 3.0


def test_piecewise_linear_interpolates_midpoint():
    assert piecewise_linear(25, [0, 50, 100], [0.0, 2.0, 4.0]) == pytest.approx(1.0)


def test_piecewise_linear_hits_exact_breakpoints():
    breakpoints, values = [0, 50, 100], [1.0, 5.0, 9.0]
    for bp, val in zip(breakpoints, values):
        assert piecewise_linear(bp, breakpoints, values) == pytest.approx(val)


# --- derive_episode_camera_seed -------------------------------------------


def test_derive_episode_camera_seed_is_deterministic():
    episode = SimpleNamespace(scene_dataset="procthor-10k", data_split="val", house_index=3)
    seed_a = derive_episode_camera_seed(episode)
    seed_b = derive_episode_camera_seed(episode)
    assert seed_a == seed_b
    assert isinstance(seed_a, int)


def test_derive_episode_camera_seed_differs_across_episodes():
    ep1 = SimpleNamespace(scene_dataset="procthor-10k", data_split="val", house_index=3)
    ep2 = SimpleNamespace(scene_dataset="procthor-10k", data_split="val", house_index=4)
    assert derive_episode_camera_seed(ep1) != derive_episode_camera_seed(ep2)


def test_derive_episode_camera_seed_uses_source_and_seed_fields_when_present():
    base = SimpleNamespace(scene_dataset="procthor-10k", data_split="val", house_index=3)
    with_source = SimpleNamespace(
        **vars(base), source=SimpleNamespace(h5_file="a.h5", traj_key="traj_0")
    )
    with_seed = SimpleNamespace(**vars(base), seed=42)
    # Both extra fields should participate in the hash, so all three differ.
    seeds = {
        derive_episode_camera_seed(base),
        derive_episode_camera_seed(with_source),
        derive_episode_camera_seed(with_seed),
    }
    assert len(seeds) == 3


# --- _decompose_to_spherical -----------------------------------------------


def test_decompose_to_spherical_along_x_axis():
    camera_pos = np.array([3.0, 0.0, 1.0])
    workspace_center = np.array([0.0, 0.0, 0.0])
    azimuth, distance, height = _decompose_to_spherical(camera_pos, workspace_center)
    assert azimuth == pytest.approx(0.0)
    assert distance == pytest.approx(3.0)
    assert height == pytest.approx(1.0)


def test_decompose_to_spherical_along_y_axis():
    camera_pos = np.array([0.0, 2.0, 0.0])
    workspace_center = np.array([0.0, 0.0, 0.0])
    azimuth, distance, height = _decompose_to_spherical(camera_pos, workspace_center)
    assert azimuth == pytest.approx(np.pi / 2)
    assert distance == pytest.approx(2.0)
    assert height == pytest.approx(0.0)


def test_decompose_to_spherical_relative_to_nonzero_center():
    camera_pos = np.array([5.0, 5.0, 5.0])
    workspace_center = np.array([5.0, 5.0, 2.0])
    azimuth, distance, height = _decompose_to_spherical(camera_pos, workspace_center)
    assert distance == pytest.approx(0.0)
    assert height == pytest.approx(3.0)


# --- apply_camera_perturbation ----------------------------------------------


def _make_cam(**overrides):
    defaults = dict(
        pos=(2.0, 0.0, 1.0),
        azimuth_range=None,
        distance_range=None,
        height_range=None,
        workspace_center_weight=0.0,
        lookat_noise_range=None,
        fov=60.0,
        fov_range=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_apply_camera_perturbation_no_noise_reconstructs_reference_pose():
    """With every *_range unset, the perturbed pose should equal the reference
    pose reconstructed from its own spherical decomposition (a round trip)."""
    cam = _make_cam()
    ref_forward = np.array([-1.0, 0.0, 0.0])
    ref_up = np.array([0.0, 0.0, 1.0])
    workspace_center = np.array([0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)

    pos, forward, up, fov = apply_camera_perturbation(
        cam, ref_forward, ref_up, workspace_center, rng
    )

    np.testing.assert_allclose(pos, np.array(cam.pos, dtype=np.float32), atol=1e-5)
    assert fov == pytest.approx(cam.fov)
    assert np.linalg.norm(forward) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-5)


def test_apply_camera_perturbation_is_deterministic_given_a_seed():
    cam = _make_cam(azimuth_range=(-0.2, 0.2), distance_range=(-0.1, 0.1), fov_range=(50, 70))
    ref_forward = np.array([-1.0, 0.0, 0.0])
    ref_up = np.array([0.0, 0.0, 1.0])
    workspace_center = np.array([0.0, 0.0, 0.0])

    result_a = apply_camera_perturbation(
        cam, ref_forward, ref_up, workspace_center, np.random.RandomState(7)
    )
    result_b = apply_camera_perturbation(
        cam, ref_forward, ref_up, workspace_center, np.random.RandomState(7)
    )
    for a, b in zip(result_a, result_b):
        np.testing.assert_allclose(a, b)


def test_apply_camera_perturbation_lookat_weight_one_points_at_workspace_center():
    cam = _make_cam(workspace_center_weight=1.0)
    ref_forward = np.array([-1.0, 0.0, 0.0])
    ref_up = np.array([0.0, 0.0, 1.0])
    workspace_center = np.array([0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)

    pos, forward, up, _fov = apply_camera_perturbation(
        cam, ref_forward, ref_up, workspace_center, rng
    )
    # At full lookat weight the camera should face directly toward the
    # workspace center from its (unperturbed) position.
    expected_dir = workspace_center - pos.astype(np.float64)
    expected_dir /= np.linalg.norm(expected_dir)
    np.testing.assert_allclose(forward, expected_dir, atol=1e-4)


def test_apply_camera_perturbation_distance_floor_is_enforced():
    # A distance_range that would push the camera to (or past) the workspace
    # center should be clamped to the 0.10m floor, not go to zero/negative.
    cam = _make_cam(pos=(0.15, 0.0, 0.0), distance_range=(-1.0, -1.0))
    ref_forward = np.array([-1.0, 0.0, 0.0])
    ref_up = np.array([0.0, 0.0, 1.0])
    workspace_center = np.array([0.0, 0.0, 0.0])
    rng = np.random.RandomState(0)

    pos, _forward, _up, _fov = apply_camera_perturbation(
        cam, ref_forward, ref_up, workspace_center, rng
    )
    distance = np.linalg.norm(pos[:2] - workspace_center[:2])
    assert distance == pytest.approx(0.10, abs=1e-5)
