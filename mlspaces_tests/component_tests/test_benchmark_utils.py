"""Unit tests for the pure/near-pure helpers in utils/benchmark_utils.py.

benchmark_utils.py's outlier-detection entry point
(episodes_with_kinematics_outliers) fans out over a directory of H5 files via
multiprocessing.Pool. This file instead calls the per-file worker functions
directly with a small synthetic H5 trajectory (built with the same
dict_to_byte_array encoding save_utils.py uses), and unit-tests the smaller
statistics/parsing helpers those workers depend on.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from molmo_spaces.utils.benchmark_utils import (
    _check_bounds_and_append,
    _collect_stats_worker,
    _find_outliers_worker,
    _safe_decode_json_array,
    _sigma_clip,
    compute_bounds_std,
    resolve_asset_id,
)
from molmo_spaces.utils.save_utils import dict_to_byte_array

STR_MAX_LEN = 256


# --- resolve_asset_id -------------------------------------------------------


def test_resolve_asset_id_from_added_objects_dict():
    task_config = {"added_objects": {"place_receptacle/Bowl_25": "objaverse/abc123.xml"}}
    assert resolve_asset_id("place_receptacle/Bowl_25", task_config) == "abc123"


def test_resolve_asset_id_from_added_objects_attr():
    # Non-dict task configs (e.g. a real PickAndPlaceTaskConfig) expose
    # added_objects as an attribute instead.
    task_config = SimpleNamespace(added_objects={"Bowl_25": "objaverse/xyz789.xml"})
    assert resolve_asset_id("Bowl_25", task_config) == "xyz789"


def test_resolve_asset_id_returns_none_when_unresolvable():
    # No added_objects entry and no scene_dataset/data_split/house_index to
    # fall back to the scene-metadata lookup -> can't resolve.
    task_config = SimpleNamespace(added_objects={})
    assert resolve_asset_id("Mug_12", task_config) is None


# --- compute_bounds_std -----------------------------------------------------


def test_compute_bounds_std_computes_mean_std_bounds():
    # M2 is the running sum of squared deviations (Welford's algorithm), so
    # variance = M2 / count.
    stats = {
        ("reltrack_arm", 0): {"count": 20, "mean": 1.0, "M2": 20 * 4.0},  # variance=4, std=2
    }
    bounds = compute_bounds_std(stats, std_mult=2.0)
    lower, upper, mean, std = bounds[("reltrack_arm", 0)]
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(2.0)
    assert lower == pytest.approx(1.0 - 2.0 * 2.0)
    assert upper == pytest.approx(1.0 + 2.0 * 2.0)


def test_compute_bounds_std_skips_low_count_keys():
    stats = {("reltrack_arm", 0): {"count": 9, "mean": 0.0, "M2": 0.0}}
    assert compute_bounds_std(stats, std_mult=4.0) == {}


# --- _sigma_clip -------------------------------------------------------------


def test_sigma_clip_matches_plain_mean_std_without_outliers():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=0.0, scale=1.0, size=2000)
    mu, sigma, n_kept = _sigma_clip(values, clip_sigma=6.0)
    assert mu == pytest.approx(np.mean(values), abs=0.05)
    assert sigma == pytest.approx(np.std(values), abs=0.05)
    assert n_kept == len(values)  # nothing clipped at 6 sigma for a clean normal sample


def test_sigma_clip_removes_extreme_outliers():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=0.0, scale=1.0, size=1000)
    values = np.concatenate([values, [1000.0, -1000.0]])
    mu, sigma, n_kept = _sigma_clip(values, clip_sigma=4.0)
    assert n_kept < len(values)
    assert abs(mu) < 1.0  # unaffected by the outliers once clipped


def test_sigma_clip_empty_input():
    assert _sigma_clip(np.array([])) == (0.0, 0.0, 0)


# --- _safe_decode_json_array -------------------------------------------------


def _make_json_array_dataset(group, name, dicts, str_max_len=STR_MAX_LEN):
    arr = np.stack([dict_to_byte_array(d, name, str_max_len) for d in dicts])
    return group.create_dataset(name, data=arr)


def test_safe_decode_json_array_round_trips(tmp_path):
    dicts = [{"arm": [0.1, 0.2]}, {"arm": [0.3, 0.4]}]
    h5_path = tmp_path / "t.h5"
    with h5py.File(h5_path, "w") as f:
        ds = _make_json_array_dataset(f, "qpos", dicts)
        decoded = _safe_decode_json_array(ds)
    assert decoded == dicts


def test_safe_decode_json_array_bad_json_becomes_empty_dict(tmp_path):
    h5_path = tmp_path / "t.h5"
    with h5py.File(h5_path, "w") as f:
        arr = np.zeros((1, STR_MAX_LEN), dtype=np.uint8)
        bad = b"{not valid json"
        arr[0, : len(bad)] = list(bad)
        ds = f.create_dataset("qpos", data=arr)
        decoded = _safe_decode_json_array(ds)
    assert decoded == [{}]


def test_safe_decode_json_array_returns_none_for_numeric_dataset(tmp_path):
    # Non-JSON-encoded (plain numeric) datasets should fail decoding and
    # return None rather than raising or returning garbage. Values must be
    # non-zero -- all-zero bytes still decode as valid (empty) UTF-8, so this
    # wouldn't exercise the failure path the function guards against.
    h5_path = tmp_path / "t.h5"
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("qvel", data=np.full((3, 3), np.pi, dtype=np.float32))
        decoded = _safe_decode_json_array(ds)
    assert decoded is None


# --- _check_bounds_and_append ------------------------------------------------


def test_check_bounds_and_append_flags_out_of_range_value():
    outliers = []
    bounds = {("reltrack_arm", 0): (-1.0, 1.0, 0.0, 0.5)}
    _check_bounds_and_append(outliers, bounds, "house_0", "traj.h5", 0, 5, "reltrack_arm", 0, 3.0)
    assert len(outliers) == 1
    entry = outliers[0]
    assert entry["value"] == 3.0
    assert entry["std_away"] == pytest.approx(6.0)  # |3.0 - 0.0| / 0.5


def test_check_bounds_and_append_ignores_in_range_value():
    outliers = []
    bounds = {("reltrack_arm", 0): (-1.0, 1.0, 0.0, 0.5)}
    _check_bounds_and_append(outliers, bounds, "house_0", "traj.h5", 0, 5, "reltrack_arm", 0, 0.5)
    assert outliers == []


def test_check_bounds_and_append_lower_only_ignores_high_value():
    outliers = []
    bounds = {("reltrack_arm", 0): (-1.0, 1.0, 0.0, 0.5)}
    _check_bounds_and_append(
        outliers, bounds, "house_0", "traj.h5", 0, 5, "reltrack_arm", 0, 5.0, lower_only=True
    )
    assert outliers == []  # above upper bound, but lower_only means it's accepted


def test_check_bounds_and_append_no_op_for_unknown_key():
    # An (action_group, dim) key absent from `bounds` should be silently
    # skipped, not raise a KeyError.
    result = []
    _check_bounds_and_append(result, {}, "house_0", "traj.h5", 0, 5, "reltrack_arm", 0, 100.0)
    assert result == []


# --- _collect_stats_worker / _find_outliers_worker --------------------------


def _write_synthetic_trajectory_h5(h5_path, jpr, cmd, qpos):
    """Write a minimal traj_0 group matching the layout benchmark_utils reads:
    actions/joint_pos_rel, actions/joint_pos (T entries) and obs/agent/qpos
    (T+1 entries, since qpos[t+1] is compared against cmd[t])."""
    with h5py.File(h5_path, "w") as f:
        traj = f.create_group("traj_0")
        actions = traj.create_group("actions")
        _make_json_array_dataset(actions, "joint_pos_rel", jpr)
        _make_json_array_dataset(actions, "joint_pos", cmd)
        obs = traj.create_group("obs")
        agent = obs.create_group("agent")
        _make_json_array_dataset(agent, "qpos", qpos)


def test_collect_stats_worker_computes_relative_tracking_error(tmp_path):
    h5_path = tmp_path / "traj.h5"
    # One step: jpr=2.0 (joint pos range), cmd=1.0, actual qpos[t+1]=1.5
    # -> tracking_error = 1.5 - 1.0 = 0.5, ratio = 0.5 / 2.0 = 0.25
    _write_synthetic_trajectory_h5(
        h5_path,
        jpr=[{"arm": [2.0]}],
        cmd=[{"arm": [1.0]}],
        qpos=[{"arm": [0.0]}, {"arm": [1.5]}],
    )
    args = (str(h5_path), ["arm"], 0, 1e-6)
    result = _collect_stats_worker(args)
    assert result[("reltrack_arm", 0)] == pytest.approx([0.25])


def test_collect_stats_worker_respects_skip_first(tmp_path):
    h5_path = tmp_path / "traj.h5"
    _write_synthetic_trajectory_h5(
        h5_path,
        jpr=[{"arm": [2.0]}, {"arm": [2.0]}],
        cmd=[{"arm": [1.0]}, {"arm": [1.0]}],
        qpos=[{"arm": [0.0]}, {"arm": [1.5]}, {"arm": [1.5]}],
    )
    args = (str(h5_path), ["arm"], 1, 1e-6)  # skip timestep 0
    result = _collect_stats_worker(args)
    assert result[("reltrack_arm", 0)] == pytest.approx([0.25])  # only t=1 kept


def test_find_outliers_worker_flags_value_outside_bounds(tmp_path):
    h5_path = tmp_path / "traj.h5"
    _write_synthetic_trajectory_h5(
        h5_path,
        jpr=[{"arm": [2.0]}],
        cmd=[{"arm": [1.0]}],
        qpos=[{"arm": [0.0]}, {"arm": [1.5]}],
    )
    # ratio = 0.25; bounds exclude anything above 0.1 -> should be flagged.
    bounds = {("reltrack_arm", 0): (-0.1, 0.1, 0.0, 0.05)}
    args = (str(h5_path), ["arm"], 0, bounds, 1e-6, False)
    outliers = _find_outliers_worker(args)
    assert len(outliers) == 1
    assert outliers[0]["value"] == pytest.approx(0.25)
    assert outliers[0]["action_dim"] == "reltrack_arm[0]"


def test_find_outliers_worker_no_outliers_within_bounds(tmp_path):
    h5_path = tmp_path / "traj.h5"
    _write_synthetic_trajectory_h5(
        h5_path,
        jpr=[{"arm": [2.0]}],
        cmd=[{"arm": [1.0]}],
        qpos=[{"arm": [0.0]}, {"arm": [1.5]}],
    )
    bounds = {("reltrack_arm", 0): (-1.0, 1.0, 0.0, 0.5)}  # 0.25 is well within range
    args = (str(h5_path), ["arm"], 0, bounds, 1e-6, False)
    assert _find_outliers_worker(args) == []
