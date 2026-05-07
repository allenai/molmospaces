"""Tests for MlSpacesKinematics and SimpleWarpKinematics.

Tests cover:
- Forward kinematics (FK) correctness and frame transforms
- Inverse kinematics (IK) convergence and failure modes
- Batch vs single-element consistency (SimpleWarpKinematics)
- Cross-solver agreement between MlSpacesKinematics and SimpleWarpKinematics
- Non-identity base poses for both solvers
"""

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import FrankaRobotConfig, RBY1MConfig
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics

_WARP_DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


# --- Helpers ---


def rby1m_base_qpos_to_pose(base_qpos: np.ndarray) -> np.ndarray:
    """Convert RBY1M base joint positions [x, y, theta] to a 4x4 pose matrix."""
    pose = np.eye(4)
    pose[0, 3] = base_qpos[0]
    pose[1, 3] = base_qpos[1]
    pose[:3, :3] = R.from_euler("z", base_qpos[2]).as_matrix()
    return pose


def _assert_valid_transform(pose: np.ndarray, atol=1e-6):
    rot = pose[:3, :3]
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=atol)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=atol)
    np.testing.assert_array_equal(pose[3, :], [0, 0, 0, 1])


def _make_offset_base_pose():
    """Non-trivial base pose: translated and 45-deg yaw rotation."""
    pose = np.eye(4)
    pose[:3, 3] = [1.5, -0.8, 0.6]
    pose[:3, :3] = R.from_euler("z", 45, degrees=True).as_matrix()
    return pose


# --- Fixtures ---


@pytest.fixture(scope="module")
def franka_config():
    return FrankaRobotConfig()


@pytest.fixture(scope="module")
def rby1m_config():
    return RBY1MConfig()


@pytest.fixture(scope="module")
def franka_mujoco_kin(franka_config):
    return MlSpacesKinematics(franka_config)


@pytest.fixture(scope="module")
def rby1m_mujoco_kin(rby1m_config):
    return MlSpacesKinematics(rby1m_config)


@pytest.fixture(scope="module", params=_WARP_DEVICES)
def franka_warp_kin(franka_config, request):
    return SimpleWarpKinematics(franka_config, device=request.param)


@pytest.fixture(scope="module", params=_WARP_DEVICES)
def rby1m_warp_kin(rby1m_config, request):
    return SimpleWarpKinematics(rby1m_config, device=request.param)


@pytest.fixture()
def franka_q0(franka_config):
    q0 = {k: np.array(v, dtype=np.float64) for k, v in franka_config.init_qpos.items()}
    q0.setdefault("base", np.array([]))
    return q0


@pytest.fixture()
def rby1m_q0(rby1m_config):
    return {k: np.array(v, dtype=np.float64) for k, v in rby1m_config.init_qpos.items()}


# --- MlSpacesKinematics: FrankaDroid ---


class TestMlSpacesKinematicsFranka:
    def test_fk_returns_valid_transforms(self, franka_mujoco_kin, franka_q0):
        result = franka_mujoco_kin.fk(franka_q0, np.eye(4))
        assert "arm" in result and "gripper" in result
        for pose in result.values():
            _assert_valid_transform(pose)

    def test_fk_sensitive_to_qpos(self, franka_mujoco_kin, franka_q0):
        result1 = franka_mujoco_kin.fk(franka_q0, np.eye(4))
        q0_perturbed = {k: v.copy() for k, v in franka_q0.items()}
        q0_perturbed["arm"] = q0_perturbed["arm"] + 0.1
        result2 = franka_mujoco_kin.fk(q0_perturbed, np.eye(4))
        assert not np.allclose(result1["gripper"], result2["gripper"], atol=1e-4)

    def test_fk_rel_to_base(self, franka_mujoco_kin, franka_q0):
        base_pose = _make_offset_base_pose()
        fk_world = franka_mujoco_kin.fk(franka_q0, base_pose, rel_to_base=False)
        fk_rel = franka_mujoco_kin.fk(franka_q0, base_pose, rel_to_base=True)

        # world = base @ rel
        np.testing.assert_allclose(fk_world["gripper"], base_pose @ fk_rel["gripper"], atol=1e-6)
        # rel_to_base is invariant to base_pose
        fk_identity_rel = franka_mujoco_kin.fk(franka_q0, np.eye(4), rel_to_base=True)
        np.testing.assert_allclose(fk_rel["gripper"], fk_identity_rel["gripper"], atol=1e-6)

    def test_ik_converges(self, franka_mujoco_kin, franka_q0):
        fk_result = franka_mujoco_kin.fk(franka_q0, np.eye(4))
        target = fk_result["gripper"].copy()
        target[:3, 3] += [0.02, -0.02, 0.03]

        result = franka_mujoco_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4))
        assert result is not None
        fk_check = franka_mujoco_kin.fk(result, np.eye(4))
        np.testing.assert_allclose(fk_check["gripper"][:3, 3], target[:3, 3], atol=1e-3)

    def test_ik_unreachable_returns_none(self, franka_mujoco_kin, franka_q0):
        target = np.eye(4)
        target[:3, 3] = [10.0, 10.0, 10.0]
        result = franka_mujoco_kin.ik(
            "gripper", target, ["arm"], franka_q0, np.eye(4), max_iter=100
        )
        assert result is None

    def test_ik_with_non_identity_base(self, franka_mujoco_kin, franka_q0):
        base_pose = _make_offset_base_pose()
        fk_result = franka_mujoco_kin.fk(franka_q0, base_pose)
        target = fk_result["gripper"].copy()
        target[:3, 3] += [0.03, -0.02, 0.01]

        result = franka_mujoco_kin.ik("gripper", target, ["arm"], franka_q0, base_pose)
        assert result is not None
        fk_check = franka_mujoco_kin.fk(result, base_pose)
        np.testing.assert_allclose(fk_check["gripper"][:3, 3], target[:3, 3], atol=1e-3)

    def test_ik_rel_to_base(self, franka_mujoco_kin, franka_q0):
        base_pose = _make_offset_base_pose()
        fk_rel = franka_mujoco_kin.fk(franka_q0, base_pose, rel_to_base=True)
        target_rel = fk_rel["gripper"].copy()
        target_rel[:3, 3] += [0.02, 0.0, 0.02]

        result = franka_mujoco_kin.ik(
            "gripper", target_rel, ["arm"], franka_q0, base_pose, rel_to_base=True
        )
        assert result is not None
        fk_check = franka_mujoco_kin.fk(result, base_pose, rel_to_base=True)
        np.testing.assert_allclose(fk_check["gripper"][:3, 3], target_rel[:3, 3], atol=1e-3)

    def test_ik_q0_key_validation(self, franka_mujoco_kin):
        with pytest.raises(ValueError, match="q0 keys must match move group ids"):
            franka_mujoco_kin.ik("gripper", np.eye(4), ["arm"], {"arm": np.zeros(7)}, np.eye(4))


# --- MlSpacesKinematics: RBY1M ---


class TestMlSpacesKinematicsRBY1M:
    def test_fk_returns_valid_transforms(self, rby1m_mujoco_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        result = rby1m_mujoco_kin.fk(rby1m_q0, base_pose)
        expected = {
            "base",
            "torso",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
            "head",
        }
        assert expected.issubset(set(result.keys()))
        for pose in result.values():
            _assert_valid_transform(pose)

    def test_fk_arms_are_not_coincident(self, rby1m_mujoco_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        result = rby1m_mujoco_kin.fk(rby1m_q0, base_pose)
        dist = np.linalg.norm(result["left_gripper"][:3, 3] - result["right_gripper"][:3, 3])
        assert dist > 0.1

    @pytest.mark.parametrize(
        "gripper,arm", [("left_gripper", "left_arm"), ("right_gripper", "right_arm")]
    )
    def test_ik_converges(self, rby1m_mujoco_kin, rby1m_q0, gripper, arm):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk_result = rby1m_mujoco_kin.fk(rby1m_q0, base_pose)
        target = fk_result[gripper].copy()
        target[:3, 3] += [0.02, 0.0, 0.02]

        result = rby1m_mujoco_kin.ik(gripper, target, [arm], rby1m_q0, base_pose)
        assert result is not None
        result_bp = rby1m_base_qpos_to_pose(result["base"])
        fk_check = rby1m_mujoco_kin.fk(result, result_bp)
        np.testing.assert_allclose(fk_check[gripper][:3, 3], target[:3, 3], atol=1e-3)

    def test_ik_with_base_unlocked(self, rby1m_mujoco_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk_result = rby1m_mujoco_kin.fk(rby1m_q0, base_pose)
        target = fk_result["left_gripper"].copy()
        target[:3, 3] += [0.05, 0.05, 0.0]

        result = rby1m_mujoco_kin.ik(
            "left_gripper", target, ["left_arm", "base"], rby1m_q0, base_pose
        )
        assert result is not None

    def test_ik_unreachable_returns_none(self, rby1m_mujoco_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        target = np.eye(4)
        target[:3, 3] = [10.0, 10.0, 10.0]
        result = rby1m_mujoco_kin.ik(
            "left_gripper", target, ["left_arm"], rby1m_q0, base_pose, max_iter=100
        )
        assert result is None

    def test_fk_changes_with_base_qpos(self, rby1m_mujoco_kin, rby1m_q0, rby1m_config):
        q0_moved = {k: np.array(v, dtype=np.float64) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])

        fk_origin = rby1m_mujoco_kin.fk(rby1m_q0, rby1m_base_qpos_to_pose(rby1m_q0["base"]))
        fk_moved = rby1m_mujoco_kin.fk(q0_moved, rby1m_base_qpos_to_pose(q0_moved["base"]))
        assert not np.allclose(
            fk_origin["left_gripper"][:3, 3], fk_moved["left_gripper"][:3, 3], atol=0.1
        )

    def test_fk_rel_to_base_invariant_to_base_qpos(self, rby1m_mujoco_kin, rby1m_q0, rby1m_config):
        q0_moved = {k: np.array(v, dtype=np.float64) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])
        moved_bp = rby1m_base_qpos_to_pose(q0_moved["base"])

        fk_origin_rel = rby1m_mujoco_kin.fk(rby1m_q0, np.eye(4), rel_to_base=True)
        fk_moved_rel = rby1m_mujoco_kin.fk(q0_moved, moved_bp, rel_to_base=True)
        np.testing.assert_allclose(
            fk_origin_rel["left_gripper"], fk_moved_rel["left_gripper"], atol=1e-5
        )

    def test_ik_with_moved_base(self, rby1m_mujoco_kin, rby1m_config):
        q0_moved = {k: np.array(v, dtype=np.float64) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])
        moved_bp = rby1m_base_qpos_to_pose(q0_moved["base"])

        fk_result = rby1m_mujoco_kin.fk(q0_moved, moved_bp)
        target = fk_result["left_gripper"].copy()
        target[:3, 3] += [0.02, 0.0, 0.02]

        result = rby1m_mujoco_kin.ik(
            "left_gripper", target, ["left_arm", "base"], q0_moved, moved_bp
        )
        assert result is not None

    def test_ik_rel_to_base_with_moved_base(self, rby1m_mujoco_kin, rby1m_config):
        q0_moved = {k: np.array(v, dtype=np.float64) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])
        moved_bp = rby1m_base_qpos_to_pose(q0_moved["base"])

        fk_rel = rby1m_mujoco_kin.fk(q0_moved, moved_bp, rel_to_base=True)
        target_rel = fk_rel["left_gripper"].copy()
        target_rel[:3, 3] += [0.02, 0.0, 0.02]

        # Only unlock arm so base frame stays fixed for verification
        result = rby1m_mujoco_kin.ik(
            "left_gripper", target_rel, ["left_arm"], q0_moved, moved_bp, rel_to_base=True
        )
        assert result is not None
        fk_check = rby1m_mujoco_kin.fk(result, moved_bp, rel_to_base=True)
        np.testing.assert_allclose(fk_check["left_gripper"][:3, 3], target_rel[:3, 3], atol=1e-3)


# --- SimpleWarpKinematics: FrankaDroid ---


class TestSimpleWarpKinematicsFranka:
    def test_fk_single_and_batch(self, franka_warp_kin, franka_q0):
        single = franka_warp_kin.fk(franka_q0, np.eye(4))
        assert "gripper" in single
        _assert_valid_transform(single["gripper"], atol=1e-5)

        batch = franka_warp_kin.fk([franka_q0, franka_q0], np.eye(4))
        assert len(batch) == 2
        np.testing.assert_allclose(single["gripper"], batch[0]["gripper"], atol=1e-5)
        np.testing.assert_allclose(single["gripper"], batch[1]["gripper"], atol=1e-5)

    def test_ik_single(self, franka_warp_kin, franka_q0):
        fk_result = franka_warp_kin.fk(franka_q0, np.eye(4))
        target = fk_result["gripper"].copy()
        target[:3, 3] += [0.02, -0.02, 0.03]

        result = franka_warp_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4))
        assert result is not None
        fk_check = franka_warp_kin.fk(result, np.eye(4))
        np.testing.assert_allclose(fk_check["gripper"][:3, 3], target[:3, 3], atol=2e-3)

    def test_ik_batch(self, franka_warp_kin, franka_q0):
        fk_result = franka_warp_kin.fk(franka_q0, np.eye(4))
        base_target = fk_result["gripper"].copy()
        offsets = np.array([[0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]])
        targets = np.repeat(base_target[None], 4, axis=0)
        targets[:, :3, 3] += offsets

        results = franka_warp_kin.ik("gripper", targets, ["arm"], [franka_q0] * 4, np.eye(4))
        assert len(results) == 4
        for i, r in enumerate(results):
            assert r is not None
            fk_check = franka_warp_kin.fk(r, np.eye(4))
            np.testing.assert_allclose(fk_check["gripper"][:3, 3], targets[i, :3, 3], atol=2e-3)

    def test_ik_unreachable_returns_none(self, franka_warp_kin, franka_q0):
        target = np.eye(4)
        target[:3, 3] = [10.0, 10.0, 10.0]
        result = franka_warp_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4), max_iter=50)
        assert result is None

    def test_ik_preserves_non_actuated_groups(self, franka_warp_kin, franka_q0):
        fk_result = franka_warp_kin.fk(franka_q0, np.eye(4))
        target = fk_result["gripper"].copy()
        target[:3, 3] += [0.01, 0.0, 0.0]

        result = franka_warp_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4))
        assert result is not None
        np.testing.assert_array_equal(result["gripper"], franka_q0["gripper"])

    def test_ik_invalid_move_group_raises(self, franka_warp_kin, franka_q0):
        with pytest.raises(ValueError, match="not a MJCFFrameMixin"):
            franka_warp_kin.ik("nonexistent", np.eye(4), ["arm"], franka_q0, np.eye(4))

    def test_ik_with_non_identity_base(self, franka_warp_kin, franka_q0):
        """IK converts world-frame targets to base frame internally."""
        base_pose = _make_offset_base_pose()
        fk_base = franka_warp_kin.fk(franka_q0, np.eye(4))
        target_world = base_pose @ fk_base["gripper"].copy()
        target_world[:3, 3] += [0.03, -0.02, 0.01]

        result = franka_warp_kin.ik("gripper", target_world, ["arm"], franka_q0, base_pose)
        assert result is not None
        fk_check = franka_warp_kin.fk(result, np.eye(4))
        result_world = base_pose @ fk_check["gripper"]
        np.testing.assert_allclose(result_world[:3, 3], target_world[:3, 3], atol=2e-3)

    def test_ik_rel_to_base_consistent_across_base_poses(self, franka_warp_kin, franka_q0):
        """Same base-relative target yields same joints regardless of base_pose."""
        base_pose = _make_offset_base_pose()
        fk_base = franka_warp_kin.fk(franka_q0, np.eye(4))
        target = fk_base["gripper"].copy()
        target[:3, 3] += [0.02, 0.0, 0.0]

        r1 = franka_warp_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4), rel_to_base=True)
        r2 = franka_warp_kin.ik("gripper", target, ["arm"], franka_q0, base_pose, rel_to_base=True)
        assert r1 is not None and r2 is not None
        np.testing.assert_allclose(r1["arm"], r2["arm"], atol=1e-4)

    def test_sequential_calls_not_cached(self, franka_warp_kin, franka_q0):
        """Successive calls with different inputs must produce different correct results."""
        fk1 = franka_warp_kin.fk(franka_q0, np.eye(4))

        q0_perturbed = {k: v.copy() for k, v in franka_q0.items()}
        q0_perturbed["arm"] = q0_perturbed["arm"] + 0.2
        fk2 = franka_warp_kin.fk(q0_perturbed, np.eye(4))

        assert not np.allclose(fk1["gripper"], fk2["gripper"], atol=1e-4)

        # Call again with original input to confirm it still returns the first result
        fk3 = franka_warp_kin.fk(franka_q0, np.eye(4))
        np.testing.assert_allclose(fk1["gripper"], fk3["gripper"], atol=1e-6)

    def test_sequential_ik_not_cached(self, franka_warp_kin, franka_q0):
        """Two IK calls with different targets must converge to their respective targets."""
        fk_result = franka_warp_kin.fk(franka_q0, np.eye(4))
        base_target = fk_result["gripper"].copy()

        target_a = base_target.copy()
        target_a[:3, 3] += [0.04, 0.0, 0.0]
        target_b = base_target.copy()
        target_b[:3, 3] += [0.0, 0.04, 0.0]

        result_a = franka_warp_kin.ik("gripper", target_a, ["arm"], franka_q0, np.eye(4))
        result_b = franka_warp_kin.ik("gripper", target_b, ["arm"], franka_q0, np.eye(4))
        assert result_a is not None and result_b is not None
        assert not np.allclose(result_a["arm"], result_b["arm"], atol=1e-3)

        fk_a = franka_warp_kin.fk(result_a, np.eye(4))
        fk_b = franka_warp_kin.fk(result_b, np.eye(4))
        np.testing.assert_allclose(fk_a["gripper"][:3, 3], target_a[:3, 3], atol=2e-3)
        np.testing.assert_allclose(fk_b["gripper"][:3, 3], target_b[:3, 3], atol=2e-3)

    def test_warmup_ik(self, franka_warp_kin):
        franka_warp_kin.warmup_ik(4)


# --- SimpleWarpKinematics: RBY1M ---


class TestSimpleWarpKinematicsRBY1M:
    def test_fk_single_and_batch(self, rby1m_warp_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        single = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        assert isinstance(single, dict)

        batch = rby1m_warp_kin.fk([rby1m_q0, rby1m_q0, rby1m_q0], base_pose)
        assert len(batch) == 3

    @pytest.mark.parametrize(
        "gripper,unlocked",
        [
            ("left_gripper", ["left_arm", "base"]),
            ("right_gripper", ["right_arm"]),
        ],
    )
    def test_ik_converges(self, rby1m_warp_kin, rby1m_q0, gripper, unlocked):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk_result = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        target_key = gripper if gripper in fk_result else gripper.replace("gripper", "arm")
        target = fk_result[target_key].copy()
        target[:3, 3] += [0.02, 0.0, 0.02]

        result = rby1m_warp_kin.ik(target_key, target, unlocked, rby1m_q0, base_pose)
        assert result is not None
        result_bp = rby1m_base_qpos_to_pose(result["base"])
        fk_check = rby1m_warp_kin.fk(result, result_bp)
        np.testing.assert_allclose(fk_check[target_key][:3, 3], target[:3, 3], atol=2e-3)

    def test_ik_batch(self, rby1m_warp_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk_result = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        target_key = "left_gripper" if "left_gripper" in fk_result else "left_arm"
        base_target = fk_result[target_key].copy()

        targets = np.repeat(base_target[None], 3, axis=0)
        targets[0, :3, 3] += [0.02, 0, 0]
        targets[1, :3, 3] += [0, 0.02, 0]
        targets[2, :3, 3] += [0, 0, 0.02]

        results = rby1m_warp_kin.ik(
            target_key, targets, ["left_arm", "base"], [rby1m_q0] * 3, base_pose
        )
        assert all(r is not None for r in results)

    def test_ik_unreachable_returns_none(self, rby1m_warp_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk_result = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        target_key = "left_gripper" if "left_gripper" in fk_result else "left_arm"

        target = np.eye(4)
        target[:3, 3] = [10.0, 10.0, 10.0]
        result = rby1m_warp_kin.ik(
            target_key, target, ["left_arm"], rby1m_q0, base_pose, max_iter=50
        )
        assert result is None

    def test_fk_changes_with_base_qpos(self, rby1m_warp_kin, rby1m_q0, rby1m_config):
        q0_moved = {k: np.array(v) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])

        fk_origin = rby1m_warp_kin.fk(rby1m_q0, rby1m_base_qpos_to_pose(rby1m_q0["base"]))
        fk_moved = rby1m_warp_kin.fk(q0_moved, rby1m_base_qpos_to_pose(q0_moved["base"]))
        target_key = "left_gripper" if "left_gripper" in fk_origin else "left_arm"
        assert not np.allclose(fk_origin[target_key][:3, 3], fk_moved[target_key][:3, 3], atol=0.1)

    def test_fk_rel_to_base_invariant_to_base_qpos(self, rby1m_warp_kin, rby1m_q0, rby1m_config):
        q0_moved = {k: np.array(v) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])
        moved_bp = rby1m_base_qpos_to_pose(q0_moved["base"])

        fk_origin_rel = rby1m_warp_kin.fk(
            rby1m_q0, rby1m_base_qpos_to_pose(rby1m_q0["base"]), rel_to_base=True
        )
        fk_moved_rel = rby1m_warp_kin.fk(q0_moved, moved_bp, rel_to_base=True)
        target_key = "left_gripper" if "left_gripper" in fk_origin_rel else "left_arm"
        np.testing.assert_allclose(fk_origin_rel[target_key], fk_moved_rel[target_key], atol=1e-4)

    def test_ik_with_moved_base(self, rby1m_warp_kin, rby1m_config):
        q0_moved = {k: np.array(v) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])
        moved_bp = rby1m_base_qpos_to_pose(q0_moved["base"])

        fk_result = rby1m_warp_kin.fk(q0_moved, moved_bp)
        target_key = "left_gripper" if "left_gripper" in fk_result else "left_arm"
        target = fk_result[target_key].copy()
        target[:3, 3] += [0.02, 0.0, 0.02]

        result = rby1m_warp_kin.ik(target_key, target, ["left_arm", "base"], q0_moved, moved_bp)
        assert result is not None
        fk_check = rby1m_warp_kin.fk(result, moved_bp)
        np.testing.assert_allclose(fk_check[target_key][:3, 3], target[:3, 3], atol=2e-3)

    def test_sequential_calls_not_cached(self, rby1m_warp_kin, rby1m_q0, rby1m_config):
        """Successive calls with different inputs must produce different correct results."""
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk1 = rby1m_warp_kin.fk(rby1m_q0, base_pose)

        q0_perturbed = {k: v.copy() for k, v in rby1m_q0.items()}
        q0_perturbed["left_arm"] = q0_perturbed["left_arm"] + 0.2
        fk2 = rby1m_warp_kin.fk(q0_perturbed, base_pose)

        target_key = "left_gripper" if "left_gripper" in fk1 else "left_arm"
        assert not np.allclose(fk1[target_key], fk2[target_key], atol=1e-4)

        fk3 = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        np.testing.assert_allclose(fk1[target_key], fk3[target_key], atol=1e-6)

    def test_sequential_ik_not_cached(self, rby1m_warp_kin, rby1m_q0):
        """Two IK calls with different targets must converge to their respective targets."""
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk_result = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        target_key = "left_gripper" if "left_gripper" in fk_result else "left_arm"
        base_target = fk_result[target_key].copy()

        target_a = base_target.copy()
        target_a[:3, 3] += [0.04, 0.0, 0.0]
        target_b = base_target.copy()
        target_b[:3, 3] += [0.0, 0.04, 0.0]

        result_a = rby1m_warp_kin.ik(target_key, target_a, ["left_arm"], rby1m_q0, base_pose)
        result_b = rby1m_warp_kin.ik(target_key, target_b, ["left_arm"], rby1m_q0, base_pose)
        assert result_a is not None and result_b is not None
        assert not np.allclose(result_a["left_arm"], result_b["left_arm"], atol=1e-3)

        fk_a = rby1m_warp_kin.fk(result_a, rby1m_base_qpos_to_pose(result_a["base"]))
        fk_b = rby1m_warp_kin.fk(result_b, rby1m_base_qpos_to_pose(result_b["base"]))
        np.testing.assert_allclose(fk_a[target_key][:3, 3], target_a[:3, 3], atol=2e-3)
        np.testing.assert_allclose(fk_b[target_key][:3, 3], target_b[:3, 3], atol=2e-3)

    def test_warmup_ik(self, rby1m_warp_kin):
        rby1m_warp_kin.warmup_ik(4)


# --- Cross-solver consistency ---


class TestCrossSolverConsistency:
    """Verify MlSpacesKinematics and SimpleWarpKinematics agree on FK and IK."""

    def test_franka_fk_agreement(self, franka_mujoco_kin, franka_warp_kin, franka_q0):
        mujoco_fk = franka_mujoco_kin.fk(franka_q0, np.eye(4))
        warp_fk = franka_warp_kin.fk(franka_q0, np.eye(4))
        for key in warp_fk:
            if key in mujoco_fk:
                np.testing.assert_allclose(mujoco_fk[key], warp_fk[key], atol=1e-4)

    def test_franka_ik_agreement(self, franka_mujoco_kin, franka_warp_kin, franka_q0):
        fk_result = franka_mujoco_kin.fk(franka_q0, np.eye(4))
        target = fk_result["gripper"].copy()
        target[:3, 3] += [0.03, -0.02, 0.01]

        mujoco_r = franka_mujoco_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4))
        warp_r = franka_warp_kin.ik("gripper", target, ["arm"], franka_q0, np.eye(4))
        assert mujoco_r is not None and warp_r is not None

        mujoco_fk = franka_mujoco_kin.fk(mujoco_r, np.eye(4))
        warp_fk = franka_warp_kin.fk(warp_r, np.eye(4))
        np.testing.assert_allclose(mujoco_fk["gripper"][:3, 3], target[:3, 3], atol=1e-3)
        np.testing.assert_allclose(warp_fk["gripper"][:3, 3], target[:3, 3], atol=2e-3)

    def test_rby1m_fk_agreement(self, rby1m_mujoco_kin, rby1m_warp_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        mujoco_fk = rby1m_mujoco_kin.fk(rby1m_q0, base_pose)
        warp_fk = rby1m_warp_kin.fk(rby1m_q0, base_pose)
        for key in warp_fk:
            if key in mujoco_fk:
                np.testing.assert_allclose(mujoco_fk[key], warp_fk[key], atol=1e-4)

    def test_rby1m_ik_agreement(self, rby1m_mujoco_kin, rby1m_warp_kin, rby1m_q0):
        base_pose = rby1m_base_qpos_to_pose(rby1m_q0["base"])
        fk = rby1m_mujoco_kin.fk(rby1m_q0, base_pose)
        target_key = "left_gripper" if "left_gripper" in fk else "left_arm"
        target = fk[target_key].copy()
        target[:3, 3] += [0.02, 0.0, 0.02]

        mujoco_r = rby1m_mujoco_kin.ik(
            target_key, target, ["left_arm", "base"], rby1m_q0, base_pose
        )
        warp_r = rby1m_warp_kin.ik(target_key, target, ["left_arm", "base"], rby1m_q0, base_pose)
        assert mujoco_r is not None and warp_r is not None

        mujoco_fk = rby1m_mujoco_kin.fk(mujoco_r, rby1m_base_qpos_to_pose(mujoco_r["base"]))
        warp_fk = rby1m_warp_kin.fk(warp_r, rby1m_base_qpos_to_pose(warp_r["base"]))
        np.testing.assert_allclose(mujoco_fk[target_key][:3, 3], target[:3, 3], atol=1e-3)
        np.testing.assert_allclose(warp_fk[target_key][:3, 3], target[:3, 3], atol=2e-3)

    def test_rby1m_fk_agreement_with_moved_base(
        self, rby1m_mujoco_kin, rby1m_warp_kin, rby1m_config
    ):
        q0_moved = {k: np.array(v, dtype=np.float64) for k, v in rby1m_config.init_qpos.items()}
        q0_moved["base"] = np.array([1.0, 0.5, np.pi / 6])
        moved_bp = rby1m_base_qpos_to_pose(q0_moved["base"])

        mujoco_fk = rby1m_mujoco_kin.fk(q0_moved, moved_bp)
        warp_fk = rby1m_warp_kin.fk(q0_moved, moved_bp)
        for key in warp_fk:
            if key in mujoco_fk:
                np.testing.assert_allclose(mujoco_fk[key], warp_fk[key], atol=1e-4)
