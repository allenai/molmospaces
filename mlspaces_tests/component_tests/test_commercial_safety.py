"""Commercial-safety regression tests against installed asset metadata.

These tests use real Objaverse UIDs from ``ObjectMeta`` / ``molmospaces_resources``
whenever possible. Pure wiring checks live in ``test_license_policy.py``.
Tests skip if object metadata is unavailable.

First run may take ~30--60s while metadata LMDB and (optionally) ObjectRetriever
index are built; later runs reuse caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import pytest

from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils import lazy_loading_utils
from molmo_spaces.utils.license_policy import LicensePolicy, set_license_policy
from molmo_spaces.utils.license_utils import list_asset_identifiers
from molmo_spaces.utils.object_metadata import ObjectMeta


@dataclass(frozen=True)
class LicenseTestAssets:
    nc_uid: str
    ok_uid: str
    nc_uid_with_grasp: str
    ok_uid_with_grasp: str
    nc_receptacle_uid: str
    ok_receptacle_uid: str
    nc_uids: frozenset[str]


@pytest.fixture(scope="session")
def license_test_assets() -> LicenseTestAssets:
    try:
        from molmo_spaces.utils.grasps import get_pickup_grasp_path
        from molmo_spaces.utils.license_policy import non_commercial_objaverse_uids
    except ImportError as exc:
        pytest.skip(f"Object metadata unavailable: {exc}")

    nc_uids = non_commercial_objaverse_uids()
    if not nc_uids:
        pytest.skip("No non-commercial Objaverse UIDs in installed metadata")

    nc_uid = next(iter(nc_uids))
    ok_uid = None
    ok_uid_with_grasp = None
    nc_uid_with_grasp = None

    for uid, anno in ObjectMeta.annotation().items():
        if not anno.get("isObjaverse"):
            continue
        lic = (anno.get("license_info") or {}).get("license")
        if lic in {"by-nc", "by-nc-sa"}:
            if nc_uid_with_grasp is None and get_pickup_grasp_path(uid):
                nc_uid_with_grasp = uid
            continue
        ok_uid = ok_uid or uid
        if ok_uid_with_grasp is None and get_pickup_grasp_path(uid):
            ok_uid_with_grasp = uid
        if ok_uid and ok_uid_with_grasp and nc_uid_with_grasp:
            break

    if ok_uid is None:
        pytest.skip("No commercial Objaverse UIDs in installed metadata")
    if nc_uid_with_grasp is None:
        pytest.skip("No NC Objaverse UID with grasp files found in metadata sample")
    if ok_uid_with_grasp is None:
        pytest.skip("No commercial Objaverse UID with grasp files found in metadata sample")

    nc_receptacle_uid = None
    for uid in nc_uids:
        anno = ObjectMeta.annotation(uid) or {}
        if not anno.get("receptacle"):
            continue
        synset = anno.get("synset")
        if synset:
            from molmo_spaces.utils import synset_utils

            if synset_utils.is_valid_receptacle_synset(synset):
                nc_receptacle_uid = uid
                break
    if nc_receptacle_uid is None:
        pytest.skip("No NC receptacle UID with valid receptacle synset in metadata")

    ok_receptacle_uid = next(
        (
            uid
            for uid, anno in ObjectMeta.annotation().items()
            if anno.get("isObjaverse") and anno.get("receptacle") and uid not in nc_uids
        ),
        ok_uid,
    )

    return LicenseTestAssets(
        nc_uid=nc_uid,
        ok_uid=ok_uid,
        nc_uid_with_grasp=nc_uid_with_grasp,
        ok_uid_with_grasp=ok_uid_with_grasp,
        nc_receptacle_uid=nc_receptacle_uid,
        ok_receptacle_uid=ok_receptacle_uid,
        nc_uids=nc_uids,
    )


@pytest.fixture(scope="session")
def receptacle_uid_sets(license_test_assets):
    from molmo_spaces.utils import synset_utils

    set_license_policy(LicensePolicy.NONE)
    none_ids = set(synset_utils.get_valid_receptacle_uids())
    set_license_policy(LicensePolicy.COMMERCIAL_SAFE)
    safe_ids = set(synset_utils.get_valid_receptacle_uids())
    set_license_policy(LicensePolicy.NONE)
    return none_ids, safe_ids


@pytest.fixture(scope="session")
def pickupable_uid_sets(license_test_assets):
    from molmo_spaces.utils import synset_utils

    set_license_policy(LicensePolicy.NONE)
    none_ids = set(synset_utils.get_valid_pickupable_obja_uids())
    set_license_policy(LicensePolicy.COMMERCIAL_SAFE)
    safe_ids = set(synset_utils.get_valid_pickupable_obja_uids())
    set_license_policy(LicensePolicy.NONE)
    return none_ids, safe_ids


@pytest.fixture(scope="session")
def object_retriever():
    from molmo_spaces.utils.object_retriever import ObjectRetriever

    return ObjectRetriever(sim_thres=0.25, max_results=100)


@pytest.fixture(autouse=True)
def reset_license_policy():
    set_license_policy(LicensePolicy.NONE)
    yield
    set_license_policy(LicensePolicy.NONE)


def test_real_metadata_marks_nc_uid_blocked(license_test_assets):
    assert ObjectMeta.is_commercial_use_allowed(license_test_assets.ok_uid) is True
    assert ObjectMeta.is_commercial_use_allowed(license_test_assets.nc_uid) is False


@pytest.mark.parametrize(
    "policy,nc_listed",
    [
        (LicensePolicy.NONE, True),
        (LicensePolicy.COMMERCIAL_SAFE, False),
    ],
)
def test_list_asset_identifiers_real_objaverse(policy, nc_listed, license_test_assets):
    identifiers = list_asset_identifiers("objects", "objaverse", policy)
    assert license_test_assets.ok_uid in identifiers
    assert (license_test_assets.nc_uid in identifiers) is nc_listed


@pytest.mark.parametrize(
    "policy,nc_located",
    [
        (LicensePolicy.NONE, True),
        (LicensePolicy.COMMERCIAL_SAFE, False),
    ],
)
def test_locate_uid_package_real_nc(policy, nc_located, license_test_assets):
    set_license_policy(policy)
    source, _package, _path = lazy_loading_utils.locate_uid_package(
        license_test_assets.nc_uid,
        license_policy=policy,
    )
    assert (source is not None) is nc_located


def test_locate_uid_package_real_commercial_uid_allowed(license_test_assets):
    source, _package, _path = lazy_loading_utils.locate_uid_package(
        license_test_assets.ok_uid,
        license_policy=LicensePolicy.COMMERCIAL_SAFE,
    )
    assert source is not None


@pytest.mark.parametrize(
    "policy,nc_allowed",
    [
        (LicensePolicy.NONE, True),
        (LicensePolicy.COMMERCIAL_SAFE, False),
    ],
)
def test_is_object_install_allowed_real_nc(policy, nc_allowed, license_test_assets):
    set_license_policy(policy)
    allowed = lazy_loading_utils._is_object_install_allowed(
        license_test_assets.nc_uid, "objaverse", policy
    )
    assert allowed is nc_allowed


def test_require_objaverse_license_known_real_ok(license_test_assets):
    from molmo_spaces.utils.license_policy import require_objaverse_license_known

    require_objaverse_license_known(license_test_assets.ok_uid)


@pytest.mark.parametrize(
    "policy,nc_installed",
    [
        (LicensePolicy.NONE, True),
        (LicensePolicy.COMMERCIAL_SAFE, False),
    ],
)
def test_install_objects_for_scene_skips_real_nc(
    policy, nc_installed, license_test_assets, monkeypatch
):
    nc_uid = license_test_assets.nc_uid
    installed: dict[str, list[str]] = {}
    real_rm = lazy_loading_utils.get_resource_manager()
    archive_name = f"objaverse_{nc_uid}.tar.zst"

    def track_install(data_type, source_to_archives):
        installed.update(source_to_archives)

    monkeypatch.setattr(
        lazy_loading_utils,
        "find_object_paths",
        lambda xml_path, exclude_thor=True: [("objaverse", Path(f"{nc_uid}.xml"))],
    )
    monkeypatch.setattr(
        real_rm,
        "find_archives",
        lambda data_type, source, rel_assets: [archive_name],
    )
    monkeypatch.setattr(
        real_rm,
        "install_packages",
        track_install,
    )

    set_license_policy(policy)
    lazy_loading_utils.install_objects_for_scene(Path("scene.xml"))

    objaverse_archives = installed.get("objaverse", [])
    assert (archive_name in objaverse_archives) is nc_installed


@pytest.mark.parametrize(
    "policy,blocked",
    [
        (LicensePolicy.NONE, False),
        (LicensePolicy.COMMERCIAL_SAFE, True),
    ],
)
def test_install_uid_real_nc_policy(policy, blocked, license_test_assets):
    set_license_policy(policy)
    if blocked:
        with pytest.raises(ValueError, match="blocked by the active license policy"):
            lazy_loading_utils.install_uid(license_test_assets.nc_uid)
    else:
        xml_path = lazy_loading_utils.install_uid(license_test_assets.nc_uid)
        assert xml_path is not None


@pytest.mark.parametrize(
    "policy,nc_stripped",
    [
        (LicensePolicy.NONE, False),
        (LicensePolicy.COMMERCIAL_SAFE, True),
    ],
)
def test_delete_license_blocked_bodies_real_nc_uid(policy, nc_stripped, license_test_assets):
    from molmo_spaces.utils.scene_maps import _delete_license_blocked_bodies

    nc_uid = license_test_assets.nc_uid
    set_license_policy(policy)

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name=f"obja_{nc_uid}_0_0_0")
    spec.worldbody.add_body(name=f"obja_{license_test_assets.ok_uid}_0_0_0")

    deleted = _delete_license_blocked_bodies(spec)
    assert (deleted >= 1) is nc_stripped


def test_get_valid_receptacle_uids_real(receptacle_uid_sets, license_test_assets):
    none_ids, safe_ids = receptacle_uid_sets
    assert license_test_assets.ok_receptacle_uid in none_ids
    assert license_test_assets.ok_receptacle_uid in safe_ids
    assert license_test_assets.nc_receptacle_uid in none_ids
    assert license_test_assets.nc_receptacle_uid not in safe_ids


def test_get_valid_pickupable_obja_uids_real(pickupable_uid_sets, license_test_assets):
    none_ids, safe_ids = pickupable_uid_sets
    assert license_test_assets.ok_uid_with_grasp in none_ids
    assert license_test_assets.ok_uid_with_grasp in safe_ids
    assert license_test_assets.nc_uid_with_grasp in none_ids
    assert license_test_assets.nc_uid_with_grasp not in safe_ids
    assert safe_ids.isdisjoint(license_test_assets.nc_uids)


def test_object_retriever_real_query_excludes_nc(object_retriever, license_test_assets):
    set_license_policy(LicensePolicy.NONE)
    none_uids, _ = object_retriever.query("wooden bowl")
    nc_in_none = sum(1 for uid in none_uids if uid in license_test_assets.nc_uids)

    set_license_policy(LicensePolicy.COMMERCIAL_SAFE)
    safe_uids, _ = object_retriever.query("wooden bowl")
    nc_in_safe = sum(1 for uid in safe_uids if uid in license_test_assets.nc_uids)

    assert nc_in_none > 0, "Expected some NC UIDs in unfiltered semantic search"
    assert nc_in_safe == 0


@pytest.mark.parametrize(
    "policy,nc_blocked",
    [
        (LicensePolicy.NONE, False),
        (LicensePolicy.COMMERCIAL_SAFE, True),
    ],
)
def test_sampler_is_license_blocked_real(policy, nc_blocked, license_test_assets):
    set_license_policy(policy)
    sampler = BaseMujocoTaskSampler.__new__(BaseMujocoTaskSampler)
    sampler.config = MagicMock()
    assert sampler.is_license_blocked(license_test_assets.nc_uid) is nc_blocked
    assert sampler.is_license_blocked(license_test_assets.ok_uid) is False
