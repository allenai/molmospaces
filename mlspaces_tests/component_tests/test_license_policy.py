import pytest

from molmo_spaces.utils.license_policy import (
    LicensePolicy,
    filter_uids,
    is_object_allowed,
    is_scene_source_allowed,
    require_objaverse_license_known,
    set_license_policy,
    validate_datagen_license_policy,
)


@pytest.fixture(autouse=True)
def reset_license_policy():
    set_license_policy(LicensePolicy.NONE)
    yield
    set_license_policy(LicensePolicy.NONE)


def test_is_commercial_use_allowed(monkeypatch):
    def fake_annotation(uid):
        if uid == "nc_uid":
            return {"isObjaverse": True, "license_info": {"license": "by-nc"}}
        if uid == "ok_uid":
            return {"isObjaverse": True, "license_info": {"license": "by"}}
        if uid == "Bowl_1":
            return {"isObjaverse": False}
        if uid == "missing_lic_uid":
            return {"isObjaverse": True, "license_info": {}}
        return None

    monkeypatch.setattr(
        "molmo_spaces.utils.object_metadata.ObjectMeta.annotation",
        fake_annotation,
    )
    from molmo_spaces.utils.object_metadata import ObjectMeta

    assert ObjectMeta.is_commercial_use_allowed("ok_uid") is True
    assert ObjectMeta.is_commercial_use_allowed("Bowl_1") is True
    assert ObjectMeta.is_commercial_use_allowed("nc_uid") is False
    assert ObjectMeta.is_commercial_use_allowed("missing_lic_uid") is False


def test_require_objaverse_license_known_raises(monkeypatch):
    def fake_annotation(uid):
        if uid == "missing_lic_uid":
            return {"isObjaverse": True, "license_info": {}}
        if uid == "ok_uid":
            return {"isObjaverse": True, "license_info": {"license": "by"}}
        return None

    monkeypatch.setattr(
        "molmo_spaces.utils.object_metadata.ObjectMeta.annotation",
        fake_annotation,
    )
    require_objaverse_license_known("ok_uid")
    require_objaverse_license_known("Bowl_1")
    with pytest.raises(ValueError, match="no license metadata"):
        require_objaverse_license_known("missing_lic_uid")


def test_filter_uids_commercial_safe(monkeypatch):
    monkeypatch.setattr(
        "molmo_spaces.utils.object_metadata.ObjectMeta.is_commercial_use_allowed",
        lambda uid: uid != "nc_uid",
    )
    filtered = filter_uids(["a", "nc_uid", "b"], LicensePolicy.COMMERCIAL_SAFE)
    assert filtered == ["a", "b"]


def test_is_scene_source_allowed_commercial_safe():
    assert is_scene_source_allowed("ithor", LicensePolicy.COMMERCIAL_SAFE)
    assert is_scene_source_allowed("procthor-10k-train", LicensePolicy.COMMERCIAL_SAFE)


def test_validate_datagen_license_policy():
    validate_datagen_license_policy("ithor", LicensePolicy.COMMERCIAL_SAFE)


def test_is_object_allowed_respects_none():
    assert is_object_allowed("anything", LicensePolicy.NONE)


def test_object_uid_from_rel_asset():
    """UIDs come from the package directory, not the mesh file name."""
    from pathlib import Path

    from molmo_spaces.utils.lazy_loading_utils import _object_uid_from_rel_asset

    uid = "0001b3c342c74d6987571b531a325199"
    assert _object_uid_from_rel_asset(Path(uid) / f"{uid}_visual.obj") == uid
    assert _object_uid_from_rel_asset(Path(uid) / f"{uid}_collider0.obj") == uid
    assert _object_uid_from_rel_asset(Path(uid) / f"{uid}.xml") == uid
    assert _object_uid_from_rel_asset(Path(f"{uid}.xml")) == uid
    # Nothing UID-shaped to recover.
    assert _object_uid_from_rel_asset(Path(f"{uid}_visual.obj")) is None


def test_is_object_install_allowed_fails_closed_on_unknown_uid():
    """An unidentifiable asset must not slip through an active policy."""
    from molmo_spaces.utils.lazy_loading_utils import _is_object_install_allowed

    # No policy: nothing to check, so an unknown UID is fine.
    assert _is_object_install_allowed(None, "objaverse", LicensePolicy.NONE) is True
    assert _is_object_install_allowed(None, "thor", LicensePolicy.COMMERCIAL_SAFE) is True
    with pytest.raises(ValueError, match="Could not determine the object UID"):
        _is_object_install_allowed(None, "objaverse", LicensePolicy.COMMERCIAL_SAFE)


@pytest.mark.parametrize(
    "config_class_path",
    [
        "molmo_spaces.configs.task_sampler_configs.NavToObjTaskSamplerConfig",
        "molmo_spaces.configs.task_sampler_configs.DoorOpeningTaskSamplerConfig",
        "molmo_spaces.configs.task_sampler_configs.PickTaskSamplerConfig",
    ],
)
def test_apply_license_policy_to_configs_without_uid_pools(config_class_path, monkeypatch):
    """Configs that declare no UID pool must pass through, not raise AttributeError."""
    import importlib

    from molmo_spaces.utils.license_policy import apply_license_policy_to_task_sampler_config

    monkeypatch.setattr(
        "molmo_spaces.utils.object_metadata.ObjectMeta.is_commercial_use_allowed",
        lambda uid: uid != "nc_uid",
    )
    module_path, class_name = config_class_path.rsplit(".", 1)
    config = getattr(importlib.import_module(module_path), class_name)()

    apply_license_policy_to_task_sampler_config(config, LicensePolicy.COMMERCIAL_SAFE)

    if hasattr(config, "added_pickup_objects"):
        config.added_pickup_objects = ["ok_uid", "nc_uid"]
        apply_license_policy_to_task_sampler_config(config, LicensePolicy.COMMERCIAL_SAFE)
        assert config.added_pickup_objects == ["ok_uid"]


def test_locate_uid_package_cache_is_keyed_by_policy(monkeypatch):
    """A lookup made before the policy is resolved must not pin an unfiltered result."""
    from molmo_spaces.utils import grasps

    grasps._locate_uid_package_cached.cache_clear()

    def fake_locate(uid, license_policy=None):
        if license_policy == LicensePolicy.COMMERCIAL_SAFE and uid == "nc_uid":
            return (None, None, None)
        return ("objaverse", "pkg.tar.zst", "path.xml")

    monkeypatch.setattr(grasps, "locate_uid_package", fake_locate)

    set_license_policy(LicensePolicy.NONE)
    assert grasps._locate_uid_package("nc_uid")[0] == "objaverse"

    set_license_policy(LicensePolicy.COMMERCIAL_SAFE)
    assert grasps._locate_uid_package("nc_uid")[0] is None

    # The earlier entry is retained, not clobbered.
    set_license_policy(LicensePolicy.NONE)
    assert grasps._locate_uid_package("nc_uid")[0] == "objaverse"
    assert grasps._locate_uid_package_cached.cache_info().currsize == 2

    grasps._locate_uid_package_cached.cache_clear()


def test_get_valid_receptacle_uids_is_policy_independent(monkeypatch):
    """Receptacle lookup is cached process-wide, so it must not bake in a policy."""
    from molmo_spaces.utils import synset_utils

    monkeypatch.setattr(
        "molmo_spaces.utils.object_metadata.ObjectMeta.annotation",
        lambda *a: {
            "ok_uid": {"synset": "bowl.n.01", "receptacle": True},
            "nc_uid": {"synset": "bowl.n.01", "receptacle": True},
        },
    )
    monkeypatch.setattr(synset_utils, "is_valid_receptacle_synset", lambda synset: True)
    monkeypatch.setattr(
        "molmo_spaces.utils.object_metadata.ObjectMeta.is_commercial_use_allowed",
        lambda uid: uid != "nc_uid",
    )

    set_license_policy(LicensePolicy.NONE)
    under_none = list(synset_utils.get_valid_receptacle_uids())
    set_license_policy(LicensePolicy.COMMERCIAL_SAFE)
    under_safe = list(synset_utils.get_valid_receptacle_uids())

    assert under_none == under_safe == ["ok_uid", "nc_uid"]
    # Filtering happens at the point of use instead.
    assert filter_uids(under_safe, LicensePolicy.COMMERCIAL_SAFE) == ["ok_uid"]
