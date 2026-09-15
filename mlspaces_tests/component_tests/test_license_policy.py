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
