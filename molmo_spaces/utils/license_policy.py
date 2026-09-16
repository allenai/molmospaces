"""Runtime license policy for datagen and asset inspection.

Eval code paths must not set or pass a non-default policy. Use
``molmo_spaces.molmo_spaces_constants.get_license_policy`` as the single
entry point for reading the active policy.
"""

from __future__ import annotations

import hashlib
import os
from contextvars import ContextVar
from enum import StrEnum

NON_COMMERCIAL_OBJAVERSE_LICENSES = frozenset({"by-nc", "by-nc-sa"})

LICENSE_POLICY_ENV_VAR = "MLSPACES_LICENSE_POLICY"

_NC_OBJAVERSE_UIDS_CACHE: frozenset[str] | None = None


class LicensePolicy(StrEnum):
    NONE = "none"
    COMMERCIAL_SAFE = "commercial_safe"


_LICENSE_POLICY: ContextVar[LicensePolicy] = ContextVar(
    "license_policy", default=LicensePolicy.NONE
)


def get_license_policy() -> LicensePolicy:
    return _LICENSE_POLICY.get()


def set_license_policy(policy: LicensePolicy | str) -> None:
    if isinstance(policy, str):
        policy = LicensePolicy(policy)
    _LICENSE_POLICY.set(policy)


def parse_license_policy(value: str | None) -> LicensePolicy:
    if value is None or value == "":
        return LicensePolicy.NONE
    return LicensePolicy(value)


def resolve_license_policy(
    explicit: str | LicensePolicy | None = None,
    *,
    config_policy: LicensePolicy | None = None,
) -> LicensePolicy:
    """Resolve the effective license policy.

    Precedence (highest first): explicit CLI/config override, non-NONE config
    field, ``MLSPACES_LICENSE_POLICY`` env var, default ``NONE``.
    """
    if explicit is not None and explicit != "":
        if isinstance(explicit, LicensePolicy):
            return explicit
        return parse_license_policy(explicit)
    if config_policy is not None and config_policy != LicensePolicy.NONE:
        return config_policy
    env_value = os.environ.get(LICENSE_POLICY_ENV_VAR)
    if env_value:
        return parse_license_policy(env_value)
    return LicensePolicy.NONE


def license_blocked_body_keys() -> frozenset[str]:
    """NC Objaverse UIDs plus MD5 hashes for fast scene-body lookup."""
    blocked = non_commercial_objaverse_uids()
    keys = set(blocked)
    for uid in blocked:
        keys.add(hashlib.md5(uid.encode()).hexdigest())
    return frozenset(keys)


def non_commercial_objaverse_uids() -> frozenset[str]:
    global _NC_OBJAVERSE_UIDS_CACHE
    if _NC_OBJAVERSE_UIDS_CACHE is None:
        from molmo_spaces.utils.object_metadata import ObjectMeta

        blocked = set()
        for uid, anno in ObjectMeta.annotation().items():
            if not anno.get("isObjaverse"):
                continue
            lic = (anno.get("license_info") or {}).get("license")
            if lic in NON_COMMERCIAL_OBJAVERSE_LICENSES:
                blocked.add(uid)
        _NC_OBJAVERSE_UIDS_CACHE = frozenset(blocked)
    return _NC_OBJAVERSE_UIDS_CACHE


def require_objaverse_license_known(uid: str) -> None:
    """Raise when an Objaverse asset lacks license metadata."""
    from molmo_spaces.utils.object_metadata import ObjectMeta

    anno = ObjectMeta.annotation(uid)
    if anno is None or not anno.get("isObjaverse"):
        return
    lic = (anno.get("license_info") or {}).get("license")
    if not lic:
        raise ValueError(
            f"Objaverse asset {uid!r} has no license metadata; "
            "cannot load under a commercial-safe license policy"
        )


def is_object_allowed(uid: str, policy: LicensePolicy) -> bool:
    if policy == LicensePolicy.NONE:
        return True
    if policy == LicensePolicy.COMMERCIAL_SAFE:
        from molmo_spaces.utils.object_metadata import ObjectMeta

        return ObjectMeta.is_commercial_use_allowed(uid)
    raise ValueError(f"Unknown {policy=}")


def filter_uids(uids, policy: LicensePolicy | None = None) -> list[str]:
    if policy is None:
        policy = get_license_policy()
    if policy == LicensePolicy.NONE:
        return list(uids)
    return [uid for uid in uids if is_object_allowed(uid, policy)]


def apply_license_policy_to_task_sampler_config(task_sampler_config, policy: LicensePolicy) -> None:
    """Filter config UID pools after the effective policy has been resolved."""
    if policy == LicensePolicy.NONE:
        return
    if task_sampler_config.added_pickup_objects:
        task_sampler_config.added_pickup_objects = filter_uids(
            task_sampler_config.added_pickup_objects, policy
        )


def is_scene_source_allowed(scene_dataset: str, policy: LicensePolicy | None = None) -> bool:
    if policy is None:
        policy = get_license_policy()
    if policy == LicensePolicy.NONE:
        return True
    if policy == LicensePolicy.COMMERCIAL_SAFE:
        return True
    raise ValueError(f"Unknown {policy=}")


def validate_datagen_license_policy(scene_dataset: str, policy: LicensePolicy) -> None:
    if policy == LicensePolicy.NONE:
        return
    if not is_scene_source_allowed(scene_dataset, policy):
        raise ValueError(
            f"scene_dataset={scene_dataset!r} is not allowed under license_policy={policy!r}"
        )
