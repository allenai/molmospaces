"""Ensure evaluation entrypoints do not expose license-policy filtering."""

from pathlib import Path


def test_eval_main_has_no_license_policy_cli():
    eval_main = Path(__file__).resolve().parents[2] / "molmo_spaces/evaluation/eval_main.py"
    source = eval_main.read_text()
    assert "license_policy" not in source
    assert "license-policy" not in source


def test_task_sampler_config_default_is_none():
    from molmo_spaces.configs.task_sampler_configs import BaseMujocoTaskSamplerConfig
    from molmo_spaces.utils.license_policy import LicensePolicy

    cfg = BaseMujocoTaskSamplerConfig(
        task_batch_size=1,
        house_inds=[0],
        samples_per_house=1,
        max_tasks=1,
    )
    assert cfg.license_policy == LicensePolicy.NONE
