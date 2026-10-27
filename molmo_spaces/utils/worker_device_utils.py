"""Per-worker policy device and policy-server assignment for parallel data generation."""

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.policy.base_policy import BasePolicy


def resolve_worker_policy_device(base_device: str | None, worker_id: int) -> str:
    """Resolve a worker-local policy device."""
    resolved_device = str(base_device or "cpu").strip()
    if not resolved_device.startswith("cuda"):
        return resolved_device

    if not torch.cuda.is_available():
        return resolved_device

    if resolved_device != "cuda":
        return resolved_device

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 1:
        return resolved_device

    return f"cuda:{worker_id % visible_gpu_count}"


def resolve_cuda_device_index(device: str | None) -> int | None:
    resolved_device = str(device or "").strip().lower()
    if not resolved_device.startswith("cuda"):
        return None
    if resolved_device == "cuda":
        return 0
    if ":" not in resolved_device:
        return None
    try:
        return int(resolved_device.split(":", 1)[1])
    except ValueError:
        return None


def configure_worker_policy_device(
    exp_config: "MlSpacesExpConfig",
    preloaded_policy: "BasePolicy | None",
    worker_id: int,
    worker_logger,
) -> None:
    """Assign a policy device and remote server for this worker process."""
    policy_config = getattr(exp_config, "policy_config", None)
    if policy_config is None or not hasattr(policy_config, "device"):
        return

    base_device = getattr(policy_config, "device", None)
    resolved_device = resolve_worker_policy_device(base_device, worker_id)
    policy_config.device = resolved_device
    render_device_index = resolve_cuda_device_index(resolved_device)
    remote_config = getattr(policy_config, "remote_config", None)
    selected_server_url = None

    if isinstance(remote_config, dict):
        server_urls = remote_config.get("server_urls") or []
        if isinstance(server_urls, (list, tuple)) and server_urls:
            selected_server_url = str(server_urls[worker_id % len(server_urls)])
            remote_config["selected_server_url"] = selected_server_url
        elif remote_config.get("server_url"):
            selected_server_url = str(remote_config["server_url"])
            remote_config["selected_server_url"] = selected_server_url

    if render_device_index is None:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_device_index)

    if preloaded_policy is not None and hasattr(preloaded_policy, "device"):
        preloaded_policy.device = resolved_device

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    worker_logger.info(
        "Worker %s policy device assignment: base=%s resolved=%s visible_gpus=%s "
        "mujoco_egl_device=%s remote_server=%s",
        worker_id,
        base_device,
        resolved_device,
        visible_gpu_count,
        os.environ.get("MUJOCO_EGL_DEVICE_ID", "<unset>"),
        selected_server_url or "<none>",
    )
