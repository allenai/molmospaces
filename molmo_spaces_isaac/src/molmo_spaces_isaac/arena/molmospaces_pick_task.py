"""Pick task for MolmoSpaces benchmark: success = lift_height >= succ_pos_threshold."""

from __future__ import annotations

import logging
from typing import Any

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_isaac_lab

log = logging.getLogger(__name__)

try:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
    from isaaclab_arena.tasks.task_base import TaskBase

    _ARENA_AVAILABLE = True
except ImportError:
    IsaacLabArenaManagerBasedRLEnvCfg = None  # type: ignore[misc, assignment]
    TaskBase = None
    _ARENA_AVAILABLE = False



# Steps to track running-minimum z for the resting baseline after each reset.
_BASELINE_SETTLE_STEPS = 60
# Consecutive steps the object must be above the lift threshold to confirm a genuine lift
# (not a transient bounce from initial landing).
_LIFT_CONFIRM_STEPS = 5


def _make_pick_success_func(pick_name: str, pick_start_z: float, lift_threshold_m: float):
    """Return termination func: success when object Z exceeds baseline by lift_threshold_m for _LIFT_CONFIRM_STEPS consecutive steps."""
    _baseline: list[torch.Tensor | None] = [None]
    _consec: list[torch.Tensor | None] = [None]  # per-env consecutive-above-threshold counter
    _init_z = float(pick_start_z)

    def _pick_success(env) -> Any:
        try:
            base_env = getattr(env, "unwrapped", env)
            scene = base_env.scene
            current_z = scene[pick_name].data.root_pos_w[..., 2]
            n = current_z.shape[0] if current_z.dim() > 0 else 1
            device = current_z.device

            ep_buf = getattr(base_env, "episode_length_buf", None)
            if _baseline[0] is None:
                _baseline[0] = current_z.detach().clone()
                _consec[0] = torch.zeros(n, dtype=torch.long, device=device)
            elif ep_buf is not None:
                # During the settle window, track the minimum z (resting position).
                in_settle = ep_buf <= _BASELINE_SETTLE_STEPS
                if in_settle.any():
                    bl = _baseline[0].to(device=device).clone()
                    just_reset = ep_buf <= 1
                    bl[just_reset] = current_z[just_reset].detach()
                    later_settle = in_settle & ~just_reset
                    if later_settle.any():
                        bl[later_settle] = torch.min(bl[later_settle], current_z[later_settle].detach())
                    _baseline[0] = bl
                # Reset consecutive counter on episode reset.
                if (ep_buf <= 1).any():
                    cc = _consec[0].to(device=device).clone()
                    cc[ep_buf <= 1] = 0
                    _consec[0] = cc

            baseline = _baseline[0].to(device=device) if _baseline[0] is not None else torch.full((n,), _init_z, device=device)
            above = (current_z - baseline) >= lift_threshold_m

            # Update consecutive counter: increment if above, reset to 0 if not.
            cc = _consec[0].to(device=device) if _consec[0] is not None else torch.zeros(n, dtype=torch.long, device=device)
            cc = torch.where(above, cc + 1, torch.zeros_like(cc))
            _consec[0] = cc

            return cc >= _LIFT_CONFIRM_STEPS
        except (KeyError, AttributeError) as e:
            log.debug("Pick success check failed: %s", e)
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return _pick_success


@configclass
class MolmoSpacesPickTerminationsCfg:
    # time_out=True so this term sets truncated (not terminated); zero policy yields FAIL.
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    object_picked: TerminationTermCfg | None = None


class MolmoSpacesPickTask(TaskBase):
    """Pick task: success = lift_height >= succ_pos_threshold."""

    DEFAULT_EPISODE_LENGTH_S: float = 60.0

    def __init__(
        self,
        pick_up_object,
        background_scene,
        *,
        episode_length_s: float | None = None,
        pick_start_z: float | None = None,
        pick_lift_threshold_m: float = 0.01,
    ):
        if not _ARENA_AVAILABLE:
            raise ImportError("isaaclab_arena is required for MolmoSpacesPickTask.")
        if pick_start_z is None:
            raise ValueError("pick_start_z is required for MolmoSpacesPickTask.")
        self.pick_up_object = pick_up_object
        self.background_scene = background_scene
        self.episode_length_s = episode_length_s if episode_length_s is not None else self.DEFAULT_EPISODE_LENGTH_S
        pick_name = getattr(pick_up_object, "name", "pick_object")
        self._termination_cfg = MolmoSpacesPickTerminationsCfg()
        self._termination_cfg.object_picked = TerminationTermCfg(
            func=_make_pick_success_func(pick_name, pick_start_z, pick_lift_threshold_m)
        )

    def get_episode_length_s(self) -> float:
        return self.episode_length_s

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self._termination_cfg

    def get_events_cfg(self):
        return None

    def get_prompt(self) -> str:
        return "Pick the object."

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("MolmoSpacesPickTask has no mimic config.")

    def get_metrics(self):
        return []

    def modify_env_cfg(self, env_cfg: "IsaacLabArenaManagerBasedRLEnvCfg") -> "IsaacLabArenaManagerBasedRLEnvCfg":
        """Enable CCD and raise PhysX solver iterations to prevent THOR objects tunnelling through the scene."""
        if IsaacLabArenaManagerBasedRLEnvCfg is None:
            return env_cfg
        sim = getattr(env_cfg, "sim", None)
        if sim is not None and hasattr(sim, "physx") and sim.physx is not None:
            sim.physx.enable_ccd = True
            if getattr(sim.physx, "min_position_iteration_count", 1) < 4:
                sim.physx.min_position_iteration_count = 4
        return env_cfg
