"""Pick-and-place task with MolmoSpaces success (geometric supported + displacement) or pose-based success_radius."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_isaac_lab

log = logging.getLogger(__name__)

try:
    from isaaclab_arena.tasks.task_base import TaskBase

    _ARENA_AVAILABLE = True
except ImportError:
    TaskBase = None
    _ARENA_AVAILABLE = False


def _make_pose_success_func(pick_name: str, dest_name: str, radius: float = 0.12):
    """Return a termination function: True when pick object center is within radius of destination center."""

    def _object_in_destination(env) -> Any:
        try:
            base_env = getattr(env, "unwrapped", env)
            scene = base_env.scene
            pick_pos = scene[pick_name].data.root_pos_w
            dest_pos = scene[dest_name].data.root_pos_w
            dist = (pick_pos - dest_pos).norm(dim=-1)
            return dist < radius
        except (KeyError, AttributeError) as e:
            log.debug("Pose success check failed: %s", e)
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return _object_in_destination


def _make_object_lifted_success_func(
    pick_name: str,
    pick_start_z: float,
    lift_threshold_m: float = 0.01,
):
    """True when pick object is lifted above baseline by at least lift_threshold_m (Z-up)."""
    MIN_Z_TO_CACHE = 0.15
    initial_z_ref: list[torch.Tensor | None] = [None]

    def _object_lifted(env) -> Any:
        try:
            base_env = getattr(env, "unwrapped", env)
            scene = base_env.scene
            pick_pos = scene[pick_name].data.root_pos_w
            current_z = pick_pos[..., 2]
            if initial_z_ref[0] is None:
                # Only cache when we have a plausible height (avoid caching 0/stale buffer)
                if (current_z > MIN_Z_TO_CACHE).all():
                    initial_z_ref[0] = current_z.detach().clone()
                return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            lift_height = current_z - initial_z_ref[0]
            return lift_height >= lift_threshold_m
        except (KeyError, AttributeError) as e:
            log.debug("Object lifted check failed: %s", e)
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return _object_lifted


def _make_molmospaces_success_func(
    pick_name: str,
    dest_name: str,
    place_start_pose_7: list[float],
    *,
    fallback_thres: float = 0.01,
    max_place_receptacle_pos_displacement: float = 0.1,
    max_place_receptacle_rot_displacement: float = 0.785,
    default_extent: float = 0.06,
):
    """Termination: MolmoSpaces success (geometric supported + receptacle stable)."""

    from molmo_spaces_isaac.arena.molmo_success import molmospaces_pick_and_place_success

    place_start = torch.tensor(place_start_pose_7, dtype=torch.float32)

    def _object_placed_molmo(env) -> Any:
        try:
            base_env = getattr(env, "unwrapped", env)
            scene = base_env.scene
            pick_pos = scene[pick_name].data.root_pos_w
            place_pos = scene[dest_name].data.root_pos_w
            data_place = scene[dest_name].data
            quat = getattr(data_place, "root_quat_w", None)
            if quat is None:
                quat = getattr(data_place, "root_quat_world", None)
            if quat is None:
                log.debug("Place object has no root_quat_w; using identity")
                quat = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],
                    device=env.device,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(pick_pos.shape[0], 4)
            else:
                quat = quat.to(env.device)
                if quat.shape[-1] == 4 and quat.dim() == 2:
                    if quat[0, 0].abs() < 0.9 and quat[0, 3].abs() > 0.9:
                        quat = quat[:, [3, 0, 1, 2]]
                elif quat.dim() == 1 and quat.shape[0] == 4:
                    if quat[0].abs() < 0.9 and quat[3].abs() > 0.9:
                        quat = quat[[3, 0, 1, 2]].unsqueeze(0).expand(pick_pos.shape[0], 4)
                    else:
                        quat = quat.unsqueeze(0).expand(pick_pos.shape[0], 4)
            ext = torch.tensor(
                [default_extent, default_extent, default_extent],
                device=env.device,
                dtype=torch.float32,
            )
            place_start_dev = place_start.to(env.device)
            return molmospaces_pick_and_place_success(
                pick_center=pick_pos,
                pick_extent=ext.unsqueeze(0).expand_as(pick_pos),
                place_center=place_pos,
                place_extent=ext.unsqueeze(0).expand_as(place_pos),
                place_start_pose_7=place_start_dev.unsqueeze(0).expand(pick_pos.shape[0], 7),
                place_current_pos=place_pos,
                place_current_quat_wxyz=quat,
                fallback_thres=fallback_thres,
                max_place_receptacle_pos_displacement=max_place_receptacle_pos_displacement,
                max_place_receptacle_rot_displacement=max_place_receptacle_rot_displacement,
            )
        except Exception as e:
            log.debug("MolmoSpaces success check failed: %s", e)
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return _object_placed_molmo


@configclass
class MolmoSpacesPickAndPlaceSceneCfg:
    """No contact sensor; keeps THOR USDs and Arena unchanged."""


@configclass
class MolmoSpacesPickAndPlaceTerminationsCfg:
    # time_out=True so this term sets truncated (not terminated). Zero policy then yields FAIL.
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    object_placed: TerminationTermCfg | None = None


class MolmoSpacesPickAndPlaceTask(TaskBase):
    """Pick/pick_and_place with MolmoSpaces success (lift height for pick; supported+displacement for place) or pose-based."""

    DEFAULT_EPISODE_LENGTH_S: float = 60.0

    def __init__(
        self,
        pick_up_object,
        destination_location,
        background_scene,
        *,
        success_radius: float = 0.12,
        episode_length_s: float | None = None,
        use_molmospaces_success: bool = True,
        place_receptacle_start_pose_7: list[float] | None = None,
        max_place_receptacle_pos_displacement: float = 0.1,
        max_place_receptacle_rot_displacement: float = 0.785,
        supported_fallback_thres: float = 0.01,
        default_extent: float = 0.06,
        task_type: str = "pick_and_place",
        pick_start_z: float | None = None,
        pick_lift_threshold_m: float = 0.01,
        get_contact_supported: Callable[[Any], torch.Tensor | None] | None = None,
    ):
        if not _ARENA_AVAILABLE:
            raise ImportError("isaaclab_arena is required for MolmoSpacesPickAndPlaceTask.")
        self.pick_up_object = pick_up_object
        self.destination_location = destination_location
        self.background_scene = background_scene
        self.success_radius = success_radius
        self.episode_length_s = (
            episode_length_s if episode_length_s is not None else self.DEFAULT_EPISODE_LENGTH_S
        )
        pick_name = getattr(pick_up_object, "name", "pick_object")
        dest_name = getattr(destination_location, "name", "destination")

        # Pick-only: success = object lifted above start z (mirrors MolmoSpaces PickTask).
        if task_type == "pick" and pick_start_z is not None:
            success_func = _make_object_lifted_success_func(
                pick_name, pick_start_z, lift_threshold_m=pick_lift_threshold_m
            )
        else:
            use_molmo = use_molmospaces_success and place_receptacle_start_pose_7 is not None
            if use_molmo:
                success_func = _make_molmospaces_success_func(
                    pick_name,
                    dest_name,
                    place_receptacle_start_pose_7,
                    get_contact_supported=get_contact_supported,
                    fallback_thres=supported_fallback_thres,
                    max_place_receptacle_pos_displacement=max_place_receptacle_pos_displacement,
                    max_place_receptacle_rot_displacement=max_place_receptacle_rot_displacement,
                    default_extent=default_extent,
                )
            else:
                success_func = _make_pose_success_func(
                    pick_name, dest_name, radius=success_radius
                )
        self._termination_cfg = MolmoSpacesPickAndPlaceTerminationsCfg(
            object_placed=TerminationTermCfg(func=success_func),
        )

    def get_episode_length_s(self) -> float:
        """Episode length in seconds (used by Arena)."""
        return self.episode_length_s

    def get_scene_cfg(self):
        return MolmoSpacesPickAndPlaceSceneCfg()

    def get_termination_cfg(self):
        return self._termination_cfg

    def get_events_cfg(self):
        return None

    def get_prompt(self) -> str:
        return "Pick the object and place it in the destination."

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("MolmoSpacesPickAndPlaceTask has no mimic config.")

    def get_metrics(self):
        return []
