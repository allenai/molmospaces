"""Pick task for MolmoSpaces benchmark: success = lift_height >= succ_pos_threshold."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_isaac_lab

log = logging.getLogger(__name__)

try:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )
except Exception:
    IsaacLabArenaManagerBasedRLEnvCfg = None  # type: ignore[misc, assignment]

try:
    from isaaclab_arena.tasks.task_base import TaskBase
except Exception:

    class TaskBase:  # type: ignore[no-redef]
        """Small compatibility base when Arena's TaskBase import pulls optional embodiments."""

        def __init__(self, episode_length_s: float | None = None, task_description: str | None = None):
            self.episode_length_s = episode_length_s
            self.task_description = task_description

        def get_observation_cfg(self):
            return None

        def get_rewards_cfg(self):
            return None

        def get_curriculum_cfg(self):
            return None

        def get_commands_cfg(self):
            return None

        def get_recorder_term_cfg(self):
            return None

        def modify_env_cfg(self, env_cfg):
            return env_cfg

        def get_viewer_cfg(self):
            return None

        def get_episode_length_s(self) -> float | None:
            return self.episode_length_s

        def get_task_description(self) -> str | None:
            return self.task_description


_ARENA_AVAILABLE = True


# MuJoCo's ``PickTask`` treats an object as supported when a contact point is in
# the lower half of the object's visual AABB and the contact normal is within
# 30 degrees of vertical.  In Arena we reconstruct the visual AABB center as a
# local offset from the pickup rigid body and apply the same support-contact test
# to non-robot scene contacts.
_SUPPORT_NORMAL_Z_THRESHOLD = math.cos(math.radians(30))
_SUPPORT_FORCE_THRESHOLD_N = 1e-5
_SUPPORT_POINT_CENTER_EPS_M = 0.0


def _contact_glob(prim_path_template: str) -> str:
    return prim_path_template.replace("{ENV_REGEX_NS}", "/World/envs/env_*").replace(".*", "*")


def _contact_env0_path(prim_path_template: str) -> str:
    return prim_path_template.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace(".*", "*")


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_rotate_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q_vec = quat[..., 1:]
    q_w = quat[..., :1]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def _make_pick_success_func(
    pick_name: str,
    pick_prim_path: str,
    pick_start_z: float,
    lift_threshold_m: float,
    support_exclude_prim_path: str | None = None,
):
    """Return termination func matching MolmoSpaces/MuJoCo ``PickTask.get_info``.

    MuJoCo success is:

    ``lift_height >= succ_pos_threshold``
    and ``(supporting_geom is None or supporting_geom in robot_geoms)``

    where ``supporting_geom`` is a heuristic support contact, not any contact:
    a contact in the lower half of the visual AABB with a mostly vertical normal.
    Robot contacts are allowed because this Arena version only checks support
    contacts against non-robot scene rigid bodies.
    """
    _init_z = float(pick_start_z)
    _views: dict[str, Any] = {}

    def _visual_center_offset_local(
        env,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return visual AABB center offset in pickup-root local coordinates."""
        base_env = getattr(env, "unwrapped", env)
        device = root_pos_w.device
        dtype = root_pos_w.dtype
        if "visual_center_offset_local" in _views:
            return _views["visual_center_offset_local"].to(device=device, dtype=dtype)
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("USD stage is not available")
            prim_path = _contact_env0_path(support_exclude_prim_path or pick_prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"pickup visual root is not valid: {prim_path}")
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            center = bbox.GetCenter()
            center_w = torch.tensor(
                [center[0], center[1], center[2]],
                device=device,
                dtype=dtype,
            )
            root_pos0 = root_pos_w.reshape(-1, 3)[0]
            root_quat0 = root_quat_w.reshape(-1, 4)[0]
            offset_w = center_w - root_pos0
            offset_local = _quat_rotate_wxyz(
                _quat_conjugate_wxyz(root_quat0.unsqueeze(0)),
                offset_w.unsqueeze(0),
            )[0]
            _views["visual_center_offset_local"] = offset_local.detach().cpu()
        except Exception as e:
            if "visual_center_offset_failed" not in _views:
                log.debug(
                    "Could not compute visual AABB center offset for %s; falling back to root: %s",
                    pick_name,
                    e,
                )
                _views["visual_center_offset_failed"] = True
            _views["visual_center_offset_local"] = torch.zeros(3)
        return _views["visual_center_offset_local"].to(device=device, dtype=dtype)

    def _visual_center_z(
        env,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        offset_local = _visual_center_offset_local(env, root_pos_w, root_quat_w)
        offsets = offset_local.unsqueeze(0).expand(root_pos_w.reshape(-1, 3).shape[0], -1)
        center_w = root_pos_w.reshape(-1, 3) + _quat_rotate_wxyz(
            root_quat_w.reshape(-1, 4),
            offsets,
        )
        return center_w[..., 2]

    def _scene_support_filter_patterns() -> list[str]:
        """Return non-pickup iTHOR scene rigid-body paths as env-wildcard patterns."""
        if "scene_support_filter_patterns" in _views:
            return _views["scene_support_filter_patterns"]
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return []
            scene_prim = stage.GetPrimAtPath("/World/envs/env_0/molmospaces_scene")
            if not scene_prim.IsValid():
                return []
            exclude_path_env0 = _contact_env0_path(support_exclude_prim_path or pick_prim_path)
            patterns: set[str] = set()
            for prim in Usd.PrimRange(scene_prim):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                path = str(prim.GetPath())
                if path == exclude_path_env0 or path.startswith(f"{exclude_path_env0}/"):
                    continue
                patterns.add(path.replace("/World/envs/env_0/", "/World/envs/env_*/"))
            _views["scene_support_filter_patterns"] = sorted(patterns)
            return _views["scene_support_filter_patterns"]
        except Exception as e:
            if "scene_filter_failed" not in _views:
                log.debug("Could not enumerate scene support filters for %s: %s", pick_name, e)
                _views["scene_filter_failed"] = True
            _views["scene_support_filter_patterns"] = []
            return []

    def _robot_support_filter_patterns() -> list[str]:
        if "robot_support_filter_patterns" in _views:
            return _views["robot_support_filter_patterns"]
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
                if robot_prim.IsValid():
                    patterns: set[str] = set()
                    for prim in Usd.PrimRange(robot_prim):
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            path = str(prim.GetPath())
                            patterns.add(path.replace("/World/envs/env_0/", "/World/envs/env_*/"))
                    if patterns:
                        _views["robot_support_filter_patterns"] = sorted(patterns)
                        return _views["robot_support_filter_patterns"]
        except Exception as e:
            if "robot_filter_failed" not in _views:
                log.debug("Could not enumerate robot support filters for %s: %s", pick_name, e)
                _views["robot_filter_failed"] = True

        _views["robot_support_filter_patterns"] = [
            "/World/envs/env_*/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
            "/World/envs/env_*/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
        ]
        return _views["robot_support_filter_patterns"]

    def _max_contact_data_count(filter_count: int) -> int:
        env_value = (os.environ.get("MOLMO_ARENA_MAX_CONTACT_DATA_COUNT") or "").strip()
        if env_value:
            return max(1, int(env_value))
        return max(256, min(4096, max(1, filter_count) * 8))

    def _make_contact_view(key: str, filter_patterns: list[str]) -> Any:
        if not filter_patterns:
            raise RuntimeError(f"no {key} support filter patterns")
        if key not in _views:
            from isaacsim.core.simulation_manager import SimulationManager

            sim_view = SimulationManager.get_physics_sim_view()
            last_error: Exception | None = None
            if support_exclude_prim_path and pick_prim_path != support_exclude_prim_path:
                body_patterns = (
                    _contact_glob(pick_prim_path),
                    _contact_glob(f"{pick_prim_path}/*"),
                    _contact_glob(f"{pick_prim_path}/*/*"),
                    _contact_glob(f"{pick_prim_path}/Geometry/*"),
                )
            else:
                body_patterns = (
                    _contact_glob(f"{pick_prim_path}/Geometry/*"),
                    _contact_glob(pick_prim_path),
                    _contact_glob(f"{pick_prim_path}/*"),
                    _contact_glob(f"{pick_prim_path}/*/*"),
                )
            for body_pattern in body_patterns:
                try:
                    _views[key] = sim_view.create_rigid_contact_view(
                        body_pattern,
                        filter_patterns=filter_patterns,
                        max_contact_data_count=_max_contact_data_count(len(filter_patterns)),
                    )
                    _views[f"{key}_body_pattern"] = body_pattern
                    _views[f"{key}_filter_count_requested"] = len(filter_patterns)
                    break
                except Exception as e:
                    last_error = e
            if key not in _views:
                raise RuntimeError(
                    f"no pickup contact view matched for {pick_prim_path}: {last_error}"
                )
        return _views[key]

    def _get_support_contacts(
        env,
        key: str,
        center_z: torch.Tensor,
        filter_patterns: list[str],
    ) -> torch.Tensor:
        base_env = getattr(env, "unwrapped", env)
        n = int(base_env.num_envs)
        device = base_env.device
        try:
            view = _make_contact_view(key, filter_patterns)
            filter_count = int(getattr(view, "filter_count", 0))
            if filter_count <= 0:
                return torch.zeros(n, dtype=torch.bool, device=device)
            sim_cfg = getattr(getattr(base_env, "sim", None), "cfg", object())
            dt_value = getattr(base_env, "physics_dt", None)
            if dt_value is None:
                dt_value = getattr(sim_cfg, "dt", 1.0 / 60.0)
            dt = float(dt_value)
            forces, points, normals, _distances, counts, starts = view.get_contact_data(dt=dt)
            counts_flat = counts.reshape(-1).to(device=device, dtype=torch.long)
            if int(counts_flat.sum().item()) <= 0:
                return torch.zeros(n, dtype=torch.bool, device=device)

            sensor_count = int(getattr(view, "sensor_count", int(counts.shape[0])))
            body_count = max(1, sensor_count // max(1, n))
            pair_ids = torch.nonzero(counts_flat > 0, as_tuple=False).flatten()
            pair_counts = counts_flat[pair_ids]
            pair_rows = pair_ids // filter_count
            pair_envs = torch.clamp(pair_rows // body_count, max=n - 1)

            contact_pair_ids = torch.repeat_interleave(pair_ids, pair_counts)
            contact_envs = torch.repeat_interleave(pair_envs, pair_counts)
            block_starts = pair_counts.cumsum(0) - pair_counts
            deltas = torch.arange(
                contact_pair_ids.numel(), device=device
            ) - torch.repeat_interleave(block_starts, pair_counts)
            flat_indices = (
                starts.reshape(-1).to(device=device, dtype=torch.long)[contact_pair_ids] + deltas
            )

            contact_forces = forces.reshape(-1).to(device=device)[flat_indices]
            contact_points = points.reshape(-1, 3).to(device=device)[flat_indices]
            contact_normals = normals.reshape(-1, 3).to(device=device)[flat_indices]
            center_z_for_contact = center_z.reshape(-1).to(device=device)[contact_envs]
            support_like = (
                (contact_forces.abs() > _SUPPORT_FORCE_THRESHOLD_N)
                & (contact_normals[:, 2].abs() >= _SUPPORT_NORMAL_Z_THRESHOLD)
                & (contact_points[:, 2] < center_z_for_contact - _SUPPORT_POINT_CENTER_EPS_M)
            )

            supported = torch.zeros(n, dtype=torch.bool, device=device)
            if bool(support_like.any().item()):
                supported[contact_envs[support_like].unique()] = True
            return supported
        except Exception as e:
            failed_key = f"{key}_failed"
            if failed_key not in _views:
                log.debug(
                    "Could not initialize %s support contact view for %s: %s",
                    key,
                    pick_name,
                    e,
                )
                _views[failed_key] = True
            return torch.zeros(n, dtype=torch.bool, device=device)

    def _pick_success(env) -> Any:
        try:
            base_env = getattr(env, "unwrapped", env)
            scene = base_env.scene
            pick_object = scene[pick_name]
            root_pos_w = pick_object.data.root_pos_w
            root_quat_w = pick_object.data.root_quat_w
            current_z = root_pos_w[..., 2]
            center_z = _visual_center_z(env, root_pos_w, root_quat_w)
            n = current_z.shape[0] if current_z.dim() > 0 else 1
            device = current_z.device

            baseline = torch.full((n,), _init_z, dtype=current_z.dtype, device=device)
            lifted = (current_z.reshape(-1) - baseline) >= lift_threshold_m
            scene_supported = _get_support_contacts(
                env,
                "scene",
                center_z,
                _scene_support_filter_patterns(),
            )

            return lifted & ~scene_supported
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
        terminate_on_success: bool = True,
    ):
        if not _ARENA_AVAILABLE:
            raise ImportError("isaaclab_arena is required for MolmoSpacesPickTask.")
        if pick_start_z is None:
            raise ValueError("pick_start_z is required for MolmoSpacesPickTask.")
        self.pick_up_object = pick_up_object
        self.background_scene = background_scene
        self.episode_length_s = (
            episode_length_s if episode_length_s is not None else self.DEFAULT_EPISODE_LENGTH_S
        )
        pick_name = getattr(pick_up_object, "name", "pick_object")
        pick_prim_path = getattr(pick_up_object, "prim_path", f"{{ENV_REGEX_NS}}/{pick_name}")
        support_exclude_prim_path = getattr(pick_up_object, "scene_object_root_prim_path", pick_prim_path)
        self._termination_cfg = MolmoSpacesPickTerminationsCfg()
        self._pick_success_func = _make_pick_success_func(
            pick_name,
            pick_prim_path,
            pick_start_z,
            pick_lift_threshold_m,
            support_exclude_prim_path=support_exclude_prim_path,
        )
        if terminate_on_success:
            self._termination_cfg.object_picked = TerminationTermCfg(func=self._pick_success_func)

    def get_episode_length_s(self) -> float:
        return self.episode_length_s

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self._termination_cfg

    def pick_success(self, env):
        return self._pick_success_func(env)

    def get_events_cfg(self):
        return None

    def get_prompt(self) -> str:
        return "Pick the object."

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("MolmoSpacesPickTask has no mimic config.")

    def get_metrics(self):
        return []

    def modify_env_cfg(
        self,
        env_cfg: "IsaacLabArenaManagerBasedRLEnvCfg",
    ) -> "IsaacLabArenaManagerBasedRLEnvCfg":
        """Enable CCD and raise PhysX solver iterations to prevent THOR objects tunnelling."""
        if IsaacLabArenaManagerBasedRLEnvCfg is None:
            return env_cfg
        sim = getattr(env_cfg, "sim", None)
        if sim is not None and hasattr(sim, "physx") and sim.physx is not None:
            sim.physx.enable_ccd = True
            if getattr(sim.physx, "min_position_iteration_count", 1) < 4:
                sim.physx.min_position_iteration_count = 4
        return env_cfg
