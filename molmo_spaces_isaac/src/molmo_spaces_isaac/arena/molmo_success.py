"""MolmoSpaces pick-and-place success for Arena: geometry-only (supported_by_receptacle + receptacle displacement limits)."""

from __future__ import annotations

from typing import Any

import torch


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    t = torch.tensor(x, device=device, dtype=dtype)
    return t


def supported_by_receptacle_geometric(
    pick_center: torch.Tensor,
    pick_extent: torch.Tensor,
    place_center: torch.Tensor,
    place_extent: torch.Tensor,
    fallback_thres: float = 0.01,
) -> torch.Tensor:
    """True where pick center is inside receptacle top AABB (xy) and base z in [receptacle_top - half_h, receptacle_top + fallback_thres]."""
    device = pick_center.device
    pick_center = _to_tensor(pick_center, device).reshape(-1, 3)
    pick_extent = _to_tensor(pick_extent, device).reshape(-1, 3)
    place_center = _to_tensor(place_center, device).reshape(-1, 3)
    place_extent = _to_tensor(place_extent, device).reshape(-1, 3)
    n = pick_center.shape[0]
    if place_center.shape[0] == 1 and n > 1:
        place_center = place_center.expand(n, 3)
        place_extent = place_extent.expand(n, 3)
    if pick_extent.shape[0] == 1 and n > 1:
        pick_extent = pick_extent.expand(n, 3)

    place_half = place_extent / 2
    place_top_z = place_center[:, 2] + place_half[:, 2]
    place_min_xy = place_center[:, :2] - place_half[:, :2]
    place_max_xy = place_center[:, :2] + place_half[:, :2]

    pick_half = pick_extent / 2
    pick_base_z = pick_center[:, 2] - pick_half[:, 2]

    # xy: pick center inside receptacle top face
    in_xy = (
        (pick_center[:, 0] >= place_min_xy[:, 0])
        & (pick_center[:, 0] <= place_max_xy[:, 0])
        & (pick_center[:, 1] >= place_min_xy[:, 1])
        & (pick_center[:, 1] <= place_max_xy[:, 1])
    )
    # z: object base between (receptacle top - half height) and (receptacle top + fallback_thres)
    z_lo = place_top_z - place_half[:, 2]
    z_hi = place_top_z + fallback_thres
    in_z = (pick_base_z >= z_lo) & (pick_base_z <= z_hi)

    return in_xy & in_z


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix (batch or single)."""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _rotmat_to_magnitude(R: torch.Tensor) -> torch.Tensor:
    """Rotation magnitude in radians. angle = arccos(clamp((trace(R)-1)/2, -1, 1))."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = ((trace - 1) / 2).clamp(-1.0, 1.0)
    return torch.acos(cos_angle)


def receptacle_displacement_ok(
    place_start_pose_7: torch.Tensor,
    place_current_pos: torch.Tensor,
    place_current_quat_wxyz: torch.Tensor,
    max_pos_displacement: float,
    max_rot_displacement_rad: float,
) -> torch.Tensor:
    """True where pos and rot displacement (in start frame) are within max_pos and max_rot_rad."""
    device = place_current_pos.device
    dtype = place_current_pos.dtype
    start = _to_tensor(place_start_pose_7, device, dtype)
    if start.dim() == 1:
        start = start.unsqueeze(0)
    pos_cur = _to_tensor(place_current_pos, device, dtype).reshape(-1, 3)
    quat_cur = _to_tensor(place_current_quat_wxyz, device, dtype).reshape(-1, 4)
    n = pos_cur.shape[0]
    if start.shape[0] == 1 and n > 1:
        start = start.expand(n, 7)

    start_pos = start[:, :3]
    start_quat = start[:, 3:7]
    R_start = _quat_wxyz_to_rotmat(start_quat)
    R_cur = _quat_wxyz_to_rotmat(quat_cur)
    # displacement in start frame: inv(T_start) @ T_cur
    R_disp = R_start.transpose(-2, -1) @ R_cur
    t_disp = (R_start.transpose(-2, -1) @ (pos_cur - start_pos).unsqueeze(-1)).squeeze(-1)
    pos_norm = t_disp.norm(dim=-1)
    rot_rad = _rotmat_to_magnitude(R_disp)
    ok = (pos_norm <= max_pos_displacement) & (rot_rad <= max_rot_displacement_rad)
    return ok


def molmospaces_pick_and_place_success(
    pick_center: torch.Tensor,
    pick_extent: torch.Tensor,
    place_center: torch.Tensor,
    place_extent: torch.Tensor,
    place_start_pose_7: torch.Tensor,
    place_current_pos: torch.Tensor,
    place_current_quat_wxyz: torch.Tensor,
    *,
    fallback_thres: float = 0.01,
    max_place_receptacle_pos_displacement: float = 0.1,
    max_place_receptacle_rot_displacement: float = 0.785,
) -> torch.Tensor:
    """Success = supported_by_receptacle (geometric) AND receptacle displacement within limits."""
    supported = supported_by_receptacle_geometric(
        pick_center, pick_extent, place_center, place_extent, fallback_thres=fallback_thres
    )
    stable = receptacle_displacement_ok(
        place_start_pose_7,
        place_current_pos,
        place_current_quat_wxyz,
        max_place_receptacle_pos_displacement,
        max_place_receptacle_rot_displacement,
    )
    return supported & stable
