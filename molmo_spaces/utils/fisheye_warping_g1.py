"""Cubemap-based fisheye renderer for MuJoCo (5 pinhole faces, equidistant warp).

Ported from the G1 humanoid research stack (g1_molmo/molmospaces/components/fisheye.py).
Unlike fisheye_warping.py (an offline single-image/video distortion utility built around
a fixed GoPro K/D), this module renders a live fisheye view directly from 5 named MuJoCo
"tile" cameras sharing an optical center, compositing them into one warped image per call.
Kept as a parallel module rather than merged with fisheye_warping.py.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

_FACES = (
    (
        "center",
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
    ),
    (
        "up",
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
        np.array([0, -1, 0], dtype=np.float32),
    ),
    (
        "down",
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 0, -1], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
    ),
    (
        "left",
        np.array([0, 0, -1], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([1, 0, 0], dtype=np.float32),
    ),
    (
        "right",
        np.array([0, 0, 1], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
        np.array([-1, 0, 0], dtype=np.float32),
    ),
)
HEAD_FISHEYE_K = np.array(
    [
        [801.6382129934864, 0.0, 976.1246839545557],
        [0.0, 802.1081824931498, 542.7122090223202],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
HEAD_FISHEYE_D = np.array(
    [
        -0.02559442829261663,
        0.008371943913215045,
        -0.006921566406199126,
        0.0010132813066123071,
    ],
    dtype=np.float64,
)
HEAD_FISHEYE_IMAGE_SIZE = (1920, 1080)  # (W, H) of the calibration capture.


class FisheyeRenderer:
    def __init__(
        self,
        model,
        tile_cam_names,
        tile_size=256,
        output_h=240,
        output_w=240,
        weight_power=4.0,
        K=None,
        D=None,
        image_size=None,
    ):
        """Pixel→ray uses the OpenCV fisheye model with the K/D constants at the top
        of this module (real-lens calibration), so the simulated view matches the
        real head camera. K, D, image_size can be passed in to override the
        module-level defaults (used by calibration tooling)."""
        if len(tile_cam_names) != 5:
            raise ValueError(f"need exactly 5 tile cameras, got {len(tile_cam_names)}")
        self.model = model
        self.tile_cam_ids = []
        for name in tile_cam_names:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"tile camera {name!r} not found in model")
            self.tile_cam_ids.append(cid)
        self.tile_size = int(tile_size)
        self.output_h = int(output_h)
        self.output_w = int(output_w)
        self.weight_power = float(weight_power)
        tile_fovy = float(model.cam_fovy[self.tile_cam_ids[0]])
        if tile_fovy < 90.0:
            raise ValueError(f"tile fovy {tile_fovy} too small; need >=90 (recommend 100)")
        self.tile_fovy_rad = np.radians(tile_fovy)
        self.K = (HEAD_FISHEYE_K if K is None else np.asarray(K, dtype=np.float64)).copy()
        self.D = (HEAD_FISHEYE_D if D is None else np.asarray(D, dtype=np.float64)).copy()
        self.image_size = tuple(HEAD_FISHEYE_IMAGE_SIZE if image_size is None else image_size)
        self._build_lut()

    def set_intrinsics(self, K=None, D=None):
        """Replace K and/or D in place and rebuild the LUT. Invalidates the GPU
        cache so the next render uploads fresh grids."""
        if K is not None:
            self.K[:] = K
        if D is not None:
            self.D[:] = D
        self._build_lut()
        self._gpu_ready = False
        self._grids_gpu = None
        self._weights_gpu = None
        self._weight_sum_gpu = None
        self._in_circle_gpu = None

    def project_camera_point(self, p_cam):
        """Forward-project a 3D point in the MuJoCo camera frame (+x right, +y up,
        -z forward) to a pixel (u, v) in the rendered fisheye output. This is the
        inverse of the pixel->ray LUT: the OpenCV equidistant fisheye model with
        this renderer's K/D, scaled to the output resolution. Returns None if the
        point is at/behind the optical plane (not imageable)."""
        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        X, Y, Z = x, -y, -z  # MuJoCo cam -> OpenCV cam (z forward, y down)
        if Z <= 1e-6:
            return None
        a, b = X / Z, Y / Z
        r = float(np.hypot(a, b))
        theta = np.arctan(r)
        k1, k2, k3, k4 = self.D
        theta_d = theta * (1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
        scale = (theta_d / r) if r > 1e-9 else 1.0
        xp, yp = a * scale, b * scale
        cal_w, cal_h = self.image_size
        sx = self.output_w / float(cal_w)
        sy = self.output_h / float(cal_h)
        u = self.K[0, 0] * sx * xp + self.K[0, 2] * sx
        v = self.K[1, 1] * sy * yp + self.K[1, 2] * sy
        return u, v

    def _build_lut(self):
        out_H, out_W = self.output_h, self.output_w
        ys, xs = np.mgrid[:out_H, :out_W].astype(np.float64)

        # OpenCV fisheye unprojection using the instance K, D (defaulting to the
        # module-level real-lens constants). Intrinsics scale from the calibration
        # image_size to the renderer's output size so the FOV/distortion shape
        # match at any resolution.
        cal_w, cal_h = self.image_size
        sx = out_W / float(cal_w)
        sy = out_H / float(cal_h)
        fx = self.K[0, 0] * sx
        fy = self.K[1, 1] * sy
        cx = self.K[0, 2] * sx
        cy = self.K[1, 2] * sy
        x_norm = (xs - cx) / fx
        y_norm = (ys - cy) / fy
        theta_d = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        phi = np.arctan2(y_norm, x_norm)
        # Invert theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        # via Newton-Raphson. theta_d <~1.3 rad even at lens edge -> converges in 10 iters.
        k1, k2, k3, k4 = self.D
        theta = theta_d.copy()
        for _ in range(10):
            t2 = theta * theta
            t4 = t2 * t2
            t6 = t4 * t2
            t8 = t4 * t4
            f = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
            df = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
            theta = theta - f / np.where(np.abs(df) > 1e-12, df, 1e-12)
        # Pixel is valid if it corresponds to a finite forward-hemisphere ray.
        in_circle = np.isfinite(theta) & (theta >= 0.0) & (theta < np.pi * 0.5 + 0.3)

        sin_t = np.sin(theta)
        rx = sin_t * np.cos(phi)
        ry = -sin_t * np.sin(phi)
        rz = -np.cos(theta)

        f = (self.tile_size * 0.5) / np.tan(self.tile_fovy_rad * 0.5)
        cx_t = cy_t = (self.tile_size - 1) * 0.5

        self._projs = []
        grids_norm = []
        weights = []
        for _, ax_v, ay_v, az_v in _FACES:
            fx = rx * ax_v[0] + ry * ax_v[1] + rz * ax_v[2]
            fy_ = rx * ay_v[0] + ry * ay_v[1] + rz * ay_v[2]
            fz = rx * az_v[0] + ry * az_v[1] + rz * az_v[2]
            depth = -fz
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx / depth * f + cx_t
                v = -fy_ / depth * f + cy_t
            in_bounds = (
                (depth > 1e-6)
                & (u >= 0)
                & (u <= self.tile_size - 1)
                & (v >= 0)
                & (v <= self.tile_size - 1)
            )
            weight = np.where(in_bounds, np.clip(depth, 0.0, 1.0) ** self.weight_power, 0.0)
            u = np.clip(u, 0, self.tile_size - 1).astype(np.float32)
            v = np.clip(v, 0, self.tile_size - 1).astype(np.float32)
            self._projs.append((u, v, weight.astype(np.float32)))
            gx = 2.0 * u / max(self.tile_size - 1, 1) - 1.0
            gy = 2.0 * v / max(self.tile_size - 1, 1) - 1.0
            grids_norm.append(np.stack([gx, gy], axis=-1).astype(np.float32))
            weights.append(weight.astype(np.float32))
        self._in_circle = in_circle

        # Pre-stage on GPU for torch grid_sample (lazy upload — happens on first render
        # to ensure CUDA is initialized in the right worker process).
        self._grids_np = np.stack(grids_norm, axis=0)  # (5, H, W, 2)
        self._weights_np = np.stack(weights, axis=0)  # (5, H, W)
        self._gpu_ready = False
        self._grids_gpu = None
        self._weights_gpu = None
        self._weight_sum_gpu = None
        self._in_circle_gpu = None

    def render(self, data, renderer, scene_option=None):
        # Cubemap faces share position but differ in orientation; MuJoCo's view-attached
        # headlight would shade each face differently, producing visible seams. Move
        # diffuse/specular into ambient (direction-independent) for the duration of the
        # 5 face renders, then restore.
        hl = self.model.vis.headlight
        orig_amb = hl.ambient.copy()
        orig_dif = hl.diffuse.copy()
        orig_spc = hl.specular.copy()
        hl.ambient[:] = orig_amb + orig_dif
        hl.diffuse[:] = 0.0
        hl.specular[:] = 0.0
        try:
            tiles = []
            # First face: full scene update (traverses all geoms + lights).
            if scene_option is not None:
                renderer.update_scene(data, self.tile_cam_ids[0], scene_option)
            else:
                renderer.update_scene(data, self.tile_cam_ids[0])
            tiles.append(renderer.render())
            # Remaining faces: scene/lights already populated, only camera changes.
            for cid in self.tile_cam_ids[1:]:
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cid
                mujoco.mjv_updateCamera(self.model, data, cam, renderer.scene)
                tiles.append(renderer.render())
        finally:
            hl.ambient[:] = orig_amb
            hl.diffuse[:] = orig_dif
            hl.specular[:] = orig_spc

        if torch.cuda.is_available():
            if not self._gpu_ready:
                dev = torch.device("cuda")
                self._grids_gpu = torch.from_numpy(self._grids_np).to(dev, non_blocking=True)
                self._weights_gpu = torch.from_numpy(self._weights_np).to(dev, non_blocking=True)
                self._weight_sum_gpu = self._weights_gpu.sum(dim=0).clamp(min=1e-6)
                self._in_circle_gpu = torch.from_numpy(self._in_circle).to(dev, non_blocking=True)
                self._gpu_ready = True
            dev = self._grids_gpu.device
            tiles_np = np.stack(tiles, axis=0)
            tiles_gpu = (
                torch.from_numpy(tiles_np).to(dev, non_blocking=True).float().permute(0, 3, 1, 2)
            )
            sampled = F.grid_sample(
                tiles_gpu,
                self._grids_gpu,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            weighted = (sampled * self._weights_gpu.unsqueeze(1)).sum(dim=0)
            out_gpu = (weighted / self._weight_sum_gpu.unsqueeze(0)).clamp(0, 255)
            out_gpu = out_gpu.permute(1, 2, 0)
            mask = self._in_circle_gpu.unsqueeze(-1)
            out_gpu = torch.where(mask, out_gpu, torch.zeros_like(out_gpu))
            return out_gpu.byte().cpu().numpy()

        # CPU fallback: cv2.remap
        sample_acc = np.zeros((self.output_h, self.output_w, 3), dtype=np.float32)
        weight_acc = np.zeros((self.output_h, self.output_w), dtype=np.float32)
        for tile, (u, v, w) in zip(tiles, self._projs):
            sample = cv2.remap(
                tile, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            sample_acc += sample.astype(np.float32) * w[..., None]
            weight_acc += w

        valid = (weight_acc > 1e-6) & self._in_circle
        out = np.zeros_like(sample_acc)
        out[valid] = sample_acc[valid] / weight_acc[valid][..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def render_mask(self, data, renderer, robot_geom_ids):
        """Binary mask (uint8, 0/255) of the robot through the same fisheye projection
        used by `render`. Returns a (H, W) uint8 mask (255 = robot pixel, 0 = background)
        at the renderer's output resolution.

        Implementation note: each tile is rendered in segmentation mode so we get
        per-pixel geom IDs; the tile masks are then resampled with nearest-neighbor
        through the cubemap LUT and combined with logical-or."""
        robot_set = set(int(g) for g in robot_geom_ids)
        seg_tiles = []
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(data, self.tile_cam_ids[0])
            seg_tiles.append(renderer.render())
            for cid in self.tile_cam_ids[1:]:
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cid
                mujoco.mjv_updateCamera(self.model, data, cam, renderer.scene)
                seg_tiles.append(renderer.render())
        finally:
            renderer.disable_segmentation_rendering()

        out = np.zeros((self.output_h, self.output_w), dtype=np.uint8)
        for seg, (u, v, w) in zip(seg_tiles, self._projs):
            gid = seg[..., 0]
            tile_mask = np.isin(gid, list(robot_set) if robot_set else [-1]).astype(np.uint8) * 255
            sampled = cv2.remap(
                tile_mask,
                u,
                v,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            out = np.maximum(out, np.where(w > 0, sampled, 0).astype(np.uint8))
        out[~self._in_circle] = 0
        return out
