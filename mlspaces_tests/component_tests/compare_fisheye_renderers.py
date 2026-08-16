"""Apples-to-apples image-quality comparison of the two fisheye implementations.

The repo has two, and they work very differently:

1. `molmo_spaces/utils/fisheye_warping.py` -- post-hoc *warp* of a single
   pinhole render. One 640x480 image is resampled through a radial polynomial
   (k1..k4), centre-cropped and resized. Every output pixel comes from one
   finite-resolution pinhole frame, so the periphery is stretched from few
   source pixels.

2. `molmo_spaces/utils/fisheye_cubemap.py` -- a cubemap `FisheyeRenderer`.
   Five pinhole "tile" cameras sharing one optical centre are rendered at
   `tile_size` each and composited through an OpenCV-fisheye (equidistant)
   pixel->ray LUT. Peripheral rays come from the *side* tiles at full resolution.

The two use different lens models, so raw outputs cover different fields of
view and are not directly comparable. This script equalises everything that is
not the thing under test:

  * Same lens and same FOV. Both are put on one common projection: the g1
    renderer's own distortion coefficients D, with K scaled so a square
    +/-TARGET_HALF_FOV_DEG cone fills the frame. The cubemap renderer takes
    these as an intrinsics override; the warp output is resampled onto the
    same pixel->ray grid. Pixel (i, j) of each image is then the same ray.
  * Same output size -- both 240x240.
  * Same render budget. The warp path renders 640x480 = 307200 pixels; the
    tile size is chosen so 5 tiles come to the same total, so neither method
    is winning on raw pixels rendered.
  * The pinhole feeding the warp path is widened (bisection on `fovy`) until
    its post-crop output just covers the target cone, so it is not starved of
    coverage either.

Wall-clock render time for both paths is measured and reported.

Note the warp path cannot reach the g1 lens's native 72.8 deg horizontal
half-FOV at all: r*(1 + k1 r^2 + ... + k4 r^8) stops being monotone past
r ~ 1.65, capping its output at ~68.3 deg. That is why the common cone is set
below both, rather than at the g1's native FOV.

The scene is a room whose six walls carry a regular checker grid.

Run:
    python mlspaces_tests/component_tests/compare_fisheye_renderers.py [--out PATH]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from molmo_spaces.configs.camera_configs import G1_HEAD_FISHEYE_D
from molmo_spaces.utils.constants.camera_constants import (
    DEFAULT_CROP_PERCENT,
    DEFAULT_DISTORTION_PARAMETERS,
    GOPRO_CAMERA_HEIGHT,
    GOPRO_CAMERA_WIDTH,
    GOPRO_VERTICAL_FOV,
)
from molmo_spaces.utils.fisheye_cubemap import _FACES, FisheyeRenderer
from molmo_spaces.utils.fisheye_warping import calc_camera_intrinsics, warp_image_gpu

OUT_H = OUT_W = 240
TILE_FOVY = 100.0
ROOM = 2.0  # half-extent of the grid room, metres
FACE_NAMES = tuple(f[0] for f in _FACES)

# The square cone both paths are asked to cover. Below the warp path's hard
# ~68.3 deg ceiling (see module docstring) with margin to spare.
TARGET_HALF_FOV_DEG = 60.0

# Same render budget for both: 5 tiles totalling the warp path's 640x480.
PINHOLE_PIXELS = GOPRO_CAMERA_WIDTH * GOPRO_CAMERA_HEIGHT
TILE_SIZE = int(round((PINHOLE_PIXELS / 5) ** 0.5))

# Base camera frame, expressed as world axes: the camera looks along world +x
# with world +z up. MuJoCo cameras look down their own -z with +y up, so
# cam_x = -Y, cam_y = +Z, cam_z = -X. Columns are (cam_x, cam_y, cam_z).
BASE_R = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def tile_camera_xml() -> str:
    """One `<camera>` per cubemap face, oriented straight from `_FACES`.

    `_FACES` gives each face's (x, y, z) axes in the base camera frame; mapping
    them through BASE_R gives world axes, which MuJoCo accepts as `xyaxes`
    (it derives cam_z = cam_x x cam_y itself). Deriving these rather than
    hand-writing eulers keeps the test scene locked to the renderer's own
    face convention.
    """
    lines = []
    for name, ax, ay, _az in _FACES:
        x_w = BASE_R @ ax
        y_w = BASE_R @ ay
        vals = " ".join(f"{v:g}" for v in (*x_w, *y_w))
        lines.append(f'      <camera name="tile_{name}" fovy="{TILE_FOVY}" xyaxes="{vals}"/>')
    return "\n".join(lines)


def build_scene_xml(pinhole_fovy: float) -> str:
    """Camera at the origin inside a closed grid room. `pinhole` is the single
    wide camera the warp path consumes; the five `tile_*` cameras share its
    optical centre and feed the cubemap path."""
    return f"""
<mujoco model="fisheye_grid_test">
  <visual>
    <headlight ambient="0.9 0.9 0.9" diffuse="0.1 0.1 0.1" specular="0 0 0"/>
    <global offwidth="2048" offheight="2048"/>
    <quality offsamples="8"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.05 0.05 0.08" rgb2="0.95 0.95 0.95"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" specular="0" shininess="0"/>
    <texture name="fine" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.10 0.35 0.75" rgb2="0.95 0.95 0.95"/>
    <material name="fine_mat" texture="fine" texrepeat="8 8" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <!-- Six inward-facing walls of a cube; the wall the camera faces gets the
         finer grid so the image centre has a high-frequency target. -->
    <geom name="wall_front" type="plane" size="{ROOM} {ROOM} 0.1"
          pos="{ROOM} 0 0" zaxis="-1 0 0" material="fine_mat"/>
    <geom name="wall_back" type="plane" size="{ROOM} {ROOM} 0.1"
          pos="{-ROOM} 0 0" zaxis="1 0 0" material="grid_mat"/>
    <geom name="wall_left" type="plane" size="{ROOM} {ROOM} 0.1"
          pos="0 {ROOM} 0" zaxis="0 -1 0" material="grid_mat"/>
    <geom name="wall_right" type="plane" size="{ROOM} {ROOM} 0.1"
          pos="0 {-ROOM} 0" zaxis="0 1 0" material="grid_mat"/>
    <geom name="ceiling" type="plane" size="{ROOM} {ROOM} 0.1"
          pos="0 0 {ROOM}" zaxis="0 0 -1" material="grid_mat"/>
    <geom name="floor" type="plane" size="{ROOM} {ROOM} 0.1"
          pos="0 0 {-ROOM}" zaxis="0 0 1" material="grid_mat"/>

    <body name="cam_rig" pos="0 0 0">
      <!-- Wide pinhole for the warp path: GoPro raw format, +x forward. -->
      <camera name="pinhole" fovy="{pinhole_fovy}" xyaxes="0 -1 0 0 0 1"/>
      <!-- Cubemap tiles: same optical centre, orientations derived from _FACES. -->
{tile_camera_xml()}
    </body>
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------
# Lens models
# --------------------------------------------------------------------------


def gopro_distortion_factor(r: np.ndarray) -> np.ndarray:
    """The radial factor `fisheye_warping.make_distorted_grid` applies: an output
    pixel at normalized radius r samples the source at radius r * factor(r)."""
    k1, k2, k3, k4 = (DEFAULT_DISTORTION_PARAMETERS[k] for k in ("k1", "k2", "k3", "k4"))
    return 1 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8


def invert_gopro_distortion(r_src: np.ndarray, iters: int = 30) -> np.ndarray:
    """Solve r_out * factor(r_out) = r_src for r_out (source radius -> output
    radius), the direction `make_distorted_grid` does not provide. Newton on a
    monotone polynomial over this parameter range; converges in a few iters."""
    k1, k2, k3, k4 = (DEFAULT_DISTORTION_PARAMETERS[k] for k in ("k1", "k2", "k3", "k4"))
    t = r_src.copy()
    for _ in range(iters):
        t2 = t * t
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f = t * (1 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - r_src
        df = 1 + 3 * k1 * t2 + 5 * k2 * t4 + 7 * k3 * t6 + 9 * k4 * t8
        t = t - f / np.where(np.abs(df) > 1e-12, df, 1e-12)
    return t


def warp_output_half_fov(fovy_deg: float) -> tuple[float, float]:
    """(horizontal, vertical) half-FOV in radians that the warp path's final
    240x240 output covers, for a source pinhole of the given fovy.

    The outermost surviving pixel after the `DEFAULT_CROP_PERCENT` crop sits at
    normalized radius r; it samples the source at r * factor(r), and the source
    is a pinhole, so that radius is tan(theta)."""
    f = 0.5 * GOPRO_CAMERA_HEIGHT / np.tan(np.radians(fovy_deg) / 2)
    crop_w = int(GOPRO_CAMERA_WIDTH * DEFAULT_CROP_PERCENT)
    crop_h = int(GOPRO_CAMERA_HEIGHT * DEFAULT_CROP_PERCENT)
    xn = (GOPRO_CAMERA_WIDTH / 2 - crop_w) / f
    yn = (GOPRO_CAMERA_HEIGHT / 2 - crop_h) / f
    return (
        float(np.arctan(xn * gopro_distortion_factor(np.array(xn)))),
        float(np.arctan(yn * gopro_distortion_factor(np.array(yn)))),
    )


def warp_monotone_limit() -> tuple[float, float]:
    """(max invertible normalized radius, max reachable output half-angle in rad).

    r -> r * factor(r) must be monotone for the warp to be a well-defined
    mapping; with the shipped k1..k4 it turns over near r = 1.65. Past that the
    warp path simply cannot render a wider view, whatever fovy it is given."""
    r = np.linspace(0, 4, 20000)
    g = r * gopro_distortion_factor(r)
    turn = int(np.argmax(np.diff(g) < 0))
    return float(r[turn]), float(np.arctan(g[turn]))


def solve_pinhole_fovy(target_half: float) -> float:
    """Bisect on the source pinhole's fovy so the warp path's output *vertical*
    half-FOV reaches `target_half`.

    Vertical is the binding axis: the warp path's two axes are locked to the 4:3
    source aspect, so once vertical reaches the target the horizontal
    over-covers. That excess is not free -- it is source pixels spent on rays
    outside the compared cone -- and is reported rather than hidden."""
    r_limit, theta_limit = warp_monotone_limit()
    if target_half >= theta_limit:
        raise ValueError(
            f"target half-FOV {np.degrees(target_half):.2f} deg exceeds the warp path's "
            f"ceiling of {np.degrees(theta_limit):.2f} deg (r*factor(r) turns over at "
            f"r={r_limit:.3f})"
        )
    lo, hi = 10.0, 179.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        # Stay inside the invertible branch while bisecting, else the objective
        # is non-monotone and the search diverges.
        f = 0.5 * GOPRO_CAMERA_HEIGHT / np.tan(np.radians(mid) / 2)
        yn = (GOPRO_CAMERA_HEIGHT / 2 - int(GOPRO_CAMERA_HEIGHT * DEFAULT_CROP_PERCENT)) / f
        if yn > r_limit:  # past the turnover: too wide, and the objective is garbage
            hi = mid
        elif warp_output_half_fov(mid)[1] < target_half:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def common_lens_intrinsics(half_fov: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """K, D, image_size for the shared projection: the g1 lens's real distortion
    coefficients, with focal length set so a square +/-`half_fov` cone exactly
    fills the OUT_H x OUT_W frame.

    image_size is set to the output size so `FisheyeRenderer`'s
    calibration->output rescaling is the identity and K means what it says."""
    k1, k2, k3, k4 = G1_HEAD_FISHEYE_D
    t = half_fov
    theta_d = t * (1 + k1 * t**2 + k2 * t**4 + k3 * t**6 + k4 * t**8)
    f = ((OUT_W - 1) / 2) / theta_d
    K = np.array([[f, 0.0, (OUT_W - 1) / 2], [0.0, f, (OUT_H - 1) / 2], [0.0, 0.0, 1.0]])
    return K, list(G1_HEAD_FISHEYE_D), (OUT_W, OUT_H)


def g1_ray_grid(fisheye: FisheyeRenderer) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-pixel unit ray (MuJoCo camera frame: +x right, +y up, -z fwd)
    for the g1 renderer, plus its validity mask.

    This mirrors the unprojection inside `FisheyeRenderer._build_lut`, which
    keeps only the resulting tile grids. `main` round-trips it against the
    renderer's own `project_camera_point` so the replication cannot drift."""
    out_h, out_w = fisheye.output_h, fisheye.output_w
    ys, xs = np.mgrid[:out_h, :out_w].astype(np.float64)
    cal_w, cal_h = fisheye.image_size
    sx, sy = out_w / float(cal_w), out_h / float(cal_h)
    fx, fy = fisheye.K[0, 0] * sx, fisheye.K[1, 1] * sy
    cx, cy = fisheye.K[0, 2] * sx, fisheye.K[1, 2] * sy
    x_norm = (xs - cx) / fx
    y_norm = (ys - cy) / fy
    theta_d = np.hypot(x_norm, y_norm)
    phi = np.arctan2(y_norm, x_norm)

    k1, k2, k3, k4 = fisheye.D
    theta = theta_d.copy()
    for _ in range(10):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
        df = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
        theta = theta - f / np.where(np.abs(df) > 1e-12, df, 1e-12)

    valid = np.isfinite(theta) & (theta >= 0.0) & (theta < np.pi * 0.5 + 0.3)
    sin_t = np.sin(theta)
    rays = np.stack([sin_t * np.cos(phi), -sin_t * np.sin(phi), -np.cos(theta)], axis=-1)
    return rays, valid


def resample_warp_onto_rays(warped: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample the warp path's 240x240 output along the given rays, producing an
    image in the g1 renderer's angular coordinates.

    Forward model, inverting each stage of `warp_image_gpu`:
      ray -> pinhole normalized coords (source radius)
           -> output radius (invert the k1..k4 polynomial)
           -> pixel in the uncropped 640x480 output frame
           -> subtract the crop, rescale to 240x240 (align_corners=True)
    """
    # MuJoCo cam -> OpenCV cam (z forward, y down), same convention as
    # FisheyeRenderer.project_camera_point.
    X, Y, Z = rays[..., 0], -rays[..., 1], -rays[..., 2]
    forward = Z > 1e-6
    Zs = np.where(forward, Z, 1.0)
    a, b = X / Zs, Y / Zs
    r_src = np.hypot(a, b)
    r_out = invert_gopro_distortion(r_src)
    scale = np.where(r_src > 1e-9, r_out / np.where(r_src > 1e-9, r_src, 1.0), 1.0)

    f = 0.5 * GOPRO_CAMERA_HEIGHT / np.tan(np.radians(PINHOLE_FOVY) / 2)
    u_full = a * scale * f + GOPRO_CAMERA_WIDTH / 2
    v_full = b * scale * f + GOPRO_CAMERA_HEIGHT / 2

    crop_w = int(GOPRO_CAMERA_WIDTH * DEFAULT_CROP_PERCENT)
    crop_h = int(GOPRO_CAMERA_HEIGHT * DEFAULT_CROP_PERCENT)
    cropped_w = GOPRO_CAMERA_WIDTH - 2 * crop_w
    cropped_h = GOPRO_CAMERA_HEIGHT - 2 * crop_h
    u = (u_full - crop_w) * (OUT_W - 1) / (cropped_w - 1)
    v = (v_full - crop_h) * (OUT_H - 1) / (cropped_h - 1)

    valid = forward & (u >= 0) & (u <= OUT_W - 1) & (v >= 0) & (v <= OUT_H - 1)
    out = cv2.remap(
        warped,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out[~valid] = 0
    return out, valid


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def time_it(fn, repeats: int, warmup: int = 3) -> tuple[object, float, float]:
    """Run `fn` and return (last result, mean ms, stdev ms) over `repeats` timed
    calls after `warmup` untimed ones."""
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return result, float(np.mean(times)), float(np.std(times))


def render_warp_fisheye(model, data, repeats: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Path 1: one wide pinhole render, radially warped, cropped, resized."""
    renderer = mujoco.Renderer(model, height=GOPRO_CAMERA_HEIGHT, width=GOPRO_CAMERA_WIDTH)
    K = calc_camera_intrinsics(PINHOLE_FOVY, GOPRO_CAMERA_HEIGHT, GOPRO_CAMERA_WIDTH)

    def once():
        renderer.update_scene(data, "pinhole")
        raw = renderer.render()
        img = torch.from_numpy(raw).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        warped = warp_image_gpu(
            image=img,
            K=K,
            distortion_parameters=DEFAULT_DISTORTION_PARAMETERS,
            crop_percent=DEFAULT_CROP_PERCENT,
            output_shape=(OUT_H, OUT_W),
        )
        return raw, (warped[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    try:
        (raw, warped), mean_ms, std_ms = time_it(once, repeats)
    finally:
        renderer.close()
    return raw, warped, mean_ms, std_ms


def render_cubemap_fisheye(
    model, data, K, D, image_size, repeats: int
) -> tuple[np.ndarray, list[np.ndarray], FisheyeRenderer, float, float]:
    """Path 2: five pinhole tiles composited through the equidistant fisheye LUT,
    using the shared K/D so its output covers exactly the common cone."""
    fisheye = FisheyeRenderer(
        model,
        tile_cam_names=[f"tile_{n}" for n in FACE_NAMES],
        tile_size=TILE_SIZE,
        output_h=OUT_H,
        output_w=OUT_W,
        K=K,
        D=D,
        image_size=image_size,
    )
    renderer = mujoco.Renderer(model, height=TILE_SIZE, width=TILE_SIZE)
    try:
        out, mean_ms, std_ms = time_it(lambda: fisheye.render(data, renderer), repeats)
        # Re-render the tiles individually for the contact sheet.
        tiles = []
        for name in FACE_NAMES:
            renderer.update_scene(data, f"tile_{name}")
            tiles.append(renderer.render())
    finally:
        renderer.close()
    return out, tiles, fisheye, mean_ms, std_ms


# --------------------------------------------------------------------------
# Metrics and layout
# --------------------------------------------------------------------------


def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    """Variance of Laplacian over the masked region -- a standard blur/detail
    proxy. Higher = crisper. The mask is eroded so the Laplacian never straddles
    the validity boundary and reads the black fill as an edge."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return float(lap[inner].var()) if inner.any() else float("nan")


def radial_sharpness(img: np.ndarray, mask: np.ndarray, n_rings: int = 4) -> list[float]:
    """Variance of Laplacian per concentric ring, centre outward -- shows *where*
    a method loses detail."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (min(h, w) / 2)
    edges = np.linspace(0, 1, n_rings + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = inner & (r >= lo) & (r < hi)
        out.append(float(lap[sel].var()) if sel.any() else float("nan"))
    return out


def label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Add a caption bar under an image."""
    bar = np.full((34 if sub else 22, img.shape[1], 3), 245, dtype=np.uint8)
    out = np.vstack([img, bar])
    cv2.putText(
        out,
        text,
        (4, img.shape[0] + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )
    if sub:
        cv2.putText(
            out,
            sub,
            (4, img.shape[0] + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    return out


def letterbox(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize preserving aspect ratio, padding to (h, w)."""
    scale = min(h / img.shape[0], w / img.shape[1])
    small = cv2.resize(
        img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )
    out = np.full((h, w, 3), 245, dtype=np.uint8)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    return out


def zoom_patch(img: np.ndarray, cy_frac: float, cx_frac: float, size: int = 60) -> np.ndarray:
    """Nearest-neighbour blow-up of a square patch, so per-pixel detail is visible."""
    h, w = img.shape[:2]
    cy, cx = int(cy_frac * h), int(cx_frac * w)
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return cv2.resize(
        img[y0 : y0 + size, x0 : x0 + size], (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST
    )


# Set in main() once the FOV match is solved; read by the warp render/resample.
PINHOLE_FOVY = GOPRO_VERTICAL_FOV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fisheye_renderer_comparison.png"),
        help="Where to write the comparison PNG.",
    )
    parser.add_argument(
        "--half-fov",
        type=float,
        default=TARGET_HALF_FOV_DEG,
        help="Half-angle (deg) of the square cone both renderers must cover.",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Timed render repeats per path.")
    args = parser.parse_args()

    global PINHOLE_FOVY

    target_half = np.radians(args.half_fov)
    K, D, image_size = common_lens_intrinsics(target_half)
    PINHOLE_FOVY = solve_pinhole_fovy(target_half)
    warp_half_h, warp_half_v = warp_output_half_fov(PINHOLE_FOVY)
    r_limit, theta_limit = warp_monotone_limit()

    print(f"common projection: g1 lens D, square +/-{args.half_fov:.1f} deg, {OUT_H}x{OUT_W} out")
    print(f"  warp path ceiling: {np.degrees(theta_limit):.2f} deg half-angle (r<{r_limit:.3f})")
    print(f"  source pinhole fovy solved to {PINHOLE_FOVY:.2f} deg ->")
    print(
        f"    warp output covers horiz={np.degrees(warp_half_h):6.2f}  "
        f"vert={np.degrees(warp_half_v):6.2f} deg"
        f"  (horizontal over-covers: 4:3 source aspect is not a free knob)"
    )
    print(
        f"  render budget: pinhole {GOPRO_CAMERA_WIDTH}x{GOPRO_CAMERA_HEIGHT}="
        f"{PINHOLE_PIXELS} px vs 5x{TILE_SIZE}^2={5 * TILE_SIZE**2} px "
        f"({5 * TILE_SIZE**2 / PINHOLE_PIXELS:.3f}x)"
    )

    model = mujoco.MjModel.from_xml_string(build_scene_xml(PINHOLE_FOVY))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    raw, warped, warp_ms, warp_sd = render_warp_fisheye(model, data, args.repeats)
    cubemap, tiles, fisheye, cube_ms, cube_sd = render_cubemap_fisheye(
        model, data, K, D, image_size, args.repeats
    )

    rays, ref_valid = g1_ray_grid(fisheye)
    # Guard the replicated unprojection against the renderer's own forward model.
    for py, px in ((OUT_H // 2, OUT_W // 2), (30, 30), (OUT_H - 20, OUT_W - 20)):
        uv = fisheye.project_camera_point(rays[py, px])
        assert uv is not None and abs(uv[0] - px) < 0.5 and abs(uv[1] - py) < 0.5, (
            f"ray grid disagrees with project_camera_point at ({px}, {py}): {uv}"
        )
    warp_on_rays, warp_valid = resample_warp_onto_rays(warped, rays)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nrender time ({args.repeats} reps, {device}, ms per frame):")
    print(f"  warp     {warp_ms:7.2f} +/- {warp_sd:5.2f}   ({1000 / warp_ms:6.1f} fps)")
    print(
        f"  cubemap  {cube_ms:7.2f} +/- {cube_sd:5.2f}   ({1000 / cube_ms:6.1f} fps)"
        f"   ({cube_ms / warp_ms:.2f}x the warp path)"
    )

    common = ref_valid & warp_valid
    print(f"\ncommon valid region: {100 * common.mean():.1f}% of the {OUT_H}x{OUT_W} frame")
    s_warp = sharpness(warp_on_rays, common)
    s_cube = sharpness(cubemap, common)
    print("sharpness (variance of Laplacian, common region only):")
    print(f"  warp    {s_warp:8.1f}")
    print(f"  cubemap {s_cube:8.1f}   ({s_cube / s_warp:.2f}x)")
    print("radial sharpness (centre -> edge):")
    print("  warp   ", [f"{v:7.1f}" for v in radial_sharpness(warp_on_rays, common)])
    print("  cubemap", [f"{v:7.1f}" for v in radial_sharpness(cubemap, common)])

    diff = cv2.absdiff(warp_on_rays, cubemap)
    diff[~common] = 0
    print(f"mean abs diff over common region: {diff[common].mean():.1f}/255")
    diff_panel = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_INFERNO)[
        ..., ::-1
    ].copy()
    diff_panel[~common] = 0  # rays only one method can see are not a "difference"

    row1 = [
        label(
            letterbox(raw, OUT_H, OUT_W),
            f"source pinhole {GOPRO_CAMERA_WIDTH}x{GOPRO_CAMERA_HEIGHT}",
            f"fovy={PINHOLE_FOVY:.1f} deg (solved), undistorted",
        ),
        label(
            warp_on_rays,
            "fisheye_warping.py (warp)",
            f"sharp={s_warp:.0f}, {warp_ms:.1f} ms",
        ),
        label(
            cubemap,
            "fisheye_cubemap.py (cubemap)",
            f"5x{TILE_SIZE} tiles, sharp={s_cube:.0f}, {cube_ms:.1f} ms",
        ),
        label(
            diff_panel,
            "abs difference",
            f"mean={diff[common].mean():.1f}/255 over common rays",
        ),
    ]

    # 4x blow-ups of the same rays from each method: the centre, and the left
    # periphery where the warp path is stretching a handful of source pixels.
    row2 = [
        label(zoom_patch(warp_on_rays, 0.5, 0.5), "warp: centre 4x"),
        label(zoom_patch(cubemap, 0.5, 0.5), "cubemap: centre 4x"),
        label(zoom_patch(warp_on_rays, 0.5, 0.20), "warp: left edge 4x"),
        label(zoom_patch(cubemap, 0.5, 0.20), "cubemap: left edge 4x"),
    ]

    row3 = [label(letterbox(t, OUT_H, OUT_W), f"tile {n}") for t, n in zip(tiles, FACE_NAMES)]

    def pack(row: list[np.ndarray], width: int) -> np.ndarray:
        strip = np.hstack(row)
        pad = np.full((strip.shape[0], max(0, width - strip.shape[1]), 3), 245, np.uint8)
        return np.hstack([strip, pad])

    width = max(sum(im.shape[1] for im in r) for r in (row1, row2, row3))
    canvas = np.vstack([pack(r, width) for r in (row1, row2, row3)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
