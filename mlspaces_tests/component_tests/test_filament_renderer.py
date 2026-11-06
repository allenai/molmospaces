"""MjFilamentRenderer (mujoco.mjrf) against the classic MjOpenGLRenderer on a small scene."""

import subprocess
import sys

import mujoco as mj
import numpy as np
import pytest

from molmo_spaces.renderer.filament_rendering import HAS_FILAMENT, MjFilamentRenderer
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

W, H = 320, 240


def _filament_probe() -> str | None:
    """Renders one frame in a subprocess; returns why Filament is unusable, or None.

    A Filament context failure (e.g. no usable GPU/GL on a headless runner) can abort
    the process, so it is probed out of process rather than letting it take down the
    whole test session.
    """
    if not HAS_FILAMENT:
        return "needs mujoco>=4.0 (mujoco-studio)"
    code = (
        "import mujoco as mj\n"
        "from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer\n"
        "m = mj.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>')\n"
        "r = MjFilamentRenderer(m, width=8, height=8); r.update(mj.MjData(m)); r.render()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return f"Filament cannot render here (exit {proc.returncode}): {proc.stderr[-2000:]}"
    return None


_UNUSABLE = _filament_probe()
pytestmark = pytest.mark.skipif(_UNUSABLE is not None, reason=str(_UNUSABLE))

XML = f"""
<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".8 .8 .8"
             width="64" height="64"/>
    <texture name="red" type="2d" builtin="flat" rgb1="1 0 0" width="8" height="8"/>
    <material name="floor" texture="grid" texrepeat="4 4"/>
    <material name="boxmat" texture="grid"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="hidden" type="box" pos="0 0 .5" size=".1 .1 .1" group="3"/>
    <site name="marker" pos="-.6 0 .3" size=".15" rgba="1 0 1 1"/>
    <geom name="floor" type="plane" size="3 3 .1" material="floor"/>
    <body name="box" pos="0 0 .3">
      <freejoint/>
      <geom name="box" type="box" size=".2 .2 .2" material="boxmat"/>
    </body>
    <geom name="sph" type="sphere" pos=".6 0 .2" size=".2" rgba="0 1 0 1"/>
    <camera name="cam" pos="0 -2 1.5" xyaxes="1 0 0 0 .6 .8"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def scene():
    model = mj.MjModel.from_xml_string(XML)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    filament = MjFilamentRenderer(model, width=W, height=H)
    opengl = MjOpenGLRenderer(model, width=W, height=H)
    yield model, data, filament, opengl
    filament.close()
    opengl.close()


def _render(renderer, data, mode=None):
    renderer.update(data, "cam")
    if mode == "depth":
        renderer.enable_depth_rendering()
    elif mode == "seg":
        renderer.enable_segmentation_rendering()
    try:
        return renderer.render()
    finally:
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()


def test_rgb(scene):
    model, data, filament, _ = scene
    rgb = _render(filament, data)
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    # The green sphere is lit and sits right of the box, as in the classic render.
    sph_px = rgb[100, 233]
    assert rgb.std() > 10
    assert sph_px[1] > sph_px[0] and sph_px[1] > sph_px[2], sph_px


def test_depth_matches_opengl(scene):
    _, data, filament, opengl = scene
    fd = _render(filament, data, "depth")
    od = _render(opengl, data, "depth")
    assert fd.shape == (H, W) and fd.dtype == np.float32
    near = od < 10  # skip the far-plane background, where both just report zfar
    assert near.mean() > 0.5
    assert np.median(np.abs(fd[near] - od[near])) < 1e-3
    assert np.mean(np.abs(fd[near] - od[near]) < 1e-2) > 0.95


def test_segmentation_matches_opengl(scene):
    model, data, filament, opengl = scene
    fs = _render(filament, data, "seg")
    os_ = _render(opengl, data, "seg")
    assert fs.shape == (H, W, 3) and fs.dtype == np.int32
    # Geom IDs agree away from silhouettes (rasterization differs by a pixel or so).
    assert np.mean(fs[..., 0] == os_[..., 0]) > 0.98
    for name in ("floor", "box", "sph"):
        gid = model.geom(name).id
        mask = fs[..., 0] == gid
        assert mask.any(), name
        assert (fs[mask, 1] == mj.mjtObj.mjOBJ_GEOM).all()
        assert (fs[mask, 2] == model.geom_bodyid[gid]).all()
    assert not (fs[..., 0] == model.geom("hidden").id).any()
    assert (fs[fs[..., 0] == -1] == -1).all()


def test_scene_camera_override_is_used(scene):
    """Callers (scene_maps, env.render_at_pose) edit scene.camera after update()."""
    _, data, filament, _ = scene
    filament.update(data, "cam")
    for camera in filament.scene.camera:
        camera.pos = [0.0, 0.0, 5.0]
        camera.forward = [0.0, 0.0, -1.0]
        camera.up = [0.0, 1.0, 0.0]
    filament.enable_depth_rendering()
    depth = filament.render()
    filament.disable_depth_rendering()
    # Straight down from 5 m: the box top (z=0.5) is 4.5 m away at the image center.
    assert abs(depth[H // 2, W // 2] - 4.5) < 0.05


def test_material_texture_swap_is_rendered(scene):
    """Texture randomization edits mat_texid; the next update() must pick it up."""
    model, data, filament, _ = scene
    boxmat = model.material("boxmat").id
    orig = model.mat_texid[boxmat].copy()
    try:
        before = _render(filament, data)[100, 160].astype(int)
        model.mat_texid[boxmat, mj.mjtTextureRole.mjTEXROLE_RGB] = model.texture("red").id
        after = _render(filament, data)[100, 160].astype(int)
    finally:
        model.mat_texid[boxmat] = orig
    assert after[0] > 150 and after[1] < 50 and after[2] < 50, (before, after)
