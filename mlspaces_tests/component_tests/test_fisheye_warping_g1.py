"""Smoke tests for the ported G1 cubemap fisheye renderer.

Builds a minimal 5-tile-camera MuJoCo scene and checks that FisheyeRenderer
constructs, renders, and masks without error and with the expected output
shapes. Not a visual-regression test (see test_fisheye_warping.py for that
style, applied to the offline distortion utility) -- this only exercises the
live cubemap-compositing path that fisheye_warping.py doesn't cover.
"""

import sys

import mujoco
import numpy as np
import pytest

from molmo_spaces.utils.fisheye_warping_tiled import FisheyeRenderer

TILE_CAM_NAMES = ("tile_center", "tile_up", "tile_down", "tile_left", "tile_right")

_TILE_EULERS = {
    "tile_center": "0 0 0",
    "tile_up": "60 0 0",
    "tile_down": "-60 0 0",
    "tile_left": "0 60 0",
    "tile_right": "0 -60 0",
}

_MODEL_XML = f"""
<mujoco>
  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom type="box" pos="0 0.5 0.3" size="0.2 0.2 0.3" rgba="0.8 0.1 0.1 1"/>
    <body pos="0 0 1">
      {
    "".join(
        f'<camera name="{name}" fovy="100" euler="{euler}"/>'
        for name, euler in _TILE_EULERS.items()
    )
}
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(_MODEL_XML)


@pytest.fixture(scope="module")
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    return d


@pytest.fixture
def renderer(model):
    return FisheyeRenderer(
        model, tile_cam_names=TILE_CAM_NAMES, tile_size=64, output_h=48, output_w=48
    )


@pytest.fixture
def mj_renderer(model):
    if sys.platform == "darwin":
        pytest.skip("mujoco.Renderer/CGL can't create an offscreen context headlessly on macOS")
    r = mujoco.Renderer(model, 64, 64)
    yield r
    r.close()


class TestFisheyeRendererConstruction:
    def test_requires_exactly_five_cameras(self, model):
        with pytest.raises(ValueError, match="need exactly 5 tile cameras"):
            FisheyeRenderer(model, tile_cam_names=TILE_CAM_NAMES[:4])

    def test_unknown_camera_name_raises(self, model):
        with pytest.raises(ValueError, match="not found in model"):
            FisheyeRenderer(model, tile_cam_names=(*TILE_CAM_NAMES[:4], "does_not_exist"))

    def test_rejects_narrow_fov(self, model):
        # tile_center etc all have fovy=100 in the fixture model; this only
        # confirms the >=90 guard fires when the model doesn't satisfy it.
        narrow_xml = _MODEL_XML.replace('fovy="100"', 'fovy="60"')
        narrow_model = mujoco.MjModel.from_xml_string(narrow_xml)
        with pytest.raises(ValueError, match="too small"):
            FisheyeRenderer(narrow_model, tile_cam_names=TILE_CAM_NAMES)


class TestFisheyeRendererRender:
    def test_render_shape_and_dtype(self, renderer, data, mj_renderer):
        out = renderer.render(data, mj_renderer)
        assert out.shape == (48, 48, 3)
        assert out.dtype == np.uint8

    def test_render_mask_shape_and_dtype(self, renderer, data, mj_renderer):
        # geom id 1 is the box (0 is the ground plane) per declaration order in _MODEL_XML.
        mask = renderer.render_mask(data, mj_renderer, robot_geom_ids=[1])
        assert mask.shape == (48, 48)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_set_intrinsics_rebuilds_lut(self, renderer, data, mj_renderer):
        # Should not raise, and should still produce a valid render afterwards.
        renderer.set_intrinsics(K=renderer.K * 1.1)
        out = renderer.render(data, mj_renderer)
        assert out.shape == (48, 48, 3)


class TestProjectCameraPoint:
    def test_point_in_front_projects(self, renderer):
        # MuJoCo camera frame: +x right, +y up, -z forward (see project_camera_point docstring).
        pt = renderer.project_camera_point(np.array([0.0, 0.0, -1.0]))
        assert pt is not None
        u, v = pt
        assert np.isfinite(u) and np.isfinite(v)

    def test_point_behind_returns_none(self, renderer):
        pt = renderer.project_camera_point(np.array([0.0, 0.0, 1.0]))
        assert pt is None
