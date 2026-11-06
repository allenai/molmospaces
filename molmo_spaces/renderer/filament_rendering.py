import mujoco as mj
import numpy as np

from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer

try:
    # Registers the `filament:` resource provider that serves Filament's compiled
    # materials (pbr.filamat, ...). Without it, the first render aborts the process.
    import mujoco.studio.window  # noqa: F401
    from mujoco import mjrf
except ImportError:
    mjrf = None

# MuJoCo >= 4.0 ships Filament in the `mujoco-studio` package (installed by `mujoco`),
# so no custom wheel is needed.
HAS_FILAMENT: bool = mjrf is not None

# Model numeric read by Filament when no light sets a positive `intensity` (the
# "fallback" lighting mode); see MuJoCo's rendering docs, `reFlags`.
_FALLBACK_HEADLIGHT_INTENSITY = "filament.fallback.head_light_intensity"
_DEFAULT_FALLBACK_HEADLIGHT_INTENSITY = 40000.0


class MjFilamentRenderer(MjAbstractRenderer):
    """Renders through MuJoCo's Filament backend (`mujoco.mjrf`).

    Filament keeps its own retained scene (`ModelRenderables`/`ModelLights`), which
    `update()` syncs from MjData. An `MjvScene` is still filled on `update()` so
    callers can read or override `scene.camera` before `render()`, as they do with
    the OpenGL renderer.
    """

    def __init__(
        self,
        model: mj.MjModel,
        device_id: int | None = None,
        height: int = 720,
        width: int = 1280,
        max_geom: int = 10000,
    ) -> None:
        if mjrf is None:
            raise ImportError(
                "Filament rendering requires mujoco>=4.0 with the `mujoco-studio` package."
            )
        super().__init__(model, device_id)

        self._width = width
        self._height = height

        self._scene = mj.MjvScene(model=model, maxgeom=max_geom)
        self._scene_option = mj.MjvOption()
        self._scene_option.sitegroup *= 0

        self._depth_rendering = False
        self._segmentation_rendering = False

        self._ctx = mjrf.Context(mjrf.ContextConfig())
        self._ctx.set_clear_color([0.0, 0.0, 0.0])
        self._resources = mjrf.ModelResources(self._ctx, model)
        self._fscene = self._ctx.create_scene(mjrf.SceneParams())
        self._fscene.configure_from_model(model)
        self._lights = mjrf.ModelLights(self._fscene, self._resources)
        self._renderables = mjrf.ModelRenderables(self._fscene, self._resources)
        self._renderables.set_options(self._scene_option)
        self._headlight_intensity = self._get_headlight_intensity(model)

        # (pixel_format, width, height) -> (RenderTarget, ReadRequest, buffer)
        self._targets: dict[tuple, tuple] = {}

    @staticmethod
    def _get_headlight_intensity(model: mj.MjModel) -> float:
        # Filament turns the headlight off once any light is photometric.
        if model.nlight and np.any(model.light_intensity > 0):
            return 0.0
        nid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_NUMERIC, _FALLBACK_HEADLIGHT_INTENSITY)
        if nid >= 0:
            return float(model.numeric_data[model.numeric_adr[nid]])
        return _DEFAULT_FALLBACK_HEADLIGHT_INTENSITY

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def enable_depth_rendering(self) -> None:
        self._segmentation_rendering = False
        self._depth_rendering = True

    def disable_depth_rendering(self) -> None:
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = True
        self._depth_rendering = False

    def disable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = False

    def geomid_to_bodyid(self, geomid):
        return self.model.geom_bodyid[geomid]

    def _target(self, pixel_format, width: int, height: int, channels: int, dtype) -> tuple:
        key = (pixel_format, width, height)
        if key not in self._targets:
            target = self._ctx.create_render_target(
                mjrf.RenderTargetConfig(width=width, height=height, color_format=pixel_format)
            )
            shape = (height, width, channels) if channels > 1 else (height, width)
            buffer = np.zeros(shape, dtype=dtype)
            read = mjrf.ReadRequest(target=target)
            # The ReadRequest holds a raw pointer, so the buffer lives as long as it.
            read.set_buffer(buffer)
            self._targets[key] = (target, read, buffer)
        return self._targets[key]

    def _camera(self) -> "mjrf.Camera":
        gl = self._scene.camera[0]
        return mjrf.Camera(
            orthographic=bool(gl.orthographic),
            pos=gl.pos,
            forward=gl.forward,
            up=gl.up,
            frustum_center=gl.frustum_center,
            frustum_width=gl.frustum_width,
            frustum_bottom=gl.frustum_bottom,
            frustum_top=gl.frustum_top,
            frustum_near=gl.frustum_near,
            frustum_far=gl.frustum_far,
        )

    def render(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        height = height or self._height
        width = width or self._width

        if self._depth_rendering:
            out_shape = (height, width)
            draw_mode = mjrf.DrawMode.DRAW_MODE_DEPTH
            target, read, buffer = self._target(
                mjrf.PixelFormat.PIXEL_FORMAT_R32F, width, height, 1, np.float32
            )
        else:
            out_shape = (height, width, 3)
            draw_mode = (
                mjrf.DrawMode.DRAW_MODE_SEGMENTATION_BY_ID
                if self._segmentation_rendering
                else mjrf.DrawMode.DRAW_MODE_DEFAULT
            )
            target, read, buffer = self._target(
                mjrf.PixelFormat.PIXEL_FORMAT_RGB8, width, height, 3, np.uint8
            )

        if out is not None and out.shape != out_shape:
            raise ValueError(
                f"Expected `out.shape == {out_shape}`. Got `out.shape={out.shape}`"
                " instead. When using depth rendering, the out array should be of"
                " shape `(height, width)` and otherwise (height, width, 3)."
                f" Got `self._depth_rendering={self._depth_rendering}`."
            )

        # Depth and segmentation must not be tone mapped or dithered.
        plain = self._depth_rendering or self._segmentation_rendering
        request = mjrf.RenderRequest()
        request.scene = self._fscene
        request.target = target
        request.camera = self._camera()
        request.viewport.left = 0
        request.viewport.bottom = 0
        request.viewport.width = width
        request.viewport.height = height
        request.draw_mode = draw_mode
        request.post_processing = not plain
        request.shadows = True
        request.reflections = True
        request.skybox = True
        request.haze = True
        request.headlight = self.model.vis.headlight.active != 0
        request.headlight_color = tuple(self.model.vis.headlight.diffuse)
        request.headlight_intensity = self._headlight_intensity

        self._ctx.wait_for_frame(self._ctx.render([request], [read]))

        if self._depth_rendering:
            # Filament writes reverse-Z depth with a finite far plane:
            #   z = near * (far - d) / (d * (far - near))
            gl = self._scene.camera[0]
            near = np.float64(gl.frustum_near)
            far = np.float64(gl.frustum_far)
            z = buffer.astype(np.float64)
            if gl.orthographic:
                depth = far - z * (far - near)
            else:
                depth = near * far / (z * (far - near) + near)
            result = depth.astype(np.float32)
        elif self._segmentation_rendering:
            result = self._decode_segmentation(buffer)
        else:
            result = buffer

        if out is None:
            return result.copy() if result is buffer else result
        out[:] = result
        return out

    def _decode_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Maps Filament's per-pixel IDs to (object ID, object type, body ID).

        Filament colors each pixel by ID + 1 over all model geoms, then all sites
        (0 is background). Background and any other object maps to (-1, -1, -1).
        """
        image3 = image.astype(np.int64)
        ids = image3[..., 0] + image3[..., 1] * (2**8) + image3[..., 2] * (2**16) - 1

        model = self.model
        ngeom, nsite = model.ngeom, model.nsite
        lut = np.full((ngeom + nsite + 1, 3), -1, dtype=np.int32)
        lut[:ngeom, 0] = np.arange(ngeom)
        lut[:ngeom, 1] = int(mj.mjtObj.mjOBJ_GEOM)
        lut[:ngeom, 2] = model.geom_bodyid
        lut[ngeom : ngeom + nsite, 0] = np.arange(nsite)
        lut[ngeom : ngeom + nsite, 1] = int(mj.mjtObj.mjOBJ_SITE)
        lut[ngeom : ngeom + nsite, 2] = model.site_bodyid

        # Background (-1) and IDs past the sites index the trailing (-1, -1, -1) row.
        ids[(ids < 0) | (ids >= ngeom + nsite)] = ngeom + nsite
        return lut[ids]

    def render_rgb(
        self,
        *,
        out: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        return self.render(out=out, width=width, height=height)

    def upload_textures(self) -> None:
        # Filament's ModelRenderables re-read materials and texture assignments
        # (mat_texid, mat_rgba, geom_rgba) on every update(), so nothing to do.
        pass

    def mark_textures_dirty(self) -> None:
        pass

    def update(
        self,
        data: mj.MjData,
        camera: int | str | mj.MjvCamera = -1,
        scene_option: mj.MjvOption | None = None,
    ) -> None:
        if not isinstance(camera, mj.MjvCamera):
            camera_id = camera
            if isinstance(camera_id, str):
                camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA.value, camera_id)
                if camera_id == -1:
                    raise ValueError(f'The camera "{camera}" does not exist.')
            if camera_id < -1 or camera_id >= self.model.ncam:
                raise ValueError(
                    f"The camera id {camera_id} is out of range [-1, {self.model.ncam})."
                )

            camera = mj.MjvCamera()
            camera.fixedcamid = camera_id

            if camera_id == -1:
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultFreeCamera(self.model, camera)
            else:
                camera.type = mj.mjtCamera.mjCAMERA_FIXED

        scene_option = scene_option or self._scene_option
        # Fills scene.camera, which callers may override before render().
        mj.mjv_updateScene(
            self.model,
            data,
            scene_option,
            None,
            camera,
            mj.mjtCatBit.mjCAT_ALL.value,
            self._scene,
        )
        self._renderables.set_options(scene_option)
        self._lights.update(data)
        self._renderables.update(data)

    def close(self) -> None:
        # Release Filament objects before the context that owns them.
        self._targets = {}
        self._renderables = None
        self._lights = None
        self._fscene = None
        self._resources = None
        self._ctx = None


if __name__ == "__main__":
    from dataclasses import dataclass
    from pathlib import Path

    import tyro
    from PIL import Image

    @dataclass
    class Args:
        model: Path

    args = tyro.cli(Args)

    if not args.model.is_file():
        raise RuntimeError(f"Given path @ {args.model} doesn't point to a valid file")

    model = mj.MjModel.from_xml_path(args.model.as_posix())
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    renderer = MjFilamentRenderer(model)
    renderer.update(data=data)

    image = renderer.render()
    pil_image = Image.fromarray(image)
    pil_image.save("test_render_filament.png")

    renderer.close()
