import mujoco as mj
import mujoco._render_filament as mjrf  # ty: ignore
import mujoco.experimental.studio.window  # ty: ignore
import numpy as np
from mujoco.rendering.filament.renderer import Renderer  # ty: ignore

from ..base import IMSRenderer, MSRenderMode


class MSFilamentRenderer(IMSRenderer):
    def __init__(
        self,
        model: mj.MjModel,
        width: int = 1280,
        height: int = 720,
        max_geom: int = 10000,
        device_id: int | None = None,
    ) -> None:
        super().__init__(model, width, height, max_geom, device_id)

        self._ctx = mjrf.Context(
            mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL)
        )

        self._impl: Renderer | None = Renderer(self._ctx)

        self._rf_objects = mjrf.ModelObjects(self._ctx, model)
        self._rf_scene = self._impl.scene("scn")  # ty: ignore
        self._rf_lights = mjrf.ModelLights(self._rf_scene, self._rf_objects)
        self._rf_renderables = mjrf.ModelRenderables(self._rf_scene, self._rf_objects)
        self._rf_views_names = set()

        self.set_mode(MSRenderMode.RGB)

    def set_mode(self, mode: MSRenderMode) -> None:
        assert self._impl is not None, "Must have the renderer initialized by now"

        self._mode = mode
        view_name = f"view_{self._mode}"

        if view_name in self._rf_views_names:
            return

        self._rf_views_names.add(view_name)

        match mode:
            case MSRenderMode.RGB:
                self._impl.view(
                    view_name,
                    scene="scn",
                    target="out",
                    draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT,
                )
            case MSRenderMode.DEPTH:
                self._impl.view(
                    view_name,
                    scene="scn",
                    target="out",
                    draw_mode=mjrf.DrawMode.DRAW_MODE_DEPTH,
                )
            case MSRenderMode.SEGMENTATION:
                self._impl.view(
                    view_name,
                    scene="scn",
                    target="out",
                    draw_mode=mjrf.DrawMode.DRAW_MODE_SEGMENTATION_BY_ID,
                )

    def update(self, data: mj.MjData, cam_id_or_name: int | str = -1) -> None:
        assert self._impl is not None, "Must have the renderer initialized by now"

        self._rf_lights.update(data)
        self._rf_renderables.update(data)

        camera_id = -1
        if isinstance(cam_id_or_name, str):
            camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA.value, cam_id_or_name)
            if camera_id == -1:
                raise ValueError(f'The camera "{cam_id_or_name}" does not exist.')
        if camera_id < -1 or camera_id >= self._model.ncam:
            raise ValueError(f"The camera id {camera_id} is out of range [-1, {self.model.ncam}).")

        camera = mj.MjvCamera()
        camera.fixedcamid = camera_id

        if camera_id == -1:
            camera.type = mj.mjtCamera.mjCAMERA_FREE
            mj.mjv_defaultFreeCamera(self.model, camera)
        else:
            camera.type = mj.mjtCamera.mjCAMERA_FIXED

        glcam = mj.mjv_camera2GLCamera(self._model, data, camera)
        self._impl.update_camera(f"view_{self._mode}", glcam)
        self._impl.target("out", (self._width, self._height))

    def render(self) -> np.ndarray:
        assert self._impl is not None, "Must have the renderer initialized by now"

        self._impl.render()
        return self._impl.get_image("out").__array__()

    def close(self) -> None:
        self._impl = None


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

    renderer = MSFilamentRenderer(model)

    # for rmode in (MSRenderMode.RGB, MSRenderMode.DEPTH, MSRenderMode.SEGMENTATION):
    for rmode in (MSRenderMode.RGB,):
        renderer.set_mode(rmode)
        renderer.update(data=data)

        image = renderer.render()
        pil_image = Image.fromarray(image)
        pil_image.save(f"test_render_filament_{rmode}_{args.model.stem}.png")

    renderer.close()
