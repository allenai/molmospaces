import mujoco as mj
import numpy as np
from mujoco.rendering.classic.renderer import Renderer

from ..base import IMSRenderer, MSRenderMode


class MSClassicRenderer(IMSRenderer):
    def __init__(
        self,
        model: mj.MjModel,
        width: int = 1280,
        height: int = 720,
        max_geom: int = 10000,
        device_id: int | None = None,
    ) -> None:
        super().__init__(model, width, height, max_geom, device_id)

        self._impl: Renderer | None = Renderer(
            model=model, width=width, height=height, max_geom=max_geom
        )

    def set_mode(self, mode: MSRenderMode) -> None:
        assert self._impl is not None, "Must have the renderer initialized by now"

        self._mode = mode

        match mode:
            case MSRenderMode.RGB:
                self._impl.disable_depth_rendering()
                self._impl.disable_segmentation_rendering()
            case MSRenderMode.DEPTH:
                self._impl.enable_depth_rendering()
            case MSRenderMode.SEGMENTATION:
                self._impl.enable_segmentation_rendering()

    def update(self, data: mj.MjData, cam_id_or_name: int | str = -1) -> None:
        assert self._impl is not None, "Must have the renderer initialized by now"
        self._impl.update_scene(data, cam_id_or_name)

    def render(self) -> np.ndarray:
        assert self._impl is not None, "Must have the renderer initialized by now"
        return self._impl.render()

    def close(self) -> None:
        if self._impl:
            self._impl.close()
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

    renderer = MSClassicRenderer(model)

    # for rmode in (MSRenderMode.RGB, MSRenderMode.DEPTH, MSRenderMode.SEGMENTATION):
    for rmode in (MSRenderMode.RGB,):
        renderer.set_mode(rmode)
        renderer.update(data=data)

        image = renderer.render()
        pil_image = Image.fromarray(image)

        img_name = f"test_render_classic_{rmode}_{args.model.stem}.png"
        pil_image.save(img_name)

        print(f"Saved: {img_name}")

    renderer.close()
