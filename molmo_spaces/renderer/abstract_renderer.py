import abc
from typing import Any

import mujoco as mj

RENDERING_COMPLETE = "RENDERING_COMPLETE"


class MjAbstractRenderer(abc.ABC):
    render_outputs: list[Any]

    def __init__(self, model: mj.MjModel, device_id: int | None = None) -> None:
        self._model = model
        self._device_id = device_id

        self._scene: mj.MjvScene | None = None

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def device_id(self) -> int | None:
        return self._device_id

    @property
    def scene(self) -> mj.MjvScene:
        assert self._scene is not None, "Internal scene:MjvScene must be initialized by now"
        return self._scene

    @abc.abstractmethod
    def enable_depth_rendering(self) -> None: ...

    @abc.abstractmethod
    def disable_depth_rendering(self) -> None: ...

    @abc.abstractmethod
    def enable_segmentation_rendering(self) -> None: ...

    @abc.abstractmethod
    def disable_segmentation_rendering(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> Any: ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None: ...
