import abc
from typing import Any

import mujoco as mj

RENDERING_COMPLETE = "RENDERING_COMPLETE"


class MjAbstractRenderer(abc.ABC):
    render_outputs: list[Any]

    def __init__(self, model: mj.MjModel, device_id: int | None = None) -> None:
        self._model = model
        self._device_id = device_id

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def device_id(self) -> int | None:
        return self._device_id

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> Any: ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None: ...
