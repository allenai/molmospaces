import abc
from enum import StrEnum

import mujoco as mj
import numpy as np


class MSRenderMode(StrEnum):
    RGB = "rgb"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"


class IMSRenderer(abc.ABC):
    def __init__(
        self,
        model: mj.MjModel,
        width: int = 1280,
        height: int = 720,
        max_geom: int = 10000,
        device_id: int | None = None,
    ) -> None:
        self._model = model
        self._width = width
        self._height = height
        self._max_geom = max_geom

        self._mode = MSRenderMode.RGB

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @abc.abstractmethod
    def set_mode(self, mode: MSRenderMode) -> None: ...

    @abc.abstractmethod
    def update(self, data: mj.MjData, cam_id_or_name: int | str = -1) -> None: ...

    @abc.abstractmethod
    def render(self) -> np.ndarray: ...

    @abc.abstractmethod
    def close(self) -> None: ...
