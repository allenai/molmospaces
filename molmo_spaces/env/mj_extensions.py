from functools import cached_property

import mujoco
import numpy as np
from lxml import etree
from mujoco import (
    MjModel,
    mjtObj,
)

from molmo_spaces.utils.mj_model_and_data_utils import extract_mj_names


class MjModelBindings:
    model: MjModel
    id: int
    body_names: list[str]
    body_name2id: dict[str, int]
    body_id2name: dict[int, str]
    joint_names: list[str]
    joint_name2id: dict[str, int]
    joint_id2name: dict[int, str]
    geom_names: list[str]
    geom_name2id: dict[str, int]
    geom_id2name: dict[int, str]
    site_names: list[str]
    site_name2id: dict[str, int]
    site_id2name: dict[int, str]
    light_names: list[str]
    light_name2id: dict[str, int]
    light_id2name: dict[int, str]
    camera_names: list[str]
    camera_name2id: dict[str, int]
    camera_id2name: dict[int, str]
    actuator_names: list[str]
    actuator_name2id: dict[str, int]
    actuator_id2name: dict[int, str]
    sensor_names: list[str]
    sensor_name2id: dict[str, int]
    sensor_id2name: dict[int, str]
    tendon_names: list[str]
    tendon_name2id: dict[str, int]
    tendon_id2name: dict[int, str]
    mesh_names: list[str]
    mesh_name2id: dict[str, int]
    mesh_id2name: dict[int, str]
    equality_names: list[str]
    equality_name2id: dict[str, int]
    equality_id2name: dict[int, str]

    # noinspection PyMissingConstructor
    def __init__(self, model: MjModel, xml_path: str | None = None) -> None:
        self.model = model
        self.xml_path = xml_path
        self.id = id(model)

    @cached_property
    def xml(self) -> etree.Element:
        assert self.xml_path is not None
        with open(self.xml_path) as f:
            return etree.fromstring(f.read())

    # noinspection PyMethodOverriding
    @classmethod
    def from_xml_path(cls, model: MjModel, filename: str) -> "MjModelBindings":
        return cls(model, xml_path=filename)

    def extract_mj_names(self, name_adr: np.ndarray | None, num_obj: int, obj_type: mjtObj):
        return extract_mj_names(self.model, name_adr, num_obj, obj_type)

    def __getattr__(self, item):
        if item.endswith("_names") or item.endswith("_name2id") or item.endswith("_id2name"):
            # Lazy-load the names, name2id, and id2name mappings

            k = item.split("_")[0]

            assert k in [
                "body",
                "joint",
                "geom",
                "site",
                "light",
                "camera",
                "actuator",
                "sensor",
                "tendon",
                "mesh",
                "equality",
            ]

            def key_to_nkey(key: str):
                return {
                    "actuator": "nu",
                    "joint": "njnt",
                    "camera": "ncam",
                    "equality": "neq",
                }.get(key, f"n{key}")

            def key_to_short_key(key: str):
                return {
                    "joint": "jnt",
                    "camera": "cam",
                    "equality": "eq",
                }.get(key, f"{key}")

            names, name2id, id2name = self.extract_mj_names(
                getattr(self.model, f"name_{key_to_short_key(k)}adr"),
                getattr(self.model, key_to_nkey(k)),
                getattr(mujoco.mjtObj, f"mjOBJ_{k.upper()}"),
            )
            setattr(self, f"{k}_names", names)
            setattr(self, f"{k}_name2id", name2id)
            setattr(self, f"{k}_id2name", id2name)
            return getattr(self, item)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
