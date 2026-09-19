from __future__ import annotations

import logging
import random
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.utils.randomization import speckle_texture

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig


log = logging.getLogger(__name__)


class FrankaRobot(Robot):
    def __init__(self, mj_data: mj.MjData, config: BaseRobotConfig) -> None:
        super().__init__(mj_data, config)

        assert config.robot_view_factory, (
            "Something went wrong, 'robot_view_factory' shouldn't be None"
        )
        self._robot_view = config.robot_view_factory(mj_data, config.robot_namespace)
        self._kinematics = MlSpacesKinematics(config)

        # TODO(wilbert): use this one only if not in Darwin|MacOS, as warp works poorly when doing
        # emulation on non-nvidia hardware
        self._parallel_kinematics = SimpleWarpKinematics(config)
        arm_controller_cls = (
            JointPosController
            if config.command_mode == {} or config.command_mode["arm"] == "joint_position"
            else JointRelPosController
        )
        self._controllers: dict[str, Controller] = {
            "arm": arm_controller_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self):
        return self.config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def create_robot_sensors(self):
        return super().create_robot_sensors() + [
            TCPPoseSensor(uuid="tcp_pose"),
        ]

    def get_arm_move_group_ids(self) -> list[str]:
        """Franka has a single arm move group."""
        return ["arm"]

    def reset(self) -> None:
        for mg_id, default_pos in self.config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = np.array(default_pos)

    @staticmethod
    def robot_model_root_name() -> str:
        return "fr3_link0"

    @classmethod
    def create_robot_base_material(
        cls,
        robot_config: FrankaRobotConfig,
        spec: mj.MjSpec,
        prefix: str,
        randomize_base_texture: bool,
    ) -> str:
        texture_dir = robot_config.get_robot_dir() / "assets" / "base_textures"
        assert texture_dir.is_dir(), f"Texture directory {texture_dir} does not exist"
        texture_path: Path | None = None
        if randomize_base_texture:
            texture_paths = list(texture_dir.glob("*.png"))
            texture_paths.sort(key=lambda x: x.name)
            assert len(texture_paths) > 0, f"No robot base texture paths found in {texture_dir}"
            log.debug(f"Found {len(texture_paths)} robot base texture paths")
            texture_path = random.choice(texture_paths)
        else:
            texture_path = texture_dir / "DarkWood2.png"
            assert texture_path.is_file(), f"Default texture {texture_path} does not exist"

        texture_name = f"{prefix}robot_base_texture"
        spec.add_texture(
            name=texture_name,
            type=mj.mjtTexture.mjTEXTURE_CUBE,
            file=str(texture_path),
        )
        log.debug(f"Successfully created texture from {texture_path}")

        material_name = f"{prefix}robot_base_material"
        robot_base_mat = spec.add_material(name=material_name)
        robot_base_mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = texture_name
        log.debug(f"Successfully created material {material_name}")
        return material_name

    @classmethod
    def randomize_robot_textures(
        cls,
        robot_config: FrankaRobotConfig,
        spec: mj.MjSpec,
        prefix: str,
        robot_spec: mj.MjSpec,
    ):
        if random.random() > robot_config.perturb_texture_probability:
            log.info(f"Skipping texture randomization for robot '{robot_config.name}'")
            return

        perturbed_materials: dict[str, str] = {}
        for material in robot_spec.materials:
            material: mj.MjsMaterial
            is_rgb_mat = all(
                material.textures[i] == "" for i in range(mj.mjtTextureRole.mjNTEXROLE)
            )
            if not is_rgb_mat:
                continue

            speckle_img = speckle_texture(material.rgba[:3])
            buffer = BytesIO()
            speckle_img.save(buffer, format="PNG")
            buffer.seek(0)

            tex_name = f"{material.name}_perturbed_tex"
            mat_name = f"{material.name}_perturbed"
            fn = f"{prefix}{tex_name}.png".replace("/", "__")
            spec.assets[fn] = buffer.getvalue()
            robot_spec.add_texture(name=tex_name, type=mj.mjtTexture.mjTEXTURE_2D, file=fn)
            perturbed_mat = robot_spec.add_material(name=mat_name)
            perturbed_mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = tex_name
            perturbed_materials[material.name] = mat_name

        def set_material(body: mj.MjsBody):
            for geom in body.geoms:
                geom: mj.MjsGeom
                if geom.material in perturbed_materials:
                    log.debug(
                        f"Setting material {geom.material} to {perturbed_materials[geom.material]} "
                        f"for geom '{geom.name}' in body '{body.name}'"
                    )
                    geom.material = perturbed_materials[geom.material]
            for child in body.bodies:
                set_material(child)

        robot_body = robot_spec.body(cls.robot_model_root_name())
        set_material(robot_body)
        log.info(f"Successfully randomized robot textures for robot '{robot_config.name}'")

    @classmethod
    def merge_gripper_pads(cls, robot_spec: mj.MjSpec) -> int:
        """Fuse each finger's two pad boxes into one, so they share no seam.

        Each Robotiq finger carries its pad as two stacked boxes on one body,
        pad1 and pad2. An object thin enough to penetrate that boundary ends up
        inside both boxes at once, and each pushes it out of itself and therefore
        into the other. The surviving collider is ``pad2``, which is the geom the
        robot views identify each finger by. Returns how many fingers were merged.
        """
        pads: dict[str, dict[str, mj.MjsGeom]] = {}
        for geom in robot_spec.geoms:
            if geom.name.endswith("_pad1") or geom.name.endswith("_pad2"):
                pads.setdefault(geom.name[:-1], {})[geom.name[-1]] = geom
        merged = 0
        for finger, pair in sorted(pads.items()):
            if set(pair) != {"1", "2"}:
                continue
            first, second = pair["1"], pair["2"]
            low = min(first.pos[2] - first.size[2], second.pos[2] - second.size[2])
            high = max(first.pos[2] + first.size[2], second.pos[2] + second.size[2])
            second.pos = np.array([second.pos[0], second.pos[1], (low + high) / 2.0])
            second.size = np.array([second.size[0], second.size[1], (high - low) / 2.0])
            # Kept as geometry but no longer colliding: the merged box already
            # covers its span, and two colliders over one span is the whole bug.
            first.contype = 0
            first.conaffinity = 0
            merged += 1
            log.debug("merged gripper pads for %s into %s2", finger, finger)
        return merged

    # TODO(wilbert): uhmm, this part should be moved to a regular free function, or a factory fcn
    # that is registered via metaclasses when creating the robot class
    @classmethod
    def add_robot_to_scene(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        cls,
        robot_config: FrankaRobotConfig,
        spec: mj.MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        # TODO(wilbert): this part is iffy, we shouldn't expect to receive either both vec3 and vec2
        # here. The caller should just send a vec3 and if he just wants xy then the last element
        # should be just 0. We could maybe change list[float] to tuple[float,float,float] and so on
        pos_xyz = pos + [0.0] if len(pos) == 2 else pos.copy()

        material_name = cls.create_robot_base_material(
            robot_config, spec, prefix, randomize_textures
        )

        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos_xyz,
            quat=quat,
            mocap=True,
        )
        if robot_config.base_size is not None:
            assert robot_config.base_size, (
                "If using 'base' must provide 'base_size' in configuration"
            )
            base_height = robot_config.base_size[2]

            # Add base geometry (wooden platform)
            robot_body.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[x / 2 for x in robot_config.base_size],
                pos=[0, 0, base_height / 2],
                material=material_name,
                group=0,  # Visual group
            )
            attach_frame = robot_body.add_frame(pos=[0, 0, base_height])
        else:
            attach_frame = robot_body.add_frame()

        robot_spec = cls._load_robot_spec(robot_config, strip_meshes=strip_meshes)
        cls.merge_gripper_pads(robot_spec)

        if randomize_textures:
            cls.randomize_robot_textures(robot_config, spec, prefix, robot_spec)

        robot_root_name = cls.robot_model_root_name()
        robot_root = robot_spec.body(robot_root_name)
        if robot_root is None:
            raise ValueError(f"Robot {robot_root_name=} not found in {robot_spec}")
        attach_frame.attach_body(robot_root, prefix, "")
