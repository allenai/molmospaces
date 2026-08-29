import logging
import random
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjsBody, MjSpec, mjtGeom
from PIL import Image, ImageDraw

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.env.sensors import TCPPoseSensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.robots.abstract import Robot

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig


log = logging.getLogger(__name__)


def _speckle_texture(
    base_color,
    size=256,
    noise_strength=0.1,
    num_blobs=80,
    blob_size_range=(5, 25),
    blob_variation=0.1,
):
    img = np.ones((size, size, 3)) * np.array(base_color)
    noise = np.random.normal(0, noise_strength, (size, size, 1))
    img += noise
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Draw chunky rectangular or elliptical blobs
    for _ in range(num_blobs):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        w = np.random.randint(*blob_size_range)
        h = np.random.randint(*blob_size_range)

        variation = np.random.uniform(-blob_variation, blob_variation)
        blob_color = tuple(int(np.clip((c + variation) * 255, 0, 255)) for c in base_color)

        if np.random.random() > 0.5:
            draw.ellipse((x, y, x + w, y + h), fill=blob_color)
        else:
            draw.rectangle((x, y, x + w, y + h), fill=blob_color)

    return pil_img


class FrankaRobot(Robot):
    """Franka robot implementation for the framework."""

    def __init__(
        self,
        mj_data: MjData,
        config: "MlSpacesExpConfig",
    ) -> None:
        super().__init__(mj_data, config)
        self._robot_view = config.robot_config.robot_view_factory(
            mj_data, config.robot_config.robot_namespace
        )
        self._kinematics = MlSpacesKinematics(config.robot_config)

        self._parallel_kinematics = SimpleWarpKinematics(config.robot_config)
        arm_controller_cls = (
            JointPosController
            if config.robot_config.command_mode == {}
            or config.robot_config.command_mode["arm"] == "joint_position"
            else JointRelPosController
        )
        self._controllers = {
            "arm": arm_controller_cls(self._robot_view.get_move_group("arm")),
            "gripper": JointPosController(self._robot_view.get_move_group("gripper")),
        }

    @property
    def namespace(self):
        return self.exp_config.robot_config.robot_namespace

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
        for mg_id, default_pos in self.exp_config.robot_config.init_qpos.items():
            if mg_id in self._robot_view.move_group_ids():
                self._robot_view.get_move_group(mg_id).joint_pos = default_pos

    @staticmethod
    def robot_model_root_name() -> str:
        return "fr3_link0"

    @classmethod
    def create_robot_base_material(
        cls,
        robot_config: "FrankaRobotConfig",
        spec: MjSpec,
        prefix: str,
        randomize_base_texture: bool,
    ) -> None:
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
            type=mujoco.mjtTexture.mjTEXTURE_CUBE,
            file=str(texture_path),
        )
        log.debug(f"Successfully created texture from {texture_path}")

        material_name = f"{prefix}robot_base_material"
        robot_base_mat = spec.add_material(name=material_name)
        robot_base_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
        log.debug(f"Successfully created material {material_name}")
        return material_name

    @classmethod
    def randomize_robot_textures(
        cls,
        robot_config: "FrankaRobotConfig",
        spec: MjSpec,
        prefix: str,
        robot_spec: MjSpec,
    ):
        if random.random() > robot_config.perturb_texture_probability:
            log.info(f"Skipping texture randomization for robot '{robot_config.name}'")
            return

        perturbed_materials: dict[str, str] = {}
        for material in robot_spec.materials:
            material: mujoco.MjsMaterial
            is_rgb_mat = all(
                material.textures[i] == "" for i in range(mujoco.mjtTextureRole.mjNTEXROLE)
            )
            if not is_rgb_mat:
                continue

            speckle_img = _speckle_texture(material.rgba[:3])
            buffer = BytesIO()
            speckle_img.save(buffer, format="PNG")
            buffer.seek(0)

            tex_name = f"{material.name}_perturbed_tex"
            mat_name = f"{material.name}_perturbed"
            fn = f"{prefix}{tex_name}.png".replace("/", "__")
            spec.assets[fn] = buffer.getvalue()
            robot_spec.add_texture(name=tex_name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=fn)
            perturbed_mat = robot_spec.add_material(name=mat_name)
            perturbed_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex_name
            perturbed_materials[material.name] = mat_name

        def set_material(body: MjsBody):
            for geom in body.geoms:
                geom: mujoco.MjsGeom
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
    def merge_gripper_pads(cls, robot_spec: MjSpec) -> int:
        """Fuse each finger's two pad boxes into one, so they share no seam.

        Each Robotiq finger carries its pad as two stacked boxes on one body,
        18.75 mm tall each, meeting flush: pad1 spans 0.11063-0.12937 and pad2
        0.12938-0.14813. An object thin enough to penetrate that boundary ends up
        inside both boxes at once, and each pushes it out of itself and therefore
        into the other. The contacts oppose -- measured on a paper cup, normals at
        -0.983 carrying 1190 N and 853 N from two geoms on the *same* finger,
        against 26 N on the whole opposite finger -- there is no equilibrium to
        settle into, and nothing can release the object, because opening the jaws
        moves the finger, the seam and whatever is caught in it together. The cup
        was carried away wedged in one finger at 2703 N.

        That is a collision artefact rather than a grip: one box over the same
        span is the same shape to everything outside the gripper, and has no
        interior boundary to catch anything on. Any thin wall, rim, lid, plate or
        sheet can reach the seam, so this is applied unconditionally.

        The surviving collider is ``pad2``, which is the geom the robot views
        identify each finger by (see ``franka_droid_view``/``franka_cap_view``).
        Only the local z extent changes, so the distance between the two fingers
        -- and hence ``inter_finger_dist`` -- is untouched.

        Returns how many fingers were merged.
        """
        pads: dict[str, dict[str, mujoco.MjsGeom]] = {}
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
            second.pos = [second.pos[0], second.pos[1], (low + high) / 2.0]
            second.size = [second.size[0], second.size[1], (high - low) / 2.0]
            # Kept as geometry but no longer colliding: the merged box already
            # covers its span, and two colliders over one span is the whole bug.
            first.contype = 0
            first.conaffinity = 0
            merged += 1
            log.debug("merged gripper pads for %s into %s2", finger, finger)
        return merged

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "FrankaRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        robot_config = cast("FrankaRobotConfig", robot_config)
        add_base = robot_config.base_size is not None
        pos = pos + [0.0] if len(pos) == 2 else pos

        material_name = cls.create_robot_base_material(
            robot_config, spec, prefix, randomize_textures
        )

        robot_body = spec.worldbody.add_body(
            name=f"{prefix}base",
            pos=pos,
            quat=quat,
            mocap=True,
        )
        if add_base:
            base_height = robot_config.base_size[2]

            # Add base geometry (wooden platform)
            robot_body.add_geom(
                type=mjtGeom.mjGEOM_BOX,
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
