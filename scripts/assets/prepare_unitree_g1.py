"""Prepare the Unitree G1 Dex1.1 model for MolmoSpaces.

The upstream Unitree Dex1.1 URDF has the geometry we want, but it does not
include a free base or MuJoCo actuators. This script creates a deterministic
MuJoCo XML asset with the MolmoSpaces control contract.
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR

ROBOT_ASSET_NAME = "unitree_g1_29dof_dex1_1"
SOURCE_URDF_REL_PATH = Path("g1_description/g1_29dof_mode_15_with_dex1_1.urdf")
SOURCE_MESH_DIR_REL_PATH = Path("g1_description/meshes")
OUTPUT_XML_NAME = "model.xml"
DEFAULT_PELVIS_HEIGHT_M = 0.793
DEX1_GRASP_SITE_POS = [0.148, 0.0, 0.0]
DEX1_FINGERTIP_PAD_SIZE = [0.02, 0.01, 0.045]
DEX1_FINGERTIP_PAD_POSITIONS = {
    "dex1_finger_link_1": [0.1065, -0.0285, 0.0],
    "dex1_finger_link_2": [0.1065, 0.0285, 0.0],
}
DEX1_FINGERTIP_PAD_FRICTION = [12.0, 0.2, 0.02]
DEX1_FINGERTIP_PAD_CONDIM = 6
DEX1_FINGERTIP_PAD_GROUP = 4
DEX1_HAND_FORCE_LIMIT_MULTIPLIER = 3.0
LEFT_ARM_STOW_QPOS = {
    "left_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": 0.35,
    "left_shoulder_yaw_joint": -0.1,
    "left_elbow_joint": 1.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}
LEFT_HAND_OPEN_QPOS = {
    "left_dex1_finger_joint_1": 0.0245,
    "left_dex1_finger_joint_2": 0.0245,
}


GAIN_BY_JOINT_GROUP = {
    "leg": (120.0, 12.0),
    "waist": (80.0, 8.0),
    "arm": (60.0, 6.0),
    "hand": (80.0, 8.0),
}


@dataclass(frozen=True)
class JointSpec:
    name: str
    lower: float
    upper: float
    effort: float | None
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unitree-urdf-root",
        type=Path,
        default=Path(os.environ["UNITREE_URDF_ROOT"])
        if "UNITREE_URDF_ROOT" in os.environ
        else None,
        help="Path to the Unitree `unitree_ros/robots` directory. Defaults to UNITREE_URDF_ROOT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROBOTS_DIR / ROBOT_ASSET_NAME,
        help=f"Output robot asset directory. Defaults to $MLSPACES_ASSETS_DIR/robots/{ROBOT_ASSET_NAME}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def _joint_group(joint_name: str) -> str:
    if "_hip_" in joint_name or "_knee_" in joint_name or "_ankle_" in joint_name:
        return "leg"
    if joint_name.startswith("waist_"):
        return "waist"
    if "_shoulder_" in joint_name or "_elbow_" in joint_name or "_wrist_" in joint_name:
        return "arm"
    if "_dex1_" in joint_name:
        return "hand"
    raise ValueError(f"Unexpected controllable G1 joint: {joint_name}")


def _parse_controllable_joints(urdf_path: Path) -> list[JointSpec]:
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name is None or joint_type in {None, "fixed", "floating"}:
            continue

        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Joint {joint_name} is missing a <limit> element")

        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            raise ValueError(f"Joint {joint_name} is missing lower/upper limits")

        effort = limit.get("effort")
        joints.append(
            JointSpec(
                name=joint_name,
                lower=float(lower),
                upper=float(upper),
                effort=float(effort) if effort is not None else None,
                group=_joint_group(joint_name),
            )
        )
    return joints


def _mesh_files_referenced_by_urdf(urdf_path: Path) -> set[Path]:
    root = ET.parse(urdf_path).getroot()
    mesh_files = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue
        mesh_files.add(Path(filename))
    return mesh_files


def _copy_referenced_meshes(source_root: Path, urdf_path: Path, output_dir: Path) -> None:
    source_mesh_root = source_root / SOURCE_MESH_DIR_REL_PATH
    output_mesh_root = output_dir / "meshes"
    output_mesh_root.mkdir(parents=True, exist_ok=True)

    for mesh_file in sorted(_mesh_files_referenced_by_urdf(urdf_path)):
        if mesh_file.is_absolute():
            source_mesh_path = mesh_file
            relative_mesh_path = Path(mesh_file.name)
        else:
            relative_mesh_path = Path(mesh_file.name)
            source_mesh_path = source_mesh_root / relative_mesh_path

        if not source_mesh_path.is_file():
            raise FileNotFoundError(f"Referenced mesh does not exist: {source_mesh_path}")

        destination = output_mesh_root / relative_mesh_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_mesh_path, destination)


def _add_position_actuators(spec: mujoco.MjSpec, joints: list[JointSpec]) -> None:
    for joint in joints:
        stiffness, damping = GAIN_BY_JOINT_GROUP[joint.group]
        kwargs = {}
        if joint.effort is not None:
            effort = (
                joint.effort * DEX1_HAND_FORCE_LIMIT_MULTIPLIER
                if joint.group == "hand"
                else joint.effort
            )
            kwargs["forcelimited"] = 1
            kwargs["forcerange"] = [-effort, effort]

        spec.add_actuator(
            name=joint.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint.name,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[stiffness, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0.0, -stiffness, -damping, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=1,
            ctrlrange=[joint.lower, joint.upper],
            **kwargs,
        )


def _add_kinematics_sites(spec: mujoco.MjSpec) -> None:
    for side in ("left", "right"):
        wrist_body = spec.body(f"{side}_wrist_yaw_link")
        if wrist_body is None:
            raise ValueError(f"Expected body `{side}_wrist_yaw_link` in Unitree G1 model")
        wrist_body.add_site(
            name=f"{side}_wrist_site",
            pos=[0.0, 0.0, 0.0],
            size=[0.015],
            rgba=[0.1, 0.4, 1.0, 1.0],
        )
        wrist_body.add_site(
            name=f"{side}_grasp_site",
            pos=DEX1_GRASP_SITE_POS,
            size=[0.015],
            rgba=[0.1, 1.0, 0.4, 1.0],
        )


def _add_dex1_fingertip_contact_pads(spec: mujoco.MjSpec) -> None:
    for side in ("left", "right"):
        for finger_suffix, pad_pos in DEX1_FINGERTIP_PAD_POSITIONS.items():
            finger_body_name = f"{side}_{finger_suffix}"
            finger_body = spec.body(finger_body_name)
            if finger_body is None:
                raise ValueError(f"Expected body `{finger_body_name}` in Unitree G1 model")

            finger_id = finger_suffix.rsplit("_", maxsplit=1)[-1]
            finger_body.add_geom(
                name=f"{side}_dex1_fingertip_pad_{finger_id}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pad_pos,
                size=DEX1_FINGERTIP_PAD_SIZE,
                contype=1,
                conaffinity=1,
                condim=DEX1_FINGERTIP_PAD_CONDIM,
                priority=2,
                friction=DEX1_FINGERTIP_PAD_FRICTION,
                density=0.0,
                group=DEX1_FINGERTIP_PAD_GROUP,
                rgba=[0.05, 0.8, 0.2, 0.35],
            )


def prepare_unitree_g1(
    unitree_urdf_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    force: bool = False,
) -> Path:
    source_root = Path(unitree_urdf_root).expanduser().resolve()
    urdf_path = source_root / SOURCE_URDF_REL_PATH
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Could not find Unitree G1 Dex1.1 URDF: {urdf_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    joints = _parse_controllable_joints(urdf_path)
    if len(joints) != 33:
        raise ValueError(f"Expected 33 controllable joints, found {len(joints)}")

    _copy_referenced_meshes(source_root, urdf_path, output_dir)

    spec = mujoco.MjSpec.from_file(str(urdf_path))
    pelvis = spec.body("pelvis")
    if pelvis is None:
        raise ValueError("Expected root body `pelvis` in Unitree G1 URDF")
    pelvis.pos = [0.0, 0.0, DEFAULT_PELVIS_HEIGHT_M]
    pelvis.add_freejoint(name="floating_base_joint")
    _add_kinematics_sites(spec)
    _add_dex1_fingertip_contact_pads(spec)
    _add_position_actuators(spec, joints)

    model = spec.compile()
    expected_shape = (40, 39, 33)
    actual_shape = (model.nq, model.nv, model.nu)
    if actual_shape != expected_shape:
        raise ValueError(f"Expected model shape nq/nv/nu={expected_shape}, got {actual_shape}")

    output_xml_path = output_dir / OUTPUT_XML_NAME
    spec.to_file(str(output_xml_path))
    mujoco.MjModel.from_xml_path(str(output_xml_path))
    return output_xml_path


def main() -> None:
    args = parse_args()
    if args.unitree_urdf_root is None:
        raise SystemExit("Set UNITREE_URDF_ROOT or pass --unitree-urdf-root")

    output_xml_path = prepare_unitree_g1(args.unitree_urdf_root, args.output_dir, args.force)
    print(f"Wrote Unitree G1 Dex1.1 model: {output_xml_path}")


if __name__ == "__main__":
    main()
