import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from scipy.spatial.transform import Rotation as R

from molmo_spaces.molmo_spaces_constants import ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR

__all__ = ["MotionGenStatus"]


def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def xml_to_world_cfg(xml_file: str, ignore_strings: list[str] = []) -> dict:
    """
    Parses a mujoco xml file for geometries and returns a world config dict expected by curobo.
    Currently supports collision types: "obb" ("cuboid"), "mesh", "capsule", "sphere", and "cylinder".

    Args:
        xml_file: Path to the mujoco xml file.
        ignore_strings: List of strings to ignore when parsing the xml file.
    """

    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Collision classes for filtering
    collision_classes = [
        "__STRUCTURAL_MJT__",
        "__DYNAMIC_MJT__",
    ]  # "__VISUAL_MJT__" # not including visual geoms

    # store the mesh file paths as a dict
    xml_path = Path(xml_file)
    mesh_dict = {}
    for mesh in root.findall(".//mesh"):  # asset/mesh
        mesh_name = mesh.get("name", "")
        mesh_file = mesh.get("file", "")
        mesh_scale = mesh.get("scale", "1 1 1")
        mesh_path = (xml_path.parent / mesh_file).resolve()
        mesh_scale_list = [float(x) for x in mesh_scale.split()]
        mesh_dict[mesh_name] = {"file": mesh_path, "scale": mesh_scale_list}
    # print(f"Extracted {len(mesh_dict)} meshes from XML.")

    # Extract body and mesh information
    mesh_collision_geoms = {}
    cuboid_collision_geoms = {}
    capsule_collision_geoms = {}
    sphere_collision_geoms = {}
    cylinder_collision_geoms = {}

    def process_body(body, parent_pos=None, parent_quat=None) -> None:
        if parent_pos is None:
            parent_pos = np.array([0.0, 0.0, 0.0])
        if parent_quat is None:
            parent_quat = np.array([1.0, 0.0, 0.0, 0.0])

        """Recursively process bodies and apply transformations."""
        # Get local pose
        local_pos = np.fromstring(body.get("pos", "0.0 0.0 0.0"), sep=" ")
        local_quat = np.fromstring(body.get("quat", "1.0 0.0 0.0 0.0"), sep=" ")  # (w, x, y, z)
        # Compute global pose by adding parent's position and rotation
        parent_rotation = R.from_quat(parent_quat, scalar_first=True)
        global_pos = parent_pos + parent_rotation.apply(local_pos)
        global_quat = quaternion_multiply(parent_quat, local_quat)
        global_rotation = R.from_quat(global_quat, scalar_first=True)
        global_pose = np.concatenate([global_pos, global_quat]).tolist()

        # Process geoms
        for geom in body.findall("geom"):
            # Ensure this is a collision-related geom
            parent_classes = geom.get("class", "").split()
            if not any(cls in parent_classes for cls in collision_classes):
                continue  # Skip non-collision geoms

            geom_name = geom.get("name", "")
            geom_type = geom.get("type", "")

            if geom_type == "mesh":
                mesh_name = geom.get("mesh", "")
                mesh_path = mesh_dict[mesh_name]["file"]
                mesh_scale = mesh_dict[mesh_name]["scale"]
                mesh_collision_geoms[geom_name] = {
                    "pose": global_pose,  # [x, y, z, qw, qx, qy, qz]
                    "file_path": mesh_path,
                    "scale": mesh_scale,  # if not body.get("name", "").startswith("wall") else [-1.0, 1.0, 1.0]
                }
            else:
                # for all non-mesh geom types, further geom local pos and quat may be set
                geom_local_pos = np.fromstring(geom.get("pos", "0.0 0.0 0.0"), sep=" ")
                geom_local_quat = np.fromstring(geom.get("quat", "1.0 0.0 0.0 0.0"), sep=" ")
                geom_global_pos = global_pos + global_rotation.apply(geom_local_pos)
                geom_global_quat = quaternion_multiply(global_quat, geom_local_quat)
                geom_global_pose = np.concatenate([geom_global_pos, geom_global_quat]).tolist()
                geom_size = np.fromstring(geom.get("size"), sep=" ")

                if geom_type == "box":
                    # MuJoCo box: size is half-extents
                    bbox_extents = geom_size * 2.0
                    cuboid_collision_geoms[geom_name] = {
                        "pose": geom_global_pose,  # [x, y, z, qw, qx, qy, qz]
                        "dims": bbox_extents,  # [x, y, z] extents
                    }
                elif geom_type == "capsule":
                    # MuJoCo capsule: size is radius, half-height (along Z axis)
                    radius = geom_size[0]
                    half_height = geom_size[1]
                    capsule_collision_geoms[geom_name] = {
                        "pose": geom_global_pose,  # [x, y, z, qw, qx, qy, qz]
                        "radius": radius,
                        "base": [0.0, 0.0, -half_height],
                        "tip": [0.0, 0.0, half_height],
                    }
                elif geom_type == "sphere":
                    # MuJoCo sphere: size is radius
                    radius = geom_size[0]
                    sphere_collision_geoms[geom_name] = {
                        "pose": geom_global_pose,  # [x, y, z, qw, qx, qy, qz]
                        "radius": radius,
                    }
                elif geom_type == "cylinder":
                    # MuJoCo cylinder: size is radius, half-height (along Z axis)
                    radius = geom_size[0]
                    half_height = geom_size[1]
                    cylinder_collision_geoms[geom_name] = {
                        "pose": geom_global_pose,  # [x, y, z, qw, qx, qy, qz]
                        "radius": radius,
                        "height": 2 * half_height,
                    }
                else:
                    print(
                        f'Curobo world xml warning: Ignoring unsupported geom "{geom_name}" of type: {geom_type}'
                    )

        # Recursively process child bodies
        for child_body in body.findall("body"):
            process_body(child_body, parent_pos=global_pos, parent_quat=global_quat)

    # Start recursion from the root body elements
    for body in root.findall("./worldbody/body"):
        # Skip bodies with ignore strings
        if any(ignore_string in body.get("name", "") for ignore_string in ignore_strings):
            continue
        process_body(body)
    # print(f"Extracted {len(collision_geoms)} collision geoms for world config.")

    world_dict = {}
    world_dict["mesh"] = mesh_collision_geoms
    world_dict["cuboid"] = cuboid_collision_geoms
    world_dict["capsule"] = capsule_collision_geoms
    world_dict["sphere"] = sphere_collision_geoms
    world_dict["cylinder"] = cylinder_collision_geoms

    return world_dict


def extract_nearby_geoms(world_dict, origin_xy_pos, radius, consider_strings=[], ignore_strings=[]):
    nearby_geoms_dict = {}
    nearby_geoms_dict["mesh"] = {}
    nearby_geoms_dict["cuboid"] = {}
    nearby_geoms_dict["capsule"] = {}
    nearby_geoms_dict["sphere"] = {}
    nearby_geoms_dict["cylinder"] = {}

    for geom_type in world_dict:
        for name, geom in world_dict[geom_type].items():
            if any(ignore_string in name for ignore_string in ignore_strings):
                # geoms with these strings can be ignored
                continue
            geom_xy_pos = geom["pose"][:2]
            dist = np.linalg.norm(geom_xy_pos - origin_xy_pos)
            # Always include geoms within the radius, or those matching consider_strings even if outside radius
            if dist < radius or any(
                consider_string in name for consider_string in consider_strings
            ):
                nearby_geoms_dict[geom_type][name] = geom

    return nearby_geoms_dict


def extract_local_mesh_objects(world_dict, target_name, base_pose_world, radius=1.5):
    base_pos = np.array(base_pose_world[:3])
    base_quat = np.array(base_pose_world[3:])
    base_rot = R.from_quat(base_quat, scalar_first=True)

    local_mesh_dict = {}

    for name, obj in world_dict["mesh"].items():
        # if target_name in name:
        #     continue
        obj_pose = np.array(obj["pose"])
        obj_pos = obj_pose[:3]
        obj_quat = obj_pose[3:]

        dist = np.linalg.norm(obj_pos[:2] - base_pos[:2])  # only consider x and y
        if dist > radius:
            continue

        rel_pos = base_rot.inv().apply(obj_pos - base_pos)
        rel_rot = R.from_quat(obj_quat, scalar_first=True)
        rel_quat = (base_rot.inv() * rel_rot).as_quat(scalar_first=True)

        local_pose = np.concatenate([rel_pos, rel_quat])

        local_mesh_dict[name] = {
            "pose": local_pose.tolist(),
            "file_path": obj["file_path"],
            "scale": obj["scale"],
        }

    return {"mesh": local_mesh_dict}


if __name__ == "__main__":
    save_path = (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR + "/debug/procthor_train/train_22_worldconfig.obj"
    )
    world_dict = xml_to_world_cfg(
        xml_file=ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR + "/debug/procthor_train/train_22_xarm7.xml"
    )
    world_model = WorldConfig.from_dict(world_dict)
    world_model.save_world_as_mesh(file_path=save_path, save_as_scene_graph=True)
