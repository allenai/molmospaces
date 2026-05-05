import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import NoReturn

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.utils.linalg_utils import inverse_homogeneous_matrix
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_pos_quat


class MlSpacesObjectAbstract(ABC):
    def __init__(self, data: mujoco.MjData, name: str) -> None:
        self.mj_data = data
        self._name = name

    @cached_property
    def mj_model(self):
        return self.mj_data.model

    @property
    def name(self) -> str:
        return self._name

    @property
    def pose(self) -> np.ndarray:
        return pos_quat_to_pose_mat(self.position, self.quat)

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        self.position, self.quat = pose_mat_to_pos_quat(pose)

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        raise NotImplementedError

    @position.setter
    @abstractmethod
    def position(self, position: np.ndarray) -> NoReturn:
        raise NotImplementedError

    @property
    @abstractmethod
    def quat(self) -> np.ndarray:
        raise NotImplementedError

    @quat.setter
    @abstractmethod
    def quat(self, quat: np.ndarray) -> NoReturn:
        raise NotImplementedError


class MlSpacesBody(MlSpacesObjectAbstract):
    def __init__(self, data: mujoco.MjData, name: str) -> None:
        super().__init__(data, name)

    @cached_property
    def body_id(self) -> int:
        return int(self.mj_data.body(self.name).id)


class MlSpacesFreeJointBody(MlSpacesBody):
    def __init__(self, data: mujoco.MjData, body_name: str) -> None:
        super().__init__(data, body_name)
        self._jnt_id = self.mj_model.body_jntadr[self.body_id]
        self._qposadr = self.mj_model.jnt_qposadr[self._jnt_id]
        # TODO(all): this should be a ValueError
        assert (
            self._jnt_id != -1
            and self.mj_model.jnt_type[self._jnt_id] == mujoco.mjtJoint.mjJNT_FREE
        ), f"Body {body_name} does not have a free joint! {self._jnt_id}"

    @property
    def position(self) -> np.ndarray:
        # if this is kept, then the entire base data is also kept, so make a copy, for goodness sake
        return self.mj_data.qpos[self._qposadr : self._qposadr + 3].copy()

    @position.setter
    def position(self, position: np.ndarray) -> None:
        self.mj_data.qpos[self._qposadr : self._qposadr + 3] = position

    @property
    def quat(self) -> np.ndarray:
        return self.mj_data.qpos[self._qposadr + 3 : self._qposadr + 7].copy()

    @quat.setter
    def quat(self, quat: np.ndarray) -> None:
        self.mj_data.qpos[self._qposadr + 3 : self._qposadr + 7] = quat


class MlSpacesMocapBody(MlSpacesBody):
    def __init__(self, data: mujoco.MjData, body_name: str) -> None:
        super().__init__(data, body_name)
        self._mocap_id = int(self.mj_model.body_mocapid[self.body_id])
        assert self._mocap_id != -1, f"Body {body_name} is not a mocap body!"

    @property
    def mocap_id(self) -> int:
        return self._mocap_id

    @property
    def position(self) -> np.ndarray:
        return self.mj_data.mocap_pos[self._mocap_id].copy()

    @position.setter
    def position(self, position: np.ndarray) -> None:
        self.mj_data.mocap_pos[self._mocap_id] = position

    @property
    def quat(self) -> np.ndarray:
        return self.mj_data.mocap_quat[self._mocap_id].copy()

    @quat.setter
    def quat(self, quat: np.ndarray) -> None:
        self.mj_data.mocap_quat[self._mocap_id] = quat


class MlSpacesImmovableBody(MlSpacesBody):
    def __init__(self, data: mujoco.MjData, body_name: str) -> None:
        super().__init__(data, body_name)

    @property
    def position(self) -> np.ndarray:
        return self.mj_data.xpos[self.body_id].copy()

    @position.setter
    def position(self, position: np.ndarray) -> NoReturn:
        raise ValueError(f"Body {self.name} is not movable!")

    @property
    def quat(self) -> np.ndarray:
        return self.mj_data.xquat[self.body_id].copy()

    @quat.setter
    def quat(self, quat: np.ndarray) -> NoReturn:
        raise ValueError(f"Body {self.name} is not movable!")


class MlSpacesCamera(MlSpacesObjectAbstract):
    def __init__(self, data: mujoco.MjData, camera_name: str) -> None:
        super().__init__(data, camera_name)

    @cached_property
    def camera_id(self) -> int:
        return int(self.mj_data.camera(self.name).id)

    def get_focal_length(self, height: int, width: int) -> float:
        """Focal length in pixels"""
        return height / (2 * np.tan(np.radians(self.fovy) / 2))

    def get_fovx(self, height: int, width: int) -> float:
        """Horizontal field of view in degrees"""
        return np.degrees(2 * np.arctan(width / (2 * self.get_focal_length(height, width))))

    def get_fovy(self, height: int, width: int) -> float:
        """Vertical field of view in degrees"""
        return self.fovy

    def get_intrinsic_matrix(self, height: int, width: int) -> np.ndarray:
        """
        3x3 intrinsic matrix, see: https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
        """
        f = self.get_focal_length(height, width)
        return np.array(
            [
                [f, 0, width / 2],
                [0, f, height / 2],
                [0, 0, 1],
            ]
        )

    def get_camera_matrix(self, height: int, width: int) -> np.ndarray:
        """
        3x4 camera matrix, see: https://en.wikipedia.org/wiki/Camera_resectioning#Projection
        """
        return self.get_intrinsic_matrix(height, width) @ inverse_homogeneous_matrix(self.pose)[:3]

    @property
    def fovy(self) -> float:
        return float(self.mj_model.cam_fovy[self.camera_id])

    @fovy.setter
    def fovy(self, fovy: float) -> None:
        self.mj_model.cam_fovy[self.camera_id] = fovy

    @property
    def pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = self.position
        pose[:3, :3] = self.mj_data.cam_xmat[self.camera_id].reshape(3, 3)
        return pose

    @property
    def position(self) -> np.ndarray:
        return self.mj_data.cam_xpos[self.camera_id].copy()

    @position.setter
    def position(self, position: np.ndarray) -> NoReturn:
        raise ValueError(f"Camera {self.name} is not movable!")

    @property
    def quat(self) -> np.ndarray:
        return R.from_matrix(self.mj_data.cam_xmat[self.camera_id].reshape(3, 3)).as_quat(
            scalar_first=True
        )

    @quat.setter
    def quat(self, quat: np.ndarray) -> NoReturn:
        raise ValueError(f"Camera {self.name} is not movable!")


def create_mlspaces_body(data: mujoco.MjData, body_name_or_id: str | int) -> MlSpacesBody:
    if isinstance(body_name_or_id, int):
        body_id = body_name_or_id
        body_name = data.model.body(body_id).name
    else:
        body_name = body_name_or_id
        body_id = data.body(body_name).id

    if data.model.body_mocapid[body_id] != -1:
        return MlSpacesMocapBody(data, body_name)

    elif (
        data.model.body_jntadr[body_id] != -1
        and data.model.jnt_type[data.model.body_jntadr[body_id]] == mujoco.mjtJoint.mjJNT_FREE
    ):
        return MlSpacesFreeJointBody(data, body_name)

    else:
        return MlSpacesImmovableBody(data, body_name)


class MlSpacesObject(MlSpacesBody):
    """
    Class to represent scene object data
    """

    def __init__(self, object_name: str, data: mujoco.MjData) -> None:
        super().__init__(data, object_name)

        # Note: _object_root_id, _geom_ids and _body_ids are no longer needed as backing stores
        # since @cached_property handles caching automatically

        # Local position of object center of mass in world coordinate
        self._center_of_mass_ref = self.mj_model.body_ipos[
            self.object_id
        ].copy()  # inertial frame position

        # Lazy axes-aligned bounding box (initial pose from model)
        self._bvhadr = None
        self._aabb_size: np.ndarray | None = None

    @property
    def bvh_root(self):
        # Address of bounding volume hierarchy (bvh) root
        if self._bvhadr is None:
            bvhadr = self.mj_model.body_bvhadr[self.object_id]
            if bvhadr == -1:
                #  find a <body> with a <geom>
                for b in range(self.mj_model.nbody):
                    if b == self.object_id:
                        continue
                    if self.mj_model.body(b).rootid.item() == self.object_id:
                        bvhadr = self.mj_model.body_bvhadr[b]
                        if bvhadr != -1:
                            break
                else:
                    raise ValueError(f"Object {self.name} has no bvh root!")
            self._bvhadr = int(bvhadr)

        return self._bvhadr

    @property
    def aabb_center(self) -> np.ndarray:
        # axis aligned bounding box center (from model, so initial pose)
        return self.mj_model.bvh_aabb[self.bvh_root][:3].copy()

    @property
    def aabb_size(self) -> np.ndarray:
        # axis aligned bounding box size (from model, so initial pose)
        if self._aabb_size is None:
            aabb_size = self.mj_model.bvh_aabb[self.bvh_root][3:6]
            self._aabb_size = np.ceil(aabb_size / 0.001) * 0.001  # round float to mm

        return self._aabb_size

    @property
    def object_id(self) -> int:
        return self.body_id

    @property
    def center_of_mass(self):
        return self._center_of_mass_ref + self.position

    @property
    def position(self):
        return self.mj_data.xpos[self.body_id].copy()

    @property
    def quat(self):
        return self.mj_data.xquat[self.body_id].copy()

    def is_object_picked_up(self) -> None:
        pass

    def is_object_in_gripper(self) -> None:
        pass

    @cached_property
    def object_root_id(self):
        return int(self.mj_model.body(self.object_id).rootid[0])

    @cached_property
    def geom_ids(self):
        """Get all geom IDs belonging to this object (lazy, cached)."""
        geom_ids = []
        for geom_id in range(0, self.mj_model.ngeom):
            body_id = self.mj_model.geom(geom_id).bodyid.item()
            root_id = self.mj_model.body(body_id).rootid
            if root_id == self.object_root_id:
                geom_ids.append(int(geom_id))
        return geom_ids

    @cached_property
    def body_ids(self):
        """Get all body IDs belonging to this object including descendants (lazy, cached)."""
        from molmo_spaces.utils import mj_model_and_data_utils

        # Use descendant_bodies to get all descendant bodies including the root
        body_ids_set = mj_model_and_data_utils.descendant_bodies(self.mj_model, self.object_id)
        return list(body_ids_set)

    def get_geom_infos(
        self, include_descendants: bool = True, max_geoms: int | None = 2048
    ) -> list[dict[str, object]]:
        """Get geom information for this object.

        Args:
            include_descendants: If True, includes geoms from all descendant bodies.
                If False, only includes geoms directly attached to this object's body.
            max_geoms: Maximum number of geoms to return. If None, returns all.

        Returns:
            List of dicts with geom info: id, name, position, size, type, type_name
        """
        # Build set of body IDs to include
        if include_descendants:
            body_ids = set(self.body_ids)
        else:
            body_ids = {self.object_id}

        # Find geoms belonging to these bodies
        geoms: list[dict[str, object]] = []
        count = 0
        for geom_id in range(self.mj_model.ngeom):
            if max_geoms is not None and count >= max_geoms:
                break
            if int(self.mj_model.geom_bodyid[geom_id]) in body_ids:
                name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
                pos = self.mj_data.geom_xpos[geom_id].copy()
                size = self.mj_model.geom_size[geom_id].copy()
                gtype = int(self.mj_model.geom_type[geom_id])
                geoms.append(
                    {
                        "id": int(geom_id),
                        "name": name,
                        "position": pos,
                        "size": size,
                        "type": gtype,
                        "type_name": MlSpacesObject.get_geom_type_name(gtype),
                    }
                )
                count += 1

        return geoms

    def _set_friction(self, friction: float) -> None:
        # set friction of all geoms in the object
        for geom_id in self.geom_ids:
            self.mj_model.geom(geom_id).friction = friction

    def _set_mass(self, mass: float) -> None:
        # Use body_ids property which uses descendant_bodies() to get all descendant bodies
        body_ids = self.body_ids
        # Calculate total original mass to preserve proportional distribution
        total_original_mass = sum(self.mj_model.body_mass[body_id] for body_id in body_ids)

        if total_original_mass > 0:
            # Scale each body's mass proportionally to maintain original mass ratios
            scale_factor = mass / total_original_mass
            for body_id in body_ids:
                self.mj_model.body_mass[body_id] *= scale_factor
        else:
            # If no original mass, distribute equally
            for body_id in body_ids:
                self.mj_model.body_mass[body_id] = mass / len(body_ids)

    def _set_max_mass(self, max_mass: float) -> None:
        # Use body_ids property which uses descendant_bodies() to get all descendant bodies
        body_ids = self.body_ids
        # Calculate total original mass
        total_original_mass = sum(self.mj_model.body_mass[body_id] for body_id in body_ids)

        # If total mass exceeds max_mass, scale proportionally to preserve mass distribution
        if total_original_mass > max_mass:
            scale_factor = max_mass / total_original_mass
            for body_id in body_ids:
                self.mj_model.body_mass[body_id] *= scale_factor

    def get_friction(self):
        return [self.mj_model.geom(geom_id).friction for geom_id in self.geom_ids]

    def get_mass(self):
        return [self.mj_model.body_mass[body_id] for body_id in self.body_ids]

    @staticmethod
    def body_parent_id(model: mujoco.MjModel, body_id: int) -> int:
        try:
            return int(model.body_parentid[body_id])
        except Exception as e:
            print(f"Error getting body parent id: {e}")
            return int(model.body(body_id).parentid)

    @staticmethod
    def build_children_lists(model: mujoco.MjModel) -> list[list[int]]:
        children: list[list[int]] = [[] for _ in range(model.nbody)]
        for child_id in range(model.nbody):
            pid = MlSpacesObject.body_parent_id(model, child_id)
            if pid >= 0:
                children[pid].append(child_id)
        return children

    @staticmethod
    def get_ancestors(model: mujoco.MjModel, body_id: int) -> list[int]:
        ancestors: list[int] = []
        visited = {body_id}
        pid = MlSpacesObject.body_parent_id(model, body_id)
        while pid >= 0 and pid not in visited:
            ancestors.append(pid)
            visited.add(pid)
            pid = MlSpacesObject.body_parent_id(model, pid)
        return ancestors

    @staticmethod
    def get_descendants(children_lists: list[list[int]], body_id: int) -> list[int]:
        stack = [body_id]
        desc: list[int] = []
        while stack:
            cur = stack.pop()
            for c in children_lists[cur]:
                desc.append(int(c))
                stack.append(c)
        return desc

    @staticmethod
    def get_direct_children(children_lists: list[list[int]], body_id: int) -> list[int]:
        return list(children_lists[body_id])

    @staticmethod
    def is_child_name_of(parent_name: str, child_name: str) -> bool:
        return child_name.startswith(parent_name + "_")

    @staticmethod
    def find_top_object_body_id(model: mujoco.MjModel, body_id: int) -> int:
        cur = body_id
        visited = set()
        while True:
            if cur in visited:
                break
            visited.add(cur)
            pid = MlSpacesObject.body_parent_id(model, cur)
            if pid < 0:
                break
            pname = model.body(pid).name
            cname = model.body(cur).name
            if MlSpacesObject.is_child_name_of(pname, cname):
                cur = pid
            else:
                break
        return int(cur)

    @staticmethod
    def get_geom_type_name(geom_type: int) -> str:
        names = {
            0: "PLANE",
            1: "HFIELD",
            2: "SPHERE",
            3: "CAPSULE",
            4: "ELLIPSOID",
            5: "CYLINDER",
            6: "BOX",
            7: "MESH",
        }
        return names.get(int(geom_type), f"UNKNOWN({geom_type})")

    @staticmethod
    def body_name2id(model: mujoco.MjModel) -> dict[str, int]:
        return {model.body(i).name: i for i in range(model.nbody)}

    @staticmethod
    def get_top_level_bodies(model: mujoco.MjModel) -> list[int]:
        """Return bodies whose parent is the world body."""
        # Identify world body id: the one with parent < 0
        try:
            world_id = next(
                b for b in range(model.nbody) if MlSpacesObject.body_parent_id(model, b) < 0
            )
        except StopIteration:
            world_id = 0
        tops: list[int] = []
        for b in range(model.nbody):
            try:
                pid = MlSpacesObject.body_parent_id(model, b)
                if pid == world_id:
                    tops.append(b)
            except Exception as e:
                print(f"Error getting top-level bodies: {e}")
                continue
        return tops


class MlSpacesArticulationObject(MlSpacesObject):
    def __init__(self, object_name: str, data: mujoco.MjData) -> None:
        super().__init__(object_name, data)

        # gather all joint information of the articulation object
        self.joint_ids = []
        self.joint_names = []
        self.joint_id2name = {}
        self.joint_id2qpos_adr = {}
        self._get_joint_info()

    @property
    def njoints(self) -> int:
        return len(self.joint_ids)

    # getters
    def _get_joint_info(self) -> None:
        """Get joint information for the articulation object."""
        for joint_id in range(self.mj_model.njnt):
            body_id = self.mj_model.joint(joint_id).bodyid[0]
            root_body_id = self.mj_model.body(body_id).rootid[0]
            if root_body_id == self.object_root_id:
                self.joint_ids.append(joint_id)
                self.joint_names.append(self.mj_model.joint(joint_id).name)
                self.joint_id2qpos_adr[joint_id] = self.mj_model.joint(joint_id).qposadr[0]
                self.joint_id2name[joint_id] = self.mj_model.joint(joint_id).name

    def get_joint_position(self, i: int) -> float:
        return self.mj_data.qpos[self.get_joint_qpos_adr(i)].copy()

    def get_joint_range(self, i: int) -> tuple[float, float]:
        return self.mj_model.joint(self.joint_ids[i]).range

    def get_joint_axis(self, i: int) -> np.ndarray:
        return self.mj_model.joint(self.joint_ids[i]).axis

    def get_joint_type(self, i: int) -> mujoco.mjtJoint:
        return self.mj_model.joint(self.joint_ids[i]).type

    def get_joint_qpos_adr(self, i: int) -> int:
        return int(self.mj_model.joint(self.joint_ids[i]).qposadr[0])

    def get_joint_leaf_body_position(self, i: int) -> np.ndarray:
        # position of the joint body not the root center of the object
        body_id = self.mj_model.joint(self.joint_ids[i]).bodyid[0]
        # find child body of this joint body
        for child_body_id in range(self.mj_model.nbody):
            if self.mj_model.body(child_body_id).parentid[0] == body_id:
                body_id = child_body_id
                break
        return self.mj_data.xpos[body_id].copy()

    def get_joint_body_orientation(self, i: int) -> np.ndarray:
        # orientation of the joint body not the root orientation of the object
        body_id = self.mj_model.joint(self.joint_ids[i]).bodyid[0]
        return self.mj_data.xmat[body_id].copy().reshape(3, 3)

    def get_joint_anchor_position(self, i: int) -> np.ndarray:
        # Get the position in world frame of the joint anchor point
        joint_id = self.joint_ids[i]
        body_id = self.mj_model.jnt_bodyid[joint_id]
        local_anchor = self.mj_model.jnt_pos[joint_id]
        body_pos = self.mj_data.xpos[body_id]
        body_quat = self.mj_data.xquat[body_id]
        body_rot = R.from_quat(body_quat, scalar_first=True).as_matrix()
        world_anchor = body_pos + body_rot @ local_anchor
        return world_anchor

    def get_joint_frictionloss(self):
        return [self.mj_model.joint(joint_id).frictionloss for joint_id in self.joint_ids]

    def get_joint_stiffness(self):
        return [self.mj_model.joint(joint_id).stiffness for joint_id in self.joint_ids]

    def get_joint_damping(self):
        return [self.mj_model.joint(joint_id).damping for joint_id in self.joint_ids]

    def get_joint_armature(self):
        return [self.mj_model.joint(joint_id).armature for joint_id in self.joint_ids]

    def set_joint_position(self, i: int, position: float) -> None:
        if isinstance(position, np.ndarray):
            position = position.item()
        self.mj_data.qpos[self.get_joint_qpos_adr(i)] = position

    def _set_joint_frictionloss(self, frictionloss: float) -> None:
        for joint_id in self.joint_ids:
            self.mj_model.joint(joint_id).frictionloss = frictionloss

    def _set_joint_stiffness(self, stiffness: float) -> None:
        for joint_id in self.joint_ids:
            self.mj_model.joint(joint_id).stiffness = stiffness

    def _set_joint_damping(self, damping: float) -> None:
        for joint_id in self.joint_ids:
            self.mj_model.joint(joint_id).damping = damping

    def _set_joint_armature(self, armature: float) -> None:
        for joint_id in self.joint_ids:
            self.mj_model.joint(joint_id).armature = armature

    def _set_joint_reference(self, reference: float) -> None:
        """Set the joint reference position (where the spring pulls to)"""
        for joint_id in self.joint_ids:
            self.mj_model.joint(joint_id).ref = reference

    def _set_joint_springref(self, springref: float) -> None:
        """Set the joint spring reference position"""
        for joint_id in self.joint_ids:
            self.mj_model.joint(joint_id).springref = springref


class Door(MlSpacesArticulationObject):
    def __init__(
        self,
        door_name: str,
        data: mujoco.MjData,
    ) -> None:
        super().__init__(data=data, object_name=door_name)

    @property
    def door_name(self) -> str:
        return self.name

    @property
    def num_handles(self) -> int:
        return len(self._get_handle_leaf_body_ids())

    def handle_name(self, handle_id=0) -> str:
        return self.mj_model.body(self._get_handle_leaf_body_ids()[handle_id]).name

    def set_joint_position(self, i: int, position: float) -> None:
        super().set_joint_position(i, position)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _get_joint_info(self) -> None:
        """Get joint information for the door object.

        Differs from parent by only considering joints that belong to child bodies of the door body."""
        children_lists = MlSpacesObject.build_children_lists(self.mj_model)
        child_body_ids = {
            self.object_id,
            *MlSpacesObject.get_descendants(children_lists, self.object_id),
        }

        for joint_id in range(self.mj_model.njnt):
            body_id = self.mj_model.joint(joint_id).bodyid[0]
            # Only include joints that belong to child bodies of the door body
            if body_id in child_body_ids:
                self.joint_ids.append(joint_id)
                self.joint_names.append(self.mj_model.joint(joint_id).name)
                self.joint_id2qpos_adr[joint_id] = self.mj_model.joint(joint_id).qposadr[0]
                self.joint_id2name[joint_id] = self.mj_model.joint(joint_id).name

    def get_hinge_joint_index(self) -> int:
        """Get the index of the door hinge joint (the joint that does not have 'handle' in its name).

        Returns:
            int: Index of the hinge joint in self.joint_ids list (not the joint ID itself)
        Warns:
            UserWarning: If no joint without 'handle' in its name is found.
        """
        for i, joint_name in enumerate(self.joint_names):
            if "handle" not in joint_name.lower():
                return i

        raise ValueError(
            f"No joint found on door '{self.name}' that does not contain 'handle' in its name. "
            f"Available joints: {self.joint_names}"
        )

    def get_handle_joint_index(self) -> int:
        """Get the index of the handle joint (the joint that has 'handle' in its name).

        Returns:
            int: Index of the handle joint in self.joint_ids list
        """
        for i, joint_name in enumerate(self.joint_names):
            if "handle" in joint_name.lower():
                return i

        raise ValueError(
            f"No joint found on door '{self.name}' that contains 'handle' in its name. "
            f"Available joints: {self.joint_names}"
        )

    def _get_handle_leaf_body_ids(self) -> list[int]:
        """Get leaf body IDs in door hierarchy (assumed to be handles).

        Returns:
            list[int]: List of leaf body IDs (excluding the root door body)
        """
        model = self.mj_model
        # Find leaf bodies in door hierarchy (assume handles are always the leaves)
        children_lists = MlSpacesObject.build_children_lists(model)
        body_ids = [self.object_id]
        body_ids += MlSpacesObject.get_descendants(children_lists, self.object_id)

        # Find bodies with no children (leaf bodies)
        leaf_body_ids = [
            bid for bid in body_ids if len(children_lists[bid]) == 0 and bid != self.object_id
        ]
        return leaf_body_ids

    def get_handle_bboxes_array(self) -> np.ndarray:
        """Get handle bounding boxes as an array.
        Finds leaf bodies in the door hierarchy (assumed to be handles) and
        returns their visual geom AABBs.

        Returns:
            np.ndarray: Array of AABBs (center, size) for handle visual geoms
        """
        model = self.mj_model
        leaf_body_ids = self._get_handle_leaf_body_ids()
        handle_bboxes_array = []
        # Find visual geoms for leaf bodies
        for leaf_body_id in leaf_body_ids:
            leaf_ginds = np.where(model.geom_bodyid == leaf_body_id)[0]
            for leaf_gind in leaf_ginds:
                if model.geom(leaf_gind).contype == 0 and model.geom(leaf_gind).conaffinity == 0:
                    # visual geom (this is usually the target geom of the handle)
                    aabb = model.geom_aabb[leaf_gind]  # axis aligned bounding box (center, size)
                    handle_bboxes_array.append(aabb)

        return np.array(handle_bboxes_array)

    def get_handle_pose(self) -> np.ndarray:
        """Get handle pose (position + quaternion) as a single array.
        Finds the first handle visual geom and returns its pose.
        If multiple handles exist, returns the first one.

        Returns:
            np.ndarray: Handle pose [x, y, z, qw, qx, qy, qz]
        """
        model = self.mj_model
        leaf_body_ids = self._get_handle_leaf_body_ids()

        # Find first visual geom for leaf bodies
        for leaf_body_id in leaf_body_ids:
            leaf_ginds = np.where(model.geom_bodyid == leaf_body_id)[0]
            for leaf_gind in leaf_ginds:
                if model.geom(leaf_gind).contype == 0 and model.geom(leaf_gind).conaffinity == 0:
                    # visual geom - get its pose
                    geom_pos = self.mj_data.geom_xpos[leaf_gind]
                    geom_rot_mat = self.mj_data.geom_xmat[leaf_gind].reshape(3, 3)
                    geom_quat = R.from_matrix(geom_rot_mat).as_quat(scalar_first=True)
                    geom_pose = np.concatenate([geom_pos, geom_quat])
                    return geom_pose

        # No handle found, return zero pose
        warnings.warn(
            f"No handle visual geom found for door '{self.name}'", UserWarning, stacklevel=2
        )
        return np.zeros(7)  # 3 for position + 4 for quaternion

    def get_swing_arc_circle(self) -> dict[str, np.ndarray | float]:
        hinge_joint_idx = self.get_hinge_joint_index()
        center = self.get_joint_anchor_position(hinge_joint_idx)

        # Calculate radius as door extent on Z axis (height)
        # Use the door's AABB size in Z direction
        radius = self.aabb_size[2]

        return {
            "center": center,
            "radius": float(radius),
        }

    def is_point_in_swing_arc(self, point: np.ndarray, safety_margin: float = 0.1) -> bool:
        circle_info = self.get_swing_arc_circle()
        center = circle_info["center"]
        radius = circle_info["radius"] + safety_margin

        # Extract 2D position
        point_2d = point[:2] if len(point) >= 2 else point

        # Check if point is within circle
        dist_from_center = np.linalg.norm(point_2d - center[:2])
        return (dist_from_center <= radius).item()
