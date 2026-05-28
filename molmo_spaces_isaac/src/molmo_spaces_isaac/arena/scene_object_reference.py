"""Arena object wrapper for a rigid body already present inside a MolmoSpaces scene USD."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.utils.pose import Pose


class SceneRigidObjectReference(ObjectBase):
    """Track and reset a scene-embedded rigid object as an Arena pickup object.

    MolmoSpaces benchmark pick tasks often target existing scene objects instead of
    objects listed under ``scene_modifications.added_objects``. The iTHOR USD scene is
    spawned as ``{ENV_REGEX_NS}/molmospaces_scene``; this wrapper points Isaac Lab's
    ``RigidObject`` at the matching child prim so Arena terms can read and move it.
    """

    def __init__(
        self,
        *,
        name: str,
        scene_object_name: str,
        body_prim_suffix: str | None = None,
        initial_pose: Pose | None = None,
    ):
        scene_object_root = f"{{ENV_REGEX_NS}}/molmospaces_scene/Geometry/{scene_object_name}"
        if body_prim_suffix:
            prim_path = f"{scene_object_root}/{body_prim_suffix.strip('/')}"
        else:
            prim_path = scene_object_root
        super().__init__(name=name, prim_path=prim_path, object_type=ObjectType.RIGID)
        self.scene_object_name = scene_object_name
        self.scene_object_root_prim_path = scene_object_root
        self.body_prim_suffix = body_prim_suffix
        self.initial_pose = initial_pose
        self.object_cfg = self._init_object_cfg()
        self.event_cfg = self._init_event_cfg()

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        object_cfg = RigidObjectCfg(prim_path=self.prim_path, spawn=None)
        initial_pose = self._get_initial_pose_as_pose()
        if initial_pose is not None:
            object_cfg.init_state.pos = initial_pose.position_xyz
            object_cfg.init_state.rot = initial_pose.rotation_wxyz
        return object_cfg

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        raise TypeError("SceneRigidObjectReference only supports rigid objects.")

    def _generate_base_cfg(self) -> AssetBaseCfg:
        raise TypeError("SceneRigidObjectReference only supports rigid objects.")
