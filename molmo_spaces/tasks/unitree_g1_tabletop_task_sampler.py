"""Unitree G1 tabletop task samplers."""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.grasp_sample import has_valid_grasp_file
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d

log = logging.getLogger(__name__)


class UnitreeG1RightArmTabletopPickAndPlaceTaskSampler(PickAndPlaceTaskSampler):
    """Pick-and-place sampler for the fixed G1 real-setup tabletop user scene."""

    def add_auxiliary_objects(self, spec: mujoco.MjSpec) -> None:
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)

    def init_scene(self, env) -> None:
        BaseMujocoTaskSampler.init_scene(self, env)
        self._task_counter = 0
        self._grasp_failure_counts = {}
        self._metadata_adder.add_meta(env.current_scene_metadata)
        self.candidate_objects = self._get_added_pickup_candidates(env)
        if not self.candidate_objects:
            raise HouseInvalidForTask("No dynamically added tabletop pickup objects available")

    def _get_added_pickup_candidates(self, env: CPUMujocoEnv):
        om = env.object_managers[env.current_batch_index]
        candidates = []
        for name in self._added_pickup_names:
            try:
                candidates.append(om.get_object_by_name(name))
            except KeyError:
                log.debug("Skipping unavailable tabletop pickup object %s", name)
        return candidates

    def _restore_pickupables_to_staging(self, env: CPUMujocoEnv) -> None:
        for name, pose7 in self._added_pickup_staging_poses.items():
            body = create_mlspaces_body(env.current_data, name)
            body.pose = pos_quat_to_pose_mat(pose7[:3], pose7[3:7])
            self._zero_free_body_velocity(env, name)

    def _zero_free_body_velocity(self, env: CPUMujocoEnv, body_name: str) -> None:
        model = env.current_model
        data = env.current_data
        body_id = model.body(body_name).id
        joint_id = model.body_jntadr[body_id]
        if joint_id < 0:
            return
        dof_adr = model.jnt_dofadr[joint_id]
        dof_num = 6 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else 1
        data.qvel[dof_adr : dof_adr + dof_num] = 0.0

    def _table_top_z(self, env: CPUMujocoEnv) -> float:
        table_body = env.current_data.body(self.config.task_sampler_config.table_body_name).id
        center, size = body_aabb(env.current_model, env.current_data, table_body, visual_only=False)
        return float(center[2] + size[2] / 2.0)

    def _sample_xy(self, center_xy: tuple[float, float], size_xy: tuple[float, float]) -> np.ndarray:
        center = np.asarray(center_xy, dtype=float)
        half_size = np.asarray(size_xy, dtype=float) / 2.0
        return center + np.random.uniform(-half_size, half_size)

    def _place_pickup_on_table(self, env: CPUMujocoEnv, pickup_obj_name: str) -> None:
        body = create_mlspaces_body(env.current_data, pickup_obj_name)
        model = env.current_model
        data = env.current_data
        mujoco.mj_forward(model, data)
        center, size = body_aabb(model, data, body.body_id, visual_only=False)
        bottom_z = center[2] - size[2] / 2.0
        body_to_bottom = body.position[2] - bottom_z

        xy = self._sample_xy(
            self.config.task_sampler_config.pickup_workspace_center_xy,
            self.config.task_sampler_config.pickup_workspace_size_xy,
        )
        body.position = np.array(
            [
                xy[0],
                xy[1],
                self._table_top_z(env)
                + body_to_bottom
                + self.config.task_sampler_config.object_table_clearance,
            ]
        )
        self._zero_free_body_velocity(env, pickup_obj_name)
        mujoco.mj_forward(model, data)

    def _place_receptacle_on_table(self, env: CPUMujocoEnv) -> None:
        receptacle_name = self.config.task_sampler_config.place_receptacle_name
        body = create_mlspaces_body(env.current_data, receptacle_name)
        xy = self._sample_xy(
            self.config.task_sampler_config.place_workspace_center_xy,
            self.config.task_sampler_config.place_workspace_size_xy,
        )
        body.position = np.array(
            [
                xy[0],
                xy[1],
                self._table_top_z(env) + self.config.task_sampler_config.receptacle_table_clearance,
            ]
        )
        body.quat = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(env.current_model, env.current_data)

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        robot_view = env.current_robot.robot_view
        base_qpos = self.config.robot_config.init_qpos["base"]
        robot_view.base.pose = pos_quat_to_pose_mat(np.asarray(base_qpos[:3]), base_qpos[3:7])
        robot_view.base.joint_vel = np.zeros(robot_view.base.vel_dim)
        env.current_robot.sync_pinned_base_pose()
        self.config.task_config.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

    def _select_pickup_object(self, env: CPUMujocoEnv) -> int:
        if not self._added_pickup_names:
            raise HouseInvalidForTask("No added pickup objects were loaded")

        max_attempts = len(self._added_pickup_names)
        for _ in range(max_attempts):
            object_index = self._task_counter % len(self._added_pickup_names)
            self._task_counter += 1
            pickup_obj_name = self._added_pickup_names[object_index]
            self._added_pickup_obj_name = pickup_obj_name
            self.config.task_config.pickup_obj_name = pickup_obj_name

            self._restore_pickupables_to_staging(env)
            self._place_pickup_on_table(env, pickup_obj_name)
            self._place_receptacle_on_table(env)
            self._sample_and_place_robot(env)

            om = env.object_managers[env.current_batch_index]
            pickup_obj = om.get_object_by_name(pickup_obj_name)
            self.config.task_config.pickup_obj_start_pose = pose_mat_to_7d(
                pickup_obj.pose
            ).tolist()
            pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
            pickup_obj_goal_pose[2] += 0.05
            self.config.task_config.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()
            self.config.task_config.object_poses = {
                pickup_obj_name: pose_mat_to_7d(pickup_obj.pose).tolist()
            }

            self.place_receptacle_name = self.config.task_sampler_config.place_receptacle_name
            receptacle = om.get_object_by_name(self.place_receptacle_name)
            self.config.task_config.place_receptacle_name = self.place_receptacle_name
            self.config.task_config.place_target_name = self.place_receptacle_name
            self.config.task_config.place_receptacle_start_pose = pose_mat_to_7d(
                receptacle.pose
            ).tolist()

            asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
            if asset_uid and not has_valid_grasp_file(asset_uid):
                log.info("Skipping %s because no valid grasp file exists", pickup_obj_name)
                continue

            supporting_geom_id = get_supporting_geom(env.current_data, pickup_obj.body_id)
            if supporting_geom_id is None:
                supporting_geom_id = env.current_model.geom(
                    self.config.task_sampler_config.table_geom_name
                ).id

            self.setup_cameras(env)
            return supporting_geom_id

        raise HouseInvalidForTask("Unable to sample a valid G1 tabletop pickup object")

    def _build_context_objects(self, env, om, pickup_obj_name, supporting_geom_id):
        del supporting_geom_id
        return [
            om.get_object_by_name(pickup_obj_name),
            om.get_object_by_name(self.config.task_sampler_config.place_receptacle_name),
        ]

    def _generate_referral_expressions(self, env, pickup_obj_name: str, context_objects: list):
        if pickup_obj_name == self.config.task_sampler_config.place_receptacle_name:
            priority = [(1.0, 1.0, "box")]
            return priority, priority
        try:
            return super()._generate_referral_expressions(env, pickup_obj_name, context_objects)
        except (AssertionError, KeyError, ValueError):
            om = env.object_managers[env.current_batch_index]
            try:
                expression = om.fallback_expression(pickup_obj_name)
            except (KeyError, ValueError):
                expression = pickup_obj_name.split("/")[-1].replace("_", " ")
            priority = [(1.0, 1.0, expression)]
            return priority, priority

    def _get_place_target_candidates(self, env, pickup_obj_name, supporting_geom_id) -> list[str]:
        del env, pickup_obj_name, supporting_geom_id
        return [self.config.task_sampler_config.place_receptacle_name]

    def _prepare_place_target(
        self,
        env,
        place_target_name,
        pickup_obj_name,
        pickup_obj_pos,
        supporting_geom_id,
    ) -> bool:
        del env, place_target_name, pickup_obj_name, pickup_obj_pos, supporting_geom_id
        return True

    def _filter_place_target(self, env, pickup_obj_name, place_target_name) -> bool:
        del env, pickup_obj_name, place_target_name
        return True

    def _finalize_task_config(self, env: CPUMujocoEnv):
        del env
        added = {}
        if self._added_pickup_obj_name in self.added_objects:
            added[self._added_pickup_obj_name] = self.added_objects[self._added_pickup_obj_name]
        self.config.task_config.added_objects = added

    def _sample_task(self, env: CPUMujocoEnv):
        self._configure_pick_and_place(env)
        return PickAndPlaceTask(env, self.config)
