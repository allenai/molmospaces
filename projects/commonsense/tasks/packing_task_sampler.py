"""CommonSense packing datagen: the core PackingTaskSampler, tuned for the
bench_pack_{covering,occlusion} benchmarks.

Places the robot within reach of both the pickup object and the box, clutters the
scene around the pickup object, and packs every placed clutter object (not just the
pickup object) -- PackingTask.get_info() scores all of task_config.packing_object_names.
"""

import logging

from molmo_spaces.configs.task_sampler_configs import PackingTaskSamplerConfig
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.packing_task import PackingTask
from molmo_spaces.tasks.packing_task_sampler import PackingTaskSampler
from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class CommonSensePackingTaskSamplerConfig(PackingTaskSamplerConfig):
    base_pose_sampling_radius_range: tuple[float, float] = (0.0, 0.7)
    min_object_to_receptacle_dist: float = 0.30  # 15cm further than default (0.15)
    robot_object_z_offset: float = -0.58
    robot_object_z_offset_random_min: float = -0.1
    robot_object_z_offset_random_max: float = 0.1
    # Deliberately the shallowest box (0.061m walls): taller boxes put the place pose
    # out of comfortable IK range and hide the contents from the exo cameras.
    box_uids: list[str] | None = ["Box_11"]


class CommonSensePackingTaskSampler(PackingTaskSampler):
    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Place robot within reach of both pickup object and box receptacle."""
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        log.debug(f"Selected pickup object: {task_cfg.pickup_obj_name}")

        # Randomize pickup object texture
        if (
            self.texture_randomizer is not None
            and self.config.task_sampler_config.randomize_textures
        ):
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("robot_randomize_pickup_obj")
            self.texture_randomizer.randomize_object(pickup_obj)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("robot_randomize_pickup_obj")

        robot_view = env.current_robot.robot_view
        if not isinstance(pickup_obj, MlSpacesObject):
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")
        target_pos = pickup_obj.position

        initial_robot_z = (
            target_pos[2]
            + self.config.task_sampler_config.robot_object_z_offset
            + np.random.uniform(
                self.config.task_sampler_config.robot_object_z_offset_random_min,
                self.config.task_sampler_config.robot_object_z_offset_random_max,
            )
        )

        # Get box receptacle object
        box_obj = om.get_object_by_name(self.place_receptacle_name)
        log.info(f"[PACKING] Placing robot near both '{pickup_obj.name}' and box '{box_obj.name}'")

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("robot_place_near")
        robot_placed = env.place_robot_near_multiple(
            robot_view=robot_view,
            targets=[pickup_obj, box_obj],
            max_tries=10,
            sampling_radius_range=self.config.task_sampler_config.base_pose_sampling_radius_range,
            robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
            preserve_z=initial_robot_z,
            face_target=True,
            check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
            visibility_resolver=self.get_visibility_resolver(env),
            excluded_positions=self.used_robot_positions[pickup_obj.name],
            save_visibility_frames_dir=self.config.output_dir,
        )
        if self._datagen_profiler is not None:
            self._datagen_profiler.end("robot_place_near")

        if not robot_placed:
            log.info(
                f"[PACKING] Failed to place robot near '{pickup_obj.name}' and box '{box_obj.name}'"
            )
            raise RobotPlacementError(
                f"Failed to place robot near object '{pickup_obj.name}' and box '{box_obj.name}'"
            )

        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05  # 5 cm
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

        log.info(f"Supporting receptacle: {task_cfg.receptacle_name}")

    def _sample_task(self, env: CPUMujocoEnv) -> PackingTask:
        """Core packing sample (placement + open flaps), then clutter the scene."""
        _ = super()._sample_task(env)

        # Clutter scene with additional objects around the pickup object (same as pick task)
        self._clutter_scene_around_pickup_object(env)

        # Pass the list of objects to pack (clutter + pickup) to the task config
        self.config.task_config.packing_object_names = self._placed_clutter_object_names

        # Apply robot qpos after all placement (PickAndPlaceTaskSampler doesn't apply
        # _init_robot_qpos unlike PickTaskSampler, so we do it here)
        robot_view = env.current_robot.robot_view
        for group_name, joint_pos in self._init_robot_qpos.items():
            robot_view.get_move_group(group_name).joint_pos = joint_pos

        # Banish clutter penetrating the robot at its home pose, then re-sync the
        # packing target list (the helper rebinds _placed_clutter_object_names, and
        # banished objects must not count toward the packing success condition).
        self._resolve_robot_clutter_penetrations(env)
        self.config.task_config.packing_object_names = self._placed_clutter_object_names

        return PackingTask(env, self.config)
