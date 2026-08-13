import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from molmo_spaces.env.arena.arena_utils import modify_mjmodel_thor_articulated
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.policy.solvers.navigation.fetchman_base_planner_policy import (
    _astar,
    _simplify_path,
)
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.task_sampler import (
    BaseMujocoTaskSampler,
)
from molmo_spaces.tasks.task_sampler_errors import (
    HouseInvalidForTask,
    ObjectPlacementError,
    RobotPlacementError,
)
from molmo_spaces.utils.mujoco_scene_utils import place_object_near
from molmo_spaces.utils.pose import pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_nav_to_obj_config import NavToObjBaseConfig


log = logging.getLogger(__name__)


class RolloutFailure(Exception):
    """Exception for when scene setup fails."""

    pass


def _point_along_path(path_xy: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and unit direction at `frac` (0-1) of the total length along the
    polyline `path_xy` (N, 2), walking segment by segment."""
    seg_vecs = path_xy[1:] - path_xy[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    total_len = float(seg_lens.sum())
    if total_len < 1e-6:
        return path_xy[0].copy(), np.array([1.0, 0.0])

    target_dist = frac * total_len
    cum = 0.0
    for i, seg_len in enumerate(seg_lens):
        if cum + seg_len >= target_dist or i == len(seg_lens) - 1:
            t = 0.0 if seg_len < 1e-9 else float(np.clip((target_dist - cum) / seg_len, 0.0, 1.0))
            point = path_xy[i] + t * seg_vecs[i]
            direction = seg_vecs[i] / seg_len if seg_len > 1e-9 else np.array([1.0, 0.0])
            return point, direction
        cum += seg_len
    return path_xy[-1].copy(), np.array([1.0, 0.0])


class NavToObjTaskSampler(BaseMujocoTaskSampler):
    """
    Default task sampler for RBY1 navigation to object tasks with house iteration control.
    House order (`house_inds`) and samples per house are provided via config.
    """

    def __init__(self, config: "NavToObjBaseConfig") -> None:
        super().__init__(config)
        self.candidate_objects: None | list[MlSpacesObject] = None
        self._task_counter = None  # Track tasks within the same house for variety
        self._cached_thormap = None  # Cache occupancy map per house

        # If pickup_types is None, default to empty list which matches any object type.
        # Objects are then filtered by navigability and visibility in _get_scene_objects().
        if config.task_sampler_config.pickup_types is None:
            config.task_sampler_config.pickup_types = []

    def init_scene(self, env) -> None:
        # Initialize base randomizers (texture, lighting, dynamics)
        super().init_scene(env)

        log.info(
            f"Setting up scene for house {self.current_house_index}, task {self._task_counter}..."
        )
        model = env.mj_model
        data = env.mj_datas[0]
        modify_mjmodel_thor_articulated(model, data)

        # New house - reset counters
        self._task_counter = 0
        log.debug(f"New house {self.current_house_index} - resetting object tracking")

        # Generate occupancy map ONCE per house (for A* planner)
        import gc

        from molmo_spaces.utils.scene_maps import ProcTHORMap

        if self._cached_thormap is not None:
            del self._cached_thormap
            gc.collect()

        log.info(f"Generating occupancy map for house {self.current_house_index}")
        self._cached_thormap = ProcTHORMap.from_mj_model_path(
            model_path=env.current_model_path,
            agent_radius=self.config.task_sampler_config.robot_safety_radius,
            px_per_m=200,
            device_id=None,
        )
        log.info("Occupancy map generated successfully")

        candidate_objects = self._get_scene_objects(env)
        candidate_objects = self.balance_sample_names(candidate_objects)
        # Shuffle order deterministically per house/task for variety
        np.random.shuffle(candidate_objects)
        self.candidate_objects = candidate_objects

        np.random.shuffle(self.candidate_objects)

    def randomize_scene(self, env: CPUMujocoEnv, robot_view) -> None:
        """Setup scene state: robot joints, texture randomization, cameras."""
        # randomize scene here
        super().randomize_scene(env, robot_view)

        model = env.current_model
        data = env.current_data
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Set robot joints
        for group_name, qpos in self.config.robot_config.init_qpos.items():
            qpos = np.array(qpos)
            noise_range = self.config.robot_config.init_qpos_noise_range
            if noise_range is not None and group_name in noise_range:
                noise_mag = np.array(noise_range[group_name])
                perturb = np.random.uniform(-noise_mag, noise_mag)
            else:
                perturb = np.zeros_like(qpos)
            robot_view.get_move_group(group_name).joint_pos = qpos + perturb

        # Reset controllers to hold current positions (important for torso/head)
        for robot in env.robots:
            for controller in robot.controllers.values():
                controller.reset()

        log.info("Scene setup completed.\n")

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> str | None:
        """Resolve special visibility object keys.

        Handles:
        - __pickup_object__: Current pickup/nav object (returns first candidate instance)
        """
        if key == "__pickup_object__":
            # Return the first candidate object instance for visibility checking
            if (
                hasattr(self.config.task_config, "pickup_obj_candidates")
                and self.config.task_config.pickup_obj_candidates
                and len(self.config.task_config.pickup_obj_candidates) > 0
            ):
                return self.config.task_config.pickup_obj_candidates[0]
            elif (
                hasattr(self.config.task_config, "pickup_obj_name")
                and self.config.task_config.pickup_obj_name
            ):
                return self.config.task_config.pickup_obj_name
            return None

        # Delegate to base class for other keys (e.g., __gripper__)
        return super().resolve_visibility_object(env, key)

    def _sample_task(self, env: CPUMujocoEnv) -> NavToObjTask:
        """Sample a navigation to object task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        # Get ObjectManager for type extraction
        om = env.object_managers[env.current_batch_index]

        keep_task_cfg = self.config.task_config.pickup_obj_name is not None

        excluded_types = set()
        num_robot_placement_failures = 0

        unique_objects = set(self.candidate_objects)

        sample_success = False
        num_attempts_left = len(self.candidate_objects)
        while not sample_success and num_attempts_left > 0:
            num_attempts_left -= 1

            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_select_object")

            if not keep_task_cfg:
                self.config.task_config.pickup_obj_name = None

            # Sample nav object
            if self.config.task_config.pickup_obj_name is None:
                object_index = self._task_counter % len(self.candidate_objects)
                selected_obj = self.candidate_objects[object_index]

                # Extract object type (e.g., "lettuce" from "lettuce_58a2d909...")
                pickup_obj_type = om.fallback_expression(selected_obj.name)

                if pickup_obj_type in excluded_types:
                    continue

                # Get synset for semantic information
                synset = om.get_annotation_synset(selected_obj.name)

                # Collect all objects of this type in the scene
                same_type_candidates = [
                    obj.name
                    for obj in unique_objects
                    if om.fallback_expression(obj.name) == pickup_obj_type
                ]

                if len(same_type_candidates) > self.config.task_sampler_config.max_valid_candidates:
                    log.info(
                        f"Skipping {pickup_obj_type} with {len(same_type_candidates)} instances in scene."
                    )
                    excluded_types.add(pickup_obj_type)
                    continue

                # Set the instance name and store all candidates
                self.config.task_config.pickup_obj_name = selected_obj.name
                self.config.task_config.pickup_obj_candidates = same_type_candidates

                # Store semantic category for eval reconstruction
                self.config.task_config.pickup_obj_category = pickup_obj_type
                self.config.task_config.pickup_obj_synset = synset

                log.info(
                    f"[OK] Attempting object type '{pickup_obj_type}' (synset: {synset}) with {len(same_type_candidates)} instances: {same_type_candidates}"
                )
                log.info(f"Selected initial instance for robot placement: {selected_obj.name}")
            else:
                # If pickup_obj_name is pre-specified, it might be a type or specific instance
                # Try to interpret it as a type and collect candidates
                if self.config.task_config.pickup_obj_candidates is None:
                    pickup_obj_type = om.category_from_name(self.config.task_config.pickup_obj_name)
                    synset = om.get_annotation_synset(self.config.task_config.pickup_obj_name)
                    same_type_candidates = [
                        obj.name
                        for obj in self.candidate_objects
                        if om.fallback_expression(obj.name) == pickup_obj_type
                    ]

                    if len(same_type_candidates) == 0:
                        # Might be a specific instance name, use it as-is
                        same_type_candidates = [self.config.task_config.pickup_obj_name]
                        log.info(
                            f"[OK] Using pre-specified object instance: {self.config.task_config.pickup_obj_name}"
                        )
                    else:
                        self.config.task_config.pickup_obj_candidates = same_type_candidates
                        log.info(
                            f"[OK] Using pre-specified object type '{pickup_obj_type}' (synset: {synset}) with {len(same_type_candidates)} instances"
                        )

                    # Store semantic category
                    self.config.task_config.pickup_obj_category = pickup_obj_type
                    self.config.task_config.pickup_obj_synset = synset
                else:
                    log.info("[OK] Using pre-configured pickup_obj_name and pickup_obj_candidates")
                    # If not set, try to infer from existing pickup_obj_name
                    if self.config.task_config.pickup_obj_category is None:
                        self.config.task_config.pickup_obj_category = om.category_from_name(
                            self.config.task_config.pickup_obj_name
                        )
                        self.config.task_config.pickup_obj_synset = om.get_annotation_synset(
                            self.config.task_config.pickup_obj_name
                        )

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_select_object")

            self._task_counter += 1  # update counter, so we don't re-try same object

            try:
                self._sample_and_place_robot(env)
            except RobotPlacementError:
                log.exception("Caught when attempting to place robot. Retrying")
                num_robot_placement_failures += 1
                continue

            if self.config.task_sampler_config.sample_trajectory_obstacles:
                om = env.object_managers[env.current_batch_index]
                goal_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
                robot_xy = env.current_robot.robot_view.base.pose[:2, 3]
                # Exclude every instance in pickup_obj_candidates, not just the
                # selected pickup_obj_name -- NavToObjTask.get_nearest_nav_object
                # treats ALL of them as valid targets (e.g. "any painting"), so
                # relocating an alternate instance would still be moving a goal.
                exclude_names = set(self.config.task_config.pickup_obj_candidates or [])
                exclude_names.add(goal_obj.name)
                exclude_names.add(self.config.task_config.start_obj_name)
                self._sample_trajectory_obstacles(
                    env,
                    start_xy=robot_xy,
                    goal_xy=goal_obj.position[:2],
                    exclude_names=exclude_names,
                )

            # Ensure robot is in final position before camera setup
            mujoco.mj_forward(env.current_model, env.current_data)

            # Setup cameras after navigation object and robot placement
            # This allows cameras to use task-specific info (navigation object)
            self.setup_cameras(env)

            sample_success = True
            break

        if not sample_success:
            # HouseInvalidForTask builds its message from `reason`/`house_info`, not a
            # plain positional string -- passing one positionally (as this used to)
            # silently falls through to its generic default message, hiding exactly
            # the diagnosis this is trying to report.
            reason_parts = [
                f"tried all {len(self.candidate_objects)} candidate objects, "
                "none produced a valid task"
            ]
            if excluded_types:
                reason_parts.append(
                    f"{len(excluded_types)} type(s) excluded for exceeding "
                    f"max_valid_candidates={self.config.task_sampler_config.max_valid_candidates}: "
                    f"{sorted(excluded_types)}"
                )
            if num_robot_placement_failures:
                reason_parts.append(
                    f"{num_robot_placement_failures} robot-placement attempt(s) failed "
                    "(see preceding 'Caught when attempting to place robot' logs)"
                )
            raise HouseInvalidForTask(reason="; ".join(reason_parts))

        # Here we just copy the ObjNavTask target name for completeness, even if unused
        pickup_obj_name = self.config.task_config.pickup_obj_name

        # Get natural name if available
        try:
            object_name = om.fallback_expression(pickup_obj_name)
        except Exception:
            # Fallback to raw name if natural name lookup fails
            object_name = pickup_obj_name.replace("_", " ").title()

        self.config.task_config.referral_expressions["object_name"] = object_name
        self.config.task_config.referral_expressions_priority["object_name"] = [
            (1.0, 1.0, object_name)
        ]

        task: NavToObjTask = NavToObjTask(env, exp_config=self.config)
        # Store occupancy map reference in task for policy access
        task.occupancy_map = self._cached_thormap
        return task

    def _get_scene_objects(self, env: CPUMujocoEnv) -> list[MlSpacesObject]:
        """
        Get the list of candidate probjects in the scene for interactions.
        Filter by object types and prefer objects on the floor (not on furniture).
        """
        # Discover candidate nav_to objects
        om = env.object_managers[env.current_batch_index]
        candidates = om.get_objects_of_type(self.config.task_sampler_config.pickup_types)
        log.info(f"Found {len(candidates)} candidate nav objects in the scene")

        if not len(candidates) > 0:
            log.info("[WARN] No candidate nav objects found in the scene")
            # print all the top-level objects in the scene for debugging
            om = env.object_managers[env.current_batch_index]
            all_objects = MlSpacesObject.get_top_level_bodies(model=self.env.mj_model)
            for b in all_objects[:30]:
                name = self.env.mj_model.body(b).name
                pos = self.env.current_data.xpos[b]
                possible_types = om.get_possible_object_types(b)
                log.info(
                    f"  - #{b:02d} {name} (types={possible_types}) pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

            # log.info(f"Scene objects (no candidates): {[obj.name for obj in all_objects]}")
            raise HouseInvalidForTask("No nav candidates found in the scene")

        return candidates

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Sample a nav object, place robot using occupancy map, and return sampled params.

        Returns:
            dict with keys: pickup_obj_name, robot_base_pose
        Raise:
            RobotPlacementError if robot placement fails
        """
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        robot_view = env.current_robot.robot_view
        log.debug(f"Selected pickup object: {task_cfg.pickup_obj_name}")
        log.debug(f"[TASK SAMPLING] Trying to place robot near '{pickup_obj.name}'")

        # randomize pickup object texture
        if (
            self.texture_randomizer is not None
            and self.config.task_sampler_config.randomize_textures
        ):
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("robot_randomize_pickup_obj")
            self.texture_randomizer.randomize_object(pickup_obj)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("robot_randomize_pickup_obj")

        if isinstance(pickup_obj, MlSpacesObject):
            pickup_obj_pos = pickup_obj.position
        else:
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        # Base-motion evaluation: with probability start_near_object_probability, place
        # the robot near a different scene object instead of the nav target itself.
        place_target, place_target_pos = pickup_obj, pickup_obj_pos
        start_near_object_probability = (
            self.config.task_sampler_config.start_near_object_probability
        )
        if (
            start_near_object_probability > 0
            and np.random.uniform(0, 100) < start_near_object_probability
        ):
            start_obj = self._sample_start_object_near(env, pickup_obj)
            if start_obj is not None:
                place_target, place_target_pos = start_obj, start_obj.position
                task_cfg.start_obj_name = start_obj.name
                log.info(
                    f"[BaseMotion] Placing robot near start object '{start_obj.name}' "
                    f"(nav target remains '{pickup_obj.name}')"
                )
            else:
                log.info(
                    f"[BaseMotion] No valid start object found within "
                    f"[{self.config.task_sampler_config.min_start_object_dist}, "
                    f"{self.config.task_sampler_config.max_start_object_dist}]m of "
                    f"'{pickup_obj.name}'; falling back to placing near the nav target"
                )

        # Check if robot_base_pose is already set (e.g., from frozen_config)
        if task_cfg.robot_base_pose is not None:
            # Restore robot to saved pose instead of sampling
            from molmo_spaces.utils.pose import pos_quat_to_pose_mat

            log.info(f"Restoring robot from frozen_config: {task_cfg.robot_base_pose}")

            saved_pose = np.array(task_cfg.robot_base_pose)
            robot_view.base.pose = pos_quat_to_pose_mat(saved_pose[:3], saved_pose[3:])

            final_pos = robot_view.base.pose[:3, 3]
            distance_to_obj = np.linalg.norm(final_pos[:2] - pickup_obj_pos[:2])
            log.info("[OK] Robot restored from config")
            log.info(
                f"Final robot position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
            )
            log.info(f"Distance to object: {distance_to_obj:.3f}m")
        else:
            # Sample a new robot position
            # Log placement parameters
            sampling_radius_range = (
                self.config.task_sampler_config.start_object_sampling_radius_range
                if place_target is not pickup_obj
                else self.config.task_sampler_config.base_pose_sampling_radius_range
            )
            robot_safety_radius = self.config.task_sampler_config.robot_safety_radius
            # Robots whose base height is held constant by their own controller
            # regardless of placement (e.g. G1WalkController's WBC) can't actually
            # spawn at an offset-derived height -- see BaseRobotConfig.fixed_base_height
            # and the identical fix in PickTaskSampler._sample_and_place_robot.
            # Without this, G1 was being placed with its pelvis at
            # robot_object_z_offset (0.1m default) instead of ~0.79m, so the pelvis
            # immediately collided with the floor at every sampled point.
            fixed_base_height = self.config.robot_config.fixed_base_height
            if fixed_base_height is not None:
                initial_robot_z = fixed_base_height
            else:
                initial_robot_z = self.config.task_sampler_config.robot_object_z_offset
            max_robot_placement_attempts = (
                self.config.task_sampler_config.max_robot_placement_attempts
            )
            face_target = self.config.task_sampler_config.face_target

            log.info(
                f"Attempting to place robot near '{place_target.name}' in radius range {sampling_radius_range[0]:.3f}m - {sampling_radius_range[1]:.3f}m"
            )

            # place robot near the placement target (the nav target, or a distinct
            # start object if start_near_object_probability fired above)
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("robot_place_near_pickup_obj")

            robot_placed = env.place_robot_near(
                robot_view=robot_view,
                target=place_target,
                max_tries=max_robot_placement_attempts,
                sampling_radius_range=sampling_radius_range,
                robot_safety_radius=robot_safety_radius,
                preserve_z=initial_robot_z,
                face_target=face_target,
                # check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
                # visibility_resolver=self.get_visibility_resolver(env),
                # excluded_positions=self.used_robot_positions[pickup_obj.name],
            )
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("robot_place_near_pickup_obj")

            if not robot_placed:
                log.info(f"[FAIL] Failed to place robot near '{place_target.name}'")
                raise RobotPlacementError(f"Failed to place robot near object: {place_target.name}")

            # Get final robot pose for return data
            task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()
            final_pos = robot_view.base.pose[:3, 3]
            log.info("[OK] Successfully placed robot")
            log.info(
                f"Final robot position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
            )
            log.info(
                f"Object position: ({place_target_pos[0]:.3f}, {place_target_pos[1]:.3f}, {place_target_pos[2]:.3f})"
            )

    def _sample_start_object_near(
        self, env: CPUMujocoEnv, goal_obj: MlSpacesObject
    ) -> MlSpacesObject | None:
        """Pick a random scene object within [min_start_object_dist, max_start_object_dist]
        of `goal_obj`, for base-motion evaluation (start the robot near a different
        object than the nav target). Uses plain XY position distance, matching
        NavToObjTask.calculate_distance's own reward metric. Returns None if no
        candidate qualifies.
        """
        task_sampler_config = self.config.task_sampler_config
        min_dist = task_sampler_config.min_start_object_dist
        max_dist = task_sampler_config.max_start_object_dist

        candidates = [
            obj
            for obj in self.candidate_objects
            if obj.name != goal_obj.name
            and min_dist <= np.linalg.norm(obj.position[:2] - goal_obj.position[:2]) <= max_dist
        ]
        if not candidates:
            return None

        return candidates[np.random.randint(len(candidates))]

    def _sample_trajectory_obstacles(
        self,
        env: CPUMujocoEnv,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        exclude_names: set[str],
    ) -> None:
        """Scatter num_trajectory_obstacles random scene objects along the robot's
        actual walkable start->goal path (A*, not a straight line -- see below) to
        force navigation detours. Individual placement failures are skipped, not
        fatal to the task sample.
        """
        task_sampler_config = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]
        data = env.current_data

        if np.linalg.norm(goal_xy - start_xy) < 1e-6:
            return

        # Interpolating along the straight line between start and goal (the
        # original approach) breaks down in any multi-room house: that line
        # commonly crosses a wall between rooms, so every point sampled near the
        # crossing gets rejected by the occupancy check below, leaving a dead
        # zone in the middle of the frac range and bunching successful
        # placements toward whichever end has more open floor -- confirmed this
        # is exactly what was producing the reported "obstacles bunched at one
        # end" pattern. Use the same A* implementation FetchManBasePlannerPolicy
        # already uses to actually walk this path (ported from g1_molmo, reused
        # here rather than duplicated) so obstacles are scattered along the real
        # walkable route instead.
        path_xy = None
        if self._cached_thormap is not None:
            start_rc = self._cached_thormap.pos_m_to_px(np.array([*start_xy, 0.0]))
            goal_rc = self._cached_thormap.pos_m_to_px(np.array([*goal_xy, 0.0]))
            path_rc = _astar(self._cached_thormap.occupancy, start_rc, goal_rc)
            if len(path_rc) >= 2:
                path_rc = _simplify_path(path_rc, self._cached_thormap.occupancy)
                path_xy = self._cached_thormap.pos_px_to_m(np.array(path_rc, dtype=np.float64))[
                    :, :2
                ]
        if path_xy is None or len(path_xy) < 2:
            log.info(
                "[Trajectory obstacles] A* path unavailable/degenerate, "
                "falling back to a straight line between start and goal"
            )
            path_xy = np.array([start_xy, goal_xy])

        # candidate_objects are valid *nav targets* (walkable-to), which doesn't imply
        # they're movable -- e.g. a bed has no free joint and can't be relocated.
        # place_object_near requires a free-jointed body, so filter to those upfront
        # rather than discovering the mismatch via a ValueError per attempt.
        movable_objects = [obj for obj in self.candidate_objects if om.has_free_joint(obj.name)]

        used_names = set(exclude_names)
        placed = 0
        max_point_attempts_per_obstacle = 20
        for _ in range(task_sampler_config.num_trajectory_obstacles):
            candidates = [obj for obj in movable_objects if obj.name not in used_names]
            if not candidates:
                log.info("[Trajectory obstacles] No more candidate objects to place")
                break

            # Find a collision-free point on its own retry budget, separate from the
            # outer per-obstacle loop above. Previously the occupancy check's
            # `continue` consumed one of the num_trajectory_obstacles iterations
            # per miss, so a cluttered scene (frequent wall/furniture hits) could
            # exhaust the entire obstacle count on collision misses alone, well
            # before ever attempting most of the requested obstacles.
            point_xy = None
            for _ in range(max_point_attempts_per_obstacle):
                frac = np.random.uniform(
                    task_sampler_config.trajectory_obstacle_min_frac,
                    task_sampler_config.trajectory_obstacle_max_frac,
                )
                jitter = np.random.uniform(
                    -task_sampler_config.trajectory_obstacle_lateral_jitter,
                    task_sampler_config.trajectory_obstacle_lateral_jitter,
                )
                path_point, path_direction = _point_along_path(path_xy, frac)
                perpendicular = np.array([-path_direction[1], path_direction[0]])
                candidate_point_xy = path_point + jitter * perpendicular

                if self._cached_thormap is not None:
                    point_xyz = np.array([candidate_point_xy[0], candidate_point_xy[1], 0.0])
                    if bool(self._cached_thormap.check_collision(point_xyz)):
                        continue  # unnavigable cell (wall/etc) -- retry within this obstacle's budget

                point_xy = candidate_point_xy
                break

            if point_xy is None:
                log.info(
                    "[Trajectory obstacles] Could not find a free point after "
                    f"{max_point_attempts_per_obstacle} attempts, skipping this obstacle"
                )
                continue

            obstacle = candidates[np.random.randint(len(candidates))]
            used_names.add(obstacle.name)

            # place_object_near interprets placement_point[2] as the target BASE
            # (bottom) height. The obstacle's own resting height isn't a valid
            # reference for the new XY -- an object normally resting on a
            # counter/shelf (e.g. an apple at z=1.0m) would float in midair at a
            # floor-level path point, with nothing underneath it. Instead, always
            # target a small height above the floor and let gravity settle it
            # once the episode steps, regardless of where the object originally
            # rested -- lets any movable object (not just already-floor-level
            # ones) be scattered as an obstacle.
            floor_z = 0.0  # ProcTHOR scenes are single-story with floor at world z=0
            base_z = floor_z + task_sampler_config.trajectory_obstacle_drop_height
            point_3d = np.array([point_xy[0], point_xy[1], base_z])
            try:
                place_object_near(
                    data=data,
                    object_id=obstacle.body_id,
                    placement_point=point_3d,
                    min_dist=0.0,
                    max_dist=task_sampler_config.trajectory_obstacle_placement_radius,
                )
                placed += 1
                final_pos = obstacle.position
                log.info(
                    f"[Trajectory obstacles] Placed '{obstacle.name}' at "
                    f"({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
                )
            except (ObjectPlacementError, ValueError) as e:
                # Best-effort scattering -- a single obstacle failing (occupancy,
                # geometry edge cases, etc.) should never fail the whole task sample.
                log.info(f"[Trajectory obstacles] Failed to place '{obstacle.name}': {e}, skipping")

        log.info(
            f"[Trajectory obstacles] Placed {placed}/{task_sampler_config.num_trajectory_obstacles} "
            f"obstacles between start and goal"
        )
