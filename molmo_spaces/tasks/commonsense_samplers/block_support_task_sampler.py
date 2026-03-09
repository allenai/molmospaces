import logging
from typing import TYPE_CHECKING

import numpy as np
from mujoco import MjSpec, mjtGeom

from molmo_spaces.env.data_views import MjThorObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.commonsense_tasks.block_support_task import BlockSupportTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.object_constants import THOR_PICKUP_OBJECTS_LOWERCASE
from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos
from molmo_spaces.utils.mujoco_scene_utils import (
    get_supporting_geom,
    place_object_near,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

MAX_BLOCKS = 4
BLOCK_COLORS = [
    [1, 0, 0, 1],  # red
    [0, 0, 1, 1],  # blue
    [0, 1, 0, 1],  # green
    [1, 1, 0, 1],  # yellow
]


def block_name(i: int) -> str:
    """Return the body name for block index i (1-based)."""
    return f"block_{i}"


class BlockSupportTaskSampler(PickTaskSampler):
    """
    Task sampler for block support tasks that extends PickTaskSampler.
    Adds 2-4 colored cubes to the scene and places them near random graspable objects.
    """

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Add all possible blocks (up to MAX_BLOCKS) at spec time.
        Unused blocks will be moved off-scene during task sampling.
        """
        for i in range(1, MAX_BLOCKS + 1):
            self._add_support_cube(spec, name=block_name(i), color=BLOCK_COLORS[i - 1])

        # Call parent class to add any additional auxiliary objects from policy
        super().add_auxiliary_objects(spec)

    def _get_scene_objects(self, env: CPUMujocoEnv) -> list[MjThorObject]:
        """Override to return only the active blocks as candidate objects."""
        om = env.object_managers[env.current_batch_index]
        task_sampler_config = self.config.task_sampler_config
        _, num_blocks_max = task_sampler_config.num_blocks_range
        # Return all possible blocks; _sample_task will decide how many to use
        cubes = []
        for i in range(1, num_blocks_max + 1):
            name = block_name(i)
            try:
                cube_obj = om.get_object_by_name(name)
                log.info(f"Found block '{name}' as candidate pickup object")
                cubes.append(cube_obj)
            except Exception as e:
                log.error(f"Could not find block '{name}': {e}")
                raise
        return cubes

    def _sample_task(self, env: CPUMujocoEnv) -> BlockSupportTask:
        """Place support cubes near a random graspable object, then place robot near all blocks."""
        import mujoco

        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        om = env.object_managers[env.current_batch_index]
        task_sampler_config = self.config.task_sampler_config

        # Sample number of blocks for this episode
        num_blocks_min, num_blocks_max = task_sampler_config.num_blocks_range
        num_blocks = np.random.randint(num_blocks_min, num_blocks_max + 1)
        active_block_names = [block_name(i) for i in range(1, num_blocks + 1)]
        log.info(f"[BLOCK SUPPORT] Using {num_blocks} blocks: {active_block_names}")

        # Store block names in task config
        self.config.task_config.block_names = active_block_names

        # Pick a random graspable object as a reference point for block placement
        all_graspable_objects = om.get_objects_of_type(THOR_PICKUP_OBJECTS_LOWERCASE)
        tmp_pickup_obj = np.random.choice(all_graspable_objects)
        self.config.task_config.pickup_obj_name = tmp_pickup_obj.name

        # Place active blocks near the reference object
        self._place_cubes_near_reference(env, tmp_pickup_obj, active_block_names)

        # Move unused blocks far away
        for i in range(num_blocks + 1, MAX_BLOCKS + 1):
            unused_name = block_name(i)
            try:
                unused_obj = om.get_object_by_name(unused_name)
                self._move_object_away(env, unused_obj)
            except Exception as e:
                log.warning(f"[BLOCK SUPPORT] Failed to move unused block '{unused_name}': {e}")

        # Set pickup/receptacle for initial stacking (block_2 onto block_1)
        self.config.task_config.pickup_obj_name = active_block_names[1]
        self.config.task_config.place_receptacle_name = active_block_names[0]
        self._task_counter += 1

        # Place robot near all active blocks
        self._sample_and_place_robot(env, active_block_names)

        mujoco.mj_forward(env.current_model, env.current_data)

        self.setup_cameras(env)

        # Apply robot qpos after all placement
        robot_view = env.current_robot.robot_view
        for group_name, joint_pos in self._init_robot_qpos.items():
            robot_view.get_move_group(group_name).joint_pos = joint_pos

        return BlockSupportTask(env, self.config)

    def _sample_and_place_robot(self, env: CPUMujocoEnv, active_block_names: list[str]) -> None:
        """Place robot within reach of all blocks, like packing does for pickup + box."""
        from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
        from molmo_spaces.utils.pose import pose_mat_to_7d

        om = env.object_managers[env.current_batch_index]
        task_cfg = self.config.task_config
        task_sampler_config = self.config.task_sampler_config

        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()

        # Gather all active block objects as targets for robot placement
        cube_objects = []
        for name in active_block_names:
            cube_objects.append(om.get_object_by_name(name))

        robot_view = env.current_robot.robot_view
        target_pos = pickup_obj.position

        initial_robot_z = (
            target_pos[2]
            + task_sampler_config.robot_object_z_offset
            + np.random.uniform(
                task_sampler_config.robot_object_z_offset_random_min,
                task_sampler_config.robot_object_z_offset_random_max,
            )
        )

        robot_placed = env.place_robot_near_multiple(
            robot_view=robot_view,
            targets=cube_objects,
            face_target_index=0,
            max_tries=10,
            sampling_radius_range=task_sampler_config.base_pose_sampling_radius_range,
            robot_safety_radius=task_sampler_config.robot_safety_radius,
            preserve_z=initial_robot_z,
            face_target=True,
            check_camera_visibility=task_sampler_config.check_robot_placement_visibility,
            visibility_resolver=self.get_visibility_resolver(env),
            excluded_positions=self.used_robot_positions[pickup_obj.name],
            save_visibility_frames_dir=self.config.output_dir,
        )

        if not robot_placed:
            raise RobotPlacementError(f"Failed to place robot near blocks {active_block_names}")

        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

    def _place_cubes_near_reference(
        self, env: CPUMujocoEnv, reference_object: MjThorObject, active_block_names: list[str]
    ) -> None:
        """Place support cubes near a reference graspable object."""
        import mujoco

        om = env.object_managers[env.current_batch_index]
        task_sampler_config = self.config.task_sampler_config

        target_obj_name = self.config.task_config.pickup_obj_name
        target_obj_id = env.current_model.body(target_obj_name).id
        target_obj_pos = body_base_pos(env.current_data, target_obj_id)

        supporting_geom_id = get_supporting_geom(env.current_data, target_obj_id)
        log.info(
            f"[BLOCK SUPPORT] Placing cubes near '{target_obj_name}' at position "
            f"({target_obj_pos[0]:.3f}, {target_obj_pos[1]:.3f}, {target_obj_pos[2]:.3f})"
        )

        # Move the reference object far away to avoid interference
        try:
            self._move_object_away(env, reference_object)
        except Exception as e:
            log.warning(f"[BLOCK SUPPORT] Failed to move reference object away: {e}")

        mujoco.mj_forward(env.current_model, env.current_data)

        # Place each active cube near the reference position
        for name in active_block_names:
            cube = om.get_object_by_name(name)

            try:
                place_object_near(
                    data=env.current_data,
                    object_id=cube.object_id,
                    placement_point=target_obj_pos,
                    min_dist=task_sampler_config.min_block_placement_dist,
                    max_dist=task_sampler_config.max_block_placement_dist,
                    max_tries=task_sampler_config.max_block_placement_tries,
                    supporting_geom_id=supporting_geom_id,
                    z_eps=0.003,
                )
                mujoco.mj_forward(env.current_model, env.current_data)
                cube_pos = cube.position
                log.info(
                    f"[BLOCK SUPPORT] Placed {name} at "
                    f"({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})"
                )
            except Exception as e:
                log.error(f"[BLOCK SUPPORT] Failed to place {name} near target object: {e}")
                raise

    def _move_object_away(self, env: CPUMujocoEnv, obj: MjThorObject) -> None:
        """Move an object far away from the scene."""
        away_pos = np.array([10.0, 10.0, 10.0])

        # Get the object's joint information
        body_jntadr = env.current_model.body_jntadr[obj.object_id]
        body_jntnum = env.current_model.body_jntnum[obj.object_id]

        if body_jntnum > 0:
            jnt_id = body_jntadr
            jnt_type = env.current_model.jnt_type[jnt_id]

            # Check if it's a free joint (which includes position)
            if jnt_type == 0:  # mjJNT_FREE
                qposadr = env.current_model.jnt_qposadr[jnt_id]
                # Set position (first 3 values of qpos for free joint)
                env.current_data.qpos[qposadr : qposadr + 3] = away_pos
                log.info(f"[BLOCK SUPPORT] Moved '{obj.name}' to {away_pos}")

        import mujoco

        mujoco.mj_forward(env.current_model, env.current_data)

    @staticmethod
    def _add_support_cube(spec: MjSpec, pos=None, name="block_1", color=None) -> None:
        """
        Add a support cube to the scene with separate visual and collision geometry.
        Args:
            spec: MuJoCo MjSpec object
            pos: [x, y, z] position. If None, uses a default position
            name: Name for the cube body
            color: RGBA color for the visual geometry. Defaults to red [1, 0, 0, 1]
        """
        if pos is None:
            # Default position on table surface
            pos = [0.0, 0.5, 0.71]

        if color is None:
            color = [1, 0, 0, 1]  # Default to red

        # Create cube body with free joint for physics simulation
        cube_body = spec.worldbody.add_body(name=name, pos=pos)
        cube_body.add_freejoint()

        # Add visual geometry
        cube_body.add_geom(
            name=f"{name}_visual",
            type=mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.03, 0.03],  # Half-size of 3cm per side = 6cm cube
            rgba=color,  # Specified color (RGBA)
            contype=0,  # Visual-only geometry
            conaffinity=0,
        )

        # Add collision geometry (invisible collider for physics)
        cube_body.add_geom(
            name=f"{name}_collision",
            type=mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.03, 0.03],  # Same size as visual
            rgba=[0, 0, 0, 0],  # Invisible (alpha=0)
            friction=[1.0, 0.005, 0.0001],  # Add friction for stability
        )
