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
    clear_surface,
    get_supporting_geom,
    place_object_near,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

RED_CUBE_NAME = "red_cube"
RED_CUBE_2_NAME = "red_cube_2"


class BlockSupportTaskSampler(PickTaskSampler):
    """
    Task sampler for block support tasks that extends PickTaskSampler.
    Adds red and blue cubes to the scene and places them near random graspable objects.
    """

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Add auxiliary objects for block support task.
        Adds red and blue cubes to the scene as target pickup objects.
        """
        # Add red cube
        self._add_support_cube(spec, name=RED_CUBE_NAME, color=[1, 0, 0, 1])

        # Add blue cube
        self._add_support_cube(
            spec, name=RED_CUBE_2_NAME, color=[1, 0, 0, 1]
        )  # color=[0, 0, 1, 1])

        # Call parent class to add any additional auxiliary objects from policy
        super().add_auxiliary_objects(spec)

    def _get_scene_objects(self, env: CPUMujocoEnv) -> list[MjThorObject]:
        """Override to return only the support cubes as the candidate objects."""
        cubes = []
        for cube_name in [RED_CUBE_NAME, RED_CUBE_2_NAME]:
            try:
                cube_obj = env.get_object_by_name(cube_name)
                log.info(f"Found cube '{cube_name}' as candidate pickup object")
                cubes.append(cube_obj)
            except Exception as e:
                log.error(f"Could not find cube '{cube_name}': {e}")
                raise
        return cubes

    def _sample_task(self, env: CPUMujocoEnv) -> BlockSupportTask:
        """Override to place support cubes near a random graspable object before sampling task."""
        import mujoco

        # Do all the parent task sampling setup, then create BlockSupportTask
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        # Set a temporary pickup object, used for robot placement. We need this because
        # robot should be positioned before we place multiple blocks, so that we can ensure
        # all blocks are within the workspace (we could position robot after, but hard to do
        # that s.t. many already positioned blocks are all within worksapce, easier to just
        # place blocks after so that they're within positioned robot's workspace)
        all_graspable_objects = env.get_objects_of_type(THOR_PICKUP_OBJECTS_LOWERCASE)
        tmp_pickup_obj = np.random.choice(all_graspable_objects)
        self.config.task_config.pickup_obj_name = tmp_pickup_obj.name

        # Clutter scene and place robot
        print(
            f"[DEBUG block_support_task_sampler] BEFORE _sample_and_place_robot: qpos = {env.current_robot.robot_view.get_qpos_dict()}"
        )
        self._sample_and_place_robot(env)
        print(
            f"[DEBUG block_support_task_sampler] AFTER _sample_and_place_robot: qpos = {env.current_robot.robot_view.get_qpos_dict()}"
        )
        # Place the support cubes near a random graspable object
        self._place_cube_near_graspable_object(env, tmp_pickup_obj)

        # Sample pickup object for real now
        # TODO: if we have more than 2 blocks, we should have a list of pickup objects, and
        # one place receptacle for the bottom block
        if self.config.task_config.pickup_obj_name is None:
            object_index = self._task_counter % len(self.candidate_objects)
            self.config.task_config.pickup_obj_name = self.candidate_objects[object_index].name
            # FIXME: Below will not work if more than 2 objects, update this when I generalize to more cubes
            self.config.task_config.place_receptacle_name = self.candidate_objects[
                1 - object_index
            ].name
        self._task_counter += 1

        mujoco.mj_forward(env.current_model, env.current_data)

        # Setup cameras and create the task
        self.setup_cameras(env)

        # Create and return BlockSupportTask instead of PickTask
        task = BlockSupportTask(env, self.config)
        print(
            f"[DEBUG block_support_task_sampler] AFTER BlockSupportTask.__init__: qpos = {env.current_robot.robot_view.get_qpos_dict()}"
        )
        return task

    def _place_cube_near_graspable_object(self, env: CPUMujocoEnv, target_object) -> None:
        """Place both support cubes near the same randomly selected graspable object in the scene.
        After placing the cubes, moves the reference object far away to avoid interference.
        """
        import mujoco

        # Get all graspable objects in the scene
        graspable_objects = env.get_objects_of_type(THOR_PICKUP_OBJECTS_LOWERCASE)

        if not graspable_objects:
            log.warning(
                "[BLOCK SUPPORT] No graspable objects found in scene, skipping cube placement"
            )
            return

        task_sampler_config = self.config.task_sampler_config

        target_obj_name = self.config.task_config.pickup_obj_name
        target_obj_id = env.current_model.body(target_obj_name).id
        target_obj_pos = body_base_pos(env.current_data, target_obj_id)

        supporting_geom_id = get_supporting_geom(env.current_data, target_obj_id)
        log.info(
            f"[BLOCK SUPPORT] Placing cubes near '{target_obj_name}' at position ({target_obj_pos[0]:.3f}, {target_obj_pos[1]:.3f}, {target_obj_pos[2]:.3f})"
        )

        # First, move the reference object far away to avoid interference
        try:
            self._move_object_away(env, target_object)
        except Exception as e:
            log.warning(f"[BLOCK SUPPORT] Failed to move reference object away: {e}")

        # Ensure forward kinematics are up to date
        mujoco.mj_forward(env.current_model, env.current_data)

        # Place each cube near the same graspable object
        cube_object_ids = []
        for cube_name in [RED_CUBE_NAME, RED_CUBE_2_NAME]:
            # Get the cube object ID
            cube = env.get_object_by_name(cube_name)
            cube_object_id = cube.object_id
            cube_object_ids.append(cube_object_id)

            # Place the cube near the target object
            try:
                # place_object_near(
                #     data=env.current_data,
                #     object_id=cube_object_id,
                #     placement_point=target_pos,
                #     min_dist=0.05,  # 5cm minimum distance
                #     max_dist=0.15,  # 15cm maximum distance
                #     max_tries=50,
                # )
                place_object_near(
                    data=env.current_data,
                    object_id=cube_object_id,
                    placement_point=target_obj_pos,
                    min_dist=0.05,
                    max_dist=0.15,
                    max_tries=100,
                    reference_pos=env.current_robot.robot_view.base.pose[:3, 3],
                    max_dist_to_reference=task_sampler_config.max_robot_to_block_dist,
                    supporting_geom_id=supporting_geom_id,
                    z_eps=0.003,
                    env=env,
                )
                mujoco.mj_forward(env.current_model, env.current_data)
                cube_pos = cube.position
                log.info(
                    f"[BLOCK SUPPORT] Successfully placed {cube_name} at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})"
                )
            except Exception as e:
                log.error(f"[BLOCK SUPPORT] Failed to place {cube_name} near target object: {e}")
                raise

        # We now clear the surface of all other objects
        # TODO: should just clear objects in immediate vicinity of placed blocks
        clear_surface(
            data=env.current_data,
            supporting_geom_id=supporting_geom_id,
            object_ids_to_keep=cube_object_ids,
        )

        # We now set the pickup and receptacle object
        object_index = self._task_counter % len(self.candidate_objects)
        self.config.task_config.pickup_obj_name = self.candidate_objects[object_index].name
        self.config.task_config.place_receptacle_name = self.candidate_objects[
            1 - object_index
        ].name
        log.info(
            f"✅ Attempting object {self.config.task_config.pickup_obj_name} {object_index}/{len(self.candidate_objects)}"
        )

        # Settle the scene to let cubes fall naturally
        # for _ in range(500):
        #     mujoco.mj_step(env.current_model, env.current_data)

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
    def _add_support_cube(spec: MjSpec, pos=None, name=RED_CUBE_NAME, color=None) -> None:
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

        # Add visual geometry (6cm = 0.06m, so half-size = 0.02m per side)
        cube_body.add_geom(
            name=f"{name}_visual",
            type=mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],  # Half-size of 2cm per side = 4cm cube (was scaled by linter)
            rgba=color,  # Specified color (RGBA)
            contype=0,  # Visual-only geometry
            conaffinity=0,
        )

        # Add collision geometry (invisible collider for physics)
        cube_body.add_geom(
            name=f"{name}_collision",
            type=mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],  # Same size as visual
            rgba=[0, 0, 0, 0],  # Invisible (alpha=0)
            friction=[1.0, 0.005, 0.0001],  # Add friction for stability
        )
