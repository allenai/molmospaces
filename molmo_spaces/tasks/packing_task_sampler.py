import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.packing_task import PackingTask
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler
from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.pose import pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_packing_configs import PackingDataGenConfig

log = logging.getLogger(__name__)

# Default THOR cardboard box UIDs
DEFAULT_BOX_UIDS = [f"Box_{i}" for i in range(1, 31)]


class PackingTaskSampler(PickAndPlaceTaskSampler):
    def __init__(self, config: "PackingDataGenConfig") -> None:
        assert config.task_type == "packing"
        super().__init__(config)
        self.config: PackingDataGenConfig

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        # Call grandparent (PickTaskSampler) to add pickup objects, skip PickAndPlaceTaskSampler's
        # receptacle selection logic since we use THOR box assets directly.
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        PickTaskSampler.add_auxiliary_objects(self, spec)

        task_sampler_config = self.config.task_sampler_config

        box_uids = task_sampler_config.box_uids or DEFAULT_BOX_UIDS
        uid = np.random.choice(box_uids)
        uid = "Box_11"  # TODO: temporary — shortest box (0.061m) for IK debugging
        box_xml = install_uid(uid)

        box_spec = MjSpec.from_file(str(box_xml))
        if len(box_spec.worldbody.bodies) != 1:
            log.warning(
                f"{box_xml} has {len(box_spec.worldbody.bodies)} bodies, expected 1. Using first one."
            )
        box_obj: mujoco.MjsBody = box_spec.worldbody.bodies[0]
        if not box_obj.first_joint():
            box_obj.add_joint(
                name=f"{uid}_jntfree",
                type=mujoco.mjtJoint.mjJNT_FREE,
                damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
            )

        attach_frame = spec.worldbody.add_frame(
            pos=[10, 10, 10], quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
        )
        attach_frame.attach_body(box_obj, task_sampler_config.place_receptacle_namespace, "")
        self.place_receptacle_name = box_obj.name
        self._receptacle_names = [box_obj.name]

        # Save added object for scene recreation
        xml_path_rel = box_xml.relative_to(ASSETS_DIR)
        self.config.task_config.added_objects[box_obj.name] = xml_path_rel

        self._metadata_adder.update(
            {
                box_obj.name: {
                    "asset_id": uid,
                    "category": "Box",
                    "object_enum": "temp_object",
                    "is_static": False,
                }
            }
        )

    def _filter_place_target(self, env, pickup_obj_name, place_target_name) -> bool:
        """Skip size filtering — the box is always large enough for pickup objects."""
        return True

    def _open_box_flaps(self, env: CPUMujocoEnv) -> None:
        """Open all flap joints on the box so objects can be placed inside."""
        model, data = env.current_model, env.current_data
        namespace = self.config.task_sampler_config.place_receptacle_namespace

        opened_count = 0
        for i in range(model.njnt):
            jnt_name = model.joint(i).name
            if namespace in jnt_name and "flap" in jnt_name:
                jnt_range = model.jnt_range[i]
                # Open = extreme of joint range away from 0
                if abs(jnt_range[0]) > abs(jnt_range[1]):
                    target_val = jnt_range[0]
                else:
                    target_val = jnt_range[1]

                qposadr = model.jnt_qposadr[i]
                dofadr = model.jnt_dofadr[i]

                data.qpos[qposadr] = target_val
                data.qvel[dofadr] = 0
                # Also update qpos0 so any reset defaults to open
                model.qpos0[qposadr] = target_val
                # Freeze the joint with very high damping so flaps stay open during simulation
                model.dof_damping[dofadr] = 1e6
                opened_count += 1
                log.info(
                    f"Flap '{jnt_name}': qposadr={qposadr}, set qpos={target_val:.3f}, "
                    f"verified qpos={data.qpos[qposadr]:.3f}"
                )

        if opened_count == 0:
            log.warning(
                f"No flap joints found with namespace '{namespace}'. "
                f"All joints: {[model.joint(i).name for i in range(model.njnt)]}"
            )
        else:
            log.info(f"Opened and frozen {opened_count} box flap joints")

        mujoco.mj_forward(model, data)

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
            face_target_index=0,
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
        """Sample a packing task — open box flaps, then delegate placement to parent."""
        # First let parent handle all placement (box, robot, cameras)
        _ = super()._sample_task(env)

        # Open flaps AFTER placement so nothing can overwrite the qpos values
        self._open_box_flaps(env)

        # Verify flap qpos values survived
        model, data = env.current_model, env.current_data
        namespace = self.config.task_sampler_config.place_receptacle_namespace
        for i in range(model.njnt):
            jnt_name = model.joint(i).name
            if namespace in jnt_name and "flap" in jnt_name:
                log.info(f"VERIFY '{jnt_name}': qpos={data.qpos[model.jnt_qposadr[i]]:.3f}")

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
