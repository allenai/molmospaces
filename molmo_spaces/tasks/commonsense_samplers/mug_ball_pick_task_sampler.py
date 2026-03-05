import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec, mjtGeom
from PIL import Image
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MjThorObject, create_mjthor_body
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.commonsense_tasks.mug_ball_pick_task import MugBallPickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler, SameClassClutterMetadataAdder
from molmo_spaces.tasks.task_sampler_errors import RobotPlacementError
from molmo_spaces.utils.constants.object_constants import THOR_PICKUP_OBJECTS_LOWERCASE
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos, geom_aabb
from molmo_spaces.utils.mujoco_scene_utils import (
    get_supporting_geom,
    place_object_near,
)
from molmo_spaces.utils.pose import pose_mat_to_7d

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Asset IDs for the two mugs (iThor mug models)
MUG_ASSET_IDS = ["Mug_1", "Mug_2"]

BALL_NAME = "mug_ball_pick_ball"
INVISIBLE_TARGET_NAME = "mug_ball_pick_place_target"

# Upside-down quaternion: mug opening faces +Y in local frame, scene is Z-up.
# Rotate -90° around X so local +Y maps to world -Z (opening faces down).
UPSIDE_DOWN_QUAT = R.from_euler("x", -90, degrees=True).as_quat(scalar_first=True)

# Ball dimensions
BALL_RADIUS = 0.012  # 12mm radius

# Height above surface for mug placement
MUG_DROP_HEIGHT = 0.15  # 15cm above surface

# Minimum distance from ball/mug landing spots to other graspable objects
MIN_DIST_TO_GRASPABLE = 0.10  # 10cm
# Minimum distance from ball/mug positions to the edge of the receptacle
MIN_DIST_TO_EDGE = 0.05  # 5cm
# Minimum XY distance between the two mug landing positions
MIN_MUG_DISTANCE = 0.10  # 10cm
MAX_PLACEMENT_RETRIES = 20


class MugBallPickTaskSampler(PickTaskSampler):
    """Task sampler for the mug-ball pick task.

    Adds a ball and two iThor mugs to the scene. The ball is placed on a counter
    surface, and both mugs are spawned upside-down 10cm above the surface.
    One mug is directly above the ball, the other above an empty spot.
    When the scene starts, the mugs fall: one lands covering the ball,
    the other lands on empty surface.
    """

    def __init__(self, config):
        super().__init__(config)
        self.correct_mug_name: str | None = None
        self._mug_metadata_adder: SameClassClutterMetadataAdder | None = None
        # Actual body names in the compiled model (set during add_auxiliary_objects)
        self._mug_body_names: list[str] = []
        # Set to True after ball/mugs are in their final positions
        self._objects_placed = False

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Add the ball, two iThor mugs, and an invisible placement target."""
        self._add_ball(spec)

        # Load and add two iThor mug assets
        name_to_meta = {}
        self._mug_body_names = []
        for i, asset_id in enumerate(MUG_ASSET_IDS):
            body_name = self._add_ithor_mug(spec, asset_id, f"mbp_mug_{i}", name_to_meta)
            self._mug_body_names.append(body_name)

        if name_to_meta:
            self._mug_metadata_adder = SameClassClutterMetadataAdder(name_to_meta)

        self._add_invisible_target(spec)
        super().add_auxiliary_objects(spec)

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> list[str]:
        """Resolve visibility keys — include the ball once objects are placed."""
        if key == "__task_objects__":
            objects = super().resolve_visibility_object(env, key)
            if self._objects_placed:
                objects.append(BALL_NAME)
            return objects
        return super().resolve_visibility_object(env, key)

    def _get_scene_objects(self, env: CPUMujocoEnv) -> list[MjThorObject]:
        """Return only the two mugs as candidate pickup objects."""
        # Inject metadata for the dynamically added mugs
        if self._mug_metadata_adder is not None:
            self._mug_metadata_adder.add_meta(env.current_scene_metadata)

        mugs = []
        for mug_name in self._mug_body_names:
            try:
                mug_obj = MjThorObject(object_name=mug_name, data=env.current_data)
                log.info(f"Found mug '{mug_name}' as candidate pickup object")
                mugs.append(mug_obj)
            except Exception as e:
                log.error(f"Could not find mug '{mug_name}': {e}")
                raise
        return mugs

    def _sample_task(self, env: CPUMujocoEnv) -> MugBallPickTask:
        """Place ball and mugs, then create the task."""
        self._objects_placed = False
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0
        assert len(self._mug_body_names) == 2

        mug_1_name = self._mug_body_names[0]
        mug_2_name = self._mug_body_names[1]

        task_sampler_config = self.config.task_sampler_config

        # Step 1: Get a temporary pickup object from existing scene objects.
        # This gives us a reference point on a counter surface for placement.
        # Exclude our dynamically added mugs (they're off-screen at this point).
        om = env.object_managers[env.current_batch_index]
        all_graspable_objects = [
            obj
            for obj in om.get_objects_of_type(THOR_PICKUP_OBJECTS_LOWERCASE)
            if obj.name not in self._mug_body_names
        ]
        tmp_pickup_obj = np.random.choice(all_graspable_objects)
        self.config.task_config.pickup_obj_name = tmp_pickup_obj.name

        # Step 2: Place robot near the temporary pickup object
        self._sample_and_place_robot(env)

        # Fix robot qpos to a known configuration
        robot_view = env.current_robot.robot_view
        fixed_qpos = {
            "base": [],
            "arm": [[0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0]],
            "gripper": [0.00296, 0.00296],
        }
        for group_name in self.config.robot_config.init_qpos:
            robot_view.get_move_group(group_name).joint_pos = fixed_qpos[group_name]

        # Step 3: Get the supporting surface (counter/table) under the temp object
        tmp_obj_id = env.current_model.body(tmp_pickup_obj.name).id
        tmp_obj_pos = body_base_pos(env.current_data, tmp_obj_id)
        supporting_geom_id = get_supporting_geom(env.current_data, tmp_obj_id)

        if supporting_geom_id is None:
            raise RuntimeError(
                f"Could not find supporting surface for temp object '{tmp_pickup_obj.name}'"
            )

        # Move the temp pickup object out of the way
        self._move_object_away(env, tmp_pickup_obj)
        mujoco.mj_forward(env.current_model, env.current_data)

        # Step 4-5: Place ball and invisible target (mug 2 landing spot),
        # retrying if either is within 10cm of any non-receptacle scene object.
        ball_body_id = env.current_model.body(BALL_NAME).id
        inv_target_id = env.current_model.body(INVISIBLE_TARGET_NAME).id
        excluded_names = set(self._mug_body_names) | {
            tmp_pickup_obj.name,
            BALL_NAME,
            INVISIBLE_TARGET_NAME,
        }
        # Also exclude the supporting receptacle body
        supporting_body_id = env.current_model.geom_bodyid[supporting_geom_id]
        supporting_root_id = env.current_model.body_rootid[supporting_body_id]
        supporting_root_name = env.current_model.body(supporting_root_id).name
        excluded_names.add(supporting_root_name)

        all_scene_objects = om.list_top_level_objects()
        nearby_objects = [obj for obj in all_scene_objects if obj.name not in excluded_names]

        # Compute the supporting surface AABB for edge-distance checks
        support_aabb_center, support_aabb_size = geom_aabb(
            env.current_model, env.current_data, [supporting_geom_id]
        )
        support_min_xy = support_aabb_center[:2] - support_aabb_size[:2] / 2
        support_max_xy = support_aabb_center[:2] + support_aabb_size[:2] / 2

        ball_pos = None
        inv_target_pos = None
        for attempt in range(MAX_PLACEMENT_RETRIES):
            # Place ball on the counter surface near where the temp object was
            place_object_near(
                data=env.current_data,
                object_id=ball_body_id,
                placement_point=tmp_obj_pos,
                min_dist=0.03,
                max_dist=0.12,
                max_tries=100,
                reference_pos=env.current_robot.robot_view.base.pose[:3, 3],
                max_dist_to_reference=task_sampler_config.max_robot_to_block_dist,
                supporting_geom_id=supporting_geom_id,
                z_eps=0.003,
            )
            mujoco.mj_forward(env.current_model, env.current_data)
            ball_view = create_mjthor_body(env.current_data, BALL_NAME)
            ball_pos = ball_view.position.copy()

            # Place invisible target (mug 2 landing spot) near the ball
            place_object_near(
                data=env.current_data,
                object_id=inv_target_id,
                placement_point=ball_pos,
                min_dist=MIN_MUG_DISTANCE,
                max_dist=0.18,
                max_tries=100,
                reference_pos=env.current_robot.robot_view.base.pose[:3, 3],
                max_dist_to_reference=task_sampler_config.max_robot_to_block_dist,
                supporting_geom_id=supporting_geom_id,
                z_eps=0.003,
            )
            mujoco.mj_forward(env.current_model, env.current_data)
            inv_target_view = create_mjthor_body(env.current_data, INVISIBLE_TARGET_NAME)
            inv_target_pos = inv_target_view.position.copy()

            # Check proximity to other scene objects (toasters, etc.)
            too_close = self._find_nearby_objects(
                env, nearby_objects, [ball_pos, inv_target_pos], MIN_DIST_TO_GRASPABLE
            )
            if too_close:
                log.info(
                    f"[MUG BALL PICK] Placement attempt {attempt + 1}/{MAX_PLACEMENT_RETRIES}: "
                    f"objects {[o.name for o in too_close]} within {MIN_DIST_TO_GRASPABLE * 100:.0f}cm, resampling..."
                )
                continue

            # Check that both positions are at least MIN_DIST_TO_EDGE from the receptacle edges
            too_close_to_edge = False
            for label, pos in [("ball", ball_pos), ("mug2 target", inv_target_pos)]:
                dist_to_min = pos[:2] - support_min_xy
                dist_to_max = support_max_xy - pos[:2]
                min_edge_dist = min(dist_to_min.min(), dist_to_max.min())
                if min_edge_dist < MIN_DIST_TO_EDGE:
                    log.info(
                        f"[MUG BALL PICK] Placement attempt {attempt + 1}/{MAX_PLACEMENT_RETRIES}: "
                        f"{label} too close to receptacle edge ({min_edge_dist * 100:.1f}cm < {MIN_DIST_TO_EDGE * 100:.0f}cm), resampling..."
                    )
                    too_close_to_edge = True
                    break
            if too_close_to_edge:
                continue

            # Temporarily place mugs in their floating positions to check for collisions
            mug1_view = create_mjthor_body(env.current_data, mug_1_name)
            mug1_pos = ball_pos.copy()
            mug1_pos[2] += MUG_DROP_HEIGHT
            mug1_view.position = mug1_pos
            mug1_view.quat = UPSIDE_DOWN_QUAT

            mug2_view = create_mjthor_body(env.current_data, mug_2_name)
            mug2_pos = inv_target_pos.copy()
            mug2_pos[2] += MUG_DROP_HEIGHT
            mug2_view.position = mug2_pos
            mug2_view.quat = UPSIDE_DOWN_QUAT

            mujoco.mj_forward(env.current_model, env.current_data)
            log.info(
                f"[MUG BALL PICK] Placed '{mug_1_name}' upside-down at "
                f"({mug1_pos[0]:.3f}, {mug1_pos[1]:.3f}, {mug1_pos[2]:.3f})"
            )
            log.info(
                f"[MUG BALL PICK] Placed '{mug_2_name}' upside-down at "
                f"({mug2_pos[0]:.3f}, {mug2_pos[1]:.3f}, {mug2_pos[2]:.3f})"
            )

            # Check that neither mug collides with anything (except the supporting receptacle)
            mug1_body_id = env.current_model.body(mug_1_name).id
            mug2_body_id = env.current_model.body(mug_2_name).id
            mug_in_collision = False
            for mug_label, mug_bid in [("mug1", mug1_body_id), ("mug2", mug2_body_id)]:
                for c in env.current_data.contact:
                    root1 = env.current_model.body_rootid[env.current_model.geom_bodyid[c.geom1]]
                    root2 = env.current_model.body_rootid[env.current_model.geom_bodyid[c.geom2]]
                    if (root1 == mug_bid) ^ (root2 == mug_bid):
                        other_root = root1 if root1 != mug_bid else root2
                        if other_root != supporting_root_id:
                            other_name = env.current_model.body(other_root).name
                            log.info(
                                f"[MUG BALL PICK] Placement attempt {attempt + 1}/{MAX_PLACEMENT_RETRIES}: "
                                f"{mug_label} in collision with '{other_name}' at floating height, resampling..."
                            )
                            mug_in_collision = True
                            break
                if mug_in_collision:
                    break

            if mug_in_collision:
                # Move mugs back off-screen before retrying
                mug1_view.position = np.array([20.0, 20.0, 5.0])
                mug2_view.position = np.array([20.0, 20.0, 5.0])
                mujoco.mj_forward(env.current_model, env.current_data)
                continue

            break
        else:
            # Max retries reached — move offending objects away as fallback
            too_close = self._find_nearby_objects(
                env, nearby_objects, [ball_pos, inv_target_pos], MIN_DIST_TO_GRASPABLE
            )
            for obj in too_close:
                log.warning(
                    f"[MUG BALL PICK] Moving nearby object '{obj.name}' away after "
                    f"{MAX_PLACEMENT_RETRIES} retries"
                )
                self._move_object_away(env, obj)

            # Place mugs in their final floating positions (best-effort after retries)
            mug1_view = create_mjthor_body(env.current_data, mug_1_name)
            mug1_pos = ball_pos.copy()
            mug1_pos[2] += MUG_DROP_HEIGHT
            mug1_view.position = mug1_pos
            mug1_view.quat = UPSIDE_DOWN_QUAT

            mug2_view = create_mjthor_body(env.current_data, mug_2_name)
            mug2_pos = inv_target_pos.copy()
            mug2_pos[2] += MUG_DROP_HEIGHT
            mug2_view.position = mug2_pos
            mug2_view.quat = UPSIDE_DOWN_QUAT

            mujoco.mj_forward(env.current_model, env.current_data)

        log.info(
            f"[MUG BALL PICK] Placed ball at "
            f"({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f})"
        )
        log.info(
            f"[MUG BALL PICK] Placed invisible target at "
            f"({inv_target_pos[0]:.3f}, {inv_target_pos[1]:.3f}, {inv_target_pos[2]:.3f})"
        )

        # Move invisible target away (it served its purpose as a placement reference)
        inv_target_view = create_mjthor_body(env.current_data, INVISIBLE_TARGET_NAME)
        inv_target_view.position = np.array([10.0, 10.0, 10.0])

        # Setup cameras early so we can capture debug images
        self.setup_cameras(env)

        # Capture debug image
        self._save_debug_image(env, "scene")

        # Step 9: Track which mug is over the ball. Mug 1 is always over the ball.
        self.correct_mug_name = mug_1_name
        self._objects_placed = True

        # Step 10: Set the correct mug as the pickup target
        self.config.task_config.pickup_obj_name = self.correct_mug_name

        # Verify that ball and mugs are visible from exo camera
        exo_cam_name = "exo_camera_1"
        if exo_cam_name in env.camera_manager.registry.cameras:
            task_objects = self.resolve_visibility_object(env, "__task_objects__")
            if task_objects:
                vis = env.check_visibility(exo_cam_name, *task_objects)
                if not isinstance(vis, dict):
                    vis = {task_objects[0]: vis}
                invisible = [obj for obj, frac in vis.items() if frac <= 0.0]
                if invisible:
                    raise RobotPlacementError(
                        f"[MUG BALL PICK] Objects {invisible} not visible from '{exo_cam_name}'"
                    )

        # Update task config with pickup object pose info (use settled pose)
        mug1_obj = MjThorObject(object_name=mug_1_name, data=env.current_data)
        self.config.task_config.pickup_obj_start_pose = pose_mat_to_7d(mug1_obj.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(mug1_obj.pose)
        pickup_obj_goal_pose[2] += 0.05  # 5cm above current position
        self.config.task_config.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

        # Set settle duration so the task forces no-op actions while mugs fall
        self.config.task_config.scene_settle_duration = 5.0

        self._task_counter += 1

        # Create and return the task
        task = MugBallPickTask(env, self.config)
        task.correct_mug_name = self.correct_mug_name
        log.info(f"[MUG BALL PICK] Task created. Correct mug: {self.correct_mug_name}")
        return task

    @staticmethod
    def _find_nearby_objects(
        env: CPUMujocoEnv,
        scene_objects: list[MjThorObject],
        check_positions: list[np.ndarray],
        min_dist: float,
    ) -> list[MjThorObject]:
        """Return scene objects that are within min_dist of any check position."""
        too_close = []
        for obj in scene_objects:
            obj_pos = body_base_pos(env.current_data, obj.object_id)
            for pos in check_positions:
                if np.linalg.norm(obj_pos - pos) < min_dist:
                    too_close.append(obj)
                    break
        return too_close

    def _move_object_away(self, env: CPUMujocoEnv, obj: MjThorObject) -> None:
        """Move an object far away from the scene."""
        away_pos = np.array([10.0, 10.0, 10.0])
        body_jntadr = env.current_model.body_jntadr[obj.object_id]
        body_jntnum = env.current_model.body_jntnum[obj.object_id]

        if body_jntnum > 0:
            jnt_id = body_jntadr
            jnt_type = env.current_model.jnt_type[jnt_id]
            if jnt_type == 0:  # mjJNT_FREE
                qposadr = env.current_model.jnt_qposadr[jnt_id]
                env.current_data.qpos[qposadr : qposadr + 3] = away_pos
                # Zero velocities and disable gravity so displaced object
                # doesn't fall forever through empty space
                dofadr = env.current_model.jnt_dofadr[jnt_id]
                env.current_data.qvel[dofadr : dofadr + 6] = 0
                env.current_model.body_gravcomp[obj.object_id] = 1.0
                log.info(f"[MUG BALL PICK] Moved '{obj.name}' to {away_pos}")

        mujoco.mj_forward(env.current_model, env.current_data)

    def _save_debug_image(self, env: CPUMujocoEnv, label: str) -> None:
        """Save a debug image from the exo camera."""
        try:
            frame = env.render_rgb_frame("exo_camera_1")
            debug_dir = Path("debug_mug_ball_pick")
            debug_dir.mkdir(exist_ok=True)
            filename = debug_dir / f"{label}_task{self._task_counter}.png"
            Image.fromarray(frame).save(filename)
            log.info(f"[MUG BALL PICK] Saved debug image: {filename}")
        except Exception as e:
            log.warning(f"[MUG BALL PICK] Could not save debug image '{label}': {e}")

    @staticmethod
    def _add_ball(spec: MjSpec, name: str = BALL_NAME, pos=None) -> None:
        """Add a small ball (sphere) to the scene."""
        if pos is None:
            pos = [0.0, 0.5, 0.71]

        ball_body = spec.worldbody.add_body(name=name, pos=pos)
        ball_body.add_freejoint()

        # Visual geom
        ball_body.add_geom(
            name=f"{name}_visual",
            type=mjtGeom.mjGEOM_SPHERE,
            size=[BALL_RADIUS, 0, 0],
            rgba=[1, 1, 0, 1],  # Yellow
            contype=0,
            conaffinity=0,
        )
        # Collision geom
        ball_body.add_geom(
            name=f"{name}_collision",
            type=mjtGeom.mjGEOM_SPHERE,
            size=[BALL_RADIUS, 0, 0],
            rgba=[0, 0, 0, 0],
            friction=[0.5, 0.005, 0.0001],
        )

    def _add_ithor_mug(
        self,
        spec: MjSpec,
        asset_id: str,
        namespace_prefix: str,
        name_to_meta: dict,
    ) -> str:
        """Load an iThor mug asset and add it to the scene.

        Args:
            spec: The MjSpec to add the mug to.
            asset_id: The iThor asset ID (e.g. "Mug_1").
            namespace_prefix: Prefix for the attach namespace.
            name_to_meta: Dict to accumulate metadata for the added object.

        Returns:
            The actual body name of the mug in the compiled model.
        """
        mug_xml = install_uid(asset_id)
        mug_spec = MjSpec.from_file(str(mug_xml))

        if len(mug_spec.worldbody.bodies) == 0:
            raise RuntimeError(f"Mug asset {asset_id} has no bodies in its XML")

        mug_obj: mujoco.MjsBody = mug_spec.worldbody.bodies[0]

        # Ensure the body has a free joint
        if not mug_obj.first_joint():
            mug_obj.add_joint(
                name=f"{namespace_prefix}_jntfree",
                type=mujoco.mjtJoint.mjJNT_FREE,
                damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
            )

        # Place off-screen initially (will be repositioned in _sample_task)
        attach_frame = spec.worldbody.add_frame(
            pos=[20, 20, 5],
        )
        namespace = f"{namespace_prefix}/"
        attach_frame.attach_body(mug_obj, namespace, "")

        # The body name in the compiled model is the original XML body name
        body_name = mug_obj.name

        # Track metadata for the added mug
        name_to_meta[body_name] = {
            "asset_id": asset_id,
            "category": "Mug",
            "object_enum": "temp_object",
            "is_static": False,
            "boundingBox": {"x": 0.09, "y": 0.10, "z": 0.09},
        }

        # Save added object path for scene recreation
        xml_path_rel = mug_xml.relative_to(ASSETS_DIR)
        self.config.task_config.added_objects[body_name] = xml_path_rel

        log.info(f"[MUG BALL PICK] Added iThor mug '{asset_id}' as body '{body_name}'")
        return body_name

    @staticmethod
    def _add_invisible_target(spec: MjSpec, name: str = INVISIBLE_TARGET_NAME, pos=None) -> None:
        """Add an invisible body with no colliders.

        Used as a placement reference for mug 2. Placed on the counter surface
        via place_object_near, then mug 2 is positioned above it.
        """
        if pos is None:
            pos = [0.0, 0.5, 0.71]

        target_body = spec.worldbody.add_body(name=name, pos=pos)
        target_body.add_freejoint()

        # Invisible sphere with no collision — just a placement anchor
        target_body.add_geom(
            name=f"{name}_geom",
            type=mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],
            rgba=[0, 0, 0, 0],
            contype=0,
            conaffinity=0,
        )
