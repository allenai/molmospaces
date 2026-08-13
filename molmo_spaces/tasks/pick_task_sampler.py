import logging
from collections.abc import Collection
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec, mjtGeom
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.arena.arena_utils import modify_mjmodel_thor_articulated
from molmo_spaces.env.data_views import (
    MlSpacesArticulationObject,
    MlSpacesObject,
    create_mlspaces_body,
)
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import Context, ObjectManager
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.tasks.task_sampler_errors import (
    HouseInvalidForTask,
    ObjectPlacementError,
    RobotPlacementError,
)
from molmo_spaces.utils.asset_names import get_thor_name
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.grasp_sample import (
    get_noncolliding_grasp_mask,
    select_grasp_pose,
)
from molmo_spaces.utils.grasps import (
    get_pickup_grasps,
    has_pickup_grasp_path,
    has_valid_pickup_grasps,
)
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos
from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom, place_object_near
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.task_relevant_objects_and_workspace_utils import (
    compute_workspace_center,
    get_task_relevant_objects,
)

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_config import PickBaseConfig


log = logging.getLogger(__name__)

MAX_BOTTOM_Z_DIFFERENCE = 0.05  # 5cm

_VALID_PICKUPABLE_CACHE: dict[str, dict] | None = None


def get_valid_pickupable_uids(
    pickupable_synsets_categories_or_uids: Collection[str], split: str
) -> dict[str, dict]:
    """Get all asset UIDs that are valid pickupables from synsets, categories, or direct UIDs."""
    valid_uids = {}

    asset_ids = set(pickupable_synsets_categories_or_uids) & set(ObjectMeta.annotation().keys())
    for uid in asset_ids:
        valid_uids[uid] = ObjectMeta.annotation(uid)

    categories_and_synsets = {
        reference.lower().replace("_", " ").strip()
        for reference in set(pickupable_synsets_categories_or_uids) - asset_ids
    }

    if categories_and_synsets:
        import hashlib

        from tqdm import tqdm

        class DummyEnv:
            mj_datas = [None]

        om = ObjectManager(DummyEnv(), -1)  # type:ignore
        om.scene_metadata = {"objects": {}}

        for uid, anno in tqdm(ObjectMeta.annotation().items(), "caching pickupables"):
            if uid in valid_uids:
                continue

            if split == "test" and anno["split"] not in ["test", None]:
                continue
            elif split == "train" and anno["split"] != split:
                continue
            elif split == "val" and anno["split"] not in ["val", "train"]:
                continue

            category = anno["category"]
            name = f"{category.lower()}_{hashlib.md5(uid.encode()).hexdigest()}_0_0_0"

            om.scene_metadata["objects"][name] = {
                "asset_id": uid,
                "category": category,
                "object_enum": "temp_object",
            }

            possible_types = om.get_possible_object_types(name)
            possible_types = {
                reference.lower().replace("_", " ").strip() for reference in possible_types
            }

            added = False
            for cat_or_synset in categories_and_synsets:
                for possible_type in possible_types:
                    if cat_or_synset in possible_type:
                        valid_uids[uid] = anno
                        added = True
                        break
                if added:
                    break

            om.scene_metadata["objects"].pop(name)
            om._object_name_to_possible_type_names = {}
            om._object_name_and_context_to_source_to_natural_names = {}

    return valid_uids


def _get_cached_valid_pickupables(
    pickupable_synsets_categories_or_uids: Collection[str], split: str
) -> dict[str, dict]:
    """Get cached valid pickupable UIDs filtered by synset rules."""
    global _VALID_PICKUPABLE_CACHE
    if _VALID_PICKUPABLE_CACHE is None:
        _VALID_PICKUPABLE_CACHE = get_valid_pickupable_uids(
            pickupable_synsets_categories_or_uids, split=split
        )
    return _VALID_PICKUPABLE_CACHE


# G1 arm/waist joint names and standalone-model reach envelope, for the
# reset-time reachability precheck (_precheck_grasp_reachable). Duplicated
# from FetchmanPickPlannerPolicy's own copies (rather than imported) to keep
# task-sampling-time code independent of the policy layer -- these are
# G1 MJCF joint names, unlikely to drift out of sync.
_PRECHECK_ARM_JOINTS = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)
_PRECHECK_WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_PRECHECK_HEIGHT_MIN, _PRECHECK_HEIGHT_MAX = 0.35, 0.793
# g1_molmo's own fast-precheck mode ("just smell-test the best candidate")
# uses a single lumped pos+rot error threshold of 0.1 and few iterations --
# it only needs to reject clearly-unreachable spawns, not certify precision.
_PRECHECK_MAX_ITERS = 300
_PRECHECK_ERROR_THRESHOLD = 0.1


class PickTaskSampler(BaseMujocoTaskSampler):
    """
    Default task sampler for pick tasks with house iteration control.
    House order (`house_inds`) and samples per house are provided via config.
    """

    def __init__(self, config: "PickBaseConfig") -> None:
        super().__init__(config)
        self.candidate_objects: None | list[MlSpacesObject] = None
        self._task_counter = None  # Track tasks within the same house for variety
        self._grasp_failure_counts: dict[str, int] = {}  # Track grasp failures per object name

        # If pickup_types is None, default to empty list which matches any object type.
        # Objects are then filtered by grasp file availability in _get_scene_objects().
        if config.task_sampler_config.pickup_types is None:
            config.task_sampler_config.pickup_types = []

        # Added pickup objects state (pick-from-set mode)
        self._added_pickup_obj_name: str | None = None
        self._added_pickup_cache: dict = {}
        self._added_pickup_names: list[str] = []
        self._added_pickup_uids: list[str] = []
        self._current_added_pickup_index: int = 0
        self._episodes_with_current_added_pickup: int = 0
        self._added_pickup_multiplier: int = 1
        self._added_pickup_staging_poses: dict = {}
        self.added_objects: dict = {}
        self._valid_candidate_uids: list[str] | None = None

    def _remove_candidate_object(self, obj_name: str) -> None:
        """Remove an object from candidate_objects list."""
        if self.candidate_objects is not None:
            original_len = len(self.candidate_objects)
            self.candidate_objects = [obj for obj in self.candidate_objects if obj.name != obj_name]
            if len(self.candidate_objects) < original_len:
                log.info(
                    f"Removed {obj_name} from candidates, {len(self.candidate_objects)} remaining"
                )

    def report_grasp_failure(self, obj_name: str, max_failures: int = 2) -> None:
        """Report a grasp failure for an object. Remove from candidates if threshold exceeded.

        Args:
            obj_name: Name of the object that failed grasp finding
            max_failures: Remove object after this many failures (default 2)
        """
        self._grasp_failure_counts[obj_name] = self._grasp_failure_counts.get(obj_name, 0) + 1
        count = self._grasp_failure_counts[obj_name]
        if count > max_failures:
            self._remove_candidate_object(obj_name)
            log.info(f"Removed {obj_name} after {count} grasp failures (threshold: {max_failures})")

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Use this function to put task specific assets into the scene."""
        self.config.policy_config.policy_cls.add_auxiliary_objects(self.config, spec)
        if self.config.task_sampler_config.added_pickup_objects is not None:
            self._add_pickupables_to_scene(spec)

    def init_scene(self, env) -> None:
        # initialize randomizers here
        super().init_scene(env)

        log.info(
            f"Setting up scene for house {self.current_house_index}, task {self._task_counter}..."
        )
        model = env.mj_model
        data = env.mj_datas[0]
        modify_mjmodel_thor_articulated(model, data)

        # New house - reset counters
        self._task_counter = 0
        self._grasp_failure_counts = {}
        log.debug(f"New house {self.current_house_index} - resetting object tracking")

        # Shuffle order deterministically per house/task for variety
        candidate_objects = self._get_scene_objects(env)
        candidate_objects = self.balance_sample_names(candidate_objects)
        np.random.shuffle(candidate_objects)
        self.candidate_objects = candidate_objects

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
            if (
                self.config.robot_config.init_qpos_noise_range is not None
                and group_name in self.config.robot_config.init_qpos_noise_range
            ):
                noise_mag = np.array(self.config.robot_config.init_qpos_noise_range[group_name])
                perturb = np.random.uniform(-noise_mag, noise_mag)
            else:
                perturb = np.zeros_like(qpos)
            robot_view.get_move_group(group_name).joint_pos = qpos + perturb

        # Reset controllers and sync head ctrl with qpos.
        # mj_resetData zeros all ctrl values. For move groups with controllers, reset() re-syncs
        # the ctrl targets. For the head (which has actuators but no controller), we must
        # manually set ctrl = noop_ctrl to prevent the head from snapping to position 0.
        for robot in env.robots:
            for controller in robot.controllers.values():
                controller.reset()
            if "head" in robot.robot_view.move_group_ids():
                head_mg = robot.robot_view.get_move_group("head")
                head_mg.ctrl = head_mg.noop_ctrl

        # robot_color = None
        # robot_color = [.941, .322, .612,1.]  # example: red
        # if robot_color:
        #     # Get robot geometry ids
        #     robot_geoms = descendant_geoms(
        #         self.env._mj_model,
        #         self.env.current_robot.robot_view.base.root_body_id,
        #     )
        #     # Set color
        #     for geom_id in robot_geoms:
        #         model.geom_rgba[geom_id] = robot_color
        log.info("Scene setup completed.\n")

    def get_workspace_center(self, env: CPUMujocoEnv) -> np.ndarray:
        """Workspace center as centroid of task-relevant objects and gripper.

        Collects positions for every object returned by
        :func:`get_task_relevant_objects` that can be found in the environment,
        plus the robot gripper. Falls back to the base implementation (gripper
        only) when the pickup object is not yet set.
        """
        if (
            not hasattr(self.config.task_config, "pickup_obj_name")
            or not self.config.task_config.pickup_obj_name
        ):
            return super().get_workspace_center(env)

        try:
            om = env.object_managers[env.current_batch_index]
            positions: dict[str, np.ndarray] = {}

            for name in get_task_relevant_objects(self.config.task_config):
                obj = om.get_object_by_name(name)
                if obj is not None:
                    positions[name] = obj.position

            # For pure pick tasks the goal pose acts as implicit place target
            if len(positions) == 1 and hasattr(self.config.task_config, "pickup_obj_goal_pose"):
                goal = self.config.task_config.pickup_obj_goal_pose
                if goal is not None:
                    positions["goal_pose"] = np.asarray(goal[:3])

            robot_view = env.current_robot.robot_view
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            ee_pose = (
                robot_view.base.pose @ robot_view.get_move_group(gripper_mg_id).leaf_frame_to_robot
            )
            positions["gripper"] = ee_pose[:3, 3]

            return compute_workspace_center(positions)
        except Exception as e:
            log.debug(f"[CAMERA SETUP] Could not compute workspace center: {e}, using default")
            return super().get_workspace_center(env)

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> list[str]:
        """Resolve special visibility object keys.

        Handles:
        - __task_objects__: Task-relevant objects via shared utility
        - __gripper__: Robot gripper (via base class)
        """
        if key == "__task_objects__":
            return get_task_relevant_objects(self.config.task_config)

        return super().resolve_visibility_object(env, key)

    # ── Added pickup objects (pick-from-set) ──────────────────────────────

    def _add_pickupables_to_scene(self, spec: MjSpec) -> None:
        """Add external pickupable objects to the scene for pick-from-set mode."""
        task_sampler_config = self.config.task_sampler_config

        max_size = np.array([0.25, 0.25, -1])

        def valid_pickupable(anno):
            xyz = [anno["boundingBox"][x] for x in "xyz"]
            return (
                anno["primaryProperty"] == "CanPickup"
                and max_size[0] >= xyz[0]
                and max_size[1] >= xyz[1]
            ) or (anno["assetId"] in task_sampler_config.added_pickup_objects)

        cache_key = "valid_pickupables"
        added_objects = list(task_sampler_config.added_pickup_objects)
        num_pickupables_target = task_sampler_config.num_added_pickups

        # if len(added_objects) > 1000:
        # Random pre-selection before metadata load; skip CLIP for large lists
        # HARD ASSUMPTION: inputs are UIDs. Need to santize this better.
        num_pre_select = min(num_pickupables_target * 3, len(added_objects))
        pre_selected = list(np.random.choice(added_objects, size=num_pre_select, replace=False))
        all_valid = {}
        for uid in pre_selected:
            anno = ObjectMeta.annotation(uid)
            if anno is not None:
                all_valid[uid] = anno
        valid_uids = sorted([uid for uid, anno in all_valid.items() if valid_pickupable(anno)])
        self._added_pickup_cache[cache_key] = {uid: all_valid[uid] for uid in valid_uids}
        # elif cache_key not in self._added_pickup_cache:
        #     all_valid = _get_cached_valid_pickupables(task_sampler_config.added_pickup_objects, split=self.config.data_split)
        #     valid_uids = sorted([uid for uid, anno in all_valid.items() if valid_pickupable(anno)])
        #     valid_uids = ObjectManager.prefilter_with_clip(
        #         list(task_sampler_config.added_pickup_objects), valid_uids
        #     )
        #     valid_uids = sorted(
        #         set(valid_uids)
        #         | (
        #             set(task_sampler_config.added_pickup_objects)
        #             & set(ObjectMeta.annotation().keys())
        #         )
        #     )
        #     self._added_pickup_cache[cache_key] = {uid: all_valid[uid] for uid in valid_uids}
        valid_uids = sorted(self._added_pickup_cache[cache_key].keys())

        if len(valid_uids) == 0:
            raise ValueError("No valid pickupable assets found")

        num_pickupables = min(task_sampler_config.num_added_pickups, len(valid_uids))
        selected_uids = list(np.random.choice(valid_uids, size=num_pickupables, replace=False))

        multiplier = self._added_pickup_multiplier

        self._added_pickup_names = []
        self._added_pickup_uids = []
        self._current_added_pickup_index = 0
        name_to_meta = {}

        staging_size = np.array([num_pickupables, multiplier, 1]) / 2
        staging_center = np.array([5, 5, 25])
        staging_start = staging_center + np.array(
            [0.5 - staging_size[0], 0.5 - staging_size[1], staging_size[2]]
        )

        mocap_body = spec.worldbody.add_body(
            name="pickupable_staging_floor",
            mocap=True,
            pos=staging_center,
        )
        mocap_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=staging_size,
            contype=8,
            conaffinity=15,
            group=4,
        )

        self._added_pickup_staging_poses = {}

        for i, uid in enumerate(selected_uids):
            pickupable_xml = install_uid(uid)
            for j in range(multiplier):
                pickupable_spec = MjSpec.from_file(str(pickupable_xml))
                if len(pickupable_spec.worldbody.bodies) != 1:
                    log.warning(
                        f"{pickupable_xml} has {len(pickupable_spec.worldbody.bodies)} bodies, expected 1."
                    )
                pickupable_obj: mujoco.MjsBody = pickupable_spec.worldbody.bodies[0]

                if not pickupable_obj.first_joint():
                    pickupable_obj.add_joint(
                        name=f"{uid}_copy{j}_jntfree",
                        type=mujoco.mjtJoint.mjJNT_FREE,
                        damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
                    )

                z_shift = self._added_pickup_cache[cache_key][uid]["boundingBox"]["z"] / 2 + 0.01
                position = staging_start + np.array([i, j, z_shift])
                quat = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)

                attach_frame = spec.worldbody.add_frame(pos=position, quat=quat)
                namespace = f"{task_sampler_config.added_pickup_namespace}{i}_{j}/"
                original_body_name = pickupable_obj.name
                attach_frame.attach_body(pickupable_obj, namespace, "")

                full_name = namespace + original_body_name
                self._added_pickup_names.append(full_name)
                self._added_pickup_uids.append(uid)
                self._added_pickup_staging_poses[full_name] = np.concatenate((position, quat))

                xml_path_rel = pickupable_xml.relative_to(ASSETS_DIR)
                self.added_objects[full_name] = xml_path_rel

                uid_anno = self._added_pickup_cache[cache_key][uid]
                name_to_meta[full_name] = {
                    "asset_id": uid,
                    "category": uid_anno["category"],
                    "object_enum": "temp_object",
                    "is_static": False,
                    "boundingBox": uid_anno.get("boundingBox", {}),
                }

        self._added_pickup_obj_name = self._added_pickup_names[0]
        log.info(
            f"Added {num_pickupables} (x {multiplier}) pickupables to scene: "
            f"{self._added_pickup_uids}"
        )

        self._metadata_adder.update(name_to_meta)

    def _advance_to_next_added_pickupable(self, env: CPUMujocoEnv) -> bool:
        """Advance to the next added pickupable (loops around)."""
        multiplier = self._added_pickup_multiplier

        if len(self._added_pickup_names) == 0:
            log.info("No added pickupables available to try")
            return False

        self._current_added_pickup_index = (self._current_added_pickup_index + multiplier) % len(
            self._added_pickup_names
        )
        self._added_pickup_obj_name = self._added_pickup_names[self._current_added_pickup_index]
        self._episodes_with_current_added_pickup = 1
        log.info(
            f"Advanced to pickupable "
            f"{self._current_added_pickup_index // multiplier + 1}/"
            f"{len(self._added_pickup_names) // multiplier}: "
            f"{self._added_pickup_uids[self._current_added_pickup_index]}"
        )
        return True

    @property
    def current_added_pickup_uid(self) -> str | None:
        """Get the UID of the currently active added pickupable."""
        if self._added_pickup_uids and self._current_added_pickup_index < len(
            self._added_pickup_uids
        ):
            return self._added_pickup_uids[self._current_added_pickup_index]
        return None

    @property
    def active_added_pickup_names(self) -> list[str]:
        multiplier = self._added_pickup_multiplier
        return self._added_pickup_names[
            self._current_added_pickup_index : self._current_added_pickup_index + multiplier
        ]

    def _prepare_added_pickupable(
        self,
        env: CPUMujocoEnv,
        reference_obj_name: str,
        reference_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Position the added pickupable near the reference scene object."""
        task_sampler_config = self.config.task_sampler_config
        om = env.object_managers[env.current_batch_index]

        for pickupable_name in self.active_added_pickup_names:
            pickupable_id = om.get_object_body_id(pickupable_name)

            try:
                place_object_near(
                    data=env.current_data,
                    object_id=pickupable_id,
                    placement_point=reference_obj_pos,
                    min_dist=task_sampler_config.min_reference_to_added_pickup_dist,
                    max_dist=task_sampler_config.max_reference_to_added_pickup_dist,
                    max_tries=task_sampler_config.max_added_pickup_placement_attempts,
                    max_dist_to_reference=task_sampler_config.max_robot_to_added_pickup_dist,
                    supporting_geom_id=supporting_geom_id,
                    z_eps=0.003,
                )
            except ObjectPlacementError:
                log.info(f"Failed to place pickupable {pickupable_name} near {reference_obj_name}")
                return False

            r_obj = om.get_object(pickupable_name)
            r_base_pos = body_base_pos(env.current_data, r_obj.body_id)

            if abs(r_base_pos[2] - reference_obj_pos[2]) > MAX_BOTTOM_Z_DIFFERENCE:
                raise ValueError(
                    f"Failed to place pickupable {pickupable_name} at same height as "
                    f"reference object {reference_obj_name}"
                )

        return True

    def _on_candidate_selected(
        self,
        env: CPUMujocoEnv,
        reference_obj_name: str,
        reference_obj_id: int,
        supporting_geom_id: int,
    ) -> bool:
        """Hook called after a valid candidate scene object is found.

        Override in subclasses to inject task-specific preparation (e.g.,
        positioning a place target).  The base implementation handles from-set
        mode (positioning an added pickupable near the reference object).

        Returns True to proceed with robot placement, False to skip this
        candidate.  Raise ``ValueError`` to skip *and* permanently remove
        the candidate from the list.
        """
        from_set_mode = self.config.task_sampler_config.added_pickup_objects is not None
        if not from_set_mode:
            return True

        om = env.object_managers[env.current_batch_index]
        reference_obj_pos = body_base_pos(env.current_data, reference_obj_id)

        if not self._prepare_added_pickupable(
            env, reference_obj_name, reference_obj_pos, supporting_geom_id
        ):
            log.info(
                f"No valid placement for {self._added_pickup_obj_name} near {reference_obj_name}"
            )
            return False

        self.config.task_config.pickup_obj_name = self._added_pickup_obj_name
        pickupable_obj = om.get_object_by_name(self._added_pickup_obj_name)
        self.config.task_config.object_poses = {}
        self.config.task_config.object_poses[self._added_pickup_obj_name] = pose_mat_to_7d(
            pickupable_obj.pose
        ).tolist()
        return True

    def _select_pickup_object(self, env: CPUMujocoEnv) -> int:
        """Run the pickup object selection retry loop.

        Iterates candidate scene objects and for each one:
        1. Finds a supporting surface.
        2. Calls :meth:`_on_candidate_selected` (hook for subclass logic).
        3. Sets up initial cameras, places the robot, checks grasp feasibility.
        4. Sets up final cameras.

        On success, ``self.config.task_config.pickup_obj_name`` holds the
        selected pickup object (which may differ from the scene candidate
        when from-set mode is active or a subclass swaps it).

        Returns:
            The ``supporting_geom_id`` of the surface the reference object
            sits on.

        Raises:
            HouseInvalidForTask: If no valid candidate is found.
        """
        om = env.object_managers[env.current_batch_index]

        from_set_mode = self.config.task_sampler_config.added_pickup_objects is not None
        if from_set_mode:
            self._episodes_with_current_added_pickup += 1
            if (
                self._episodes_with_current_added_pickup
                > self.config.task_sampler_config.episodes_per_added_pickup
            ):
                self._advance_to_next_added_pickupable(env)

        keep_task_cfg = self.config.task_config.pickup_obj_name is not None

        max_attempts = len(self.candidate_objects)
        attempts = 0
        while len(self.candidate_objects) > 0 and attempts < max_attempts:
            attempts += 1

            if not keep_task_cfg:
                self.config.task_config.pickup_obj_name = None

            # Select candidate scene object
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_select_object")

            if self.config.task_config.pickup_obj_name is None:
                object_index = self._task_counter % len(self.candidate_objects)
                self.config.task_config.pickup_obj_name = self.candidate_objects[object_index].name
                log.info(
                    f"Attempting object {self.config.task_config.pickup_obj_name} "
                    f"{object_index}/{len(self.candidate_objects)}"
                )
            else:
                log.info(
                    f"Attempting object {self.config.task_config.pickup_obj_name} "
                    f"of {len(self.candidate_objects)}"
                )

            self._task_counter += 1

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_select_object")

            reference_obj_name = self.config.task_config.pickup_obj_name
            reference_obj_id = om.get_object_body_id(reference_obj_name)

            supporting_geom_id = get_supporting_geom(env.current_data, reference_obj_id)
            if supporting_geom_id is None:
                log.info(f"Failed to get a valid supporting geom_id for {reference_obj_name}")
                self._remove_candidate_object(reference_obj_name)
                continue

            # Hook: subclass-specific preparation (place target, from-set swap, etc.)
            try:
                if not self._on_candidate_selected(
                    env, reference_obj_name, reference_obj_id, supporting_geom_id
                ):
                    continue
            except ValueError:
                log.exception(f"Removing {reference_obj_name}.")
                self._remove_candidate_object(reference_obj_name)
                continue

            pickup_obj_name = self.config.task_config.pickup_obj_name

            # Initial camera setup (for visibility checks during placement)
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_cameras_initial")
            self.setup_cameras(env, deterministic_only=True)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_cameras_initial")

            # Place robot
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_place_robot")
            try:
                self._sample_and_place_robot(env)
            except RobotPlacementError as e:
                log.info(f"Robot placement failed for {pickup_obj_name}: {e}")
                if reference_obj_name == pickup_obj_name:
                    asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
                    if asset_uid:
                        self.report_asset_failure(asset_uid, f"robot placement failed: {e}")
                self._remove_candidate_object(reference_obj_name)
                continue
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_place_robot")

            if not self._check_placement_walk_reachable(env, pickup_obj_name):
                log.info(f"Robot placement not walk-reachable for {pickup_obj_name}")
                self.report_grasp_failure(pickup_obj_name)
                continue

            mujoco.mj_forward(env.current_model, env.current_data)

            # Height randomization (g1_molmo ports) -- must run before the
            # reachability precheck right below, on the same (object,
            # placement) attempt: g1_molmo randomizes support/robot height
            # as part of the same reset attempt it then reachability-checks,
            # not as a separate step after an attempt is already committed.
            self._randomize_target_support_height(env, pickup_obj_name, supporting_geom_id)
            self._randomize_robot_standing_height(env)
            mujoco.mj_forward(env.current_model, env.current_data)

            # Re-capture pickup_obj_start_pose here, AFTER height
            # randomization -- _sample_and_place_robot (above) already set it
            # once, but against the object's pre-randomization pose. Left
            # uncorrected, every downstream lift_height computation
            # (PickTask.get_reward/get_info, PickG1Task) measures against a
            # stale reference: since randomize_height_favored skews the
            # random draw toward the object's own (higher) natural height,
            # the pre-randomization pose is typically well above the actual
            # post-randomization starting height, so lift_height reads
            # spuriously negative even when the object never moved at all
            # after being placed. Matches g1_molmo's own ordering exactly:
            # its env randomizes support height, *then* calls
            # task.init_target_tracking (which sets _target_z0) --
            # never the other way around.
            pickup_obj_for_start_pose = om.get_object_by_name(pickup_obj_name)
            self.config.task_config.pickup_obj_start_pose = pose_mat_to_7d(
                pickup_obj_for_start_pose.pose
            ).tolist()

            # Reset-time grasp-reachability precheck (port of g1_molmo's
            # agent.precheck_grasp / env reset_precheck_grasp=True default):
            # reject this (object, placement) attempt outright if not even
            # the best grasp candidate is plausibly IK-reachable from here,
            # instead of committing to an episode that can only discover
            # this later as a guaranteed-fail rollout during policy
            # execution ("IK failed for pregrasp pose").
            if self.config.task_sampler_config.reset_precheck_grasp:
                pickup_obj_for_precheck = om.get_object_by_name(pickup_obj_name)
                if not self._precheck_grasp_reachable(env, pickup_obj_for_precheck):
                    log.info(f"Reachability precheck failed for {pickup_obj_name}")
                    self.report_grasp_failure(pickup_obj_name)
                    continue

            # Check grasp feasibility
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_check_grasps")

            pickup_obj = om.get_object_by_name(pickup_obj_name)
            asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
            if asset_uid:
                try:
                    grasp_poses_world = get_pickup_grasps(
                        env,
                        pickup_obj,
                        include_flipped=False,
                        grasp_libraries=self.config.task_sampler_config.grasp_libraries,
                    )
                    if len(grasp_poses_world) > 0:
                        noncolliding_mask = get_noncolliding_grasp_mask(
                            env.current_model, env.current_data, grasp_poses_world, 64
                        )
                        n_feasible = int(np.sum(noncolliding_mask))
                        if n_feasible == 0:
                            log.info(
                                f"No feasible grasps for {pickup_obj_name} (uid={asset_uid}): "
                                f"0/{len(grasp_poses_world)} non-colliding"
                            )
                            if self._datagen_profiler is not None:
                                self._datagen_profiler.end("sample_check_grasps")
                            self.report_grasp_failure(pickup_obj_name)
                            continue
                except KeyError:
                    # Grasp collision bodies not in scene (e.g. learned policy instead of planner)
                    log.debug(
                        f"Skipping grasp collision check for {pickup_obj_name} — no collision bodies in scene"
                    )
                except ValueError:
                    log.info(f"No grasps found for {pickup_obj_name} (uid={asset_uid})")
                    if self._datagen_profiler is not None:
                        self._datagen_profiler.end("sample_check_grasps")
                    self.report_grasp_failure(pickup_obj_name)
                    if from_set_mode:
                        self._advance_to_next_added_pickupable(env)
                    continue

            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_check_grasps")

            # Final camera setup
            if self._datagen_profiler is not None:
                self._datagen_profiler.start("sample_cameras_final")
            self.setup_cameras(env)
            if self._datagen_profiler is not None:
                self._datagen_profiler.end("sample_cameras_final")

            if from_set_mode:
                self.config.task_config.added_objects = {
                    self._added_pickup_obj_name: self.added_objects[self._added_pickup_obj_name]
                }

            return supporting_geom_id

        raise HouseInvalidForTask(
            f"Unable to sample a valid task after {attempts} attempts, "
            f"{len(self.candidate_objects)} candidates remaining"
        )

    def _generate_referral_expressions(
        self, env: CPUMujocoEnv, pickup_obj_name: str, context_objects: list
    ) -> tuple[list, list]:
        """Generate referral expressions for the pickup object.

        Args:
            env: The environment.
            pickup_obj_name: Name of the object to generate expressions for.
            context_objects: Objects to consider for disambiguation. The caller
                is responsible for including all relevant objects (pickup object
                itself, place targets, bench neighbours, etc.).

        Returns:
            (expression_priority, filtered_expression_priority)
        """
        om = env.object_managers[env.current_batch_index]

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("generate_context_expressions")

        if self.config.task_sampler_config.referral_expression_clip_filter:
            try:
                expression_priority = om.referral_expression_priority(
                    pickup_obj_name, context_objects
                )
                filtered_expression_priority = om.thresholded_expression_priority(
                    expression_priority
                )
                if len(filtered_expression_priority) == 0:
                    log.info(
                        f"No filtered expression priorities for {pickup_obj_name}, "
                        f"using unfiltered ({len(expression_priority)} expressions)"
                    )
                    filtered_expression_priority = expression_priority
            except ImportError:
                expression_priority = [(1.0, 1.0, om.fallback_expression(pickup_obj_name))]
                filtered_expression_priority = expression_priority
        else:
            expression_priority = [(1.0, 1.0, om.fallback_expression(pickup_obj_name))]
            filtered_expression_priority = expression_priority

        if len(filtered_expression_priority) == 0:
            log.info(f"No expression priorities for {pickup_obj_name}, using fallback")
            expression_priority = [(1.0, 1.0, om.fallback_expression(pickup_obj_name))]
            filtered_expression_priority = expression_priority

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("generate_context_expressions")

        return expression_priority, filtered_expression_priority

    def _sample_task(self, env: CPUMujocoEnv) -> PickTask:
        """Sample a pick task configuration and create the task."""
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        supporting_geom_id = self._select_pickup_object(env)
        pickup_obj_name = self.config.task_config.pickup_obj_name

        om = env.object_managers[env.current_batch_index]

        # Build context and generate referral expressions
        bench_geom_body_id = env.mj_model.geom_bodyid[supporting_geom_id]
        context_objects = om.get_context_objects(
            pickup_obj_name, Context.BENCH, bench_geom_ids=[bench_geom_body_id]
        )
        context_names = {obj.name for obj in context_objects}
        if pickup_obj_name not in context_names:
            context_objects.append(om.get_object(pickup_obj_name))

        expression_priority, filtered_expression_priority = self._generate_referral_expressions(
            env, pickup_obj_name, context_objects
        )

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("sample_context_expressions")

        self.config.task_config.referral_expressions["pickup_obj_name"] = om.sample_expression(
            filtered_expression_priority
        )
        self.config.task_config.referral_expressions_priority["pickup_obj_name"] = (
            expression_priority
        )

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("sample_context_expressions")

        if self._datagen_profiler is not None:
            self._datagen_profiler.start("sample_task_create")

        task = self._task_cls()(env, self.config)

        if self._datagen_profiler is not None:
            self._datagen_profiler.end("sample_task_create")

        return task

    def _task_cls(self) -> type[PickTask]:
        """Hook: task class to instantiate in _sample_task. Override to swap
        in a task with different success/reward semantics -- see
        PickG1Task, which matches g1_molmo's own success criteria instead of
        this class's own (see PickG1Task's docstring for the difference).
        """
        return PickTask

    def _get_scene_objects(self, env: CPUMujocoEnv, mass_limit=100) -> list[MlSpacesObject]:
        """
        Get the list of candidate probjects in the scene for interactions.
        Filter by object types.

        Arguments:
            env: and environment
            mass_limit: don't choose objects with mass greater than limit
            oversample_obja: oversample objaverse assets by factor n
        """
        # Discover candidate pickup objects
        om = env.object_managers[env.current_batch_index]
        candidates = om.get_objects_of_type(self.config.task_sampler_config.pickup_types)
        log.info(f"Found {len(candidates)} candidate pickup objects in the scene")

        if not len(candidates) > 0:
            log.info("[TASK SAMPLING] ⚠️ No candidate pickup objects found in the scene")
            # print all the top-level objects in the scene for debugging
            om = env.object_managers[env.current_batch_index]
            all_objects = MlSpacesObject.get_top_level_bodies(model=self.env.mj_model)
            all_non_structural_or_excluded = [
                obj for obj in all_objects if not (om.is_structural(obj) or om.is_excluded(obj))
            ]
            for b in all_non_structural_or_excluded[:30]:
                name = self.env.mj_model.body(b).name
                pos = self.env.current_data.xpos[b]
                possible_types = om.get_possible_object_types(name or "")
                log.info(
                    f"  - #{b:02d} {name} (types={possible_types}) pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

            # log.info(f"[TASK SAMPLING] Scene objects (no candidates): {[obj.name for obj in all_objects]}")
            raise HouseInvalidForTask("No pickup candidates found in the scene")

        model = self._env.current_model

        # #Mass computation setup
        children = {}
        for bid in range(model.nbody):
            pid = model.body_parentid[bid]  # ID of parent
            children.setdefault(pid, []).append(bid)

        def get_children(body_id):
            return children.get(body_id, [])

        candidate_objects = []
        blacklisted_count = 0
        for pickup_obj in candidates:
            # Rule for added objects
            parts = pickup_obj.name.split("/")
            if len(parts) == 3 and len(parts[1].split("_")) == 2:
                log.info(f"Skipping possibly added object {pickup_obj.name}")
                continue

            # Check if grasp files exist for this object
            asset_uid = None

            if not isinstance(pickup_obj, MlSpacesArticulationObject) and not om.has_free_joint(
                pickup_obj
            ):
                log.info(f"Skipping {pickup_obj.name} (uid={asset_uid}) - static in scene")
                continue

            scene_metadata = env.current_scene_metadata
            if scene_metadata is not None:
                asset_uid = (
                    scene_metadata.get("objects", {}).get(pickup_obj.name, {}).get("asset_id", None)
                )

            if asset_uid is None:
                asset_uid = get_thor_name(model, pickup_obj)

            # Check if asset is blacklisted (static or dynamic)
            if asset_uid and self.is_asset_blacklisted(asset_uid):
                log.debug(f"Skipping {pickup_obj.name} (uid={asset_uid}) - blacklisted")
                blacklisted_count += 1
                continue

            if self.config.task_sampler_config.filter_for_grasps:
                if not self._has_grasps(pickup_obj, asset_uid):
                    continue

            # Mass computation
            masses = [model.body_mass[bid] for bid in get_children(pickup_obj.object_id)]
            if np.sum(masses) > mass_limit:
                continue

            candidate_objects.append(pickup_obj)

        if blacklisted_count > 0:
            log.info(f"Skipped {blacklisted_count} blacklisted objects")
        log.info(
            f"Filtered to {len(candidate_objects)} valid candidate pickup objects, {len(candidate_objects) / len(candidates) * 100:.1f} %"
        )

        return candidate_objects

    def _has_grasps(self, pickup_obj: MlSpacesObject, asset_uid: str):
        if not has_pickup_grasp_path(
            asset_uid,
            grasp_libraries=self.config.task_sampler_config.grasp_libraries,
        ):
            log.info(f"Skipping {pickup_obj.name} (uid={asset_uid}) - no grasp file available")
            return False

        if not has_valid_pickup_grasps(
            asset_uid,
            grasp_libraries=self.config.task_sampler_config.grasp_libraries,
            num_grasps=1,
        ):
            log.info(
                f"Skipping {pickup_obj.name} (uid={asset_uid}) - grasp file exists but has no valid transforms"
            )
            return False
        return True

    def _randomize_target_support_height(
        self, env: CPUMujocoEnv, pickup_obj_name: str, supporting_geom_id: int
    ) -> None:
        """Port of g1_molmo's env._randomize_target_support_height: move the
        pickup object's supporting surface (and the object with it) to a
        randomized height, so the pickup object doesn't always sit at
        whatever height it happened to be authored at.

        The sampled height is drawn from a triangular distribution over
        [randomize_height_min, upper] (upper = the object's own current
        height, further capped by randomize_height_max if set) with mode
        randomize_height_favored -- for most objects (whose natural height is
        below the g1_molmo-matching default of 0.95m), the mode ends up
        clipped to the object's own current height, so most draws land near
        the unmodified default and only occasionally go much lower.

        Scoped down from g1_molmo's reference implementation, which also (a)
        traces through multiple stacked free-jointed supports, (b) unions in
        contact-graph neighbors for edge-perched objects a pure ancestor walk
        misses, and (c) cascades the move to furniture sitting underneath the
        support when lowering. This handles the common case of a single
        object resting on one fixed (non-free-jointed) support -- the large
        majority of pick targets.
        """
        cfg = self.config.task_sampler_config
        if not cfg.randomize_height:
            return
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(pickup_obj_name)
        if getattr(pickup_obj, "is_articulated", False):
            return

        model, data = env.current_model, env.current_data
        sup_root = int(model.geom_bodyid[supporting_geom_id])
        for _ in range(10):
            if int(model.body_parentid[sup_root]) == 0:
                break
            sup_root = int(model.body_parentid[sup_root])
        else:
            return
        if sup_root == 0:
            return

        sup_top_z = float(pickup_obj.position[2])
        upper = sup_top_z
        if cfg.randomize_height_max is not None:
            upper = min(upper, cfg.randomize_height_max)
        if upper <= cfg.randomize_height_min:
            return

        mode = float(np.clip(cfg.randomize_height_favored, cfg.randomize_height_min, upper))
        new_top = float(np.random.triangular(cfg.randomize_height_min, mode, upper))
        dz = new_top - sup_top_z
        if abs(dz) < 1e-3:
            return

        # Static support furniture moves via body_pos -- MuJoCo's "simple"/
        # "sameframe" compile-time flags can otherwise cache a stale
        # transform for a body with a fixed offset from its parent, silently
        # ignoring this change.
        model.body_simple[sup_root] = 0
        model.body_sameframe[sup_root] = 0
        model.body_pos[sup_root, 2] += dz

        # The pickup object itself moves via its own free joint's qpos.
        body_id = om.get_object_body_id(pickup_obj_name)
        jnt_adr = int(model.body_jntadr[body_id])
        assert jnt_adr >= 0 and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE, (
            f"{pickup_obj_name} has no free joint to reposition for height randomization"
        )
        qposadr = int(model.jnt_qposadr[jnt_adr])
        data.qpos[qposadr + 2] += dz

        mujoco.mj_forward(model, data)
        log.info(
            f"[HEIGHT RANDOMIZATION] {pickup_obj_name}: support surface "
            f"{sup_top_z:.3f}m -> {new_top:.3f}m (dz={dz:+.3f}m)"
        )

    def _randomize_robot_standing_height(self, env: CPUMujocoEnv) -> None:
        """Port of g1_molmo's env randomize_robot_height: draw a uniform
        random initial WBC height command for the robot each episode instead
        of always starting at the controller's fixed default (0.74m).
        g1_molmo only applies this when the robot spawns already at its
        final grasp position (spawn_at_grasp); applied unconditionally here.
        No-op for robots without a legs_waist WBC controller (i.e. anything
        but G1 in its default walking mode).
        """
        cfg = self.config.task_sampler_config
        if not cfg.randomize_robot_height:
            return
        controller = env.current_robot.controllers.get("legs_waist")
        if controller is None or not hasattr(controller, "set_target"):
            return
        height = float(
            np.random.uniform(cfg.randomize_robot_height_min, cfg.randomize_robot_height_max)
        )
        controller.set_target(np.array([0.0, 0.0, 0.0, height, 0.0, 0.0, 0.0], dtype=np.float32))
        log.info(f"[ROBOT HEIGHT RANDOMIZATION] init height -> {height:.3f}m")

    def _ensure_ik_precheck_setup(self, env: CPUMujocoEnv) -> None:
        """Lazily build a standalone, robot-only mink model for
        _precheck_grasp_reachable, cached on this (long-lived, spans many
        houses) task sampler instance. Mirrors FetchmanPickPlannerPolicy.
        _ensure_mink_setup's standalone-model approach (see that method's
        docstring for why: solving on the live scene model, with its
        free joint per movable object, is ~2600x slower than a standalone
        ~35-DOF robot-only model), but kept independent of the policy layer
        since this runs during task sampling, before a policy exists.
        """
        # Deferred: mink is an optional dependency (see pyproject.toml's `mink`
        # extra) needed only for this G1-only reset_precheck_grasp path -- a
        # module-level import here would make it a hard dependency for every
        # caller of this module, which is imported unconditionally by
        # base_pick_config.py and therefore nearly every config/test in the repo.
        import mink

        if getattr(self, "_precheck_mink_cfg", None) is None:
            robot_config = env.current_robot.exp_config.robot_config
            model = mujoco.MjModel.from_xml_path(str(robot_config.get_robot_xml_path()))
            self._precheck_mink_model = model
            self._precheck_mink_cfg = mink.Configuration(model)

            def jid(name):
                return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

            self._precheck_arm_dofadr = np.array(
                [model.jnt_dofadr[jid(f"right_{n}")] for n in _PRECHECK_ARM_JOINTS]
            )
            self._precheck_waist_dofadr = np.array(
                [model.jnt_dofadr[jid(n)] for n in _PRECHECK_WAIST_JOINTS]
            )
            fj_id = jid("floating_base_joint")
            self._precheck_fj_dofadr = model.jnt_dofadr[fj_id]
            self._precheck_fj_qposadr = model.jnt_qposadr[fj_id]
            # A posture task is not optional here, even for a "just check
            # reachability" precheck: without one, the null space of this
            # redundant (7 arm + 3 waist + 1 height = 11 DOF for a 6-DOF
            # target) IK has no preferred direction at all, and the QP
            # solver can wander/stall inside it well short of
            # _PRECHECK_MAX_ITERS. Confirmed empirically: omitting this
            # entirely (an earlier bug in this method) produced huge,
            # inconsistent errors (0.1-1.4) across essentially every
            # candidate in every house tested, unlike
            # FetchmanPickPlannerPolicy's real solve (which always includes
            # one) converging to within 0.02-0.05 on comparable poses.
            posture_cost = np.full(model.nv, 0.1)
            posture_cost[self._precheck_waist_dofadr] = 0.2
            posture_cost[self._precheck_fj_dofadr + 2] = 0.1
            self._precheck_posture_cost = posture_cost
            self._precheck_synced_scene_model = None

        # The scene model changes identity every time the task sampler moves
        # to a new house -- rebuild the (cheap, ~35-joint) sync pairs
        # whenever that happens rather than caching them forever against a
        # since-replaced scene.
        if env.current_model is not self._precheck_synced_scene_model:
            model = self._precheck_mink_model
            scene_model = env.current_model
            sync_pairs = []
            for sjid in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, sjid)
                cjid = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_0/{name}")
                if cjid < 0:
                    continue
                ndim = 7 if model.jnt_type[sjid] == mujoco.mjtJoint.mjJNT_FREE else 1
                sync_pairs.append((model.jnt_qposadr[sjid], scene_model.jnt_qposadr[cjid], ndim))
            self._precheck_sync_pairs = sync_pairs
            self._precheck_synced_scene_model = scene_model

    def _precheck_grasp_reachable(self, env: CPUMujocoEnv, pickup_obj: MlSpacesObject) -> bool:
        """Cheap IK feasibility check for the grasp pose only (no
        pregrasp/lift), mirroring g1_molmo's agent.precheck_grasp /
        GraspPolicy.plan(_fast_precheck=True): reject this (object,
        placement) attempt if none of the top-ranked grasp candidates are
        plausibly reachable, instead of committing to an episode that can
        only discover this later as a guaranteed-fail rollout during policy
        execution. Tries several candidates (not just the single best, as
        g1_molmo's own fast-precheck mode does) since without that
        cushion, an otherwise-fine spawn is one bad top-ranked candidate
        away from a false rejection -- FetchmanPickPlannerPolicy's real
        execution-time planning also falls through multiple candidates for
        exactly this reason (see its grasp_candidates_to_try). Also applies
        the same 180-degree roll-flip disambiguation the real policy uses
        (_closer_roll_flip) -- skipping it was an earlier bug in this
        method that made otherwise-reachable candidates look unreachable
        purely from an avoidable orientation mismatch.

        G1 only -- returns True (don't block) for any robot without a
        legs_waist WBC controller, or if no grasp data exists to check
        against.
        """
        if env.current_robot.controllers.get("legs_waist") is None:
            return True
        try:
            candidate_grasps = get_pickup_grasps(
                env, pickup_obj, grasp_libraries=self.config.task_sampler_config.grasp_libraries
            )
        except (KeyError, ValueError):
            return True
        if len(candidate_grasps) == 0:
            return True

        # Deferred: see _ensure_ik_precheck_setup's docstring -- mink is an
        # optional dependency, only pulled in once we know this attempt is
        # actually G1 and has candidates worth IK-checking.
        import mink

        try:
            top_grasps = select_grasp_pose(
                env,
                candidate_grasps,
                pickup_obj.pose,
                check_collision=True,
                n_collision_checks=512,
                collision_batch_size=64,
                check_ik=False,
                n_ik_checks=0,
                ik_batch_size=0,
                # Same orientation preference as FetchmanPickPlannerPolicyConfig's
                # defaults (see that config's grasp_vertical/horizontal_cost_weight
                # comment) -- the precheck should reject/accept based on the same
                # candidates the real policy would actually attempt.
                vertical_cost_weight=0.0,
                horizontal_cost_weight=2.0,
                top_k=5,
            )
        except ValueError:
            # No non-colliding candidate at all -- the existing collision
            # -based feasibility check right after this call already
            # handles rejecting this attempt for that reason.
            return True
        if top_grasps.ndim == 2:
            top_grasps = top_grasps[None]

        self._ensure_ik_precheck_setup(env)
        data = env.current_data
        model_prefix = "robot_0/"
        site_id = mujoco.mj_name2id(
            env.current_model, mujoco.mjtObj.mjOBJ_SITE, f"{model_prefix}right_grasp"
        )
        current_rot = R.from_matrix(data.site_xmat[site_id].reshape(3, 3))

        for grasp_pose_world in top_grasps:
            original_rot = R.from_matrix(grasp_pose_world[:3, :3])
            flipped_rot = original_rot * R.from_euler("z", np.pi)
            if (current_rot.inv() * flipped_rot).magnitude() < (
                current_rot.inv() * original_rot
            ).magnitude():
                grasp_pose_world = grasp_pose_world.copy()
                grasp_pose_world[:3, :3] = flipped_rot.as_matrix()

            config = self._precheck_mink_cfg
            q = config.q
            for s_adr, c_adr, ndim in self._precheck_sync_pairs:
                q[s_adr : s_adr + ndim] = data.qpos[c_adr : c_adr + ndim]
            config.update(q)

            mask = np.zeros(config.model.nv)
            mask[self._precheck_arm_dofadr] = 1.0
            mask[self._precheck_waist_dofadr] = 1.0
            mask[self._precheck_fj_dofadr + 2] = 1.0

            frame_task = mink.FrameTask(
                frame_name="right_grasp",
                frame_type="site",
                position_cost=100,
                orientation_cost=1,
                lm_damping=1,
            )
            rot = mink.SO3.from_matrix(grasp_pose_world[:3, :3])
            frame_task.set_target(
                mink.SE3.from_rotation_and_translation(rot, np.asarray(grasp_pose_world[:3, 3]))
            )
            posture_task = mink.PostureTask(config.model, cost=self._precheck_posture_cost)
            posture_task.set_target_from_configuration(config)
            posture_task.target_q[self._precheck_fj_qposadr + 2] = _PRECHECK_HEIGHT_MAX
            limits = [mink.ConfigurationLimit(config.model)]

            err = float("inf")
            for _ in range(_PRECHECK_MAX_ITERS):
                try:
                    vel = mink.solve_ik(
                        config,
                        [frame_task, posture_task],
                        1e-2,
                        "daqp",
                        damping=1e-1,
                        limits=limits,
                    )
                except Exception:
                    break
                vel = vel * mask
                config.integrate_inplace(vel, 1e-2)
                q = config.q.copy()
                q[self._precheck_fj_qposadr + 2] = np.clip(
                    q[self._precheck_fj_qposadr + 2], _PRECHECK_HEIGHT_MIN, _PRECHECK_HEIGHT_MAX
                )
                config.update(q)
                raw_err = frame_task.compute_error(config)
                err = float(np.linalg.norm(raw_err[:3]) + np.linalg.norm(raw_err[3:]))
                if err < _PRECHECK_ERROR_THRESHOLD:
                    break

            if err < _PRECHECK_ERROR_THRESHOLD:
                return True

        return False

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Sample a pickup object and receptacle, place robot using occupancy map, and return sampled params.

        Returns:
            dict with keys: pickup_obj_name, receptacle_name, placement_region, robot_base_pose

        Raises:
            RobotPlacementError
        """
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        log.debug(f"Selected pickup object: {self.config.task_config.pickup_obj_name}")
        log.debug(f"[TASK SAMPLING] Trying to place robot near '{pickup_obj.name}'")

        # randomize pickup object
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
        if isinstance(pickup_obj, MlSpacesObject):
            target_pos = pickup_obj.position
        else:
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        fixed_base_height = self.config.robot_config.fixed_base_height
        if fixed_base_height is not None:
            # This robot's base height is held constant by its own controller
            # regardless of placement (see BaseRobotConfig.fixed_base_height) --
            # deriving a target-relative spawn height below would place it
            # somewhere physics/the controller immediately corrects away from,
            # silently invalidating any grasp/reach poses planned against it.
            initial_robot_z = fixed_base_height
        else:
            initial_robot_z = (
                target_pos[2]
                + self.config.task_sampler_config.robot_object_z_offset
                + np.random.uniform(
                    self.config.task_sampler_config.robot_object_z_offset_random_min,
                    self.config.task_sampler_config.robot_object_z_offset_random_max,
                )
            )

        # place robot near receptacle - this is the expensive call with collision/visibility checks
        if self._datagen_profiler is not None:
            self._datagen_profiler.start("robot_place_near")
        robot_placed = env.place_robot_near(
            robot_view=robot_view,
            target=pickup_obj,
            max_tries=self.config.task_sampler_config.max_robot_placement_attempts,
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
            log.info(f"[TASK SAMPLING] Failed to place robot near '{pickup_obj.name}'")
            raise RobotPlacementError(f"Failed to place robot near object: {pickup_obj.name}")

        # Add successful position to cache
        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])

        # Get final robot pose for return data
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()

        pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose)
        pickup_obj_goal_pose[2] += 0.05  # 5 cm
        task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

        log.info(f"Supporting receptacle: {self.config.task_config.receptacle_name}")

    def _check_placement_walk_reachable(self, env: CPUMujocoEnv, pickup_obj_name: str) -> bool:
        """Hook: verify the just-placed robot can actually reach a walk
        standoff point near the pickup object, using the same nav-goal-
        sampling + A*/line-of-sight machinery a walk-phase policy would use
        at reset() time -- rejecting an unreachable placement here (which
        retries with a freshly, independently sampled robot position) rather
        than only discovering the mismatch during policy.reset(), after the
        rest of this attempt's setup (height randomization, camera setup,
        grasp feasibility) has already been paid for.

        Robot placement (this class's own occupancy-map-based sampling) and
        walk-goal sampling (a policy's separate NavGoalSampler/AStarPlanner)
        are otherwise two independent computations with no guarantee of
        agreeing on what's reachable from what -- g1_molmo's own env avoids
        this class of problem entirely by computing a single standoff point
        and spawning the robot directly there, so start==goal by
        construction. We don't share that architecture (this class's
        placement supports considerations -- visibility, exclusion zones --
        a walk-goal sampler doesn't), so this hook instead validates
        consistency after the fact rather than guaranteeing it up front.

        No-op by default (True) -- meaningful only for configs with an
        actual walk phase; see PickG1TaskSampler's override.
        """
        return True

    def _place_target_near_object(
        self, env: CPUMujocoEnv, object_pos: np.ndarray, placement_region=None
    ) -> None:
        """Place the placement target on the same receptacle as the pickup object."""
        log.debug(
            f"[TARGET POSITIONING] Finding receptacle for pickup object '{self.config.task_sampler_config.pickup_obj_name}'"
        )

        # Get the pickup object using Object class
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_sampler_config.pickup_obj_name)

        support_name = None
        if placement_region is None:
            support_name = pickup_obj.get_support_below()
            if support_name is None:
                log.debug(
                    "[TARGET POSITIONING] ⚠️ No support found for pickup object, falling back to simple positioning"
                )

                # Fallback to original simple positioning if no support found
                offset_distance = 0.3
                offset_angle = np.random.uniform(0, 2 * np.pi)
                target_x = object_pos[0] + offset_distance * np.cos(offset_angle)
                target_y = object_pos[1] + offset_distance * np.sin(offset_angle)
                target_z = np.clip(object_pos[2], 0.7, 1.2)
                place_target = create_mlspaces_body(
                    env.current_data, self.config.task_sampler_config.place_target_name
                )
                place_target.position = [target_x, target_y, target_z]
                return

            log.debug(f"[TARGET POSITIONING] Pickup object is on receptacle: '{support_name}'")

            om = env.object_managers[env.current_batch_index]
            receptacle_obj = om.get_object_by_name(support_name)
            placement_region = receptacle_obj.compute_placement_region()

        xy_min = placement_region["xy_min"]
        xy_max = placement_region["xy_max"]
        top_z = placement_region["top_z"]

        log.debug(
            f"[TARGET POSITIONING] Receptacle placement region: xy_min=({xy_min[0]:.3f}, {xy_min[1]:.3f}), xy_max=({xy_max[0]:.3f}, {xy_max[1]:.3f}), top_z={top_z:.3f}"
        )

        # Apply minimum separation constraint from pickup object
        max_attempts = 50
        min_separation = self.config.task_sampler_config.min_object_separation
        pickup_xy = pickup_obj.position[:2]

        for attempt in range(max_attempts):
            # Sample a random position within the receptacle's placement region
            target_x = np.random.uniform(xy_min[0], xy_max[0])
            target_y = np.random.uniform(xy_min[1], xy_max[1])
            target_xy = np.array([target_x, target_y])

            # Check minimum separation from pickup object
            separation = np.linalg.norm(target_xy - pickup_xy)
            if separation >= min_separation:
                break

            log.debug(
                f"[TARGET POSITIONING]   Attempt {attempt + 1}: separation {separation:.3f}m < {min_separation:.3f}m, retrying..."
            )

        else:
            # If we couldn't find a position with minimum separation, use the last attempt
            log.debug(
                f"[TARGET POSITIONING] ⚠️ Could not achieve minimum separation of {min_separation:.3f}m after {max_attempts} attempts, using separation {separation:.3f}m"
            )

        # Position target slightly above the receptacle surface
        target_z = top_z + 0.01  # 1cm above surface to avoid z-fighting/embedding

        # Set placement target position
        place_target = create_mlspaces_body(
            env.current_data, self.config.task_sampler_config.place_target_name
        )
        place_target.position = [target_x, target_y, target_z]
        mujoco.mj_forward(env.current_model, env.current_data)

        distance_to_pickup = np.linalg.norm(
            np.array([target_x, target_y, target_z]) - pickup_obj.position
        )
        log.debug(
            f"[TARGET POSITIONING] Positioned target at ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
        )
        log.debug(f"[TARGET POSITIONING] Distance to pickup object: {distance_to_pickup:.3f}m")
        log.debug(
            f"[TARGET POSITIONING] XY separation: {separation:.3f}m (min: {min_separation:.3f}m)"
        )
        if support_name:
            log.debug(
                f"[TARGET POSITIONING] Target on receptacle '{support_name}' at height {target_z:.3f}m"
            )

    @staticmethod
    def add_placement_target(
        spec: MjSpec, pos=None, randomize=False, name="place_target"
    ) -> MjSpec:
        """
        Add a placement target (red cylinder) to the scene.

        Args:
            spec: MuJoCo MjSpec object
            pos: [x, y, z] position. If None, uses default or random
            randomize: Whether to randomize the position
            name: Name for the target body

        Returns:
            spec: Updated MjSpec with placement target added
        """

        if pos is None:
            if randomize:
                # Random position on table surface (approximate table bounds)
                pos = [
                    np.random.uniform(-0.4, 0.4),  # x: table width
                    np.random.uniform(0.2, 1.0),  # y: table depth
                    0.71,  # z: table height
                ]
            else:
                pos = [-0.1, 0.4, 0.71]  # Default position from XML

        # Create target body
        target_body = spec.worldbody.add_body(name=name, pos=pos, mocap=True)

        # Add red cylinder geometry
        target_body.add_geom(
            name=f"{name}_geom",  # Add geometry name for identification
            type=mjtGeom.mjGEOM_CYLINDER,
            size=[0.05, 0.001, 0],  # For cylinders in Python API: [radius, radius, half-height]
            rgba=[1, 0, 0, 1],  # Red color
            group=2,  # Visual group
        )

        return spec

    @staticmethod
    def add_pickup_target(
        spec: MjSpec, pos=None, randomize=False, name="obj_0", color=[0, 1, 0, 1]
    ) -> MjSpec:
        """
        Add a pickup target (cube) to the scene.

        Args:
            spec: MuJoCo MjSpec object
            pos: [x, y, z] position. If None, uses default or random
            randomize: Whether to randomize the position
            name: Name for the object body
            color: RGBA color for the cube

        Returns:
            spec: Updated MjSpec with pickup object added
        """
        if pos is None:
            if randomize:
                pos = [np.random.uniform(-0.4, 0.4), np.random.uniform(0.2, 1.0), 0.735]
            else:
                pos = [6.0, 3.0, 0.76]

        obj_body = spec.worldbody.add_body(name=name, pos=pos)
        obj_body.add_freejoint()
        obj_body.add_geom(
            name=f"{name}_geom",
            type=mjtGeom.mjGEOM_BOX,
            pos=[0, 0, 0.025],
            size=[0.025, 0.025, 0.025],
            rgba=color,
        )

        return spec
