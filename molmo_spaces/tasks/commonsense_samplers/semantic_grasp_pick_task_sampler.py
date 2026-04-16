import functools
import logging

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.commonsense_tasks.semantic_grasp_pick_task import SemanticGraspPickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.grasp_sample import has_valid_grasp_file

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=10000)
def has_grasp_classification_file(asset_id: str) -> bool:
    """Check if a grasp classification file exists for the given asset."""
    path = ASSETS_DIR / f"grasps/droid/{asset_id}/{asset_id}_grasp_classifications.json"
    return path.exists()


class SemanticGraspPickTaskSampler(PickTaskSampler):
    """Task sampler for the semantic grasp pick task.

    Extends PickTaskSampler by:
    1. Filtering candidate objects to only those with grasp classification files.
    2. Creating SemanticGraspPickTask instances with loaded classification data.
    """

    def _get_scene_objects(self, env: CPUMujocoEnv, mass_limit=100) -> list[MlSpacesObject]:
        """Get candidate objects, filtered to those with grasp classification data."""
        candidates = super()._get_scene_objects(env, mass_limit=mass_limit)

        filtered = []
        for obj in candidates:
            asset_uid = self.get_asset_uid_from_object(env, obj.name)
            if asset_uid is None:
                from molmo_spaces.utils.asset_names import get_thor_name

                asset_uid = get_thor_name(env.current_model, obj)

            if asset_uid and has_grasp_classification_file(asset_uid):
                filtered.append(obj)
            else:
                log.debug(f"Skipping {obj.name} (uid={asset_uid}) - no grasp classification file")

        log.info(
            f"Filtered to {len(filtered)}/{len(candidates)} objects with grasp classifications"
        )
        return filtered

    def has_valid_grasp_file(self, pickup_obj, asset_uid):
        """Override to also require a grasp classification file."""
        if not has_valid_grasp_file(asset_uid):
            return False
        return has_grasp_classification_file(asset_uid)

    def _sample_task(self, env: CPUMujocoEnv) -> SemanticGraspPickTask:
        """Sample a task, then swap in a SemanticGraspPickTask with classification data."""
        # Let parent do all the heavy lifting (object selection, robot placement, etc.)
        _parent_task = super()._sample_task(env)

        # Create our task using the same config (already populated by the parent)
        task = SemanticGraspPickTask(env, self.config)

        # Look up the asset_uid for the selected pickup object
        pickup_obj_name = self.config.task_config.pickup_obj_name
        asset_uid = self.get_asset_uid_from_object(env, pickup_obj_name)
        if asset_uid is None:
            from molmo_spaces.utils.asset_names import get_thor_name

            pickup_obj = MlSpacesObject(data=env.current_data, object_name=pickup_obj_name)
            asset_uid = get_thor_name(env.current_model, pickup_obj)

        # Load grasp classifications — skip scene if file is missing
        if not has_grasp_classification_file(asset_uid):
            raise HouseInvalidForTask(
                f"No grasp classification file for {pickup_obj_name} (uid={asset_uid}), skipping scene"
            )

        task.load_grasp_classifications(asset_uid)

        log.info(f"[SEMANTIC GRASP PICK] Task created for {pickup_obj_name} (uid={asset_uid})")
        return task
