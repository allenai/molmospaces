import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.grasps import load_pickup_grasps
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)

# Gripper dimensions (matching classify_grasps.py / view_grasps.py)
GRASP_WIDTH = 0.088
GRASP_LENGTH = 0.12
GRASP_BASE_POS = np.array([0.0, 0.0, -0.06])


def _compute_tcp_points(grasp_transforms: np.ndarray) -> np.ndarray:
    """Compute the tool center point (midpoint between fingertips) for each grasp.

    In the local TCP frame the fingertip midpoint is at
    GRASP_BASE_POS + [0, 0, GRASP_LENGTH/2]. We transform that by each
    grasp pose to get object-frame coordinates.
    """
    local_tcp = GRASP_BASE_POS + np.array([0.0, 0.0, GRASP_LENGTH / 2])
    local_h = np.array([local_tcp[0], local_tcp[1], local_tcp[2], 1.0])
    world_h = grasp_transforms @ local_h
    return world_h[:, :3]


class SemanticGraspPickTask(PickTask):
    """Pick task that additionally checks whether the grasp is semantically correct.

    A semantically correct grasp means the robot grasps the object at a
    functionally appropriate location (e.g., a pan by its handle, not its
    surface). This is determined by comparing the executed grasp pose against
    pre-classified grasp data using a KNN majority vote.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.grasp_transforms: np.ndarray | None = None
        self.grasp_classifications: np.ndarray | None = None
        self.grasp_tcp_points: np.ndarray | None = None
        self._asset_id: str | None = None
        self._vis_counter: int = 0

        pickup_obj_name = config.task_config.pickup_obj_name
        asset_uid = (
            (env.current_scene_metadata or {})
            .get("objects", {})
            .get(pickup_obj_name, {})
            .get("asset_id")
        )
        if asset_uid is None:
            from molmo_spaces.utils.asset_names import get_thor_name

            pickup_obj = MlSpacesObject(data=env.current_data, object_name=pickup_obj_name)
            asset_uid = get_thor_name(env.current_model, pickup_obj)
        self.load_grasp_classifications(asset_uid)

    def load_grasp_classifications(self, asset_id: str) -> None:
        """Load grasp classification data for the given asset.

        Loads:
        - All grasp transforms from the NPZ file (via load_pickup_grasps)
        - The classification JSON that labels each grasp as good/bad

        The classification JSON keys are string indices "0", "1", "2"...
        matching the order of grasps in the NPZ file.
        """
        self._asset_id = asset_id

        # Load all grasps (use large num_grasps to get them all)
        # num_grasps large enough to return every transform in file order -- the
        # classification JSON is keyed by positional index, so order must match.
        grasps = load_pickup_grasps(asset_id, num_grasps=int(1e6))
        self.grasp_transforms = np.array(grasps)

        # Compute TCP points in object frame for distance calculations
        self.grasp_tcp_points = _compute_tcp_points(self.grasp_transforms)

        # Find and load the classification JSON
        classification_paths = [
            ASSETS_DIR / f"grasps/droid/{asset_id}/{asset_id}_grasp_classifications.json",
        ]

        classifications_data = None
        for path in classification_paths:
            if path.exists():
                with open(path, "r") as f:
                    classifications_data = json.load(f)
                log.info(f"Loaded grasp classifications from {path}")
                break

        if classifications_data is None:
            raise FileNotFoundError(
                f"No grasp classification file found for {asset_id}. "
                f"Searched: {[str(p) for p in classification_paths]}"
            )

        # Build boolean array aligned with grasp transforms
        good_grasp_map = classifications_data.get("semantically_good_grasp", {})
        n_grasps = len(self.grasp_transforms)
        self.grasp_classifications = np.zeros(n_grasps, dtype=bool)
        for i in range(n_grasps):
            self.grasp_classifications[i] = good_grasp_map.get(str(i), False)

        n_good = int(self.grasp_classifications.sum())
        log.info(
            f"Loaded {n_grasps} grasp classifications for {asset_id}: "
            f"{n_good} good, {n_grasps - n_good} bad"
        )

    def classify_current_grasp(self) -> bool:
        """Classify the current gripper grasp as semantically good or bad.

        Uses KNN majority vote: finds the k nearest classified grasps (by
        position in object frame) and returns True if the majority are "good".
        """
        if self.grasp_transforms is None or self.grasp_classifications is None:
            log.warning("Grasp classifications not loaded, defaulting to False")
            return False

        k = getattr(self.config.task_config, "k_nearest_grasps", 5)

        # Get current TCP pose in world frame
        tcp_pose_arr = self.sensor_suite.sensors["tcp_pose"].get_observation(self._env, self)
        tcp_pose = pos_quat_to_pose_mat(tcp_pose_arr[:3], tcp_pose_arr[3:7])
        robot_view = self._env.current_robot.robot_view
        tcp_world = robot_view.base.pose @ tcp_pose

        # Get pickup object pose
        pickup_obj = MlSpacesObject(
            data=self._env.current_data,
            object_name=self.config.task_config.pickup_obj_name,
        )
        object_pose = pos_quat_to_pose_mat(pickup_obj.position, pickup_obj.quat)

        # Compute current grasp in object frame
        grasp_in_obj = np.linalg.inv(object_pose) @ tcp_world

        # Compute TCP point for the current grasp in object frame
        local_tcp = GRASP_BASE_POS + np.array([0.0, 0.0, GRASP_LENGTH / 2])
        current_tcp = (grasp_in_obj @ np.array([*local_tcp, 1.0]))[:3]

        # Compute L2 distances to all classified TCP points
        distances = np.linalg.norm(self.grasp_tcp_points - current_tcp, axis=1)

        # KNN majority vote
        k_actual = min(k, len(distances))
        nearest_indices = np.argpartition(distances, k_actual)[:k_actual]
        nearest_classifications = self.grasp_classifications[nearest_indices]
        is_good = nearest_classifications.sum() > k_actual / 2

        return bool(is_good)

    def _find_object_mesh(self, asset_id: str) -> Path | None:
        """Locate the .obj mesh file for the given asset_id. Returns None if not found."""
        objects_root = DATA_CACHE_DIR / "objects" / "thor"

        candidates = glob.glob(str(objects_root / "**" / f"{asset_id}.obj"), recursive=True)
        candidates = [c for c in candidates if "collision" not in c]
        if candidates:
            return Path(candidates[0])

        lower_id = asset_id.lower()
        candidates = glob.glob(
            str(objects_root / "**" / f"{asset_id}_{lower_id}.obj"), recursive=True
        )
        candidates = [c for c in candidates if "collision" not in c]
        if candidates:
            return Path(candidates[0])

        return None

    def _get_mesh_body_offset(self, asset_id: str) -> np.ndarray:
        """Get the position offset of the mesh child body from the asset XML.

        In iTHOR assets the mesh geoms live on a child body with a pos offset
        from the root body. Grasps are stored in the root body frame, so we
        need this offset to align the raw .obj vertices with the grasp frame.
        Returns (3,) array, defaults to zeros if the XML can't be parsed.
        """
        import xml.etree.ElementTree as ET

        objects_root = DATA_CACHE_DIR / "objects" / "thor"
        xml_candidates = glob.glob(str(objects_root / "**" / f"{asset_id}.xml"), recursive=True)
        if not xml_candidates:
            return np.zeros(3)

        try:
            tree = ET.parse(xml_candidates[0])
            root = tree.getroot()
            # Find the root body, then its first child body with a pos attribute
            worldbody = root.find("worldbody")
            if worldbody is None:
                return np.zeros(3)
            root_body = worldbody.find("body")
            if root_body is None:
                return np.zeros(3)
            child_body = root_body.find("body")
            if child_body is None:
                return np.zeros(3)
            pos_str = child_body.get("pos")
            if pos_str is None:
                return np.zeros(3)
            offset = np.array([float(x) for x in pos_str.split()])
            log.debug(f"Mesh body offset for {asset_id}: {offset}")
            return offset
        except Exception as e:
            log.debug(f"Could not parse mesh body offset for {asset_id}: {e}")
            return np.zeros(3)

    def get_task_description(self) -> str:
        pickup_obj_name = self.config.task_config.referral_expressions.get(
            "pickup_obj_name", self.config.task_config.pickup_obj_name
        )
        return f"Pick up the {pickup_obj_name} with a semantically correct grasp."

    def judge_success(self) -> bool:
        """Success requires both lifting the object AND a semantically correct grasp."""
        if self.config.task_type == "semantic_grasp_pick":
            return self.get_info()[0]["success"]
        else:
            raise ValueError(f"Invalid task_type {self.config.task_type}")

    def get_info(self) -> list[dict[str, Any]]:
        """Get metrics including semantic grasp correctness."""
        infos = super().get_info()
        for info in infos:
            base_success = info["success"]

            pickup_obj = MlSpacesObject(
                data=self._env.current_data,
                object_name=self.config.task_config.pickup_obj_name,
            )
            lift_height = pickup_obj.position[2] - self.config.task_config.pickup_obj_start_pose[2]
            succ_threshold = self.config.task_config.succ_pos_threshold

            if self.config.task_config.require_no_receptacle_contact:
                # Use the full base success condition (lift height + no receptacle contact)
                lifted = base_success
            else:
                # Only check lift height, skip the contact check
                lifted = lift_height >= succ_threshold

            if lifted:
                grasp_correct = self.classify_current_grasp()
            else:
                grasp_correct = False
            info["grasp_semantically_correct"] = grasp_correct
            info["success"] = lifted and grasp_correct
        return infos
