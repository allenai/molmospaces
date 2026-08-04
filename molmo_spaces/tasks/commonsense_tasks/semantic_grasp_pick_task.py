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

DEBUG_VIS_DIR = Path("debug_semantic_grasp")


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

        # Debug visualization disabled — PNG dumps per step are a
        # per-episode bottleneck. Re-enable by uncommenting.
        # self._save_grasp_debug_visualization(
        #     current_tcp=current_tcp,
        #     nearest_indices=nearest_indices,
        #     is_good=bool(is_good),
        # )

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

    def _save_grasp_debug_visualization(
        self,
        current_tcp: np.ndarray,
        nearest_indices: np.ndarray,
        is_good: bool,
    ) -> None:
        """Save a 3D debug plot showing grasp classification results.

        Colors:
        - Red: bad grasps (all classified-bad TCP points)
        - Green: good grasps (all classified-good TCP points)
        - Yellow: k nearest grasps used for the vote
        - Blue: current gripper TCP position
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            log.warning("matplotlib not available, skipping grasp debug visualization")
            return

        DEBUG_VIS_DIR.mkdir(exist_ok=True)
        self._vis_counter += 1

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Try to load and render the object mesh
        if self._asset_id:
            mesh_path = self._find_object_mesh(self._asset_id)
            if mesh_path is not None:
                try:
                    import trimesh

                    mesh = trimesh.load(str(mesh_path), force="mesh")
                    verts = np.array(mesh.vertices)
                    # Shift mesh vertices from sub-body frame to parent body frame
                    # (grasps are in parent body frame, mesh .obj is in child body frame)
                    mesh_offset = self._get_mesh_body_offset(self._asset_id)
                    verts = verts + mesh_offset
                    faces = np.array(mesh.faces)
                    face_verts = verts[faces]
                    mesh_coll = Poly3DCollection(
                        face_verts,
                        alpha=0.15,
                        facecolor="lightsteelblue",
                        edgecolor="slategray",
                        linewidth=0.1,
                    )
                    ax.add_collection3d(mesh_coll)
                except Exception as e:
                    log.debug(f"Could not load mesh for visualization: {e}")

        tcp_pts = self.grasp_tcp_points
        good_mask = self.grasp_classifications
        bad_mask = ~good_mask

        # Layer 1: Bad grasps in red
        if bad_mask.any():
            ax.scatter(
                tcp_pts[bad_mask, 0],
                tcp_pts[bad_mask, 1],
                tcp_pts[bad_mask, 2],
                c="red",
                s=8,
                alpha=0.4,
                label=f"Bad grasps ({bad_mask.sum()})",
                depthshade=False,
            )

        # Layer 2: Good grasps in green
        if good_mask.any():
            ax.scatter(
                tcp_pts[good_mask, 0],
                tcp_pts[good_mask, 1],
                tcp_pts[good_mask, 2],
                c="limegreen",
                s=8,
                alpha=0.4,
                label=f"Good grasps ({good_mask.sum()})",
                depthshade=False,
            )

        # Layer 3: K nearest grasps in yellow (on top)
        nn_pts = tcp_pts[nearest_indices]
        ax.scatter(
            nn_pts[:, 0],
            nn_pts[:, 1],
            nn_pts[:, 2],
            c="gold",
            s=60,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            label=f"K nearest ({len(nearest_indices)})",
            depthshade=False,
        )

        # Layer 4: Current grasp TCP in blue (largest, on top)
        ax.scatter(
            [current_tcp[0]],
            [current_tcp[1]],
            [current_tcp[2]],
            c="dodgerblue",
            s=120,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.0,
            marker="*",
            label="Current grasp",
            depthshade=False,
            zorder=10,
        )

        # Set axis limits from all points
        all_pts = np.vstack([tcp_pts, current_tcp.reshape(1, 3)])
        margin = 0.02
        mins = all_pts.min(axis=0) - margin
        maxs = all_pts.max(axis=0) + margin
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect(maxs - mins)

        result_str = "GOOD" if is_good else "BAD"
        ax.set_title(
            f"{self._asset_id} — grasp classified as {result_str}\n"
            f"(object frame, blue★=current, yellow=KNN, green=good, red=bad)",
            fontsize=11,
        )
        ax.legend(loc="upper left", fontsize=8)

        filename = DEBUG_VIS_DIR / f"grasp_vis_{self._asset_id}_{self._vis_counter}.png"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        # log.info(f"[SEMANTIC GRASP PICK] Saved debug visualization: {filename}")

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
