import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.task_configs import PackingTaskConfig
from molmo_spaces.env.data_views import create_mjthor_body
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
from molmo_spaces.utils.mujoco_scene_utils import is_object_supported_by_body
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)


class PackingTask(PickAndPlaceTask):
    """Pick and place into a box (packing) task implementation."""

    # Object must be inside or just above the box (up to 10cm above box top)
    _support_fallback_z_threshold: float = 0.10
    objects_in_receptacle: set[str] = set()

    def get_task_description(self) -> str:
        pickup_name = self.config.task_config.referral_expressions["pickup_name"]
        place_name = self.config.task_config.referral_expressions["place_name"]
        return f"Pick up the {pickup_name} and place it into the {place_name}"

    def get_info(self) -> list[dict[str, Any]]:
        task_config = self.config.task_config
        if not isinstance(task_config, PackingTaskConfig) or not task_config.packing_object_names:
            return super().get_info()

        packing_names = task_config.packing_object_names
        metrics = []

        for i in range(self._env.n_batch):
            data = self._env.mj_datas[i]
            place_receptacle = create_mjthor_body(data, task_config.place_receptacle_name)

            # Check receptacle displacement
            start_pose = pos_quat_to_pose_mat(
                task_config.place_receptacle_start_pose[0:3],
                task_config.place_receptacle_start_pose[3:7],
            )
            curr_pose = place_receptacle.pose
            displacement = np.linalg.inv(start_pose) @ curr_pose
            pos_displacement = displacement[:3, 3]
            rot_displacement = R.from_matrix(displacement[:3, :3]).magnitude()
            pos_disp_norm = np.linalg.norm(pos_displacement)
            pos_disp_ok = pos_disp_norm <= task_config.max_place_receptacle_pos_displacement
            rot_disp_ok = rot_displacement <= task_config.max_place_receptacle_rot_displacement

            # Check each packing object is supported by the receptacle
            per_object_supported = {}
            om = self._env.object_managers[i]
            for obj_name in packing_names:
                pickup_obj = create_mjthor_body(data, obj_name)
                supported = is_object_supported_by_body(
                    data,
                    pickup_obj.body_id,
                    place_receptacle.body_id,
                    frac_weight_threshold=task_config.receptacle_supported_weight_frac,
                )
                if not supported:
                    # Cascading fallback: 1) contact, 2) raycast XY, 3) AABB with 8cm margin
                    objects_on_receptacle = om.objects_on_receptacle(
                        [om.get_object_by_name(obj_name)],
                        om.get_object_by_name(task_config.place_receptacle_name).geom_ids,
                        fallback_thres=self._support_fallback_z_threshold,
                        full_depth=True,
                        use_raycast_xy=True,
                    )
                    names_on_receptacle = {obj.name for obj in objects_on_receptacle}
                    supported = obj_name in names_on_receptacle
                per_object_supported[obj_name] = supported

            self.objects_in_receptacle = {name for name, s in per_object_supported.items() if s}
            num_packed = len(self.objects_in_receptacle)
            task_progress = num_packed / len(packing_names) if packing_names else 0.0
            all_supported = all(per_object_supported.values())
            success = all_supported and pos_disp_ok and rot_disp_ok

            if self.is_done().any():
                packed = [name for name, s in per_object_supported.items() if s]
                not_packed = [name for name, s in per_object_supported.items() if not s]
                log.info(
                    f"[PACKING RESULT] batch={i} success={success} progress={task_progress:.0%} "
                    f"({num_packed}/{len(packing_names)}) | "
                    f"packed={packed} | not_packed={not_packed} | "
                    f"receptacle_pos_disp={pos_disp_norm:.4f}m (ok={pos_disp_ok}) "
                    f"rot_disp={np.degrees(rot_displacement):.1f}deg (ok={rot_disp_ok})"
                )

            metrics.append(
                {
                    "success": success,
                    "task_progress": task_progress,
                    "all_objects_supported": all_supported,
                    "per_object_supported": per_object_supported,
                    "supported_by_receptacle": all_supported,
                    "receptacle_pos_displacement": pos_displacement,
                    "receptacle_pos_displacement_norm": pos_disp_norm,
                    "receptacle_rot_displacement": rot_displacement,
                    "episode_step": self.episode_step_count,
                }
            )

        return metrics
