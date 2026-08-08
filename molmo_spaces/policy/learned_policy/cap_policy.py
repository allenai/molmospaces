import logging
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.rum_client import RUMClient

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def action_tensor_to_matrix(action_tensor, rot_unit):
    affine = np.eye(4)
    if rot_unit == "euler":
        r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
    elif rot_unit == "axis":
        r = R.from_rotvec(action_tensor[3:6])
    else:
        raise NotImplementedError
    affine[:3, :3] = r.as_matrix()
    affine[:3, -1] = action_tensor[:3]
    return affine


class CAP_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.grasping_type = exp_config.policy_config.grasping_type
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.use_vlm = exp_config.policy_config.use_vlm
        self.use_exo = exp_config.policy_config.exo_vlm
        self.model = None

    def prepare_model(self):
        host = "localhost"
        port = 8765

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.model = RUMClient(host=host, port=port)
                metadata = self.model.get_server_metadata()
                self.model_name = metadata.get("checkpoint", "rum")
                log.info(f"Connected to RUM server at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to RUM server after {max_retries} attempts")
                    raise

    def reset(self):
        # The server keeps per-connection history (3-frame image buffer, 2-step action
        # buffer) that outlives an episode: prepare_model() connects once and the session
        # is only torn down on disconnect. Without this, the opening steps of every
        # episode after the first are conditioned on the previous episode's frames and
        # actions. Guarded because task.reset() runs before the lazy connect in
        # obs_to_model_input(), so self.model is still None on the first episode.
        if self.model is not None:
            self.model.reset()
        self.starting_time = None
        self.T_world_object = None
        self.T_world_camera = None
        self.T_world_rum = None
        self.is_grasping = False
        self.step_counter = 0

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        # The rollout loop passes a batched list (one entry per env); every other
        # learned policy unwraps it the same way (cf. dreamzero_policy, pi_policy).
        # Without this, obs["object_poses"] below raises
        # "TypeError: list indices must be integers or slices, not str".
        if isinstance(obs, list):
            if len(obs) > 1:
                log.warning("obs list has %d elements, only using the first", len(obs))
            obs = obs[0]
        if self.model is None:
            self.prepare_model()

        # Invalidate the cached anchor on the ground-truth path so it is recomputed every
        # step. A task may rewrite pickup_obj_start_pose mid-episode -- MugBallPickTask
        # does exactly that when its settle window ends (mug_ball_pick_task.py:37-50),
        # once the mugs have finished falling. Caching would pin the anchor to the value
        # read on the first call, and the first call is always inside the settle window:
        # the rollout loop invokes get_action() before task.step() applies the no-op
        # substitution (pipeline.py:864-882), so step 0 sees the pre-fall pose.
        #
        # Only the ground-truth path is invalidated. The VLM path below must stay cached
        # because infer_point() is a paid API call, and re-running it per step would also
        # make the anchor jitter with detection noise.
        if not self.use_vlm:
            self.T_world_object = None

        if hasattr(self, "T_world_object") is False or self.T_world_object is None:
            if not self.use_vlm:
                # "object_poses" is only registered for RBY1 (rby1_sensors.py:272) and
                # "pickup_obj_pose" only for nav tasks (sensors.py:1110). The pick task's
                # suite (pick_task.py:63-79) exposes the pickup object's WORLD-frame pose
                # as "obj_start" (7D: x, y, z, qw, qx, qy, qz), so no base composition is
                # needed. Reading it is an array lookup, not a sim query, so recomputing
                # every step is cheap.
                obj_start = np.asarray(obs["obj_start"], dtype=np.float64)
                self.T_world_object = np.eye(4)
                self.T_world_object[:3, 3] = obj_start[:3]
                self.T_world_object[:3, :3] = R.from_quat(
                    obj_start[3:7], scalar_first=True
                ).as_matrix()
            else:
                exo_depth = obs["exo_camera_1_depth"]
                exo_rgb = obs["exo_camera_1"]
                ego_depth = obs["wrist_camera_depth"]
                ego_rgb = obs["wrist_camera"]
                point_norm = self._infer_point_norm(obs, exo_rgb if self.use_exo else ego_rgb)
                if self.use_exo:
                    K = np.array(obs["sensor_param_exo_camera_1"]["intrinsic_cv"])
                else:
                    K = np.array(obs["sensor_param_wrist_camera"]["intrinsic_cv"])
                fovy = 2 * np.arctan((2 * K[1, 2]) / (2 * K[1, 1]))
                x_norm, y_norm = point_norm
                width, height = (
                    (exo_rgb.shape[1], exo_rgb.shape[0])
                    if self.use_exo
                    else (ego_rgb.shape[1], ego_rgb.shape[0])
                )
                x = int(x_norm * width)
                y = int(y_norm * height)
                depth_value = exo_depth[y, x] + 0.03 if self.use_exo else ego_depth[y, x] + 0.03
                f = height / (2 * np.tan(fovy / 2))
                cam_mat = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
                cx = cam_mat[0, 2]
                cy = cam_mat[1, 2]
                fx = cam_mat[0, 0]
                fy = cam_mat[1, 1]
                z_cam = -depth_value
                x_cam = -(x - cx) * z_cam / fx
                y_cam = -(cy - y) * z_cam / fy
                p_cam = np.array([x_cam, y_cam, z_cam])
                T_corr = np.eye(4)
                T_corr[:3, :3] = np.diag([1, -1, -1])
                if self.use_exo:
                    camera_pose = (
                        np.array(obs["sensor_param_exo_camera_1"]["cam2world_gl"].copy()) @ T_corr
                    )
                else:
                    camera_pose = (
                        np.array(obs["sensor_param_wrist_camera"]["cam2world_gl"].copy()) @ T_corr
                    )
                p_world = camera_pose[:3, :3] @ p_cam + camera_pose[:3, 3]
                self.T_world_object = np.eye(4)
                self.T_world_object[:3, 3] = p_world

        T_base_ego = np.eye(4)
        T_base_ego[:3, 3] = obs["tcp_pose"][:3]
        T_base_ego[:3, :3] = R.from_quat(obs["tcp_pose"][3:7], scalar_first=True).as_matrix()

        T_world_base = np.eye(4)
        T_world_base[:3, 3] = obs["robot_base_pose"][:3]
        T_world_base[:3, :3] = R.from_quat(
            obs["robot_base_pose"][3:7], scalar_first=True
        ).as_matrix()

        T_world_ego = T_world_base @ T_base_ego
        self.T_world_camera = T_world_ego
        T_camera_object = np.linalg.inv(T_world_ego) @ self.T_world_object
        object_3d_position = T_camera_object[:3, 3]
        object_3d_position = np.array(
            [-T_camera_object[0, 3], -T_camera_object[2, 3], -T_camera_object[1, 3]]
        )

        if self.is_grasping:
            object_3d_position = np.array([0.00, 0.18, 0.04])
        return {
            "rgb_ego": cv2.resize(obs["wrist_camera"], (224, 224)),
            "object_3d_position": object_3d_position,
        }

    def _infer_point_norm(self, obs, rgb: np.ndarray) -> np.ndarray:
        """Return the 2D point to anchor on, as (x, y) normalized to [0, 1].

        Overridable hook: everything downstream of this (intrinsics, depth lookup, the
        world-frame lift) is independent of how the point was obtained. GeminiCAP_Policy
        overrides it to supply a point reasoned over a sequence of frames instead of a
        single-image object query.
        """
        return self.model.infer_point(
            rgb=rgb,
            object_name=self.task.config.task_config.referral_expressions["pickup_obj_name"],
            task=self.config.task_type,
        )

    def inference_model(self, model_input):
        self.step_counter += 1

        model_output = self.model.infer(model_input)
        model_output[0][:3] = np.array(
            [-model_output[0][0], -model_output[0][2], -model_output[0][1]]
        )
        model_output[0][3:6] = np.array(
            [-model_output[0][3], -model_output[0][5], -model_output[0][4]]
        )
        # Per-step, NOT latched. This mirrors upstream, which re-evaluates
        # `self.gripper <= closing_threshold` every step (cap-policy robot/controller.py:368)
        # and additionally breaks the rollout at the first close (:426) -- so a sticky flag
        # there would be harmless. Here the rollout runs the full horizon, and a
        # `max()` latch made the first dip below threshold irreversible: the gripper was
        # welded shut and object_3d_position was pinned to GRIPPER_FROM_CAMERA for the rest
        # of the episode, so CAP got exactly one grasp attempt.
        #
        # Replaying real episodes through the policy shows the raw gripper signal
        # oscillates and recovers on its own (e.g. 0.65 0.53 0.49 0.34 0.22 -> 0.65 0.71
        # 1.00 1.00), so latching discards a re-attempt the model is actively trying to
        # make. It also fires inside the settle no-op window on most episodes, which would
        # burn the single attempt before the arm has moved at all.
        self.is_grasping = bool(model_output[0][6] < self.grasping_threshold)
        delta_pose_mat = action_tensor_to_matrix(model_output[0][:6], "euler")
        T_world_ego = self.T_world_camera @ delta_pose_mat
        self.T_world_rum = T_world_ego

        goal_pose_7d = np.array(
            list(self.T_world_rum[:3, 3])
            + list(R.from_matrix(self.T_world_rum[:3, :3]).as_quat(scalar_first=True))
        )

        goal_pose_homogeneous = np.eye(4)
        goal_pose_homogeneous[:3, 3] = goal_pose_7d[:3]
        goal_pose_homogeneous[:3, :3] = R.from_quat(
            goal_pose_7d[3:7], scalar_first=True
        ).as_matrix()

        kinematics = self.task.env.current_robot.kinematics
        robot_view = self.task.env.current_robot.robot_view
        gripper_mgs = set(robot_view.get_gripper_movegroup_ids())
        mgs_except_gripper = [x for x in robot_view.move_group_ids() if x not in gripper_mgs]
        new_pose = goal_pose_homogeneous.copy()

        jp = kinematics.ik(
            "arm",
            new_pose,
            mgs_except_gripper,
            robot_view.get_qpos_dict(),
            robot_view.base.pose,
            rel_to_base=False,
        )
        action = robot_view.get_ctrl_dict()
        if jp is not None:
            action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})

        if self.grasping_type == "binary":
            if self.is_grasping:
                action["gripper"] = np.array([-255.0])
            else:
                action["gripper"] = np.array([0.0])
        else:
            action["gripper"] = (1 - model_output[0][6]) * np.array([-255.0])
            if self.is_grasping:
                action["gripper"] = np.array([-255.0])
        return action

    def model_output_to_action(self, model_output):
        return model_output

    def get_info(self) -> dict:
        info = super().get_info()

        info["policy_checkpoint"] = self.model_name
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
