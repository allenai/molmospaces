"""G1-specific PickTaskSampler: g1_molmo's reset-time support/robot height
randomization and grasp-reachability precheck, behind
`PickTaskSampler._post_placement_setup`. The native counterpart of the
fetchman fork's G1TaskSampler; the gold-parity harness exercises the
fork, not this class.
"""

import logging

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


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


class PickTaskSamplerG1(PickTaskSampler):
    """`PickTaskSampler` plus g1_molmo's reset-time height randomization and
    grasp-reachability precheck. Behaviorally identical to its base class
    unless the corresponding config flags are set."""

    def _post_placement_setup(
        self, env: CPUMujocoEnv, pickup_obj_name: str, supporting_geom_id: int
    ) -> bool:
        """Height randomization (g1_molmo ports) -- must run before the
        reachability precheck right below, on the same (object, placement)
        attempt: g1_molmo randomizes support/robot height as part of the same
        reset attempt it then reachability-checks, not as a separate step
        after an attempt is already committed.
        """
        om = env.object_managers[env.current_batch_index]

        self._randomize_target_support_height(env, pickup_obj_name, supporting_geom_id)
        self._randomize_robot_standing_height(env)
        mujoco.mj_forward(env.current_model, env.current_data)

        # Re-capture pickup_obj_start_pose after height randomization, or
        # PickTask measures lift_height from the pre-randomization pose.
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
                return False

        return True

    def _randomize_target_support_height(
        self, env: CPUMujocoEnv, pickup_obj_name: str, supporting_geom_id: int
    ) -> None:
        """g1_molmo's _randomize_target_support_height: move the object's
        support (and the object) to a height drawn from a triangular
        distribution over [randomize_height_min, its current height] with mode
        randomize_height_favored. Handles one object on one fixed support; the
        reference also traces stacked supports and contact neighbours.
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
        """g1_molmo's randomize_robot_height: a uniform random initial WBC height
        command per episode (applied unconditionally here). No-op without a
        legs_waist WBC controller."""
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
        """Standalone robot-only mink model for _precheck_grasp_reachable,
        cached on the sampler (solving on the live scene model is ~2600x slower)."""
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
            # The posture task is required: without it the 11-DOF null space
            # of this 6-DOF target lets the QP wander and stall.
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
        """g1_molmo's precheck_grasp: reject the placement if none of the
        top-ranked grasp candidates is IK-reachable (grasp pose only). Tries
        several candidates and the policy's roll-flip disambiguation so one
        bad top candidate does not reject a fine spawn. True (don't block)
        without a legs_waist WBC controller or grasp data.
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
