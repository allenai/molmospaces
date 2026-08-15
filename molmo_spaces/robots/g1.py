from typing import TYPE_CHECKING, Any

import mink
import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtEq, mjtObj
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.controllers.g1_walk import (
    _DEFAULT_HEIGHT_CMD,
    NUM_TARGET_DIMS,
    G1WalkController,
)
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.robot_views.g1_view import (
    ARM_JOINT_SUFFIXES,
    HOLO_BASE_TARGET_BODY_NAME,
    LEGS_WAIST_JOINT_SUFFIXES,
    G1RobotView,
)
from molmo_spaces.utils.linalg_utils import normalize_ang_error

# Velocity clamps for the AStarPlannerPolicy waypoint -> G1WalkController command
# bridge (see G1Robot._waypoint_to_velocity_target). Not from the reference (which
# doesn't have this bridge) -- chosen to match the forward speed already validated
# empirically in the standing/walking smoke test.
_MAX_LINEAR_VEL = 0.5
_MAX_YAW_RATE = 0.5
# G1WalkController switches from its walk policy to its (non-turning, non-walking)
# stand policy whenever norm(cmd) < 0.05 (see g1_walk.py). A pure proportional law
# decays toward zero as the error shrinks, so once any residual error's raw command
# drops below that switch threshold, it gets stuck standing just short of the goal --
# confirmed empirically via rotate() plateauing ~6-7deg short of target. Once the raw
# error exceeds _VELOCITY_DEADBAND, floor the command at _MIN_* (comfortably above
# 0.05) instead of letting it decay through the stand/walk switch point.
_MIN_LINEAR_VEL = 0.15
_MIN_YAW_RATE = 0.15
_VELOCITY_DEADBAND = 0.08
# Turn-then-drive gate (see _waypoint_to_velocity_target): only suppress translation
# while heading error is *large*. Gating on "any nonzero yaw_rate" instead (i.e. the
# full _VELOCITY_DEADBAND) was too strict -- G1WalkController's yaw tracking has its
# own residual convergence ceiling around 15deg (confirmed via rotate()), so a target
# heading landing inside that dead zone could never fully clear the gate, permanently
# blocking translation even though the heading is already close enough to walk toward
# the waypoint. Confirmed empirically: nav_to() stalling flat (no distance progress
# for 30+ steps) at exactly this kind of near-but-not-exact heading alignment.
_YAW_GATE_THRESHOLD = np.radians(30)

# kinematics_wbc() constants -- exact port of g1_molmo's own GraspPolicy/
# G1Controller mink whole-body IK setup (see molmo_spaces.g1_molmo_port.
# components.robot_g1ms.G1Robot.kinematics_wbc, which prototypes this same
# split onto a standalone G1Robot before this move). Torso-tilt safety
# envelope kept in sync with g1_server deploy clips.
_WBC_HEIGHT_MIN, _WBC_HEIGHT_MAX = 0.35, 0.793
_WBC_WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_WBC_TORSO_TILT_RANGES = [
    ("waist_yaw_joint", -0.5, 0.5),
    ("waist_roll_joint", -0.4, 0.4),
    ("waist_pitch_joint", -0.1, 0.4),
]


def _wbc_hand_arm_joints(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_{suffix}" for suffix in ARM_JOINT_SUFFIXES)


_WBC_HAND_ARM_JOINTS = {side: _wbc_hand_arm_joints(side) for side in ("left", "right")}
_WBC_HAND_SITE = {side: f"{side}_grasp" for side in ("left", "right")}

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig


class G1Robot(Robot):
    """G1 humanoid robot (Phase 3: whole-body walking via `G1WalkController`).

    The combined `legs_waist` move group (see `G1LegsWaistGroup`) is driven by
    `G1WalkController`, a ported whole-body walking controller (WBC): a Python
    PD-torque law plus an ONNX policy selecting between standing and walking
    gaits, conditioned on a commanded [vx, vy, yaw_rate, height, waist] target
    (see `G1WalkController.set_target`). This requires the `legs_waist`
    actuators to behave as raw torque passthroughs rather than the MJCF's own
    position-PD gains (see `apply_control_overrides`).

    Both arms and the right gripper stay on plain JointPosControllers -- the
    MJCF's own tuned PD actuator gains (biastype="affine") do the rest. The
    base has no actuators; the pelvis is a free-floating body whose dynamics
    MuJoCo integrates directly.
    """

    def __init__(self, mj_data: MjData, exp_config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, exp_config)

        # Matches g1_molmo's own components/controller.py set_env() (`m.opt.
        # timestep = 0.005`) -- see G1Config.physics_timestep's docstring for
        # why. Applied here (before any physics stepping happens for this
        # scene/robot) rather than left at the scene's own default.
        physics_timestep = self.exp_config.robot_config.physics_timestep
        if physics_timestep is not None:
            mj_data.model.opt.timestep = physics_timestep

        self._namespace = self.exp_config.robot_config.robot_namespace
        self._use_holo_base = self.exp_config.robot_config.use_holo_base
        self._robot_view = G1RobotView(mj_data, self.namespace, use_holo_base=self._use_holo_base)
        self._kinematics = MlSpacesKinematics(self.exp_config.robot_config)
        self._parallel_kinematics = DummyParallelKinematics(
            self.exp_config.robot_config, self._kinematics
        )
        # Holo-base mode only: pending mocap-target ctrl for the "base" move group,
        # applied in compute_control(). None means "no explicit waypoint command
        # yet" -- falls back to base.noop_ctrl (the pelvis's live current pose,
        # recomputed fresh every tick), so a freshly (re)placed robot never gets
        # yanked toward a stale target left over from an earlier episode/pose.
        self._pending_base_ctrl: np.ndarray | None = None

        if self._use_holo_base:
            legs_waist_controller = JointPosController(self.robot_view.get_move_group("legs_waist"))
        else:
            legs_waist_controller = G1WalkController(
                self.robot_view.get_move_group("legs_waist"),
                base_move_group=self.robot_view.get_move_group("base"),
                left_arm_move_group=self.robot_view.get_move_group("left_arm"),
                right_arm_move_group=self.robot_view.get_move_group("right_arm"),
                models_dir=self.exp_config.robot_config.get_robot_dir() / "policies",
            )

        self._controllers = {
            "legs_waist": legs_waist_controller,
            "left_arm": JointPosController(self.robot_view.get_move_group("left_arm")),
            "right_arm": JointPosController(self.robot_view.get_move_group("right_arm")),
            "right_gripper": JointPosController(self.robot_view.get_move_group("right_gripper")),
        }
        assert set(self._controllers.keys()).issubset(set(self._robot_view.move_group_ids())), (
            "All controller keys must be move group IDs"
        )

        # kinematics_wbc() state -- lazily built on first use (grasp-planning-
        # specific, irrelevant to a freshly constructed robot with no grasp
        # policy attached yet). See _ensure_wbc_ik_setup.
        self._wbc_ik_cfg = None
        self._wbc_ik_fj_dof = None
        self._wbc_ik_fj_qa = None
        self._wbc_scene_to_ik_qpos = None
        self._wbc_hand_cfg = None
        self._wbc_waist_qa = None
        self._wbc_posture_task = None
        self._wbc_pelvis_task = None
        self._wbc_limits = None
        self._wbc_self_collision_limit = None

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    @property
    def namespace(self):
        return self._namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    def _build_wbc_self_collision_limit(self, ik_model):
        """Self-collision limit for kinematics_wbc: arm<->torso/pelvis/waist/
        hip + arm<->arm geom pairs. Returns None if no collidable pairs exist.
        Moved here from FetchmanPickPlannerPolicy._build_self_collision_limit
        -- same statements, so grasp planning stays behaviorally unaffected
        by this move."""
        right_arm, left_arm, body = [], [], []
        for gid in range(ik_model.ngeom):
            if ik_model.geom_contype[gid] == 0 and ik_model.geom_conaffinity[gid] == 0:
                continue
            bid = ik_model.geom_bodyid[gid]
            bname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(s in bname for s in ("shoulder", "elbow", "wrist", "gripper")):
                (right_arm if "right" in bname else left_arm if "left" in bname else []).append(gid)
            elif any(s in bname for s in ("pelvis", "torso", "hip", "waist")):
                body.append(gid)
        arm = right_arm + left_arm
        pairs = []
        if arm and body:
            pairs.append((arm, body))
        if right_arm and left_arm:
            pairs.append((right_arm, left_arm))
        if not pairs:
            return None
        return mink.CollisionAvoidanceLimit(
            model=ik_model,
            geom_pairs=pairs,
            minimum_distance_from_collisions=0.02,
            collision_detection_distance=0.08,
        )

    def _ensure_wbc_ik_setup(self):
        """Lazily build the standalone-model mink whole-body IK setup
        kinematics_wbc() needs. Moved here from FetchmanPickPlannerPolicy.
        __init__'s own IK-setup block (same statements, same order) so grasp
        IK lives on the robot instead of being reimplemented per-policy --
        solving mink IK against a standalone ~35-DOF model instead of the
        live scene model (1000+ DOF in a cluttered procthor house) measured
        ~2600x faster per iteration, which is why this doesn't just reuse
        self.kinematics/self.mj_model directly.
        """
        if self._wbc_ik_cfg is not None:
            return
        ik_model = mujoco.MjModel.from_xml_path(
            str(self.exp_config.robot_config.get_robot_xml_path())
        )
        # Torso-tilt safety envelope, tighter than the MJCF's raw joint
        # ranges -- must be applied before mink.Configuration(ik_model) below.
        for jn, lo, hi in _WBC_TORSO_TILT_RANGES:
            jid_ = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid_ >= 0:
                ik_model.jnt_range[jid_] = [lo, hi]

        self._wbc_ik_cfg = mink.Configuration(ik_model)
        ik_fj = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._wbc_ik_fj_dof = ik_model.jnt_dofadr[ik_fj]
        self._wbc_ik_fj_qa = ik_model.jnt_qposadr[ik_fj]
        self._wbc_scene_to_ik_qpos = []
        for jid in range(ik_model.njnt):
            jname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jname:
                continue
            scene_jid = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.namespace + jname
            )
            if scene_jid < 0:
                continue
            qsz = 7 if ik_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            self._wbc_scene_to_ik_qpos.append(
                (self.mj_model.jnt_qposadr[scene_jid], ik_model.jnt_qposadr[jid], qsz)
            )

        self._wbc_hand_cfg = {}
        for hand in ("right", "left"):
            arm_joints = list(_WBC_HAND_ARM_JOINTS[hand])
            site = _WBC_HAND_SITE[hand]
            mask = np.zeros(ik_model.nv)
            for jn in arm_joints + list(_WBC_WAIST_JOINTS):
                jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    mask[ik_model.jnt_dofadr[jid]] = 1.0
            mask_h = mask.copy()
            mask_h[self._wbc_ik_fj_dof + 2] = 1.0
            task = mink.FrameTask(
                frame_name=site,
                frame_type="site",
                position_cost=100,
                orientation_cost=1,
                lm_damping=1,
            )
            arm_qa = np.array(
                [
                    ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                    for jn in arm_joints
                ],
                dtype=np.int32,
            )
            self._wbc_hand_cfg[hand] = {
                "mask": mask,
                "mask_h": mask_h,
                "task": task,
                "arm_joints": arm_joints,
                "arm_qa": arm_qa,
            }

        posture_cost = np.full(ik_model.nv, 0.1)
        for jn in _WBC_WAIST_JOINTS:
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                posture_cost[ik_model.jnt_dofadr[jid]] = 0.2
        posture_cost[self._wbc_ik_fj_dof + 2] = 0.1
        self._wbc_waist_qa = np.array(
            [
                ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in _WBC_WAIST_JOINTS
            ],
            dtype=np.int32,
        )
        self._wbc_posture_task = mink.PostureTask(ik_model, cost=posture_cost)
        self._wbc_pelvis_task = mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=[5.0, 5.0, 0.3],
            orientation_cost=0,
            lm_damping=1,
        )
        self._wbc_limits = [mink.ConfigurationLimit(ik_model)]
        self._wbc_self_collision_limit = self._build_wbc_self_collision_limit(ik_model)

    def kinematics_wbc(
        self, target_pos, target_rot=None, hand="right", avoid_self_collision=False, precision=False
    ):
        """Whole-body grasp IK: solves for arm+waist+height that places `hand`'s
        gripper site at `target_pos`/`target_rot`, against a standalone
        robot-only model synced from the live scene qpos through a
        precomputed joint correspondence table. Moved here from
        FetchmanPickPlannerPolicy._solve_ik_wbc (same statements, same order,
        same every early-break) so grasp-reach IK lives on the robot instead
        of being duplicated per-policy. `precision` replaces the caller's own
        `self._grasp_phase in (...)` check (that's grasp-policy state, not
        robot-generic) -- True selects the tighter max_iters/conv_thresh used
        during PHASE_DESCEND/CLOSE/POST_CLOSE/LIFT.

        Returns (arm, waist, ik_h, err): arm joint qpos (7,), waist joint
        qpos (3,), clamped pelvis height, final position error norm.
        """
        self._ensure_wbc_ik_setup()
        hcfg = self._wbc_hand_cfg[hand]

        ik_q = np.zeros(self._wbc_ik_cfg.model.nq, dtype=np.float64)
        for scene_qa, ik_qa, qsz in self._wbc_scene_to_ik_qpos:
            ik_q[ik_qa : ik_qa + qsz] = self.mj_data.qpos[scene_qa : scene_qa + qsz]
        self._wbc_ik_cfg.update(ik_q)

        q_post = self._wbc_ik_cfg.q.copy()
        q_post[self._wbc_waist_qa] = 0.0
        self._wbc_posture_task.set_target(q_post)
        # Anchor pelvis xy to current (walking unaffected) but z to standing height
        # so the IK actively wants to stand back up when the wrist target allows it.
        pelvis_T = self._wbc_ik_cfg.get_transform_frame_to_world("pelvis", "body")
        pelvis_pos = pelvis_T.translation().copy()
        pelvis_pos[2] = _WBC_HEIGHT_MAX
        self._wbc_pelvis_task.set_target(
            mink.SE3.from_rotation_and_translation(pelvis_T.rotation(), pelvis_pos)
        )
        rot = mink.SO3.from_matrix(target_rot) if target_rot is not None else mink.SO3.identity()
        hcfg["task"].set_target(
            mink.SE3.from_rotation_and_translation(rot, np.asarray(target_pos, dtype=np.float64))
        )

        max_iters = 60 if precision else 20
        conv_thresh = 0.0015 if precision else 0.005
        prev_err = float("inf")
        err = float("inf")
        limits = self._wbc_limits
        if avoid_self_collision and self._wbc_self_collision_limit is not None:
            limits = self._wbc_limits + [self._wbc_self_collision_limit]
        for step in range(max_iters):
            try:
                vel = mink.solve_ik(
                    self._wbc_ik_cfg,
                    [hcfg["task"], self._wbc_posture_task, self._wbc_pelvis_task],
                    1e-2,
                    "daqp",
                    damping=1e-1,
                    limits=limits,
                )
            except Exception:
                break
            vel *= hcfg["mask_h"]
            self._wbc_ik_cfg.integrate_inplace(vel, 1e-2)
            q_tmp = self._wbc_ik_cfg.q.copy()
            q_tmp[self._wbc_ik_fj_qa + 2] = np.clip(
                q_tmp[self._wbc_ik_fj_qa + 2], _WBC_HEIGHT_MIN, _WBC_HEIGHT_MAX
            )
            self._wbc_ik_cfg.update(q_tmp)
            err = float(np.linalg.norm(hcfg["task"].compute_error(self._wbc_ik_cfg)[:3]))
            if err < conv_thresh:
                break
            if step > 10 and err > prev_err - 1e-5:
                break
            prev_err = err
        q = self._wbc_ik_cfg.q
        arm = q[hcfg["arm_qa"]].astype(np.float32)
        waist = q[self._wbc_waist_qa].astype(np.float32)
        ik_h = float(np.clip(q[self._wbc_ik_fj_qa + 2], _WBC_HEIGHT_MIN, _WBC_HEIGHT_MAX))
        return arm, waist, ik_h, err

    def get_arm_move_group_ids(self) -> list[str]:
        """Both arms get independent TCP-bounded action noise."""
        return ["left_arm", "right_arm"]

    def update_control(self, action_command_dict: dict[str, Any]) -> None:
        """Bridge nav policies' base actions to whichever base control mode is active.

        Nav policies built for holonomic/mocap-driven bases (see
        AStarPlannerPolicy._build_navigation_action) command an absolute
        world-frame [x, y, theta] waypoint under the "base" key -- but G1 has no
        "base" move group controller in either mode (walking: the pelvis has no
        actuators at all; holo: the mocap target is written directly in
        compute_control(), not via the Controller/set_target() flow). Translate
        the waypoint into whichever interface is active before delegating to the
        normal per-move-group update_control() flow.

        FetchManBasePlannerPolicy instead computes a [vx, vy, yaw_rate] velocity
        command itself every step (see its _update_nav_command) and sends it
        directly under "base_velocity", bypassing _waypoint_to_velocity_target
        entirely -- there's no waypoint pose to re-derive it from. WBC mode only;
        holo-base mode has no notion of a velocity command (its mocap target is
        an absolute pose), so "base_velocity" is ignored there.
        """
        action_command_dict = dict(action_command_dict)
        waypoint = action_command_dict.pop("base", None)
        base_velocity = action_command_dict.pop("base_velocity", None)
        # G1BaseGroup.noop_ctrl is an empty array (the pelvis has no actuators), so
        # e.g. AStarPlannerPolicy._build_done_action's {"base": get_noop_ctrl_dict(...)}
        # sends an empty array here rather than omitting the key or sending None.
        has_waypoint = waypoint is not None and len(waypoint) == 3
        if self._use_holo_base:
            self._pending_base_ctrl = (
                self._waypoint_to_pose_ctrl(waypoint) if has_waypoint else None
            )
        elif has_waypoint:
            action_command_dict["legs_waist"] = self._waypoint_to_velocity_target(waypoint)

        if base_velocity is not None and not self._use_holo_base:
            # Apply the same floored-clip treatment _waypoint_to_velocity_target
            # already does for the "base" waypoint path -- without it, a nav
            # policy's own smoothstep brake (ramping speed continuously down to
            # 0 on final approach, e.g. FetchManBasePlannerPolicy/
            # FetchmanPickPlannerPolicy's _update_nav_command, ported directly
            # from g1_molmo) spends real time commanding small-but-nonzero
            # speeds in [_VELOCITY_DEADBAND, _MIN_LINEAR_VEL) -- exactly the
            # regime _floored_clip's docstring describes G1WalkController's
            # stand/walk switch getting stuck in, since a raw pass-through
            # command like this bypassed the fix entirely. Confirmed
            # empirically: FetchmanPickPlannerPolicy's walk phase stalled
            # ~0.28m short of its goal indefinitely, sending a steady ~0.09
            # m/s (between the 0.08 deadband and the 0.15 floor) the whole
            # time with essentially zero actual displacement.
            vx = self._floored_clip(
                base_velocity[0], _MIN_LINEAR_VEL, _MAX_LINEAR_VEL, _VELOCITY_DEADBAND
            )
            vy = self._floored_clip(
                base_velocity[1], _MIN_LINEAR_VEL, _MAX_LINEAR_VEL, _VELOCITY_DEADBAND
            )
            yaw_rate = self._floored_clip(
                base_velocity[2], _MIN_YAW_RATE, _MAX_YAW_RATE, _VELOCITY_DEADBAND
            )
            action_command_dict["legs_waist"] = np.array(
                [vx, vy, yaw_rate, _DEFAULT_HEIGHT_CMD, 0.0, 0.0, 0.0], dtype=np.float32
            )

        legs_waist_action = action_command_dict.get("legs_waist")
        if (
            not self._use_holo_base
            and legs_waist_action is not None
            and len(legs_waist_action) != NUM_TARGET_DIMS
        ):
            # Generic policies unaware of G1's WBC (e.g. base_object_manipulation_
            # planner_policy's get_noop_ctrl_dict(), used to hold non-manipulated
            # move groups still during e.g. door/cabinet opening) send legs_waist's
            # own MoveGroup-shaped noop ctrl -- a 15-dim joint-position array,
            # matching its actuator count -- not realizing G1WalkController's
            # target is a 7-dim [vx, vy, yaw_rate, height, waist] velocity command.
            # Drop it so update_control()'s normal "no command -> set_to_stationary()"
            # fallback applies instead, which is what "hold legs_waist still"
            # actually means for a WBC-driven biped (see G1WalkController.
            # set_to_stationary's docstring).
            action_command_dict.pop("legs_waist")

        super().update_control(action_command_dict)

    def compute_control(self) -> None:
        super().compute_control()
        if self._use_holo_base:
            base = self._robot_view.get_move_group("base")
            base.ctrl = (
                self._pending_base_ctrl if self._pending_base_ctrl is not None else (base.noop_ctrl)
            )

    @staticmethod
    def _floored_clip(value: float, min_mag: float, max_mag: float, deadband: float) -> float:
        """Proportional clip, but floor the magnitude at `min_mag` once |value| exceeds
        `deadband` -- see _VELOCITY_DEADBAND for why a pure proportional-to-zero law
        stalls G1WalkController's stand/walk switch short of convergence."""
        if abs(value) <= deadband:
            return 0.0
        return float(np.sign(value)) * float(np.clip(abs(value), min_mag, max_mag))

    def _waypoint_to_velocity_target(self, waypoint) -> np.ndarray:
        pose = self._robot_view.base.pose
        x, y = pose[0, 3], pose[1, 3]
        yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2]

        dx, dy = waypoint[0] - x, waypoint[1] - y
        local_vx = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_vy = -np.sin(yaw) * dx + np.cos(yaw) * dy
        yaw_error = normalize_ang_error(waypoint[2] - yaw)

        yaw_rate = self._floored_clip(yaw_error, _MIN_YAW_RATE, _MAX_YAW_RATE, _VELOCITY_DEADBAND)
        if abs(yaw_error) > _YAW_GATE_THRESHOLD:
            # Turn-then-drive: while heading is substantially off, correcting it and
            # translating simultaneously fights itself (the walking gait's own
            # turning drift keeps re-triggering a position correction, which never
            # converges -- confirmed empirically via rotate() plateaux). Prioritize
            # closing the heading error first; translate once roughly facing the
            # waypoint, same as a differential-drive "rotate then drive" strategy.
            vx = vy = 0.0
        else:
            vx = self._floored_clip(local_vx, _MIN_LINEAR_VEL, _MAX_LINEAR_VEL, _VELOCITY_DEADBAND)
            vy = self._floored_clip(local_vy, _MIN_LINEAR_VEL, _MAX_LINEAR_VEL, _VELOCITY_DEADBAND)

        return np.array([vx, vy, yaw_rate, _DEFAULT_HEIGHT_CMD, 0.0, 0.0, 0.0], dtype=np.float32)

    def _waypoint_to_pose_ctrl(self, waypoint) -> np.ndarray:
        """Holo-base mode: convert an absolute [x, y, theta] waypoint directly into
        the mocap target's [x, y, z, qw, qx, qy, qz] ctrl (see G1HoloBaseGroup),
        keeping the current height and a level (roll=pitch=0) orientation."""
        current_z = self._robot_view.base.pose[2, 3]
        quat = R.from_euler("z", waypoint[2]).as_quat(scalar_first=True)
        return np.array([waypoint[0], waypoint[1], current_z, *quat], dtype=np.float32)

    def reset(self) -> None:
        """Reset the robot to its initial standing state."""
        init_qpos_dict = self.exp_config.robot_config.init_qpos
        self.set_joint_pos(init_qpos_dict)
        self._pending_base_ctrl = None
        for controller in self._controllers.values():
            controller.reset()

    @staticmethod
    def robot_model_root_name() -> str:
        return "pelvis"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "BaseRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        pos = pos + [0.0] if len(pos) == 2 else pos
        super().add_robot_to_scene(
            robot_config=robot_config,
            spec=spec,
            prefix=prefix,
            pos=pos,
            quat=quat,
            randomize_textures=randomize_textures,
            strip_meshes=strip_meshes,
        )

        if getattr(robot_config, "use_holo_base", False):
            # Mirrors FloatingRUMRobot.add_robot_to_scene: weld a mocap target body
            # to the pelvis so G1HoloBaseGroup can drive the base directly via ctrl,
            # since G1's pelvis is a real free joint (no virtual holonomic joints to
            # actuate the way RBY1's base does).
            target_body_name = f"{prefix}{HOLO_BASE_TARGET_BODY_NAME}"
            spec.worldbody.add_body(name=target_body_name, pos=pos, quat=quat, mocap=True)
            eq = spec.add_equality()
            eq.name1 = target_body_name
            eq.name2 = f"{prefix}{cls.robot_model_root_name()}"
            eq.solref = np.array([0.02, 1])
            eq.solimp = np.array([0.9, 0.95, 0.0, 1, 2])
            eq.objtype = mjtObj.mjOBJ_BODY
            eq.type = mjtEq.mjEQ_WELD

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "BaseRobotConfig"):
        super().apply_control_overrides(spec, robot_config)
        # Match the solver settings the source G1 stack validated for stable
        # bipedal contact: implicit integration, pyramidal friction cones,
        # extra no-slip iterations, and unit impedance ratio. The robot's own
        # MJCF <option> block can't take effect on its own -- MjSpec.attach_body
        # only merges body subtrees, not <option> blocks -- so whatever the
        # target scene's <option> says would otherwise silently win.
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.noslip_iterations = 5
        spec.option.impratio = 1.0

        # The MJCF's "walk_*" actuators for legs_waist are authored as tuned
        # position-PD actuators (biastype="affine"), matching Phase 2's plain
        # JointPosController usage. G1WalkController computes its own PD torque
        # in Python (see molmo_spaces.controllers.g1_walk), so these need to
        # become raw torque passthroughs instead: fixed unit gain, no bias.
        # Holo-base mode doesn't use G1WalkController (legs_waist just holds a
        # static pose via plain JointPosController), so it keeps the MJCF's
        # original tuned PD gains untouched.
        namespace = robot_config.robot_namespace
        if not getattr(robot_config, "use_holo_base", False):
            for joint_name in LEGS_WAIST_JOINT_SUFFIXES:
                actuator = spec.actuator(f"{namespace}walk_{joint_name}")
                assert actuator is not None, f"Missing walk_{joint_name} actuator"
                actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
                actuator.gainprm[0] = 1.0
                actuator.biasprm[:] = 0.0

            # The arms' own "walk_{side}_*" actuators are authored in the MJCF
            # with much weaker gains than G1WalkController's IK targets assume
            # (e.g. the wrist actuators cap out around +-5 N*m) -- reconfigure
            # them to the same kp/kd the reference g1_molmo stack uses
            # (components/controller_g1ms.py's G1Controller.setup()), or the
            # arm/wrist can't physically reach IK-commanded orientations:
            # position converges "close enough" while rotation error stalls
            # at a fixed residual forever. Left arm is zero-gained (hangs
            # passively under gravity, matching the reference) even though it
            # still sits behind a JointPosController -- gain=0 makes whatever
            # that controller writes moot.
            for joint_name in ARM_JOINT_SUFFIXES:
                left_actuator = spec.actuator(f"{namespace}walk_left_{joint_name}")
                assert left_actuator is not None, f"Missing walk_left_{joint_name} actuator"
                left_actuator.gainprm[0] = 0.0
                left_actuator.biasprm[:] = 0.0
                left_actuator.forcerange[:] = [-400, 400]

                right_actuator = spec.actuator(f"{namespace}walk_right_{joint_name}")
                assert right_actuator is not None, f"Missing walk_right_{joint_name} actuator"
                right_actuator.gainprm[0] = 2000.0
                right_actuator.biasprm[0] = 0.0
                right_actuator.biasprm[1] = -2000.0
                right_actuator.biasprm[2] = -60.0
                right_actuator.forcerange[:] = [-400, 400]

        # The MJCF puts the head/mount/logo visual geoms on group 5 (every other
        # visual geom uses the "visual" class default of group 2). mujoco.MjvOption's
        # default geomgroup is [1, 1, 1, 0, 0, 0] and this codebase's renderers never
        # enable group 5 (only sitegroup gets touched -- see MjOpenGLRenderer), so the
        # head/logo are silently invisible in every render. Move them to group 2 so
        # they render like the rest of the robot.
        head_mesh_suffixes = ("head_link", "head_mount", "logo_link")
        for body_name in ("torso_link", "head_camera_mount"):
            body = spec.body(f"{namespace}{body_name}")
            assert body is not None, f"Missing {body_name} body"
            for geom in body.geoms:
                if geom.group == 5 and geom.meshname.endswith(head_mesh_suffixes):
                    geom.group = 2
