"""
Implementation of the Unitree G1 robot model.

The G1 (as configured here) is a bipedal humanoid with:
- A free-floating pelvis (real 6-DOF dynamics, not a kinematic/wheeled base)
- 12 leg joints + a 3-DOF waist/torso, combined into one 15-DOF move group
  driven by a whole-body walking controller (see `G1WalkController`)
- Two 7-DOF arms (left arm has no gripper and is commanded to its natural
  hanging pose; right arm has a dexterous gripper)
- A single right-hand gripper, tendon-actuated, with two mechanically coupled
  finger joints (joint2 = joint1, enforced by an MJCF <equality> constraint)

Each component is implemented as a MoveGroup, with the overall robot structure
managed by the G1RobotView class. There is no independent head MoveGroup
(the head/cameras are rigidly mounted to the torso) and no left-hand gripper
(no hardware exists for it).
"""

from functools import cached_property

import numpy as np
from mujoco import MjData
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.linalg_utils import normalize_ang_error
from molmo_spaces.utils.mj_model_and_data_utils import body_pose

# Name of the mocap body G1Robot.add_robot_to_scene adds (weld-constrained to the
# pelvis) when use_holo_base=True. See G1HoloBaseGroup.
HOLO_BASE_TARGET_BODY_NAME = "g1_target_base_pose"

# Dex gripper: positive qpos closes the fingers, negative opens (matches the
# actuator's ctrlrange and the MJCF <equality joint1="right_Joint1_1"
# joint2="right_Joint2_1" polycoef="0 1 0 0 0"/>, i.e. joint2 = joint1).
GRIPPER_OPEN = -0.0222
GRIPPER_CLOSED = 0.0245

_LEG_JOINT_SUFFIXES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
_WAIST_JOINT_SUFFIXES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

# The `legs_waist` MoveGroup's joints, in order -- used by G1Robot.apply_control_overrides
# to reconfigure the corresponding "walk_*" actuators for the whole-body walking
# controller (see molmo_spaces.controllers.g1_walk.G1WalkController).
LEGS_WAIST_JOINT_SUFFIXES = _LEG_JOINT_SUFFIXES + _WAIST_JOINT_SUFFIXES
_ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)


class G1BaseGroup(FreeJointRobotBaseGroup):
    """The G1's free-floating pelvis. No wheels, no holonomic joints -- true
    6-DOF rigid-body dynamics, integrated directly by MuJoCo's physics."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}floating_base_joint").id
        super().__init__(mj_data, base_joint_id, [], [])

    @property
    def noop_ctrl(self) -> np.ndarray:
        return np.array([])


class G1HoloBaseGroup(FreeJointRobotBaseGroup):
    """Alternate pelvis base for G1Config(use_holo_base=True): mocap-weld driven,
    like FloatingRUMBaseGroup, rather than passive (see G1BaseGroup).

    G1's pelvis is a real 6-DOF free joint in the MJCF (unlike RBY1's purpose-built
    virtual x/y/theta holonomic joints), so rather than retrofit new joint types
    into the compiled model, this reuses the same mocap-body + weld-equality trick
    FloatingRUMBaseGroup uses: a physics weld constraint pulls the pelvis toward a
    commanded [x, y, z, qw, qx, qy, qz] target pose written via `ctrl`. Paired with
    a static (non-WBC) hold on legs_waist, this lets the whole robot be repositioned
    directly through its base -- "similar to RBY1" in that base motion is decoupled
    from individual leg actuation, though the underlying mechanism differs (RBY1
    drives real holonomic joint actuators; this drives a weld target).
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}floating_base_joint").id
        self._target_pose_body = create_mlspaces_body(
            mj_data, f"{namespace}{HOLO_BASE_TARGET_BODY_NAME}"
        )
        super().__init__(mj_data, base_joint_id, [], [], floating=True)

    @cached_property
    def is_mobile(self):
        return True

    @cached_property
    def n_actuators(self):
        return 7

    @property
    def ctrl(self) -> np.ndarray:
        ret = np.zeros(7)
        ret[:3] = self._target_pose_body.position
        ret[3:] = self._target_pose_body.quat
        return ret

    @ctrl.setter
    def ctrl(self, ctrl: np.ndarray) -> None:
        self._target_pose_body.position = ctrl[:3]
        self._target_pose_body.quat = ctrl[3:]

    @property
    def noop_ctrl(self) -> np.ndarray:
        return self.joint_pos.copy()

    @cached_property
    def ctrl_limits(self) -> np.ndarray:
        ctrl_range = np.empty((self.n_actuators, 2))
        ctrl_range[:, 0] = -np.inf
        ctrl_range[:, 1] = np.inf
        return ctrl_range


class G1LegsWaistGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """The G1's 12 leg joints + 3-DOF waist (yaw, roll, pitch), as one 15-DOF group.

    Legs and waist are combined into a single MoveGroup (rather than kept separate,
    as Phase 2 did) because the Phase 3 whole-body walking controller (see
    `molmo_spaces.controllers.g1_walk.G1WalkController`) computes one coupled
    15-DOF PD-torque law across both -- the waist's control law folds in gravity
    compensation terms that reference the legs' state, and the ONNX walking
    policy's action space treats legs+waist as a single block (matching the
    joint ordering here: legs first, then waist). Order matters: it must match
    the reference policy's training-time joint order.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        suffixes = _LEG_JOINT_SUFFIXES + _WAIST_JOINT_SUFFIXES
        joint_ids = [model.joint(f"{namespace}{n}").id for n in suffixes]
        act_ids = [model.actuator(f"{namespace}walk_{n}").id for n in suffixes]
        self._pelvis_id = model.body(f"{namespace}pelvis").id
        self._waist_leaf_id = model.body(f"{namespace}torso_link").id
        super().__init__(mj_data, joint_ids, act_ids, self._pelvis_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._waist_leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._pelvis_id)


class G1ArmGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """One of the G1's 7-DOF arms, excluding the gripper.

    The left arm has no gripper hardware and is normally commanded to its
    natural hanging pose; the right arm carries the dexterous gripper.
    """

    def __init__(
        self, mj_data: MjData, side: str, base: RobotBaseGroup, namespace: str = ""
    ) -> None:
        model = mj_data.model
        self.side = side
        joint_ids = [model.joint(f"{namespace}{side}_{n}").id for n in _ARM_JOINT_SUFFIXES]
        act_ids = [model.actuator(f"{namespace}walk_{side}_{n}").id for n in _ARM_JOINT_SUFFIXES]
        self._ee_site_id = model.site(f"{namespace}{side}_grasp").id
        self._arm_root_id = model.body(f"{namespace}{side}_shoulder_pitch_link").id
        super().__init__(mj_data, joint_ids, act_ids, self._arm_root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._arm_root_id)


class G1GripperGroup(MJCFFrameMixin, GripperGroup):
    """The G1's right-hand dexterous gripper.

    Two finger joints (right_Joint1_1, right_Joint2_1) are mechanically
    coupled by an MJCF <equality> constraint enforcing joint2 = joint1 (unlike
    RBY1's gripper, which couples finger2 = -finger1). One tendon-driven
    position actuator (right_grip) drives both.
    """

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [
            model.joint(f"{namespace}right_Joint1_1").id,
            model.joint(f"{namespace}right_Joint2_1").id,
        ]
        act_ids = [model.actuator(f"{namespace}right_grip").id]
        self._ee_site_id = model.site(f"{namespace}right_grasp").id
        root_body_id = model.body(f"{namespace}right_dex_base").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = np.array([GRIPPER_OPEN if open else GRIPPER_CLOSED])

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        # Distance is defined as -joint_pos (more negative qpos == more open).
        return -GRIPPER_CLOSED, -GRIPPER_OPEN

    @property
    def inter_finger_dist(self) -> float:
        return float(-self.joint_pos[0])

    @property
    def joint_pos(self) -> np.ndarray:
        return self.mj_data.qpos[self._joint_posadr]

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set joint positions, applying the joint2 = joint1 coupling.

        Args:
            joint_pos: Either a single value (applied to both coupled joints)
                or two values (used directly).
        """
        if len(joint_pos) == 1:
            coupled_pos = np.array([joint_pos[0], joint_pos[0]])
        else:
            coupled_pos = joint_pos
        self.mj_data.qpos[self._joint_posadr] = coupled_pos

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class G1RobotView(RobotView):
    """Implementation of the complete G1 robot (whole-body walking, see G1WalkController).

    No `head` move group (head/cameras are rigidly mounted to the torso) and
    no `left_gripper` move group (no hardware exists for it).
    """

    def __init__(self, mj_data: MjData, namespace: str = "", use_holo_base: bool = False) -> None:
        self._namespace = namespace
        base_cls = G1HoloBaseGroup if use_holo_base else G1BaseGroup
        base = base_cls(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "legs_waist": G1LegsWaistGroup(mj_data, base, namespace=namespace),
            "left_arm": G1ArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": G1ArmGroup(mj_data, "right", base, namespace=namespace),
            "right_gripper": G1GripperGroup(mj_data, base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return "g1"

    @property
    def base(self):
        return self.get_move_group("base")

    def is_close_to(
        self, move_group_ids: list[str], target_pose: list, threshold: float = 0.1
    ) -> bool:
        """Check if the current planar base pose is close to the target pose.

        AStarPlannerPolicy.current_waypoint() calls this with no explicit threshold,
        so this default is what actually governs "waypoint reached" for G1 nav. 0.1
        (looser than FloatingRUMRobotView's 0.05) because G1WalkController's velocity
        tracking has an empirically observed residual steady-state error around
        0.07 combined (x, y, theta) units near a target -- confirmed via the
        interactive shell's rotate()/nav_to() debug traces, where distance plateaus
        there rather than continuing to converge. 0.05 was unreachable in practice.
        """
        return self.distance_to(move_group_ids, target_pose) < threshold

    def distance_to(self, move_group_ids: list[str], target_pose: list) -> float:
        """Calculate the planar (x, y, theta) distance from the base's current pose to a target pose.

        The pelvis's pose is a full 6-DOF transform (see FreeJointRobotBaseGroup.pose),
        unlike a holonomic base's native [x, y, theta] joints, so we project it down to the
        world-frame yaw for comparison against the [x, y, theta] waypoints the A* nav planner uses.
        """
        assert move_group_ids == ["base"], f"Expected ['base'], got {move_group_ids}"
        assert len(target_pose) == 3, f"Expected [x, y, theta] pose, got {target_pose}"
        pose = self.base.pose
        theta = R.from_matrix(pose[:3, :3]).as_euler("xyz")[2]
        x_delta = pose[0, 3] - target_pose[0]
        y_delta = pose[1, 3] - target_pose[1]
        theta_delta = normalize_ang_error(theta - target_pose[2])
        return float(np.linalg.norm(np.array([x_delta, y_delta, theta_delta])))
