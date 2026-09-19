"""The G1 humanoid robot: g1_molmo's components/robot.py on molmo_spaces'
Robot interface, verified byte-identical against the gold pick rollout
(projects/fetchman/scripts/). `G1Config` constructs it through
`from_mj_data`; its whole-body controller is controllers/g1_wbc.py.
"""

import contextlib
import re
from typing import TYPE_CHECKING

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as _R

from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.utils.linalg_utils import normalize_ang_error

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig

XML_PATH = str(ASSETS_DIR / "robots/g1/g1_dex.xml")
PREFIX = "robot_0/"
ROOT_BODY = f"{PREFIX}pelvis"
STANDING_HEIGHT = 0.75

# Waypoint -> base-velocity bridge (see G1Robot.waypoint_to_velocity_target);
# the reference stack emits base velocities directly.
# Width of the "legs_waist" target: [vx, vy, yaw_rate, height, waist(3)].
NUM_LEGS_WAIST_TARGET_DIMS = 7

MAX_LINEAR_VEL = 0.5
MAX_YAW_RATE = 0.5
MIN_LINEAR_VEL = 0.15
MIN_YAW_RATE = 0.15
VELOCITY_DEADBAND = 0.08
YAW_GATE_THRESHOLD = np.radians(30)

# Body-frame +x offset (m) shifting reported "robot xy" forward of the pelvis to better
# match the footprint center. Applied by get_xy()/set_pose() and the controller's _xy().
PELVIS_FORWARD_OFFSET = 0.05

# Dex gripper: positive qpos closes the fingers, negative opens.
GRIPPER_OPEN = -0.0222
GRIPPER_CLOSED = 0.0245

# 30 joints = 12 legs + 3 waist + 14 arms + 1 right gripper (Joint2_1 is <equality>-coupled).
JOINT_NAMES = [
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_Joint1_1",
]

ACTUATOR_NAME_MAP = {
    "right_Joint1_1": "right_grip",
}

# Left arm values are near the gravity-settled hanging pose (the arm is unactuated).
_DEFAULT_QPOS_PATTERNS = {
    r"left_shoulder_pitch_joint": 0.212,
    r"left_shoulder_roll_joint": -0.017,
    r"left_shoulder_yaw_joint": 0.062,
    r"left_elbow_joint": 1.216,
    r"left_wrist_roll_joint": 0.005,
    r"left_wrist_pitch_joint": 0.258,
    r"left_wrist_yaw_joint": 0.006,
    r".*_hip_pitch_joint": -0.312,
    r".*_knee_joint": 0.669,
    r".*_ankle_pitch_joint": -0.363,
    r"right_elbow_joint": -0.2,
    r"right_shoulder_roll_joint": -0.2,
    r"right_shoulder_pitch_joint": 0.2,
    r"right_Joint1_1": GRIPPER_OPEN,
    r"right_Joint2_1": GRIPPER_OPEN,
}


def _resolve_defaults():
    result = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES):
        for pat, val in _DEFAULT_QPOS_PATTERNS.items():
            if re.fullmatch(pat, name):
                result[i] = val
                break
    return result


DEFAULT_QPOS = _resolve_defaults()


def _is_floor_geom(name: str) -> bool:
    name = (name or "").lower()
    return (
        name == "floor" or name.startswith("room|") or name.startswith("room_") or "floor" in name
    )


class G1RobotView:
    """Pose/contact/visibility helpers for the G1 (the reference stack's
    view); G1Robot keeps same-named pass-throughs."""

    def __init__(
        self, model, data, body_id, freejoint_id, cam_ids, dof_ids, namespace: str = PREFIX
    ):
        self.model = model
        self.data = data
        self._body_id = body_id
        self._freejoint_id = freejoint_id
        self._cam_ids = cam_ids
        self._dof_ids = dof_ids
        self._namespace = namespace
        self._renderer = None

    def set_pose(self, xy, yaw, z=STANDING_HEIGHT):
        qadr = self.model.jnt_qposadr[self._freejoint_id]
        c, s = np.cos(yaw), np.sin(yaw)
        px = xy[0] - c * PELVIS_FORWARD_OFFSET
        py = xy[1] - s * PELVIS_FORWARD_OFFSET
        self.data.qpos[qadr : qadr + 3] = [px, py, z]
        self.data.qpos[qadr + 3 : qadr + 7] = _R.from_euler("z", yaw).as_quat(scalar_first=True)

    def zero_velocities(self):
        dadr = self.model.jnt_dofadr[self._freejoint_id]
        self.data.qvel[dadr : dadr + 6] = 0.0
        for i in range(len(JOINT_NAMES)):
            self.data.qvel[self._dof_ids[i]] = 0.0

    def get_xy(self):
        return self.data.xpos[self._body_id, :2].copy()

    def get_yaw(self):
        quat = self.data.xquat[self._body_id]
        return _R.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")[2]

    def pelvis_height(self):
        qadr = self.model.jnt_qposadr[self._freejoint_id]
        return float(self.data.qpos[qadr + 2])

    def has_bad_contacts(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]) or ""
            r1, r2 = n1.startswith(self._namespace), n2.startswith(self._namespace)
            if r1 == r2:
                continue
            scene_geom = g2 if r1 else g1
            if _is_floor_geom(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, scene_geom) or ""):
                continue
            return True
        return False

    def check_object_visibility(self, body_id, threshold=0.00002):
        # threshold = minimum fraction of the 224x224 segmentation frame the
        # object must occupy in at least one camera. ~0.00002 (~1px) restores
        # the original presence check: a hard 0.2% (~100px) gate is unsatisfiable
        # for small/distant objects at far nav spawns, so placement fails after
        # its retry budget and crashes rollout workers. Keep this near 1px.
        if not self._cam_ids:
            return True
        if self._renderer is None:
            # procthor scenes can exceed the default max_geom.
            self._renderer = mujoco.Renderer(
                self.model, 224, 224, max_geom=max(20000, self.model.ngeom * 4)
            )
        m = self.model
        geom_ids = set()
        stack = [body_id]
        while stack:
            bid = stack.pop()
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == bid:
                    geom_ids.add(gid)
            for cbid in range(m.nbody):
                if m.body_parentid[cbid] == bid and cbid != bid:
                    stack.append(cbid)

        for cam_id in self._cam_ids:
            try:
                self._renderer.update_scene(self.data, cam_id)
                self._renderer.enable_segmentation_rendering()
                seg = self._renderer.render()
                self._renderer.disable_segmentation_rendering()
            except IndexError:
                return True  # segid overflow — treat as visible to avoid crashing reset.
            seg0 = seg[:, :, 0]
            vis = np.isin(seg0, list(geom_ids)).sum()  # object's visible pixels
            if vis / seg0.size >= threshold:  # fraction of the frame
                return True
        return False

    def close(self):
        """Free the lazily-created visibility renderer (and its GL/EGL context).
        Must be called before dropping the robot on scene reload — otherwise the
        renderer's framebuffer leaks on the render GPU because __del__-based EGL
        teardown is unreliable, so VRAM creeps to OOM across reloads."""
        r = getattr(self, "_renderer", None)
        if r is not None:
            with contextlib.suppress(Exception):
                r.close()
            self._renderer = None


# IK constants/joint groups mirroring agents/policy_g1ms.py's module-level
# HEIGHT_MIN/HEIGHT_MAX, _WAIST, _hand()/_HANDS -- duplicated here (not
# imported) because policy_g1ms.py imports FROM this module's predecessor
# (components/robot.py) and would create a cycle; kept byte-identical to
# those definitions since G1Robot.kinematics below is a verbatim relocation
# of GraspPolicy._solve_ik, which depends on matching them exactly.
HEIGHT_MIN, HEIGHT_MAX = 0.35, 0.793
IK_DT = 1e-2
IK_HEIGHT_DAMPING = 5e5
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _hand_joints(side):
    return tuple(
        f"{side}_{j}"
        for j in (
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
        )
    )


HAND_ARM_JOINTS = {side: _hand_joints(side) for side in ("left", "right")}
HAND_SITE = {side: f"{side}_grasp" for side in ("left", "right")}


class G1Robot(Robot):
    def __init__(
        self,
        model,
        data,
        env=None,
        low_level=None,
        namespace: str = PREFIX,
        xml_path: str = XML_PATH,
        exp_config: "MlSpacesExpConfig | None" = None,
        gripper_friction: tuple[float, float, float] | None = None,
    ):
        # Robot.__init__ stores its second arg as self.robot_config, but this
        # class is constructed with an explicit (model, data) pair by the
        # reference stack and reads self.exp_config everywhere below, so pass
        # `data` through and set self.exp_config explicitly rather than rely
        # on the (mis-typed, for us) base-class attribute.
        super().__init__(data, exp_config)
        self.exp_config = exp_config
        self.model = model
        self.data = data
        self._env = env
        # Robot ABC solver properties, built lazily (see `kinematics`).
        self._kinematics = None
        self._parallel_kinematics = None
        # Native MoveGroup-based RobotView, built lazily (see `robot_view`).
        self._native_robot_view = None
        # Set by a policy that advances the WBC gait clock itself (see
        # compute_control).
        self._external_gait_clock = False
        # Namespace/asset path are per-instance rather than the module-level
        # PREFIX/XML_PATH the reference stack hardcoded, matching how every
        # other molmo_spaces Robot takes these from its BaseRobotConfig
        # (robot_namespace / get_robot_xml_path). The module constants remain
        # the defaults, so the reference stack's own call sites are unchanged.
        self._namespace = namespace
        self._xml_path = xml_path
        self._root_body = f"{namespace}pelvis"
        self._gripper_friction = gripper_friction
        # The reference stack sets the physics rate in G1Controller.set_env
        # (m.opt.timestep = 0.005), which molmo_spaces' own env never calls --
        # it constructs robots through the factory and drives them through the
        # task. Apply the configured rate here so a natively-constructed G1
        # gets the 5ms step its WBC was trained at, instead of inheriting the
        # scene default (2ms) and failing BaseMujocoTask's
        # control-dt-divisible-by-sim-dt check.
        if exp_config is not None:
            physics_timestep = getattr(exp_config.robot_config, "physics_timestep", None)
            if physics_timestep is not None:
                data.model.opt.timestep = physics_timestep

        # The WBC/PD controller is owned by the robot. `low_level` lets a scene
        # reload reuse the instance (its ONNX sessions are expensive); setup()
        # rebinds its index arrays to the new model either way.
        if low_level is None:
            # Deferred import -- controllers/g1_wbc.py imports JOINT_NAMES/
            # DEFAULT_QPOS from this module, so a module-level import here
            # would be circular.
            from molmo_spaces.controllers.g1_wbc import G1WholeBodyController as _LowLevelController

            low_level = _LowLevelController()
        self._low_level = low_level
        self._low_level.setup(model, data, prefix=self._namespace)

        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._root_body)
        if self._body_id < 0:
            raise RuntimeError(f"G1 root body '{self._root_body}' not found")
        self._freejoint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}floating_base_joint"
        )

        # kinematics() state -- lazily-built mink.Configuration cache (see
        # kinematics' own docstring) plus the full scene joint-name list
        # _solve_ik reads back solved qpos through.
        self._stj = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
        ]
        self._fj_scene = int(model.jnt_qposadr[self._freejoint_id])
        self._ik_cfg = None
        self._ik_mdl = None
        self._ik_fj_qa = None

        # kinematics_wbc() state -- lazily-built standalone-model mink setup,
        # ported from agents/policy_g1ms.py's G1Controller.setup(). Built on
        # first use rather than here since it's WBC-controller-specific
        # (irrelevant to a G1Robot with no agent attached yet).
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

        self._qpos_ids = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{n}")
                ]
                for n in JOINT_NAMES
            ]
        )
        self._dof_ids = np.array(
            [
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{n}")
                ]
                for n in JOINT_NAMES
            ]
        )

        def _find_act(jname):
            act_name = ACTUATOR_NAME_MAP.get(jname, jname)
            aid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}walk_{act_name}"
            )
            if aid < 0:
                aid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}{act_name}"
                )
            return aid

        self.act_ids = np.array([_find_act(n) for n in JOINT_NAMES])

        self.right_gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}right_grip"
        )
        self.left_gripper_aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}left_grip"
        )

        self.n_substeps = max(1, round(0.02 / model.opt.timestep))

        # Visibility uses the egocentric head camera so checks match policy POV.
        self._cam_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{self._namespace}{n}")
            for n in ("head_pov",)
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{self._namespace}{n}") >= 0
        ]
        self._pose_view = G1RobotView(
            model,
            data,
            self._body_id,
            self._freejoint_id,
            self._cam_ids,
            self._dof_ids,
            namespace=self._namespace,
        )

        self._apply_solver_overrides()
        self._fix_contacts()

    @property
    def robot_view(self):
        """Robot ABC's RobotView: the MoveGroup view of robot_views/g1_view.py
        when constructed with an exp_config (molmo_spaces' factory), else the
        reference stack's pose helper view. The pass-through methods always use
        the pose view, whichever is published here.
        """
        if self.exp_config is None:
            return self._pose_view
        if self._native_robot_view is None:
            from molmo_spaces.robots.robot_views.g1_view import (
                G1RobotView as _NativeG1RobotView,
            )

            self._native_robot_view = _NativeG1RobotView(
                self.data,
                self._namespace,
                use_holo_base=getattr(self.exp_config.robot_config, "use_holo_base", False),
            )
        return self._native_robot_view

    @property
    def namespace(self):
        return self._namespace

    @property
    def controllers(self):
        """{move_group_name: Controller}, matching molmo_spaces/robots/
        abstract.py's Robot.controllers."""
        controllers = getattr(self._low_level, "_controllers", None)
        if not controllers:
            return {}
        return {c.move_group.name: c for c in controllers}

    @classmethod
    def apply_control_overrides(cls, spec, robot_config):
        """Actuator/gain configuration is done at runtime by
        G1WholeBodyController.setup() on the compiled MjModel, so this only carries the
        spec-time fix that cannot be done later: the MJCF puts the head/mount/
        logo visual geoms on group 5, which no renderer in this codebase
        enables, so the G1 renders headless. Move them to group 2 like the rest
        of the robot."""
        super().apply_control_overrides(spec, robot_config)
        namespace = robot_config.robot_namespace
        head_mesh_suffixes = ("head_link", "head_mount", "logo_link")
        for body_name in ("torso_link", "head_camera_mount"):
            body = spec.body(f"{namespace}{body_name}")
            if body is None:
                continue
            for geom in body.geoms:
                if geom.group == 5 and geom.meshname.endswith(head_mesh_suffixes):
                    geom.group = 2

    @staticmethod
    def robot_model_root_name() -> str:
        """Robot ABC hook used by the scene builder to find the robot's root
        body in its MJCF. Same body the deleted g1_old_reference used -- both wrapped the same
        g1_dex.xml."""
        return "pelvis"

    def reset(self) -> None:
        """Robot ABC reset: DEFAULT_QPOS (the reference stack's settled standing
        pose, not G1Config.init_qpos) plus the WBC's internal state."""
        self.set_defaults()
        reset_wbc = getattr(self._low_level, "_reset_wbc_state", None)
        if reset_wbc is not None:
            reset_wbc()

    def _legs_waist_target(self, cmd3) -> np.ndarray:
        """[vx, vy, yaw_rate] -> the full legs_waist 7-vector, holding the
        current commanded pelvis height and a level torso."""
        out = np.zeros(NUM_LEGS_WAIST_TARGET_DIMS, dtype=np.float32)
        out[0:3] = np.asarray(cmd3, dtype=np.float32)
        out[3] = float(self._low_level._height_cmd)
        return out

    def create_robot_sensors(self):
        """Robot.create_robot_sensors: this robot's own state sensors, which
        BaseMujocoTask (and fetchman's G1Task) append to the task's suite."""
        from molmo_spaces.env.sensors_g1 import robot_sensors

        return robot_sensors()

    def update_control(self, action_command_dict) -> None:
        """Robot.update_control: a move-group dict, whose "legs_waist" entry is
        [vx, vy, yaw_rate, height, waist(3)], translated to the reference
        controller's (cmd3, height, waist3). `execute_action` is the reference
        stack's own flat-15 path and is untouched.
        """
        # The pelvis has no actuators, so "base" (an absolute [x, y, theta]
        # waypoint) and "base_velocity" ([vx, vy, yaw_rate]) both become a
        # legs_waist command here.
        action_command_dict = dict(action_command_dict)
        waypoint = action_command_dict.pop("base", None)
        base_velocity = action_command_dict.pop("base_velocity", None)
        if waypoint is not None and len(waypoint) == 3:
            action_command_dict["legs_waist"] = self._legs_waist_target(
                self.waypoint_to_velocity_target(waypoint)
            )
        elif base_velocity is not None and len(base_velocity) >= 3:
            bv = np.asarray(base_velocity, dtype=np.float32)
            # Same floored clip the waypoint path applies: a nav policy's
            # smoothstep brake otherwise spends real time in
            # [VELOCITY_DEADBAND, MIN_LINEAR_VEL), the regime the WBC's
            # stand/walk switch gets stuck in.
            action_command_dict["legs_waist"] = self._legs_waist_target(
                np.array(
                    [
                        self._floored_clip(
                            bv[0], MIN_LINEAR_VEL, MAX_LINEAR_VEL, VELOCITY_DEADBAND
                        ),
                        self._floored_clip(
                            bv[1], MIN_LINEAR_VEL, MAX_LINEAR_VEL, VELOCITY_DEADBAND
                        ),
                        self._floored_clip(bv[2], MIN_YAW_RATE, MAX_YAW_RATE, VELOCITY_DEADBAND),
                    ],
                    dtype=np.float32,
                )
            )

        controllers = self.controllers
        for mg_id, target in action_command_dict.items():
            controller = controllers.get(mg_id)
            if controller is None or target is None:
                continue
            target = np.asarray(target, dtype=np.float32)
            if mg_id == "legs_waist":
                if target.shape[0] != NUM_LEGS_WAIST_TARGET_DIMS:
                    raise ValueError(
                        f"legs_waist target must be {NUM_LEGS_WAIST_TARGET_DIMS} values "
                        f"[vx, vy, yaw_rate, height, waist(3)], got {target.shape[0]}"
                    )
                controller.set_target((target[0:3], float(target[3]), target[4:7]))
            else:
                controller.set_target(target if target.size > 1 else float(target[0]))

    def compute_control(self) -> None:
        """Robot.compute_control, dispatching in G1WholeBodyController.execute_action's
        controller order (load-bearing: legs_waist torque uses the previous
        tick's _target_lower before the WBC updates it).
        """
        # Advance the WBC gait clock (ONNX inference every 4th tick) unless the
        # driving policy does it itself (G1PickPlannerPolicy sets
        # _external_gait_clock). Pinned at 0 the network ran every tick and the
        # robot fell over.
        if not self._external_gait_clock:
            self._low_level._step_counter += 1
        data = self.data
        step_counter = self._low_level._step_counter
        for controller in self._low_level._controllers:
            # A controller with no target holds its joints (native policies
            # address only the move groups they care about). legs_waist is
            # exempt: it keeps its command on the owning G1WholeBodyController, so its
            # `target` is always None and skipping it would stop the balancing.
            if controller.target is None and controller.move_group.name != "legs_waist":
                # Hold the current joint positions rather than skipping. These
                # are position actuators (the right arm at kp=2000), so leaving
                # ctrl at whatever was written last drives the arm hard toward a
                # stale target -- enough to throw the robot over during nav.
                # Groups with no joint indices (right_gripper tracks a tendon)
                # have nothing to hold, so leave their ctrl untouched.
                jqpos = controller.move_group.jqpos
                if len(jqpos) == 0:
                    continue
                controller.move_group.set_ctrl(data, controller.move_group.joint_pos(data))
                continue
            values = controller.compute_ctrl_inputs(data, step_counter)
            controller.move_group.set_ctrl(data, np.atleast_1d(values))

    @classmethod
    def from_mj_data(cls, mj_data, exp_config) -> "G1Robot":
        """molmo_spaces' `(mj_data, exp_config)` robot-factory constructor;
        `__init__` keeps the reference stack's explicit `(model, data)` pair."""
        robot_config = exp_config.robot_config
        return cls(
            mj_data.model,
            mj_data,
            namespace=robot_config.robot_namespace,
            xml_path=str(robot_config.get_robot_xml_path()),
            exp_config=exp_config,
        )

    def set_env(self, env):
        """Wires the low-level controller to `env` (mj model opt overrides,
        n_substeps, floor friction -- see g1_wbc.G1WholeBodyController.
        set_env). Called by env_g1ms.py's _load_scene once `self` is
        assigned to `env.robot` (set_env reads env.robot.n_substeps, so it
        can't run from inside __init__, before that assignment exists)."""
        self._env = env
        self._low_level.set_env(env)

    def execute_action(self, action):
        return self._low_level.execute_action(action)

    @staticmethod
    def _floored_clip(value: float, min_mag: float, max_mag: float, deadband: float) -> float:
        """Proportional clip, but floor the magnitude at `min_mag` once |value|
        exceeds `deadband` -- a pure proportional-to-zero law stalls the WBC's
        stand/walk switch (norm(cmd) < 0.05) just short of convergence."""
        if abs(value) <= deadband:
            return 0.0
        return float(np.sign(value)) * float(np.clip(abs(value), min_mag, max_mag))

    def waypoint_to_velocity_target(self, waypoint) -> np.ndarray:
        """Absolute [x, y, yaw] waypoint -> the WBC's [vx, vy, yaw_rate] base
        command, in the robot's own body frame. Same bridge (and same
        constants) as native molmo_spaces/robots/g1.py, which is what lets a
        molmo_spaces navigation policy drive this robot: the reference stack
        has no such bridge because its policy emits base velocities directly.
        """
        xy = self._pose_view.get_xy()
        yaw = self._pose_view.get_yaw()

        dx, dy = waypoint[0] - xy[0], waypoint[1] - xy[1]
        local_vx = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_vy = -np.sin(yaw) * dx + np.cos(yaw) * dy
        yaw_error = normalize_ang_error(waypoint[2] - yaw)

        yaw_rate = self._floored_clip(yaw_error, MIN_YAW_RATE, MAX_YAW_RATE, VELOCITY_DEADBAND)
        if abs(yaw_error) > YAW_GATE_THRESHOLD:
            # Turn-then-drive: while heading is substantially off, correcting it
            # and translating at once fights itself (the gait's own turning drift
            # keeps re-triggering a position correction that never converges).
            vx = vy = 0.0
        else:
            vx = self._floored_clip(local_vx, MIN_LINEAR_VEL, MAX_LINEAR_VEL, VELOCITY_DEADBAND)
            vy = self._floored_clip(local_vy, MIN_LINEAR_VEL, MAX_LINEAR_VEL, VELOCITY_DEADBAND)
        return np.array([vx, vy, yaw_rate], dtype=np.float32)

    def advance_control_clock(self) -> None:
        """Advance the WBC's decimation clock by one tick. The reference policy
        owns the control loop and does this itself; anything else driving the
        robot has to, or the gait clock never ticks."""
        self._low_level._step_counter += 1

    def nav_action(self, waypoint, height=None) -> np.ndarray:
        """A full flat-15 action that walks toward `waypoint` while holding the
        current arm/gripper pose -- the navigation counterpart to the grasp
        policy's own action assembly (see g1_wbc.ACTION_DIM's layout:
        [vx, vy, yaw_rate, height, waist(3), right_arm(7), gripper])."""
        action = np.zeros(15, dtype=np.float32)
        action[0:3] = self.waypoint_to_velocity_target(waypoint)
        action[3] = float(self._low_level._height_cmd if height is None else height)
        # waist stays 0 (level torso); arm/gripper hold their live pose so
        # navigating never disturbs whatever the arm was doing.
        q = self.data.qpos
        action[7:14] = q[self._qpos_ids[22:29]]
        action[14] = q[self._qpos_ids[29]]
        return action

    def _set_groot_defaults(self):
        return self._low_level._set_groot_defaults()

    def solve_scene_ik(
        self, pos, rot=None, hand="right", ik_joints=None, col_limit=None, use_height=True
    ):
        """Full-scene mink IK (gold's GraspPolicy._solve_ik) on the live scene
        model. `ik_joints` and `col_limit` are the caller's, being
        grasp-specific. Returns {unprefixed_joint_name: qpos} for every joint.
        """
        if ik_joints is None:
            ik_joints = set(HAND_ARM_JOINTS[hand]) | set(WAIST_JOINTS)
        site = HAND_SITE[hand]

        if self._ik_cfg is None:
            self._ik_cfg = mink.Configuration(self.model)
            self._ik_mdl = self._ik_cfg.model
            self._ik_fj_qa = self._ik_mdl.jnt_qposadr[
                mujoco.mj_name2id(
                    self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + "floating_base_joint"
                )
            ]

        config = self._ik_cfg
        config.update(self.data.qpos.copy())

        mask = np.zeros(self._ik_mdl.nv)
        for jn in ik_joints:
            jid = mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + jn)
            if jid >= 0:
                mask[self._ik_mdl.jnt_dofadr[jid]] = 1.0
        if use_height:
            fj_dof = self._ik_mdl.jnt_dofadr[
                mujoco.mj_name2id(
                    self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + "floating_base_joint"
                )
            ]
            mask[fj_dof + 2] = 1.0

        ht = mink.FrameTask(
            frame_name=self._namespace + site,
            frame_type="site",
            position_cost=100,
            orientation_cost=1,
            lm_damping=1,
        )
        post = mink.PostureTask(self._ik_mdl, cost=1e-2)
        post.set_target_from_configuration(config)
        r = mink.SO3.from_matrix(rot) if rot is not None else mink.SO3.identity()
        ht.set_target(mink.SE3.from_rotation_and_translation(r, np.asarray(pos, dtype=np.float64)))

        limits = [mink.ConfigurationLimit(self._ik_mdl)]
        if col_limit is not None:
            limits.append(col_limit)
        prev = float("inf")
        for step in range(300):
            try:
                vel = mink.solve_ik(config, [ht, post], IK_DT, "daqp", damping=1e-1, limits=limits)
            except Exception:
                break
            vel *= mask
            config.integrate_inplace(vel, IK_DT)
            q = config.q.copy()
            q[self._ik_fj_qa + 2] = np.clip(q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
            config.update(q)
            err = np.linalg.norm(ht.compute_error(config)[:3])
            if err < 0.001:
                break
            if step == 100 and err > 0.1:
                break
            if step > 100 and err > prev - 1e-5:
                break
            prev = err

        if use_height:
            ik_h = np.clip(config.q[self._ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX)
            self.model.qpos_spring[self._fj_scene + 2] = ik_h
            self.model.dof_damping[self.model.jnt_dofadr[self._freejoint_id] + 2] = (
                IK_HEIGHT_DAMPING
            )

        out = {}
        for jn in self._stj:
            if jn == self._namespace + "floating_base_joint":
                continue
            jid = mujoco.mj_name2id(self._ik_mdl, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            key = jn[len(self._namespace) :] if jn.startswith(self._namespace) else jn
            out[key] = config.q[self._ik_mdl.jnt_qposadr[jid]]
        return out

    def _build_wbc_self_collision_limit(self, ik_model):
        """Exact relocation of G1Controller._build_self_collision_limit onto
        G1Robot -- self-collision limit for kinematics_wbc: arm<->torso/
        pelvis/waist/hip + arm<->arm geom pairs. Returns None if no
        collidable pairs exist."""
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
        """Lazily build the standalone-model mink setup kinematics_wbc needs,
        exact relocation of G1Controller.setup()'s own WBC-IK construction
        block (agents/policy_g1ms.py). Built once per G1Robot instance (this
        robot is itself reconstructed on every scene (re)load -- see
        env_g1ms.py's _load_scene -- matching the old per-setup()-call
        lifetime the ported code had)."""
        if self._wbc_ik_cfg is not None:
            return
        self._wbc_ik_cfg = mink.Configuration(mujoco.MjModel.from_xml_path(self._xml_path))
        ik_model = self._wbc_ik_cfg.model
        ik_fj = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._wbc_ik_fj_dof = ik_model.jnt_dofadr[ik_fj]
        self._wbc_ik_fj_qa = ik_model.jnt_qposadr[ik_fj]
        self._wbc_scene_to_ik_qpos = []
        for jid in range(ik_model.njnt):
            jname = mujoco.mj_id2name(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jname:
                continue
            scene_jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, self._namespace + jname
            )
            if scene_jid < 0:
                continue
            qsz = 7 if ik_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            self._wbc_scene_to_ik_qpos.append(
                (
                    self.model.jnt_qposadr[scene_jid],
                    ik_model.jnt_qposadr[jid],
                    qsz,
                )
            )

        self._wbc_hand_cfg = {}
        for hand in ("right", "left"):
            arm_joints = list(HAND_ARM_JOINTS[hand])
            site = HAND_SITE[hand]
            mask = np.zeros(ik_model.nv)
            for jn in arm_joints + list(WAIST_JOINTS):
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
        for jn in WAIST_JOINTS:
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                posture_cost[ik_model.jnt_dofadr[jid]] = 0.2
        posture_cost[self._wbc_ik_fj_dof + 2] = 0.1
        self._wbc_waist_qa = np.array(
            [
                ik_model.jnt_qposadr[mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
                for jn in WAIST_JOINTS
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
        for jn, lo, hi in [
            ("waist_yaw_joint", -0.5, 0.5),
            ("waist_roll_joint", -0.4, 0.4),
            ("waist_pitch_joint", -0.1, 0.4),
        ]:
            jid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                ik_model.jnt_range[jid] = [lo, hi]
        self._wbc_limits = [mink.ConfigurationLimit(ik_model)]
        self._wbc_self_collision_limit = self._build_wbc_self_collision_limit(ik_model)

    def kinematics_wbc(
        self, target_pos, target_rot=None, hand="right", avoid_self_collision=False, precision=False
    ):
        """gold's G1Controller._solve_ik_wbc: mink IK on a standalone robot-only
        model synced from the live scene qpos. `precision` selects the tighter
        iteration/convergence settings gold used from PHASE_DESCEND on.
        Returns (arm, waist, ik_h, err).
        """
        self._ensure_wbc_ik_setup()
        hcfg = self._wbc_hand_cfg[hand]

        ik_q = np.zeros(self._wbc_ik_cfg.model.nq, dtype=np.float64)
        for scene_qa, ik_qa, qsz in self._wbc_scene_to_ik_qpos:
            ik_q[ik_qa : ik_qa + qsz] = self.data.qpos[scene_qa : scene_qa + qsz]
        self._wbc_ik_cfg.update(ik_q)

        q_post = self._wbc_ik_cfg.q.copy()
        q_post[self._wbc_waist_qa] = 0.0
        self._wbc_posture_task.set_target(q_post)
        # Anchor pelvis xy to current (walking unaffected) but z to standing height
        # so the IK actively wants to stand back up when the wrist target allows it.
        pelvis_T = self._wbc_ik_cfg.get_transform_frame_to_world("pelvis", "body")
        pelvis_pos = pelvis_T.translation().copy()
        pelvis_pos[2] = HEIGHT_MAX
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
                q_tmp[self._wbc_ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX
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
        ik_h = float(np.clip(q[self._wbc_ik_fj_qa + 2], HEIGHT_MIN, HEIGHT_MAX))
        return arm, waist, ik_h, err

    @property
    def kinematics(self):
        """Robot ABC's kinematics solver. Built lazily: the reference stack
        constructs this robot with no exp_config at all (it drives the arm
        through solve_scene_ik/kinematics_wbc, never through
        MlSpacesKinematics), so requiring one up front would break that path.
        Native callers that do pass an exp_config get the same solver every
        other molmo_spaces Robot exposes."""
        if self._kinematics is None:
            self._kinematics = MlSpacesKinematics(self._require_exp_config("kinematics"))
        return self._kinematics

    @property
    def parallel_kinematics(self):
        """Robot ABC's parallel kinematics. DummyParallelKinematics matches
        what native G1Robot uses -- G1 has no batched-IK backend."""
        if self._parallel_kinematics is None:
            self._parallel_kinematics = DummyParallelKinematics(
                self._require_exp_config("parallel_kinematics"), self.kinematics
            )
        return self._parallel_kinematics

    def _require_exp_config(self, what: str):
        if self.exp_config is None:
            raise RuntimeError(
                f"G1Robot.{what} needs an exp_config, but this robot was constructed "
                "without one (the fetchman stack builds it straight from "
                "model/data). Construct it with exp_config= to use this."
            )
        return self.exp_config.robot_config

    def _apply_solver_overrides(self):
        m = self.model
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        m.opt.noslip_iterations = 5
        m.opt.gravity[:] = [0, 0, -9.81]
        m.opt.impratio = 1.0
        m.opt.jacobian = 2  # auto
        m.opt.enableflags = 0
        for gid in range(m.ngeom):
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if _is_floor_geom(gname):
                m.geom_friction[gid] = [1.0, 0.005, 0.0001]
        self._apply_gripper_friction_override()

    def _apply_gripper_friction_override(self):
        """Override the right gripper pad geoms' friction, if requested via
        either the constructor's `gripper_friction=` (the reference stack's
        own (model, data) call sites, which don't pass exp_config -- see
        env_g1ms.py's _make_robot) or exp_config.robot_config.gripper_friction
        (the standard molmo_spaces factory path, from_mj_data). See
        G1Config.gripper_friction.

        The gripper_pad geoms in g1_dex.xml are unnamed -- only grouped by an
        MJCF `class`, which the compiled model doesn't expose -- so they're
        matched here by their distinctive default tangential friction (5.0,
        vs. every other geom in the model) rather than by name.
        """
        friction = self._gripper_friction
        if friction is None and self.exp_config is not None:
            friction = getattr(self.exp_config.robot_config, "gripper_friction", None)
        if friction is None:
            return
        m = self.model
        for gid in range(m.ngeom):
            if abs(m.geom_friction[gid][0] - 5.0) < 1e-9:
                m.geom_friction[gid] = list(friction)

    def _fix_contacts(self):
        m = self.model
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith(self._namespace) and "ankle_roll" in bname:
                if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    m.geom_conaffinity[gid] = 15

    def set_defaults(self):
        self.data.qpos[self._qpos_ids] = DEFAULT_QPOS
        valid = self.act_ids >= 0
        self.data.ctrl[self.act_ids[valid]] = DEFAULT_QPOS[valid]
        for i, name in enumerate(JOINT_NAMES):
            for prefix in ("walk_", "grasp_"):
                aid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self._namespace}{prefix}{name}"
                )
                if aid >= 0:
                    self.data.ctrl[aid] = DEFAULT_QPOS[i]
        if self.right_gripper_aid >= 0:
            self.data.ctrl[self.right_gripper_aid] = GRIPPER_OPEN
        for jname, qval in (
            ("right_Joint1_1", GRIPPER_OPEN),
            ("right_Joint2_1", GRIPPER_OPEN),
        ):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{jname}"
            )
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = qval

    # waist(3) + right arm(7) + right Joint1_1(1).
    _UPPER_RAND_IDX = np.array([12, 13, 14, 22, 23, 24, 25, 26, 27, 28, 29], dtype=np.int64)
    _UPPER_GRIPPER_LOCAL = 10

    def sample_upper_pose(self, np_random, radius=0.15):
        idx = self._UPPER_RAND_IDX
        # Waist gets 0.4x the arm's noise range so the trunk stays closer to neutral.
        per_dim = np.full(len(idx), radius, dtype=np.float64)
        per_dim[:3] = radius * 0.4
        sampled = DEFAULT_QPOS[idx] + np_random.uniform(-per_dim, per_dim)
        sampled[self._UPPER_GRIPPER_LOCAL] = float(np_random.uniform(GRIPPER_OPEN, GRIPPER_CLOSED))
        for i, jidx in enumerate(idx):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}{JOINT_NAMES[jidx]}"
            )
            if jid >= 0:
                lo, hi = self.model.jnt_range[jid]
                sampled[i] = float(np.clip(sampled[i], lo, hi))
        return sampled

    def apply_upper_pose(self, values):
        idx = self._UPPER_RAND_IDX
        self.data.qpos[self._qpos_ids[idx]] = values
        aids = self.act_ids[idx]
        valid = aids >= 0
        if valid.any():
            self.data.ctrl[aids[valid]] = values[valid]
        # Mirror Joint2_1 to Joint1_1 — the equality only enforces during sim, not on init.
        j2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}right_Joint2_1"
        )
        if j2 >= 0:
            self.data.qpos[self.model.jnt_qposadr[j2]] = values[self._UPPER_GRIPPER_LOCAL]

    def apply_arm_pose(self, joints_dict):
        """Apply a {joint_name: qpos_value} dict produced by IK to the scene qpos.
        Only writes joints in JOINT_NAMES (skips legs/freejoint). Also mirrors
        right_Joint2_1 to right_Joint1_1."""
        if not joints_dict:
            return
        for i, jname in enumerate(JOINT_NAMES):
            if jname in joints_dict:
                self.data.qpos[self._qpos_ids[i]] = float(joints_dict[jname])
                aid = int(self.act_ids[i])
                if aid >= 0:
                    self.data.ctrl[aid] = float(joints_dict[jname])
        j2 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._namespace}right_Joint2_1"
        )
        if j2 >= 0 and "right_Joint1_1" in joints_dict:
            self.data.qpos[self.model.jnt_qposadr[j2]] = float(joints_dict["right_Joint1_1"])

    def set_pose(self, xy, yaw, z=STANDING_HEIGHT):
        self._pose_view.set_pose(xy, yaw, z=z)

    def zero_velocities(self):
        self._pose_view.zero_velocities()

    def get_xy(self):
        return self._pose_view.get_xy()

    def get_yaw(self):
        return self._pose_view.get_yaw()

    def pelvis_height(self):
        return self._pose_view.pelvis_height()

    def place(self, xy, yaw):
        self.set_pose(np.asarray(xy, dtype=np.float64), float(yaw))
        self.set_defaults()
        self.zero_velocities()
        mujoco.mj_forward(self.model, self.data)

    def has_bad_contacts(self):
        return self._pose_view.has_bad_contacts()

    def check_object_visibility(self, body_id, threshold=0.00002):
        return self._pose_view.check_object_visibility(body_id, threshold=threshold)

    def close(self):
        """Free the lazily-created visibility renderer (and its GL/EGL context).
        Must be called before dropping the robot on scene reload — otherwise the
        renderer's framebuffer leaks on the render GPU because __del__-based EGL
        teardown is unreliable, so VRAM creeps to OOM across reloads."""
        self._pose_view.close()

    def state_is_finite(self):
        d = self.data
        return np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.ctrl).all()
