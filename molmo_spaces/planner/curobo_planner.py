import copy
import logging

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)

# cuRobo
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.kinematics import KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.sphere_fit import SphereFitType
from curobo.types import GoalToolPose, JointState, Pose

from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.planner.abstract import Planner

# Re-exported for back-compat: the config lives in its own curobo-free module so
# that naming it does not require importing this one (torch + curobo).
from molmo_spaces.planner.curobo_planner_config import CuroboPlannerConfig  # noqa: F401
from molmo_spaces.utils.pose import pose_mat_to_7d

# Keys of the cuRobo v1 robot YAML `kinematics` section that cuRoboV2 no longer accepts.
_V1_ONLY_KINEMATICS_KEYS = (
    "use_usd_kinematics",
    "usd_path",
    "usd_robot_root",
    "isaac_usd_path",
    "usd_flip_joints",
    "usd_flip_joint_limits",
    "ee_link",
    "link_names",
)


def _kinematics_v1_to_v2(kinematics: dict) -> dict:
    """Translate a cuRobo v1 robot YAML `kinematics` dict to the cuRoboV2 format.

    The RB-Y1 cuRobo configs ship in the assets bundle in the v1 format; translating
    here keeps them usable without an assets release.
    """
    kinematics = copy.deepcopy(kinematics)
    if "tool_frames" not in kinematics:
        tool_frames = [kinematics["ee_link"]]
        tool_frames += [
            link for link in kinematics.get("link_names") or [] if link not in tool_frames
        ]
        kinematics["tool_frames"] = tool_frames
    for key in _V1_ONLY_KINEMATICS_KEYS:
        kinematics.pop(key, None)
    cspace = kinematics["cspace"]
    if "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
    kinematics["format_version"] = 2.0
    return kinematics


class CuroboPlanner(Planner):
    def __init__(
        self,
        config: CuroboPlannerConfig,
    ) -> None:
        self.config = config
        self.world_config = (
            self.config.world_config if self.config.world_config is not None else Scene()
        )
        self._build_robot_config(
            self.config.curobo_robot_config_path, self.config.kinematics_config
        )

        # Curobo motion planner parameters
        self.interpolation_dt = self.config.interpolation_dt
        self.collision_activation_distance = self.config.collision_activation_distance
        self.max_attempts = self.config.max_attempts
        self.collision_cache = self.config.collision_cache
        self.num_ik_seeds = self.config.num_ik_seeds
        self.num_trajopt_seeds = self.config.num_trajopt_seeds
        self.finetune_attempts = 1 if self.config.enable_finetune_trajopt else 0

        self.attached_object2link_dict = {}
        self.attached_link2object_dict = {}
        self._build_motion_gen()

    def reset(self) -> None:
        """
        Reset the motion generator.
        """
        for planner in self._all_planners():
            planner.reset_seed()
        self.attached_object2link_dict = {}
        self.attached_link2object_dict = {}
        torch.cuda.empty_cache()

    def _all_planners(self) -> list:
        """The single-query planner plus the batch planner, if one has been built."""
        planners = [self.motion_planner]
        if self.batch_motion_planner is not None:
            planners.append(self.batch_motion_planner)
        return planners

    def _get_curobo_config_paths(self) -> dict:
        # Use config paths if provided, otherwise default to rby1 for backward compatibility
        paths = {
            "urdf_path": (
                self.config.urdf_path
                if self.config.urdf_path is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config/urdf/model_holobase.urdf")
            ),
            "asset_root_path": (
                self.config.asset_root_path
                if self.config.asset_root_path is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config/urdf/meshes")
            ),
            "collision_spheres": (
                self.config.collision_spheres_path
                if self.config.collision_spheres_path is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config/rby1m_holobase_spheres.yml")
            ),
        }

        return paths

    def _build_robot_config(
        self, curobo_robot_config_path: str, kinematics_config: dict = None
    ) -> None:
        with open(curobo_robot_config_path) as f:
            curobo_robot_config_dict = yaml.safe_load(f)["robot_cfg"]
        if kinematics_config is None:
            kinematics_config = {}
        kinematics_config.update(self._get_curobo_config_paths())
        if "robot_cfg" in curobo_robot_config_dict:
            curobo_robot_config_dict = curobo_robot_config_dict["robot_cfg"]

        if kinematics_config is not None:
            for config_key in kinematics_config:
                curobo_robot_config_dict["kinematics"][config_key] = kinematics_config[config_key]

        # Merge lock_joints into kinematics if provided
        if self.config.lock_joints is not None:
            if "lock_joints" not in curobo_robot_config_dict["kinematics"]:
                curobo_robot_config_dict["kinematics"]["lock_joints"] = {}
            curobo_robot_config_dict["kinematics"]["lock_joints"].update(self.config.lock_joints)

        curobo_robot_config_dict["kinematics"] = _kinematics_v1_to_v2(
            curobo_robot_config_dict["kinematics"]
        )

        self.curobo_robot_config_path = curobo_robot_config_path
        self.curobo_robot_config_dict = curobo_robot_config_dict  # Store the dict for later use
        self.attached_link_names = list(
            curobo_robot_config_dict["kinematics"]["extra_links"].keys()
        )

    def _motion_planner_cfg(self, max_batch_size: int = 1) -> MotionPlannerCfg:
        return MotionPlannerCfg.create(
            robot={"robot_cfg": copy.deepcopy(self.curobo_robot_config_dict)},
            scene_model=None,
            collision_cache=self.collision_cache,
            self_collision_check=True,
            num_ik_seeds=self.num_ik_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            use_cuda_graph=False,
            optimizer_collision_activation_distance=self.collision_activation_distance,
            interpolation_dt=self.interpolation_dt,
            max_batch_size=max_batch_size,
        )

    def _build_motion_gen(self) -> None:
        self.motion_planner = MotionPlanner(self._motion_planner_cfg())
        self.motion_planner.update_world(self.world_config)
        self.motion_planner.warmup(enable_graph=False)
        # Built lazily by plan_batch: BatchMotionPlanner allocates a fixed batch size.
        self.batch_motion_planner = None

        self.joint_names = self.motion_planner.joint_names
        self.default_config = self.motion_planner.default_joint_state.position
        self.joint_limits = self.motion_planner.kinematics.get_joint_limits().position.cpu().numpy()
        self.link_names = self.motion_planner.tool_frames
        self.ee_link_name = self.link_names[0]

    def _get_batch_motion_planner(self, batch_size: int) -> BatchMotionPlanner:
        if self.batch_motion_planner is None or self.batch_motion_planner.batch_size < batch_size:
            if self.batch_motion_planner is not None:
                self.batch_motion_planner.destroy()
            log.info(f"Building cuRobo batch motion planner for batch size {batch_size}")
            self.batch_motion_planner = BatchMotionPlanner(self._motion_planner_cfg(batch_size))
            self.batch_motion_planner.update_world(self.world_config)
            self.batch_motion_planner.warmup(enable_graph=False)
            if self.attached_object2link_dict:
                log.warning(
                    "Batch motion planner was built after objects were attached; "
                    "attachments are not replayed onto it."
                )
        return self.batch_motion_planner

    def _start_state(self, joint_positions: torch.Tensor) -> JointState:
        joint_state = JointState.from_position(joint_positions, joint_names=self.joint_names)
        return joint_state.reorder(self.motion_planner.joint_names)

    def update_locked_joints(self, lock_joints: dict) -> None:
        """Change the values of already-locked joints without rebuilding the planners."""
        kinematics = self.curobo_robot_config_dict["kinematics"]
        kinematics.setdefault("lock_joints", {}).update(lock_joints)
        updated = KinematicsCfg.from_data_dict(copy.deepcopy(kinematics))
        torch.cuda.synchronize()
        for planner in self._all_planners():
            planner.kinematics.update_kinematics_config(updated.kinematics_config)
            planner.reset_seed()

    def plan_batch(
        self,
        joint_positions: list[list],
        goal_pose_lists: list[list],
        kinematics_config: dict = None,
        scene_objects: list[MlSpacesObject] = None,
        verbose: bool = False,
    ):
        """Plan motion for multiple start states and goal poses in parallel.

        Args:
            joint_positions: List of joint position lists, one for each query in the batch.
                Each joint position list should follow the same format as motion_plan.
            goal_pose_lists: List of goal pose lists (7D: [x, y, z, qw, qx, qy, qz]), one for each query.
                For now, only single end-effector goals are supported (no additional link poses).
            kinematics_config: Optional kinematics configuration dict (rebuilds planner if provided).
            scene_objects: Optional list of scene objects for collision checking.
            verbose: If True, prints detailed debug information about the planning result.

        Returns:
            tuple: (successes, trajectories) where successes is a list of bools and
                trajectories is a list of joint position waypoint lists (empty for failed
                queries).
        """
        # Update the kinematics config if provided
        if kinematics_config is not None:
            print(
                "[WARNING] Rebuilding motion gen with new kinematics config. This will take a while."
            )
            self._build_robot_config(self.curobo_robot_config_path, kinematics_config)
            self._build_motion_gen()

        batch_size = len(joint_positions)
        assert len(goal_pose_lists) == batch_size, (
            "Number of joint positions and goal poses must match"
        )
        planner = self._get_batch_motion_planner(batch_size)

        # Update scene objects if provided
        if scene_objects is not None:
            if verbose:
                print("Updating scene objects...")
            excluded_scene_objects = []
            for scene_object in scene_objects:
                if scene_object.name not in self.attached_object2link_dict:
                    excluded_scene_objects.append(scene_object)
            self.update_world(excluded_scene_objects)

        # BatchMotionPlanner solves a fixed batch size: pad by repeating the last query.
        pad = planner.batch_size - batch_size
        joint_positions = list(joint_positions) + [joint_positions[-1]] * pad
        goal_pose_lists = list(goal_pose_lists) + [goal_pose_lists[-1]] * pad

        positions_tensor = torch.tensor(joint_positions, dtype=torch.float32, device="cuda")
        start_state = self._start_state(positions_tensor)
        goal_poses_tensor = torch.tensor(goal_pose_lists, dtype=torch.float32, device="cuda")
        goal_pose = GoalToolPose.from_poses(
            {
                self.ee_link_name: Pose(
                    position=goal_poses_tensor[:, :3], quaternion=goal_poses_tensor[:, 3:]
                )
            },
            ordered_tool_frames=self.link_names,
        )

        result = planner.plan_pose(
            goal_pose,
            start_state,
            max_attempts=self.max_attempts,
            enable_graph_attempt=self.max_attempts,  # never: v1 ran with enable_graph=False
            finetune_attempts=self.finetune_attempts,
        )
        # Free GPU input tensors — they are no longer needed after planning completes.
        del start_state, goal_pose, positions_tensor, goal_poses_tensor

        if result is None:
            successes = [False] * batch_size
            trajectories = [[] for _ in range(batch_size)]
        else:
            successes = result.success.reshape(planner.batch_size, -1).any(dim=-1)
            successes = successes[:batch_size].cpu().tolist()
            # The optimized (not interpolated) plan, as v1's `optimized_plan` was.
            plan = result.js_solution.reorder(self.joint_names).position
            if plan.ndim == 4:  # [batch, seeds, horizon, dof]: seeds are ranked, take the best
                plan = plan[:, 0]
            plan = plan[:batch_size].cpu()
            trajectories = [plan[i].tolist() if ok else [] for i, ok in enumerate(successes)]

        if verbose:
            print("=== Curobo Batch Plan Result ===")
            print(f"Batch size: {batch_size}")
            print(f"Successes: {np.sum(successes)}/{batch_size}")
            if result is not None:
                print("Position errors:", result.position_error)
                print("Rotation errors:", result.rotation_error)
                print("Solve time:", result.solve_time)
            print("==========================")

        del result
        return successes, trajectories

    def motion_plan(
        self,
        joint_position: list,
        goal_pose_lists: list[list],
        kinematics_config: dict = None,
        scene_objects: list[MlSpacesObject] = None,
        verbose: bool = False,
    ) -> list:
        """
        Args:
            joint_position: A list of joint positions representing the current state of the robot.
                - Only include the subset of joints that are relevant to the IK/planning task.
                - For example, if solving full-body IK, provide all joint positions (e.g., a list of length 20);
                  if solving single-arm IK, provide only the arm's joint positions (e.g., a list of length 7).
            goal_pose_lists: A list of goal poses to plan for. The first pose is for the main end-effector,
                and additional poses (if any) are assigned to the remaining link names in `self.link_names[1:]`.
            verbose: If True, prints detailed debug information about the planning result.

        Returns:
            trajectory: A list of joint position waypoints representing the planned trajectory.
            result: The cuRobo TrajOptSolverResult, or None if IK found no solution.
        """
        # Update the kinematics config with the lock joints
        if kinematics_config is not None:
            print(
                "[WARNING] Rebuilding motion gen with new kinematics config. This will take a while."
            )
            self._build_robot_config(self.curobo_robot_config_path, kinematics_config)
            self._build_motion_gen()

        if scene_objects is not None:
            if verbose:
                print("Updating scene objects...")

            # exclude the attached objects from the world config
            excluded_scene_objects = []
            for scene_object in scene_objects:
                if scene_object.name not in self.attached_object2link_dict:
                    excluded_scene_objects.append(scene_object)
            self.update_world(excluded_scene_objects)

        position = torch.tensor(joint_position).unsqueeze(0).float().cuda()
        start_state = self._start_state(position)

        # In bimanual case, left arm is the main arm.
        goal_poses = {}
        for link_name, goal_pose in zip(self.link_names, goal_pose_lists):
            goal_poses[link_name] = Pose.from_list(goal_pose)
        goal = GoalToolPose.from_poses(goal_poses, ordered_tool_frames=self.link_names)

        result = self.motion_planner.plan_pose(
            goal,
            start_state,
            max_attempts=self.max_attempts,
            enable_graph_attempt=self.max_attempts,  # never: v1 ran with enable_graph=False
            finetune_attempts=self.finetune_attempts,
        )
        # Free GPU input tensors — they are no longer needed after planning completes.
        del start_state, goal, goal_poses, position

        trajectory = []
        if result is not None and result.success.any().item():
            traj = result.get_interpolated_plan().reorder(self.joint_names)
            trajectory = traj.position.reshape(-1, len(self.joint_names)).cpu().tolist()
            del traj
        else:
            trajectory.append(joint_position)
            if verbose:
                print("=== Curobo Plan Result ===")
                if result is None:
                    print("No IK solution found")
                else:
                    print("Position error:", result.position_error)
                    print("Rotation error:", result.rotation_error)
                    print("Cspace error:", result.cspace_error)
                    print("Solve time:", result.solve_time)
                    print("Debug info:", result.debug_info)
                print("==========================")
                print(self.joint_limits)
                print(joint_position)
                print(self.joint_names)
                print("==========================")
        return trajectory, result

    def update_world_obstacle_poses(
        self,
        obstacle_names_to_poses: dict[str, np.ndarray],
        pre_invert_poses: bool = True,
    ) -> None:
        # pre_invert_poses was a v1 speed-up that needed a patched cuRobo; cuRoboV2 only
        # takes world-frame poses, so it is accepted and ignored.
        for obstacle_name, pose in obstacle_names_to_poses.items():
            for planner in self._all_planners():
                planner.scene_collision_checker.update_obstacle_pose(
                    obstacle_name, Pose.from_list(list(pose))
                )

    def enable_obstacles(
        self,
        obstacle_names: list[str],
        enable: bool,
    ) -> None:
        # Enable or disable obstacles in the collision checker
        for obstacle_name in obstacle_names:
            for planner in self._all_planners():
                planner.scene_collision_checker.enable_obstacle(obstacle_name, enable)

    def set_world(self, world_config: Scene) -> None:
        """Replace the collision world of every planner."""
        for planner in self._all_planners():
            planner.update_world(world_config)
        self.world_config = world_config

    def update_world(
        self,
        scene_objects: list[MlSpacesObject],
    ) -> None:
        curobo_cuboids = []
        for object in scene_objects:
            if object.name == "scene/table":
                curobo_cuboids.append(
                    Cuboid(
                        name=object.name,
                        pose=pose_mat_to_7d(object.pose).tolist(),
                        dims=(np.array([0.3, 0.4, 0.02]) * 2).tolist(),
                    )
                )
            else:
                curobo_cuboids.append(
                    Cuboid(
                        name=object.name,
                        pose=pose_mat_to_7d(object.pose).tolist(),
                        dims=(object.aabb_size * 2).tolist(),
                    )
                )
        self.set_world(Scene(cuboid=curobo_cuboids))

    def attach_obj(
        self,
        object_names: list[str],
        joint_position: list,
        attach_link_names: list[str],
    ) -> None:
        for link_name in attach_link_names:
            if link_name not in self.attached_link_names:
                raise ValueError(f"Link name {link_name} not found in attached link names")

        joint_position = torch.tensor(joint_position).unsqueeze(0).float().cuda()
        cu_js = self._start_state(joint_position)

        for object_name, link_name in zip(object_names, attach_link_names, strict=True):
            # Check what objects are in the world before attaching
            world_objs = [c.name for c in self.world_config.cuboid or []]
            world_objs += [m.name for m in self.world_config.mesh or []]
            log.debug(f"Objects in CuRobo world before attach: {world_objs}")
            if object_name not in world_objs:
                log.warning(f"Object '{object_name}' not found in CuRobo world model!")

            log.debug(f"Attaching '{object_name}' to link '{link_name}'")
            for planner in self._all_planners():
                planner.attachment_manager.attach_from_scene(
                    cu_js,
                    [object_name],
                    link_name=link_name,
                    sphere_fit_type=SphereFitType.VOXEL,
                )
            self.attached_object2link_dict[object_name] = link_name
            self.attached_link2object_dict[link_name] = object_name
            log.debug(f"Successfully attached '{object_name}' to '{link_name}'")

    def detach_obj(self, attach_link_names: list[str]) -> None:
        for link_name in attach_link_names:
            if link_name not in self.attached_link_names:
                raise ValueError(f"Link name {link_name} not found in attached link names")

        for link_name in attach_link_names:
            for planner in self._all_planners():
                # v1 left the detached object disabled in the world; keep that behavior.
                planner.attachment_manager.detach(link_name, enable_obstacle_names=[])
            del self.attached_object2link_dict[self.attached_link2object_dict[link_name]]
            del self.attached_link2object_dict[link_name]

    def fk_solve(
        self,
        joint_config: list,
    ) -> np.ndarray:
        """Compute forward kinematics for a given joint configuration.

        Args:
            joint_config: Joint configuration as a list.

        Returns:
            np.ndarray: TCP pose in 7D format [x, y, z, qw, qx, qy, qz].
        """
        q_tensor = torch.tensor(joint_config).unsqueeze(0).float().cuda()
        kin_state = self.motion_planner.compute_kinematics(self._start_state(q_tensor))
        ee_pose = kin_state.tool_poses.get_link_pose(self.ee_link_name)
        position = ee_pose.position.reshape(-1, 3)[0].cpu().numpy()
        quaternion = ee_pose.quaternion.reshape(-1, 4)[0].cpu().numpy()  # [qw, qx, qy, qz]
        return np.concatenate([position, quaternion])

    def _set_ik_collision_costs(self, enable: bool) -> None:
        for rollout in self.motion_planner.ik_solver.get_all_rollout_instances():
            for manager in (
                rollout.cost_manager,
                rollout.constraint_manager,
                rollout.hybrid_cost_constraint_manager,
                rollout.metrics_cost_manager,
                rollout.metrics_constraint_manager,
                rollout.metrics_hybrid_cost_constraint_manager,
            ):
                if manager is None:
                    continue
                for name in ("scene_collision", "self_collision"):
                    if manager.has_cost(name):
                        if enable:
                            manager.enable_cost_component(name)
                        else:
                            manager.disable_cost_component(name)

    def ik_solve(
        self,
        goal_pose: list,
        seed_config: list | None = None,
        return_seeds: int = 1,
        disable_collision: bool = False,
    ) -> tuple:
        """Solve inverse kinematics for a given goal pose.

        Args:
            goal_pose: Goal pose in 7D format [x, y, z, qw, qx, qy, qz].
            seed_config: Optional seed joint configuration. If provided, used as initial guess.
            return_seeds: Number of IK solutions to return.
            disable_collision: If True, ignore collision checking during IK solve.

        Returns:
            tuple: (joint_config, success) where joint_config is a list of joint positions
                   if successful, None otherwise.
        """
        goal = GoalToolPose.from_poses(
            {self.ee_link_name: Pose.from_list(goal_pose)}, ordered_tool_frames=self.link_names
        )

        # Prepare seed config if provided
        # Shape must be (batch, num_seeds, dof) for curobo IK solver
        seed_config_tensor = None
        if seed_config is not None:
            # unsqueeze twice: once for batch dim, once for num_seeds dim
            seed_config_tensor = torch.tensor(seed_config).unsqueeze(0).unsqueeze(0).float().cuda()

        # Disable collision checking if requested
        if disable_collision:
            self._set_ik_collision_costs(enable=False)

        try:
            ik_result = self.motion_planner.ik_solver.solve_pose(
                goal,
                seed_config=seed_config_tensor,
                return_seeds=return_seeds,
            )
        finally:
            # Re-enable collision checking
            if disable_collision:
                self._set_ik_collision_costs(enable=True)

        success = ik_result.success.reshape(-1)[0].item()
        if success:
            # Extract the solution
            joint_config = ik_result.solution[0].squeeze().cpu().tolist()
            del ik_result
            return joint_config, None
        else:
            log.debug(
                f"IK solve failed: pos_err={ik_result.position_error.reshape(-1)[0].item():.4f}, "
                f"rot_err={ik_result.rotation_error.reshape(-1)[0].item():.4f}"
            )
            del ik_result
            return None, None
