import logging
import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from jaxlie import SE3, SO3
from mujoco import MjData, MjModel, MjSpec

from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics
from molmo_spaces.kinematics.parallel.utils import get_joint_zero, joint_difference
from molmo_spaces.molmo_spaces_constants import get_robot_path

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig


logger = logging.getLogger(__name__)
logging.getLogger("jax").setLevel(logging.WARNING)

if "JAX_ENABLE_X64" not in os.environ:
    jax.config.update("jax_enable_x64", True)


def get_error(mjx_model, q0, target_se3: SE3, site_id: int):
    data = mjx.make_data(mjx_model)
    data = data.replace(qpos=q0)
    data: mjx.Data = mjx.kinematics(mjx_model, data)

    ee_se3 = SE3.from_rotation_and_translation(
        SO3.from_matrix(data.site_xmat[site_id].reshape(3, 3)), data.site_xpos[site_id]
    )
    error_twist = (ee_se3.inverse() @ target_se3).log()
    return error_twist


def solve_step(
    mjx_model: mjx.Model,
    q0: jnp.ndarray,
    q: jnp.ndarray,
    target_se3: SE3,
    site_id: int,
    dt: jnp.ndarray,
    posture_weight: jnp.ndarray,
    damping: jnp.ndarray,
):
    jac_fn = jax.jacobian(get_error, 1)

    err: jnp.ndarray = get_error(mjx_model, q, target_se3, site_id)
    J: jnp.ndarray = jac_fn(mjx_model, q, target_se3, site_id)
    singular = jnp.linalg.det(J @ J.T) < 1e-20

    H = J.T @ J + damping * jnp.eye(J.shape[1])
    q_dot = -jnp.linalg.solve(H, J.T @ err)
    q_dot = jnp.where(singular, jnp.zeros_like(q_dot), q_dot)

    # try to keep the solution near the original joint positions
    # we do this by moving within the nullspace of the error Jacobian towards the original joint positions
    U, S, Vt = jnp.linalg.svd(J)
    # we assume a redundant manipulator, so the nullspace is the last columns of V
    null_space = Vt[J.shape[0] :].T
    r = q0 - q
    # null_space has orthonormal columns, so this is equivalent to solving least-squares
    q_dot_posture = null_space @ null_space.T @ r

    q_dot = q_dot + posture_weight * q_dot_posture
    next_q = mjx._src.forward._integrate_pos(mjx_model.jnt_type, q, q_dot, dt)
    return next_q


class FrankaParallelKinematics(ParallelKinematics):
    """
    A Levenberg-Marquardt-based parallel inverse kinematics solver for Franka robots.

    This solver is optimized for parallel execution. For single robot IK, MlSpacesKinematics may be faster.
    """

    def __init__(self, robot_config: "FrankaRobotConfig"):
        """
        Args:
            robot_config: The robot configuration. The base_size must be configured properly.
        """
        super().__init__(robot_config)
        spec = MjSpec()
        robot_xml_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
        self.jax_device = jax.devices("cpu")[0]

        robot_spec = MjSpec.from_file(str(robot_xml_path))
        for body in robot_spec.bodies:
            body: mujoco.MjsBody
            for geom in body.geoms:
                geom: mujoco.MjsGeom
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    robot_spec.delete(geom)

        robot_config.robot_cls.add_robot_to_scene(
            robot_config, spec, robot_spec, "", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        self._mj_model: MjModel = spec.compile()
        self._mjx_model = mjx.put_model(self._mj_model, device=self.jax_device)

        self._mj_data = MjData(self._mj_model)

        self._robot_view = robot_config.robot_view_factory(self._mj_data, "")

        if (
            site_id := mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper/grasp_site"
            )
        ) != -1:
            self._site_id = site_id
        elif (
            site_id := mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        ) != -1:
            self._site_id = site_id
        else:
            raise ValueError("Could not infer grasp site name from model")

        self._solve_step_jit = jax.jit(
            jax.vmap(solve_step, in_axes=(None, 0, 0, 0, None, None, None, None)),
            device=self.jax_device,
        )
        self._get_error_jit = jax.jit(
            jax.vmap(get_error, in_axes=(None, 0, 0, None)),
            device=self.jax_device,
        )

    def warmup_ik(self, batch_size: int):
        """
        JIT the IK solver to incur startup costs up-front.
        The warmed up batch size should match the batch size of the subsequent calls to ik,
        otherwise a recompilation will be triggered.

        Args:
            batch_size: The batch size to warmup.
        """
        data = MjData(self._mj_model)
        robot_view = self._robot_config.robot_view_factory(data, "")
        for mg_id, qpos in self._robot_config.init_qpos.items():
            robot_view.get_move_group(mg_id).joint_pos = qpos
        mujoco.mj_forward(self._mj_model, data)

        pose = np.broadcast_to(
            robot_view.get_move_group("gripper").leaf_frame_to_robot[None], (batch_size, 4, 4)
        )

        self.ik(pose, robot_view.get_qpos_dict(), np.eye(4), rel_to_base=True, max_iter=1)

    def fk(
        self,
        qpos_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        is_batch, batch_size, qpos_dicts, base_poses = self._batchify(qpos_dicts, base_poses)

        ret = [
            {mg_id: np.empty((4, 4)) for mg_id in self._robot_view.move_group_ids()}
            for _ in range(batch_size)
        ]
        for i, qpos_dict in enumerate(qpos_dicts):
            self._robot_view.set_qpos_dict(qpos_dict)
            mujoco.mj_fwdPosition(self._mj_model, self._mj_data)
            for mg_id in self._robot_view.move_group_ids():
                ret[i][mg_id] = self._robot_view.get_move_group(mg_id).leaf_frame_to_robot
                if not rel_to_base:
                    ret[i][mg_id] = base_poses[i] @ ret[i][mg_id]
        return ret if is_batch else ret[0]

    def _solve_step(
        self,
        q0_batch: jnp.ndarray,
        q_batch: jnp.ndarray,
        target_se3: SE3,
        dt: jnp.ndarray,
        posture_weight: jnp.ndarray,
        damping: jnp.ndarray,
    ):
        # block_until_ready makes profiling more accurate, no real effect on performance since sync is necessary anyway
        return jax.block_until_ready(
            self._solve_step_jit(
                self._mjx_model,
                q0_batch,
                q_batch,
                target_se3,
                self._site_id,
                dt,
                posture_weight,
                damping,
            )
        )

    def ik(
        self,
        poses: np.ndarray,
        q0_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
        dt: float = 0.5,
        max_iter: int = 50,
        converge_eps: float = 1e-3,
        success_eps: float = 5e-4,
        damping: float = 1e-12,
        posture_weight: float = 1.0,
    ):
        """
        Finds joint positions that would place the end-effector at the target pose.

        Args:
            pose: The target pose(s) to reach. Shape: (batch_size, 4, 4) or (4, 4)
            q0_dicts: The initial joint positions.
            base_pose: The base pose(s) of the robots. Shape: (batch_size, 4, 4) or (4, 4)
            rel_to_base: Whether the pose(s) are relative to the base frame.
            dt: The time step for integration.
            max_iter: The maximum number of iterations for the solver.
            converge_eps: The threshold for convergence, in joint space.
            success_eps: The threshold for success, in twist space.
            damping: The damping factor for the solver.
            posture_weight: The weight for the posture constraint, relative to the error minimization.
                If the solver frequently gets stuck in local minima, decreasing this value (or setting to 0) may help.

        Returns:
            A list of qpos dictionaries for each robot in the batch, or a single qpos dictionary if unbatched.
            If the solver fails to converge for a given robot, the corresponding qpos dictionary is None.
        """
        is_batch, batch_size, q0_dicts, base_poses, poses = self._batchify(
            q0_dicts, base_poses, poses
        )

        with jax.default_device(self.jax_device):
            q_batch = np.empty((batch_size, self._mjx_model.nq))
            q_batch[:] = get_joint_zero(self._mjx_model)
            for i, q0_dict in enumerate(q0_dicts):
                for mg_id, q0 in q0_dict.items():
                    q_batch[i, self._robot_view.get_move_group(mg_id)._joint_posadr] = q0
            q_batch = jnp.array(q_batch, device=self.jax_device)
            q0_batch = q_batch

            if not rel_to_base:
                poses = np.linalg.inv(base_poses) @ poses

            target_se3 = jax.vmap(SE3.from_matrix)(jnp.array(poses, device=self.jax_device))

            for n_iter in range(max_iter):
                next_q_batch = self._solve_step(
                    q0_batch,
                    q_batch,
                    target_se3,
                    jnp.array(dt, device=self.jax_device),
                    jnp.array(posture_weight, device=self.jax_device),
                    jnp.array(damping, device=self.jax_device),
                )
                q_vel = joint_difference(self._mjx_model, q_batch, next_q_batch) / dt
                q_batch = next_q_batch
                converged = jnp.linalg.norm(q_vel, axis=-1) < converge_eps

                if jnp.all(converged):
                    logger.debug(
                        f"[FrankaParallelKinematics] Batch of size {batch_size} converged in {n_iter} iterations"
                    )
                    break
            else:
                final_vel_norm = jnp.linalg.norm(q_vel, axis=-1)
                logger.debug(
                    f"[FrankaParallelKinematics] Batch of size {batch_size} failed to converge in {max_iter} iterations, {converged=}, {final_vel_norm=}"
                )

            final_twist = self._get_error_jit(self._mjx_model, q_batch, target_se3, self._site_id)
            final_err = jnp.linalg.norm(final_twist, axis=-1)
            success = np.array(final_err < success_eps)
            q_batch = np.array(q_batch)

            if not np.all(success):
                logger.debug(
                    f"[FrankaParallelKinematics] IK failed for indices {np.where(~success)[0]}, final error: {final_err}"
                )

        ret: list[dict[str, np.ndarray] | None] = []
        for i, succ in enumerate(success):
            if succ:
                q_dict = {}
                for mg_id in q0_dicts[i]:
                    q_dict[mg_id] = q_batch[i, self._robot_view.get_move_group(mg_id)._joint_posadr]
                ret.append(q_dict)
            else:
                ret.append(None)

        return ret if is_batch else ret[0]
