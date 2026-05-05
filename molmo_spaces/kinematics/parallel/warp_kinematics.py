from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING
from collections import OrderedDict
import logging

import numpy as np
import mujoco
from mujoco import MjSpec, MjModel, MjData
import mujoco_warp as mjw
import warp as wp

from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.robot_views.abstract import GripperGroup, SimplyActuatedMoveGroup, SimpleMoveGroup

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

logger = logging.getLogger(__name__)


@dataclass
class IKBuffers:
    """Preallocated buffers for the IK solver"""
    pos_err: wp.array(dtype=wp.vec3f)
    rot_err: wp.array(dtype=wp.vec3f)
    jacp: wp.array3d(dtype=float)
    jacr: wp.array3d(dtype=float)
    q_dot: wp.array2d(dtype=float)
    dq: wp.array2d(dtype=float)


@dataclass
class SolverData:
    data: mjw.Data
    ik_buffers: IKBuffers
    ik_capture: wp.ScopedCapture | None = None


@wp.kernel
def get_err(
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    site_id: int,
    poses: wp.array(dtype=wp.mat44f),
    pos_err: wp.array(dtype=wp.vec3f),
    rot_err: wp.array(dtype=wp.vec3f),
):
    """Calculate the body-frame position and rotation error between the current and target poses"""
    i = wp.tid()
    site_pos = site_xpos[i, site_id]
    site_rotmat = site_xmat[i, site_id]
    target_pose = poses[i]

    site_pose = wp.mat44(
        site_rotmat[0, 0], site_rotmat[0, 1], site_rotmat[0, 2], site_pos[0],
        site_rotmat[1, 0], site_rotmat[1, 1], site_rotmat[1, 2], site_pos[1],
        site_rotmat[2, 0], site_rotmat[2, 1], site_rotmat[2, 2], site_pos[2],
        0.0,       0.0,       0.0,       1.0,
    )
    err_trf = wp.inverse(site_pose) @ target_pose

    rotmat = wp.mat33(
        err_trf[0, 0], err_trf[0, 1], err_trf[0, 2],
        err_trf[1, 0], err_trf[1, 1], err_trf[1, 2],
        err_trf[2, 0], err_trf[2, 1], err_trf[2, 2],
    )
    q = wp.quat_from_matrix(rotmat)
    axis, angle = wp.quat_to_axis_angle(q)
    t = wp.vec3f(err_trf[0, 3], err_trf[1, 3], err_trf[2, 3])

    if wp.abs(angle) < 1e-6:
        pos_err[i] = site_rotmat @ t
        rot_err[i] = wp.vec3f()
    else:
        w = axis * angle
        V = (
            wp.identity(3, dtype=wp.float32)
            + (1.0 - wp.cos(angle)) / angle**2.0 * wp.skew(w)
            + (angle - wp.sin(angle)) / angle**3.0 * wp.skew(w) @ wp.skew(w)
        )
        t = wp.inverse(V) @ t
        pos_err[i] = site_rotmat @ t
        rot_err[i] = site_rotmat @ w


mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def cholesky_solve6(H: mat66f, b: vec6f) -> vec6f:
    """Solve Hx=b via Cholesky decomposition for 6x6 symmetric positive-definite matrix"""
    L = mat66f()
    for i in range(6):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = wp.sqrt(H[i, i] - s)
            else:
                L[i, j] = (H[i, j] - s) / L[j, j]

    # Forward substitution: L @ y = b
    y = vec6f()
    for i in range(6):
        s = float(0.0)
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = (b[i] - s) / L[i, i]

    # Backward substitution: L^T @ x = y
    x = vec6f()
    for i in range(5, -1, -1):
        s = float(0.0)
        for k in range(i + 1, 6):
            s += L[k, i] * x[k]
        x[i] = (y[i] - s) / L[i, i]

    return x


@wp.kernel
def lm_step(
    jacp: wp.array3d(dtype=float),
    jacr: wp.array3d(dtype=float),
    pos_err: wp.array(dtype=wp.vec3f),
    rot_err: wp.array(dtype=wp.vec3f),
    damping: float,
    nv: int,
    dq: wp.array2d(dtype=float),
):
    """Single step of the Levenberg-Marquardt solver."""
    i = wp.tid()

    err = vec6f(
        pos_err[i][0], pos_err[i][1], pos_err[i][2],
        rot_err[i][0], rot_err[i][1], rot_err[i][2],
    )

    # H = J @ J^T + damping * I, where J = [jacp; jacr] is (6, nv)
    H = mat66f()
    for a in range(6):
        for b in range(6):
            val = float(0.0)
            for k in range(nv):
                Ja = jacp[i, a, k] if a < 3 else jacr[i, a - 3, k]
                Jb = jacp[i, b, k] if b < 3 else jacr[i, b - 3, k]
                val += Ja * Jb
            if a == b:
                val += damping
            H[a, b] = val

    # x = H^{-1} @ err
    x = cholesky_solve6(H, err)

    # q_dot = J^T @ x * dt
    for k in range(nv):
        val = float(0.0)
        for a in range(3):
            val += jacp[i, a, k] * x[a]
            val += jacr[i, a, k] * x[a + 3]
        dq[i, k] = val


class SimpleWarpKinematics(ParallelKinematics):
    """
    A warp-based general-purpose parallel inverse kinematics solver for robots.
    This solver only supports optimizing `SimplyActuatedMoveGroups` to reach a target pose for a given `SimpleMoveGroup`.
    """
    def __init__(self, robot_config: "BaseRobotConfig", device: str = "cpu"):
        super().__init__(robot_config)
        self._device = device
        self._datas: dict[int, SolverData] = {}

        spec = MjSpec()
        robot_xml_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
        robot_spec = MjSpec.from_file(str(robot_xml_path))
        robot_config.robot_cls.add_robot_to_scene(
            robot_config, spec, robot_spec, "", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        self._mj_model: MjModel = spec.compile()
        self._mjw_model = mjw.put_model(self._mj_model)

        mj_data = MjData(self._mj_model)
        self._robot_view = robot_config.robot_view_factory(mj_data, "")

        self._actuated_move_groups: OrderedDict[str, SimplyActuatedMoveGroup] = OrderedDict()
        for mg_id in self._robot_view.move_group_ids():
            mg = self._robot_view.get_move_group(mg_id)
            assert mg.n_joints == 0 or isinstance(mg, SimplyActuatedMoveGroup) or isinstance(mg, GripperGroup)
            if isinstance(mg, SimplyActuatedMoveGroup):
                self._actuated_move_groups[mg_id] = mg

        if self._mj_model.nq != self._mj_model.nv:
            raise ValueError("Number of position variables (nq) must equal number of velocity variables (nv) for warp-based IK solver")

    @cache
    def _get_data(self, batch_size: int) -> SolverData:
        data = mjw.make_data(self._mj_model, nworld=batch_size)
        return SolverData(
            data,
            IKBuffers(
                pos_err=wp.zeros(batch_size, dtype=wp.vec3f),
                rot_err=wp.zeros(batch_size, dtype=wp.vec3f),
                jacp=wp.zeros((batch_size, 3, self._mj_model.nv), dtype=wp.float32),
                jacr=wp.zeros((batch_size, 3, self._mj_model.nv), dtype=wp.float32),
                q_dot=wp.zeros((batch_size, self._mj_model.nv), dtype=wp.float32),
                dq=wp.zeros((batch_size, self._mj_model.nv), dtype=wp.float32),
            ),
        )

    def _dicts_to_qpos_arr(self, qpos_dicts: list[dict[str, np.ndarray]]) -> np.ndarray:
        ret = np.empty((len(qpos_dicts), self._mj_model.nq), dtype=np.float32)
        for i, qpos_dict in enumerate(qpos_dicts):
            for mg_id, mg in self._actuated_move_groups.items():
                ret[i, mg.joint_posadr] = qpos_dict[mg_id]
        return ret

    def _qpos_arr_to_dicts(self, qpos_arr: np.ndarray) -> list[dict[str, np.ndarray]]:
        ret = [{} for _ in range(qpos_arr.shape[0])]
        for i, qpos_dict in enumerate(ret):
            idx = 0
            for mg_id, mg in self._actuated_move_groups.items():
                qpos_dict[mg_id] = qpos_arr[i, idx:idx+mg.n_joints]
                idx += mg.n_joints
        return ret

    def warmup_ik(self, batch_size: int):
        self._get_data(batch_size)

        mj_data = MjData(self._mj_model)
        robot_view = self._robot_config.robot_view_factory(mj_data, "")
        for mg_id, qpos in self._robot_config.init_qpos.items():
            robot_view.get_move_group(mg_id).joint_pos = qpos
        mujoco.mj_forward(self._mj_model, mj_data)

        mg_id = next(name for name, mg in self._actuated_move_groups.items() if isinstance(mg, SimpleMoveGroup))
        pose = np.broadcast_to(
            robot_view.get_move_group(mg_id).leaf_frame_to_robot[None], (batch_size, 4, 4)
        )

        self.ik(pose, robot_view.get_qpos_dict(), np.eye(4), rel_to_base=True, max_iter=1, move_group_id=mg_id)

    def fk(
        self,
        qpos_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Compute forward kinematics for all simple move groups.
        Non simple move groups (e.g. grippers) are not included in the output.

        Args:
            qpos_dicts: The joint positions.
            base_poses: The base pose(s) of the robots. Shape: (batch_size, 4, 4) or (4, 4)
            rel_to_base: Whether the returned pose(s) should be relative to the base frame.

        Returns:
            A list of qpos dictionaries for each robot in the batch, or a single qpos dictionary if unbatched.
        """
        is_batch, batch_size, qpos_dicts, base_poses = self._batchify(qpos_dicts, base_poses)
        solver_data = self._get_data(batch_size)
        data = solver_data.data

        qpos_arr = self._dicts_to_qpos_arr(qpos_dicts)
        wp.copy(data.qpos, wp.from_numpy(qpos_arr))

        mjw.fwd_position(self._mjw_model, data)

        dol = {}
        for mg_id, mg in self._actuated_move_groups.items():
            if isinstance(mg, SimpleMoveGroup):
                trf = np.repeat(np.expand_dims(np.eye(4), axis=0), batch_size, axis=0)
                trf[:, :3, 3] = data.site_xpos[:, mg.leaf_site_id].numpy()
                trf[:, :3, :3] = data.site_xmat[:, mg.leaf_site_id].numpy()
                if rel_to_base:
                    trf = base_poses @ trf
                dol[mg_id] = trf

        ret = []
        for i in range(batch_size):
            d = {}
            for mg_id, trf in dol.items():
                d[mg_id] = trf[i]
            ret.append(d)
        return ret if is_batch else ret[0]

    def _ik_solve_step(
        self,
        solver_data: SolverData,
        batch_size: int,
        leaf_site_id: int,
        poses: wp.array(dtype=wp.mat44f),
        damping: float,
        dt: float,
    ):
        data = solver_data.data
        ik_buffers = solver_data.ik_buffers
        mjw.fwd_position(self._mjw_model, data)

        # calculate error
        pos_err, rot_err = ik_buffers.pos_err, ik_buffers.rot_err
        wp.launch(
            get_err,
            dim=batch_size,
            inputs=[data.site_xpos, data.site_xmat, leaf_site_id, poses, pos_err, rot_err],
            device=self._device,
        )

        # calculate Jacobian
        jacp, jacr = ik_buffers.jacp, ik_buffers.jacr
        site_pos = data.site_xpos[:, leaf_site_id]
        site_bodyid = wp.full(batch_size, self._mj_model.site_bodyid[leaf_site_id], dtype=int)
        mjw.jac(self._mjw_model, data, jacp, jacr, site_pos, site_bodyid)

        # solve for joint velocities
        q_dot = ik_buffers.q_dot
        wp.launch(
            lm_step,
            dim=batch_size,
            inputs=[jacp, jacr, pos_err, rot_err, damping, self._mjw_model.nv],
            outputs=[q_dot],
            device=self._device,
        )
        dq = q_dot * dt
        wp.copy(ik_buffers.dq, dq)

        # update joint positions
        wp.copy(data.qpos, data.qpos + dq)

    def ik(
        self,
        poses: np.ndarray,
        q0_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
        move_group_id: str | None = None,
        unlocked_move_group_ids: list[str] | None = None,
        converge_eps: float = 1e-3,
        success_eps: float = 5e-4,
        max_iter: int = 1000,
        damping: float = 1e-12,
        dt: float = 1.0,
    ):
        if move_group_id is None:
            simple_move_groups = [name for name, mg in self._actuated_move_groups.items() if isinstance(mg, SimpleMoveGroup)]
            if len(simple_move_groups) == 0:
                raise ValueError("Robot does not contain any simple move groups!")
            move_group_id = simple_move_groups[0]
            if len(simple_move_groups) > 1:
                logger.warning(f"Multiple simple move groups found, using the first one as target move group: {move_group_id}")
        elif not isinstance(self._actuated_move_groups[move_group_id], SimpleMoveGroup):
            raise ValueError(f"Move group {move_group_id} is not a SimpleMoveGroup")

        if unlocked_move_group_ids is None:
            unlocked_move_group_ids = list(self._actuated_move_groups.keys())
        else:
            for mg_id in unlocked_move_group_ids:
                if mg_id not in self._actuated_move_groups:
                    raise ValueError(f"Move group {mg_id} is not a simply actuated move group!")

        is_batch, batch_size, q0_dicts, base_poses, poses = self._batchify(q0_dicts, base_poses, poses)
        solver_data = self._get_data(batch_size)
        data = solver_data.data

        if not rel_to_base:
            poses = np.linalg.solve(base_poses, poses)
        poses = wp.from_numpy(poses.astype(np.float32), dtype=wp.mat44f)

        q0_arr = self._dicts_to_qpos_arr(q0_dicts)
        wp.copy(data.qpos, wp.from_numpy(q0_arr))

        ee_mg = self._actuated_move_groups[move_group_id]
        assert isinstance(ee_mg, SimpleMoveGroup)

        for i in range(max_iter):
            if solver_data.ik_capture is not None:
                wp.capture_launch(solver_data.ik_capture.graph)
            elif self._device == "cuda":
                with wp.ScopedCapture(device=self._device) as capture:
                    self._ik_solve_step(solver_data, batch_size, ee_mg.leaf_site_id, poses, damping, dt)
                solver_data.ik_capture = capture
            else:
                self._ik_solve_step(solver_data, batch_size, ee_mg.leaf_site_id, poses, damping, dt)
            wp.synchronize()

            if np.all(np.linalg.norm(solver_data.ik_buffers.dq.numpy(), axis=-1) < converge_eps):
                logger.debug(f"[SimpleWarpKinematics] Batch of size {batch_size} converged in {i} iterations")
                break
        else:
            logger.debug(f"[SimpleWarpKinematics] Batch of size {batch_size} failed to converge in {max_iter} iterations")

        mjw.kinematics(self._mjw_model, data)
        pos_err = wp.zeros(batch_size, dtype=wp.vec3f)
        rot_err = wp.zeros(batch_size, dtype=wp.vec3f)
        wp.launch(
            get_err,
            dim=batch_size,
            inputs=[data.site_xpos, data.site_xmat, ee_mg.leaf_site_id, poses, pos_err, rot_err],
            device=self._device,
        )

        err = np.sqrt(np.linalg.norm(pos_err.numpy(), axis=-1)**2 + np.linalg.norm(rot_err.numpy(), axis=-1)**2)
        success = np.array(err < success_eps)

        ret_q_arr = data.qpos.numpy()
        ret_q_dicts = self._qpos_arr_to_dicts(ret_q_arr)
        for q0_dict, ret_q_dict in zip(q0_dicts, ret_q_dicts):
            for k, v in q0_dict.items():
                if k not in ret_q_dict:
                    ret_q_dict[k] = v

        ret = [d if s else None for d, s in zip(ret_q_dicts, success)]
        return ret if is_batch else ret[0]


if __name__ == "__main__":
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics

    robot_config = FrankaRobotConfig()
    wp_kinematics = SimpleWarpKinematics(robot_config)
    ml_kinematics = MlSpacesKinematics.create(robot_config)

    logging.basicConfig(level=logging.DEBUG)

    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    def test_fk():
        init_qpos = []
        for _ in range(10):
            d = {}
            for mg_id, q in robot_config.init_qpos.items():
                d[mg_id] = q + np.random.randn(len(q)) * 0.1
            init_qpos.append(d)

        wp_ret = wp_kinematics.fk(init_qpos, np.eye(4))
        assert isinstance(wp_ret, list)

        ml_ret = []
        for d in init_qpos:
            ml_ret.append(ml_kinematics.fk(d, np.eye(4)))

        for wp_val, ml_val in zip(wp_ret, ml_ret):
            for mg_id in wp_val.keys():
                assert np.allclose(wp_val[mg_id], ml_val[mg_id], atol=1e-4)

        print("FK test passed")

    def test_ik():
        pose = ml_kinematics.fk(robot_config.init_qpos, np.eye(4))["gripper"]
        # pose[0, 3] += 0.05
        # pose[1, 3] += 0.05
        # pose[2, 3] += 0.05

        init_qpos = []
        for _ in range(10):
            d = robot_config.init_qpos.copy()
            d["arm"] = d["arm"] + np.random.randn(len(d["arm"])) * 0.1
            init_qpos.append({**d, "base": []})

        ml_ret = []
        for q in init_qpos:
            ml_ret.append(ml_kinematics.ik(
                "arm",
                pose,
                ["arm"],
                q,
                np.eye(4)
            ))

        wp_kinematics.warmup_ik(len(init_qpos))
        wp_ret = wp_kinematics.ik(
            pose,
            init_qpos,
            np.eye(4),
            move_group_id="arm",
        )

        for wp_val, ml_val in zip(wp_ret, ml_ret):
            for mg_id in wp_val.keys():
                if not np.allclose(wp_val[mg_id], ml_val[mg_id], atol=1e-2):
                    breakpoint()

        print("IK test passed")
    
    test_fk()
    test_ik()
