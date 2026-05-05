import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.franka_parallel_kinematics import FrankaParallelKinematics
from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics
from molmo_spaces.kinematics.parallel.warp_kinematics import SimpleWarpKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics


def benchmark_parallel_kinematics(min_time: float, kinematics: ParallelKinematics, ik_kwargs: dict):
    start_time = time.perf_counter()
    n_iter = 0
    while time.perf_counter() - start_time < min_time:
        kinematics.ik(**ik_kwargs)
        n_iter += 1
    end_time = time.perf_counter()
    return (end_time - start_time) / n_iter

def main():
    robot_config = FrankaRobotConfig()

    cpu_kinematics = MlSpacesKinematics.create(robot_config)

    kinematics_solvers: dict[str, ParallelKinematics] = {
        # "dummy": DummyParallelKinematics(robot_config, cpu_kinematics, "arm", ["arm"]),
        # "jax": FrankaParallelKinematics(robot_config),
        "warp": SimpleWarpKinematics(robot_config),
    }

    batch_sizes = [1, 4, 16, 64]
    max_batch_size = max(batch_sizes)

    q0s = [robot_config.init_qpos.copy()] * max_batch_size
    poses = np.repeat(cpu_kinematics.fk(robot_config.init_qpos, np.eye(4))["arm"][None], max_batch_size, axis=0)
    for i, q0_dict in enumerate(q0s):
        q0_dict["arm"] = q0_dict["arm"] + np.random.randn(len(q0_dict["arm"])) * 0.5
        q0_dict["base"] = []
        poses[i, :3, :3] = poses[i, :3, :3] @ R.from_rotvec(np.random.randn(3) * np.radians(30)).as_matrix()
        poses[i, :3, 3] = poses[i, :3, 3] + np.random.randn(3) * 0.07


    for kinematics_name, kinematics in kinematics_solvers.items():
        for batch_size in batch_sizes:
            for _ in range(10):
                kinematics.warmup_ik(batch_size)
            ik_kwargs = {
                "poses": poses[:batch_size],
                "q0_dicts": q0s[:batch_size],
                "base_poses": np.eye(4),
            }
            time = benchmark_parallel_kinematics(5.0, kinematics, ik_kwargs)
            print(f"{kinematics_name},{batch_size},{time}")

if __name__ == "__main__":
    main()
