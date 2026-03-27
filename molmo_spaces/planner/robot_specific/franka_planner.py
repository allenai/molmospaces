import mujoco
import numpy as np
from molmo_spaces.planner.ompl_planner import OMPLPlanner
from molmo_spaces.robots.robot_views.franka_fr3 import FrankaFR3RobotView
from mujoco import MjData, MjModel
from ompl import base as ob
from scipy.spatial.transform import Rotation as R

from molmo_spaces.kinematics import FrankaKinematics


class FrankaPlanner(OMPLPlanner):
    agent: FrankaFR3RobotView

    def __init__(self, model: MjModel, namespace: str = "") -> None:
        robot_root_id = FrankaFR3Agent.robot_root_id(model, namespace)
        model = OMPLPlanner.optimize_model_for_planning(model, robot_root_id)
        data = MjData(model)
        agent = FrankaFR3RobotView(model, data, ns=namespace)
        kinematics = FrankaKinematics(model, data, namespace=namespace)
        super().__init__(model, data, agent, kinematics)
        self.state_space_low, self.state_space_high = agent.joint_limits
        assert self.state_space_low.shape[0] == self.STATE_SPACE_DIM

        state_space = ob.RealVectorStateSpace(self.STATE_SPACE_DIM)
        bounds = ob.RealVectorBounds(self.STATE_SPACE_DIM)
        for i in range(self.STATE_SPACE_DIM):
            bounds.setLow(i, self.state_space_low[i])
            bounds.setHigh(i, self.state_space_high[i])
        state_space.setBounds(bounds)

        si = ob.SpaceInformation(state_space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        si.setStateValidityCheckingResolution(0.01)
        si.setup()

        self.space = state_space
        self.space_info = si

    @property
    def STATE_SPACE_DIM(self):
        return len(self.agent.joint_names)

    def _set_start_state(self) -> None:
        state = ob.State(self.space)
        jp = self.agent.joint_pos
        for i in range(self.STATE_SPACE_DIM):
            state[i] = jp[i]
        self.start_state = state


def ik_test() -> None:
    model = MjModel.from_xml_path("assets/franka_fr3/empty_scene.xml")
    data = MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    agent = FrankaFR3RobotView(model, data)
    planner = FrankaPlanner(model)

    start_pose = agent.ee_pose_from_base.copy()
    start_pos = start_pose[:3, 3]
    start_rot = R.from_matrix(start_pose[:3, :3])
    import time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            elapsed = time.time() - start_time
            target_pos = start_pos + np.array([0.1 * np.sin(elapsed), 0.1 * np.cos(elapsed), 0])
            target_rot = start_rot
            q = planner.kinematics.ik(
                target_pos, target_rot.as_quat(scalar_first=True), agent.joint_pos
            )
            if q is not None:
                data.qpos[: len(q)] = q
            else:
                print("No solution found")
            mujoco.mj_forward(model, data)
            viewer.sync()


def plan_test() -> None:
    ns = "robot_0/"
    model = MjModel.from_xml_path("houses/RoboTHOR_fr3.xml")
    data = MjData(model)
    agent = FrankaFR3RobotView(model, data, ns)
    # TODO: add keyframes to procthor generation
    data.qpos[agent.qpos_adrs] = [0, -0.7853, 0, -2.35619, 0, 1.57079, 0.7853]
    # mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    mujoco.mj_forward(model, data)
    agent(model, data)
    planner = FrankaPlanner(model, ns)
    planner.create_planner("ait_star")

    goal_pos = agent.ee_pose_from_base[:3, 3].copy()
    goal_pos[0] *= -1
    # goal_pos[1] -= 0.25
    # goal_pos[2] = 0
    goal_quat = R.from_matrix(agent.ee_pose_from_base[:3, :3]).as_quat(scalar_first=True)

    plan = planner.motion_plan(data, goal_pos, goal_quat, goal_frame="base", planning_time=20.0)
    assert plan is not None, "Planning failed!"
    plan = np.array(plan)
    print(f"Plan:\n{plan}")

    TRAJ_DUR = 6.0
    seg_lens = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    seg_durs = TRAJ_DUR * seg_lens / np.sum(seg_lens)
    t = np.cumsum(np.concatenate(([0], seg_durs)))

    import time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        reverse = False
        while viewer.is_running():
            viewer.sync()
            elapsed = time.time() - start_time
            if elapsed >= t[-1]:
                start_time = time.time()
                reverse = not reverse
                continue
            setpoint = []
            for i in range(agent.ndof):
                if reverse:
                    setpoint.append(np.interp(t[-1] - elapsed, t, plan[:, i]))
                else:
                    setpoint.append(np.interp(elapsed, t, plan[:, i]))
            data.ctrl[: agent.ndof] = setpoint
            end_render = data.time + 0.02
            while data.time < end_render:
                mujoco.mj_step(model, data)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    ik_test()
    plan_test()
