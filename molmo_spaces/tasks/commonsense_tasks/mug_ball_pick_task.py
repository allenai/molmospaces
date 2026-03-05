import logging
from typing import Any

from molmo_spaces.env.data_views import MjThorObject
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class MugBallPickTask(PickTask):
    """Pick the mug that has a ball hidden under it.

    Two mugs are placed upside-down on a counter surface. One covers a ball,
    the other covers nothing. The robot must pick up the correct mug (the one
    with the ball under it). The correct mug name is tracked as a member
    variable and is set by the task sampler.
    """

    def __init__(self, env, config):
        super().__init__(env, config)
        self.correct_mug_name: str | None = None
        self._settled = False

    def reset(self):
        self._settled = False
        return super().reset()

    def step(self, action):
        settle_duration = self.config.task_config.scene_settle_duration
        if settle_duration > 0 and not self._settled:
            sim_time = self._env.mj_datas[0].time
            if sim_time < settle_duration:
                # Replace action with no-op during settling
                robot_view = self._env.current_robot.robot_view
                action = robot_view.get_noop_ctrl_dict()
            else:
                # Settling just finished — update poses to reflect settled object position
                self._settled = True
                pickup_obj_name = self.config.task_config.pickup_obj_name
                pickup_obj = MjThorObject(object_name=pickup_obj_name, data=self._env.current_data)
                old_start_z = self.config.task_config.pickup_obj_start_pose[2]
                new_start_pose = pose_mat_to_7d(pickup_obj.pose)
                new_goal_pose = new_start_pose.copy()
                new_goal_pose[2] += 0.05  # 5cm above settled position

                self.config.task_config.pickup_obj_start_pose = new_start_pose.tolist()
                self.config.task_config.pickup_obj_goal_pose = new_goal_pose.tolist()

                log.info(
                    f"[MUG BALL PICK] Scene settled. "
                    f"start_pose_z: {old_start_z:.4f} -> {new_start_pose[2]:.4f}"
                )

        return super().step(action)

    def get_task_description(self) -> str:
        return "Pick up the mug with the ball under it."

    def judge_success(self) -> bool:
        """Success requires lifting the correct mug (the one over the ball)."""
        if self.config.task_type == "mug_ball_pick":
            return self.get_info()[0]["success"]
        else:
            raise ValueError(f"Invalid task_type {self.config.task_type}")

    def get_info(self) -> list[dict[str, Any]]:
        """Get metrics including which mug is the correct one."""
        infos = super().get_info()
        for info in infos:
            info["correct_mug_name"] = self.correct_mug_name
        return infos
