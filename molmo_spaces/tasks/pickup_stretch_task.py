from collections import deque

import numpy as np

from molmo_spaces.tasks.atomic.grasp_sampler import TopDownGraspPoseSampler
from molmo_spaces.tasks.atomic.objectnav_task import ObjectNavTask
from molmo_spaces.tasks.atomic.pickup_task import PickupTask

TOPDOWN_JOINT_POS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.04, 0.0, 0.0])


class StretchPickupTask(PickupTask):
    FIXED_INIT_JOINT_POS = TOPDOWN_JOINT_POS

    def __init__(self, n_obs_steps, *args, **kwargs) -> None:
        super().__init__(n_obs_steps=n_obs_steps, *args, **kwargs)
        self._grasp_and_lift_up_ctrl_inputs = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0],  # grasp
            [0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0],  # lift up while grasping
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0],  # lift up while grasping
        ]
        self._last_actions = None
        self._last_observations = None
        self._last_rewards = None
        self._last_dones = None
        # Initialize observation history as a deque with fixed maxlen
        self.observation_history = deque(maxlen=self.n_obs_steps)

    @property
    def last_observations(self):
        return self._last_observations

    @last_observations.setter
    def last_observations(self, value) -> None:
        self._last_observations = value
        # Update the observation history with the new observation.
        # This fixed-size queue will always keep the last n_obs_steps observations.
        self._update_observation_history(self._last_observations)

    def get_observations(self):
        return list(self.observation_history)  # self._last_observations

    def _update_observation_history(self, new_obs) -> None:
        """
        Update the internal observation history as a fixed-size queue.
        If the history is empty, populate it with new_obs repeated.
        Otherwise, append new_obs so that the oldest observation is automatically dropped.

        Returns:
            The updated list of observations (length == n_obs_steps).
        """
        if len(self.observation_history) == 0:
            # Initialize the deque with new_obs repeated
            for _ in range(self.n_obs_steps):
                self.observation_history.append(new_obs)
        else:
            self.observation_history.append(new_obs)

    def step(self, action, obs_sensor=None):
        # Step the environment for this task.
        self.env.step_single(self.batch_index, action, debug_teleport=False)

        # Get observation after stepping.
        if obs_sensor is not None:
            obs = {}
            for sensor in obs_sensor:
                sensor(self.env.mj_datas[self.batch_index])
                obs[sensor.name] = sensor.last_data
            self.last_observations = obs

        # Compute reward and done status using task-specific functions.
        reward = self.get_reward() if hasattr(self, "get_reward") else 0.0
        done = self.judge_success() if hasattr(self, "judge_success") else False
        self._last_rewards = reward
        self._last_dones = done
        return self._last_observations, reward, done

    def get_last_mile_ctrl_inputs(self, data, **kwargs):
        """Heuristic Task specific actions for completing the task"""
        curr_joint_pos = data.ctrl.copy()
        return [curr_joint_pos + ctrl_input for ctrl_input in self._grasp_and_lift_up_ctrl_inputs]


class StretchObjectNavTask(ObjectNavTask):
    FIXED_JOINT_POS = TOPDOWN_JOINT_POS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_actions = None
        self._last_observations = None
        self._last_rewards = None
        self._last_dones = None
        # Initialize observation history as a deque with fixed maxlen
        self.observation_history = deque(maxlen=self.n_obs_steps)

    @property
    def last_observations(self):
        return self._last_observations

    @last_observations.setter
    def last_observations(self, value) -> None:
        self._last_observations = value
        # Update the observation history with the new observation.
        # This fixed-size queue will always keep the last n_obs_steps observations.
        self._update_observation_history(self._last_observations)

    def get_observations(self):
        return list(self.observation_history)  # self._last_observations

    def _update_observation_history(self, new_obs) -> None:
        """
        Update the internal observation history as a fixed-size queue.
        If the history is empty, populate it with new_obs repeated.
        Otherwise, append new_obs so that the oldest observation is automatically dropped.

        Returns:
            The updated list of observations (length == n_obs_steps).
        """
        if len(self.observation_history) == 0:
            # Initialize the deque with new_obs repeated
            for _ in range(self.n_obs_steps):
                self.observation_history.append(new_obs)
        else:
            self.observation_history.append(new_obs)

    def get_last_mile_ctrl_inputs(self, data, **kwargs) -> None:
        return None

    def step(self, action, obs_sensor=None):
        # Step the environment for this task.
        self.env.step_single(self.batch_index, action, debug_teleport=False)

        # Get observation after stepping.
        if obs_sensor is not None:
            obs = {}
            for sensor in obs_sensor:
                sensor(self.env.mj_datas[self.batch_index])
                obs[sensor.name] = sensor.last_data
            self.last_observations = obs

        # Compute reward and done status using task-specific functions.
        reward = self.get_reward() if hasattr(self, "get_reward") else 0.0
        done = self.judge_success() if hasattr(self, "judge_success") else False
        self._last_rewards = reward
        self._last_dones = done
        return self._last_observations, reward, done


from molmo_spaces.robots.robot_views.stretch_dex_view import StretchDexRobotView


class StretchTopDownGraspPoseSampler(TopDownGraspPoseSampler):
    def __init__(self) -> None:
        super().__init__(StretchDexRobotView.STRETCH_TOPDOWN_WRIST_ROTATION)

    def get_delta_pregrasp_to_grasp_joint_pos(self):
        return np.array([-0.1, 0.0, 0.0, 0.0, 0.0])
