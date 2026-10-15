import numpy as np

from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.robots.robot_views.abstract import MoveGroup


class JointPosController(AbstractPositionController):
    """
    Generic joint position controller for a robot.
    Computes joint position targets for the joint's actuators, and clips to control limits.
    NOTE: Assumes joint actuators use position inputs.
    """

    def __init__(self, robot_move_group: MoveGroup) -> None:
        super().__init__(robot_move_group)

        self.ctrl_dim = robot_move_group.n_actuators
        self.ctrl_range = robot_move_group.ctrl_limits

        self._stationary = True
        self._target = self.robot_move_group.joint_pos

        self._validate_actuators()

        self.reset()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value: np.ndarray) -> None:
        self._target = value

    @property
    def target_pos(self) -> np.ndarray:
        return self._target.copy()

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target: np.ndarray) -> None:
        self._stationary = False
        self.target = np.clip(target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def set_to_stationary(self) -> None:
        """
        This method sets the robot to stationary mode and computes the targets to hold the
        robot stationary.
        This is useful when the robot needs to be stopped at a certain position and not drift.
        """
        # Set to stationary mode and set target to current joint positions to hold them stationary
        self._stationary = True
        self.target = self.robot_move_group.noop_ctrl

    def compute_ctrl_inputs(self):
        """
        Compute the control inputs based on the current state and the target set by the user.

        Returns:
            The control inputs to be applied to the robot actuators, in this case: positions
        """
        # position control inputs: just pass through the target joint positions
        return self.target.copy()

    def reset(self) -> None:
        """Reset the controller to its initial state, clearing any internal state or targets"""
        self.set_to_stationary()  # Explicit reset to stationary mode

    def _validate_actuators(self) -> None:
        # TODO(wilbert): implement this part once the joint_ids and actuator_ids are exposed
        # in the base MoveGroup, not in the SingleActuated one
        pass
