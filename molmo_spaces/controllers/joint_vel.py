import numpy as np

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.robots.robot_views.abstract import MoveGroup


class JointVelController(Controller):
    """
    Generic joint velocity controller for a robot.
    Computes joint position targets for the joint's actuators after euler integration, and clips to control limits.
    NOTE: Assumes joint actuators use position inputs.
    """

    def __init__(self, robot_config, robot_move_group: MoveGroup) -> None:
        """
        Args:
        robot_config: The configuration of the robot, containing special information about the control,
                      e.g., delta_t for integration (if applicable) etc.
        robot_move_group: The move group of the robot to be controlled
        """
        super().__init__(robot_move_group)

        self.robot_config = robot_config
        self.euler_dt = self.robot_config.delta_t  # Time step for euler integration

        self.ctrl_dim = robot_move_group.n_actuators
        self.ctrl_range = robot_move_group.ctrl_limits

        self._stationary = True
        self._target = np.zeros(self.ctrl_dim)  # Initially zero vel targets

        self._validate_actuators()

        self.reset()

    @property
    def stationary(self) -> bool:
        """Returns whether the controller is in stationary mode"""
        return self._stationary

    @property
    def target(self) -> np.ndarray:
        """Returns the current target joint velocities"""
        return self._target

    def set_target(self, target: np.ndarray) -> None:
        self._stationary = False
        self._target = target.copy()

    def set_to_stationary(self) -> None:
        """
        This method sets the robot to stationary mode and computes the targets to hold the
        robot stationary.
        This is useful when the robot needs to be stopped at a certain position and not drift.
        """
        # Set to stationary mode and set target to zero joint velocities to hold joints stationary
        self._stationary = True
        self._target = np.zeros(self.ctrl_dim)

    def compute_ctrl_inputs(self) -> np.ndarray:
        """
        Compute the control inputs based on the current state and the target set by the user.

        Returns:
            The control inputs to be applied to the robot actuators, in this case: positions
        """
        # position control inputs: perform euler integration on the target velocities
        ctrl_inputs = self.robot_move_group.joint_pos + self._target * self.euler_dt
        # Clip the control inputs to the actuator limits
        ctrl_inputs = np.clip(ctrl_inputs, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

        return ctrl_inputs

    def reset(self) -> None:
        self.set_to_stationary()

    def _validate_actuators(self) -> None:
        # TODO(wilbert): implement this part once the joint_ids and actuator_ids are exposed
        # in the base MoveGroup, not in the SingleActuated one
        pass
