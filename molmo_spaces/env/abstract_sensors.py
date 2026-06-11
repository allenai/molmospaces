# rmh: pulled from allenact and dependencies stripped out
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import (
    Any,
)

import gymnasium as gym
import gymnasium.spaces as gyms

SpaceDict = gyms.Dict


class Sensor(ABC):
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    # Attributes

    uuid : universally unique id.
    observation_space : ``gym.Space`` object corresponding to observation of
        sensor.
    is_dict : whether the observation is a dictionary
    str_max_len : maximum length of the string representation of the encoded dictionary, if is_dict is True
    """

    uuid: str
    observation_space: gym.Space
    is_dict: bool = False
    str_max_len: int = 2000

    def __init__(self, uuid: str, observation_space: gym.Space, **kwargs: Any) -> None:
        self.uuid = uuid
        self.observation_space = observation_space

    @abstractmethod
    def get_observation(self, env, task, *args: Any, **kwargs: Any) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the sensor to its initial state."""
        return None


class SensorSuite:
    """Represents a set of sensors, with each sensor being identified through a
    unique id.

    # Attributes

    sensors: list containing sensors for the environment, uuid of each
        sensor must be unique.
    """

    sensors: dict[str, Sensor]
    observation_spaces: gyms.Dict

    def __init__(self, sensors: Sequence[Sensor]) -> None:
        """Initializer.

        # Parameters

        param sensors: the sensors that will be included in the suite.
        """
        self.sensors = OrderedDict()
        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for sensor in sensors:
            assert sensor.uuid not in self.sensors, f"'{sensor.uuid}' is duplicated sensor uuid"
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        """Return sensor with the given `uuid`.

        # Parameters

        uuid : The unique id of the sensor

        # Returns

        The sensor with unique id `uuid`.
        """
        return self.sensors[uuid]

    def get_observations(self, env, task, **kwargs: Any) -> dict[str, Any]:
        """Get all observations corresponding to the sensors in the suite.

        # Parameters

        env : The environment from which to get the observation.
        task : (Optionally) the task from which to get the observation.

        # Returns

        Data from all sensors packaged inside a Dict.
        """
        return {
            uuid: sensor.get_observation(env=env, task=task, **kwargs)  # type: ignore
            for uuid, sensor in self.sensors.items()
        }

    def extend(self, sensors: Sequence[Sensor]) -> None:
        """Extend the sensor suite with additional sensors."""
        for sensor in sensors:
            self.add(sensor)

    def add(self, sensor: Sensor) -> None:
        """Add a sensor to the sensor suite."""
        assert sensor.uuid not in self.sensors, f"'{sensor.uuid}' is duplicated sensor uuid"
        self.sensors[sensor.uuid] = sensor
        self.observation_spaces[sensor.uuid] = sensor.observation_space
