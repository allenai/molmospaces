class RetriableError(Exception):
    """Error after which we can continue sampling, e.g. by picking a new target in the current house"""

    pass


class ObjectPlacementError(RetriableError):
    """Error when placing some object in the current scene given the current constraints"""

    pass


class RobotPlacementError(RetriableError):
    """Error when placing robot in the current scene given the current constraints"""

    pass


class CameraPlacementError(RetriableError):
    """Error when placing a camera with visibility constraints in the current scene"""

    pass


class HouseInvalidForTask(Exception):
    """
    Exception raised when a house cannot be used for the task being sampled.

    For example:
    - The house has no candidate objects of the requested types (no pickup
      candidates, no doors, no nav targets)
    - No valid task could be sampled within the attempt budget
    - The house has unstable physics or cannot be compiled

    `reason` is the one-line explanation and comes first, so the common
    `HouseInvalidForTask("No doors found in the scene")` reports it. `house_info`
    holds extra details (e.g. the compiler's error text).
    """

    def __init__(self, reason=None, house_info=None, error=None) -> None:
        self.house_info = house_info
        self.reason = reason
        self.error = error

        if reason is None and error is not None:
            reason = f"{type(error).__name__}: {error}"
            self.reason = reason
        message = f"House invalid for tasks: {reason or 'no reason given'}"
        if house_info:
            message += f". Details: {house_info}"

        super().__init__(message)
