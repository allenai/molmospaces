"""Compatibility shim -- the implementation moved to molmo_spaces/robots/g1.py.

This file used to hold the fork of g1_molmo's components/robot.py. That
implementation has been copied into molmo_spaces proper (see
`molmo_spaces.robots.g1`), which is where it is now being reshaped
toward `molmo_spaces/robots/abstract.py`'s Robot interface. Nothing is
re-implemented here: every name below is re-exported from that one module, so
all consumers (env_g1ms, controller_g1ms, policy_g1ms, pick_task_sampler_g1ms)
share a single class object and a single set of module-level constants.

Kept as a shim rather than deleted so the port's own import sites stay valid
while the move is verified against the gold rollout step by step.
"""

from molmo_spaces.robots.g1 import (  # noqa: F401
    DEFAULT_QPOS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    JOINT_NAMES,
    PELVIS_FORWARD_OFFSET,
    PREFIX,
    ROOT_BODY,
    STANDING_HEIGHT,
    XML_PATH,
    G1Robot,
)
