"""Apply system-identified joint/actuator parameters to a robot MjSpec at load time.

The robot XMLs keep their nominal (upstream) values. A sysid JSON, produced by
`projects/sysid/scripts/calibrate.py`, overrides them per joint when the robot is
added to a scene (see `Robot.apply_control_overrides`).

JSON layout (only the keys below are read; everything else is fit metadata):

    {
      "joints": ["fr3_joint1", ...],        # un-namespaced joint names
      "actuators": ["fr3_joint1", ...],     # un-namespaced actuator names, same order
      "params": {
        "kp": [...], "kv": [...],           # position actuator gains (optional)
        "frictionloss": [...], "armature": [...], "damping": [...]  # joint params (optional)
      }
    }
"""

import json
import logging
from functools import cache
from pathlib import Path

import mujoco

log = logging.getLogger(__name__)

SYSID_RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources" / "sysid"

JOINT_PARAMS = ("frictionloss", "armature", "damping")
ACTUATOR_PARAMS = ("kp", "kv")


@cache
def load_sysid_params(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    n = len(data["joints"])
    for key, values in data["params"].items():
        if key not in JOINT_PARAMS + ACTUATOR_PARAMS:
            raise ValueError(f"{path}: unknown sysid param {key!r}")
        if len(values) != n:
            raise ValueError(f"{path}: param {key!r} has {len(values)} values, expected {n}")
    return data


def _set_joint_scalar(joint: mujoco.MjsJoint, key: str, value: float) -> None:
    """Set a hinge-joint scalar. MuJoCo 4 made `damping` a coefficient array (linear term
    first); older versions have a plain float. Only the linear term is ever set here."""
    current = getattr(joint, key)
    if hasattr(current, "__len__"):
        current[0] = value
        setattr(joint, key, current)
    else:
        setattr(joint, key, value)


def set_position_gains(actuator: mujoco.MjsActuator, kp: float | None, kv: float | None) -> None:
    """Set kp/kv of a MuJoCo <position> actuator (gain = kp, bias = [0, -kp, -kv])."""
    if kp is not None:
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
    if kv is not None:
        actuator.biasprm[2] = -kv


def apply_sysid_params(spec: mujoco.MjSpec, namespace: str, sysid_file: Path) -> None:
    data = load_sysid_params(Path(sysid_file))
    params = data["params"]
    for i, joint_name in enumerate(data["joints"]):
        joint = spec.joint(f"{namespace}{joint_name}")
        if joint is None:
            raise ValueError(f"sysid joint {namespace}{joint_name} not found in spec")
        for key in JOINT_PARAMS:
            if key in params:
                _set_joint_scalar(joint, key, params[key][i])
    if any(k in params for k in ACTUATOR_PARAMS):
        for i, act_name in enumerate(data["actuators"]):
            actuator = spec.actuator(f"{namespace}{act_name}")
            if actuator is None:
                raise ValueError(f"sysid actuator {namespace}{act_name} not found in spec")
            set_position_gains(
                actuator,
                params["kp"][i] if "kp" in params else None,
                params["kv"][i] if "kv" in params else None,
            )
    log.debug("Applied sysid params from %s to %s", sysid_file, namespace)
