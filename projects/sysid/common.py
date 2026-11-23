"""Per-robot sysid setup and the result JSON format."""

import datetime
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig
from molmo_spaces.robots.sysid import SYSID_RESOURCES_DIR, load_sysid_params
from sysid.replay import ArmReplay, build_model

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]


@dataclass(frozen=True)
class RobotSysidSetup:
    make_config: Callable[..., BaseRobotConfig]
    joint_names: list[str]
    actuator_names: list[str]
    default_data: Path
    params_file: Path  # where the fitted JSON lives (read at load time by molmo_spaces)
    # gains of the real robot's controller during the recording; used as one optimizer start
    real_kp: list[float] | None = None
    real_kv: list[float] | None = None
    notes: str = ""


_FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]

ROBOTS: dict[str, RobotSysidSetup] = {
    "franka_droid": RobotSysidSetup(
        make_config=FrankaRobotConfig,
        joint_names=_FR3_JOINTS,
        actuator_names=_FR3_JOINTS,
        default_data=PROJECT_DIR / "data" / "fr3_2026-09-02" / "real_joints.csv",
        params_file=SYSID_RESOURCES_DIR / "franka_droid.json",
        real_kp=[100.0, 100.0, 100.0, 100.0, 75.0, 150.0, 50.0],
        real_kv=[20.0, 20.0, 20.0, 20.0, 7.5, 15.0, 5.0],
        notes=(
            "Real FR3 + Robotiq 2F85 (open, fixed), 40 s multi-sine around DROID home, 25 Hz "
            "joint-position targets, real controller kp=[100,100,100,100,75,150,50] "
            "kd=[20,20,20,20,7.5,15,5]. Recorded 2026-09-02 (rcs sysid_data3)."
        ),
    ),
}


def make_replay(robot: str, control_dt: float, use_sysid: bool) -> ArmReplay:
    """use_sysid=False builds the nominal XML model; True builds it the way envs do (JSON applied)."""
    setup = ROBOTS[robot]
    config = setup.make_config() if use_sysid else setup.make_config(sysid_params_file=None)
    return ArmReplay(
        model=build_model(config),
        namespace=config.robot_namespace,
        joint_names=setup.joint_names,
        actuator_names=setup.actuator_names,
        control_dt=control_dt,
    )


def params_from_json(path: Path) -> dict[str, np.ndarray]:
    return {
        k: np.asarray(v, dtype=float) for k, v in load_sysid_params(Path(path))["params"].items()
    }


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info(replay: ArmReplay) -> dict:
    m = replay.model
    return {
        "mujoco_version": mujoco.__version__,
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "timestep": float(m.opt.timestep),
        "integrator": mujoco.mjtIntegrator(m.opt.integrator).name,
    }


def write_result(path: Path, robot: str, replay: ArmReplay, data: dict) -> None:
    setup = ROBOTS[robot]
    out = {
        "robot": robot,
        "joints": setup.joint_names,
        "actuators": setup.actuator_names,
        "params": {k: [round(float(x), 6) for x in v] for k, v in data.pop("params").items()},
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(replay),
        **data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
