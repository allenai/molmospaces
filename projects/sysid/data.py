"""Load a real joint-tracking recording: commanded targets q_des and measured q, both (T, 7) rad.

Accepts either
  * the slim CSV  `step,timestamp,q_des1..7,q1..7`  (real_joints.csv), or
  * a directory of rcs/LeRobot-style parquet shards with
    `env_action.right.joints` (q_des) and `obs.right.joints` (q).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    q_des: np.ndarray  # (T, 7) commanded joint targets [rad]
    q: np.ndarray  # (T, 7) measured joint positions [rad]
    control_dt: float  # [s]
    source: str


def _from_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    d = d[np.argsort(d[:, 0])]
    return d[:, 1], d[:, 2:9], d[:, 9:16]


def _from_parquet(
    path: Path, robot_key: str = "right"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.dataset as pds

    rows = pds.dataset(str(path), format="parquet").to_table().to_pylist()
    rows.sort(key=lambda r: r["step"])
    t = np.array([r["timestamp"] for r in rows], dtype=float)
    q_des = np.array([r["env_action"][robot_key]["joints"] for r in rows], dtype=float)
    q = np.array([r["obs"][robot_key]["joints"] for r in rows], dtype=float)
    return t, q_des, q


def load_recording(path: str | Path, control_hz: float | None = None) -> Recording:
    path = Path(path)
    if path.suffix == ".csv":
        t, q_des, q = _from_csv(path)
    else:
        t, q_des, q = _from_parquet(path)
    if control_hz is None:
        # timestamps in the FR3 data are wall-clock with ~1.5 ms jitter; take the median step
        control_dt = float(np.median(np.diff(t)))
        control_dt = round(control_dt, 3)
    else:
        control_dt = 1.0 / control_hz
    return Recording(q_des=q_des, q=q, control_dt=control_dt, source=str(path))
