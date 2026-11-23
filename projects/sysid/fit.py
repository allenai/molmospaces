"""Fit per-joint kp, kv, frictionloss, armature so the sim replay matches the real q(t).

All 7 joints are fitted jointly (scipy least_squares, trust-region reflective) on the
stacked per-sample tracking error, so inertial coupling between joints is accounted for.
Positive scale parameters are optimised in log space.

Joint `damping` is deliberately NOT fitted: for a position actuator, kv and joint damping
are both a torque -c * qdot and therefore not separately identifiable from this data. It
stays at the XML value; kv absorbs all viscous damping.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from sysid.replay import PARAM_NAMES, ArmReplay, rms_deg, tracking_error

log = logging.getLogger(__name__)

# (lower, upper, log-space?) per parameter
BOUNDS = {
    "kp": (5.0, 2.0e4, True),
    "kv": (0.1, 2.0e3, True),
    "frictionloss": (0.0, 10.0, False),
    "armature": (1.0e-3, 2.0, True),
}


@dataclass
class FitResult:
    params: dict[str, np.ndarray]
    rms_deg_before: np.ndarray
    rms_deg_after: np.ndarray
    n_rollouts: int
    optimizer_status: str


def _pack(params: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in PARAM_NAMES:
        lo, hi, is_log = BOUNDS[name]
        v = np.clip(params[name], lo, hi)
        parts.append(np.log(v) if is_log else v)
    return np.concatenate(parts)


def _unpack(x: np.ndarray, n: int) -> dict[str, np.ndarray]:
    out = {}
    for k, name in enumerate(PARAM_NAMES):
        v = x[k * n : (k + 1) * n]
        out[name] = np.exp(v) if BOUNDS[name][2] else v.copy()
    return out


def _bounds(n: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for name in PARAM_NAMES:
        a, b, is_log = BOUNDS[name]
        lo += [np.log(a) if is_log else a] * n
        hi += [np.log(b) if is_log else b] * n
    return np.array(lo), np.array(hi)


def fit(
    replay: ArmReplay,
    q_des: np.ndarray,
    q_real: np.ndarray,
    init: dict[str, np.ndarray],
    obs_offset: int = 1,
    max_nfev: int = 60,
) -> FitResult:
    n = len(replay.joint_names)
    n_rollouts = 0

    def residuals(x: np.ndarray) -> np.ndarray:
        nonlocal n_rollouts
        n_rollouts += 1
        replay.set_params(_unpack(x, n))
        err = tracking_error(replay, q_des, q_real, obs_offset)
        # scaled so that 0.5 * ||r||^2 = 0.5 * sum_j (rms_j [deg])^2
        return np.rad2deg(err).ravel() / np.sqrt(len(err))

    replay.set_params(init)
    before = rms_deg(tracking_error(replay, q_des, q_real, obs_offset))
    log.info("initial RMS [deg] %s (mean %.3f)", np.round(before, 3).tolist(), before.mean())

    lo, hi = _bounds(n)
    x0 = np.clip(_pack(init), lo + 1e-9, hi - 1e-9)
    res = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        method="trf",
        x_scale="jac",
        diff_step=1e-3,
        max_nfev=max_nfev,
        verbose=2 if log.isEnabledFor(logging.DEBUG) else 1,
    )
    params = _unpack(res.x, n)
    replay.set_params(params)
    after = rms_deg(tracking_error(replay, q_des, q_real, obs_offset))
    log.info("fitted RMS [deg] %s (mean %.3f)", np.round(after, 3).tolist(), after.mean())

    for name in PARAM_NAMES:
        lo_b, hi_b, _ = BOUNDS[name]
        at_bound = np.isclose(params[name], lo_b, rtol=1e-3) | np.isclose(
            params[name], hi_b, rtol=1e-3
        )
        if at_bound.any():
            log.warning(
                "%s hit its bound on joints %s -- treat as poorly identified",
                name,
                np.flatnonzero(at_bound) + 1,
            )

    return FitResult(
        params=params,
        rms_deg_before=before,
        rms_deg_after=after,
        n_rollouts=n_rollouts,
        optimizer_status=res.message,
    )
