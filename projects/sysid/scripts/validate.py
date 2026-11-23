"""Check the committed sysid params still reproduce the real recording (no refit).

Builds the robot exactly as envs do (JSON applied at load), replays the recording and
compares the per-joint RMS with what was stored at fit time. Run after any change to
MuJoCo, the robot XML, or the physics options; if it fails, re-run calibrate.py.

    python projects/sysid/scripts/validate.py [--plot out.png]

Exit code 1 if the mean RMS regressed by more than --tol-deg.
"""

import _bootstrap  # noqa: F401  # isort: skip

import argparse
import json
import logging
import sys
from pathlib import Path

import mujoco
import numpy as np

from sysid.common import ROBOTS, make_replay
from sysid.data import load_recording
from sysid.replay import rms_deg, tracking_error

log = logging.getLogger("sysid")


def plot(path: Path, rec, sim_nominal, sim_fitted, offset: int, names: list[str]) -> None:
    try:
        import matplotlib
    except ImportError:
        log.error("--plot needs matplotlib (pip install matplotlib)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(sim_fitted)
    t = np.arange(T) * rec.control_dt
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2 * len(names)), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, np.rad2deg(rec.q_des[:T, j]), color="0.7", lw=1, label="commanded")
        ax.plot(t, np.rad2deg(rec.q[offset : offset + T, j]), "k", lw=1.2, label="real")
        ax.plot(t, np.rad2deg(sim_nominal[:, j]), "C3", lw=0.8, label="sim nominal")
        ax.plot(t, np.rad2deg(sim_fitted[:, j]), "C0", lw=0.8, label="sim sysid")
        ax.set_ylabel(f"{names[j]} [deg]")
    axes[0].legend(ncol=4, loc="upper right")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    log.info("wrote %s", path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--robot", default="franka_droid", choices=sorted(ROBOTS))
    p.add_argument("--data", type=Path)
    p.add_argument(
        "--tol-deg", type=float, default=0.05, help="allowed increase of the mean RMS [deg]"
    )
    p.add_argument("--plot", type=Path, help="write a q(t) overlay plot")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("molmo_spaces").setLevel(logging.WARNING)

    setup = ROBOTS[args.robot]
    with open(setup.params_file) as f:
        stored = json.load(f)
    fit_info = stored["fit"]
    offset = fit_info["obs_offset"]
    rec = load_recording(args.data or setup.default_data)

    nominal = make_replay(args.robot, rec.control_dt, use_sysid=False)
    fitted = make_replay(args.robot, rec.control_dt, use_sysid=True)
    err_nom = tracking_error(nominal, rec.q_des, rec.q, offset)
    err_fit = tracking_error(fitted, rec.q_des, rec.q, offset)
    rms_nom, rms_fit = rms_deg(err_nom), rms_deg(err_fit)
    stored_rms = np.asarray(fit_info["rms_deg_fitted"])

    log.info(
        "fitted with mujoco %s, now %s", stored["environment"]["mujoco_version"], mujoco.__version__
    )
    log.info("joint        | nominal | sysid now | sysid at fit time  [RMS deg]")
    for j, name in enumerate(setup.joint_names):
        log.info("%-12s | %7.3f | %9.3f | %7.3f", name, rms_nom[j], rms_fit[j], stored_rms[j])
    log.info(
        "mean         | %7.3f | %9.3f | %7.3f", rms_nom.mean(), rms_fit.mean(), stored_rms.mean()
    )

    if args.plot:
        real_aligned = rec.q[offset : offset + len(err_fit)]
        plot(
            args.plot,
            rec,
            real_aligned + err_nom,
            real_aligned + err_fit,
            offset,
            setup.joint_names,
        )

    regression = rms_fit.mean() - stored_rms.mean()
    if regression > args.tol_deg:
        log.error(
            "FAIL: mean RMS regressed by %.3f deg (> %.3f). Re-run calibrate.py.",
            regression,
            args.tol_deg,
        )
        sys.exit(1)
    log.info("OK (mean RMS change %+.3f deg, tolerance %.3f)", regression, args.tol_deg)


if __name__ == "__main__":
    main()
