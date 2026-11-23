"""Fit sysid params for a molmospaces robot to a real recording and write the JSON that
molmo_spaces applies at load time.

    python projects/sysid/scripts/calibrate.py                          # franka_droid, default data
    python projects/sysid/scripts/calibrate.py --starts current         # quick: warm start only
    python projects/sysid/scripts/calibrate.py --out /tmp/x.json        # don't touch the committed file

Always starts the "before" measurement from the nominal XML model, so before/after in the
JSON are comparable across MuJoCo versions.
"""

import _bootstrap  # noqa: F401  # isort: skip

import argparse
import logging
from pathlib import Path

import numpy as np

from sysid.common import ROBOTS, make_replay, params_from_json, write_result
from sysid.data import load_recording
from sysid.fit import fit
from sysid.replay import rms_deg, tracking_error

log = logging.getLogger("sysid")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--robot", default="franka_droid", choices=sorted(ROBOTS))
    p.add_argument("--data", type=Path, help="real_joints.csv or parquet dir (default: per robot)")
    p.add_argument(
        "--out", type=Path, help="output JSON (default: the robot's committed params file)"
    )
    p.add_argument(
        "--starts",
        default="xml,real,current",
        help="comma-separated optimizer starts, best result wins: xml = nominal XML values, "
        "real = the real controller's kp/kv, current = the committed params JSON (if present). "
        "The fit has local minima (high kp + high friction on the wrist), hence multi-start.",
    )
    p.add_argument(
        "--obs-offset", type=int, default=1, help="real q[i+offset] follows command q_des[i]"
    )
    p.add_argument(
        "--max-nfev", type=int, default=60, help="optimizer iterations (each ~29 rollouts)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s"
    )
    logging.getLogger("molmo_spaces").setLevel(logging.WARNING)

    setup = ROBOTS[args.robot]
    data_path = args.data or setup.default_data
    out = args.out or setup.params_file
    rec = load_recording(data_path)
    log.info("loaded %d samples @ %.0f Hz from %s", len(rec.q), 1 / rec.control_dt, data_path)

    replay = make_replay(args.robot, rec.control_dt, use_sysid=False)
    nominal = replay.get_params()
    nominal_rms = rms_deg(tracking_error(replay, rec.q_des, rec.q, args.obs_offset))

    starts: dict[str, dict] = {}
    for name in args.starts.split(","):
        if name == "xml":
            starts[name] = nominal
        elif name == "real" and setup.real_kp is not None:
            starts[name] = {**nominal, "kp": np.array(setup.real_kp), "kv": np.array(setup.real_kv)}
        elif name == "current" and setup.params_file.exists():
            starts[name] = {**nominal, **params_from_json(setup.params_file)}
    results = {}
    for name, init in starts.items():
        log.info("--- start: %s ---", name)
        results[name] = fit(
            replay, rec.q_des, rec.q, init, obs_offset=args.obs_offset, max_nfev=args.max_nfev
        )
    best = min(results, key=lambda k: results[k].rms_deg_after.mean())
    result = results[best]
    replay.set_params(result.params)
    log.info(
        "best start: %s  (%s)",
        best,
        {k: round(float(r.rms_deg_after.mean()), 4) for k, r in results.items()},
    )

    log.info("joint        |      kp |     kv | friction | armature | RMS nominal -> fitted [deg]")
    for j, name in enumerate(setup.joint_names):
        log.info(
            "%-12s | %7.1f | %6.2f | %8.3f | %8.4f | %.3f -> %.3f",
            name,
            result.params["kp"][j],
            result.params["kv"][j],
            result.params["frictionloss"][j],
            result.params["armature"][j],
            nominal_rms[j],
            result.rms_deg_after[j],
        )
    log.info("mean RMS %.3f -> %.3f deg", nominal_rms.mean(), result.rms_deg_after.mean())

    write_result(
        out,
        args.robot,
        replay,
        {
            "params": result.params,
            "nominal_params": {k: [round(float(x), 6) for x in v] for k, v in nominal.items()},
            "fit": {
                "data": str(Path(data_path).name),
                "data_notes": setup.notes,
                "n_samples": len(rec.q),
                "control_dt": rec.control_dt,
                "obs_offset": args.obs_offset,
                "start": best,
                "start_results": {
                    k: round(float(r.rms_deg_after.mean()), 4) for k, r in results.items()
                },
                "n_rollouts": sum(r.n_rollouts for r in results.values()),
                "optimizer_status": result.optimizer_status,
                "rms_deg_nominal": np.round(nominal_rms, 4).tolist(),
                "rms_deg_fitted": np.round(result.rms_deg_after, 4).tolist(),
                "rms_deg_nominal_mean": round(float(nominal_rms.mean()), 4),
                "rms_deg_fitted_mean": round(float(result.rms_deg_after.mean()), 4),
            },
        },
    )
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
