"""Navigation smoke test for molmo_spaces/robots/g1.py's G1Robot.

The pick rollout (generate_ported_rollout.py) exercises this robot's grasp
path. This exercises the *navigation* path instead: it drives the robot to a
sequence of absolute [x, y, yaw] waypoints through
`G1Robot.waypoint_to_velocity_target` -- the same waypoint -> base-velocity
bridge molmo_spaces/robots/g1_old_reference.py uses, so a molmo_spaces
navigation policy (e.g. AStarPlannerPolicy, which emits exactly these
waypoints) can drive this robot unchanged.

Waypoints are picked relative to the robot's own spawn pose and checked
against the scene occupancy map, so this works on whatever house the config
loads without hardcoding scene coordinates.

Run from the repo root:
    conda run -n mlspaces python scripts/g1_molmo_port_comparison/nav_demo.py
"""

import argparse

import numpy as np

from molmo_spaces.g1_molmo_port.configs.bowl_mixed_grasponly import get_config
from molmo_spaces.g1_molmo_port.env_g1ms import make_env
from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import G1Controller

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-steps", type=int, default=1200, help="Sim steps per waypoint.")
parser.add_argument(
    "--tolerance", type=float, default=0.25, help="Metres from waypoint that counts as arrived."
)
args = parser.parse_args()

cfg = get_config().copy_and_resolve_references()
cfg["scene"] = "scenes/procthor-10k-val/val_0.xml"
cfg["randomize_scene"] = False
cfg["randomize_object"] = False
cfg["launch_viewer"] = False
cfg["seed"] = args.seed
cfg = cfg.to_dict()

env = make_env(cfg)
raw_env = getattr(env, "env", env)
agent = G1Controller()
agent.setup(raw_env.scene.model, raw_env.scene.data)
agent.set_env(raw_env)
env.set_agent(agent)

obs, info = env.reset()
agent.reset(info)
robot = raw_env.robot

start_xy = robot.robot_view.get_xy()
start_yaw = robot.robot_view.get_yaw()
print(f"[nav] robot class: {type(robot).__module__}.{type(robot).__name__}")
print(
    f"[nav] is molmo_spaces Robot subclass: {isinstance(robot, __import__('molmo_spaces.robots.abstract', fromlist=['Robot']).Robot)}"
)
print(f"[nav] spawn xy={start_xy.tolist()} yaw={start_yaw:.3f}rad")


def free(xy):
    try:
        return bool(raw_env.occ.is_free(np.asarray(xy, dtype=float)))
    except Exception:
        return True


# Pick reachable waypoints: probe outward along several headings from spawn and
# keep the ones the occupancy map says are free.
waypoints = []
for dist in (0.6, 1.0):
    for dtheta in (0.0, np.pi / 2, -np.pi / 2, np.pi):
        th = start_yaw + dtheta
        cand = np.array([start_xy[0] + dist * np.cos(th), start_xy[1] + dist * np.sin(th)])
        if free(cand):
            waypoints.append(np.array([cand[0], cand[1], th], dtype=float))
    if waypoints:
        break

if not waypoints:
    raise SystemExit("[nav] FAIL: no free waypoint found near spawn")
waypoints = waypoints[:2]
print(f"[nav] {len(waypoints)} reachable waypoint(s) selected\n")

overall_ok = True
for wi, wp in enumerate(waypoints):
    d0 = float(np.linalg.norm(robot.robot_view.get_xy() - wp[:2]))
    best = d0
    arrived = False
    for step in range(args.max_steps):
        # The controlling policy owns the WBC gait clock (see
        # G1Robot.advance_control_clock) -- this loop is playing that role.
        robot.advance_control_clock()
        action = robot.nav_action(wp)
        env.step(action)
        d = float(np.linalg.norm(robot.robot_view.get_xy() - wp[:2]))
        best = min(best, d)
        if step % 200 == 0:
            print(f"    [wp{wi}] step {step:4d} dist={d:.3f}m")
        if d <= args.tolerance:
            arrived = True
            print(f"    [wp{wi}] step {step:4d} dist={d:.3f}m  <- ARRIVED")
            break
    d1 = float(np.linalg.norm(robot.robot_view.get_xy() - wp[:2]))
    progress = d0 - d1
    ok = arrived or progress > 0.25
    overall_ok &= ok
    print(
        f"[nav] waypoint {wi} target=({wp[0]:.2f},{wp[1]:.2f}) "
        f"start_dist={d0:.3f}m final_dist={d1:.3f}m best={best:.3f}m "
        f"progress={progress:+.3f}m arrived={arrived} -> {'PASS' if ok else 'FAIL'}"
    )

print(f"\n[nav] RESULT: {'PASS' if overall_ok else 'FAIL'}")
raise SystemExit(0 if overall_ok else 1)
