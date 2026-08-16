"""Navigation smoke test for molmo_spaces/robots/g1.py's G1Robot.

The pick rollout (generate_ported_rollout.py) exercises this robot's grasp
path. This exercises the *navigation* path instead: it drives the robot to a
sequence of absolute [x, y, yaw] waypoints through
`G1Robot.waypoint_to_velocity_target` -- the same waypoint -> base-velocity
bridge a molmo_spaces navigation policy (e.g. AStarPlannerPolicy, which emits
exactly these waypoints) drives this robot through, unchanged.

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
    # occ_safe (the dilated map the planner itself uses) rather than occ: a
    # cell that is technically free but flush against a wall is not somewhere
    # a 0.15m-radius walking robot gets to.
    try:
        return bool(raw_env.occ_safe.is_free(np.asarray(xy, dtype=float)))
    except Exception:
        return True


def corridor_free(origin, heading, length, step=0.25):
    """Every cell along the segment free, not just its endpoint. The pick
    config spawns the robot 0.2-0.5m from a target on a counter -- i.e.
    facing furniture -- so an endpoint-only check happily picks a waypoint on
    the far side of the counter and then measures the robot failing to walk
    through it. That misreads as "the G1 cannot walk" (it is what §4 of
    NEXT_STEPS.md used to report); from a clear corridor the same command
    covers ~2.2m in 4.5s.
    """
    v = np.array([np.cos(heading), np.sin(heading)])
    return all(free(origin + s * v) for s in np.arange(step, length + step, step))


# Pick reachable waypoints: probe outward along several headings from spawn and
# keep the ones whose whole path the occupancy map says is free.
waypoints = []
# 16 headings, not 4: at a grasp spawn the robot is wedged against a counter,
# and the few directions that are actually open are rarely the cardinal ones.
headings = np.arange(0.0, 2 * np.pi, np.pi / 8)
for dist in (1.0, 0.6):
    for dtheta in headings:
        th = start_yaw + dtheta
        if not corridor_free(start_xy, th, dist):
            continue
        cand = np.array([start_xy[0] + dist * np.cos(th), start_xy[1] + dist * np.sin(th)])
        waypoints.append(np.array([cand[0], cand[1], th], dtype=float))
    if waypoints:
        break

if not waypoints:
    raise SystemExit(
        "[nav] SKIP: no clear corridor from spawn -- the robot is boxed in by furniture, "
        "which measures nothing about walking. Re-run with a different --seed."
    )
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
