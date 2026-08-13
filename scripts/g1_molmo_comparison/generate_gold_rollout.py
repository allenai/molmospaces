"""Run g1_molmo's own reference bowl config against a real procthor-10k-val
house, retrying episodes until a successful pick rollout is collected --
i.e. produce a genuine "gold" rollout to compare our own pick pipeline
against.

Requires a checkout of ~/code/g1_molmo (edit G1_MOLMO_ROOT below if yours is
elsewhere) with its own conda env active, and procthor-10k-val scenes
downloaded into it (they are not present by default in a fresh checkout --
see molmospaces/scripts/download_scenes.py --project-root <path>
--procthor-10k-val N --procthor-10k-train 0 --procthor-10k-test 0
--procthor-objaverse-train 0 --procthor-objaverse-val 0
--holodeck-objaverse-train 0 --holodeck-objaverse-val 0 --ithor 0
in that checkout; a bare --project-root is required since the script's own
default points at the original author's machine, and downloading only
procthor-10k-val avoids the very large, rate-limited full grasp-archive
cascade the default --objects/--grasps behavior otherwise triggers).

Run from the g1_molmo checkout's own conda env, e.g.:
    conda run -n g1_molmo python generate_gold_rollout.py

Pass --viewer to open MuJoCo's passive viewer (mujoco.viewer.launch_passive,
via g1_molmo's own env launch_viewer option) and watch the rollout live --
requires mjpython (not plain python) on macOS, matching run_house_sweep.py's
own --viewer flag in this directory:
    conda run -n g1_molmo mjpython generate_gold_rollout.py --viewer

Uses randomize_object=False so the bowl stays the native THOR asset
(Bowl_16 in val_0.xml), whose grasps are already in the fully-installed
'droid' library -- avoiding the much larger droid_objaverse archive
randomize_object=True would otherwise need.

On success, prints the target object name/scene and saves the full reset
state to /tmp/g1_molmo_bowl_success.json via env.export_reset_state --
note this captures the state at the *end* of the episode (object already
lifted), not the initial spawn; if you need the initial spawn pose, capture
robot_xy/agent._xy() right after agent.reset() instead, as this script does
in its own per-episode print. See run_house_sweep.py to compare our own
repo's pick pipeline against a target extracted this way.
"""
import argparse
import sys
from pathlib import Path

G1_MOLMO_ROOT = Path("/Users/maxa/code/g1_molmo")
for p in (str(G1_MOLMO_ROOT), str(G1_MOLMO_ROOT / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from molmospaces.env import make_env
from molmospaces.agents.policy import G1Controller
from molmospaces.configs.bowl_mixed_grasponly import get_config

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true", help="Launch MuJoCo's passive viewer (requires mjpython on macOS).")
args = parser.parse_args()

SCENE = "scenes/procthor-10k-val/val_0.xml"
MAX_EPISODES = 15
TIME_LIMIT = 60.0

cfg = get_config().copy_and_resolve_references()
cfg["scene"] = SCENE
cfg["randomize_scene"] = False
cfg["randomize_object"] = False
cfg["launch_viewer"] = args.viewer
cfg["seed"] = 0
cfg = cfg.to_dict()

env = make_env(cfg)
agent = G1Controller()
agent.setup(env.scene.model, env.scene.data)
agent.set_env(env)
env.set_agent(agent)

for episode in range(MAX_EPISODES):
    for _ in range(20):
        obs, info = env.reset()
        agent.reset(info)
        obs = env._build_obs()
        if agent.has_path:
            break
    print(
        f"\n[gold] === episode {episode}: target={info.get('target_name')} "
        f"robot_xy={agent._xy().tolist()} robot_yaw_rad={agent._yaw():.6f} "
        f"robot_yaw_deg={np.degrees(agent._yaw()):.2f} ==="
    )

    SETTLE_STEPS = 70
    hold_action = np.zeros(15, dtype=np.float32)
    hold_action[3] = float(getattr(agent, "_height_cmd", obs["base_height"][0]))
    hold_action[4:7] = obs["joint_pos"][12:15]
    hold_action[7:14] = obs["joint_pos"][22:29]
    hold_action[14] = float(obs["joint_pos"][29])
    settle_bad = False
    for _ in range(SETTLE_STEPS):
        obs, _, terminated, truncated, info = env.step(hold_action)
        if terminated or truncated or not env.occ.is_free(env.robot.get_xy()):
            settle_bad = True
            break
    if settle_bad:
        print(f"[gold] episode {episode}: rejected after settle, retrying")
        continue
    agent.set_step_info(info)

    step_count = 0
    last_phase = None
    while env.time < TIME_LIMIT:
        phase = agent._grasp_phase if agent._arrived else "walking"
        if phase != last_phase:
            sid = agent._sites[agent._hand]
            tcp_pos = agent._data.site_xpos[sid].copy()
            obj_pos = agent._env.task.target.position(agent._data)
            dist = float(np.linalg.norm(tcp_pos - obj_pos))
            print(
                f"    [gold phase -> {phase}] t={env.time:.2f}s "
                f"tcp_pos={np.round(tcp_pos, 4).tolist()} "
                f"obj_pos={np.round(obj_pos, 4).tolist()} "
                f"dist={dist:.4f}m"
            )
            last_phase = phase

        action = agent.sample_actions(obs)
        obs, r, terminated, truncated, info = env.step(action)
        agent.set_step_info(info)
        step_count += 1
        if terminated or truncated or agent.done:
            break

    success = bool(info.get("success"))
    print(
        f"[gold] episode {episode} result: steps={step_count} "
        f"sim_time={env.time:.2f}s success={success} terminated={terminated} "
        f"truncated={truncated} agent.done={agent.done}"
    )
    if success:
        print(f"\n[gold] SUCCESS on episode {episode}!")
        print(f"[gold] target={info.get('target_name')} scene={SCENE}")
        st = env.export_reset_state(info=info)
        import json

        out_path = "/tmp/g1_molmo_bowl_success.json"
        with open(out_path, "w") as f:
            json.dump(st, f)
        print(f"[gold] saved end-of-episode reset state to {out_path}")
        break
else:
    print(f"\n[gold] no success after {MAX_EPISODES} episodes")
