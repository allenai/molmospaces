"""Run molmo_spaces.g1_molmo_port (a wholesale copy of g1_molmo's own
env_g1ms.py/agents/policy_g1ms.py pick pipeline, see molmo_spaces/
g1_molmo_port/__init__.py) against the same real procthor-10k-val house/seed
as g1_molmo's own generate_gold_rollout.py, retrying episodes until a
successful pick rollout is collected.

Unlike molmo_spaces' own independent "fetchman" rewrite (policy/solvers/
object_manipulation/fetchman_pick_planner_policy.py), this is a near-verbatim
port -- it's expected to reproduce gold's own printed trace byte-for-byte
(module names and the "[gold]"/"[ours]" print prefix aside), the same way
g1_molmo's own generate_our_rollout.py already does against generate_gold_
rollout.py.

Run from this repo's own conda env (mlspaces), from the repo root:
    conda run -n mlspaces python scripts/g1_molmo_port_comparison/generate_ported_rollout.py

Requires a g1_molmo checkout with procthor-10k-val scenes already downloaded
(this script's assets resolve through molmo_spaces.g1_molmo_port.ASSETS_DIR,
which defaults to ~/code/g1_molmo/molmospaces/assets -- override with the
G1_MOLMO_ASSETS_DIR env var if yours is elsewhere).
"""

import argparse

import numpy as np

from molmo_spaces.g1_molmo_port.configs.bowl_mixed_grasponly import get_config
from molmo_spaces.g1_molmo_port.env_g1ms import make_env
from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import G1Controller

parser = argparse.ArgumentParser()
parser.add_argument(
    "--viewer",
    action="store_true",
    help="Launch MuJoCo's passive viewer (requires mjpython on macOS).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Env seed (default 0, matching generate_gold_rollout.py's own default).",
)
args = parser.parse_args()

SCENE = "scenes/procthor-10k-val/val_0.xml"
MAX_EPISODES = 15
TIME_LIMIT = 60.0

# Trace every obs key this often (in policy steps), plus once after settle and
# once at episode end. Without this the rollout only ever *reads* two of the 14
# OBS_SENSORS values (base_height, joint_pos, for hold_action) -- the other
# twelve never reach the trace, so the strict ported-vs-ported gate in
# NEXT_STEPS.md would pass even if they broke outright. Fixed step cadence, so
# the dump points are deterministic and comparable across runs.
OBS_DUMP_EVERY = 500


def _dump_obs(episode, step_count, obs):
    for k in sorted(obs):
        v = np.asarray(obs[k], dtype=np.float64).ravel()
        print(f"    [ported obs ep{episode} step{step_count}] {k}={np.round(v, 6).tolist()}")


cfg = get_config().copy_and_resolve_references()
cfg["scene"] = SCENE
cfg["randomize_scene"] = False
cfg["randomize_object"] = False
cfg["launch_viewer"] = args.viewer
cfg["seed"] = args.seed
cfg = cfg.to_dict()

env = make_env(cfg)
raw_env = getattr(env, "env", env)
agent = G1Controller()
agent.setup(raw_env.scene.model, raw_env.scene.data)
agent.set_env(raw_env)
env.set_agent(agent)

for episode in range(MAX_EPISODES):
    for _ in range(20):
        obs, info = env.reset()
        agent.reset(info)
        obs = env._build_obs()
        if agent.has_path:
            break
    print(
        f"\n[ported] === episode {episode}: target={info.get('target_name')} "
        f"robot_xy={agent._xy().tolist()} robot_yaw_rad={agent._yaw():.6f} "
        f"robot_yaw_deg={np.degrees(agent._yaw()):.2f} ==="
    )

    SETTLE_STEPS = 70
    hold_action = np.zeros(15, dtype=np.float32)
    hold_action[3] = float(getattr(agent._low_level, "_height_cmd", obs["base_height"][0]))
    hold_action[4:7] = obs["joint_pos"][12:15]
    hold_action[7:14] = obs["joint_pos"][22:29]
    hold_action[14] = float(obs["joint_pos"][29])
    settle_bad = False
    for _ in range(SETTLE_STEPS):
        obs, _, terminated, truncated, info = env.step(hold_action)
        if terminated or truncated or not raw_env.occ.is_free(raw_env.robot.get_xy()):
            settle_bad = True
            break
    if settle_bad:
        print(f"[ported] episode {episode}: rejected after settle, retrying")
        continue
    agent.set_step_info(info)
    _dump_obs(episode, 0, obs)

    step_count = 0
    last_phase = None
    while raw_env.time < TIME_LIMIT:
        phase = agent._grasp_phase if agent._arrived else "walking"
        if phase != last_phase:
            sid = agent._sites[agent._hand]
            tcp_pos = agent._low_level._data.site_xpos[sid].copy()
            obj_pos = agent._env.task.target.position(agent._low_level._data)
            dist = float(np.linalg.norm(tcp_pos - obj_pos))
            print(
                f"    [ported phase -> {phase}] t={raw_env.time:.2f}s "
                f"tcp_pos={np.round(tcp_pos, 4).tolist()} "
                f"obj_pos={np.round(obj_pos, 4).tolist()} "
                f"dist={dist:.4f}m"
            )
            last_phase = phase

        action = agent.sample_actions(obs)
        obs, r, terminated, truncated, info = env.step(action)
        agent.set_step_info(info)
        step_count += 1
        if step_count % OBS_DUMP_EVERY == 0:
            _dump_obs(episode, step_count, obs)
        if terminated or truncated or agent.done:
            break

    _dump_obs(episode, step_count, obs)
    success = bool(info.get("success"))
    print(
        f"[ported] episode {episode} result: steps={step_count} "
        f"sim_time={raw_env.time:.2f}s success={success} terminated={terminated} "
        f"truncated={truncated} agent.done={agent.done}"
    )
    if success:
        print(f"\n[ported] SUCCESS on episode {episode}!")
        print(f"[ported] target={info.get('target_name')} scene={SCENE}")
        st = env.export_reset_state(info=info)
        import json

        out_path = "/tmp/g1_molmo_bowl_success_ported.json"
        with open(out_path, "w") as f:
            json.dump(st, f)
        print(f"[ported] saved end-of-episode reset state to {out_path}")
        break
else:
    print(f"\n[ported] no success after {MAX_EPISODES} episodes")
