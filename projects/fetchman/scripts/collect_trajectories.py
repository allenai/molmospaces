"""Collect N trajectories through either import path and dump them as JSON.

    python projects/fetchman/scripts/collect_trajectories.py --stack native --episodes 10 --out /tmp/native.json
    python projects/fetchman/scripts/collect_trajectories.py --stack port   --episodes 10 --out /tmp/port.json
    python projects/fetchman/scripts/collect_trajectories.py --compare /tmp/native.json /tmp/port.json

`--stack native` drives molmo_spaces' own modules (molmo_spaces.tasks.
fetchman.tasks, fetchman.configs); `--stack port` reaches the
same classes through the `fetchman.tasks` shim that preserves gold's original
import paths. Both must produce identical trajectories.

Each episode records the discrete invariants the gold-parity gate compares
(target, spawn pose, step count, sim time, success) plus a checksum over the
final qpos/qvel, so continuous state is compared too rather than just the
summary numbers.
"""

import _bootstrap  # noqa: F401  -- must precede any molmo_spaces or fetchman import

import argparse
import hashlib
import json

import numpy as np

SCENE = "scenes/procthor-10k-val/val_0.xml"
TIME_LIMIT = 60.0
SETTLE_STEPS = 70


def _load(stack):
    """Return (get_config, make_task_sampler) from the requested import path."""
    from fetchman.configs.bowl_fetchman import get_config

    if stack == "native":
        from fetchman.tasks import make_task_sampler
    elif stack == "port":
        from fetchman.tasks import make_task_sampler
    else:
        raise ValueError(stack)
    return get_config, make_task_sampler


def _checksum(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def collect(stack, episodes, seed):
    from molmo_spaces.controllers.g1_wbc import flat15_to_move_groups
    from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy_g1 import G1PickAgent

    get_config, make_task_sampler = _load(stack)

    cfg = get_config()
    cfg.task_sampler_config.scene = SCENE
    cfg.task_sampler_config.randomize_scene = False
    cfg.task_config.randomize_object = False
    cfg.use_passive_viewer = False
    cfg.seed = seed

    task_sampler = make_task_sampler(cfg)
    raw_env = task_sampler.env
    agent = G1PickAgent()
    task_sampler.set_agent(agent)

    out = []
    for episode in range(episodes):
        for _ in range(20):
            task = task_sampler.sample_task()
            obs, info = task.reset()
            agent.reset(info)
            obs = task._build_obs()
            if agent.has_path:
                break

        rec = {
            "episode": episode,
            "target": info.get("target_name"),
            "spawn_xy": [round(v, 12) for v in agent._xy().tolist()],
            "spawn_yaw": round(float(agent._yaw()), 12),
        }

        hold = np.zeros(15, dtype=np.float32)
        hold[3] = float(getattr(agent._low_level, "_height_cmd", obs["base_height"][0]))
        hold[4:7] = obs["joint_pos"][12:15]
        hold[7:14] = obs["joint_pos"][22:29]
        hold[14] = float(obs["joint_pos"][29])
        settle_bad = False
        for _ in range(SETTLE_STEPS):
            obs, _, terminated, truncated, info = task.step(flat15_to_move_groups(hold))
            if terminated or truncated or not raw_env.occ.is_free(raw_env.robot.get_xy()):
                settle_bad = True
                break
        if settle_bad:
            rec.update(rejected=True)
            out.append(rec)
            continue
        agent.set_step_info(info)

        steps = 0
        while raw_env.time < TIME_LIMIT:
            action = agent.sample_actions(obs)
            obs, _, terminated, truncated, info = task.step(flat15_to_move_groups(action))
            agent.set_step_info(info)
            steps += 1
            if terminated or truncated or agent.done:
                break

        d = raw_env.scene.data
        rec.update(
            rejected=False,
            steps=steps,
            sim_time=round(float(raw_env.time), 9),
            success=bool(info.get("success")),
            terminated=bool(terminated),
            truncated=bool(truncated),
            qpos_sha=_checksum(d.qpos),
            qvel_sha=_checksum(d.qvel),
        )
        out.append(rec)
        print(
            f"[{stack}] ep{episode}: target={rec['target']} steps={rec.get('steps')} "
            f"success={rec.get('success')} qpos={rec.get('qpos_sha')}"
        )
    return out


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    if len(a) != len(b):
        print(f"FAIL: {len(a)} vs {len(b)} episodes")
        return 1
    bad = 0
    for x, y in zip(a, b):
        diffs = [k for k in set(x) | set(y) if x.get(k) != y.get(k)]
        if diffs:
            bad += 1
            print(f"\n--- episode {x.get('episode')} differs on {sorted(diffs)} ---")
            for k in sorted(diffs):
                print(f"    {k}: {x.get(k)!r}  vs  {y.get(k)!r}")
    if bad:
        print(f"\nFAIL: {bad}/{len(a)} trajectories differ")
        return 1
    fields = sorted({k for r in a for k in r})
    n_run = sum(1 for r in a if not r.get("rejected"))
    print(f"PASS: {len(a)}/{len(a)} trajectories identical "
          f"({n_run} simulated, {len(a) - n_run} rejected at settle)")
    print(f"  fields compared: {fields}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["native", "port"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        raise SystemExit(compare(*args.compare))
    recs = collect(args.stack, args.episodes, args.seed)
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=1)
    print(f"wrote {len(recs)} trajectories to {args.out}")


if __name__ == "__main__":
    main()
