"""Interactive single-worker rollout collector for molmo_spaces.g1_molmo_port
(the ported copy of g1_molmo's own env_g1ms.py/agents/policy_g1ms.py pick
pipeline, see molmo_spaces/g1_molmo_port/__init__.py).

Mirrors the reset/settle/step/record loop in g1_molmo's own main.py, but
against the ported env/agent/recorder instead of g1_molmo's own molmospaces
package -- this is the entrypoint scripts/collect_single.sh execs.

Usage:
    conda run -n mlspaces python scripts/g1_molmo_port_comparison/collect_single_main.py \\
        --env=molmo_spaces/g1_molmo_port/configs/bowl_mixed_grasponly.py [flags]

--env accepts a filesystem path to a config module exposing get_config()
(ml_collections.ConfigDict), e.g. any file under molmo_spaces/g1_molmo_port/configs/.
"""

import argparse
import importlib.util
import os
import signal
import time as _time
from pathlib import Path

import numpy as np


def _load_env_config(path):
    spec = importlib.util.spec_from_file_location("_g1_port_env_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def get_args():
    parser = argparse.ArgumentParser(
        description="molmo_spaces.g1_molmo_port single-worker collector"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU device ID (-1 = no EGL/CUDA device select)"
    )
    parser.add_argument("--episodes", type=int, default=10, help="max episodes")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--render", action="store_true", help="render to viewer")
    parser.add_argument(
        "--cv2",
        action="store_true",
        help="with --render, show camera feed in a cv2 window instead of the MuJoCo viewer",
    )
    parser.add_argument(
        "--record", action="store_true", help="record successful episodes to a LeRobot dataset"
    )
    parser.add_argument(
        "--save_failures", action="store_true", help="also save failed episodes (when --record)"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="throttle rollout so wall-clock time matches sim time",
    )
    parser.add_argument(
        "--debug", action="store_true", help="overlay debug markers in the MuJoCo viewer"
    )
    parser.add_argument(
        "--repo_id", type=str, default="local/g1_pick", help="LeRobot dataset repo_id"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="local root for LeRobot dataset"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="dataset logging fps (physics runs at 200)"
    )
    parser.add_argument(
        "--env", type=str, required=True, help="path to a config module exposing get_config()"
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.gpu != -1:
        # g1_molmo's own molmospaces.egl.set_egl_device (nvidia-smi CUDA index ->
        # EGL device index) hasn't been ported here; this repo's collect_single.sh
        # only exercises GPU=-1 (macOS/CPU passive-viewer path), so just set the
        # env vars directly instead of remapping.
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from molmo_spaces.g1_molmo_port.env_g1ms import make_env
    from molmo_spaces.policy.solvers.object_manipulation.g1_pick_policy import G1Controller

    cfg = _load_env_config(args.env)
    cfg["seed"] = args.seed
    cfg["launch_viewer"] = args.render and not args.cv2
    cfg = cfg.to_dict()

    env = make_env(cfg)
    raw_env = env.env
    raw_env.debug = bool(args.debug)

    cv2 = None
    CV2_WINDOW = "cameras (face | head | wrist)"
    if args.render and args.cv2:
        import cv2  # noqa: F401

        cv2.namedWindow(CV2_WINDOW, cv2.WINDOW_NORMAL)

    agent = G1Controller()
    agent.setup(raw_env.scene.model, raw_env.scene.data)
    agent.set_env(raw_env)
    env.set_agent(agent)
    step_dt = raw_env.robot.n_substeps * raw_env.scene.model.opt.timestep

    recorder = None
    if args.record:
        from molmo_spaces.g1_molmo_port.dataset.lerobot_recorder import LeRobotRecorder

        physics_fps = int(round(1.0 / raw_env.scene.model.opt.timestep))
        recorder = LeRobotRecorder(
            repo_id=args.repo_id,
            root=Path(args.data_dir) / args.repo_id,
            env=raw_env,
            action_dim=env.action_space.shape[0],
            fps=args.fps,
            physics_fps=physics_fps,
        )

    stop = [False]

    def _shutdown(*_):
        stop[0] = True
        print("\nStopping...")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _cv2_render():
        if cv2 is None:
            return
        panel = raw_env.render_debug_panel()
        if panel is None:
            return
        cv2.imshow(CV2_WINDOW, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        key = cv2.pollKey() & 0xFF
        if key == ord(" "):
            raw_env._skip_episode = True
        elif key in (ord("q"), 27):
            stop[0] = True
        if cv2.getWindowProperty(CV2_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            stop[0] = True

    GRIP_OPEN, GRIP_CLOSED = -0.0222, 0.0245
    time_limit = 60.0
    results = []
    try:
        episode = 0
        while episode < args.episodes:
            if stop[0]:
                break
            while not stop[0]:
                try:
                    obs, info = env.reset()
                except RuntimeError as e:
                    print(f"[main] env.reset failed: {e}; retrying", flush=True)
                    continue
                reset_info = dict(info)
                agent.reset(info)
                obs = env._build_obs()
                if args.render:
                    env.sync_viewer()
                    _cv2_render()
                if agent.has_path:
                    break
            if stop[0]:
                break

            SETTLE_STEPS = 70
            hold_action = np.zeros(15, dtype=np.float32)
            hold_action[3] = float(getattr(agent._low_level, "_height_cmd", obs["base_height"][0]))
            hold_action[4:7] = obs["joint_pos"][12:15]
            hold_action[7:14] = obs["joint_pos"][22:29]
            hold_action[14] = float(obs["joint_pos"][29])
            settle_bad = False
            for _ in range(SETTLE_STEPS):
                if stop[0]:
                    break
                obs, _, terminated, truncated, info = env.step(hold_action)
                if args.render:
                    _cv2_render()
                if terminated or truncated or not raw_env.occ.is_free(raw_env.robot.get_xy()):
                    settle_bad = True
                    break
            if settle_bad:
                print(
                    f"[main] rejecting reset after settle: xy={raw_env.robot.get_xy()}", flush=True
                )
                continue
            agent.set_step_info(info)

            if recorder is not None:
                recorder.begin_episode(reset_info)

            print(f"\n[main] === episode {episode}: target={info.get('target_name')} ===")

            while raw_env.time < time_limit and not stop[0]:
                action = agent.sample_actions(obs)
                if recorder is not None:
                    save_action = action.copy()
                    intent = float(getattr(agent, "_gripper_intent", 0.0))
                    save_action[14] = GRIP_CLOSED if intent > 0.5 else GRIP_OPEN
                    recorder.add_step(obs, save_action)
                exec_action = (
                    agent.perturb_action_for_rollout(action)
                    if hasattr(agent, "perturb_action_for_rollout")
                    else action
                )
                if args.realtime:
                    step_t0 = _time.perf_counter()
                obs, r, terminated, truncated, info = env.step(exec_action)
                agent.set_step_info(info)
                if args.render:
                    env.sync_viewer()
                    _cv2_render()
                if args.render and env.consume_skip_episode():
                    break
                if terminated or truncated or agent.done:
                    break
                if args.render and not args.cv2 and not env.viewer_running:
                    stop[0] = True
                    break
                if args.realtime:
                    remaining = step_dt - (_time.perf_counter() - step_t0)
                    if remaining > 0:
                        _time.sleep(remaining)

            success = bool(info.get("success", False))
            results.append(success)
            print(
                f"[main] episode {episode} result: success={success} sim_time={raw_env.time:.2f}s"
            )

            if recorder is not None:
                if success or args.save_failures:
                    recorder.save_episode()
                else:
                    recorder.discard_episode()

            episode += 1
    finally:
        if recorder is not None:
            recorder.close()
        n = len(results)
        n_success = sum(results)
        print(f"\n[main] done: {n_success}/{n} episodes succeeded")


if __name__ == "__main__":
    main()
