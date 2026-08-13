"""Run our own pick pipeline (PickG1DataGenConfig) across a range of
procthor-10k-val houses, comparing against g1_molmo's own gold rollout (see
generate_gold_rollout.py in this directory) on the same scene/object family.

Handles HouseInvalidForTask and other per-house exceptions gracefully,
matching the real batch pipeline's own behavior (an isolated bad house
shouldn't kill the whole sweep). For each successfully-sampled house,
prints the target object, its start/final height, lift_height, whether both
gripper finger links are in contact with it at episode end (per
PickG1Task._fingers_in_contact, g1_molmo's own success criterion -- see
molmo_spaces/tasks/pick_g1_task.py), and the final success verdict.

Also logs, at every phase transition of FetchmanPickPlannerPolicy's
trajectory (walking/facing -> pregrasp -> grasp -> gripper-close -> lift),
the live TCP position (data.site_xpos for the "robot_0/right_grasp" site --
the same site g1_molmo's own GraspPolicy tracks via its per-hand `_sites`
dict, see ~/code/g1_molmo/molmospaces/agents/policy.py's `_hand`/`_HANDS`)
and the target object's position, plus the distance between them -- this is
the direct signal for whether the selected grasp candidate's IK-converged
pose actually lands on the object's surface (see
scripts/g1_molmo_comparison/generate_gold_rollout.py for the matching gold
side of this comparison).

Usage:
    conda run -n mlspaces python run_house_sweep.py [house_ind ...] [--viewer]

With no arguments, sweeps houses 0-7 (the range used throughout the session
that produced this script -- house 0's bowl in particular,
bowl_46a21212675e4d90993a86b1232e6f40_1_0_8 in procthor-10k-val/val_0.xml,
is the exact object g1_molmo's own generate_gold_rollout.py succeeds on).
Pass specific house indices to target known cases, e.g.:
    conda run -n mlspaces python run_house_sweep.py 1 2 5

--viewer opens MuJoCo's passive viewer (mujoco.viewer.launch_passive) to
watch the rollout live -- requires mjpython (not plain python) on macOS,
matching the launch mechanism generate_gold_rollout.py's own launch_viewer
option would need on the g1_molmo side:
    conda run -n mlspaces mjpython run_house_sweep.py 0 --viewer

config.seed is set to 0 (not this script's usual default of 1) whenever
--viewer is passed, matching generate_gold_rollout.py's cfg["seed"] = 0 --
so a --viewer run is the closest possible apples-to-apples match to the
gold rollout's own settings (same house/object, same seed).
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    PickG1DataGenConfig,
)
from molmo_spaces.data_generation.pipeline import setup_viewer
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

TCP_SITE_NAME = "robot_0/right_grasp"


def run_rollout_with_tcp_logging(task, policy, pickup_obj, end_on_success=True, viewer=None):
    """Re-implementation of ParallelRolloutRunner.run_single_rollout's step
    loop (see molmo_spaces/data_generation/pipeline.py), with per-phase-
    transition TCP/object position logging added. Behavior is otherwise
    identical: same is_done()/get_action_chunk/step_chunk/judge_success
    calls, same stop_on_success semantics.
    """
    observation, _info = task.reset()
    model = task.env.current_model
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_NAME)

    assert site_id >= 0, f"site {TCP_SITE_NAME!r} not found in model"

    last_phase = None
    last_conv_print_t = -1.0
    logged_candidate = False
    success = False
    while not task.is_done():
        phase = policy.get_phase() if hasattr(policy, "get_phase") else None

        # Log the selected grasp candidate exactly once, as soon as
        # _compute_target_poses has populated it (right after arrival) --
        # same format as g1_molmo's own "[G1_MOLMO_TRACE] plan(): hand=..."
        # print (~/code/g1_molmo/molmospaces/agents/policy.py) for a direct,
        # side-by-side comparison of the two pipelines' chosen grasp pose for
        # the same object. Gated on _arrived (not just "grasp" in
        # target_poses): reset() seeds target_poses["grasp"] with a
        # placeholder (the current gripper pose) solely so GraspPoseSensor
        # doesn't KeyError while still walking -- see FetchmanPickPlannerPolicy
        # .reset()'s docstring -- so checking dict membership alone would log
        # that placeholder instead of the real planned candidate.
        if not logged_candidate and getattr(policy, "_arrived", False) and "grasp" in getattr(policy, "target_poses", {}):
            grasp_pose = policy.target_poses["grasp"]
            grasp_pos = grasp_pose[:3, 3]
            grasp_rot_euler_xyz = R.from_matrix(grasp_pose[:3, :3]).as_euler("xyz")
            print(
                f"    [candidate] hand=right grasp_pos={grasp_pos.tolist()} "
                f"grasp_rot_euler_xyz={grasp_rot_euler_xyz.tolist()}"
            )
            logged_candidate = True

        if phase != last_phase:
            data = task.env.current_data
            tcp_pos = data.site_xpos[site_id].copy()
            obj_pos = pickup_obj.position.copy()
            dist = float(np.linalg.norm(tcp_pos - obj_pos))
            print(
                f"    [phase -> {phase}] t={data.time:.2f}s "
                f"tcp_pos={np.round(tcp_pos, 4).tolist()} "
                f"obj_pos={np.round(obj_pos, 4).tolist()} "
                f"dist={dist:.4f}m"
            )
            last_phase = phase
            last_conv_print_t = data.time

        # Periodic (every ~1s of sim time) within-phase convergence trace:
        # is pos_err/rot_err to the *current* segment's own target actually
        # decreasing over time, plateaued at some nonzero value, or
        # oscillating? A phase that never fires its "[phase -> ...]"
        # transition print (stuck) is otherwise invisible until the episode
        # ends -- this makes that visible while it's still happening.
        # FetchmanPickPlannerPolicy's grasp-phase state machine (direct port
        # of g1_molmo's G1Controller, not TCPMoveSegments -- see that
        # module's docstring) exposes the phase's *goal* position/rotation
        # directly as attributes (_pregrasp/_grasp_pos/_grasp_rot/
        # _lift_pos), matching g1_molmo's own attribute names exactly -- map
        # each of the 7 phase names to its goal here rather than through
        # policy.target_poses (only has one populated entry, "grasp", for
        # GraspPoseSensor compatibility -- see reset()/_start_grasp).
        phase_goal_pos = {
            "approach": lambda: policy._pregrasp,
            "descend": lambda: policy._grasp_pos,
            "open_hold": lambda: policy._grasp_pos,
            "close": lambda: policy._grasp_pos,
            "post_close": lambda: policy._grasp_pos,
            "lift": lambda: policy._lift_pos,
        }.get(phase)
        if phase_goal_pos is not None and getattr(policy, "_grasp_rot", None) is not None:
            data = task.env.current_data
            if data.time - last_conv_print_t >= 1.0:
                goal_pos = phase_goal_pos()
                goal_rot = policy._grasp_rot
                tcp_pos = data.site_xpos[site_id].copy()
                tcp_rot = R.from_matrix(data.site_xmat[site_id].reshape(3, 3))
                pos_err = float(np.linalg.norm(tcp_pos - goal_pos))
                rot_err = float((tcp_rot.inv() * R.from_matrix(goal_rot)).magnitude())
                print(
                    f"        [converging to {phase}] t={data.time:.2f}s "
                    f"pos_err={pos_err:.4f}m rot_err={np.degrees(rot_err):.1f}deg"
                )
                last_conv_print_t = data.time

        action_chunk = policy.get_action_chunk(observation) or [policy.get_action(observation)]
        if action_chunk[0] is None:
            print("    Policy returned None action, ending episode")
            break
        observation, reward, terminal, truncated, infos = task.step_chunk(
            action_chunk, stop_on_success=end_on_success
        )
        if viewer is not None:
            if not viewer.is_running():
                print("    Viewer closed, ending episode")
                break
            viewer.sync()
        if end_on_success and "success" in infos[0] and infos[0]["success"]:
            success = True
            break

    return task.judge_success() if hasattr(task, "judge_success") else success

args = sys.argv[1:]
use_viewer = "--viewer" in args
args = [a for a in args if a != "--viewer"]
house_inds = [int(a) for a in args] or list(range(8))

for house_ind in house_inds:
    config = PickG1DataGenConfig()
    # Matches generate_gold_rollout.py's cfg["seed"] = 0 whenever --viewer is
    # passed, so a --viewer run is the closest apples-to-apples match to the
    # gold rollout's own settings (same house/object, same seed) -- this
    # script's usual sweep default (seed=1) is left alone otherwise so the
    # existing house-sweep numbers already discussed aren't disturbed.
    config.seed = 0 if use_viewer else 1
    config.use_passive_viewer = use_viewer
    config.use_wandb = False
    config.task_sampler_config.house_inds = [house_ind]

    task_sampler = config.task_sampler_config.task_sampler_class(config)
    task_sampler.reset()
    try:
        task = task_sampler.sample_task()
    except HouseInvalidForTask as e:
        print(f"house {house_ind}: HouseInvalidForTask ({e})")
        task_sampler.env.close()
        continue
    except Exception as e:
        # Pre-existing, already-diagnosed scene-compatibility issue for some
        # procthor houses (e.g. house 3 in this range: "FactorizeHessian:
        # rank-deficient sparse Hessian" at the very first env mj_forward,
        # tied to mjINT_IMPLICITFAST needing to factorize a singular Hessian
        # for that house's initial contact configuration) -- unrelated to
        # the pick pipeline itself; the real batch pipeline already handles
        # this by moving on to the next house.
        print(f"house {house_ind}: task sampling RAISED {type(e).__name__}: {e}")
        try:
            task_sampler.env.close()
        except Exception:
            pass
        continue

    task.reset()
    env = task.env
    target_name = config.task_config.pickup_obj_name
    om = env.object_managers[env.current_batch_index]
    pickup_obj = om.get_object_by_name(target_name)
    start_z = config.task_config.pickup_obj_start_pose[2]

    # Hard-override the robot's spawn xy/yaw to exactly match g1_molmo's own
    # gold rollout's spawn pose (captured via generate_gold_rollout.py's
    # agent._xy()/agent._yaw() print, seed=0, episode 2 -- the episode that
    # succeeds). This removes the ~0.2m position / ~50deg yaw discrepancy
    # between task_sampler.sample_task()'s stochastic place_robot_near
    # placement and gold's own spawn as a confound when comparing TCP
    # convergence between the two pipelines on the same object. Keeps the
    # robot's current z/roll/pitch (place_robot_near's own terrain-following
    # placement) and only overrides x, y, and yaw. Mirrors the pose-setting
    # pattern in molmo_spaces/env/env.py's place_robot_near
    # (robot_view.base.pose = ...; mujoco.mj_forward(...)).
    GOLD_SPAWN_XY = np.array([8.947922122467782, 3.5005874958063803])
    GOLD_SPAWN_YAW_RAD = 1.704219
    robot_view = env.current_robot.robot_view
    robot_pose = robot_view.base.pose.copy()
    robot_pose[0, 3], robot_pose[1, 3] = GOLD_SPAWN_XY
    robot_pose[:3, :3] = R.from_euler("z", GOLD_SPAWN_YAW_RAD).as_matrix()
    robot_view.base.pose = robot_pose
    mujoco.mj_forward(env.current_model, env.current_data)

    policy = config.policy_config.policy_factory(config, task)
    task.register_policy(policy)
    viewer = setup_viewer(config, task, policy, None) if use_viewer else None
    print(f"house {house_ind}: target={target_name} start_z={start_z:.3f}")
    try:
        success = run_rollout_with_tcp_logging(
            task=task, policy=policy, pickup_obj=pickup_obj, end_on_success=True, viewer=viewer
        )
        final_z = pickup_obj.position[2]
        lift_height = final_z - start_z
        contact = (
            task._fingers_in_contact(env.current_data, pickup_obj)
            if hasattr(task, "_fingers_in_contact")
            else None
        )
        print(
            f"house {house_ind}: target={target_name} start_z={start_z:.3f} "
            f"final_z={final_z:.3f} lift_height={lift_height:.3f} "
            f"fingers_in_contact={contact} -> pick success={success}"
        )
    except Exception as e:
        print(f"house {house_ind}: target={target_name} -> pick RAISED {type(e).__name__}: {e}")

    if viewer is not None:
        viewer.close()
    task_sampler.env.close()
