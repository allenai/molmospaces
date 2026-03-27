import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjviewer
import numpy as np

from molmo_spaces.env.arena.arena_utils import load_env_with_objects_with_tweaks

TIMESTEP = 0.002
STARTING_FORCE = 10.0
N_SECS = 4  # or increasing this,
FORCE_STEPS = N_SECS * 500  # 1000/2= 500 steps = 1 second
OPEN_RANGE_PERCENT = 0.667

model: mj.MjModel | None = None
data: mj.MjData | None = None


@dataclass
class JointInfo:
    jnt_id: int = -1
    jnt_name: str = ""
    jnt_type: mj.mjtJoint = mj.mjtJoint.mjJNT_HINGE
    jnt_qposadr: int = -1
    jnt_dofadr: int = -1
    jnt_range_min: float = -np.inf
    jnt_range_max: float = np.inf
    jnt_range_diff: float = np.inf
    jnt_qpos_start: float = 0
    jnt_qpos_end: float = 0


class Context:
    def __init__(self) -> None:
        self.joints: list[JointInfo] = []
        self.joint_idx: int = 0
        self.dirty_reset: bool = False
        self.dirty_pause: bool = False
        self.dirty_start_test: bool = False
        self.dirty_start_test_batch: bool = False
        self.force_magnitude: float = STARTING_FORCE


context: Context = Context()


def find_articulated_joints(model: mj.MjModel) -> list[JointInfo]:
    articulated_joints_ids = []
    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id].item()
        jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT.value, jnt_id)
        jnt_qposadr = model.jnt_qposadr[jnt_id].item()
        jnt_dofadr = model.jnt_dofadr[jnt_id].item()
        jnt_range_min, jnt_range_max = model.jnt_range[jnt_id]
        jnt_range_diff = abs(jnt_range_max - jnt_range_min)

        if jnt_type == mj.mjtJoint.mjJNT_FREE:
            continue
        articulated_joints_ids.append(
            JointInfo(
                jnt_id,
                jnt_name,
                jnt_type,
                jnt_qposadr,
                jnt_dofadr,
                jnt_range_min,
                jnt_range_max,
                jnt_range_diff,
            )
        )
    return articulated_joints_ids


def key_callback(keycode: int) -> None:
    global model, data, context

    if model is None or data is None:
        return

    # print(f"keycode: {keycode}, chr_key: {chr(keycode)}")

    if keycode == 265:  # up arrow key
        context.joint_idx = (context.joint_idx + 1) % len(context.joints)
        c_joint = context.joints[context.joint_idx]
        print(f"Current joint: idx={context.joint_idx}, name={c_joint.jnt_name}")
    elif keycode == 264:  # down arrow key
        context.joint_idx = (context.joint_idx - 1) % len(context.joints)
        c_joint = context.joints[context.joint_idx]
        print(f"Current joint: idx={context.joint_idx}, name={c_joint.jnt_name}")
    elif keycode == 259:  # backspace key
        context.dirty_reset = True
    elif keycode == 32:  # space key
        context.dirty_pause = not context.dirty_pause
        print("Paused" if context.dirty_pause else "Running")
    elif keycode == 82:  # R key
        context.dirty_start_test = True
    elif keycode == 262:  # right arrow key
        context.force_magnitude *= 1.1
        print(f"force-magnitude: {context.force_magnitude}")
    elif keycode == 263:  # left arrow key
        context.force_magnitude *= 0.9
        print(f"force-magnitude: {context.force_magnitude}")
    elif keycode == 81:  # Q key
        context.dirty_start_test_batch = True


def main() -> int:
    global model, data, context

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--house",
        type=str,
        default="",
        help="The house to load for the test",
    )

    args = parser.parse_args()

    if args.house == "":
        print("Should give a specific house to test")
        return 1

    xml_file_path = Path(args.house)
    model, _, _ = load_env_with_objects_with_tweaks(
        xml_file_path.as_posix(),
        remove_objects_within_inner_sites=True,
        # exclude_floor_contact_with_fridges=True,
    )
    data = mj.MjData(model)

    model.vis.scale.contactwidth = 0.01
    model.vis.scale.contactheight = 0.01

    context.joints = find_articulated_joints(model)

    test_joint: JointInfo | None = None
    test_running = False
    test_done = False
    is_batch = False

    with mjviewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            if context.dirty_reset:
                context.dirty_reset = False
                mj.mj_resetData(model, data)

            if context.dirty_start_test:
                mj.mj_resetData(model, data)
                context.dirty_start_test = False
                is_batch = False

                test_joint = context.joints[context.joint_idx]
                test_joint.jnt_qpos_start = data.qpos[test_joint.jnt_qposadr].item()

                dist_to_low = abs(test_joint.jnt_qpos_start - test_joint.jnt_range_min)
                dist_to_high = abs(test_joint.jnt_qpos_start - test_joint.jnt_range_max)

                sign = 1.0 if dist_to_low < dist_to_high else -1.0
                force_to_apply = context.force_magnitude * sign

                data.qfrc_applied[test_joint.jnt_dofadr] = force_to_apply
                test_running = True
                test_done = False
                print(f"Started test for joint: {test_joint.jnt_name}")
                print(f"force applied: {force_to_apply}")

            if context.dirty_start_test_batch:
                mj.mj_resetData(model, data)
                context.dirty_start_test_batch = False
                is_batch = True

                joints_ids = np.array([jnt_info.jnt_id for jnt_info in context.joints])
                joints_qposadr = model.jnt_qposadr[joints_ids]
                joints_dofadr = model.jnt_dofadr[joints_ids]
                joints_range_min = model.jnt_range[joints_ids, 0]
                joints_range_max = model.jnt_range[joints_ids, 1]
                joints_range_diff = np.abs(joints_range_max - joints_range_min)

                joints_qpos_start = data.qpos[joints_qposadr].copy()
                joints_dist_to_low = np.abs(joints_qpos_start - joints_range_min)
                joints_dist_to_high = np.abs(joints_qpos_start - joints_range_max)

                joints_sign = np.zeros(joints_ids.shape, dtype=np.float32)
                joints_sign[joints_dist_to_low < joints_dist_to_high] = 1.0
                joints_sign[joints_dist_to_low >= joints_dist_to_high] = -1.0

                joints_forces = context.force_magnitude * joints_sign

                data.qfrc_applied[joints_dofadr] = joints_forces
                test_running = True
                test_done = False
                print("Started test for joint a batch of joints")

            if not context.dirty_pause:
                t_start = data.time
                while data.time - t_start < 1.0 / 60.0:
                    mj.mj_step(model, data)
                    if test_running and data.time > N_SECS:
                        test_done = True
                        break

            if test_running and test_done:
                test_running = False
                context.dirty_pause = True

                if is_batch:
                    is_batch = False
                    joints_qpos_end = data.qpos[joints_qposadr].copy()
                    joints_open_percent = (
                        np.abs(joints_qpos_end - joints_qpos_start) / joints_range_diff
                    )
                    joints_open_success = joints_open_percent > OPEN_RANGE_PERCENT

                    data.qfrc_applied[joints_dofadr] = 0.0
                    print(f"open_success: {joints_open_success}")
                    print(f"open_percent: {joints_open_percent}")
                else:
                    if test_joint:
                        test_joint.jnt_qpos_end = data.qpos[test_joint.jnt_qposadr].item()
                        open_percent = (
                            abs(test_joint.jnt_qpos_end - test_joint.jnt_qpos_start)
                            / test_joint.jnt_range_diff
                        )
                        open_success = open_percent > OPEN_RANGE_PERCENT
                        print(
                            f"open_success: {open_success}, qpos_start: {test_joint.jnt_qpos_start}, qpos_end: {test_joint.jnt_qpos_end}, open_percent: {open_percent}"
                        )

                        data.qfrc_applied[test_joint.jnt_dofadr] = 0.0
                    else:
                        print("Something went wrong, haven't selected a joint to run the test")

            viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
