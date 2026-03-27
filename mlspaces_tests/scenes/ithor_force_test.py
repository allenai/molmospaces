"""
iTHOR force test

1. applies force to joint of the asset in the scene for 1 seconds, then check the joint position
2. monitors for 1 second to see if the jointed asset settles
"""

import json
import os

import mujoco
import numpy as np

from molmo_spaces.editor.constants import ALL_ARTICULATION_TYPES_THOR
from molmo_spaces.env.arena.arena_utils import load_env_with_objects

folder_path = "assets/scenes/ithor_081125"
all_xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
sorted_xml_files = sorted(all_xml_files)[:1]

STARTING_FORCE = 200  # 5 # increasing this,
FORCE_STEP_SIZE = 5

n_secs = 5  # or increasing this,
FORCE_STEPS = n_secs * 500  # 1000/2= 500 steps = 1 second
MONITOR_STEPS = 100  # 1000/2= 500 steps = 1 second


ALL_CATEGORIES = ALL_ARTICULATION_TYPES_THOR + [
    "cabinet",
    "drawer",
    "oven",
    "dishwasher",
    "showerdoor",
    "other",
]


def name_to_category(name):
    for category in ALL_CATEGORIES:
        if category.lower() in name.lower():
            return category.lower()
    return "other"


forces_for_all_categories = {}
qvels_for_all_categories = {}

drawer_forces_for_all_assets = {}


for cat in ALL_CATEGORIES:
    forces_for_all_categories[cat.lower()] = []
    qvels_for_all_categories[cat.lower()] = []


failed_joints = []
for xml_file in sorted_xml_files:
    xml_path = os.path.join(folder_path, xml_file)
    model, root_bodies_dict = load_env_with_objects(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    viewer = None  #
    # viewer = mujoco.viewer.launch_passive(model, data)

    # while viewer.is_running():
    #    viewer.sync()

    # gather all joints
    for i in range(model.njnt):
        name = model.joint(i).name
        # if "cabinet" not in name.lower():
        #    continue
        # print(name)

        category = name_to_category(name)

        do_apply = False
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            do_apply = True
            force_to_apply = STARTING_FORCE

        elif model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE:
            do_apply = True
            force_to_apply = STARTING_FORCE * 2
        else:
            pass

        if do_apply:
            qposadr = model.joint(i).qposadr[0]
            dofadr = model.joint(i).dofadr[0]
            joint_range = model.joint(i).range
            range_diff = np.abs(joint_range[1] - joint_range[0])
            sign = 1
            if joint_range[1] == 0:
                sign = -1

            starting_joint_position = data.qpos[qposadr]

            # print(f"Joint {name}:")
            # print(f"  Type: {model.jnt_type[i]}")
            # print(f"  Range: {joint_range}")
            # print(f"  Stiffness: {model.jnt_stiffness[i]}")
            # print(f"  Damping: {model.dof_damping[model.joint(i).dofadr[0]]}")
            # print(f"  Frictionloss: {model.dof_frictionloss[model.joint(i).dofadr[0]]}")
            # print(f"  Armature: {model.dof_armature[model.joint(i).dofadr[0]]}")

            def apply_force_and_monitor(dofadr, force_to_apply, sign):
                # apply force to joint
                data.qfrc_applied[dofadr] = force_to_apply * sign

                # should i apply all at once or one by one? one at a time bc some might collide
                for _ in range(FORCE_STEPS // 50):
                    mujoco.mj_step(model, data, nstep=50)  # 1000/2= 500 steps = 1 second
                    if viewer is not None:
                        viewer.sync()
                    after_force_joint_position = data.qpos[qposadr]
                    # print(name, "\t", after_force_joint_position, "\t", starting_joint_position)
                open_success = (
                    np.abs(after_force_joint_position - starting_joint_position) / range_diff
                ) > 0.25

                # mointor for 1 second to see if the jointed asset settles
                data.qfrc_applied[dofadr] = 0
                for _ in range(MONITOR_STEPS // 50):
                    mujoco.mj_step(model, data, nstep=50)  # 1000/2= 500 steps = 1 second
                    if viewer is not None:
                        viewer.sync()
                dofadr = model.joint(i).dofadr[0]
                after_force_qvel = data.qvel[dofadr]
                # settle_success = np.isclose(after_force_qvel, 0, atol=1e-3)

                if not open_success:
                    print(name, "\t", starting_joint_position, "\t", after_force_joint_position)
                return open_success, after_force_qvel

            open_success = False
            n_tries = 0
            force_to_apply = STARTING_FORCE
            after_force_qvel = 0
            while not open_success:
                open_success, after_force_qvel = apply_force_and_monitor(
                    dofadr, force_to_apply, sign
                )
                force_to_apply += FORCE_STEP_SIZE
                # n_tries += 1
                # if n_tries > 10:
                #    break

            if open_success:
                forces_for_all_categories[category].append(force_to_apply)
                qvels_for_all_categories[category].append(after_force_qvel)

                if "drawer" in name.lower():
                    if xml_file not in drawer_forces_for_all_assets:
                        drawer_forces_for_all_assets[xml_file] = {}
                    drawer_forces_for_all_assets[xml_file][name] = force_to_apply

                # Close the asset
                # data.qfrc_applied[dofadr] = -force_to_apply * sign
                # for _ in range((n_secs*n_tries*FORCE_STEPS)//50):
                #    mujoco.mj_step(model, data, nstep=50) # 1000/2= 500 steps = 1 second
                #    if viewer is not None:
                #        viewer.sync()
                data.qpos[qposadr] = 0
                mujoco.mj_step(model, data, nstep=10)
                data.qfrc_applied[dofadr] = 0

            else:
                failed_joints.append((xml_path, model.joint(i).name))
                print(f"Joint {model.joint(i).name} failed to open {open_success}")

# save the json
import datetime

datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(f"results/{datetime}", exist_ok=True)

with open(f"results/{datetime}/forces_for_all_categories.json", "w") as f:
    json.dump(forces_for_all_categories, f)

with open(f"results/{datetime}/qvels_for_all_categories.json", "w") as f:
    json.dump(qvels_for_all_categories, f)

with open(f"results/{datetime}/drawer_forces_for_all_assets.json", "w") as f:
    json.dump(drawer_forces_for_all_assets, f)
with open(f"results/{datetime}/failed_joints.json", "w") as f:
    json.dump(failed_joints, f)


# remove keys with no values
nkeys_before = len(forces_for_all_categories.keys())
forces_for_all_categories = {k: v for k, v in forces_for_all_categories.items() if v}
qvels_for_all_categories = {k: v for k, v in qvels_for_all_categories.items() if v}

nkeys = len(forces_for_all_categories.keys())
removed_keys = nkeys_before - nkeys
print(f"Removed {removed_keys} keys with no values")

# count number of values
nvalues = sum([len(v) for v in forces_for_all_categories.values()])
print(f"Total values: {nvalues}")

# count number of values per key
nvalues_per_key = {k: len(v) for k, v in forces_for_all_categories.items()}
print(nvalues_per_key)

# plot with min and max and mean
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(
    range(nkeys),
    [np.min(forces_for_all_categories[cat]) for cat in forces_for_all_categories],
)
plt.bar(
    range(nkeys),
    [np.max(forces_for_all_categories[cat]) for cat in forces_for_all_categories],
)
plt.bar(
    range(nkeys),
    [np.mean(forces_for_all_categories[cat]) for cat in forces_for_all_categories],
)
plt.xticks(range(nkeys), forces_for_all_categories.keys(), rotation=90)
plt.ylabel("Force")
plt.title("Force for all categories")
# plt.show()
plt.savefig(f"results/{datetime}/force_for_all_categories.png")

# plot with min and max and mean - line and box
plt.figure(figsize=(10, 5))
plt.boxplot(qvels_for_all_categories.values())
plt.xticks(range(nkeys), qvels_for_all_categories.keys(), rotation=90)
plt.ylabel("Qvel")
plt.title("Qvel for all categories")
# plt.show()
plt.savefig(f"results/{datetime}/qvel_for_all_categories.png")
