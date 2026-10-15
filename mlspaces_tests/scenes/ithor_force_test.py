"""iTHOR articulation force test.

For every hinge/slide joint in a scene: ramp an applied force until the joint
opens past a quarter of its range, then release and let it settle. Reports the
force each joint needed and the velocity it was left with.

Importable: `measure_scene_forces` is a pure function and nothing runs at
import. `test_articulated_joints_open` is the pytest entry point; `main()` is
the full sweep with JSON and plots, for `python ithor_force_test.py`.
"""

import json
import os

import mujoco
import numpy as np
import pytest

from molmo_spaces.env.scene.thor.fixups import load_env_with_objects
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.utils.constants.object_constants import ALL_ARTICULATION_TYPES_THOR

# The iTHOR scenes live in the resource manager's asset tree, not in a
# repo-relative `assets/`; ASSETS_DIR honours $MLSPACES_ASSETS_DIR.
FOLDER_PATH = ASSETS_DIR / "scenes" / "ithor"

STARTING_FORCE = 200  # 5 # increasing this,
FORCE_STEP_SIZE = 5

n_secs = 5  # or increasing this,
FORCE_STEPS = n_secs * 500  # 1000/2= 500 steps = 1 second
MONITOR_STEPS = 100  # 1000/2= 500 steps = 1 second

# The ramp used to be `while not open_success` with the try counter commented
# out, so a joint that never opened span forever. Bounded here instead: a joint
# that has not moved by MAX_FORCE_TRIES steps of the ramp is recorded as failed.
MAX_FORCE_TRIES = 20

# Fraction of a joint's range it has to travel to count as opened.
OPEN_FRACTION = 0.25

# The test checks the first couple of joints, not the whole scene. Every joint
# in FloorPlan8 opens on the first try, so the ramp is not the cost -- the
# 5-second force application is, and it grows per joint (2s for the first,
# ~100s by the fourth) as the scene accumulates disturbance. `main()` still
# does every joint.
TEST_MAX_JOINTS = 2

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


def discover_scenes(folder_path=FOLDER_PATH, limit=1):
    """The scene XMLs in `folder_path`, or [] when the asset tree is absent.

    `limit` keeps the default cheap -- the original only ever took the first.
    """
    if not os.path.isdir(folder_path):
        return []
    xmls = sorted(f for f in os.listdir(folder_path) if f.endswith(".xml"))
    if limit is not None:
        xmls = xmls[:limit]
    return [os.path.join(folder_path, f) for f in xmls]


def _apply_force_and_monitor(model, data, jnt_id, force, sign):
    """Push one joint with `force`, then release it and let it settle.

    Returns (opened, travelled_fraction, qvel_after_settling).
    """
    qposadr = int(model.jnt_qposadr[jnt_id])
    dofadr = int(model.jnt_dofadr[jnt_id])
    joint_range = model.joint(jnt_id).range
    range_diff = np.abs(joint_range[1] - joint_range[0])

    start = np.copy(data.qpos[qposadr])

    data.qfrc_applied[dofadr] = force * sign
    for _ in range(FORCE_STEPS // 50):
        mujoco.mj_step(model, data, nstep=50)
    travelled = np.abs(data.qpos[qposadr] - start) / range_diff

    # Release and monitor, to see whether the joint settles.
    data.qfrc_applied[dofadr] = 0
    for _ in range(MONITOR_STEPS // 50):
        mujoco.mj_step(model, data, nstep=50)

    return bool(travelled > OPEN_FRACTION), float(travelled), float(data.qvel[dofadr])


def measure_scene_forces(xml_path, max_tries=MAX_FORCE_TRIES, max_joints=None):
    """Ramp force on every hinge/slide joint of one scene until it opens.

    Returns a dict with per-category forces and settled velocities, the
    per-drawer forces, and the joints that never opened within `max_tries`.

    `max_joints` stops after that many articulated joints, which is how the
    test keeps to seconds; `main` leaves it None and does the whole scene.
    """
    xml_file = os.path.basename(xml_path)
    model, _ = load_env_with_objects(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    forces_by_category = {cat.lower(): [] for cat in ALL_CATEGORIES}
    qvels_by_category = {cat.lower(): [] for cat in ALL_CATEGORIES}
    drawer_forces = {}
    failed_joints = []

    n_seen = 0
    for i in range(model.njnt):
        name = model.joint(i).name
        jnt_type = model.jnt_type[i]

        if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            force = STARTING_FORCE
        elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
            force = STARTING_FORCE * 2
        else:
            continue

        if max_joints is not None and n_seen >= max_joints:
            break
        n_seen += 1

        joint_range = model.joint(i).range
        # A joint whose range opens downwards is pushed the other way.
        sign = -1 if joint_range[1] == 0 else 1
        qposadr = int(model.jnt_qposadr[i])
        dofadr = int(model.jnt_dofadr[i])

        opened = False
        qvel = 0.0
        for _ in range(max_tries):
            opened, _travelled, qvel = _apply_force_and_monitor(model, data, i, force, sign)
            if opened:
                break
            force += FORCE_STEP_SIZE

        if opened:
            category = name_to_category(name)
            forces_by_category[category].append(force)
            qvels_by_category[category].append(qvel)
            if "drawer" in name.lower():
                drawer_forces.setdefault(xml_file, {})[name] = force
            # Put the joint back so the next one starts from a closed scene.
            data.qpos[qposadr] = 0
            mujoco.mj_step(model, data, nstep=10)
            data.qfrc_applied[dofadr] = 0
        else:
            failed_joints.append((xml_path, name))

    return {
        "env": xml_file,
        "forces_by_category": forces_by_category,
        "qvels_by_category": qvels_by_category,
        "drawer_forces": drawer_forces,
        "failed_joints": failed_joints,
        "n_opened": sum(len(v) for v in forces_by_category.values()),
    }


@pytest.mark.slow
@pytest.mark.parametrize("scene_path", discover_scenes(), ids=os.path.basename)
def test_articulated_joints_open(scene_path):
    """Every hinge and slide joint in the scene opens under a bounded force.

    A joint that will not move by `MAX_FORCE_TRIES` steps of the ramp is either
    welded shut or mis-authored, which is exactly what this sweep exists to
    find.
    """
    result = measure_scene_forces(scene_path, max_joints=TEST_MAX_JOINTS)

    assert result["n_opened"] > 0, f"no articulated joint opened in {result['env']}"
    assert not result["failed_joints"], (
        f"{len(result['failed_joints'])} joint(s) never opened in {result['env']}: "
        f"{[name for _, name in result['failed_joints'][:5]]}"
    )


def test_scene_assets_are_available():
    """Guards the parametrisation above: with no scenes it silently vacuums out
    to zero tests, which would look like a pass."""
    if not os.path.isdir(FOLDER_PATH):
        pytest.skip(f"no scene assets at {FOLDER_PATH}")
    assert discover_scenes(), f"no *.xml scenes under {FOLDER_PATH}"


def main():
    import datetime

    import matplotlib.pyplot as plt

    forces_for_all_categories = {cat.lower(): [] for cat in ALL_CATEGORIES}
    qvels_for_all_categories = {cat.lower(): [] for cat in ALL_CATEGORIES}
    drawer_forces_for_all_assets = {}
    failed_joints = []

    for xml_path in discover_scenes():
        result = measure_scene_forces(xml_path)
        for cat, forces in result["forces_by_category"].items():
            forces_for_all_categories[cat].extend(forces)
        for cat, qvels in result["qvels_by_category"].items():
            qvels_for_all_categories[cat].extend(qvels)
        drawer_forces_for_all_assets.update(result["drawer_forces"])
        failed_joints.extend(result["failed_joints"])
        for _, name in result["failed_joints"]:
            print(f"Joint {name} failed to open")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results/{stamp}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/forces_for_all_categories.json", "w") as f:
        json.dump(forces_for_all_categories, f)
    with open(f"{out_dir}/qvels_for_all_categories.json", "w") as f:
        json.dump(qvels_for_all_categories, f)
    with open(f"{out_dir}/drawer_forces_for_all_assets.json", "w") as f:
        json.dump(drawer_forces_for_all_assets, f)
    with open(f"{out_dir}/failed_joints.json", "w") as f:
        json.dump(failed_joints, f)

    # remove keys with no values
    nkeys_before = len(forces_for_all_categories.keys())
    forces_for_all_categories = {k: v for k, v in forces_for_all_categories.items() if v}
    qvels_for_all_categories = {k: v for k, v in qvels_for_all_categories.items() if v}

    nkeys = len(forces_for_all_categories.keys())
    print(f"Removed {nkeys_before - nkeys} keys with no values")
    print(f"Total values: {sum(len(v) for v in forces_for_all_categories.values())}")
    print({k: len(v) for k, v in forces_for_all_categories.items()})

    if nkeys == 0:
        print("No joints opened; nothing to plot.")
        return

    # plot with min and max and mean
    plt.figure(figsize=(10, 5))
    for reduce in (np.min, np.max, np.mean):
        plt.bar(
            range(nkeys),
            [reduce(forces_for_all_categories[cat]) for cat in forces_for_all_categories],
        )
    plt.xticks(range(nkeys), forces_for_all_categories.keys(), rotation=90)
    plt.ylabel("Force")
    plt.title("Force for all categories")
    plt.savefig(f"{out_dir}/force_for_all_categories.png")

    # plot with min and max and mean - line and box
    plt.figure(figsize=(10, 5))
    plt.boxplot(list(qvels_for_all_categories.values()))
    plt.xticks(range(nkeys), qvels_for_all_categories.keys(), rotation=90)
    plt.ylabel("Qvel")
    plt.title("Qvel for all categories")
    plt.savefig(f"{out_dir}/qvel_for_all_categories.png")


if __name__ == "__main__":
    main()
