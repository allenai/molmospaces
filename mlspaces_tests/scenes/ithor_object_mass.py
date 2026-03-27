import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm

from molmo_spaces.editor.constants import ALL_PICKUP_TYPES_THOR

# from molmo_spaces.tasks.util_samplers.grasp_sampler import TopDownGraspPoseSampler
from tests.ithor_object_mass_est import ithor_estimated_masses

# MJT_ASSET_DIR = Path("/Users/maxa/local_datasets/mujoco-thor-assets")
MJT_ASSET_DIR = Path("/Users/yejink/Repos/mujoco-thor/assets/objects/thor")


LARGE_MASS_THRESHOLD = 0.5  # 3.75


def get_moveable_objects(model):
    free_body_ids = []
    free_body_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            # Get the body attached to this joint
            body_id = model.jnt_bodyid[j]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            model.body_mass[body_id]  # Access mass directly by index
            free_body_names.append(body_name)
            free_body_ids.append(body_id)
            # print(body_name, mass)
    return free_body_names, free_body_ids


def plot_scatter(all_object_weights, ithor_estimated_masses) -> None:
    names = []
    mean_weights = []
    estimated = []

    for name, weights in all_object_weights.items():
        # remove "_" in name
        category_name = name.replace("_", "")
        if category_name not in ALL_PICKUP_TYPES_THOR:
            continue
        names.append(name)
        mean_weights.append(np.mean(weights))
        estimated.append(ithor_estimated_masses.get(name, np.nan))  # fallback to NaN if missing
    x = np.array(mean_weights)
    y = np.array(estimated)

    # remove nan index from y
    nan_indices = np.where(np.isnan(y))[0]
    x = np.delete(x, nan_indices)
    y = np.delete(y, nan_indices)
    names = np.delete(names, nan_indices)

    # compute R2 score
    r2 = 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)
    print(f"R2 score: {r2}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, c="teal", edgecolor="k", alpha=0.7)
    lims = np.array([min(np.min(x), np.nanmin(y)), max(np.max(x), np.nanmax(y))])
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for i, nm in enumerate(names):
        plt.annotate(nm, xy=(x[i], y[i]), xytext=(5, 3), textcoords="offset points", fontsize=9)
    plt.xlabel("Simulation Mean Mass (kg)", fontsize=12)
    plt.ylabel("LLM Estimated Mass (kg)", fontsize=12)
    plt.title(f"Simulation vs Estimated Object Mass (R2: {r2:.2f})", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"simulation_vs_estimated_object_mass_r2_{r2:.2f}_graspable.png")


def add_gripper_to_scene(
    scene_path, gripper_pose, gripper_path="assets/robots/rum_gripper/model.xml"
):
    # gripper_path = "assets/franka_fr3/fr3_gripper.xml"

    editor = ThorMjModelEditor.from_xml_path(scene_path)
    editor.set_options()
    editor.set_size(size=5000)
    editor.set_compiler()
    editor.add_robot(
        xml_path=gripper_path, pos=gripper_pose["position"], quat=gripper_pose["quaternion"]
    )
    editor.add_mocap_body(
        name="target_ee_pose",
        gripper_weld=True,
        gripper_name="robot_0/",
        pos=gripper_pose["position"],
        quat=gripper_pose["quaternion"],
    )
    robot_xml_path = scene_path.replace(".xml", "_gripper.xml")
    editor.save_xml(save_path=robot_xml_path)
    return robot_xml_path


def process_scene(xml_path, all_objects_weights, large_mass_objects):
    # model, root_bodies_dict = load_env_with_objects(xml_path)
    robot_xml_path = MJT_ASSET_DIR / "robots/franka_fr3/fr3_gripper.xml"

    mode = None  # "add_robot_mjspec"
    if mode == "add_robot_thor":
        new_path = add_gripper_to_scene(
            xml_path,
            gripper_pose={"position": [0, 0, 0.5], "quaternion": [1, 0, 0, 0]},
            gripper_path=robot_xml_path,
        )

        print("Saved gripper scene to:", new_path)
        model = mujoco.MjModel.from_xml_path(new_path)

    if mode == "add_robot_mjspec":
        robot_pos = [0, 0, 0.5]
        print("XXX", robot_xml_path, robot_xml_path.is_file())
        spec = mujoco.MjSpec.from_file(xml_path)
        parent_body = spec.worldbody.add_body(name="parent_body")
        parent_site = parent_body.add_site(name="attachment_site", pos=robot_pos)
        robot_spec = mujoco.MjSpec.from_file(str(robot_xml_path))
        print("robot_spec wb name", robot_spec.worldbody.name)
        print("Direct child bodies:", [b.name for b in robot_spec.bodies])
        spec.attach(robot_spec, site=parent_site, prefix="attached_", suffix="_child")

        mocap_name = "target_ee_pose"
        body = spec.worldbody.add_body(name=mocap_name)
        body.pos = robot_pos  # expect list of floats
        body.quat = [1, 0, 0, 0]  # expect list of floats
        body.mocap = True

        # Add a site to the new body
        body.add_site(
            name=mocap_name + "_site",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015, 0.015, 0.015],
            rgba=[1.0, 0.0, 0.0, 0.3],
            group=2,
        )
        gripper_weld = False  # Set to True if you want to weld the gripper to the mocap bodys
        if gripper_weld:
            spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1=mocap_name + "_site",
                name2="hand",
                solref=[1e-6, 1],
                solimp=[0.9, 0.95, 0.001, 0.00001, 1],
            )

        model = spec.compile()

    else:
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
        except ValueError:
            return

    data = mujoco.MjData(model)

    start_viewer = False  # True

    viewer = None
    if start_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        # while viewer.is_running():
        #    mujoco.mj_step(model, data)
        viewer.sync()  # Synchronize viewer with simulation state

    mujoco.mj_step(model, data)

    # get body tree
    children = {}
    for bid in range(model.nbody):
        pid = model.body_parentid[bid]  # ID of parent
        children.setdefault(pid, []).append(bid)

    def get_children(body_id):
        return children.get(body_id, [])

    def get_subtree(body_id):
        subtree = []
        stack = [body_id]
        while stack:
            bid = stack.pop()
            subtree.append(bid)
            stack.extend(get_children(bid))
        return subtree

    _, free_body_ids = get_moveable_objects(model)

    def clean_name(name):
        return "".join(ch for ch in name if not ch.isdigit()).strip("_")

    def clean_name2(name):
        return name.split("__")[0]

    for body_id in free_body_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        [model.body_mass[bid] for bid in get_children(body_id)]
        total_mass = model.body_subtreemass[body_id]  # np.sum(masses)
        if total_mass == 0:
            print("Warning 0 mass objects", name)

        v_clean_name = clean_name2(clean_name(name))

        if total_mass > LARGE_MASS_THRESHOLD:
            print("Warning large mass objects", name, total_mass)
            large_mass_objects.add(v_clean_name)

        all_object_weights[v_clean_name].append(total_mass)

    return model, data, viewer


if __name__ == "__main__":
    # folder_path = "/Users/maxa/assets/good_iTHOR/"
    folder_path = "/Users/yejink/Repos/mujoco-thor/assets/scenes/ithor_082725/"
    # sorted_xml_files = ["FloorPlan1_physics.xml",]
    all_xml_files = [f for f in os.listdir(folder_path) if f.endswith("_mesh.xml")]
    sorted_xml_files = sorted(all_xml_files)
    # sorted_xml_files = sorted_xml_files[:1]
    all_object_weights = defaultdict(list)
    large_mass_objects = set()

    for xml_file in tqdm(sorted_xml_files):
        xml_path = os.path.join(folder_path, xml_file)
        print(xml_path)
        model, data, viewer = process_scene(xml_path, all_object_weights, large_mass_objects)

        # for i in range(10_000):
        #    mujoco.mj_step(model, data)
        #    viewer.sync() # Synchronize viewer with simulation state

    print(large_mass_objects)

    # for name, weights in all_object_weights.items():
    #    print(name, len(weights), np.mean(weights), np.min(weights), np.max(weights))

    plot_scatter(all_object_weights, ithor_estimated_masses)

    # set_trace()
