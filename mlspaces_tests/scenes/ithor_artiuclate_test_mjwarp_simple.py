import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import mujoco
import mujoco.mjx as mjx
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from molmo_spaces.editor.thor_model_editor import ThorMjModelEditor
from molmo_spaces.utils.scene_maps import iTHORMap

# Performance optimization: MJX batch processing
BATCH_SIZE = 16  # Smaller batch size for better compatibility
PHYSICS_STEPS_PER_FRAME = 25  # Reduced for faster simulation


def regenerate_map(ithor_scene_path, thread_id=None, use_gpu=False) -> None:
    device_id = None

    if use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                if thread_id is not None:
                    device_id = thread_id % 4 if thread_id < 4 else None
            else:
                device_id = None
                print("Warning: GPU requested but not available, using CPU rendering")
        except ImportError:
            device_id = None
            print("Warning: GPU requested but PyTorch not available, using CPU rendering")
    else:
        device_id = None

    thormap = iTHORMap.from_mj_model_path(
        ithor_scene_path, px_per_m=200, agent_radius=0.20, device_id=device_id
    )
    if "mesh" in ithor_scene_path:
        thormap.save(ithor_scene_path.replace("_mesh.xml", "_map.png"))
    else:
        thormap.save(ithor_scene_path.replace(".xml", "_map.png"))


def get_all_handles(model, data):
    all_handles = []
    for i in range(model.nbody):
        if "handle" in model.body(i).name.lower():
            parent_id = int(model.body_parentid[i])
            parent_name = model.body(parent_id).name
            if parent_id == 0:
                continue
            root_body = int(model.body_rootid[i])
            model.body(root_body).name

            geom_id = int(model.body_geomadr[i])
            geom_num = int(model.body_geomnum[i])
            for j in range(geom_num):
                geom_group = int(model.geom(geom_id + j).group)
                if geom_group == 0:
                    geom_id = geom_id + j
                    break
            body_id = int(model.body_bvhadr[i])
            handle_bounding_box = model.bvh_aabb[body_id]
            handle_bounding_box[:3]

            all_handles.append(
                {
                    "name": parent_name,
                    "position": data.geom_xpos[geom_id],
                    "orientation": data.geom_xmat[geom_id],
                    "handle_id": parent_id,
                    "geom_id": geom_id,
                    "size": handle_bounding_box[3:],
                }
            )
    return all_handles


def get_gripper_pose_based_on_handle_pose(handle_pose, ithor_map):
    handle_position = handle_pose["position"]
    handle_orientation = handle_pose["orientation"].reshape(3, 3)
    handle_pose["size"]

    free_points = ithor_map.get_free_points()
    closest_point = min(free_points, key=lambda x: np.linalg.norm(x[:2] - handle_position[:2]))
    free_position = closest_point
    free_position[2] = handle_position[2]

    to_handle_dist = handle_position - free_position
    to_handle_dir = to_handle_dist / np.linalg.norm(to_handle_dist)

    z_axis = to_handle_dir
    handle_x_axis_global = handle_orientation @ np.array([0, 0, -1])
    up_reference = handle_x_axis_global
    if up_reference[2] > 0:
        up_reference *= -1

    x_axis = np.cross(up_reference, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up_reference = np.array([1, 0, 0])
        x_axis = np.cross(up_reference, z_axis)

    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quat = R.from_matrix(rot_matrix).as_quat(scalar_first=True)

    gripper_pose = {"position": free_position, "quaternion": quat}
    return gripper_pose


def add_gripper_to_scene(scene_path, gripper_pose, gripper_path="assets/rum_gripper/model.xml"):
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


def step_path(to_handle_dist, current_pos, current_quat, step_size, gripper_length=0.1):
    path = {"mocap_pos": [], "mocap_quat": []}
    path["mocap_pos"].append(current_pos)
    path["mocap_quat"].append(current_quat)

    dist = np.linalg.norm(to_handle_dist)
    dist -= gripper_length

    for _i in range(int(dist / step_size)):
        angle = np.arctan2(to_handle_dist[1], to_handle_dist[0])
        next_pos = current_pos + step_size * np.array([np.cos(angle), np.sin(angle), 0])
        path["mocap_pos"].append(next_pos)
        path["mocap_quat"].append(current_quat)
        dist -= np.linalg.norm(next_pos - current_pos)
        current_pos = next_pos
    return path


def process_handles_with_mjx_simple(handles, ithor_scene_path, ithor_map, device_id=0):
    """Simplified MJX processing that should work with MuJoCo 3.3.2"""

    # Load model and convert to MJX
    model = mujoco.MjModel.from_xml_path(ithor_scene_path)

    # Use CPU device for compatibility
    device = jax.devices()[0]  # Use first available device (usually CPU)
    mjx_model = mjx.put_model(model, device=device)

    results = []

    # Process handles one by one for better compatibility
    for handle in handles:
        handle_name = handle["name"]
        handle_id = handle["handle_id"]

        # Get gripper pose
        gripper_pose = get_gripper_pose_based_on_handle_pose(handle, ithor_map)

        # Create MJX data
        data = mujoco.MjData(model)

        # Set initial state
        if data.mocap_pos.size > 0:
            data.mocap_pos[0] = gripper_pose["position"]
            data.mocap_quat[0] = gripper_pose["quaternion"]

        # Convert to MJX
        mjx_data = mjx.put_data(mjx_model, data, device=device)

        # Run simulation steps
        for _step in range(PHYSICS_STEPS_PER_FRAME):
            mjx_data = mjx.step(mjx_model, mjx_data)

        # Get handle joint position
        handle_joint_id = model.body(handle_id).jntadr[0]
        if handle_joint_id != -1:
            handle_qpos_idx = model.joint(handle_joint_id).qposadr[0]

            # Get initial position
            data_init = mujoco.MjData(model)
            mujoco.mj_step(model, data_init)
            start_handle_joint_pos = data_init.qpos[handle_qpos_idx]

            # Get final position from MJX
            end_handle_joint_pos = mjx_data.qpos[handle_qpos_idx]

            success = int(np.abs(end_handle_joint_pos - start_handle_joint_pos) > np.deg2rad(5))

            results.append(
                {
                    "handle_name": handle_name,
                    "start_handle_joint_pos": float(start_handle_joint_pos),
                    "end_handle_joint_pos": float(end_handle_joint_pos),
                    "success": success,
                }
            )

    return results


def run_one_floorplan_mjx_simple(i, mesh, thread_id=None, use_gpu=False) -> None:
    if mesh:
        ithor_scene_path = f"debug/good_iTHOR/FloorPlan{i}_physics_mesh.xml"
        ithor_map_path = f"{ithor_scene_path.replace('_mesh.xml', '_map.png')}"
    else:
        ithor_scene_path = f"debug/good_iTHOR/FloorPlan{i}_physics.xml"
        ithor_map_path = f"{ithor_scene_path.replace('.xml', '_map.png')}"

    if not os.path.exists(ithor_scene_path):
        print(f"Scene path {ithor_scene_path} does not exist")
        return

    print(f"Processing FloorPlan {i} with MJX Simple (thread {thread_id})...")

    regenerate_map(ithor_scene_path, thread_id, use_gpu)
    ithor_map = iTHORMap.load(ithor_map_path)

    # Initialize scene
    model = mujoco.MjModel.from_xml_path(ithor_scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    all_handles = get_all_handles(model, data)
    print(f"Found {len(all_handles)} handles")

    # Add gripper to scene once
    if len(all_handles) > 0:
        first_handle = all_handles[0]
        gripper_pose = get_gripper_pose_based_on_handle_pose(first_handle, ithor_map)
        gripper_path = add_gripper_to_scene(ithor_scene_path, gripper_pose)
        print("Added gripper to scene")

    # Process handles in batches using simplified MJX
    success_metric = {}
    n_success = 0
    n_total = 0
    failed_handles = []

    # Split handles into batches
    handle_batches = [
        all_handles[j : j + BATCH_SIZE] for j in range(0, len(all_handles), BATCH_SIZE)
    ]

    device_id = thread_id % len(jax.devices()) if thread_id is not None else 0

    for batch_idx, handle_batch in enumerate(handle_batches):
        print(
            f"Processing MJX Simple batch {batch_idx + 1}/{len(handle_batches)} ({len(handle_batch)} handles)"
        )

        # Process batch with simplified MJX
        batch_results = process_handles_with_mjx_simple(
            handle_batch, gripper_path, ithor_map, device_id
        )

        # Process results
        for result in batch_results:
            handle_name = result["handle_name"]
            success = result["success"]

            success_metric[handle_name] = {
                "start_handle_joint_pos": result["start_handle_joint_pos"],
                "end_handle_joint_pos": result["end_handle_joint_pos"],
                "success": success,
            }
            n_success += success
            n_total += 1
            if not success:
                failed_handles.append(handle_name)

    # Save results
    success_metric["success_rate"] = n_success / n_total if n_total > 0 else 0
    success_metric["failed_handles"] = failed_handles
    print(success_metric)
    os.makedirs(f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}", exist_ok=True)
    with open(
        f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}/success_metric_mjx_simple.json", "w"
    ) as f:
        json.dump(success_metric, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i", type=int, required=True, help="Floorplan index. Use -1 to process all floorplans"
    )
    parser.add_argument(
        "--mesh", action="store_true", help="Use mesh files instead of primitive files"
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=4,
        help="Number of threads for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Enable GPU rendering if available (default: CPU-only)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="MJX batch size (default: 16)")
    parser.add_argument(
        "--physics-steps", type=int, default=25, help="Physics steps per frame (default: 25)"
    )
    args = parser.parse_args()

    # Update global constants
    global BATCH_SIZE, PHYSICS_STEPS_PER_FRAME
    BATCH_SIZE = args.batch_size
    PHYSICS_STEPS_PER_FRAME = args.physics_steps

    # Set JAX environment for GPU
    if args.gpu:
        os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
        print(f"Using {len(jax.devices())} devices: {jax.devices()}")

    i = args.i
    mesh = args.mesh
    nthread = args.nthread
    use_gpu = args.gpu

    if i == -1:
        floorplan_indices = list(range(13, 0, -1))

        print(
            f"Processing {len(floorplan_indices)} floorplans using MJX Simple with {nthread} threads..."
        )
        print(f"Floorplan range: {floorplan_indices[0]} to {floorplan_indices[-1]}")
        print(f"MJX batch size: {BATCH_SIZE}, Physics steps per frame: {PHYSICS_STEPS_PER_FRAME}")

        completed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=nthread) as executor:
            future_to_floorplan = {
                executor.submit(
                    run_one_floorplan_mjx_simple, floorplan_i, mesh, thread_id, use_gpu
                ): floorplan_i
                for thread_id, floorplan_i in enumerate(floorplan_indices)
            }

            with tqdm(
                total=len(floorplan_indices), desc="Processing floorplans with MJX Simple"
            ) as pbar:
                for future in as_completed(future_to_floorplan):
                    floorplan_i = future_to_floorplan[future]
                    try:
                        future.result()
                        completed_count += 1
                        print(
                            f"Completed FloorPlan {floorplan_i} ({completed_count}/{len(floorplan_indices)})"
                        )
                    except Exception as exc:
                        failed_count += 1
                        print(f"FloorPlan {floorplan_i} generated an exception: {exc}")
                        print(f"Failed count: {failed_count}")
                    pbar.update(1)

        print(
            f"All floorplans processed with MJX Simple! Completed: {completed_count}, Failed: {failed_count}"
        )
        return
    else:
        run_one_floorplan_mjx_simple(i, mesh, use_gpu=use_gpu)


if __name__ == "__main__":
    main()
