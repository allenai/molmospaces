"""
Test the egg grasping and picking.

For 8 Scenes - Try all 5000 grasps see what percentage actually grasps
"""

import os
import re
import sys
import json
import datetime

import mujoco
import numpy as np
from ithor_grasp_test import GraspTestEnvironment, add_robot_to_scene, load_grasps_for_object
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

# Set multiprocessing start method for Linux compatibility
if sys.platform.startswith("linux"):
    mp.set_start_method("spawn", force=True)


def find_object_base_name_in_scene(model, object_type="egg") -> tuple[int, str]:
    """
    Find the egg in the scene.
    """
    object_pattern = re.compile(r"^([A-Za-z]+_\d+)(_.*)?$")
    for i in range(model.nbody):
        if object_type in model.body(i).name.lower():
            root_body_id = model.body(i).rootid[0]
            root_body_name = model.body(root_body_id).name
            match = object_pattern.match(root_body_name)
            if match:
                base_name = match.group(1)
                return root_body_id, base_name
    raise ValueError(f"Object {object_type} not found in scene")


def test_grasp_batch(args: Tuple[str, str, List[int], int, str, bool, str, bool]) -> Dict[str, int]:
    """
    Test a batch of grasps in a worker process.

    Args:
        args: Tuple containing:
            - gripper_scene_path: Path to the scene with robot
            - object_name: Name of the object
            - grasp_indices: List of grasp indices to test
            - object_root_id: Root body ID of the object
            - object_type: Type of object to test
            - use_pregrasp: Whether to use pregrasp
            - grasp_dir: Directory to load grasps from
            - save_video: Whether to save videos

    Returns:
        Dictionary with grasp statistics
    """
    (
        gripper_scene_path,
        object_name,
        grasp_indices,
        object_root_id,
        object_type,
        use_pregrasp,
        grasp_dir,
        save_video,
    ) = args

    n_colliding = 0
    n_non_colliding = 0
    n_pickable = 0
    n_failed = 0

    # Create the environment for this worker
    env = GraspTestEnvironment(gripper_scene_path, grasps_per_object=len(grasp_indices))
    env.load_scene()
    if save_video:
        env.setup_recorders()

    # Load only the grasps this worker needs to test
    min_idx = min(grasp_indices) if grasp_indices else 0
    max_idx = max(grasp_indices) if grasp_indices else 0

    # Load the subset of grasps we need (from 0 to max_idx)
    all_grasps = load_grasps_for_object(object_name, num_grasps=max_idx + 1, grasp_dir=grasp_dir)

    # Extract only the grasps for this worker's batch
    batch_grasps = []
    for i in grasp_indices:
        if i < len(all_grasps):
            batch_grasps.append(all_grasps[i])
        else:
            print(f"Warning: Index {i} out of bounds for {len(all_grasps)} grasps")

    # Store in environment
    env.object_grasps[object_name] = batch_grasps
    env.object_body_map[object_name] = object_root_id

    print(
        f"Worker testing {len(grasp_indices)} grasps (global indices {min_idx}-{max_idx}), loaded {len(batch_grasps)} grasps"
    )

    # Test each grasp in the batch - use local indices now
    for local_idx in range(len(batch_grasps)):
        mujoco.mj_resetData(env.model, env.data)
        mujoco.mj_step(env.model, env.data, nstep=100)

        grasp_pose = env.sample_grasp(object_name, index=local_idx)

        grasp_pose = env.tcp_to_base_grasp_pose(grasp_pose, gripper_length=0.155)
        if grasp_pose:
            if env.place_gripper_at_grasp(object_name, grasp_pose, pregrasp=use_pregrasp):
                n_non_colliding += 1
                if env.pick_object(object_name, pregrasp=use_pregrasp):
                    n_pickable += 1
                else:
                    n_failed += 1
                    # Save video only if requested
                if save_video:
                    for recorder in env.recorders:
                        recorder.save(os.path.join(env.save_dir, f"{object_name}"), save_video=True)

            else:
                n_colliding += 1
        else:
            global_idx = grasp_indices[local_idx]
            print(f"Could not sample grasp for {object_name} at global index {global_idx}")
            break

    return {
        "n_non_colliding": n_non_colliding,
        "n_colliding": n_colliding,
        "n_pickable": n_pickable,
        "n_failed": n_failed,
    }


def run_scene(
    gripper_scene_path,
    num_grasps,
    object_type="egg",
    grasp_dir=f"{ASSETS_DIR}/grasps/droid",
    use_passive_viewer=False,
    use_pregrasp=True,
    num_workers=None,
    save_video=False,
):
    gripper_scene_path = gripper_scene_path.replace("_robot.xml", "_robotiq.xml")
    if not os.path.exists(gripper_scene_path):
        add_robot_to_scene(
            gripper_scene_path.replace("_robotiq.xml", ".xml"),
            gripper_scene_path,
            robot_path="assets/robots/franka_droid/robotiq_2f85_v4/2f85.xml",
        )
        print(f"Robot added to scene: {gripper_scene_path}")

    # Load scene once to find object info (no multiprocessing for this part)
    temp_env = GraspTestEnvironment(gripper_scene_path, grasps_per_object=1)
    temp_env.load_scene()

    # Find the object info
    object_root_id, object_name = find_object_base_name_in_scene(temp_env.model, object_type)
    grasps = load_grasps_for_object(object_name, num_grasps=num_grasps, grasp_dir=grasp_dir)

    print(f"Object: {object_name} (ID: {object_root_id})")
    print(f"Total grasps: {len(grasps)}")

    n_grasps = len(grasps)

    # If viewer is requested or only few grasps, run sequentially
    if use_passive_viewer or n_grasps < 10:
        return run_scene_sequential(
            gripper_scene_path,
            num_grasps,
            object_type,
            grasp_dir,
            use_passive_viewer,
            use_pregrasp,
            save_video,
        )

    # Use multiprocessing for faster testing
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    print(f"Using {num_workers} worker processes...")

    # Split grasps into batches
    grasp_indices = list(range(n_grasps))
    batch_size = max(1, n_grasps // num_workers)
    batches = [grasp_indices[i : i + batch_size] for i in range(0, n_grasps, batch_size)]

    print(f"Split into {len(batches)} batches")

    # Prepare arguments for each batch
    batch_args = []
    for batch in batches:
        batch_args.append(
            (
                gripper_scene_path,
                object_name,
                batch,
                object_root_id,
                object_type,
                use_pregrasp,
                grasp_dir,
                save_video,
            )
        )

    # Run batches in parallel
    n_colliding = 0
    n_non_colliding = 0
    n_pickable = 0
    n_failed = 0

    with Pool(processes=num_workers) as pool:
        results = pool.map(test_grasp_batch, batch_args)

    # Aggregate results
    for result in results:
        n_non_colliding += result["n_non_colliding"]
        n_colliding += result["n_colliding"]
        n_pickable += result["n_pickable"]
        n_failed += result["n_failed"]

    print(f"n_non_colliding: {n_non_colliding}")
    print(f"n_colliding: {n_colliding}")
    print(f"n_pickable: {n_pickable}")
    print(f"n_failed: {n_failed}")

    return {
        "n_non_colliding": n_non_colliding,
        "n_colliding": n_colliding,
        "n_pickable": n_pickable,
        "n_failed": n_failed,
    }


def run_scene_sequential(
    gripper_scene_path,
    num_grasps,
    object_type="egg",
    grasp_dir=f"{ASSETS_DIR}/grasps/droid",
    use_passive_viewer=False,
    use_pregrasp=True,
    save_video=False,
):
    """Run scene test sequentially (for viewer or small tests)."""
    # Create the environment
    env = GraspTestEnvironment(gripper_scene_path, grasps_per_object=num_grasps)
    env.load_scene()
    if save_video:
        env.setup_recorders()

    # Load the specific object grasps
    # object_root_id, object_name = find_object_base_name_in_scene(env.model, "egg")
    # object_root_id, object_name = find_object_base_name_in_scene(env.model, "mug")
    # object_root_id, object_name = find_object_base_name_in_scene(env.model, "pan")
    # object_root_id, object_name = find_object_base_name_in_scene(env.model, "fork")
    # object_root_id, object_name = find_object_base_name_in_scene(env.model, "knife")
    object_root_id, object_name = find_object_base_name_in_scene(env.model, "fork")

    grasps = load_grasps_for_object(object_name, num_grasps=num_grasps, grasp_dir=grasp_dir)
    print(object_root_id, object_name)
    print(len(grasps))
    env.object_grasps[object_name] = grasps
    env.object_body_map[object_name] = object_root_id

    n_grasps = len(grasps)

    n_colliding = 0
    n_non_colliding = 0
    n_pickable = 0
    n_failed = 0

    if use_passive_viewer:
        env.start_viewer()

    for i in range(n_grasps):
        mujoco.mj_resetData(env.model, env.data)
        mujoco.mj_step(env.model, env.data, nstep=100)

        grasp_pose = env.sample_grasp(object_name, index=i)

        grasp_pose = env.tcp_to_base_grasp_pose(grasp_pose, gripper_length=0.155)
        if grasp_pose:
            if env.place_gripper_at_grasp(object_name, grasp_pose, pregrasp=False):
                n_non_colliding += 1
                if env.pick_object(object_name, pregrasp=False):
                    n_pickable += 1
                    # Save video only if requested
                    if save_video:
                        for recorder in env.recorders:
                            recorder.save(
                                os.path.join(env.save_dir, f"{object_name}"), save_video=True
                            )
                else:
                    n_failed += 1

            else:
                n_colliding += 1
        else:
            print(f"Could not sample grasp for {object_name} at index {i}")
            break

        if i % 10 == 0:  # Print progress every 10 grasps
            print(
                f"Progress: {i}/{n_grasps} | Non-colliding: {n_non_colliding} | Colliding: {n_colliding} | Pickable: {n_pickable} | Failed: {n_failed}"
            )

    if use_passive_viewer:
        env.stop_viewer()

    print(f"Final results:  {object_name}")
    print(f"n_non_colliding: {n_non_colliding}")
    print(f"n_colliding: {n_colliding}")
    print(f"n_pickable: {n_pickable}")
    print(f"n_failed: {n_failed}")

    return {
        "n_non_colliding": n_non_colliding,
        "n_colliding": n_colliding,
        "n_pickable": n_pickable,
        "n_failed": n_failed,
    }


def main() -> None:
    # variables
    grasp_dir = f"{ASSETS_DIR}/grasps/static_objects"

    # Test multiple object types and scenes
    object_types = ["pencil"]  # ["knife", "fork", "spoon"]  # List of objects to test
    scene_ids = [
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
    ]  # ["1", "2", "3", "4", "5", "6", "7", "8"]  # List of scenes to test

    use_passive_viewer = False
    use_pregrasp = True
    num_workers = None  # None = auto-detect (half of CPU cores)
    save_video = True  # True

    num_grasps = 1000

    print(f"Testing {len(object_types)} object types across {len(scene_ids)} scenes")
    print(f"Total combinations: {len(object_types) * len(scene_ids)}")
    print(f"use_pregrasp: {use_pregrasp}")
    print(f"multiprocessing: {not use_passive_viewer}")
    if not use_passive_viewer:
        print(
            f"num_workers: {num_workers if num_workers is not None else 'auto (half of CPU cores)'}"
        )

    # Store all test results
    all_results = []

    # Iterate through all combinations
    for object_type in object_types:
        for scene_id in scene_ids:
            print(f"\n{'=' * 80}")
            print(f"Testing: {object_type} in Scene {scene_id}")
            print(f"{'=' * 80}")

            scene_path = f"assets/scenes/ithor/FloorPlan{scene_id}_physics_mesh.xml"
            gripper_scene_path = scene_path.replace(".xml", "_robot.xml")

            try:
                result = run_scene(
                    gripper_scene_path,
                    num_grasps,
                    object_type=object_type,
                    grasp_dir=grasp_dir,
                    use_passive_viewer=use_passive_viewer,
                    use_pregrasp=use_pregrasp,
                    num_workers=num_workers,
                    save_video=save_video,
                )

                # Store result with metadata
                result["object_type"] = object_type
                result["scene_id"] = scene_id
                all_results.append(result)

            except Exception as e:
                print(f"Error testing {object_type} in Scene {scene_id}: {e}")
                continue

    # Print per-object-type statistics
    print(f"\n{'=' * 80}")
    print("SUCCESS RATES BY OBJECT TYPE")
    print(f"{'=' * 80}")
    for obj_type in object_types:
        obj_results = [r for r in all_results if r["object_type"] == obj_type]
        obj_non_colliding = sum(r["n_non_colliding"] for r in obj_results)
        obj_pickable = sum(r["n_pickable"] for r in obj_results)
        if obj_non_colliding > 0:
            obj_success_rate = (obj_pickable / obj_non_colliding) * 100
            print(f"{obj_type}: {obj_success_rate:.2f}%")
        else:
            print(f"{obj_type}: No valid grasps")

    # Save results to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"debug/grasp_test_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
