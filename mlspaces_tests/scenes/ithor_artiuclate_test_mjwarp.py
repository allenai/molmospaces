"""Batched iTHOR articulation test on top of mujoco_warp.

This is the mjwarp port of the old MJX version: every handle is simulated in its
own mjwarp world, so a batch of handles is stepped in a single kernel launch
instead of being vmapped/pmapped across JAX devices.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from ithor_artiuclate_test_mjwarp_simple import (
    add_gripper_to_scene,
    get_all_handles,
    get_gripper_pose_based_on_handle_pose,
    get_handle_qpos_index,
    regenerate_map,
    warp_device,
)
from tqdm import tqdm

from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.utils.scene_maps import iTHORMap

# Performance settings: one mjwarp world per handle
BATCH_SIZE = 32  # Number of handles simulated in parallel (worlds per rollout)
PHYSICS_STEPS_PER_FRAME = 25  # Physics steps per rollout
VIDEO_STEPS_PER_FRAME = 5  # Physics steps between recorded frames
VIDEO_CAMERA = "robot_0/follower"  # rum gripper specific
VIDEO_FPS = 10


class WorldVideoRecorder:
    """Records a single mjwarp world by copying it back into an MjData for rendering."""

    def __init__(self, model, world_id=0, camera=VIDEO_CAMERA) -> None:
        self.model = model
        self.world_id = world_id
        self.camera = camera
        self.mj_data = mujoco.MjData(model)
        self.renderer = MjOpenGLRenderer(model=model, device_id=None)
        self.frames = []

    def __call__(self, data) -> None:
        mjw.get_data_into(self.mj_data, self.model, data, world_id=self.world_id)
        mujoco.mj_forward(self.model, self.mj_data)
        self.renderer.update(self.mj_data, camera=self.camera)
        self.frames.append(self.renderer.render().copy())

    def save(self, save_dir) -> None:
        if len(self.frames) == 0:
            return
        import imageio.v2 as imageio

        os.makedirs(save_dir, exist_ok=True)
        video_path = os.path.join(save_dir, f"world_{self.world_id}.mp4")
        imageio.mimwrite(video_path, self.frames, fps=VIDEO_FPS)
        print(f"  [VIDEO] Wrote {len(self.frames)} frames to {video_path}")

    def close(self) -> None:
        self.renderer.close()


def rollout(mjw_model, data, nsteps, recorder=None, use_graph=True):
    """Step every world `nsteps` times, optionally recording one world along the way."""
    if recorder is not None:
        for step in range(nsteps):
            mjw.step(mjw_model, data)
            if step % VIDEO_STEPS_PER_FRAME == 0:
                recorder(data)
        return

    if use_graph and wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            for _ in range(nsteps):
                mjw.step(mjw_model, data)
        wp.capture_launch(capture.graph)
        wp.synchronize()
    else:
        for _ in range(nsteps):
            mjw.step(mjw_model, data)


def process_handles_with_mjwarp(
    handles, ithor_scene_path, ithor_map, device_id=0, record_video=False, use_graph=True
):
    """Simulate one handle per mjwarp world; returns (results, recorder)."""

    if len(handles) == 0:
        return [], None

    model = mujoco.MjModel.from_xml_path(ithor_scene_path)

    device = warp_device(device_id)
    print(f"  [MJWARP] Putting model on {device} for {len(handles)} worlds...")

    recorder = None
    with wp.ScopedDevice(device):
        put_start_time = time.time()
        mjw_model = mjw.put_model(model)

        # Common initial state for every world
        mjd = mujoco.MjData(model)
        mujoco.mj_forward(model, mjd)
        data = mjw.put_data(model, mjd, nworld=len(handles))
        print(f"  [MJWARP] Model/data upload took {time.time() - put_start_time:.3f}s")

        # Place the gripper mocap of every world in front of its own handle
        if model.nmocap > 0:
            mocap_pos = data.mocap_pos.numpy()
            mocap_quat = data.mocap_quat.numpy()
            for world_id, handle in enumerate(handles):
                gripper_pose = get_gripper_pose_based_on_handle_pose(handle, ithor_map)
                mocap_pos[world_id, 0] = gripper_pose["position"]
                mocap_quat[world_id, 0] = gripper_pose["quaternion"]
            data.mocap_pos.assign(mocap_pos)
            data.mocap_quat.assign(mocap_quat)
        else:
            print("  [MJWARP] Warning: scene has no mocap body, gripper poses are not applied")

        handle_qpos_indices = [get_handle_qpos_index(model, handle) for handle in handles]
        start_qpos = data.qpos.numpy().copy()

        if record_video:
            recorder = WorldVideoRecorder(model, world_id=0)

        sim_start_time = time.time()
        rollout(mjw_model, data, PHYSICS_STEPS_PER_FRAME, recorder=recorder, use_graph=use_graph)
        print(
            f"  [MJWARP] Stepped {len(handles)} worlds x {PHYSICS_STEPS_PER_FRAME} steps "
            f"in {time.time() - sim_start_time:.3f}s"
        )

        end_qpos = data.qpos.numpy()

    results = []
    for world_id, handle in enumerate(handles):
        handle_qpos_idx = handle_qpos_indices[world_id]
        if handle_qpos_idx == -1:
            continue

        start_handle_joint_pos = start_qpos[world_id, handle_qpos_idx]
        end_handle_joint_pos = end_qpos[world_id, handle_qpos_idx]
        success = int(np.abs(end_handle_joint_pos - start_handle_joint_pos) > np.deg2rad(5))

        results.append(
            {
                "handle_name": handle["name"],
                "start_handle_joint_pos": float(start_handle_joint_pos),
                "end_handle_joint_pos": float(end_handle_joint_pos),
                "success": success,
            }
        )

    return results, recorder


def run_one_floorplan_mjwarp(
    i, mesh, thread_id=None, use_gpu=False, record_video=False, use_graph=True
) -> None:
    date = "070925"
    if mesh:
        ithor_scene_path = f"debug/good_iTHOR_{date}/FloorPlan{i}_physics_mesh.xml"
        ithor_map_path = f"{ithor_scene_path.replace('_mesh.xml', '_map.png')}"
    else:
        ithor_scene_path = f"debug/good_iTHOR_{date}/FloorPlan{i}_physics.xml"
        ithor_map_path = f"{ithor_scene_path.replace('.xml', '_map.png')}"

    if not os.path.exists(ithor_scene_path):
        print(f"Scene path {ithor_scene_path} does not exist")
        return

    print(f"Processing FloorPlan {i} with mjwarp (thread {thread_id})...")

    regenerate_map(ithor_scene_path, thread_id, use_gpu)
    ithor_map = iTHORMap.load(ithor_map_path)

    # Initialize scene
    model = mujoco.MjModel.from_xml_path(ithor_scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    all_handles = get_all_handles(model, data)
    print(f"Found {len(all_handles)} handles")
    if len(all_handles) == 0:
        return

    # Add gripper to scene once
    first_handle = all_handles[0]
    gripper_pose = get_gripper_pose_based_on_handle_pose(first_handle, ithor_map)
    gripper_path = add_gripper_to_scene(ithor_scene_path, gripper_pose)
    print("Added gripper to scene")

    success_metric = {}
    n_success = 0
    n_total = 0
    failed_handles = []

    # Split handles into batches of worlds
    handle_batches = [
        all_handles[j : j + BATCH_SIZE] for j in range(0, len(all_handles), BATCH_SIZE)
    ]

    device_id = thread_id if thread_id is not None else 0

    print(
        f"  [FLOORPLAN] Starting mjwarp processing with {len(handle_batches)} batches, "
        f"{len(all_handles)} total handles"
    )
    floorplan_start_time = time.time()

    for batch_idx, handle_batch in enumerate(handle_batches):
        print(
            f"Processing mjwarp batch {batch_idx + 1}/{len(handle_batches)} ({len(handle_batch)} handles)"
        )
        print(f"  [BATCH] Device ID: {device_id}, Worlds: {len(handle_batch)}")

        batch_start_time = time.time()

        batch_results, recorder = process_handles_with_mjwarp(
            handle_batch,
            gripper_path,
            ithor_map,
            device_id,
            record_video=record_video,
            use_graph=use_graph,
        )

        if recorder is not None:
            save_dir = f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}/batch_{batch_idx}"
            recorder.save(save_dir)
            recorder.close()

        print(f"  [BATCH] Total batch processing time: {time.time() - batch_start_time:.3f}s")

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
    floorplan_time = time.time() - floorplan_start_time
    print(f"  [FLOORPLAN] Completed mjwarp processing in {floorplan_time:.3f}s")
    print(f"  [FLOORPLAN] Average time per handle: {floorplan_time / len(all_handles):.3f}s")

    success_metric["success_rate"] = n_success / n_total if n_total > 0 else 0
    success_metric["failed_handles"] = failed_handles
    success_metric["processing_time_seconds"] = floorplan_time
    success_metric["total_handles"] = len(all_handles)
    success_metric["total_batches"] = len(handle_batches)
    print(success_metric)
    os.makedirs(f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}", exist_ok=True)
    with open(
        f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}/success_metric_mjwarp.json", "w"
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
    parser.add_argument(
        "--batch-size", type=int, default=32, help="mjwarp worlds per rollout (default: 32)"
    )
    parser.add_argument(
        "--physics-steps", type=int, default=25, help="Physics steps per frame (default: 25)"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of the first world of each batch (slower but provides visual output)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable CUDA graph capture of the stepping loop (useful for debugging)",
    )
    args = parser.parse_args()

    # Update global constants
    global BATCH_SIZE, PHYSICS_STEPS_PER_FRAME
    BATCH_SIZE = args.batch_size
    PHYSICS_STEPS_PER_FRAME = args.physics_steps

    wp.init()
    if args.gpu:
        print(f"Using {wp.get_cuda_device_count()} CUDA devices: {wp.get_cuda_devices()}")

    i = args.i
    mesh = args.mesh
    nthread = args.nthread
    use_gpu = args.gpu
    record_video = args.record_video
    use_graph = not args.no_graph

    if i == -1:
        floorplan_indices = list(range(13, 0, -1))

        print(
            f"Processing {len(floorplan_indices)} floorplans using mjwarp with {nthread} threads..."
        )
        print(f"Floorplan range: {floorplan_indices[0]} to {floorplan_indices[-1]}")
        print(f"mjwarp worlds: {BATCH_SIZE}, Physics steps per frame: {PHYSICS_STEPS_PER_FRAME}")
        if record_video:
            print("Video recording enabled (will be slower but provide visual output)")

        completed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=nthread) as executor:
            future_to_floorplan = {
                executor.submit(
                    run_one_floorplan_mjwarp,
                    floorplan_i,
                    mesh,
                    thread_id,
                    use_gpu,
                    record_video,
                    use_graph,
                ): floorplan_i
                for thread_id, floorplan_i in enumerate(floorplan_indices)
            }

            with tqdm(
                total=len(floorplan_indices), desc="Processing floorplans with mjwarp"
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
            f"All floorplans processed with mjwarp! Completed: {completed_count}, Failed: {failed_count}"
        )
        return
    else:
        run_one_floorplan_mjwarp(
            i, mesh, use_gpu=use_gpu, record_video=record_video, use_graph=use_graph
        )


if __name__ == "__main__":
    main()
