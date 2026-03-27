import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from molmo_spaces.data_generation.recorder import RGBRecorder
from molmo_spaces.editor.thor_model_editor import ThorMjModelEditor

# Add imports for OpenGL rendering and video recording
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer
from molmo_spaces.utils.scene_maps import iTHORMap

# Performance optimization: MJX batch processing
BATCH_SIZE = 2  # Further reduced batch size to prevent GPU memory issues
PHYSICS_STEPS_PER_FRAME = 25  # Reduced for faster simulation
SUB_BATCH_SIZE = 1  # Very small sub-batch size to minimize memory usage

# Memory management settings
MAX_GPU_MEMORY_GB = 6  # Maximum GPU memory to use (in GB)
MEMORY_MARGIN_GB = 1  # Safety margin (in GB)

# Cache for reusable MJX models and data templates
_mjx_model_cache = {}
_mjx_data_template_cache = {}

# Memory pool for MJX data to avoid repeated allocations
_mjx_data_pool = {}


def get_cached_mjx_model(model, device_id):
    """Get or create cached MJX model to avoid repeated conversion"""
    model_id = id(model)
    device = jax.devices()[device_id]
    cache_key = (model_id, device)

    if cache_key not in _mjx_model_cache:
        print(f"  [CACHE] Creating new MJX model for device {device_id}")
        mjx_model = mjx.put_model(model, device=device)
        _mjx_model_cache[cache_key] = mjx_model
    else:
        print(f"  [CACHE] Reusing MJX model for device {device_id}")

    return _mjx_model_cache[cache_key]


def get_cached_mjx_data_template(model, device_id):
    """Get or create cached MJX data template to avoid repeated conversion"""
    model_id = id(model)
    device = jax.devices()[device_id]
    cache_key = (model_id, device)

    if cache_key not in _mjx_data_template_cache:
        print(f"  [CACHE] Creating new MJX data template for device {device_id}")
        data = mujoco.MjData(model)
        mjx_data = mjx.put_data(model, data, device=device)
        _mjx_data_template_cache[cache_key] = mjx_data
    else:
        print(f"  [CACHE] Reusing MJX data template for device {device_id}")

    return _mjx_data_template_cache[cache_key]


def create_mjx_batch_from_template(template, batch_size):
    """Create MJX batch from template without repeated conversion"""
    return jax.tree.map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), template)


def reset_mjx_data_state(mjx_data_template, device_id):
    """Reset MJX data to initial state without recreating the entire structure"""
    # Create a fresh MuJoCo data instance
    data = mujoco.MjData(mjx_data_template.model)

    # Convert to MJX data
    device = jax.devices()[device_id]
    reset_data = mjx.put_data(mjx_data_template.model, data, device=device)

    return reset_data


def update_mjx_data_poses(mjx_data, gripper_poses, device_id):
    """Update only the necessary poses in MJX data without full reset"""
    # This is a simplified version - in practice you'd want to update specific fields
    # For now, we'll use the existing update function
    return _update_gripper_positions_jit(mjx_data, gripper_poses)


def get_mjx_data_from_pool(model, device_id, batch_size):
    """Get MJX data from pool or create new if not available"""
    model_id = id(model)
    device = jax.devices()[device_id]
    pool_key = (model_id, device, batch_size)

    if pool_key not in _mjx_data_pool:
        print(
            f"  [POOL] Creating new MJX data pool for device {device_id}, batch_size {batch_size}"
        )
        # Create template and batch
        template = get_cached_mjx_data_template(model, device_id)
        batch = create_mjx_batch_from_template(template, batch_size)
        _mjx_data_pool[pool_key] = batch
    else:
        print(
            f"  [POOL] Reusing MJX data from pool for device {device_id}, batch_size {batch_size}"
        )

    return _mjx_data_pool[pool_key]


def return_mjx_data_to_pool(model, device_id, batch_size, mjx_data) -> None:
    """Return MJX data to pool for reuse (optional - for explicit memory management)"""
    model_id = id(model)
    device = jax.devices()[device_id]
    pool_key = (model_id, device, batch_size)

    # Reset data to initial state before returning to pool
    # This is optional since we'll reset when getting from pool anyway
    _mjx_data_pool[pool_key] = mjx_data


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
            parent_id = model.body(i).parentid
            parent_name = model.body(parent_id).name
            if parent_id == 0:
                continue
            root_body = model.body(i).rootid
            model.body(root_body).name

            # Fix: Properly access geom information
            geom_id = model.body(i).geomadr[0]
            geom_num = model.body(i).geomnum[0]

            # Find the first geom with group 0 (visible geom)
            found_geom_id = None
            for j in range(geom_num):
                try:
                    geom_group = model.geom(geom_id + j).group
                    if geom_group == 0:
                        found_geom_id = geom_id + j
                        break
                except (IndexError, AttributeError):
                    # Handle case where geom access fails
                    continue

            if found_geom_id is None:
                # Fallback: use the first geom
                found_geom_id = geom_id

            # Get bounding box information safely
            body_id = model.body_bvhadr[i]
            if body_id >= 0 and body_id < len(model.bvh_aabb):
                handle_bounding_box = model.bvh_aabb[body_id]
                handle_bounding_box[:3]
                size = handle_bounding_box[3:]
            else:
                # Fallback: use default values
                np.array([0.0, 0.0, 0.0])
                size = np.array([0.1, 0.1, 0.1])

            all_handles.append(
                {
                    "name": parent_name,
                    "position": data.geom_xpos[found_geom_id],
                    "orientation": data.geom_xmat[found_geom_id],
                    "handle_id": parent_id,
                    "geom_id": found_geom_id,
                    "size": size,
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


# Pre-compile JIT functions to avoid repeated compilation
_update_gripper_positions_jit = None
_step_simulation_jit = None
_get_handle_joint_pos_jit = None


def _compile_jit_functions() -> None:
    """Pre-compile JIT functions once to avoid repeated compilation"""
    global _update_gripper_positions_jit, _step_simulation_jit, _get_handle_joint_pos_jit

    try:
        if _update_gripper_positions_jit is None:
            print("  [JIT] Compiling update_gripper_positions function...")

            def update_gripper_positions(data, gripper_pose):
                # gripper_pose is now a tuple of (position, quaternion)
                position, quaternion = gripper_pose

                # Update mocap body position
                data = data.replace(mocap_pos=data.mocap_pos.at[0].set(position))
                data = data.replace(mocap_quat=data.mocap_quat.at[0].set(quaternion))

                # Update robot joint positions
                gripper_qpos_start = 0  # Assuming first 7 DOFs are gripper
                data = data.replace(
                    qpos=data.qpos.at[gripper_qpos_start : gripper_qpos_start + 3].set(position)
                )
                data = data.replace(
                    qpos=data.qpos.at[gripper_qpos_start + 3 : gripper_qpos_start + 7].set(
                        quaternion
                    )
                )
                return data

            _update_gripper_positions_jit = jax.jit(jax.vmap(update_gripper_positions))
            print("  [JIT] Compiled update_gripper_positions function")

        if _step_simulation_jit is None:
            print("  [JIT] Compiling step_simulation function...")
            _step_simulation_jit = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
            print("  [JIT] Compiled step_simulation function")

        if _get_handle_joint_pos_jit is None:
            print("  [JIT] Compiling get_handle_joint_pos function...")

            def get_handle_joint_pos(data, handle_qpos_idx):
                return data.qpos[handle_qpos_idx]

            _get_handle_joint_pos_jit = jax.jit(jax.vmap(get_handle_joint_pos))
            print("  [JIT] Compiled get_handle_joint_pos function")

    except Exception as e:
        print(f"  [JIT] Error compiling functions: {e}")
        print("  [JIT] Falling back to non-JIT version")
        # Fallback to non-JIT versions
        _update_gripper_positions_jit = None
        _step_simulation_jit = None
        _get_handle_joint_pos_jit = None


@jax.jit
def mjx_batch_simulate_handles(
    mjx_model, mjx_data_batch, gripper_poses, handle_ids, handle_qpos_indices
):
    """Vectorized simulation of multiple handles using MJX"""

    # Ensure JIT functions are compiled
    _compile_jit_functions()

    # Use JIT functions if available, otherwise fallback to non-JIT
    if _update_gripper_positions_jit is not None:
        # Apply gripper poses to all data instances using pre-compiled vmap
        mjx_data_batch = _update_gripper_positions_jit(mjx_data_batch, gripper_poses)

        # Run physics simulation for all instances using pre-compiled vmap
        mjx_data_batch = _step_simulation_jit(mjx_model, mjx_data_batch)

        # Extract handle joint positions using pre-compiled vmap
        handle_joint_positions = _get_handle_joint_pos_jit(mjx_data_batch, handle_qpos_indices)
    else:
        # Fallback to non-JIT version
        print("  [MJX] Using non-JIT fallback execution")

        def update_gripper_positions(data, gripper_pose):
            # gripper_pose is now a tuple of (position, quaternion)
            position, quaternion = gripper_pose

            # Update mocap body position
            data = data.replace(mocap_pos=data.mocap_pos.at[0].set(position))
            data = data.replace(mocap_quat=data.mocap_quat.at[0].set(quaternion))

            # Update robot joint positions
            gripper_qpos_start = 0  # Assuming first 7 DOFs are gripper
            data = data.replace(
                qpos=data.qpos.at[gripper_qpos_start : gripper_qpos_start + 3].set(position)
            )
            data = data.replace(
                qpos=data.qpos.at[gripper_qpos_start + 3 : gripper_qpos_start + 7].set(quaternion)
            )
            return data

        # Apply gripper poses to all data instances using vmap
        mjx_data_batch = jax.vmap(update_gripper_positions)(mjx_data_batch, gripper_poses)

        # Run physics simulation for all instances using vmap
        mjx_data_batch = jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, mjx_data_batch)

        # Extract handle joint positions
        def get_handle_joint_pos(data, handle_qpos_idx):
            return data.qpos[handle_qpos_idx]

        handle_joint_positions = jax.vmap(get_handle_joint_pos)(mjx_data_batch, handle_qpos_indices)

    return mjx_data_batch, handle_joint_positions


def process_handles_with_mjx_and_video(
    handles, ithor_scene_path, ithor_map, device_id=1, record_video=False
):
    """Process handles using MJX for vectorized simulation with optional video recording"""

    # Load model and convert to MJX
    model = mujoco.MjModel.from_xml_path(ithor_scene_path)

    # Disable unsupported options for MJX
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)

    # Note: mjENBL_SPARSE not available in this MuJoCo version
    # MJX will automatically set jacobian to "dense" for compatibility

    # Set solver to NEWTON and integrator to IMPLICITFAST
    # model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Explicitly set jacobian to dense for MJX compatibility
    # model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

    print(f"  [MJX] Converting model to MJX on device {device_id}...")
    try:
        mjx.put_model(model, device=jax.devices()[device_id])
        print("  [MJX] Successfully converted model to MJX")
    except Exception as e:
        print(f"  [MJX] Error converting model to MJX: {e}")
        raise

    # Initialize video recording if requested
    recorder = None
    renderer = None
    if record_video:
        renderer = MjOpenGLRenderer(model=model, device_id=None)
        recorder = RGBRecorder(
            period_ms=100,
            camera_name="robot_0/follower",  # "robot_0/egocentric",
            renderer=renderer,
            save_video=True,
        )

    # Prepare batch data
    gripper_poses = []
    handle_ids = []
    handle_qpos_indices = []

    for handle in handles:
        gripper_pose = get_gripper_pose_based_on_handle_pose(handle, ithor_map)
        gripper_poses.append(gripper_pose)

        handle_id = handle["handle_id"]
        handle_ids.append(handle_id)

        # Get joint qpos index
        handle_joint_id = int(model.body(handle_id).jntadr[0])
        if handle_joint_id != -1:
            handle_qpos_idx = model.joint(handle_joint_id).qposadr[0]
            handle_qpos_indices.append(handle_qpos_idx)
        else:
            handle_qpos_indices.append(-1)

    # Convert to JAX arrays
    gripper_poses = jax.tree.map(lambda *args: jnp.array(args), *gripper_poses)
    handle_qpos_indices = jnp.array(handle_qpos_indices)

    # Distribute workload across all available GPUs
    import concurrent.futures

    num_gpus = min(8, len(jax.devices()))
    print(f"  [MJX] Distributing {len(handles)} handles across {num_gpus} GPUs...")

    # Convert JAX arrays to numpy arrays for splitting
    gripper_poses_np = [np.array(gp) for gp in gripper_poses]
    handle_qpos_indices_np = np.array(handle_qpos_indices)

    handle_chunks = np.array_split(handles, num_gpus)
    gripper_pose_chunks = np.array_split(gripper_poses_np, num_gpus)
    qpos_idx_chunks = np.array_split(handle_qpos_indices_np, num_gpus)

    print(f"  [MJX] Created {len(handle_chunks)} chunks: {[len(chunk) for chunk in handle_chunks]}")

    def process_on_gpu(gpu_id, handle_chunk, gripper_pose_chunk, qpos_idx_chunk):
        print(f"    [GPU{gpu_id}] Starting processing of {len(handle_chunk)} handles...")
        try:
            device = jax.devices()[gpu_id]
            print(f"    [GPU{gpu_id}] Using device: {device}")

            # Check GPU memory before starting
            allocated, reserved = get_gpu_memory_info(gpu_id)
            print(
                f"    [GPU{gpu_id}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

            # Use cached MJX model instead of creating new one
            mjx_model_gpu = get_cached_mjx_model(model, gpu_id)
            print(f"    [GPU{gpu_id}] Got MJX model from cache")

            # Get cached data template
            get_cached_mjx_data_template(model, gpu_id)
            print(f"    [GPU{gpu_id}] Got MJX data template from cache")

            # Process in very small sub-batches to avoid memory issues
            all_mjx_data_batch = []
            all_handle_joint_positions = []

            for sub_batch_start in range(0, len(handle_chunk), SUB_BATCH_SIZE):
                sub_batch_end = min(sub_batch_start + SUB_BATCH_SIZE, len(handle_chunk))
                sub_handle_chunk = handle_chunk[sub_batch_start:sub_batch_end]
                sub_gripper_pose_chunk = gripper_pose_chunk[sub_batch_start:sub_batch_end]
                sub_qpos_idx_chunk = qpos_idx_chunk[sub_batch_start:sub_batch_end]

                print(
                    f"    [GPU{gpu_id}] Processing sub-batch {sub_batch_start // SUB_BATCH_SIZE + 1}/{(len(handle_chunk) + SUB_BATCH_SIZE - 1) // SUB_BATCH_SIZE} ({len(sub_handle_chunk)} handles)"
                )

                # Check memory before processing sub-batch
                estimated_memory = estimate_batch_memory_usage(len(sub_handle_chunk))
                if not check_gpu_memory_available(gpu_id, estimated_memory):
                    print(f"    [GPU{gpu_id}] Warning: Low memory, clearing cache...")
                    clear_gpu_memory(gpu_id)

                # Get MJX data from pool instead of creating from template
                mjx_data_batch = get_mjx_data_from_pool(model, gpu_id, len(sub_handle_chunk))
                print(
                    f"    [GPU{gpu_id}] Got MJX data from pool with {len(sub_handle_chunk)} instances"
                )

                # Convert batch data to JAX arrays
                if len(sub_handle_chunk) == 0:
                    print(f"    [GPU{gpu_id}] No handles to process in sub-batch")
                    continue

                sub_gripper_pose_chunk = jax.tree.map(
                    lambda x: jnp.array(x), sub_gripper_pose_chunk
                )
                sub_qpos_idx_chunk = jnp.array(sub_qpos_idx_chunk)
                print(f"    [GPU{gpu_id}] Converted to JAX arrays, running simulation...")

                # Run simulation
                mjx_data_batch, handle_joint_positions = mjx_batch_simulate_handles(
                    mjx_model_gpu,
                    mjx_data_batch,
                    sub_gripper_pose_chunk,
                    [h["handle_id"] for h in sub_handle_chunk],
                    sub_qpos_idx_chunk,
                )
                print(
                    f"    [GPU{gpu_id}] Completed sub-batch simulation, got {len(handle_joint_positions)} results"
                )

                all_mjx_data_batch.append(mjx_data_batch)
                all_handle_joint_positions.extend(handle_joint_positions)

                # Clear memory after each sub-batch
                if len(all_mjx_data_batch) % 5 == 0:  # Clear every 5 sub-batches
                    clear_gpu_memory(gpu_id)

            print(
                f"    [GPU{gpu_id}] Completed all sub-batches, total {len(all_handle_joint_positions)} results"
            )
            return all_mjx_data_batch, all_handle_joint_positions
        except Exception as e:
            print(f"    [GPU{gpu_id}] Error: {e}")
            import traceback

            traceback.print_exc()
            raise

    print(f"  [MJX] Starting ThreadPoolExecutor with {num_gpus} workers...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            if len(handle_chunks[gpu_id]) > 0:  # Only submit if there are handles to process
                print(f"  [MJX] Submitting GPU {gpu_id} with {len(handle_chunks[gpu_id])} handles")
                future = executor.submit(
                    process_on_gpu,
                    gpu_id,
                    list(handle_chunks[gpu_id]),
                    list(gripper_pose_chunks[gpu_id]),
                    list(qpos_idx_chunks[gpu_id]),
                )
                futures.append(future)
            else:
                print(f"  [MJX] Skipping GPU {gpu_id} (no handles)")

        print(f"  [MJX] Waiting for {len(futures)} futures to complete...")
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print("  [MJX] Completed one GPU task")
            except Exception as e:
                print(f"  [MJX] Future failed: {e}")
                raise

    print("  [MJX] All GPU tasks completed, combining results...")

    # Combine results from all GPUs
    all_mjx_data_batch, all_handle_joint_positions = zip(*results)
    # Flatten
    all_mjx_data_batch = [
        item
        for sublist in all_mjx_data_batch
        for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])
    ]
    all_handle_joint_positions = [
        item
        for sublist in all_handle_joint_positions
        for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])
    ]

    # For video, only support if all_mjx_data_batch is not empty
    if record_video and recorder is not None and len(all_mjx_data_batch) > 0:
        print(
            f"  [VIDEO] Converting {len(all_mjx_data_batch)} MJX data instances back to MuJoCo for rendering..."
        )
        for mjx_data_batch in all_mjx_data_batch:
            batched_mj_data = mjx.get_data(model, mjx_data_batch)
            for data in batched_mj_data:
                recorder(data)
        print(f"  [VIDEO] Recorded {len(all_mjx_data_batch)} video frames")

    # Process results
    results = []
    idx = 0
    for _chunk_idx, handle_chunk in enumerate(handle_chunks):
        for i, handle in enumerate(handle_chunk):
            handle_name = handle["name"]
            handle_id = handle["handle_id"]
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            handle_joint_id = int(model.body(handle_id).jntadr[0])
            if handle_joint_id != -1:
                handle_qpos_idx = model.joint(handle_joint_id).qposadr[0]
                start_handle_joint_pos = data.qpos[handle_qpos_idx]
                end_handle_joint_pos = (
                    all_handle_joint_positions[idx][i]
                    if isinstance(all_handle_joint_positions[idx], (list, tuple, np.ndarray))
                    else all_handle_joint_positions[idx]
                )
                success = int(np.abs(end_handle_joint_pos - start_handle_joint_pos) > np.deg2rad(5))
                results.append(
                    {
                        "handle_name": handle_name,
                        "start_handle_joint_pos": float(start_handle_joint_pos),
                        "end_handle_joint_pos": float(end_handle_joint_pos),
                        "success": success,
                    }
                )
        idx += 1
    return results, recorder


def process_handles_with_mjx(handles, ithor_scene_path, ithor_map, device_id=1):
    """Process handles using MJX for vectorized simulation"""

    # Load model and convert to MJX
    model = mujoco.MjModel.from_xml_path(ithor_scene_path)

    # Disable unsupported options for MJX
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)

    # Additional compatibility fixes for MJX
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_FWDINV)

    # Set solver to NEWTON and integrator to IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    print(f"  [MJX] Converting model to MJX on device {device_id}...")
    try:
        mjx.put_model(model, device=jax.devices()[device_id])
        print("  [MJX] Successfully converted model to MJX")
    except Exception as e:
        print(f"  [MJX] Error converting model to MJX: {e}")
        raise

    # Prepare batch data
    gripper_poses = []
    handle_ids = []
    handle_qpos_indices = []

    for handle in handles:
        gripper_pose = get_gripper_pose_based_on_handle_pose(handle, ithor_map)
        # Convert to tuple format for JAX compatibility
        gripper_poses.append((gripper_pose["position"], gripper_pose["quaternion"]))

        handle_id = handle["handle_id"]
        handle_ids.append(handle_id)

        # Get joint qpos index
        handle_joint_id = int(model.body(handle_id).jntadr[0])
        if handle_joint_id != -1:
            handle_qpos_idx = model.joint(handle_joint_id).qposadr[0]
            handle_qpos_indices.append(handle_qpos_idx)
        else:
            handle_qpos_indices.append(-1)

    # Convert to JAX arrays - now gripper_poses is a list of tuples
    gripper_positions = [pose[0] for pose in gripper_poses]
    gripper_quaternions = [pose[1] for pose in gripper_poses]

    gripper_positions = jnp.array(gripper_positions)
    gripper_quaternions = jnp.array(gripper_quaternions)
    gripper_poses = (gripper_positions, gripper_quaternions)  # Tuple of arrays
    handle_qpos_indices = jnp.array(handle_qpos_indices)

    # Distribute workload across all available GPUs
    import concurrent.futures

    num_gpus = min(8, len(jax.devices()))
    print(f"  [MJX] Distributing {len(handles)} handles across {num_gpus} GPUs...")

    # Convert JAX arrays to numpy arrays for splitting
    gripper_positions_np = np.array(gripper_positions)
    gripper_quaternions_np = np.array(gripper_quaternions)
    handle_qpos_indices_np = np.array(handle_qpos_indices)

    handle_chunks = np.array_split(handles, num_gpus)
    gripper_positions_chunks = np.array_split(gripper_positions_np, num_gpus)
    gripper_quaternions_chunks = np.array_split(gripper_quaternions_np, num_gpus)
    qpos_idx_chunks = np.array_split(handle_qpos_indices_np, num_gpus)

    print(f"  [MJX] Created {len(handle_chunks)} chunks: {[len(chunk) for chunk in handle_chunks]}")

    def process_on_gpu(
        gpu_id, handle_chunk, gripper_positions_chunk, gripper_quaternions_chunk, qpos_idx_chunk
    ):
        print(f"    [GPU{gpu_id}] Starting processing of {len(handle_chunk)} handles...")
        try:
            device = jax.devices()[gpu_id]
            print(f"    [GPU{gpu_id}] Using device: {device}")

            # Check GPU memory before starting
            allocated, reserved = get_gpu_memory_info(gpu_id)
            print(
                f"    [GPU{gpu_id}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

            # Use cached MJX model instead of creating new one
            mjx_model_gpu = get_cached_mjx_model(model, gpu_id)
            print(f"    [GPU{gpu_id}] Got MJX model from cache")

            # Get cached data template
            get_cached_mjx_data_template(model, gpu_id)
            print(f"    [GPU{gpu_id}] Got MJX data template from cache")

            # Process in very small sub-batches to avoid memory issues
            all_handle_joint_positions = []

            for sub_batch_start in range(0, len(handle_chunk), SUB_BATCH_SIZE):
                sub_batch_end = min(sub_batch_start + SUB_BATCH_SIZE, len(handle_chunk))
                sub_handle_chunk = handle_chunk[sub_batch_start:sub_batch_end]
                sub_gripper_positions = gripper_positions_chunk[sub_batch_start:sub_batch_end]
                sub_gripper_quaternions = gripper_quaternions_chunk[sub_batch_start:sub_batch_end]
                sub_qpos_idx_chunk = qpos_idx_chunk[sub_batch_start:sub_batch_end]

                print(
                    f"    [GPU{gpu_id}] Processing sub-batch {sub_batch_start // SUB_BATCH_SIZE + 1}/{(len(handle_chunk) + SUB_BATCH_SIZE - 1) // SUB_BATCH_SIZE} ({len(sub_handle_chunk)} handles)"
                )

                # Check memory before processing sub-batch
                estimated_memory = estimate_batch_memory_usage(len(sub_handle_chunk))
                if not check_gpu_memory_available(gpu_id, estimated_memory):
                    print(f"    [GPU{gpu_id}] Warning: Low memory, clearing cache...")
                    clear_gpu_memory(gpu_id)

                # Get MJX data from pool instead of creating from template
                mjx_data_batch = get_mjx_data_from_pool(model, gpu_id, len(sub_handle_chunk))
                print(
                    f"    [GPU{gpu_id}] Got MJX data from pool with {len(sub_handle_chunk)} instances"
                )

                # Convert batch data to JAX arrays
                if len(sub_handle_chunk) == 0:
                    print(f"    [GPU{gpu_id}] No handles to process in sub-batch")
                    continue

                sub_gripper_positions = jnp.array(sub_gripper_positions)
                sub_gripper_quaternions = jnp.array(sub_gripper_quaternions)
                sub_gripper_poses = (sub_gripper_positions, sub_gripper_quaternions)
                sub_qpos_idx_chunk = jnp.array(sub_qpos_idx_chunk)
                print(f"    [GPU{gpu_id}] Converted to JAX arrays, running simulation...")

                # Run simulation
                mjx_data_batch, handle_joint_positions = mjx_batch_simulate_handles(
                    mjx_model_gpu,
                    mjx_data_batch,
                    sub_gripper_poses,
                    [h["handle_id"] for h in sub_handle_chunk],
                    sub_qpos_idx_chunk,
                )
                print(
                    f"    [GPU{gpu_id}] Completed sub-batch simulation, got {len(handle_joint_positions)} results"
                )

                all_handle_joint_positions.extend(handle_joint_positions)

                # Clear memory after each sub-batch
                if len(all_handle_joint_positions) % 5 == 0:  # Clear every 5 sub-batches
                    clear_gpu_memory(gpu_id)

            print(
                f"    [GPU{gpu_id}] Completed all sub-batches, total {len(all_handle_joint_positions)} results"
            )
            return all_handle_joint_positions
        except Exception as e:
            print(f"    [GPU{gpu_id}] Error: {e}")
            import traceback

            traceback.print_exc()
            raise

    print(f"  [MJX] Starting ThreadPoolExecutor with {num_gpus} workers...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            if len(handle_chunks[gpu_id]) > 0:  # Only submit if there are handles to process
                print(f"  [MJX] Submitting GPU {gpu_id} with {len(handle_chunks[gpu_id])} handles")
                future = executor.submit(
                    process_on_gpu,
                    gpu_id,
                    list(handle_chunks[gpu_id]),
                    list(gripper_positions_chunks[gpu_id]),
                    list(gripper_quaternions_chunks[gpu_id]),
                    list(qpos_idx_chunks[gpu_id]),
                )
                futures.append(future)
            else:
                print(f"  [MJX] Skipping GPU {gpu_id} (no handles)")

        print(f"  [MJX] Waiting for {len(futures)} futures to complete...")
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print("  [MJX] Completed one GPU task")
            except Exception as e:
                print(f"  [MJX] Future failed: {e}")
                raise

    print("  [MJX] All GPU tasks completed, combining results...")

    # Combine results from all GPUs
    all_handle_joint_positions = [
        item
        for sublist in results
        for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])
    ]

    # Process results
    results = []
    idx = 0
    for _chunk_idx, handle_chunk in enumerate(handle_chunks):
        for i, handle in enumerate(handle_chunk):
            handle_name = handle["name"]
            handle_id = handle["handle_id"]
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            handle_joint_id = int(model.body(handle_id).jntadr[0])
            if handle_joint_id != -1:
                handle_qpos_idx = model.joint(handle_joint_id).qposadr[0]
                start_handle_joint_pos = data.qpos[handle_qpos_idx]
                end_handle_joint_pos = (
                    all_handle_joint_positions[idx][i]
                    if isinstance(all_handle_joint_positions[idx], (list, tuple, np.ndarray))
                    else all_handle_joint_positions[idx]
                )
                success = int(np.abs(end_handle_joint_pos - start_handle_joint_pos) > np.deg2rad(5))
                results.append(
                    {
                        "handle_name": handle_name,
                        "start_handle_joint_pos": float(start_handle_joint_pos),
                        "end_handle_joint_pos": float(end_handle_joint_pos),
                        "success": success,
                    }
                )
        idx += 1
    return results


def run_one_floorplan_mjx(i, mesh, thread_id=None, use_gpu=False, record_video=False) -> None:
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

    print(f"Processing FloorPlan {i} with MJX (thread {thread_id})...")

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

    # Process handles in batches using MJX
    success_metric = {}
    n_success = 0
    n_total = 0
    failed_handles = []

    # Split handles into batches
    handle_batches = [
        all_handles[j : j + BATCH_SIZE] for j in range(0, len(all_handles), BATCH_SIZE)
    ]

    device_id = thread_id % len(jax.devices()) if thread_id is not None else 1

    print(
        f"  [FLOORPLAN] Starting MJX processing with {len(handle_batches)} batches, {len(all_handles)} total handles"
    )
    floorplan_start_time = time.time()

    for batch_idx, handle_batch in enumerate(handle_batches):
        print(
            f"Processing MJX batch {batch_idx + 1}/{len(handle_batches)} ({len(handle_batch)} handles)"
        )
        print(f"  [BATCH] Device ID: {device_id}, Batch size: {len(handle_batch)}")

        batch_start_time = time.time()

        # Process batch with MJX (with or without video recording)
        if record_video:
            batch_results, recorder = process_handles_with_mjx_and_video(
                handle_batch, gripper_path, ithor_map, device_id, record_video=True
            )

            # Save video if recorded
            if recorder is not None:
                save_dir = f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}/batch_{batch_idx}"
                os.makedirs(save_dir, exist_ok=True)
                recorder.save(save_dir)
        else:
            batch_results = process_handles_with_mjx(
                handle_batch, gripper_path, ithor_map, device_id
            )

        batch_time = time.time() - batch_start_time
        print(f"  [BATCH] Total batch processing time: {batch_time:.3f}s")

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
    floorplan_time = time.time() - floorplan_start_time
    print(f"  [FLOORPLAN] Completed MJX processing in {floorplan_time:.3f}s")
    print(f"  [FLOORPLAN] Average time per handle: {floorplan_time / len(all_handles):.3f}s")

    success_metric["success_rate"] = n_success / n_total if n_total > 0 else 0
    success_metric["failed_handles"] = failed_handles
    success_metric["processing_time_seconds"] = floorplan_time
    success_metric["total_handles"] = len(all_handles)
    success_metric["total_batches"] = len(handle_batches)
    print(success_metric)
    os.makedirs(f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}", exist_ok=True)
    with open(
        f"debug/mocap_data/iTHOR_rum_gripper/floorplan_{i}/success_metric_mjx.json", "w"
    ) as f:
        json.dump(success_metric, f)


def configure_jax_for_performance() -> None:
    """Configure JAX for optimal performance"""
    import os

    # Enable XLA optimizations
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=true --xla_gpu_enable_triton_softmax_fusion=true"
    )

    # Configure JAX for better performance
    jax.config.update("jax_enable_x64", False)  # Use float32 for better GPU performance
    jax.config.update("jax_default_matmul_precision", "bfloat16")  # Use bfloat16 for matrix ops

    # Enable aggressive optimizations
    jax.config.update("jax_platform_name", "gpu")  # Force GPU usage

    print("  [JAX] Configured for optimal performance")


def get_gpu_memory_info(device_id):
    """Get current GPU memory usage"""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
            allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            return allocated, reserved
    except ImportError:
        pass
    return 0.0, 0.0


def check_gpu_memory_available(device_id, required_gb=1.0):
    """Check if GPU has enough memory available"""
    allocated, reserved = get_gpu_memory_info(device_id)
    available = MAX_GPU_MEMORY_GB - allocated - MEMORY_MARGIN_GB
    return available >= required_gb


def clear_gpu_memory(device_id) -> None:
    """Clear GPU memory cache"""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  [MEMORY] Cleared GPU {device_id} memory cache")
    except ImportError:
        pass


def estimate_batch_memory_usage(batch_size, model_size_mb=100):
    """Estimate memory usage for a batch"""
    # Rough estimation: model + data + gradients + intermediate results
    estimated_gb = (model_size_mb / 1024) * batch_size * 4  # 4x for safety
    return estimated_gb


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
    parser.add_argument("--batch-size", type=int, default=32, help="MJX batch size (default: 32)")
    parser.add_argument(
        "--physics-steps", type=int, default=25, help="Physics steps per frame (default: 25)"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video during simulation (slower but provides visual output)",
    )
    parser.add_argument(
        "--optimize-jax", action="store_true", help="Enable JAX performance optimizations"
    )
    args = parser.parse_args()

    # Update global constants
    global BATCH_SIZE, PHYSICS_STEPS_PER_FRAME
    BATCH_SIZE = args.batch_size
    PHYSICS_STEPS_PER_FRAME = args.physics_steps

    # Configure JAX if optimization is requested
    if args.optimize_jax:
        configure_jax_for_performance()

    # Set JAX environment for GPU
    if args.gpu:
        os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
        print(f"Using {len(jax.devices())} devices: {jax.devices()}")

    i = args.i
    mesh = args.mesh
    nthread = args.nthread
    use_gpu = args.gpu
    record_video = args.record_video

    if i == -1:
        floorplan_indices = list(range(13, 0, -1))

        print(f"Processing {len(floorplan_indices)} floorplans using MJX with {nthread} threads...")
        print(f"Floorplan range: {floorplan_indices[0]} to {floorplan_indices[-1]}")
        print(f"MJX batch size: {BATCH_SIZE}, Physics steps per frame: {PHYSICS_STEPS_PER_FRAME}")
        if record_video:
            print("Video recording enabled (will be slower but provide visual output)")

        completed_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=nthread) as executor:
            future_to_floorplan = {
                executor.submit(
                    run_one_floorplan_mjx, floorplan_i, mesh, thread_id, use_gpu, record_video
                ): floorplan_i
                for thread_id, floorplan_i in enumerate(floorplan_indices)
            }

            with tqdm(total=len(floorplan_indices), desc="Processing floorplans with MJX") as pbar:
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
            f"All floorplans processed with MJX! Completed: {completed_count}, Failed: {failed_count}"
        )
        return
    else:
        run_one_floorplan_mjx(i, mesh, use_gpu=use_gpu, record_video=record_video)


if __name__ == "__main__":
    main()
