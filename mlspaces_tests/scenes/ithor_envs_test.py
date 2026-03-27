import os
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from tqdm import tqdm

from molmo_spaces.env.arena.arena_utils import load_env_with_objects

# Folders
date = "080425"
xml_dir = "assets/scenes/ithor_080625/"
xml_dir = "assets/scenes/procthor-100k-train/"
# xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml") and "gripper" in f and "mesh" not in f]
# xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml") and "gripper_mesh" in f]
# xml_files = [
#    f for f in os.listdir(xml_dir) if f.endswith(".xml") and "gripper" not in f and "mesh" not in f
# ]
# xml_files = [
#    f for f in os.listdir(xml_dir) if f.endswith(".xml") and "gripper" not in f and "mesh" in f
# ]
xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml") and "ceiling" in f]
# NEW: Step size configuration
steps_per_frame = 50  # Number of physics steps per frame (50 steps = 100ms)
physics_timestep = 0.002  # 2ms physics timestep
frame_timestep = steps_per_frame * physics_timestep  # 100ms frame timestep


# Sim params
settle_steps = 1500 // steps_per_frame  # 1 seconds (100 is 0.2 seconds)
monitor_steps = 3000 // steps_per_frame  # 3 second
total_steps = settle_steps + monitor_steps


# Create output directory for images
# output_dir = f"calibration_output_ranger_prim_{frame_timestep*1000:.0f}ms_steps_settle{settle_steps}_monitor{monitor_steps}"
# output_dir = f"calibration_output_ranger_mesh_{frame_timestep*1000:.0f}ms_steps_settle{settle_steps}_monitor{monitor_steps}"
# output_dir = f"calibration_output_prim_{frame_timestep*1000:.0f}ms_steps_settle{settle_steps}_monitor{monitor_steps}"
# output_dir = f"calibration_output_mesh_{frame_timestep*1000:.0f}ms_steps_settle{settle_steps}_monitor{monitor_steps}"
output_dir = f"calibration_output_procthor_{frame_timestep * 1000:.0f}ms_steps_settle{settle_steps}_monitor{monitor_steps}"


os.makedirs(output_dir, exist_ok=True)
print(f"Saving all images to: {output_dir}")
print(
    f"Using {steps_per_frame} physics steps per frame ({frame_timestep * 1000:.0f}ms simulation time per frame)"
)

# Storage
results = []
all_object_displacements = []
all_joint_drifts = []
all_frame_times = []  # Store individual frame times for analysis

for xml_file in tqdm(xml_files):
    ithor_scene_path = os.path.join(xml_dir, xml_file)

    try:
        model, root_bodies_dict = load_env_with_objects(ithor_scene_path)
        data = mujoco.MjData(model)

        # Debug information
        total_joints = model.njnt
        free_joints = sum(
            1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        )
        ball_joints = sum(
            1 for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_BALL
        )
        other_joints = total_joints - free_joints - ball_joints

        print(
            f"Scene {xml_file}: {total_joints} total joints ({free_joints} free, {ball_joints} ball, {other_joints} other)"
        )

        joint_qs = []
        joint_qs_delta = []
        obj_positions = []
        kinetic_energies = []
        contact_forces = []
        frame_times = []  # Track timing for each frame (100ms steps)

        # Pick movable bodies
        movable_bodies = [i for i in range(model.nbody) if model.body_dofnum[i] == 0 and i > 0]

        print(f"Scene {xml_file}: {len(movable_bodies)} movable bodies")

        # Store previous frame's joint positions for delta calculation
        prev_joint_qpos = None

        for _ in range(total_steps):
            step_start_time = time.time()
            # NEW: Step 50 physics steps at once (100ms simulation time)
            mujoco.mj_step(model, data, nstep=steps_per_frame)
            step_end_time = time.time()
            frame_time = step_end_time - step_start_time
            frame_times.append(frame_time)

            # Only collect joint-related qpos entries, not free joint positions
            joint_qpos = []
            for i in range(model.njnt):
                qpos_adr = int(model.joint(i).qposadr)
                joint_type = model.jnt_type[i]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    # Skip free joints (objects without constraints)
                    continue
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    continue
                else:
                    qpos_dim = 1
                joint_qpos.extend(data.qpos[qpos_adr : qpos_adr + qpos_dim])

            # Safety check: if no valid joints found, skip this scene
            if not joint_qpos:
                print(f"Warning: No valid joints found in {xml_file}, skipping joint analysis")
                # Set default values for joint-related metrics
                q_drift = 0.0
                q_drift_delta = 0.0
                max_joint_drift_body_name = "no_joints"
                max_joint_drift_body_name_delta = "no_joints"
                all_joint_drifts.extend([0.0])  # Add a default value

                # Still calculate FPS and simulation speed even without joints
                total_simulation_time = np.sum(frame_times)
                total_simulated_time = total_steps * frame_timestep
                avg_fps = total_steps / total_simulation_time
                min_fps = 1.0 / np.max(frame_times)
                max_fps = 1.0 / np.min(frame_times)
                fps_std = np.std(1.0 / frame_times)
                simulation_speed = total_simulated_time / total_simulation_time
                all_frame_times.extend(frame_times)

                # Set default values for object displacement
                obj_disp = 0.0
                max_obj_disp_name = "none"
                obj_disp_delta = 0.0
                max_obj_disp_name_delta = "none"
                all_object_displacements.extend([0.0])

                # Continue to next iteration
                continue

            joint_qpos_array = np.array(joint_qpos)
            joint_qs.append(joint_qpos_array)

            # Calculate delta from previous frame (skip first frame)
            if prev_joint_qpos is not None:
                joint_delta = joint_qpos_array - prev_joint_qpos
                joint_qs_delta.append(joint_delta)
            else:
                # For the first frame, use zero delta
                joint_qs_delta.append(np.zeros_like(joint_qpos_array))

            # Store current frame's joint positions for next iteration
            prev_joint_qpos = joint_qpos_array.copy()
            obj_pos = [np.copy(data.xpos[bid]) for bid in movable_bodies]
            obj_positions.append(obj_pos)

            KE = 0
            for i in range(model.nbody):
                vel = data.cvel[i]
                mass = model.body_mass[i]
                inertia_diag = model.body_inertia[i]
                omega = data.cvel[i][3:]
                KE += 0.5 * mass * np.sum(vel[:3] ** 2)  # Linear KE
                KE += 0.5 * np.sum(inertia_diag * vel[3:] ** 2)  # Rotational KE
            kinetic_energies.append(KE)

            contact_forces.append(np.sum(np.abs(data.cfrc_ext)))

        # Convert
        joint_qs = np.array(joint_qs)
        joint_qs_delta = np.array(joint_qs_delta)
        obj_positions = np.array(obj_positions)
        kinetic_energies = np.array(kinetic_energies)
        contact_forces = np.array(contact_forces)
        frame_times = np.array(frame_times)

        # Safety check: ensure we have valid data arrays
        if obj_positions.size == 0:
            print(f"Warning: No object positions recorded in {xml_file}")
            obj_positions = np.zeros((total_steps, 0, 3))  # Empty array with correct shape

        if joint_qs.size == 0:
            print(f"Warning: No joint data recorded in {xml_file}")
            joint_qs = np.zeros((total_steps, 0))  # Empty array with correct shape
            joint_qs_delta = np.zeros((total_steps, 0))  # Empty array with correct shape

        if kinetic_energies.size == 0:
            print(f"Warning: No kinetic energy data recorded in {xml_file}")
            kinetic_energies = np.zeros(total_steps)

        if contact_forces.size == 0:
            print(f"Warning: No contact force data recorded in {xml_file}")
            contact_forces = np.zeros(total_steps)

        if frame_times.size == 0:
            print(f"Warning: No frame time data recorded in {xml_file}")
            frame_times = np.zeros(total_steps)

        # NEW: Calculate frame rate metrics for 100ms frames
        total_simulation_time = np.sum(frame_times)
        total_simulated_time = total_steps * frame_timestep  # Total simulated time (200 seconds)

        # Safety check: avoid division by zero
        if total_simulation_time <= 0:
            print(f"Warning: Invalid simulation time in {xml_file}, using default values")
            avg_fps = 0.0
            min_fps = 0.0
            max_fps = 0.0
            fps_std = 0.0
            simulation_speed = 0.0
        else:
            # FPS calculation: frames per second (where each frame is 100ms of simulation)
            avg_fps = total_steps / total_simulation_time
            min_fps = (
                1.0 / np.max(frame_times) if np.max(frame_times) > 0 else 0.0
            )  # Worst case fps
            max_fps = 1.0 / np.min(frame_times) if np.min(frame_times) > 0 else 0.0  # Best case fps
            fps_std = (
                np.std(1.0 / frame_times) if np.all(frame_times > 0) else 0.0
            )  # Standard deviation of fps

            # NEW: Calculate simulation speed (real-time factor)
            simulation_speed = (
                total_simulated_time / total_simulation_time
            )  # How many times faster than real-time

        # Store frame times for overall analysis
        all_frame_times.extend(frame_times)

        # Safety check: only calculate joint drift if we have valid joint data
        if joint_qs.size > 0 and joint_qs.shape[1] > 0:
            qs_post_settle = joint_qs[settle_steps:]
            q_drift = np.max(np.abs(qs_post_settle - qs_post_settle[0]))
            q_drift_delta = np.max(np.abs(joint_qs_delta[-1]))
            # Find the object (body) that corresponds to the joint that drifted the most
            joint_drift_values = np.abs(qs_post_settle - qs_post_settle[0])
            # Find which joint had the maximum drift across all time steps
            max_per_joint = np.max(joint_drift_values, axis=0)
            max_joint_drift_idx = int(np.argmax(max_per_joint))
            max_joint_drift_idx_delta = int(np.argmax(np.abs(joint_qs_delta[-1])))

            # Map filtered qpos index to actual joint index, then get the body name
            max_joint_drift_body_name = "unknown"
            max_joint_drift_body_name_delta = "unknown"
            filtered_idx = 0
            for i in range(model.njnt):
                joint_type = model.jnt_type[i]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    continue  # Skip free joints

                qpos_dim = 4 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1

                if filtered_idx <= max_joint_drift_idx < filtered_idx + qpos_dim:
                    # Get the body that this joint belongs to
                    body_id = int(model.joint(i).bodyid[0])
                    max_joint_drift_body_name = model.body(body_id).name
                    if max_joint_drift_body_name_delta != "unknown":
                        break
                if filtered_idx <= max_joint_drift_idx_delta < filtered_idx + qpos_dim:
                    # Get the body that this joint belongs to
                    body_id = int(model.joint(i).bodyid[0])
                    max_joint_drift_body_name_delta = model.body(body_id).name
                    if max_joint_drift_body_name != "unknown":
                        break
                filtered_idx += qpos_dim
            # Collect all joint drifts for histogram
            all_joint_drifts.extend(np.abs(qs_post_settle - qs_post_settle[0]).flatten())
        else:
            # No valid joints, set default values
            q_drift = 0.0
            q_drift_delta = 0.0
            max_joint_drift_body_name = "no_joints"
            max_joint_drift_body_name_delta = "no_joints"
            all_joint_drifts.extend([0.0])

        if movable_bodies:
            obj_displacements = [
                np.linalg.norm(obj_positions[-1][i] - obj_positions[settle_steps][i])
                for i in range(len(movable_bodies))
            ]
            obj_disp = np.max(obj_displacements)
            max_obj_disp_idx = int(np.argmax(obj_displacements))
            max_obj_disp_name = model.body(movable_bodies[max_obj_disp_idx]).name
            # Collect all object displacements for histogram
            all_object_displacements.extend(obj_displacements)

            # NEW: Calculate object displacement delta
            # pick max from all objects displacements delta
            obj_displacement_delta = [
                np.linalg.norm(obj_positions[-1][i] - obj_positions[-2][i])
                for i in range(len(movable_bodies))
            ]
            obj_displacement_delta = np.array(obj_displacement_delta)
            obj_disp_delta = np.max(obj_displacement_delta)
            max_obj_disp_idx_delta = int(np.argmax(obj_displacement_delta))
            max_obj_disp_name_delta = model.body(movable_bodies[max_obj_disp_idx_delta]).name

        else:
            obj_disp = 0.0
            max_obj_disp_name = "none"
            obj_disp_delta = 0.0
            max_obj_disp_name_delta = "none"
            # Add default values to maintain consistent data structure
            all_object_displacements.extend([0.0])

        contact_variance = np.std(contact_forces[settle_steps:])
        kinetic_energy_final = kinetic_energies[-1]

        # Safety check: ensure we have valid data for variance calculation
        if len(contact_forces[settle_steps:]) < 2:
            print(
                f"Warning: Insufficient contact force data for variance calculation in {xml_file}"
            )
            contact_variance = 0.0

        if len(kinetic_energies) == 0:
            print(f"Warning: No kinetic energy data in {xml_file}")
            kinetic_energy_final = 0.0

        results.append(
            {
                "env": xml_file,
                "joint_drift": q_drift,
                "max_joint_drift_body_name": max_joint_drift_body_name,
                "object_displacement": obj_disp,  # object drifts or rolls
                "max_object_disp_name": max_obj_disp_name,
                "max_obj_disp_delta": obj_disp_delta,
                "max_obj_disp_delta_name": max_obj_disp_name_delta,
                "max_joint_drift_delta": q_drift_delta,
                "max_joint_drift_delta_name": max_joint_drift_body_name_delta,
                "contact_force_std": contact_variance,  # jiterring, instability in contacts
                "final_kinetic_energy": kinetic_energy_final,  # things still moving and not settled
                "avg_fps": avg_fps,  # Average frames per second (100ms simulation frames)
                "min_fps": min_fps,  # Minimum frames per second (worst case)
                "max_fps": max_fps,  # Maximum frames per second (best case)
                "fps_std": fps_std,  # Standard deviation of fps
                "simulation_speed": simulation_speed,  # Real-time factor (how many times faster than real-time)
                "total_simulation_time": total_simulation_time,  # Total wall-clock time for all frames
                "total_simulated_time": total_simulated_time,  # Total simulated time (200 seconds)
                "avg_frame_time_ms": (
                    np.mean(frame_times) * 1000 if frame_times.size > 0 else 0.0
                ),  # Average frame time in milliseconds
                "frame_timestep_ms": frame_timestep
                * 1000,  # Simulation time per frame in milliseconds
            }
        )
    except Exception as e:
        print(f"Error in {xml_file}: {e}")


# Save and display
df = pd.DataFrame(results)

# Safety check: ensure we have results to analyze
if len(results) == 0:
    print(
        "Warning: No results to analyze. Check if all scenes failed or if there are no valid XML files."
    )
    exit()

df.to_csv(os.path.join(output_dir, "env_stability_metrics.csv"), index=False)

# Safety check: ensure we have valid data for summary statistics
if len(df) > 0:
    # Filter out infinite and NaN values for summary statistics
    df_clean = df.replace([np.inf, -np.inf], np.nan)

    print("=== SUMMARY STATISTICS ===")
    print(df_clean.describe())

    if "object_displacement" in df_clean.columns and df_clean["object_displacement"].notna().any():
        print("\n=== TOP 10 SCENES BY OBJECT DISPLACEMENT ===")
        print(df_clean.sort_values("object_displacement", ascending=False).head(10))
    else:
        print("Warning: No valid object displacement data for ranking")

    if "joint_drift" in df_clean.columns and df_clean["joint_drift"].notna().any():
        print("\n=== TOP 10 SCENES BY JOINT DRIFT ===")
        print(df_clean.sort_values("joint_drift", ascending=False).head(10))
    else:
        print("Warning: No valid joint drift data for ranking")
else:
    print("Warning: No data available for summary statistics")

# Print frame rate summary
# each frame is 100ms steps
print("\n=== FRAME RATE BENCHMARKING RESULTS (100ms steps) ===")

if len(df) > 0 and "avg_fps" in df.columns and df["avg_fps"].notna().any():
    valid_fps = df["avg_fps"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_fps) > 0:
        print(f"Average FPS across all scenes: {valid_fps.mean():.2f} ± {valid_fps.std():.2f}")
        print(
            f"Best performing scene: {df.loc[df['avg_fps'].idxmax(), 'env']} ({df['avg_fps'].max():.2f} FPS)"
        )
        print(
            f"Worst performing scene: {df.loc[df['avg_fps'].idxmin(), 'env']} ({df['avg_fps'].min():.2f} FPS)"
        )
    else:
        print("Warning: No valid FPS data for summary")
else:
    print("Warning: FPS data not available")

if len(df) > 0 and "avg_frame_time_ms" in df.columns and df["avg_frame_time_ms"].notna().any():
    valid_frame_times = df["avg_frame_time_ms"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_frame_times) > 0:
        print(
            f"Average frame time: {valid_frame_times.mean():.2f} ± {valid_frame_times.std():.2f} ms"
        )
    else:
        print("Warning: No valid frame time data for summary")
else:
    print("Warning: Frame time data not available")

print(f"Simulation time per frame: {frame_timestep * 1000:.0f} ms")

if len(df) > 0 and "simulation_speed" in df.columns and df["simulation_speed"].notna().any():
    valid_speed = df["simulation_speed"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_speed) > 0:
        print(
            f"Average simulation speed: {valid_speed.mean():.2f}x real-time ± {valid_speed.std():.2f}"
        )
        print(f"Best simulation speed: {valid_speed.max():.2f}x real-time")
        print(f"Worst simulation speed: {valid_speed.min():.2f}x real-time")
    else:
        print("Warning: No valid simulation speed data for summary")
else:
    print("Warning: Simulation speed data not available")

# Plot
metrics = [
    "joint_drift",
    "object_displacement",
    "max_joint_drift_delta",
    "max_obj_disp_delta",
    "contact_force_std",
    "final_kinetic_energy",
]
for metric in metrics:
    # Safety check: ensure the metric column exists and has valid data
    if metric in df.columns and df[metric].notna().any():
        plt.figure()
        # Filter out infinite and NaN values
        valid_data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            plt.hist(valid_data, bins=20, edgecolor="black")
            plt.title(f"Max {metric.replace('_', ' ').title()} Across All Scenes")
            plt.xlabel(metric)
            plt.ylabel("Count")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_hist.png"))
            plt.show()
        else:
            print(f"Warning: No valid data for {metric} plotting")
    else:
        print(f"Warning: Metric {metric} not available or has no valid data")

# Add frame rate plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
if "avg_fps" in df.columns and df["avg_fps"].notna().any():
    valid_fps = df["avg_fps"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_fps) > 0:
        plt.hist(valid_fps, bins=20, edgecolor="black")
        plt.title("Average FPS Distribution Across Scenes")
        plt.xlabel("Average FPS")
        plt.ylabel("Count")
        plt.grid(True)
    else:
        plt.text(
            0.5, 0.5, "No valid FPS data", ha="center", va="center", transform=plt.gca().transAxes
        )
else:
    plt.text(
        0.5, 0.5, "FPS data not available", ha="center", va="center", transform=plt.gca().transAxes
    )

plt.subplot(2, 2, 2)
if "avg_frame_time_ms" in df.columns and df["avg_frame_time_ms"].notna().any():
    valid_frame_times = df["avg_frame_time_ms"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_frame_times) > 0:
        plt.hist(valid_frame_times, bins=20, edgecolor="black")
        plt.title("Average Frame Time Distribution")
        plt.xlabel("Frame Time (100 ms per step)")
        plt.ylabel("Count")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "No valid frame time data",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
else:
    plt.text(
        0.5,
        0.5,
        "Frame time data not available",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

plt.subplot(2, 2, 3)
if "avg_fps" in df.columns and "object_displacement" in df.columns:
    valid_data = df[["avg_fps", "object_displacement"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_data) > 0:
        plt.scatter(valid_data["avg_fps"], valid_data["object_displacement"], alpha=0.6)
        plt.title("FPS vs Object Displacement")
        plt.xlabel("Average FPS")
        plt.ylabel("Object Displacement")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "No valid data for scatter plot",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
else:
    plt.text(
        0.5,
        0.5,
        "Required data not available",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

plt.subplot(2, 2, 4)
if "simulation_speed" in df.columns and "joint_drift" in df.columns:
    valid_data = df[["simulation_speed", "joint_drift"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_data) > 0:
        plt.scatter(valid_data["simulation_speed"], valid_data["joint_drift"], alpha=0.6)
        plt.title("Simulation Speed vs Joint Drift")
        plt.xlabel("Simulation Speed (x real-time)")
        plt.ylabel("Joint Drift")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "No valid data for scatter plot",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
else:
    plt.text(
        0.5,
        0.5,
        "Required data not available",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frame_rate_analysis.png"))
plt.show()

# Add simulation speed plot
plt.figure(figsize=(10, 6))
if "simulation_speed" in df.columns and df["simulation_speed"].notna().any():
    valid_speed = df["simulation_speed"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_speed) > 0:
        plt.hist(valid_speed, bins=20, edgecolor="black")
        plt.title("Simulation Speed Distribution (100ms steps)")
        plt.xlabel("Simulation Speed (x real-time)")
        plt.ylabel("Count")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "No valid simulation speed data",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
else:
    plt.text(
        0.5,
        0.5,
        "Simulation speed data not available",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "simulation_speed_hist.png"))
plt.show()

# Plot histograms of all individual object displacements and joint drifts
if all_object_displacements:
    # Filter out invalid values
    valid_obj_disps = [x for x in all_object_displacements if np.isfinite(x)]
    if valid_obj_disps:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_obj_disps, bins=50, edgecolor="black", alpha=0.7)
        plt.title("All Object Displacements Across All Scenes")
        plt.xlabel("Object Displacement")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_object_displacements_hist.png"))
        plt.show()

        print(f"Total object displacements recorded: {len(valid_obj_disps)}")
        print(f"Mean object displacement: {np.mean(valid_obj_disps):.6f}")
        print(f"Std object displacement: {np.std(valid_obj_disps):.6f}")
        print(f"Max object displacement: {np.max(valid_obj_disps):.6f}")
    else:
        print("Warning: No valid object displacement data for plotting")

if all_joint_drifts:
    # Filter out invalid values
    valid_joint_drifts = [x for x in all_joint_drifts if np.isfinite(x)]
    if valid_joint_drifts:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_joint_drifts, bins=50, edgecolor="black", alpha=0.7)
        plt.title("All Joint Drifts Across All Scenes")
        plt.xlabel("Joint Drift")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_joint_drifts_hist.png"))
        plt.show()

        print(f"Total joint drifts recorded: {len(valid_joint_drifts)}")
        print(f"Mean joint drift: {np.mean(valid_joint_drifts):.6f}")
        print(f"Std joint drift: {np.std(valid_joint_drifts):.6f}")
        print(f"Max joint drift: {np.max(valid_joint_drifts):.6f}")
    else:
        print("Warning: No valid joint drift data for plotting")

# Plot frame time distribution across all scenes
if all_frame_times:
    # Filter out invalid values
    valid_frame_times = [x for x in all_frame_times if np.isfinite(x) and x > 0]
    if valid_frame_times:
        plt.figure(figsize=(10, 6))
        plt.hist(np.array(valid_frame_times) * 1000, bins=50, edgecolor="black", alpha=0.7)
        plt.title("All Frame Times Across All Scenes")
        plt.xlabel("Frame Time (100 ms per step)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_frame_times_hist.png"))
        plt.show()

        print("\n=== DETAILED FRAME TIME ANALYSIS (100ms steps) ===")
        print(f"Total frames recorded: {len(valid_frame_times)}")
        print(f"Simulation time per frame: {frame_timestep * 1000:.0f} ms")
        print(f"Mean wall-clock frame time: {np.mean(valid_frame_times) * 1000:.2f} ms")
        print(f"Std wall-clock frame time: {np.std(valid_frame_times) * 1000:.2f} ms")
        print(f"Min wall-clock frame time: {np.min(valid_frame_times) * 1000:.2f} ms")
        print(f"Max wall-clock frame time: {np.max(valid_frame_times) * 1000:.2f} ms")
        print(
            f"95th percentile wall-clock frame time: {np.percentile(valid_frame_times, 95) * 1000:.2f} ms"
        )
        print(
            f"99th percentile wall-clock frame time: {np.percentile(valid_frame_times, 99) * 1000:.2f} ms"
        )
        print(f"Average real-time factor: {np.mean(valid_frame_times) / frame_timestep:.2f}x")
    else:
        print("Warning: No valid frame time data for analysis")
else:
    print("Warning: No frame time data available")
