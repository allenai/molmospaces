import argparse
import datetime
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gripper_teleop import GripperTeleopController
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from molmo_spaces.editor.thor_model_editor import ThorMjModelEditor
from molmo_spaces.env.arena.arena_utils import load_env_with_objects
from molmo_spaces.utils.profiler_utils import Profiler
from molmo_spaces.utils.scene_maps import iTHORMap

PROFILER = Profiler()


iTHOR_DATE = "082625"
DATETIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
GRIPPER_TYPE = "RUM"  # "robotiq" # "franka" #"franka" # "RUM"
# GRIPPER_TYPE = "franka" # "franka" #"franka" # "RUM"
SAVE_DIR_PATH = f"debug/mocap_data/iTHOR_{GRIPPER_TYPE}_gripper_{iTHOR_DATE}_{DATETIME}/"

# Performance optimization: Batch size for parallel processing
# Note: This script now uses multiprocessing instead of multithreading for better performance
# and to avoid OpenGL context conflicts when using GPU rendering
BATCH_SIZE = 20  # Balanced batch size for good throughput
PHYSICS_STEPS_PER_FRAME = 50  # Balanced physics steps for accuracy vs speed
SKIP_SAVE_TRAJECTORIES = False  # Skip saving trajectory data for speed
SKIP_MAP_GENERATION = False  # Skip map generation if already exists

# Gripper configuration
GRIPPER_LENGTH = 0.125  # 0.15  # Length of the gripper in meters
if GRIPPER_TYPE == "RUM":
    GRIPPER_LENGTH = 0.125
elif GRIPPER_TYPE == "franka":
    GRIPPER_LENGTH = 0.125

# Object type categories for statistics
OBJECT_CATEGORIES = ["cabinet", "drawer", "oven", "dishwasher", "shower_door"]


def categorize_handle_by_object_type(handle_name) -> str:
    """
    Categorize a handle by its object type based on the handle name.

    Args:
        handle_name: Name of the handle (e.g., "Cabinet_12", "Drawer_5", etc.)

    Returns:
        str: Object category ("cabinet", "drawer", "oven", "dishwasher", "shower_door", or "other")
    """
    handle_name_lower = handle_name.lower()

    # Check for specific object types in the handle name
    if "cabinet" in handle_name_lower:
        return "cabinet"
    elif "drawer" in handle_name_lower:
        return "drawer"
    elif "oven" in handle_name_lower:
        return "oven"
    elif "dishwasher" in handle_name_lower:
        return "dishwasher"
    elif "shower" in handle_name_lower and "door" in handle_name_lower:
        return "shower_door"
    elif "showerdoor" in handle_name_lower:
        return "shower_door"
    else:
        return "other"


def visualize_path(
    path,
    title="Gripper Base Path Visualization",
    save_path=None,
    joint_position=None,
    show_finger_center=True,
):
    """
    Comprehensive visualization of the gripper base path and finger center arc.

    Args:
        path: Dictionary with 'mocap_pos' and 'mocap_quat' lists (representing gripper base positions)
        title: Title for the plot
        save_path: Optional path to save the plot
        joint_position: Optional joint position to visualize
        show_finger_center: If True, also show the finger center arc for reference
    """
    if not path or "mocap_pos" not in path or len(path["mocap_pos"]) == 0:
        print("No path data to visualize")
        return

    # Convert to numpy arrays for easier manipulation
    # Handle case where positions might be tuples or lists
    if len(path["mocap_pos"]) > 0:
        # Convert each position to numpy array if it isn't already
        positions = np.array(
            [np.array(pos) if not isinstance(pos, np.ndarray) else pos for pos in path["mocap_pos"]]
        )
    else:
        positions = np.array([])

    # Handle quaternions similarly
    if "mocap_quat" in path and len(path["mocap_quat"]) > 0:
        quaternions = np.array(
            [
                np.array(quat) if not isinstance(quat, np.ndarray) else quat
                for quat in path["mocap_quat"]
            ]
        )
    else:
        quaternions = None

    print(f"Visualizing gripper base path with {len(positions)} points")
    print(
        f"Base position range: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
        f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
        f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]"
    )

    # Calculate finger center positions for reference (if quaternions available)
    finger_center_positions = None
    if show_finger_center and quaternions is not None and len(quaternions) > 0:
        finger_center_positions = []
        for i, (base_pos, quat) in enumerate(zip(positions, quaternions)):
            # Calculate finger center position by offsetting from base
            rot_matrix = R.from_quat(quat, scalar_first=True).as_matrix()
            gripper_z_axis = rot_matrix[:, 2]  # Z-axis points towards handle
            finger_offset = 0.5 * GRIPPER_LENGTH * gripper_z_axis
            finger_pos = base_pos + finger_offset
            finger_center_positions.append(finger_pos)
        finger_center_positions = np.array(finger_center_positions)
        print(
            f"Finger center position range: X[{finger_center_positions[:, 0].min():.3f}, {finger_center_positions[:, 0].max():.3f}], "
            f"Y[{finger_center_positions[:, 1].min():.3f}, {finger_center_positions[:, 1].max():.3f}], "
            f"Z[{finger_center_positions[:, 2].min():.3f}, {finger_center_positions[:, 2].max():.3f}]"
        )

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 2D Top-down view (X-Y plane)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        positions[:, 0],
        positions[:, 1],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Gripper Base Path",
    )
    ax1.scatter(
        positions[0, 0], positions[0, 1], color="green", s=100, label="Base Start", zorder=5
    )
    ax1.scatter(positions[-1, 0], positions[-1, 1], color="red", s=100, label="Base End", zorder=5)

    # Add finger center path if available
    if finger_center_positions is not None:
        ax1.plot(
            finger_center_positions[:, 0],
            finger_center_positions[:, 1],
            "r--s",
            linewidth=1,
            markersize=3,
            alpha=0.7,
            label="Finger Center Arc",
        )
        ax1.scatter(
            finger_center_positions[0, 0],
            finger_center_positions[0, 1],
            color="darkgreen",
            s=80,
            marker="s",
            label="Finger Start",
            zorder=5,
        )
        ax1.scatter(
            finger_center_positions[-1, 0],
            finger_center_positions[-1, 1],
            color="darkred",
            s=80,
            marker="s",
            label="Finger End",
            zorder=5,
        )

    # Add joint position if provided
    if joint_position is not None:
        ax1.scatter(
            joint_position[0],
            joint_position[1],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position (finger center for reference)
        if finger_center_positions is not None:
            ax1.plot(
                [joint_position[0], finger_center_positions[0, 0]],
                [joint_position[1], finger_center_positions[0, 1]],
                "k--",
                alpha=0.5,
                linewidth=1,
                label="Joint-Finger Offset",
            )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Top-down View (X-Y)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # 2D Side view (X-Z plane)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(positions[:, 0], positions[:, 2], "r-o", linewidth=2, markersize=4, label="Path")
    ax2.scatter(positions[0, 0], positions[0, 2], color="green", s=100, label="Start", zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 2], color="red", s=100, label="End", zorder=5)

    # Add joint position if provided
    if joint_position is not None:
        ax2.scatter(
            joint_position[0],
            joint_position[2],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position
        ax2.plot(
            [joint_position[0], positions[0, 0]],
            [joint_position[2], positions[0, 2]],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Joint-Start Offset",
        )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Side View (X-Z)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # 2D Front view (Y-Z plane)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(positions[:, 1], positions[:, 2], "g-o", linewidth=2, markersize=4, label="Path")
    ax3.scatter(positions[0, 1], positions[0, 2], color="green", s=100, label="Start", zorder=5)
    ax3.scatter(positions[-1, 1], positions[-1, 2], color="red", s=100, label="End", zorder=5)

    # Add joint position if provided
    if joint_position is not None:
        ax3.scatter(
            joint_position[1],
            joint_position[2],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position
        ax3.plot(
            [joint_position[1], positions[0, 1]],
            [joint_position[2], positions[0, 2]],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Joint-Start Offset",
        )

    ax3.set_xlabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("Front View (Y-Z)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    # 3D view
    ax4 = plt.subplot(2, 2, 4, projection="3d")
    ax4.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Path",
    )
    ax4.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        color="green",
        s=100,
        label="Start",
        zorder=5,
    )
    ax4.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        color="red",
        s=100,
        label="End",
        zorder=5,
    )

    # Add joint position if provided
    if joint_position is not None:
        ax4.scatter(
            joint_position[0],
            joint_position[1],
            joint_position[2],
            color="purple",
            s=200,
            marker="*",
            label="Joint",
            zorder=6,
        )
        # Draw a line from joint to start position to show the offset
        ax4.plot(
            [joint_position[0], positions[0, 0]],
            [joint_position[1], positions[0, 1]],
            [joint_position[2], positions[0, 2]],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Joint-Start Offset",
        )

    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")
    ax4.set_title("3D View")

    # Add legend for orientation arrows
    from matplotlib.patches import FancyArrowPatch

    legend_elements = [
        FancyArrowPatch((0, 0), (0.1, 0), color="red", linewidth=2, label="Gripper X-axis"),
        FancyArrowPatch((0, 0), (0.1, 0), color="green", linewidth=2, label="Gripper Y-axis"),
        FancyArrowPatch((0, 0), (0.1, 0), color="blue", linewidth=2, label="Gripper Z-axis"),
    ]
    ax4.legend(handles=legend_elements, loc="upper right")

    # Add orientation arrows if quaternions are available
    if quaternions is not None and len(quaternions) > 0:
        # Show orientation at regular intervals along the path
        n_arrows = min(8, len(positions))  # Show up to 8 arrows
        arrow_indices = np.linspace(0, len(positions) - 1, n_arrows, dtype=int)

        for i, idx in enumerate(arrow_indices):
            if idx < len(quaternions):
                # Convert quaternion to rotation matrix
                rot_matrix = R.from_quat(quaternions[idx], scalar_first=True).as_matrix()

                # Create arrows representing gripper orientation (X, Y, Z axes)
                arrow_length = 0.15
                origin = positions[idx]

                # X-axis (red), Y-axis (green), Z-axis (blue) of gripper
                axes = [
                    (rot_matrix[:, 0], "red", "X"),  # X-axis
                    (rot_matrix[:, 1], "green", "Y"),  # Y-axis
                    (rot_matrix[:, 2], "blue", "Z"),  # Z-axis
                ]

                for axis_direction, color, _axis_name in axes:
                    ax4.quiver(
                        origin[0],
                        origin[1],
                        origin[2],
                        axis_direction[0],
                        axis_direction[1],
                        axis_direction[2],
                        length=arrow_length,
                        color=color,
                        alpha=0.8,
                        linewidth=2,
                    )

                # Add text label for the point
                ax4.text(
                    origin[0],
                    origin[1],
                    origin[2],
                    f"P{i}",
                    fontsize=8,
                    color="black",
                    weight="bold",
                )

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Path visualization saved to {save_path}")

    plt.show()

    # Print path statistics
    print("\nPath Statistics:")
    print("Gripper Base Path:")
    print(f"  Total distance: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.3f} m")
    print(f"  Number of waypoints: {len(positions)}")
    print(f"  Start position: {positions[0]}")
    print(f"  End position: {positions[-1]}")

    if finger_center_positions is not None:
        finger_distance = np.sum(np.linalg.norm(np.diff(finger_center_positions, axis=0), axis=1))
        print("Finger Center Arc:")
        print(f"  Total distance: {finger_distance:.3f} m")
        print(f"  Start position: {finger_center_positions[0]}")
        print(f"  End position: {finger_center_positions[-1]}")

    if joint_position is not None:
        print(f"Joint position: {joint_position}")
        print(
            f"Distance from joint to base start: {np.linalg.norm(positions[0] - joint_position):.3f} m"
        )
        print(
            f"Distance from joint to base end: {np.linalg.norm(positions[-1] - joint_position):.3f} m"
        )
        if finger_center_positions is not None:
            print(
                f"Distance from joint to finger start: {np.linalg.norm(finger_center_positions[0] - joint_position):.3f} m"
            )
            print(
                f"Distance from joint to finger end: {np.linalg.norm(finger_center_positions[-1] - joint_position):.3f} m"
            )

    # Print orientation statistics if available
    if quaternions is not None and len(quaternions) > 0:
        print(f"Start orientation (quaternion): {quaternions[0]}")
        print(f"End orientation (quaternion): {quaternions[-1]}")

        # Calculate orientation change
        start_rot = R.from_quat(quaternions[0], scalar_first=True)
        end_rot = R.from_quat(quaternions[-1], scalar_first=True)
        orientation_diff = start_rot.inv() * end_rot
        angle_diff = orientation_diff.magnitude()
        print(f"Total orientation change: {np.degrees(angle_diff):.1f}°")

    return fig


def regenerate_map(ithor_scene_path, process_id=None, use_gpu=False) -> None:
    # Skip map generation if already exists and SKIP_MAP_GENERATION is True
    if SKIP_MAP_GENERATION:
        if "mesh" in ithor_scene_path:
            map_path = ithor_scene_path.replace("_mesh.xml", "_map.png")
        else:
            map_path = ithor_scene_path.replace(".xml", "_map.png")
        if os.path.exists(map_path):
            print(f"Map already exists, skipping generation: {map_path}")
            return

    print(f"Generating map for {ithor_scene_path} (process {process_id})")

    # For CPU-only systems, use device_id=None for all processes
    # For GPU systems, use different device IDs for different processes to avoid OpenGL context conflicts
    device_id = None

    if use_gpu:
        # Check if GPU is available
        try:
            import torch

            if torch.cuda.is_available():
                # GPU available - use different device IDs for different processes
                if process_id is not None:
                    # Use process_id % 4 to limit device IDs (assuming max 4 GPUs)
                    device_id = process_id % 4 if process_id < 4 else None
            else:
                # No GPU available - use CPU rendering for all processes
                device_id = None
                print("Warning: GPU requested but not available, using CPU rendering")
        except ImportError:
            # PyTorch not available - use CPU rendering for all processes
            device_id = None
            print("Warning: GPU requested but PyTorch not available, using CPU rendering")
    else:
        # CPU-only rendering
        device_id = None

    try:
        thormap = iTHORMap.from_mj_model_path(
            ithor_scene_path, px_per_m=200, agent_radius=0.20, device_id=device_id
        )
        if "mesh" in ithor_scene_path:
            thormap.save(ithor_scene_path.replace("_mesh.xml", "_map.png"))
        else:
            thormap.save(ithor_scene_path.replace(".xml", "_map.png"))
        print(f"Map generated successfully for {ithor_scene_path}")
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def get_all_handles(model, data):
    # check name includes "handle" in body
    all_handles = []
    for i in range(model.nbody):
        if "handle" in model.body(i).name.lower():
            parent_id = int(model.body_parentid[i])
            parent_name = model.body(parent_id).name
            if parent_id == 0:  # 0th is world
                continue
            root_body = int(model.body_rootid[i])
            model.body(root_body).name

            # find handle geom for its position
            geom_id = int(model.body_geomadr[i])
            geom_num = int(model.body_geomnum[i])
            for j in range(geom_num):
                geom_group = int(model.geom(geom_id + j).group)
                if geom_group == 0:  # visual geom
                    geom_id = geom_id + j
                    break
            body_id = int(model.body_bvhadr[i])
            handle_bounding_box = model.bvh_aabb[body_id]  # (center, size)  6 values
            handle_bounding_box[:3]

            # handle info
            data.geom_xpos[geom_id]
            data.geom_xmat[geom_id]

            all_handles.append(
                {
                    "name": parent_name,
                    "position": data.geom_xpos[geom_id],  # data.xpos[model.body(i).id],
                    "orientation": data.geom_xmat[geom_id],  # data.xmat[model.body(i).id],
                    "handle_id": parent_id,
                    "geom_id": geom_id,
                    "size": handle_bounding_box[3:],
                }
            )
    return all_handles


def get_gripper_pose_based_on_handle_pose(handle_info, ithor_map, joint_info=None):
    handle_position = handle_info["position"]
    handle_orientation = handle_info["orientation"].reshape(3, 3)  # 9 matrix
    handle_info["size"]

    # get handle axis information from orientation
    # GEOM Z Axis is always alongthe handle. X-axis is up. Y is forward
    # <--- this may not be always true...
    # <--- setting 1.0 threshold and switch to using all free points to cover some cases...Floorplan 27

    if False:  #  joint_info is not None:
        joint_max_range = joint_info["max_range"]
        joint_sign = np.sign(joint_max_range)
        joint_position = joint_info["joint_position"]  # from body frame
        joint_axis = joint_info["joint_axis"]

        joint_body_orientation = joint_info["joint_body_orientation"]
        joint_body_position = joint_info["joint_body_position"]
        world_joint_position = joint_body_orientation @ joint_position + joint_body_position

        z_axis = joint_body_orientation @ joint_axis
        # z_axis = joint_axis

        # this is the axis that is parallel to the handle
        x_axis = handle_position - world_joint_position
        argmax_x_axis = np.argmax(np.abs(x_axis))
        x_axis = np.zeros(3)
        x_axis[argmax_x_axis] = 1
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(x_axis, z_axis)
        y_axis /= np.linalg.norm(y_axis)

        # position along the handle y axis
        free_position = handle_position - joint_sign * y_axis * 0.5
        free_position[2] = handle_position[2]

    else:
        y_axis = handle_orientation @ np.array([0, 1, 0])

        # Pick a free point closest to the handle position
        all_free_points = ithor_map.get_free_points()

        # filter free points wihtin 1m
        free_points = [
            p for p in all_free_points if np.linalg.norm(p[:2] - handle_position[:2]) < 2.0
        ]
        # free_points = [p for p in free_points if np.linalg.norm(p[:2] - handle_position[:2]) > 0.5]

        closest_point = min(free_points, key=lambda x: np.linalg.norm(x[:2] - handle_position[:2]))
        free_position = closest_point
        free_position[2] = handle_position[2]

        # pick a closest free point along the handle y axis but is also free in the map
        # free_points_along_y_axis = [
        #    p
        #    for p in free_points
        #    if np.abs(np.dot(p[:2] - handle_position[:2], y_axis[:2]))
        #    > 0 #0.25  # close to 1 means parallel
        # ]
        ## assert len(free_points_along_y_axis) > 0, "No free point found along the handle y axis"
        # if len(free_points_along_y_axis) == 0:
        #    print("No free point found along the handle y axis, using all free points")
        #    free_points_along_y_axis = free_points
        #    assert len(free_points_along_y_axis) > 0, "No free point found"
    #
    # free_position = min(
    #    free_points_along_y_axis, key=lambda x: np.linalg.norm(x[:2] - handle_position[:2])
    # )
    # free_position[2] = handle_position[2]

    debug_plot = False
    if debug_plot:
        # DEBUG:
        # plot the handle and the free point
        # plat ithor map
        occupancy_map = ithor_map.occupancy_map
        handle_position_pix = ithor_map.pos_m_to_px([handle_position[0], handle_position[1], 0])
        free_position_pix = ithor_map.pos_m_to_px([free_position[0], free_position[1], 0])
        plt.figure()
        plt.imshow(occupancy_map, cmap="gray")
        plt.scatter(handle_position_pix[1], handle_position_pix[0], color="red", label="Handle")
        plt.scatter(free_position_pix[1], free_position_pix[0], color="blue", label="Free Point")
        plt.legend()
        plt.show()

    # orientation towards handle
    # 1. Direction vector
    to_handle_dist = handle_position - free_position
    to_handle_dir = to_handle_dist / np.linalg.norm(to_handle_dist)

    # 1. Get handle's world orientation as rotation matrix
    z_axis = to_handle_dir

    # 2. Extract handle's local X-axis in global frame
    # Always align GEOM Z-axis as up_reference
    # GEOM Z Axis is always alongthe handle. X-axis is up. Y is forward
    handle_x_axis_global = handle_orientation @ np.array([0, 0, -1])

    # 3. Use the handle's X-axis as up reference for gripper alignment
    up_reference = handle_x_axis_global  # or [0, 0, -1]
    if up_reference[2] > 0:
        up_reference *= -1  # this should always be negative

    # 4. Compute +X as orthogonal to +Z and 'up_reference'
    x_axis = np.cross(up_reference, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # Degenerate case: handle direction is vertical
        up_reference = np.array([1, 0, 0])  # use another axis
        x_axis = np.cross(up_reference, z_axis)

    x_axis /= np.linalg.norm(x_axis)

    # 5. Recompute true +Y to complete orthonormal frame
    y_axis = np.cross(z_axis, x_axis)

    # 6. Form rotation matrix with columns: [X, Y, Z]
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # 7. Convert to quaternion
    quat = R.from_matrix(rot_matrix).as_quat(scalar_first=True)  # [x, y, z, w]
    free_quaternion = quat

    # get the gripper pose based on the handle pose
    gripper_pose = {"position": free_position, "quaternion": free_quaternion}
    return gripper_pose


def add_gripper_to_scene(
    scene_path, gripper_pose, gripper_path="assets/robots/rum_gripper/model.xml"
):
    if GRIPPER_TYPE == "RUM":
        gripper_path = "assets/robots/rum_gripper/model.xml"
    elif GRIPPER_TYPE == "franka":
        gripper_path = "assets/robots/franka_fr3/fr3_gripper.xml"
    elif GRIPPER_TYPE == "robotiq":
        gripper_path = "assets/robots/robotiq_2f85/2f85.xml"
    else:
        raise ValueError(f"Invalid gripper type: {GRIPPER_TYPE}")

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
    robot_xml_path = scene_path.replace(".xml", f"_{GRIPPER_TYPE}_robot.xml")
    editor.save_xml(save_path=robot_xml_path)
    return robot_xml_path


def step_circular_path(
    current_pos,
    current_quat,
    handle_info,
    joint_info,
    n_waypoints=10,
    gripper_length=GRIPPER_LENGTH,
):
    def rotation_matrix_from_axis_angle(axis, angle):
        """Create rotation matrix from axis and angle using scipy's reliable implementation"""
        axis = axis / np.linalg.norm(axis)
        # Use scipy's implementation which is more reliable
        from scipy.spatial.transform import Rotation as R_scipy

        R_matrix = R_scipy.from_rotvec(axis * angle).as_matrix()
        return R_matrix

    ## extract joint info
    joint_body_position = joint_info["joint_body_position"]
    joint_axis_local = joint_info["joint_axis"]

    # Convert joint axis from local body frame to global frame
    # Get the body's orientation matrix
    body_orientation = joint_info["joint_body_orientation"]
    joint_axis = body_orientation @ joint_axis_local

    # joint position is in the joint frame, so we need to convert it to the world frame
    # joint_position = joint_info["joint_position"] + need to convert to world frame by multiplying body orientation
    joint_position = body_orientation @ joint_info["joint_position"] + joint_body_position

    # Use gripper position to find the arc that gripper follows
    handle_position = handle_info["position"]  # current_pos

    # get offset from joint to gripper
    handle_offset = handle_position - joint_position

    if np.abs(joint_axis[2]) > 0.9:
        # if rotating along the global z axis, make the height same
        joint_position[2] = handle_position[2]

        # For Z-axis rotation, we need to ensure the rotation is in the XY plane
        # The gripper offset should be in the XY plane only
        handle_offset_xy = handle_offset.copy()
        handle_offset_xy[2] = 0  # Zero out Z component for XY plane rotation
        handle_offset = handle_offset_xy
    current_joint_angle = joint_info["joint_pos"]
    joint_range = joint_info["joint_range"]

    # pick non-zero angle as max
    max_joint_angle = joint_range[1]
    if max_joint_angle == 0:
        max_joint_angle = joint_range[0]

    # Calculate relative angles (change from current to max)
    angle_change = max_joint_angle - current_joint_angle
    angles = np.linspace(0, angle_change, n_waypoints + 1)

    # print(f"  Angle calculation:")
    # print(f"    Current joint angle: {current_joint_angle:.3f} rad")
    # print(f"    Max joint angle: {max_joint_angle:.3f} rad")
    # print(f"    Angle change: {angle_change:.3f} rad")
    # print(f"    Angles array: {angles}")
    # print(f"    Number of waypoints: {n_waypoints}")

    # NEW: Calculate finger center circular arc and derive gripper base positions
    # We want the finger center to follow the circular arc around the joint
    # The gripper base position is calculated by offsetting from the finger center

    # Get gripper orientation matrix
    gripper_orientation_matrix = R.from_quat(current_quat, scalar_first=True).as_matrix()

    # Calculate the finger center offset from joint (this follows the circular arc)
    # The finger center should be at the handle position initially
    finger_center_offset_from_joint = handle_position - joint_position

    if np.abs(joint_axis[2]) > 0.9:
        # For Z-axis rotation, ensure the finger center offset is in XY plane only
        finger_center_offset_from_joint_xy = finger_center_offset_from_joint.copy()
        finger_center_offset_from_joint_xy[2] = 0
        finger_center_offset_from_joint = finger_center_offset_from_joint_xy

    path = {"mocap_pos": [], "mocap_quat": []}
    for angle in angles:
        # Calculate rotation matrix for this angle
        R_matrix = rotation_matrix_from_axis_angle(joint_axis, angle)

        # Rotate finger center offset around the joint (finger center follows circular arc)
        rotated_finger_center_offset = R_matrix @ finger_center_offset_from_joint

        # Calculate new finger center position (this follows the circular arc)
        next_finger_center_pos = joint_position + rotated_finger_center_offset

        # Calculate new gripper orientation by applying the joint rotation to the original orientation
        next_gripper_orientation_matrix = R_matrix @ gripper_orientation_matrix
        next_gripper_z_axis = next_gripper_orientation_matrix[:, 2]  # Z-axis points towards handle

        # Calculate gripper base position by offsetting from finger center
        # Gripper base is offset by half gripper length in the negative Z direction (away from handle)
        gripper_base_offset_from_finger = -0.5 * gripper_length * next_gripper_z_axis
        next_gripper_base_pos = next_finger_center_pos + gripper_base_offset_from_finger

        # print(f"  Angle {angle:.3f} rad:")
        # print(f"    Finger center pos (on arc): {next_finger_center_pos}")
        # print(f"    Gripper base pos (trajectory): {next_gripper_base_pos}")

        if next_finger_center_pos[2] < max(gripper_length, 0.3):
            continue  # skip if the finger center is below floor

        # Convert to quaternion
        next_quat = R.from_matrix(next_gripper_orientation_matrix).as_quat(scalar_first=True)

        # Use gripper base position for the mocap trajectory (but finger center follows the arc)
        path["mocap_pos"].append(next_gripper_base_pos)
        path["mocap_quat"].append(next_quat)

    # visualize the path
    if False:
        visualize_path(path, joint_position=joint_position)
    return path


def step_linear_path(
    to_handle_dist,
    current_pos,
    current_quat,
    step_size,
    is_reverse=False,
    gripper_length=GRIPPER_LENGTH,
):
    path = {"mocap_pos": [], "mocap_quat": []}
    path["mocap_pos"].append(current_pos)
    path["mocap_quat"].append(current_quat)

    dist = np.linalg.norm(to_handle_dist)

    if is_reverse:
        dist += gripper_length
    else:
        dist -= gripper_length

    # path forward
    for _i in range(int(dist / step_size)):
        # in direction of to_handle_dist
        angle = np.arctan2(to_handle_dist[1], to_handle_dist[0])
        if not is_reverse:
            next_pos = current_pos + step_size * np.array([np.cos(angle), np.sin(angle), 0])
        else:
            next_pos = current_pos - step_size * np.array([np.cos(angle), np.sin(angle), 0])
        path["mocap_pos"].append(next_pos)
        path["mocap_quat"].append(current_quat)
        dist -= np.linalg.norm(next_pos - current_pos)
        current_pos = next_pos
    return path


def process_handle_batch_ultra_fast(
    handle_batch, model, ithor_map, ithor_scene_path, floorplan_num, process_id=None
):
    """Ultra-fast version that skips all non-essential operations"""
    results = []

    for handle in handle_batch:
        handle_start_time = time.time()
        handle_name = handle["name"]
        handle["position"]
        handle["orientation"]
        handle_id = handle["handle_id"]

        # if "dishwasher" in handle_name.lower():
        #    continue

        # Create fresh data for each handle (reuse model)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data, nstep=PHYSICS_STEPS_PER_FRAME)

        ####
        # collect joint info
        handle_joint_id = model.body(handle_id).jntadr[0]
        if handle_joint_id == -1:
            print(f"No handle joint found for {handle_name}")
            continue
        handle_joint_type = model.joint(handle_joint_id).type
        handle_joint_qpos = model.joint(handle_joint_id).qposadr[0]

        start_handle_joint_pos = data.qpos[handle_joint_qpos]

        joint_range = model.joint(handle_joint_id).range
        max_range = joint_range[1] if joint_range[1] != 0 else joint_range[0]
        print(f"Max range: {max_range}")

        joint_info = {
            "joint_axis": model.joint(handle_joint_id).axis,
            "joint_position": model.joint(handle_joint_id).pos,
            "joint_range": model.joint(handle_joint_id).range,
            "joint_pos": data.qpos[handle_joint_qpos],
            "joint_body_position": data.xpos[handle_id],
            "joint_body_orientation": data.xmat[handle_id].reshape(3, 3),  # Body orientation matrix
            "max_range": max_range,
        }
        ####

        gripper_pose = get_gripper_pose_based_on_handle_pose(handle, ithor_map, joint_info)

        # Update gripper position
        data.mocap_pos[0] = gripper_pose["position"]
        data.mocap_quat[0] = gripper_pose["quaternion"]
        model.body("robot_0/").id
        gripper_qpos_start = model.body("robot_0/").dofadr[0]
        data.qpos[gripper_qpos_start : gripper_qpos_start + 3] = gripper_pose["position"]
        data.qpos[gripper_qpos_start + 3 : gripper_qpos_start + 7] = gripper_pose["quaternion"]
        mujoco.mj_step(model, data, nstep=PHYSICS_STEPS_PER_FRAME)

        STEP_SIZE = 0.005
        STEP_WAYPOINTS = 80

        if handle_joint_type == mujoco.mjtJoint.mjJNT_HINGE:  # hinge joint
            handle_info = handle

            current_pos = gripper_pose["position"]
            current_quat = gripper_pose["quaternion"]
            lienar_path = step_linear_path(
                to_handle_dist=handle["position"] - gripper_pose["position"],
                current_pos=gripper_pose["position"],
                current_quat=gripper_pose["quaternion"],
                step_size=STEP_SIZE,
            )
            circular_path = step_circular_path(
                current_pos, current_quat, handle_info, joint_info, n_waypoints=STEP_WAYPOINTS
            )
            path = {
                "approach_path": lienar_path,
                "pull_path": circular_path,
            }
        elif handle_joint_type == mujoco.mjtJoint.mjJNT_SLIDE:  # slider joint
            linear_path = step_linear_path(
                to_handle_dist=handle["position"] - gripper_pose["position"],
                current_pos=gripper_pose["position"],
                current_quat=gripper_pose["quaternion"],
                step_size=STEP_SIZE,
            )
            ## TODO: linear path to open the full length
            # position that is max range slide away from handle
            joint_range = model.joint(handle_joint_id).range
            max_range = joint_range[1] if joint_range[1] != 0 else joint_range[0]
            # axis is in the local frame, so we need to convert it to the world frame
            normalize_dir_axis = (
                linear_path["mocap_pos"][-1] - linear_path["mocap_pos"][0]
            ) / np.linalg.norm(linear_path["mocap_pos"][-1] - linear_path["mocap_pos"][0])
            end_pos = linear_path["mocap_pos"][-1] + normalize_dir_axis * max_range
            linear_reverse_path = step_linear_path(
                to_handle_dist=end_pos - linear_path["mocap_pos"][-1],
                is_reverse=True,
                current_pos=linear_path["mocap_pos"][-1],
                current_quat=linear_path["mocap_quat"][-1],
                step_size=STEP_SIZE,
            )
            path = {
                "approach_path": linear_path,
                "pull_path": linear_reverse_path,
            }

        else:
            print(f"Unknown handle joint type: {handle_joint_type}")
            continue

        # Ultra-fast simulation with minimal overhead
        simulation_start_time = time.time()

        # Create minimal teleop controller without rendering or recording
        replay_teleop = GripperTeleopController(
            scene_model=model,
            use_keyboard=False,
            use_spacemouse=False,
            nstep=PHYSICS_STEPS_PER_FRAME,
            replay_trajectory=path,
            use_viewer=False,
            handle_id=handle_joint_id,  # nv
        )

        # Use dummy recorder if SKIP_SAVE_TRAJECTORIES is True, otherwise keep the real recorder
        if SKIP_SAVE_TRAJECTORIES:
            # Create a dummy recorder that does nothing to avoid NoneType errors
            class DummyRecorder:
                def __call__(self, data):
                    pass

                def save(self, save_dir) -> None:
                    pass

            # Use dummy recorder instead of None to avoid NoneType errors
            replay_teleop.recorders = [DummyRecorder()]

        PROFILER.start("trajectory")
        replay_data = replay_teleop.run()
        PROFILER.end("trajectory")

        simulation_end_time = time.time()
        simulation_time = simulation_end_time - simulation_start_time
        end_handle_joint_pos = replay_data.qpos[handle_joint_qpos]

        success = int(
            np.abs(end_handle_joint_pos - start_handle_joint_pos) / np.abs(max_range)
            > 0.25  # > 0.05
        )  # np.deg2rad(5))

        # Monitor after the trajectory if the door/drawer stays in the same position
        # if not, then the gripper is not stable
        handle_qpos_start = model.body(handle_id).dofadr[0]
        handle_qpos_end = handle_qpos_start + 3
        before_handle_qpos = replay_data.qpos[handle_qpos_start:handle_qpos_end]

        # step 1000 times after the trajectory
        # monitor for 1000 steps (1000 * 0.002 = 2 seconds)

        for _i in range(1000 // PHYSICS_STEPS_PER_FRAME):
            mujoco.mj_step(model, replay_data, nstep=PHYSICS_STEPS_PER_FRAME)
            for recorder in replay_teleop.recorders:
                recorder(replay_data)
        after_handle_qpos = replay_data.qpos[handle_qpos_start:handle_qpos_end]

        post_trajectory_handle_qpos_change = np.linalg.norm(after_handle_qpos - before_handle_qpos)

        final_success = int(success and post_trajectory_handle_qpos_change < 0.01)
        if not final_success:
            print(f"Handle {handle_name} failed to open")
        # Categorize handle by object type
        object_type = categorize_handle_by_object_type(handle_name)

        # Force applied is the force applied to the handle by the gripper
        force_applied = replay_teleop.force_applied
        joint_pos_trajectory = replay_teleop.joint_pos_trajectory

        # debug plot joint pos vs force applied
        debug_plot = False
        if debug_plot:
            # plot joint pos vs force applied alwas absolute
            plt.plot(np.abs(joint_pos_trajectory), np.abs(force_applied))
            plt.xlabel("Joint Position")
            plt.ylabel("Force Applied (N)")
            plt.title("Joint Position vs Force Applied")
            plt.show()

        # find min force applied when joint pos non zero
        # first index of joint pos trajectory where joint pos is non zero
        non_zero_joint_pos_idx = np.where(np.abs(joint_pos_trajectory) > 0.001)[0]
        if len(non_zero_joint_pos_idx) > 0:
            min_force_applied = force_applied[non_zero_joint_pos_idx[0]]
        else:
            min_force_applied = np.max(force_applied)
        print(f"Min force applied when joint pos non zero: {min_force_applied}")

        # Prepare success info for saving
        success_info = {
            "success": final_success,
            "object_type": object_type,
            "handle_joint_pos_change": np.abs(end_handle_joint_pos - start_handle_joint_pos),
            "post_trajectory_handle_qpos_change": post_trajectory_handle_qpos_change,
            "min_force_applied": min_force_applied,
        }

        results.append(
            {
                "handle_name": handle_name,
                "object_type": object_type,
                # "start_handle_joint_pos": start_handle_joint_pos,
                "handle_joint_pos_change": np.abs(end_handle_joint_pos - start_handle_joint_pos),
                f"post_trajectory_handle_qpos_change_{1000 * 0.002}s": post_trajectory_handle_qpos_change,
                "success": final_success,
                # "force_applied": force_applied,
                "min_force_applied": min_force_applied,
            }
        )

        # Calculate handle processing time
        handle_end_time = time.time()
        handle_processing_time = handle_end_time - handle_start_time

        # Skip trajectory saving for maximum speed
        if not SKIP_SAVE_TRAJECTORIES:
            save_dir = f"{SAVE_DIR_PATH}/floorplan_{floorplan_num}"

            # save only failed trajectory
            if not final_success:
                replay_teleop.save(
                    save_dir, ithor_scene_path, handle_name=handle_name, success_info=success_info
                )

        print(
            f"Handle {handle_name} processed in {handle_processing_time:.2f}s (sim: {simulation_time:.2f}s, success: {final_success})"
        )

    return results


def run_one_floorplan(i, mesh, process_id=None, use_gpu=False) -> None:
    # Start timing
    scene_start_time = time.time()

    if mesh:
        ithor_scene_path = f"assets/scenes/ithor_{iTHOR_DATE}/FloorPlan{i}_physics_mesh.xml"
        ithor_map_path = f"{ithor_scene_path.replace('_mesh.xml', '_map.png')}"
    else:
        ithor_scene_path = f"assets/scenes/ithor_{iTHOR_DATE}/FloorPlan{i}_physics.xml"
        ithor_map_path = f"{ithor_scene_path.replace('.xml', '_map.png')}"

    # check if scene path exists
    if not os.path.exists(ithor_scene_path):
        print(f"Scene path {ithor_scene_path} does not exist")
        return

    print(f"Processing FloorPlan {i} (process {process_id})...")

    try:
        # Skip map generation if already exists
        regenerate_map(ithor_scene_path, process_id, use_gpu)

        print(f"Loading map from {ithor_map_path}")
        try:
            ithor_map = iTHORMap.load(ithor_map_path)
            print("Map loaded successfully")
        except Exception as e:
            print(f"Error loading map from {ithor_map_path}: {e}")
            import traceback

            traceback.print_exc()
            raise

        # initialize the scene
        print(f"Loading scene model from {ithor_scene_path}")
        try:
            model = mujoco.MjModel.from_xml_path(ithor_scene_path)
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            print("Scene model loaded successfully")
        except Exception as e:
            print(f"Error loading scene model from {ithor_scene_path}: {e}")
            import traceback

            traceback.print_exc()
            raise

        all_handles = get_all_handles(model, data)
        print(f"Found {len(all_handles)} handles")

        # Initialize gripper_path
        gripper_path = None

        # Add gripper to scene once if there are handles
        if len(all_handles) > 0:
            first_handle = all_handles[0]
            print(f"Getting gripper pose for first handle: {first_handle['name']}")
            try:
                gripper_pose = get_gripper_pose_based_on_handle_pose(first_handle, ithor_map)
                print(
                    f"Gripper pose calculated: position={gripper_pose['position']}, quaternion={gripper_pose['quaternion']}"
                )

                print(f"Adding gripper to scene: {ithor_scene_path}")
                gripper_path = add_gripper_to_scene(ithor_scene_path, gripper_pose)
                print(f"Added gripper to scene: {gripper_path}")
            except Exception as e:
                print(f"Error adding gripper to scene: {e}")
                import traceback

                traceback.print_exc()
                raise
        else:
            print(f"No handles found in FloorPlan {i}, skipping processing")
            return

        # Process handles in batches for better performance
        success_metric = {}
        n_success = 0
        n_total = 0
        failed_handles = []

        # Initialize statistics by object type
        stats_by_type = {}
        for obj_type in OBJECT_CATEGORIES + ["other"]:
            stats_by_type[obj_type] = {
                "success_count": 0,
                "total_count": 0,
                "failed_handles": [],
                "min_force_applied": [],
            }

        # Split handles into batches
        handle_batches = [
            all_handles[j : j + BATCH_SIZE] for j in range(0, len(all_handles), BATCH_SIZE)
        ]

        # Define floorplan_num early
        floorplan_num = i

        # Load model once for all batches (major optimization)
        print(f"Loading model with gripper from {gripper_path}")
        try:
            model, root_bodies_dict = load_env_with_objects(gripper_path)
            print(f"Model loaded successfully for all {len(handle_batches)} batches")
        except Exception as e:
            print(f"Error loading model with gripper from {gripper_path}: {e}")
            import traceback

            traceback.print_exc()
            raise

        for batch_idx, handle_batch in enumerate(handle_batches):
            batch_start_time = time.time()
            print(
                f"Processing batch {batch_idx + 1}/{len(handle_batches)} ({len(handle_batch)} handles)"
            )

            try:
                # Use ultra-fast version for maximum speed
                batch_results = process_handle_batch_ultra_fast(
                    handle_batch, model, ithor_map, ithor_scene_path, floorplan_num, process_id
                )

                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                print(
                    f"Batch {batch_idx + 1} completed in {batch_processing_time:.2f}s ({len(handle_batch)} handles)"
                )

                # Process results
                for result in batch_results:
                    handle_name = result["handle_name"]
                    object_type = result["object_type"]
                    success = result["success"]

                    success_metric[handle_name] = {
                        "object_type": object_type,
                        "handle_joint_pos_change": result["handle_joint_pos_change"],
                        f"post_trajectory_handle_qpos_change_{1000 * 0.002}s": result[
                            f"post_trajectory_handle_qpos_change_{1000 * 0.002}s"
                        ],
                        "success": success,
                        "min_force_applied": result["min_force_applied"],
                    }

                    # Update overall statistics
                    n_success += success
                    n_total += 1
                    if not success:
                        failed_handles.append(handle_name)

                    # Update statistics by object type
                    stats_by_type[object_type]["total_count"] += 1
                    stats_by_type[object_type]["success_count"] += success
                    stats_by_type[object_type]["min_force_applied"].append(
                        result["min_force_applied"]
                    )
                    if not success:
                        stats_by_type[object_type]["failed_handles"].append(handle_name)

            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Calculate total proc∫-ssing time
        scene_end_time = time.time()
        scene_processing_time = scene_end_time - scene_start_time

        # Calculate success rates by object type
        all_force_applied = []
        success_rates_by_type = {}
        for obj_type in OBJECT_CATEGORIES + ["other"]:
            total = stats_by_type[obj_type]["total_count"]
            success = stats_by_type[obj_type]["success_count"]
            min_force_applied = stats_by_type[obj_type]["min_force_applied"]
            success_rates_by_type[obj_type] = {
                "success_rate": success / total if total > 0 else 1.0,
                "success_count": success,
                "total_count": total,
                "failed_handles": stats_by_type[obj_type]["failed_handles"],
                "min_min_force_applied": (
                    np.min(min_force_applied) if len(min_force_applied) > 0 else 0.0
                ),
                "max_min_force_applied": (
                    np.max(min_force_applied) if len(min_force_applied) > 0 else 0.0
                ),
                "mean_min_force_applied": (
                    np.mean(min_force_applied) if len(min_force_applied) > 0 else 0.0
                ),
                "std_min_force_applied": (
                    np.std(min_force_applied) if len(min_force_applied) > 0 else 0.0
                ),
            }
            all_force_applied.extend(min_force_applied)

        # save success metric to json
        success_metric["success_rate"] = n_success / n_total if n_total > 0 else 1
        success_metric["failed_handles"] = failed_handles
        success_metric["processing_time_seconds"] = scene_processing_time
        success_metric["handles_per_second"] = (
            n_total / scene_processing_time if scene_processing_time > 0 else 0
        )
        success_metric["success_rates_by_object_type"] = success_rates_by_type
        success_metric["min_force_applied"] = (
            np.min(min_force_applied) if len(min_force_applied) > 0 else 0.0
        )
        success_metric["max_force_applied"] = (
            np.max(min_force_applied) if len(min_force_applied) > 0 else 0.0
        )
        success_metric["all_force_applied"] = all_force_applied

        print(f"FloorPlan {i} completed in {scene_processing_time:.2f} seconds")
        print(
            f"Processed {n_total} handles at {n_total / scene_processing_time:.2f} handles/second"
        )
        print(f"Overall success rate: {success_metric['success_rate']:.3f} ({n_success}/{n_total})")

        # Print success rates by object type
        print("Success rates by object type:")
        for obj_type in OBJECT_CATEGORIES + ["other"]:
            stats = success_rates_by_type[obj_type]
            if stats["total_count"] > 0:
                print(
                    f"  {obj_type}: {stats['success_rate']:.3f} ({stats['success_count']}/{stats['total_count']})"
                )

        # print(success_metric)
        datetime = time.strftime("%Y%m%d_%H%M%S")

        # Use correct floorplan number (already defined above)
        os.makedirs(f"{SAVE_DIR_PATH}/floorplan_{floorplan_num}", exist_ok=True)
        with open(
            f"{SAVE_DIR_PATH}/floorplan_{floorplan_num}/success_metric_{success_metric['success_rate']:.2f}_{datetime}.json",
            "w",
        ) as f:
            json.dump(success_metric, f)

    except Exception as e:
        print(f"Error processing FloorPlan {i}: {e}")
        import traceback

        traceback.print_exc()
        raise


def aggregate_statistics_across_floorplans(output_dir=SAVE_DIR_PATH):
    """
    Aggregate statistics across all floorplans and generate a summary report.

    Args:
        output_dir: Directory containing the individual floorplan results
    """
    import glob

    # Initialize aggregated statistics
    aggregated_stats = {}
    for obj_type in OBJECT_CATEGORIES + ["other"]:
        aggregated_stats[obj_type] = {"success_count": 0, "total_count": 0, "failed_handles": []}

    overall_success = 0
    overall_total = 0
    overall_failed = []

    # Find all success metric files
    pattern = os.path.join(output_dir, "floorplan_*", "success_metric_*.json")
    metric_files = glob.glob(pattern)

    print(f"Found {len(metric_files)} floorplan result files")

    # Aggregate statistics from each file
    all_cat_force_applied = []
    for metric_file in metric_files:
        try:
            with open(metric_file, "r") as f:
                data = json.load(f)

            # Aggregate overall statistics
            if "success_rates_by_object_type" in data:
                for obj_type in OBJECT_CATEGORIES + ["other"]:
                    if obj_type in data["success_rates_by_object_type"]:
                        type_data = data["success_rates_by_object_type"][obj_type]
                        aggregated_stats[obj_type]["success_count"] += type_data["success_count"]
                        aggregated_stats[obj_type]["total_count"] += type_data["total_count"]
                        aggregated_stats[obj_type]["failed_handles"].extend(
                            type_data["failed_handles"]
                        )

            if "all_force_applied" in data:
                all_cat_force_applied.extend(data["all_force_applied"])

            # Also aggregate overall stats
            for handle_name, handle_data in data.items():
                if isinstance(handle_data, dict) and "success" in handle_data:
                    overall_total += 1
                    if handle_data["success"]:
                        overall_success += 1
                    else:
                        overall_failed.append(handle_name)

        except Exception as e:
            print(f"Error processing {metric_file}: {e}")
            continue

    # Calculate final success rates
    print(f"\n{'=' * 60}")
    print("AGGREGATED STATISTICS ACROSS ALL FLOORPLANS")
    print(f"{'=' * 60}")
    print(
        f"Overall success rate: {overall_success / overall_total:.3f} ({overall_success}/{overall_total})"
    )
    print("\nSuccess rates by object type:")

    for obj_type in OBJECT_CATEGORIES + ["other"]:
        stats = aggregated_stats[obj_type]
        if stats["total_count"] > 0:
            success_rate = stats["success_count"] / stats["total_count"]
            print(
                f"  {obj_type.ljust(12)}: {success_rate:.3f} ({stats['success_count']}/{stats['total_count']})"
            )
        else:
            print(f"  {obj_type.ljust(12)}: N/A (0/0)")

    # Save aggregated results
    aggregated_results = {
        "overall_success_rate": overall_success / overall_total if overall_total > 0 else 0.0,
        "overall_success_count": overall_success,
        "overall_total_count": overall_total,
        "success_rates_by_object_type": {},
    }

    for obj_type in OBJECT_CATEGORIES + ["other"]:
        stats = aggregated_stats[obj_type]
        aggregated_results["success_rates_by_object_type"][obj_type] = {
            "success_rate": (
                stats["success_count"] / stats["total_count"] if stats["total_count"] > 0 else 0.0
            ),
            "success_count": stats["success_count"],
            "total_count": stats["total_count"],
            "failed_count": len(stats["failed_handles"]),
        }

    datetime = time.strftime("%Y%m%d_%H%M%S")

    # Save histogram of force applied
    plt.figure()
    plt.hist(np.abs(all_cat_force_applied), bins=10, edgecolor="black")
    plt.title("Histogram of Force Applied")
    plt.xlabel("Force Applied")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"histogram_force_applied_{datetime}.png"))

    # Save to file
    output_file = os.path.join(output_dir, f"aggregated_statistics_{datetime}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\nAggregated statistics saved to: {output_file}")
    return aggregated_results


def main() -> None:
    # Set multiprocessing start method for Linux compatibility
    # Use 'spawn' to avoid issues with OpenGL contexts and GPU operations
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            print("Warning: Could not set multiprocessing start method to 'spawn'")
            print("This may cause issues with GPU operations in multiprocessing mode")

    # FIND ALL HANDLES IN THE SCENE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i",
        type=int,
        required=False,
        default=-1,
        help="Floorplan index. Use -1 to process all floorplans",
    )
    parser.add_argument(
        "--mesh", action="store_true", help="Use mesh files instead of primitive files"
    )
    parser.add_argument(
        "--nprocess",
        type=int,
        default=8,
        help="Number of processes for parallel processing (default: 8)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Enable GPU rendering if available (default: CPU-only)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Batch size for processing handles (default: 20)"
    )
    parser.add_argument(
        "--physics-steps", type=int, default=50, help="Physics steps per frame (default: 50)"
    )
    parser.add_argument(
        "--skip-maps", action="store_true", help="Skip map generation if already exists"
    )
    parser.add_argument("--skip-save", action="store_true", help="Skip saving trajectory data")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing results, don't run new experiments",
    )
    parser.add_argument(
        "--aggregate-path",
        type=str,
        default=None,
        help="Path to aggregate results from",
    )
    args = parser.parse_args()

    # If only aggregating, run aggregation and exit
    if args.aggregate_only:
        aggregate_statistics_across_floorplans(args.aggregate_path)
        return

    i = args.i
    mesh = args.mesh
    nprocess = args.nprocess
    use_gpu = args.gpu

    # Update global constants based on arguments
    global BATCH_SIZE, PHYSICS_STEPS_PER_FRAME, SKIP_SAVE_TRAJECTORIES, SKIP_MAP_GENERATION
    BATCH_SIZE = args.batch_size
    PHYSICS_STEPS_PER_FRAME = args.physics_steps

    if args.skip_maps:
        SKIP_MAP_GENERATION = True

    if args.skip_save:
        SKIP_SAVE_TRAJECTORIES = True

    if i == -1:
        # Process all floorplans in parallel
        floorplan_indices = list(range(0, 431, 1))  # 430.
        # (making a map took a long time) 418. 402, 302. 224. 201. 27. 23. 21. 12

        # Start overall timing
        overall_start_time = time.time()

        print(f"Processing {len(floorplan_indices)} floorplans using {nprocess} processes...")
        print(f"Floorplan range: {floorplan_indices[0]} to {floorplan_indices[-1]}")
        print(f"Batch size: {BATCH_SIZE}, Physics steps per frame: {PHYSICS_STEPS_PER_FRAME}")

        # Check GPU availability and inform user
        if use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    print(
                        "GPU mode enabled: Each process will use a different device_id to avoid OpenGL context conflicts"
                    )
                else:
                    print(
                        "GPU mode requested but no GPU detected: All processes will use CPU rendering (device_id=None)"
                    )
            except ImportError:
                print(
                    "GPU mode requested but PyTorch not available: All processes will use CPU rendering (device_id=None)"
                )
        else:
            print("CPU-only mode (default): All processes will use CPU rendering (device_id=None)")

        completed_count = 0
        failed_count = 0
        failed_floorplans = []
        with ProcessPoolExecutor(max_workers=nprocess) as executor:
            # Submit all tasks with process IDs
            future_to_floorplan = {
                executor.submit(
                    run_one_floorplan, floorplan_i, mesh, process_id, use_gpu
                ): floorplan_i
                for process_id, floorplan_i in enumerate(floorplan_indices)
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(floorplan_indices), desc="Processing floorplans") as pbar:
                for future in as_completed(future_to_floorplan):
                    floorplan_i = future_to_floorplan[future]
                    try:
                        future.result()  # This will raise any exception that occurred
                        completed_count += 1
                        print(
                            f"Completed FloorPlan {floorplan_i} ({completed_count}/{len(floorplan_indices)})"
                        )
                    except Exception as exc:
                        failed_count += 1
                        print(f"FloorPlan {floorplan_i} generated an exception: {exc}")
                        print(f"Failed count: {failed_count}")
                        failed_floorplans.append(floorplan_i)
                        print(traceback.format_exc())
                    pbar.update(1)

        overall_end_time = time.time()
        overall_processing_time = overall_end_time - overall_start_time

        print(f"All floorplans processed! Completed: {completed_count}, Failed: {failed_count}")
        print(f"Total processing time: {overall_processing_time:.2f} seconds")
        print(
            f"Average time per floorplan: {overall_processing_time / len(floorplan_indices):.2f} seconds"
        )
        print(f"Failed floorplans: {failed_floorplans}")

        # Aggregate statistics across all floorplans
        if completed_count > 0:
            print("\nAggregating statistics across all processed floorplans...")
            aggregate_statistics_across_floorplans()

        return
    else:
        run_one_floorplan(i, mesh, process_id=i, use_gpu=use_gpu)

    PROFILER.print_all()


if __name__ == "__main__":
    main()
