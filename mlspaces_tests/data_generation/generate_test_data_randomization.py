"""
Generate test data for randomization testing.

This script loads a scene and captures baseline values for:
- Dynamics properties (mass, inertia, friction)
- Texture/material properties (via camera images and material properties)
- Lighting properties

The baseline data is saved to be used by test_randomization.py to verify that
randomization works correctly.
"""

import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np
from mujoco import MjData, MjModel

from mlspaces_tests.data_generation.config import RandomizationTestConfig
from molmo_spaces.env.arena.arena_utils import get_all_bodies_with_joints_as_mlspaces_objects
from molmo_spaces.env.arena.randomization.texture import setup_empty_materials
from molmo_spaces.molmo_spaces_constants import get_scenes

RANDOMIZED_TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data" / "test_randomized_data"


def capture_dynamics_baseline(model: MjModel, data: MjData) -> dict:
    """Capture baseline dynamics properties for all objects with joints."""
    baseline = {}

    # Get all objects with joints (these are candidates for dynamics randomization)
    objects = get_all_bodies_with_joints_as_mlspaces_objects(model, data)

    for obj in objects:
        object_id = obj.object_id
        object_root_id = model.body(object_id).rootid[0]

        # Get all bodies belonging to this object
        from molmo_spaces.utils import mj_model_and_data_utils

        body_ids = mj_model_and_data_utils.descendant_bodies(model, object_id)

        # Get total mass
        total_mass = float(model.body_subtreemass[object_id])

        # Get inertia
        inertia = np.array(model.body_inertia[object_id]).tolist()

        # Get friction for all geoms
        geom_frictions = {}
        for geom_id in range(model.ngeom):
            geom_body_id = model.geom(geom_id).bodyid.item()
            geom_root_id = model.body(geom_body_id).rootid[0]
            if geom_root_id == object_root_id:
                geom_frictions[geom_id] = np.array(model.geom_friction[geom_id]).tolist()

        baseline[obj.name] = {
            "mass": total_mass,
            "inertia": inertia,
            "body_ids": body_ids,
            "body_masses": {bid: float(model.body_mass[bid]) for bid in body_ids},
            "geom_frictions": geom_frictions,
        }

    return baseline


def capture_texture_material_baseline(model: MjModel) -> dict:
    """Capture baseline texture and material properties."""
    baseline = {
        "geom_materials": {},
        "geom_rgba": {},
        "material_properties": {},
        "texture_properties": {},
    }

    # Capture geom material and color properties
    for geom_id in range(model.ngeom):
        geom = model.geom(geom_id)

        # Get material ID if assigned
        mat_id = int(geom.matid)  # Convert to int for use as dictionary key
        if mat_id >= 0:
            mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
            baseline["geom_materials"][geom_id] = mat_name if mat_name else f"material_{mat_id}"

            # Get material properties
            if mat_id not in baseline["material_properties"]:
                mat = model.material(mat_id)
                mat_data = {
                    "rgba": np.array(mat.rgba).tolist(),
                    "specular": float(mat.specular),
                    "shininess": float(mat.shininess),
                    "emission": float(mat.emission),
                }

                # Get texture ID if material has a texture
                # Access through model.mat_texid (can be 1D or 2D array)
                tex_id = -1
                if hasattr(model, "mat_texid"):
                    try:
                        if isinstance(model.mat_texid, np.ndarray):
                            if model.mat_texid.ndim == 2 and mat_id < model.mat_texid.shape[0]:
                                tex_id = int(model.mat_texid[mat_id, 0])
                            elif model.mat_texid.ndim == 1 and mat_id < len(model.mat_texid):
                                tex_id = int(model.mat_texid[mat_id])
                    except (IndexError, ValueError, TypeError):
                        tex_id = -1

                if tex_id >= 0:
                    mat_data["texid"] = tex_id

                    # Capture texture metadata (but not bitmap data to reduce file size)
                    if tex_id not in baseline["texture_properties"]:
                        try:
                            height = int(model.tex_height[tex_id])
                            width = int(model.tex_width[tex_id])
                            nchannel = int(model.tex_nchannel[tex_id])
                            tex_type = int(model.tex_type[tex_id])

                            # Get texture name
                            tex_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id)

                            baseline["texture_properties"][tex_id] = {
                                "name": tex_name if tex_name else f"texture_{tex_id}",
                                "type": int(tex_type),
                                "width": int(width),
                                "height": int(height),
                                "nchannel": int(nchannel),
                                # Note: bitmap data not saved to reduce file size
                            }
                        except (IndexError, ValueError, AttributeError) as e:
                            print(f"  Warning: Could not capture texture {tex_id}: {e}")
                            baseline["texture_properties"][tex_id] = {
                                "error": str(e),
                            }

                baseline["material_properties"][mat_id] = mat_data

        # Get geom RGBA (may override material)
        if np.any(geom.rgba != model.geom_rgba[geom_id]):
            baseline["geom_rgba"][geom_id] = np.array(geom.rgba).tolist()
        else:
            baseline["geom_rgba"][geom_id] = np.array(model.geom_rgba[geom_id]).tolist()

    return baseline


def capture_lighting_baseline(model: MjModel) -> dict:
    """Capture baseline lighting properties."""
    baseline = {}

    for light_id in range(model.nlight):
        light = model.light(light_id)
        light_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, light_id)

        baseline[light_id] = {
            "name": light_name if light_name else f"light_{light_id}",
            "pos": np.array(light.pos).tolist(),
            "dir": np.array(light.dir).tolist(),
            "specular": np.array(light.specular).tolist(),
            "ambient": np.array(light.ambient).tolist(),
            "diffuse": np.array(light.diffuse).tolist(),
            "active": int(light.active),
        }

    return baseline


def calculate_scene_center(model: MjModel, data: MjData) -> tuple[np.ndarray, float]:
    """Calculate the center of the scene structure by finding floor bodies.

    Returns:
        center: (x, y, z) center position
        max_z: Maximum z coordinate of floor bodies (for camera height)
    """
    # Find floor bodies/geoms
    floor_positions = []
    floor_body_ids = set()

    # First, find floor bodies by name
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and ("floor" in body_name.lower() or body_name.startswith("room")):
            floor_body_ids.add(body_id)
            body_pos = data.xpos[body_id]
            floor_positions.append(body_pos)

    if not floor_positions:
        # Fallback: use all bodies to estimate center
        print("  Warning: No floor bodies found, using all bodies to estimate center")
        for body_id in range(1, model.nbody):  # Skip world body (id 0)
            body_pos = data.xpos[body_id]
            floor_positions.append(body_pos)

    if not floor_positions:
        # Last resort: use origin
        print("  Warning: No bodies found, using origin as center")
        return np.array([0.0, 0.0, 2.5]), 0.0

    floor_positions = np.array(floor_positions)

    # Calculate center (mean of x, y coordinates) and max z
    center_xy = np.mean(floor_positions[:, :2], axis=0)
    max_z = float(np.max(floor_positions[:, 2])) if len(floor_positions) > 0 else 0.0

    # Camera should be above the floor
    camera_z = max_z + 2.5  # 2.5m above the highest floor point

    center = np.array([center_xy[0], center_xy[1], camera_z])

    return center, max_z


def add_test_camera(spec: mujoco.MjSpec, model: MjModel = None, data: MjData = None) -> None:
    """Add a fixed-position camera to the scene for testing.

    If model and data are provided, calculates camera position above floor center.
    Otherwise uses default position.
    Camera is pitched down at ~60 degrees to capture both floor and walls.
    """
    from scipy.spatial.transform import Rotation as R

    if model is not None and data is not None:
        # Calculate scene center from floor bodies
        center, floor_z = calculate_scene_center(model, data)
        camera_pos = center
        # Look at a point on the floor, slightly offset to capture walls better
        lookat_pos = np.array([center[0], center[1], floor_z + 0.1])  # Look at floor level
        print(f"  Camera positioned at {camera_pos} above scene center")
    else:
        # Default position (fallback)
        camera_pos = np.array([3.0, -3.0, 2.5])
        lookat_pos = np.array([0.0, 0.0, 0.1])

    # Calculate direction vector from camera to lookat point
    direction = lookat_pos - camera_pos
    direction = direction / np.linalg.norm(direction)

    # Default up vector (world Z-up)
    world_up = np.array([0.0, 0.0, 1.0])

    # Calculate right vector (perpendicular to direction and up)
    right = np.cross(direction, world_up)
    right_norm = np.linalg.norm(right)

    # If direction is parallel to up, use a different reference
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm

    # Recalculate up to be orthogonal to both direction and right
    up = np.cross(right, direction)
    up = up / np.linalg.norm(up)

    # Build rotation matrix: [right, up, -direction] (camera looks along -direction)
    # MuJoCo camera convention: camera looks along negative Z in camera frame
    base_rotation_matrix = np.column_stack([right, up, -direction])

    # Add pitch rotation (tilt camera down to see both floor and walls)
    # Pitch angle: ~60 degrees down from horizontal (in radians)
    # This angle allows seeing both the floor and vertical walls
    pitch_angle = np.radians(310)  # 310 degrees pitch down

    # Make camera upside down
    yaw_angle = np.radians(180)  # 180 degrees yaw
    yaw_rotation = R.from_rotvec(world_up * yaw_angle)  # Rotate around Z-axis (world up)

    # Create pitch rotation around the right axis (X-axis in camera frame)
    # Rotating around the right vector pitches the camera down
    pitch_axis = right  # Pitch around the right vector (camera's X-axis)
    pitch_rotation = R.from_rotvec(pitch_axis * pitch_angle)

    # Apply pitch to the base rotation
    # Compose rotations: first apply base rotation, then pitch
    base_rotation = R.from_matrix(base_rotation_matrix)
    final_rotation = base_rotation * pitch_rotation * yaw_rotation

    # Convert to quaternion (w, x, y, z format for MuJoCo)
    quat = final_rotation.as_quat()  # Returns [x, y, z, w]
    # MuJoCo uses [w, x, y, z] format, so reorder
    quat = [quat[3], quat[0], quat[1], quat[2]]

    spec.worldbody.add_camera(
        name="test_camera",
        pos=camera_pos.tolist(),
        quat=quat,
    )


def capture_camera_image(
    model: MjModel, data: MjData, save_path: Path = None, suffix: str = "baseline"
) -> dict:
    """Capture camera image for texture comparison.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        save_path: Path to save the image (optional)
        suffix: Suffix for the filename (default: "baseline", can be "randomized")
    """
    from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

    result = {}

    # Initialize renderer
    renderer = MjOpenGLRenderer(model=model)

    # Forward pass to ensure data is up to date
    mujoco.mj_forward(model, data)

    # Render from test_camera
    camera_name = "test_camera"
    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id >= 0:
            renderer.update(data, camera=camera_name)
            image = renderer.render()

            # Save PNG image
            if save_path:
                image_path = save_path / f"{camera_name}_{suffix}.png"
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                # MuJoCo renderer typically returns RGB, but check format
                if image.shape[2] == 3:
                    # Save as RGB PNG
                    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    print(f"    Saved image to: {image_path}")

            result[camera_name] = {
                "image_shape": list(image.shape),
                "image_mean": float(np.mean(image)),
                "image_std": float(np.std(image)),
                "image_hash": hash(image.tobytes()),  # Simple hash for comparison
                "image_path": str(save_path / f"{camera_name}_{suffix}.png") if save_path else None,
            }
        else:
            print(f"Warning: Camera '{camera_name}' not found in model")
    except Exception as e:
        print(f"Warning: Could not render from camera '{camera_name}': {e}")

    renderer.close()
    return result


def capture_camera_image_baseline(model: MjModel, data: MjData, save_path: Path = None) -> dict:
    """Capture baseline camera image for texture comparison."""
    return capture_camera_image(model, data, save_path, suffix="baseline")


def randomize_scene(
    model: MjModel,
    data: MjData,
    enable_texture: bool = True,
    enable_dynamics: bool = True,
    enable_lighting: bool = True,
    seed: int = None,
) -> None:
    """Randomize the scene."""
    from molmo_spaces.env.arena.randomization.dynamics import DynamicsRandomizer
    from molmo_spaces.env.arena.randomization.lighting import LightingRandomizer
    from molmo_spaces.env.arena.randomization.texture import TextureRandomizer

    # Create random state from seed
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = None

    if enable_lighting:
        lighting_randomizer = LightingRandomizer(
            model=model,
            random_state=random_state,
            randomize_position=True,
            randomize_direction=True,
            randomize_specular=True,
            randomize_ambient=True,
            randomize_diffuse=True,
            randomize_active=True,
        )
        lighting_randomizer.randomize(data)

    if enable_texture:
        texture_randomizer = TextureRandomizer(
            model=model,
            random_state=random_state,
            randomize_geom_rgba=True,
            randomize_material_rgba=True,
            randomize_material_specular=True,
            randomize_material_shininess=True,
            randomize_texture=True,
        )
        texture_randomizer.randomize_by_category(data)

    if enable_dynamics:
        dynamics_randomizer = DynamicsRandomizer(
            random_state=random_state,
            randomize_friction=True,
            randomize_mass=True,
            randomize_inertia=True,
        )
        dynamics_randomizer.randomize_objects(
            get_all_bodies_with_joints_as_mlspaces_objects(model, data)
        )

    mujoco.mj_forward(model, data)


def generate_test_data(scene_path: str, config: RandomizationTestConfig = None) -> None:
    """Generate test data for a scene."""
    if config is None:
        config = RandomizationTestConfig()

    print(f"Loading scene from: {scene_path}")

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path does not exist: {scene_path}")

    # Load model
    spec = mujoco.MjSpec.from_file(scene_path)
    setup_empty_materials(spec)  # Setup for texture randomization

    # Compile model first to calculate camera position
    model_temp = spec.compile()
    data_temp = MjData(model_temp)
    mujoco.mj_forward(model_temp, data_temp)

    # Add a fixed camera for testing (positioned above floor center)
    add_test_camera(spec, model_temp, data_temp)

    # Recompile with camera added
    model = spec.compile()
    data = MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded model: {model.ngeom} geoms, {model.nlight} lights, {model.ntex} textures")

    # Create output directory
    RANDOMIZED_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Capture camera image baseline BEFORE randomization (for image comparison)
    print("Capturing camera image baseline (before randomization)...")
    camera_baseline = capture_camera_image_baseline(model, data, RANDOMIZED_TEST_DATA_DIR)
    print(f"  Captured {len(camera_baseline)} camera images")

    # Capture original values BEFORE randomization for debugging
    print("\nCapturing original values (before randomization) for debugging...")
    original_lighting = {}
    if config.enable_lighting_randomization:
        for light_id in range(model.nlight):
            light = model.light(light_id)
            light_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, light_id)
            original_lighting[light_id] = {
                "name": light_name if light_name else f"light_{light_id}",
                "pos": np.array(light.pos).copy(),
                "dir": np.array(light.dir).copy(),
                "specular": np.array(light.specular).copy(),
                "ambient": np.array(light.ambient).copy(),
                "diffuse": np.array(light.diffuse).copy(),
                "active": int(light.active),
            }
        print(f"  Captured {len(original_lighting)} lights")

    original_dynamics = {}
    if config.enable_dynamics_randomization:
        objects = get_all_bodies_with_joints_as_mlspaces_objects(model, data)
        for obj in objects:
            object_id = obj.object_id
            total_mass = float(model.body_subtreemass[object_id])
            inertia = np.array(model.body_inertia[object_id]).copy()
            geom_frictions = {}
            object_root_id = model.body(object_id).rootid[0]
            for geom_id in range(model.ngeom):
                geom_body_id = model.geom(geom_id).bodyid.item()
                geom_root_id = model.body(geom_body_id).rootid[0]
                if geom_root_id == object_root_id:
                    geom_frictions[geom_id] = np.array(model.geom_friction[geom_id]).copy()
            original_dynamics[obj.name] = {
                "mass": total_mass,
                "inertia": inertia,
                "geom_frictions": geom_frictions,
            }
        print(f"  Captured {len(original_dynamics)} objects for dynamics")

    # Run randomization
    print("\nRunning randomization...")
    randomize_scene(
        model,
        data,
        enable_texture=config.enable_texture_randomization,
        enable_dynamics=config.enable_dynamics_randomization,
        enable_lighting=config.enable_lighting_randomization,
        seed=config.seed,
    )

    # Debug: Compare original vs randomized values
    print("\n" + "=" * 60)
    print("DEBUG: Randomization Verification")
    print("=" * 60)

    if config.enable_lighting_randomization:
        print("\nLighting Randomization Debug:")
        lights_changed = 0
        for light_id, original in original_lighting.items():
            light = model.light(light_id)
            pos_diff = np.linalg.norm(np.array(light.pos) - original["pos"])
            dir_diff = np.linalg.norm(np.array(light.dir) - original["dir"])
            specular_diff = np.linalg.norm(np.array(light.specular) - original["specular"])
            ambient_diff = np.linalg.norm(np.array(light.ambient) - original["ambient"])
            diffuse_diff = np.linalg.norm(np.array(light.diffuse) - original["diffuse"])
            light_active = light.active.item()
            active_diff = abs(int(light_active) - original["active"])

            changed = (
                pos_diff > 1e-6
                or dir_diff > 1e-6
                or specular_diff > 1e-6
                or ambient_diff > 1e-6
                or diffuse_diff > 1e-6
                or active_diff > 0
            )

            if changed:
                lights_changed += 1
                print(f"  ✓ Light {light_id} ({original['name']}) CHANGED:")
                if pos_diff > 1e-6:
                    print(f"      pos: {original['pos']} -> {light.pos} (diff: {pos_diff:.6f})")
                if dir_diff > 1e-6:
                    print(f"      dir: {original['dir']} -> {light.dir} (diff: {dir_diff:.6f})")
                if specular_diff > 1e-6:
                    print(
                        f"      specular: {original['specular']} -> {light.specular} (diff: {specular_diff:.6f})"
                    )
                if ambient_diff > 1e-6:
                    print(
                        f"      ambient: {original['ambient']} -> {light.ambient} (diff: {ambient_diff:.6f})"
                    )
                if diffuse_diff > 1e-6:
                    print(
                        f"      diffuse: {original['diffuse']} -> {light.diffuse} (diff: {diffuse_diff:.6f})"
                    )
                if active_diff > 0:
                    print(f"      active: {original['active']} -> {light.active}")
            else:
                print(f"  ✗ Light {light_id} ({original['name']}) NOT CHANGED")
        print(f"\n  Summary: {lights_changed}/{len(original_lighting)} lights were randomized")

    if config.enable_dynamics_randomization:
        print("\nDynamics Randomization Debug:")
        objects_changed = 0
        for obj_name, original in original_dynamics.items():
            # Find the object again (it might have been recreated)
            objects = get_all_bodies_with_joints_as_mlspaces_objects(model, data)
            obj = next((o for o in objects if o.name == obj_name), None)
            if obj is None:
                continue

            object_id = obj.object_id
            current_mass = float(model.body_subtreemass[object_id])
            current_inertia = np.array(model.body_inertia[object_id])
            mass_diff = abs(current_mass - original["mass"])
            inertia_diff = np.linalg.norm(current_inertia - original["inertia"])

            friction_changed = False
            object_root_id = model.body(object_id).rootid[0]
            for geom_id, original_friction in original["geom_frictions"].items():
                geom_body_id = model.geom(geom_id).bodyid.item()
                geom_root_id = model.body(geom_body_id).rootid[0]
                if geom_root_id == object_root_id:
                    current_friction = np.array(model.geom_friction[geom_id])
                    if np.linalg.norm(current_friction - original_friction) > 1e-6:
                        friction_changed = True
                        break

            changed = mass_diff > 1e-6 or inertia_diff > 1e-6 or friction_changed

            if changed:
                objects_changed += 1
                print(f"  ✓ Object '{obj_name}' CHANGED:")
                if mass_diff > 1e-6:
                    print(
                        f"      mass: {original['mass']:.6f} -> {current_mass:.6f} (diff: {mass_diff:.6f})"
                    )
                if inertia_diff > 1e-6:
                    print(f"      inertia diff: {inertia_diff:.6f}")
                if friction_changed:
                    print("      friction: changed")
            else:
                print(f"  ✗ Object '{obj_name}' NOT CHANGED")
        print(f"\n  Summary: {objects_changed}/{len(original_dynamics)} objects were randomized")

    print("=" * 60 + "\n")

    # Capture camera image AFTER randomization (for comparison with baseline)
    if config.enable_texture_randomization or config.enable_lighting_randomization:
        print("Capturing camera image after randomization...")
        camera_randomized = capture_camera_image(
            model, data, RANDOMIZED_TEST_DATA_DIR, suffix="randomized"
        )
        print(f"  Captured {len(camera_randomized)} randomized camera images")

    # Capture baselines AFTER randomization (randomized state)
    # These represent the expected randomized values when using the same seed
    print("\nCapturing baselines (after randomization - randomized state)...")
    lighting_baseline = None
    dynamics_baseline = None
    texture_baseline = None

    if config.enable_lighting_randomization:
        print("  Capturing lighting baseline...")
        lighting_baseline = capture_lighting_baseline(model)
        print(f"    Captured {len(lighting_baseline)} lights")

    if config.enable_dynamics_randomization:
        print("  Capturing dynamics baseline...")
        dynamics_baseline = capture_dynamics_baseline(model, data)
        print(f"    Captured {len(dynamics_baseline)} objects")

    if config.enable_texture_randomization:
        print("  Capturing texture/material baseline...")
        texture_baseline = capture_texture_material_baseline(model)
        print(f"    Captured {len(texture_baseline['geom_materials'])} geom materials")

    # Save baseline data as NPY files
    scene_stem = Path(scene_path).stem
    dynamics_file = RANDOMIZED_TEST_DATA_DIR / f"baseline_{scene_stem}_dynamics.npy"
    texture_file = RANDOMIZED_TEST_DATA_DIR / f"baseline_{scene_stem}_texture_material.npy"
    lighting_file = RANDOMIZED_TEST_DATA_DIR / f"baseline_{scene_stem}_lighting.npy"

    # Save dynamics baseline as NPY (if captured)
    if dynamics_baseline is not None:
        np.save(dynamics_file, dynamics_baseline, allow_pickle=True)
        print(f"  Saved dynamics to: {dynamics_file}")

    # Save texture/material baseline as NPY (if captured)
    if texture_baseline is not None:
        np.save(texture_file, texture_baseline, allow_pickle=True)
        print(f"  Saved texture/material to: {texture_file}")

    # Save lighting baseline as NPY (if captured)
    if lighting_baseline is not None:
        np.save(lighting_file, lighting_baseline, allow_pickle=True)
        print(f"  Saved lighting to: {lighting_file}")

    # Save metadata as JSON (for easy reading)
    # Use relative paths (relative to RANDOMIZED_TEST_DATA_DIR)
    baseline_metadata = {
        "scene_path": scene_path,
        "camera_images": camera_baseline,
        "model_info": {
            "ngeom": int(model.ngeom),
            "nlight": int(model.nlight),
            "ntex": int(model.ntex),
            "nbody": int(model.nbody),
        },
        "files": {
            "dynamics": dynamics_file.name,  # Just filename, relative to metadata directory
            "texture_material": texture_file.name,
            "lighting": lighting_file.name,
        },
    }

    # Update camera image paths to be relative
    for cam_name, cam_data in baseline_metadata["camera_images"].items():
        if "image_path" in cam_data and cam_data["image_path"]:
            image_path = Path(cam_data["image_path"])
            # Make path relative to RANDOMIZED_TEST_DATA_DIR
            try:
                rel_path = image_path.relative_to(RANDOMIZED_TEST_DATA_DIR)
                baseline_metadata["camera_images"][cam_name]["image_path"] = str(rel_path)
            except ValueError:
                # If path is not relative, just use filename
                baseline_metadata["camera_images"][cam_name]["image_path"] = image_path.name

    metadata_file = RANDOMIZED_TEST_DATA_DIR / f"baseline_{scene_stem}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(baseline_metadata, f, indent=2)

    print("\nBaseline data saved:")
    if dynamics_baseline is not None:
        print(f"  Dynamics: {len(dynamics_baseline)} objects -> {dynamics_file}")
    if texture_baseline is not None:
        print(
            f"  Texture/Material: {len(texture_baseline['geom_materials'])} geoms -> {texture_file}"
        )
    if lighting_baseline is not None:
        print(f"  Lighting: {len(lighting_baseline)} lights -> {lighting_file}")
    print(f"  Camera images: {len(camera_baseline)} images (PNG files)")
    print(f"  Metadata: {metadata_file}")


def get_scene_path(house_type: str, house_index: int) -> str:
    """Get scene path from house type and index.

    Prefers "physics" variant if available, otherwise "base", otherwise first available.
    """
    scenes = get_scenes(house_type)
    scene_path_or_dict = scenes["train"][house_index]

    if isinstance(scene_path_or_dict, dict):
        # Prefer "physics" variant for randomization tests, then "base", then first available
        scene_path = (
            scene_path_or_dict.get("physics")
            or scene_path_or_dict.get("base")
            or list(scene_path_or_dict.values())[0]
        )
    else:
        scene_path = scene_path_or_dict

    if scene_path is None:
        raise ValueError(f"Scene not found for {house_type} house_index {house_index}")

    return str(scene_path)


def main():
    RANDOMIZED_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = RandomizationTestConfig()
    scene_path = get_scene_path(config.house_type, config.house_index)
    generate_test_data(scene_path, config)


if __name__ == "__main__":
    main()
