"""Env wrapper to randomize the scene.

## TEST RUNTIME RANDOMIZATION
- lightings (randomize intensity and color and position)
- textures (switch among preloaded textures and randomize material properties)
- dynamics (object mass, inertia, friction, density, etc.)

"""

import contextlib

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.env.arena.randomization.dynamics import DynamicsRandomizer
from molmo_spaces.env.arena.randomization.lighting import LightingRandomizer
from molmo_spaces.env.arena.randomization.texture import TextureRandomizer, setup_empty_materials


def test(
    scene_path: str,
    texture_paths: list[str] | None = None,
    relaunch_viewer_on_randomize: bool = True,
) -> None:
    """
    Test function to test the randomizers and visualize using MuJoCo passive viewer.

    Args:
        scene_path: Path to scene XML file. Model and data will be loaded from this file.
        texture_paths: Optional list of texture file paths for texture randomization.
            If None, uses textures already loaded in the scene XML.
        relaunch_viewer_on_randomize: If True, relaunches the viewer after each randomization
            to ensure model property changes (textures, lights) are visible. Default is True.

    Loads the scene, applies randomization, and visualizes the results.
    """
    import os
    import time

    # Load model and data from scene_path
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene path does not exist: {scene_path}")

    spec = mujoco.MjSpec.from_file(scene_path)
    setup_empty_materials(spec)
    model = spec.compile()

    print(f"Loading model from: {scene_path}")
    # model = MjModel.from_xml_path(scene_path)
    data = MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Loaded model: {model.ngeom} geoms, {model.nlight} lights, {model.ntex} textures")

    print("=" * 60)
    print("Testing Scene Randomizers")
    print("=" * 60)

    # Initialize randomizers
    print("\n1. Initializing randomizers...")
    lighting_randomizer = LightingRandomizer(
        model=model,
        randomize_position=True,
        randomize_direction=True,
        randomize_specular=True,
        randomize_ambient=True,
        randomize_diffuse=True,
        randomize_active=True,
        position_perturbation_size=0.2,
        direction_perturbation_size=0.5,
    )

    # Many materials in MuJoCo scenes have textures even if mat_texid isn't accessible
    # If we have textures in the model, we should enable randomization
    # Default: use textures from model (texture_paths=None)
    # Only use external files if explicitly provided
    # Enable if we found textures via mat_texid, OR if we have textures and materials (fallback)
    enable_texture_randomization = model.ntex > 0 or (
        texture_paths is not None and len(texture_paths) > 0
    )

    # Load scene_metadata from scene_path
    from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata

    scene_metadata = get_scene_metadata(scene_path)
    if scene_metadata is None:
        print(f"   Warning: Could not load scene metadata from {scene_path}")

    texture_randomizer = TextureRandomizer(
        model=model,
        randomize_geom_rgba=True,
        randomize_material_rgba=True,
        randomize_material_specular=True,
        randomize_material_shininess=True,
        randomize_texture=enable_texture_randomization,
        texture_paths=texture_paths,  # None = use model textures (default)
        scene_metadata=scene_metadata,
        rgba_perturbation_size=0.2,
    )

    if enable_texture_randomization:
        if texture_paths:
            print(f"   Texture randomization: using {len(texture_paths)} external texture files")
        else:
            # When using model textures, texture_bitmaps is empty (we use texture_ids for on-demand extraction)
            num_texture_ids = (
                len(texture_randomizer.texture_ids)
                if hasattr(texture_randomizer, "texture_ids")
                else 0
            )
            print(
                f"   Texture randomization: using {num_texture_ids} textures from model (default, on-demand extraction)"
            )
            if num_texture_ids == 0 and model.ntex > 0:
                print(
                    f"   Warning: Could not find 2D textures in model despite {model.ntex} textures existing"
                )

    dynamics_randomizer = DynamicsRandomizer(
        randomize_friction=True,
        randomize_mass=True,
        randomize_inertia=True,
        mass_perturbation_ratio=0.2,
        friction_perturbation_ratio=0.2,
        inertia_perturbation_ratio=0.2,
    )
    print("   ✓ Randomizers initialized")

    # Create MlSpacesObject instances for dynamics randomization
    # Find all bodies with free joints or any non-fixed joints
    from molmo_spaces.env.arena.arena_utils import get_all_bodies_with_joints_as_mlspaces_objects

    test_objects = get_all_bodies_with_joints_as_mlspaces_objects(model, data)

    # Count total bodies with joints for logging
    bodies_with_joints_count = 0
    for body_id in range(1, model.nbody):
        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr >= 0:
            bodies_with_joints_count += 1

    print(
        f"   ✓ Found {bodies_with_joints_count} bodies with joints ({len(test_objects)} successfully created as MlSpacesObject)"
    )
    if bodies_with_joints_count > 0 and len(test_objects) == 0:
        print(
            f"   ⚠ Warning: Found {bodies_with_joints_count} bodies with joints but couldn't create MlSpacesObject instances"
        )

    # Apply initial randomization before launching visualizer
    print("\n2. Applying initial randomization...")
    start_time = time.perf_counter()
    lighting_randomizer.randomize(data)
    lighting_time = time.perf_counter() - start_time
    print(f"   ✓ Lighting randomized ({lighting_time * 1000:.2f} ms)")

    start_time = time.perf_counter()
    if texture_randomizer.randomize_texture:
        # texture_randomizer.randomize(data)
        texture_randomizer.randomize_by_category(data)
        texture_time = time.perf_counter() - start_time
        print(f"   ✓ Textures randomized ({texture_time * 1000:.2f} ms)")
    else:
        texture_time = 0.0
        print("   ⚠ Texture randomization is disabled")

    dynamics_time = 0.0
    if test_objects:
        start_time = time.perf_counter()
        dynamics_randomizer.randomize_objects(test_objects)
        dynamics_time = time.perf_counter() - start_time
        print(f"   ✓ Dynamics randomized ({dynamics_time * 1000:.2f} ms)")
    else:
        print("   ⚠ No objects found for dynamics randomization")

    total_time = lighting_time + texture_time + dynamics_time
    print(f"   Total randomization time: {total_time * 1000:.2f} ms")

    # Ensure forward pass is done after initial randomization
    mujoco.mj_forward(model, data)

    print("\n3. Starting visualization...")
    if relaunch_viewer_on_randomize:
        print("   Viewer will be relaunched after each randomization")
    print("   Press ESC or close window to exit")

    # Launch passive viewer
    viewer = None
    step_count = 0
    should_relaunch = False

    while True:
        # Launch or relaunch viewer
        if viewer is None or should_relaunch:
            if viewer is not None:
                with contextlib.suppress(Exception):
                    viewer.close()
                viewer = None
                time.sleep(0.2)  # Allow viewer to fully close

            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 45.0
            viewer.cam.elevation = -30.0
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.5])
            should_relaunch = False

        # Check if viewer is still running
        if viewer is None or not viewer.is_running():
            break

        # Step simulation
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        # Re-randomize every 500 steps (including step 0 for immediate randomization)
        if step_count % 500 == 0:
            if step_count > 0:
                print(f"\n   Re-randomizing at step {step_count}...")

            # Capture light properties before randomization
            light_before = {}
            if lighting_randomizer.light_ids:
                for light_id in lighting_randomizer.light_ids:
                    light_before[light_id] = {
                        "pos": np.array(lighting_randomizer.get_pos(light_id)),
                        "dir": np.array(lighting_randomizer.get_dir(light_id)),
                        "specular": np.array(lighting_randomizer.get_specular(light_id)),
                        "ambient": np.array(lighting_randomizer.get_ambient(light_id)),
                        "diffuse": np.array(lighting_randomizer.get_diffuse(light_id)),
                        "active": lighting_randomizer.get_active(light_id),
                    }

            # Capture dynamics properties before randomization
            dynamics_before = {}
            if test_objects:
                for obj in test_objects:
                    model = obj.mj_model
                    object_id = obj.object_id
                    object_root_id = model.body(object_id).rootid[0]

                    # Get all bodies belonging to this object
                    body_ids = [object_id]
                    for body_id in range(model.nbody):
                        if body_id != object_id:
                            body_root_id = model.body(body_id).rootid[0]
                            if body_root_id == object_root_id:
                                body_ids.append(body_id)

                    # Get total mass of the object (including all descendant bodies)
                    total_mass = float(model.body_subtreemass[object_id])
                    # Inertia is typically on the root body
                    inertia = np.array(model.body_inertia[object_id])

                    # Get friction for all geoms
                    geom_frictions = {}
                    for geom_id in range(model.ngeom):
                        geom_body_id = model.geom(geom_id).bodyid.item()
                        geom_root_id = model.body(geom_body_id).rootid[0]
                        if geom_root_id == object_root_id:
                            geom_frictions[geom_id] = np.array(model.geom_friction[geom_id])

                    dynamics_before[obj.name] = {
                        "mass": total_mass,
                        "inertia": inertia,
                        "body_ids": body_ids,
                        "body_masses": {bid: float(model.body_mass[bid]) for bid in body_ids},
                        "geom_frictions": geom_frictions,
                    }

            start_time = time.perf_counter()
            lighting_randomizer.randomize(data)
            lighting_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            if texture_randomizer.randomize_texture:
                # texture_randomizer.randomize(data)
                texture_randomizer.randomize_by_category(data)
                texture_time = time.perf_counter() - start_time
            else:
                texture_time = 0.0

            dynamics_time = 0.0
            if test_objects:
                start_time = time.perf_counter()
                dynamics_randomizer.randomize_objects(test_objects)
                dynamics_time = time.perf_counter() - start_time

            total_time = lighting_time + texture_time + dynamics_time
            print(
                f"   Randomization complete: {total_time * 1000:.2f} ms "
                f"(lighting: {lighting_time * 1000:.2f} ms, "
                f"texture: {texture_time * 1000:.2f} ms, "
                f"dynamics: {dynamics_time * 1000:.2f} ms)"
            )

            # Print light property diffs
            if light_before and step_count > 0:
                print("\n   Light property changes:")
                for light_id, before in light_before.items():
                    after_pos = np.array(lighting_randomizer.get_pos(light_id))
                    after_dir = np.array(lighting_randomizer.get_dir(light_id))
                    after_specular = np.array(lighting_randomizer.get_specular(light_id))
                    after_ambient = np.array(lighting_randomizer.get_ambient(light_id))
                    after_diffuse = np.array(lighting_randomizer.get_diffuse(light_id))
                    after_active = lighting_randomizer.get_active(light_id)

                    pos_diff = after_pos - before["pos"]
                    dir_diff = after_dir - before["dir"]
                    specular_diff = after_specular - before["specular"]
                    ambient_diff = after_ambient - before["ambient"]
                    diffuse_diff = after_diffuse - before["diffuse"]
                    active_diff = after_active - before["active"]

                    print(f"     Light {light_id}:")
                    print(f"       pos diff: {pos_diff} (norm: {np.linalg.norm(pos_diff):.6f})")
                    print(f"       dir diff: {dir_diff} (norm: {np.linalg.norm(dir_diff):.6f})")
                    print(
                        f"       specular diff: {specular_diff} (norm: {np.linalg.norm(specular_diff):.6f})"
                    )
                    print(
                        f"       ambient diff: {ambient_diff} (norm: {np.linalg.norm(ambient_diff):.6f})"
                    )
                    print(
                        f"       diffuse diff: {diffuse_diff} (norm: {np.linalg.norm(diffuse_diff):.6f})"
                    )
                    print(f"       active diff: {active_diff}")

            # Print dynamics property diffs
            if dynamics_before and test_objects and step_count > 0:
                print("\n   Dynamics property changes:")
                for obj in test_objects:
                    if obj.name in dynamics_before:
                        model = obj.mj_model
                        object_id = obj.object_id
                        before = dynamics_before[obj.name]

                        # Get total mass of the object (including all descendant bodies)
                        body_ids = before.get("body_ids", [object_id])
                        after_mass = float(model.body_subtreemass[object_id])
                        after_inertia = np.array(model.body_inertia[object_id])

                        mass_diff = after_mass - before["mass"]
                        inertia_diff = after_inertia - before["inertia"]

                        print(f"     {obj.name}:")
                        print(
                            f"       mass diff: {mass_diff:.6f} ({before['mass']:.6f} -> {after_mass:.6f})"
                        )
                        if len(body_ids) > 1:
                            print(
                                f"         (object has {len(body_ids)} bodies, showing total mass)"
                            )
                        print(
                            f"       inertia diff: {inertia_diff} (norm: {np.linalg.norm(inertia_diff):.6f})"
                        )

                        # Get friction diffs
                        object_root_id = model.body(object_id).rootid[0]
                        friction_diffs = []
                        for geom_id in range(model.ngeom):
                            geom_body_id = model.geom(geom_id).bodyid.item()
                            geom_root_id = model.body(geom_body_id).rootid[0]
                            if geom_root_id == object_root_id:
                                if geom_id in before["geom_frictions"]:
                                    after_friction = np.array(model.geom_friction[geom_id])
                                    friction_diff = (
                                        after_friction - before["geom_frictions"][geom_id]
                                    )
                                    friction_diffs.append((geom_id, friction_diff))

                        if friction_diffs:
                            print("       friction diffs (geom_id: diff):")
                            for geom_id, friction_diff in friction_diffs[:5]:  # Show first 5
                                print(
                                    f"         geom_{geom_id}: {friction_diff} (norm: {np.linalg.norm(friction_diff):.6f})"
                                )
                            if len(friction_diffs) > 5:
                                print(f"         ... and {len(friction_diffs) - 5} more geoms")

            # Ensure forward pass is done after all randomizations
            mujoco.mj_forward(model, data)
            viewer.sync()

            # If relaunch_viewer_on_randomize is True, close and relaunch viewer to force refresh
            if relaunch_viewer_on_randomize and step_count > 0:
                if viewer is not None:
                    with contextlib.suppress(Exception):
                        viewer.close()
                    viewer = None
                should_relaunch = True
                time.sleep(0.1)
                continue

        step_count += 1

        # Small sleep to control frame rate
        time.sleep(0.01)

    # Clean up viewer if still open
    if viewer is not None:
        viewer.close()

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test(scene_path="assets/scenes/ithor/FloorPlan1_physics.xml")
    # test(scene_path="assets/scenes/procthor-10k-train/train_8.xml")
