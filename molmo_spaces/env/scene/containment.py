import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

DISTANCE_RAYCAST_THRESHOLD = 0.5
OFFSET_UP_RAYCAST = 0.05
SITE_SIZE_TOLERANCE = np.array([0.05, 0.05, 0.05], dtype=np.float64)


def is_body_com_within_box_site(
    site_id: int, body_id: int, model: mj.MjModel, data: mj.MjData
) -> bool:
    site_type = model.site_type[site_id].item()
    if site_type != mj.mjtGeom.mjGEOM_BOX:
        return False

    # Make sure we're not considering site-body comparison of sites that belong to a body already
    site_bodyid = model.site_bodyid[site_id].item()
    site_rootid = model.body_rootid[site_bodyid].item()
    tgt_rootid = model.body_rootid[body_id].item()
    if site_rootid in (tgt_rootid, body_id):
        return False

    # Transform the body's position into the site's local frame, and do an AABB check in there
    body_com_pos = data.xpos[body_id]
    site_xpos = data.site_xpos[site_id]
    site_xmat = data.site_xmat[site_id].reshape(3, 3)
    site_size = model.site_size[site_id]

    plocal = body_com_pos - site_xpos
    plocal = site_xmat.T @ plocal

    # Converted is in the local frame of the site, so just check if it's within its AABB
    result = bool(np.all(np.abs(plocal) < (site_size + SITE_SIZE_TOLERANCE)))

    return result


def is_body_within_any_site(model: mj.MjModel, data: mj.MjData, body_id: int) -> tuple[bool, int]:
    is_within_site = False
    site_id_within = -1
    for site_id in range(model.nsite):
        if is_body_com_within_box_site(site_id, body_id, model, data):
            is_within_site = True
            site_id_within = site_id
            break

    return is_within_site, site_id_within


def does_body_aabb_intersect_box_site(
    site_id: int, body_id: int, model: mj.MjModel, data: mj.MjData
) -> bool:
    # TODO(wilbert): implement this helper function, as it might work better for the tests
    pass


def is_body_within_site_in_freespace(
    site_id: int, body_id: int, model: mj.MjModel, data: mj.MjData
) -> tuple[bool, float, str]:
    site_size = model.site_size[site_id]
    geomid = np.zeros(1, dtype=np.int32)
    world_up = np.array([0, 0, 1], dtype=np.float64)
    pnt = data.xpos[body_id] + OFFSET_UP_RAYCAST * world_up

    distance_up = mj.mj_ray(model, data, pnt, world_up, None, 1, body_id, geomid)

    if distance_up == -1.0:  # No intersection, so there's space above
        return True, -1.0, ""

    is_there_space_above = distance_up > min(DISTANCE_RAYCAST_THRESHOLD, 2.0 * site_size[2])

    geom_name = ""
    if not is_there_space_above:  # just in case, try excluding the first body we touched
        geom_bodyid = model.geom_bodyid[geomid.item()].item()
        geom_rootid = model.body_rootid[geom_bodyid].item()
        if geom_rootid == body_id:
            distance_up = mj.mj_ray(model, data, pnt, world_up, None, 1, geom_bodyid, geomid)
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geomid.item())

    if distance_up == -1.0:  # No intersection, so there's space above
        return True, -1.0, ""

    is_there_space_above = distance_up > min(DISTANCE_RAYCAST_THRESHOLD, 2.0 * site_size[2])

    return is_there_space_above, distance_up, geom_name


if __name__ == "__main__":
    import mujoco.viewer as mjviewer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--house",
        type=str,
        default="",
        help="The path to the house to be loaded for testing",
    )
    parser.add_argument(
        "--body",
        type=str,
        default="",
        help="The body to test with for site detection (optional, comes from the test results)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether or not to use the visualizer for the checks",
    )

    args = parser.parse_args()

    if args.house == "":
        print("Must give --house parameter when running this script")
        sys.exit(0)

    house_path = Path(args.house)
    if not house_path.exists():
        print(f"House: {args.house} doesnt' exist")
        sys.exit(0)

    model = mj.MjModel.from_xml_path(house_path.as_posix())
    data = mj.MjData(model)

    mj.mj_forward(model, data)

    @dataclass
    class ContextFlags:
        dirty_reset = False
        dirty_pause = False
        dirty_test_com_inside = False
        dirty_test_aabb_intersect = False
        dirty_test_in_free_space = False
        dirty_test_with_all_sites = False

    class Context:
        def __init__(self) -> None:
            self.body_id = -1
            self.site_id = -1
            self.flags = ContextFlags()

    context = Context()

    if args.body != "":
        body_handle = model.body(args.body)
        if body_handle is not None:
            context.body_id = body_handle.id
            print(f"Checking for specific body: {args.body}")

    def run_check_body_com_all_sites(body_id: int) -> tuple[bool, str]:
        found_within_site = False
        found_site_name = ""
        for site_id in range(model.nsite):
            if is_body_com_within_box_site(site_id, body_id, model, data):
                in_free_space, _, _ = is_body_within_site_in_freespace(
                    site_id, body_id, model, data
                )
                if not in_free_space:
                    found_within_site = True
                    found_site_name = model.site(site_id).name
                    break
        return found_within_site, found_site_name

    if not args.visualize:
        if context.body_id != -1:
            found_within_site, found_site_name = run_check_body_com_all_sites(context.body_id)
            body_name = model.body(context.body_id).name
            if found_within_site:
                print(f"Body {body_name} COM is within site {found_site_name}")
            else:
                print(f"Body {body_name} COM not within any sites")
        sys.exit(0)

    def key_callback(keycode: int) -> None:
        global model, data, context

        if keycode == 265:  # up arrow key
            context.body_id = (context.body_id + 1) % model.nbody
            print(f"Using body: {model.body(context.body_id).name}")
        elif keycode == 264:  # down arrow key
            context.body_id = (context.body_id - 1) % model.nbody
            print(f"Using body: {model.body(context.body_id).name}")
        elif keycode == 262:  # right arrow key
            context.site_id = (context.site_id + 1) % model.nsite
            print(f"Using site: {model.site(context.site_id).name}")
        elif keycode == 263:  # left arrow key
            context.site_id = (context.site_id - 1) % model.nsite
            print(f"Using site: {model.site(context.site_id).name}")
        elif keycode == 259:  # backspace key
            context.flags.dirty_reset = True
        elif keycode == 32:  # space key
            context.flags.dirty_pause = not context.flags.dirty_pause
            print("Paused" if context.flags.dirty_pause else "Running")
        elif keycode == 81:  # Q key
            context.flags.dirty_test_com_inside = True
        elif keycode == 80:  # P key
            context.flags.dirty_test_aabb_intersect = True
        elif keycode == 79:  # O key
            context.flags.dirty_test_with_all_sites = not context.flags.dirty_test_with_all_sites
            msg = "all sites" if context.flags.dirty_test_with_all_sites else "single site"
            print(f"Testing with {msg}")

    with mjviewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            if context.flags.dirty_reset:
                context.flags.dirty_reset = False
                mj.mj_resetData(model, data)

            if not context.flags.dirty_pause:
                t_start = data.time
                while data.time - t_start < 1.0 / 60.0:
                    mj.mj_step(model, data)
            else:
                mj.mj_forward(model, data)

            if context.flags.dirty_test_com_inside:
                context.flags.dirty_test_com_inside = False
                if context.flags.dirty_test_with_all_sites:
                    if context.body_id != -1:
                        found_within_site, found_site_name = run_check_body_com_all_sites(
                            context.body_id
                        )
                        body_name = model.body(context.body_id).name
                        if found_within_site:
                            body_name = model.body(context.body_id).name
                            print(f"Body {body_name} COM is within site {found_site_name}")
                        else:
                            print(f"Body {body_name} COM not within any sites")
                else:
                    if context.site_id != -1 and context.body_id != -1:
                        result = is_body_com_within_box_site(
                            context.site_id, context.body_id, model, data
                        )
                        print(f"is_body_com_within_box_site: {result}")

            viewer.sync()
