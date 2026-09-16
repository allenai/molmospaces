import contextlib
from pathlib import Path

import mujoco as mj

from molmo_spaces.env.data_views import MlSpacesArticulationObject, MlSpacesObject
from molmo_spaces.env.scene.containment import (
    is_body_com_within_box_site,
    is_body_within_any_site,
    is_body_within_site_in_freespace,
)
from molmo_spaces.env.scene.thor.bathroom import ShowerDoor
from molmo_spaces.env.scene.thor.cabinet import Cabinet
from molmo_spaces.env.scene.thor.drawer import Drawer
from molmo_spaces.env.scene.thor.kitchen import Dishwasher, Oven, Stoveknob

iTHOR_CATEGORIES = ["Cabinet", "Drawer", "ShowerDoor", "Oven", "Dishwasher", "StoveKnob"]
DEFAULT_Z_OFFSET_OBJS_WITHIN_SITES = 0.025


def fix_remove_objects_within_inner_sites(spec: mj.MjSpec) -> list[str]:
    model: mj.MjModel = spec.compile()
    data: mj.MjData = mj.MjData(model)
    mj.mj_forward(model, data)

    # Collect all root bodies from the model
    root_bodies: list[mj.MjsBody] = []
    current_body_spec = spec.worldbody.first_body()
    while current_body_spec:
        root_bodies.append(current_body_spec)
        current_body_spec = spec.worldbody.next_body(current_body_spec)

    bodies_deleted = []
    for idx in range(len(root_bodies)):
        body_spec: mj.MjsBody = root_bodies[idx]
        body_id = model.body(body_spec.name).id
        for site_id in range(model.nsite):
            if is_body_com_within_box_site(site_id, body_id, model, data):
                in_free_space, _, _ = is_body_within_site_in_freespace(
                    site_id, body_id, model, data
                )
                if not in_free_space:  # it's actually inside a drawer or similar
                    bodies_deleted.append(body_spec.name)
                    spec.delete(body_spec)
                    break
    return bodies_deleted


def fix_move_objects_within_inner_sites_up_abit(
    spec: mj.MjSpec, z_offset: float = DEFAULT_Z_OFFSET_OBJS_WITHIN_SITES
) -> None:
    model: mj.MjModel = spec.compile()
    data: mj.MjData = mj.MjData(model)
    mj.mj_forward(model, data)

    # Collect all root bodies from the model
    root_bodies: list[mj.MjsBody] = []
    root_bodies_z0: list[float] = []
    current_body_spec = spec.worldbody.first_body()
    while current_body_spec:
        root_bodies.append(current_body_spec)
        root_bodies_z0.append(current_body_spec.pos[2].item())
        current_body_spec = spec.worldbody.next_body(current_body_spec)

    for idx in range(len(root_bodies)):
        body_spec: mj.MjsBody = root_bodies[idx]
        body_id = model.body(body_spec.name).id
        is_within_site, _ = is_body_within_any_site(model, data, body_id)
        if is_within_site:
            body_spec.pos[2] = root_bodies_z0[idx] + z_offset


def fix_exclude_contact_floor_with_fridges(spec: mj.MjSpec) -> None:
    floor_handle = spec.body("floor")
    if not floor_handle:
        return

    fridges_bodies: list[mj.MjsBody] = []
    body_spec: mj.MjsBody = spec.worldbody.first_body()
    while body_spec:
        if "fridge" in body_spec.name.lower():
            spec.add_exclude(bodyname1="floor", bodyname2=body_spec.name)
            fridges_bodies.append(body_spec)
        body_spec = spec.worldbody.next_body(body_spec)


def fix_remove_all_toasters(spec: mj.MjSpec) -> None:
    toasters_handles: list[mj.MjsBody] = []
    root_body: mj.MjsBody = spec.worldbody.first_body()
    while root_body is not None:
        if "toaster" in root_body.name.lower():
            toasters_handles.append(root_body)
        root_body = spec.worldbody.next_body(root_body)

    for toaster_body in toasters_handles:
        spec.delete(toaster_body)


def get_all_bodies_with_joints_as_mlspaces_objects(
    model: mj.MjModel, data: mj.MjData
) -> list[MlSpacesObject]:
    """
    Get all bodies with joints as MlSpacesObject instances.

    This function finds all bodies in the model that have joints (movable bodies)
    and creates MlSpacesObject instances for them. Bodies without valid names or
    that fail to create MlSpacesObject instances are skipped.

    Args:
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        List of MlSpacesObject instances for all bodies with joints that could be
        successfully created. Bodies that fail to create MlSpacesObject instances
        are silently skipped.
    """
    mlspaces_objects = []
    bodies_with_joints = []

    # Iterate through all bodies (skip body 0 which is worldbody)
    for body_id in range(1, model.nbody):
        # Check if this body has a joint
        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr >= 0:
            # This body has a joint, check if it's a free joint or any non-fixed joint
            # In MuJoCo, joints are either FREE, BALL, HINGE, SLIDE (no fixed joint type)
            # If a body has a joint, it's movable
            joint_type = model.jnt_type[jnt_adr]
            # Get body name
            name_adr = model.name_bodyadr[body_id]
            if name_adr >= 0:
                name_bytes = model.names[name_adr:]
                body_name = name_bytes.split(b"\x00")[0].decode("utf-8")
                if body_name:
                    bodies_with_joints.append((body_id, body_name, joint_type))

    # Create MlSpacesObject instances for bodies with joints
    for _, body_name, _ in bodies_with_joints:
        with contextlib.suppress(Exception):
            obj = MlSpacesObject(object_name=body_name, data=data)
            mlspaces_objects.append(obj)

    return mlspaces_objects


def modify_mjmodel_thor_articulated(
    model: mj.MjModel, data
) -> dict[str, list[MlSpacesArticulationObject]]:
    body_name2id = {model.body(i).name: i for i in range(0, model.nbody)}

    # get root bodies
    root_bodies_dict = {
        "Cabinet": [],
        "Drawer": [],
        "ShowerDoor": [],
        "Oven": [],
        "Dishwasher": [],
        "StoveKnob": [],
    }

    root_bodies = set()
    for i in range(0, model.nbody):
        rootid = model.body(i).rootid.item()
        root_body_name = model.body(rootid).name
        root_bodies.add(root_body_name)

    for root_body_name in root_bodies:
        category = next(
            (i for i in iTHOR_CATEGORIES if root_body_name.lower().startswith(i.lower())), None
        )
        if category == "Cabinet":
            cabinet = Cabinet(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(cabinet)
        elif category == "Drawer":
            drawer = Drawer(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(drawer)
        elif category == "ShowerDoor":
            shower_door = ShowerDoor(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(shower_door)
        elif category == "Oven":
            oven = Oven(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(oven)
        elif category == "Dishwasher":
            dishwasher = Dishwasher(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(dishwasher)
        elif category == "StoveKnob":
            stove_knob = Stoveknob(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(stove_knob)

    return root_bodies_dict


def load_env_with_objects(
    xml_path: str,
) -> tuple[mj.MjModel, dict[str, list[MlSpacesArticulationObject]]]:
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    return model, modify_mjmodel_thor_articulated(model, data)


def load_env_with_objects_with_tweaks(
    xml_path: str,
    remove_objects_within_inner_sites: bool = False,
    move_objects_within_sites_up_abit: bool = False,
    remove_all_toasters: bool = False,
    # exclude_floor_contact_with_fridges: bool = False,
) -> tuple[mj.MjModel, dict[str, list[MlSpacesArticulationObject]], dict]:
    tweaks = {"bodies_deleted": []}

    spec = mj.MjSpec.from_file(xml_path)

    if remove_objects_within_inner_sites:
        tweaks["bodies_deleted"].extend(fix_remove_objects_within_inner_sites(spec))
    if move_objects_within_sites_up_abit:
        fix_move_objects_within_inner_sites_up_abit(spec, DEFAULT_Z_OFFSET_OBJS_WITHIN_SITES)
    if remove_all_toasters:
        fix_remove_all_toasters(spec)
    # if exclude_floor_contact_with_fridges:
    #     fix_exclude_contact_floor_with_fridges(spec)

    model = spec.compile()

    # filepath = Path(xml_path)
    # filepath_new = filepath.parent / f"{filepath.stem}_new.xml"
    # with open(filepath_new, "w") as fhandle:
    #     fhandle.write(spec.to_xml())

    data = mj.MjData(model)
    body_name2id = {model.body(i).name: i for i in range(0, model.nbody)}

    # get root bodies
    root_bodies_dict = {
        "Cabinet": [],
        "Drawer": [],
        "ShowerDoor": [],
        "Oven": [],
        "Dishwasher": [],
        "StoveKnob": [],
    }

    root_bodies = set()
    for i in range(0, model.nbody):
        rootid = model.body(i).rootid.item()
        root_body_name = model.body(rootid).name
        root_bodies.add(root_body_name)

    for root_body_name in root_bodies:
        category = next(
            (i for i in iTHOR_CATEGORIES if root_body_name.lower().startswith(i.lower())), None
        )
        if category == "Cabinet":
            cabinet = Cabinet(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(cabinet)
        elif category == "Drawer":
            drawer = Drawer(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(drawer)
        elif category == "ShowerDoor":
            shower_door = ShowerDoor(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(shower_door)
        elif category == "Oven":
            oven = Oven(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(oven)
        elif category == "Dishwasher":
            dishwasher = Dishwasher(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(dishwasher)
        elif category == "StoveKnob":
            stove_knob = Stoveknob(root_body_name, data, body_name2id)
            root_bodies_dict[category].append(stove_knob)

    return model, root_bodies_dict, tweaks


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    import mujoco.viewer as mjviewer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--house",
        type=str,
        default="",
        help="The procthor/ithor house to test",
    )
    parser.add_argument(
        "--fix-remove-objects-within-sites",
        action="store_true",
        help="Whether or not to apply the fix that removes bodies within inner sites",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether or not to use the visualizer when checking",
    )

    args = parser.parse_args()
    house_path = Path(args.house)

    old_model: mj.MjModel | None = None
    new_model: mj.MjModel | None = None
    if args.house != "" and house_path.exists():
        old_model = mj.MjModel.from_xml_path(house_path.as_posix())
        if args.fix_remove_objects_within_sites:
            new_model, _, tweaks = load_env_with_objects_with_tweaks(
                house_path.as_posix(), remove_objects_within_inner_sites=True
            )
        else:
            new_model, _, tweaks = load_env_with_objects_with_tweaks(house_path.as_posix())

    if not args.visualize or new_model is None:
        sys.exit(0)

    data = mj.MjData(new_model)

    with mjviewer.launch_passive(
        new_model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            t_start = data.time
            while data.time - t_start < 1.0 / 60.0:
                mj.mj_step(new_model, data)
            viewer.sync()
