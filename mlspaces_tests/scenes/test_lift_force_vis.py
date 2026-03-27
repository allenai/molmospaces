import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjviewer
import numpy as np

from molmo_spaces.env.arena.arena_utils import load_env_with_objects_with_tweaks
from molmo_spaces.env.arena.scene_tweaks import (
    is_body_com_within_box_site,
    is_body_within_any_site,
    is_body_within_site_in_freespace,
)

TIMESTEP = 0.002
TEST_TIME = 0.2  # 100 steps = 0.2 seconds
INITIAL_SETTLE_TIME = 0.5  # Settle for 1 second
TEST_STEPS = int(TEST_TIME / TIMESTEP)
INITIAL_SETTLE_STEPS = int(INITIAL_SETTLE_TIME / TIMESTEP)
MAX_LIFT_FORCE = 30
MIN_LIFT_FORCE = 0.05
DISTANCE_TO_LIFT = 0.02  # 5cm
DISTANCE_TO_LIFT_WITHIN_SITE = 0.02  # 2cm
DISTANCE_RAYCAST_THRESHOLD = 0.5
OFFSET_UP_RAYCAST = 0.05
ESTIMATED_FORCE_MULTIPLIER = 3
GRAVITY_EXTRA_VALUE = 0.1

ALL_PICKUP_TYPES_ITHOR = [
    "AlarmClock",
    "AluminumFoil",
    "Apple",
    "AppleSliced",
    "Book",
    "Boots",
    "Bottle",
    "Bowl",
    "Box",
    "Bread",
    "BreadSliced",
    "ButterKnife",
    "Candle",
    "CD",
    "CellPhone",
    "Cloth",
    "CreditCard",
    "Cup",
    "DishSponge",
    "Dumbbell",
    "Egg",
    "EggCracked",
    "Fork",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "Lettuce",
    "LettuceSliced",
    "Mug",
    "Newspaper",
    "Pan",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "PotatoSliced",
    "RemoteControl",
    "SaltShaker",
    "ScrubBrush",
    "SoapBar",
    "SoapBottle",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "TableTopDecor",
    "TeddyBear",
    "TennisRacket",
    "TissueBox",
    "ToiletPaper",
    "Tomato",
    "TomatoSliced",
    "Towel",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
]


model: mj.MjModel | None = None
data: mj.MjData | None = None


def get_all_geoms_from_body(body_id: int, model: mj.MjModel) -> list[int]:
    geoms_ids = []
    # AFAIK there's not a direct way to traverse the kinematic tree starting at a body, so the
    # best we can do for now is to check first for all the bodies that share the given as root
    for geom_id in range(model.ngeom):
        geom_bodyid = model.geom_bodyid[geom_id].item()
        geom_rootid = model.body_rootid[geom_bodyid].item()
        if geom_rootid == body_id:
            geoms_ids.append(geom_id)
    return geoms_ids


def make_body_frictionless(body_id: int, model: mj.MjModel) -> None:
    if body_id < model.nbody and body_id >= 0:
        geoms_ids = get_all_geoms_from_body(body_id, model)
        model.geom_condim[geoms_ids] = 1


class TestState(Enum):
    IDLE = 1
    RUNNING_INITIAL_SETTLE = 1
    RUNNING_MAIN_TEST = 2
    DONE = 3


@dataclass
class BodyInfo:
    body_id: int = -1
    body_name: str = ""
    body_totalmass: float = 0.0
    body_weight: float = 0.0
    body_est_force: float = 0.0

    body_pos_bef_settle: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    body_pos_start: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    body_pos_end: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    site_within: bool = False
    site_within_id: int = -1
    site_within_name: str = ""

    free_space: bool = False
    free_space_distance_up: float = 0.0
    free_space_geom_name: str = ""

    distance_to_lift: float = DISTANCE_TO_LIFT


class Context:
    def __init__(self) -> None:
        self.bodies: list[BodyInfo] = []
        self.body_idx: int = 0
        self.dirty_reset: bool = False
        self.dirty_pause: bool = False
        self.dirty_start_test: bool = False
        self.dirty_lookat: bool = False
        self.force_magnitude: float = 0.0
        self.test_state: TestState = TestState.IDLE
        self.cam_lookat: np.ndarray = np.array([0, 0, 0], dtype=np.float64)
        self.selected_body_id: int = -1


context: Context = Context()


def compute_estimated_extra_force(mass: float, distance_to_lift: float = DISTANCE_TO_LIFT) -> float:
    return 2 * mass * (ESTIMATED_FORCE_MULTIPLIER * distance_to_lift) / (TEST_TIME**2)


def find_pickupable_bodies(model: mj.MjModel) -> list[BodyInfo]:
    free_bodies_info = []
    for body_id in range(model.nbody):
        body_dofnum = model.body_dofnum[body_id].item()

        if body_dofnum != 6:  # number of dofs for a free body
            continue

        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)

        in_pickupable_list = False
        for pickupable_category in ALL_PICKUP_TYPES_ITHOR:
            if pickupable_category in body_name or pickupable_category.lower() in body_name:
                in_pickupable_list = True
                break

        if not in_pickupable_list:
            continue

        body_subtreemass = model.body_subtreemass[body_id].item()
        body_weight = body_subtreemass * abs(model.opt.gravity[2])
        body_est_force = compute_estimated_extra_force(body_subtreemass)

        free_bodies_info.append(
            BodyInfo(
                body_id,
                body_name,
                body_subtreemass,
                body_weight,
                body_est_force,
            )
        )
    return free_bodies_info


def apply_checks_for_body(body: BodyInfo, model: mj.MjModel, data: mj.MjData) -> None:
    is_within_site, site_id_within = is_body_within_any_site(model, data, body.body_id)
    body.site_within = is_within_site
    body.site_within_id = site_id_within
    body.site_within_name = model.site(site_id_within).name if site_id_within != -1 else ""
    print(f"Within site: {is_within_site}, site_name: {body.site_within_name}")
    if is_within_site and site_id_within != -1:
        is_space_above, distance_up, geom_name = is_body_within_site_in_freespace(
            site_id_within, body.body_id, model, data
        )
        body.free_space = is_space_above
        body.free_space_distance_up = distance_up
        body.free_space_geom_name = geom_name
        if is_space_above:
            message = (
                "There IS free space above. distance_up: {distance_up}, " + "geom_name: {geom_name}"
            )
            print(message.format(distance_up=distance_up, geom_name=geom_name))
        else:
            body.distance_to_lift = DISTANCE_TO_LIFT_WITHIN_SITE
            message = (
                "There is NO free space above. distance_up: {distance_up}, "
                + "geom_name: {geom_name}"
            )
            print(message.format(distance_up=distance_up, geom_name=geom_name))


def compute_force_to_apply(body: BodyInfo, model: mj.MjModel, data: mj.MjData) -> float:
    is_within_inner_site_and_no_freespace = False
    for site_id in range(model.nsite):
        if is_body_com_within_box_site(site_id, body.body_id, model, data):
            is_in_free_space, _, _ = is_body_within_site_in_freespace(
                site_id, body.body_id, model, data
            )
            if not is_in_free_space:
                is_within_inner_site_and_no_freespace = True
                break

    distance_to_lift = (
        DISTANCE_TO_LIFT_WITHIN_SITE if is_within_inner_site_and_no_freespace else DISTANCE_TO_LIFT
    )
    body.body_est_force = compute_estimated_extra_force(body.body_totalmass, distance_to_lift)
    force_magnitude = min(
        max(body.body_weight * (1. + GRAVITY_EXTRA_VALUE), MIN_LIFT_FORCE), MAX_LIFT_FORCE
    )

    return force_magnitude


def key_callback(keycode: int) -> None:
    global model, data, context

    if model is None or data is None:
        return

    if keycode == 265:  # up arrow key
        context.body_idx = (context.body_idx + 1) % len(context.bodies)
        c_body = context.bodies[context.body_idx]
        print(f"Current body: idx={context.body_idx}, name={c_body.body_name}")
        apply_checks_for_body(c_body, model, data)
        # context.force_magnitude = min(c_body.body_weight + c_body.body_est_force, MAX_LIFT_FORCE)
        context.force_magnitude = compute_force_to_apply(c_body, model, data)
    elif keycode == 264:  # down arrow key
        context.body_idx = (context.body_idx - 1) % len(context.bodies)
        c_body = context.bodies[context.body_idx]
        print(f"Current body: idx={context.body_idx}, name={c_body.body_name}")
        apply_checks_for_body(c_body, model, data)
        # context.force_magnitude = min(c_body.body_weight + c_body.body_est_force, MAX_LIFT_FORCE)
        context.force_magnitude = compute_force_to_apply(c_body, model, data)
    elif keycode == 259:  # backspace key
        context.dirty_reset = True
    elif keycode == 32:  # space key
        context.dirty_pause = not context.dirty_pause
        print("Paused" if context.dirty_pause else "Running")
    elif keycode == 82:  # R key
        context.dirty_start_test = True
    elif keycode == 262:  # right arrow key
        c_body = context.bodies[context.body_idx]
        context.force_magnitude += 1.0
        context.force_magnitude = min(context.force_magnitude, MAX_LIFT_FORCE)
        print(f"force-magnitude: {context.force_magnitude}, body-weight: {c_body.body_weight}")
    elif keycode == 263:  # left arrow key
        c_body = context.bodies[context.body_idx]
        context.force_magnitude -= 1.0
        context.force_magnitude = max(context.force_magnitude, MIN_LIFT_FORCE)
        print(f"force-magnitude: {context.force_magnitude}, body-weight: {c_body.body_weight}")
    elif keycode == 81:  # Q key
        c_body = context.bodies[context.body_idx]
    elif keycode == 80:  # P key
        c_body = context.bodies[context.body_idx]
        # print(f"Making body {c_body.body_name} frictionless")
        # make_body_frictionless(c_body.body_id, model)
    elif keycode == 79:  # O key
        if context.selected_body_id != -1:
            c_body = None
            for body_info in context.bodies:
                if body_info.body_id == context.selected_body_id:
                    c_body = body_info
                    break
            if c_body:
                print(f"Body currently selected is : {c_body.body_name}")
    elif keycode == 47:  # / key
        context.dirty_lookat = True
        c_body = context.bodies[context.body_idx]
        if data:
            context.cam_lookat = data.xpos[c_body.body_id].copy()


def main() -> int:
    global model, data, context

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--house",
        type=str,
        default="",
        help="The house to load for the test",
    )
    parser.add_argument(
        "--body",
        type=str,
        default="",
        help="The name of the body to test (optional, if you want to test directly)",
    )

    args = parser.parse_args()

    if args.house == "":
        print("Should give a specific house to test")
        return 1

    xml_file_path = Path(args.house)
    model, _, _ = load_env_with_objects_with_tweaks(
        xml_file_path.as_posix(), move_objects_within_sites_up_abit=True
    )

    data = mj.MjData(model)

    mj.mj_resetData(model, data)

    model.vis.scale.contactwidth = 0.025
    model.vis.scale.contactheight = 0.025

    context.bodies = find_pickupable_bodies(model)

    test_body: BodyInfo = context.bodies[context.body_idx]
    context.force_magnitude = compute_force_to_apply(test_body, model, data)
    force_to_apply = context.force_magnitude
    test_running = False

    if args.body != "":
        body_handle = model.body(args.body)
        if body_handle is not None:
            for idx, body_info in enumerate(context.bodies):
                if body_info.body_name == args.body:
                    context.body_idx = idx
                    test_body = body_info
                    context.force_magnitude = compute_force_to_apply(test_body, model, data)
                    force_to_apply = context.force_magnitude
                    context.dirty_lookat = True
                    context.dirty_pause = True
                    print(f"Checking body '{args.body}' first")
                    break

    with mjviewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            if context.dirty_reset:
                context.dirty_reset = False
                mj.mj_resetData(model, data)

            if context.dirty_start_test:
                mj.mj_resetData(model, data)
                context.dirty_start_test = False
                test_body = context.bodies[context.body_idx]
                test_body.body_pos_bef_settle = data.xpos[test_body.body_id].copy()

                force_to_apply = context.force_magnitude
                test_running = True
                context.test_state = TestState.RUNNING_INITIAL_SETTLE
                print(f"Started test for body: {test_body.body_name}")
                print(f"force applied: {force_to_apply}")

            if context.dirty_lookat:
                context.dirty_lookat = False
                viewer.cam.lookat = context.cam_lookat

            if not context.dirty_pause:
                t_start = data.time
                while data.time - t_start < 1.0 / 60.0:
                    mj.mj_step(model, data)
                    if not test_running:
                        continue
                    if context.test_state == TestState.RUNNING_INITIAL_SETTLE:
                        if data.time > INITIAL_SETTLE_TIME:
                            context.test_state = TestState.RUNNING_MAIN_TEST
                            test_body.body_pos_start = data.xpos[test_body.body_id].copy()
                            data.xfrc_applied[test_body.body_id][:3] = [0, 0, force_to_apply]
                            break
                    elif context.test_state == TestState.RUNNING_MAIN_TEST:
                        data.xfrc_applied[test_body.body_id][:3] = [0, 0, force_to_apply]
                        if data.time > INITIAL_SETTLE_TIME + TEST_TIME:
                            context.test_state = TestState.DONE
                            break

            if test_running and context.test_state == TestState.DONE:
                test_running = False
                context.test_state = TestState.IDLE
                context.dirty_pause = True

                test_body.body_pos_end = data.xpos[test_body.body_id].copy()
                height_start = test_body.body_pos_start[2]
                height_end = test_body.body_pos_end[2]
                diff_height = height_end - height_start
                lift_success = diff_height >= test_body.distance_to_lift
                print(f"lift_success: {lift_success}, diff-height: {diff_height}")

                data.xfrc_applied[test_body.body_id][:3] = [0, 0, 0]

            if viewer.perturb.select != -1:
                context.selected_body_id = model.body_rootid[viewer.perturb.select].item()
            viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
