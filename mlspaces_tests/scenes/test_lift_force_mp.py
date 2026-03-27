import argparse
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mujoco as mj
import numpy as np
from p_tqdm import p_uimap
from test_utils import (
    ALL_PICKUP_TYPES_ITHOR,
    collect_badqacc_body,
    collect_warnings,
    get_fail_info_lift_force_test,
    get_lift_tests_results_info,
    get_object_categories_failed_lift_test,
    sort_results_lift_force_test_by_scene_number,
)
from tqdm import tqdm

from molmo_spaces.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA as THOR_TYPE_TO_LEMMA,
)

ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_HOUSES_FOLDER_WEKA = (
    "/weka/prior-default/datasets/mujoco-thor/assets/scenes/{dataset}-{split}"
)
DEFAULT_HOUSES_FOLDER_LOCAL = "assets/scenes/{dataset}-{split}"
DEFAULT_HOUSES_FOLDER_ITHOR_WEKA = "/weka/prior-default/datasets/mujoco-thor/assets/scenes/ithor"
DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL = "assets/scenes/ithor"

HISTORY_RESULTS_INFO_PATH_TEMPLATE = "history_lift_force_test_{dataset}_{split}_{identifier}.json"
COULD_NOT_PROCESS_PATH_TEMPLATE = (
    "lift_force_test_could_not_process_{dataset}_{split}_{identifier}.txt"
)
MUJOCO_WARNINGS_FILEPATH = "lift_force_test_mujoco_warnings_{dataset}_{split}_{identifier}.json"

IS_BEAKER = "BEAKER_JOB_ID" in os.environ

TIMESTEP = 0.002
TEST_TIME = 1.0  # 500 steps = 1.0 seconds
MONITOR_STEPS = int(TEST_TIME / TIMESTEP)
MAX_LIFT_FORCE = 30.0
MIN_LIFT_FORCE = 0.1
DISTANCE_TO_LIFT = 0.05  # 5cm
GRAVITY_EXTRA_VALUE = 0.3
EXTRA_FORCE = 0.5

MJC_VERSION = tuple(map(int, mj.__version__.split(".")))
HAS_SLEEP_ISLAND_SUPPORT = MJC_VERSION >= (3, 3, 8)

TDataset = Literal["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"]
TSplit = Literal["train", "val", "test"]


@dataclass
class TestSettings:
    dataset: TDataset
    split: TSplit
    identifier: str
    houses_folder: Path

    results_filepath: Path
    could_not_process_path: Path
    mujoco_warnings_path: Path

    is_beaker: bool = IS_BEAKER
    only_failed: bool = False
    run_batch: bool = True
    track_max: bool = True
    exclude_heavy: bool = True

    copy_results_to: str = ""
    copy_results_suffix: str = ""

    max_lift_force: float = MAX_LIFT_FORCE
    min_lift_force: float = MIN_LIFT_FORCE
    gravity_extra_value: float = GRAVITY_EXTRA_VALUE
    extra_force: float = EXTRA_FORCE

    use_sleep_island: bool = HAS_SLEEP_ISLAND_SUPPORT


SETTINGS: TestSettings | None = None


def json_serializer(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def find_pickupable_bodies(model: mj.MjModel) -> np.ndarray:
    pickupable_bodies_ids = []
    for body_id in range(model.nbody):
        dofnum = model.body_dofnum[body_id].item()
        if dofnum != 6:  # objects without a free joint shouldn't be considered
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_id)
        in_pickupable_list = False
        for pickupable_category in ALL_PICKUP_TYPES_ITHOR:
            desired_category = pickupable_category
            if pickupable_category in THOR_TYPE_TO_LEMMA:
                desired_category = THOR_TYPE_TO_LEMMA[pickupable_category].replace("_", "")
            if desired_category in name or desired_category.lower() in name:
                in_pickupable_list = True
                break
        if in_pickupable_list:
            pickupable_bodies_ids.append(body_id)
    return np.array(pickupable_bodies_ids)


def get_only_failed_bodies_from_previous_run(
    house_name: str, settings: TestSettings, model: mj.MjModel
) -> np.ndarray:
    failed_bodies_ids = []
    with open(settings.results_filepath, "r") as fhandle:
        data = json.load(fhandle)

    all_results = data.get("results", {})
    if house_name in all_results:
        bodies_cant_lift_names = all_results[house_name]["counts"]["failed"]["cant_lift"]["names"]
        for body_name in bodies_cant_lift_names:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, body_name)
            if body_id != -1:
                failed_bodies_ids.append(body_id)
    else:
        print(f"There was an issue trying to get failed bodies for {house_name}")

    return np.array(failed_bodies_ids)


def get_only_failed_scenes_from_previous_run(
    houses_xmls: list[Path], settings: TestSettings
) -> list[Path]:
    if not settings.results_filepath.exists():
        return houses_xmls

    with open(settings.results_filepath, "r") as fhandle:
        data = json.load(fhandle)
    all_results = data.get("results", {})

    if len(all_results) < 1:
        return houses_xmls

    filtered_houses_xmls = []
    for house_xml in houses_xmls:
        if house_xml.stem not in all_results:
            continue
        count_failed = all_results[house_xml.stem]["counts"]["failed"]["num"]
        if count_failed == 0:
            continue
        filtered_houses_xmls.append(house_xml)

    return filtered_houses_xmls


def apply_lift_force_and_monitor_batch(
    model: mj.MjModel,
    data: mj.MjData,
    bodies_ids: np.ndarray,
    settings: TestSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    warnings = []
    badqacc_bodies = []

    bodies_dofadr = model.body_dofadr[bodies_ids]

    pos_start_all = data.xpos.copy()  # NOTE(wilbert): hacky way to handle badqacc bodies
    bodies_pos_start = data.xpos[bodies_ids, :3].copy()
    bodies_height_start = data.xpos[bodies_ids, 2].copy()

    gravity_abs = abs(model.opt.gravity[2])
    masses = model.body_subtreemass[bodies_ids]

    forces = masses * (gravity_abs + settings.gravity_extra_value) + settings.extra_force
    forces = np.minimum(np.maximum(forces, settings.min_lift_force), settings.max_lift_force)

    bodies_done: set[int] = set()
    bodies_height_max = data.xpos[bodies_ids, 2].copy()
    for _ in range(MONITOR_STEPS):
        # Early stop if all objects have passed the test already
        if len(bodies_done) == len(bodies_ids):
            break
        for body_id, force in zip(bodies_ids, forces):
            data.xfrc_applied[body_id, 2] = 0.0
            if body_id not in bodies_done:
                data.xfrc_applied[body_id, 2] = force
        mj.mj_step(model, data)
        new_bodies_height = data.xpos[bodies_ids, 2]
        bodies_height_max = np.maximum(bodies_height_max, new_bodies_height)
        for i, (body_id, body_dofadr) in enumerate(zip(bodies_ids, bodies_dofadr)):
            if body_id in bodies_done:
                continue
            dist_moved = np.linalg.norm(data.xpos[body_id] - bodies_pos_start[i])
            if dist_moved >= DISTANCE_TO_LIFT:
                bodies_done.add(body_id)
                data.qvel[body_dofadr : body_dofadr + 6] = [0, 0, 0, 0, 0, 0]
    bodies_height_end = bodies_height_max

    warnings.extend(collect_warnings(model, data))
    badbody = collect_badqacc_body(model, data)
    if badbody != "":
        badqacc_bodies.append(badbody)

    badqacc_bodies_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, name) for name in badqacc_bodies
    ]

    if len(badqacc_bodies) > 0:
        # NOTE(wilbert): for badqacc bodies we must have to overwrite the results to force a failure
        data.xpos[badqacc_bodies_ids, :3] = pos_start_all[badqacc_bodies_ids, :3]

    bodies_pos_end = data.xpos[bodies_ids, :3].copy()

    bodies_diff_height = bodies_height_end - bodies_height_start

    bodies_distance_moved = np.linalg.norm(bodies_pos_end - bodies_pos_start, axis=1)

    return (
        np.array([bid in bodies_done for bid in bodies_ids]),
        bodies_diff_height,
        bodies_distance_moved,
        warnings,
        badqacc_bodies,
    )


def apply_lift_force_and_monitor_serial(
    model: mj.MjModel,
    data: mj.MjData,
    bodies_ids: np.ndarray,
    settings: TestSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    bodies_success = []
    bodies_diff_height = []
    bodies_distance_moved = []
    gravity_abs = abs(model.opt.gravity[2])
    warnings = []
    badqacc_bodies = []

    gravity_abs = abs(model.opt.gravity[2])
    masses = model.body_subtreemass[bodies_ids]

    forces = masses * (gravity_abs + settings.gravity_extra_value) + settings.extra_force
    forces = np.minimum(np.maximum(forces, settings.min_lift_force), settings.max_lift_force)

    bodies_done: set[int] = set()
    for body_id, force in zip(bodies_ids, forces):
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

        body_pos_start = data.xpos[body_id, :3].copy()
        body_height_start = body_pos_start[2]

        dofadr = model.body_dofadr[body_id].item()

        body_height_max = data.xpos[body_id, 2]
        for _ in range(MONITOR_STEPS):
            data.xfrc_applied[body_id, 2] = 0.0
            if body_id not in bodies_done:
                data.xfrc_applied[body_id, 2] = force
            mj.mj_step(model, data)
            new_body_height = data.xpos[body_id, 2]
            body_height_max = max(body_height_max, new_body_height)
            dist_moved = np.linalg.norm(data.xpos[body_id] - body_pos_start)
            if dist_moved >= DISTANCE_TO_LIFT:
                bodies_done.add(body_id)
                data.qvel[dofadr : dofadr + 6] = [0, 0, 0, 0, 0, 0]
                break  # Early stop if the test passed already for this body
        body_pos_end = data.xpos[body_id, :3]
        body_height_end = body_height_max

        warnings.extend(collect_warnings(model, data))
        badbody = collect_badqacc_body(model, data)
        if badbody != "":
            badqacc_bodies.append(badbody)

        body_diff_height = body_height_end - body_height_start
        body_distance_moved = np.linalg.norm(body_pos_end - body_pos_start)

        badbody_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, badbody)
        if badbody_id == body_id:
            bodies_success.append(False)
        else:
            bodies_success.append(body_id in bodies_done)
        bodies_diff_height.append(body_diff_height)
        bodies_distance_moved.append(body_distance_moved)

    return (
        np.array(bodies_success),
        np.array(bodies_diff_height),
        np.array(bodies_distance_moved),
        warnings,
        badqacc_bodies,
    )


def copy_results_files(settings: TestSettings) -> None:
    if settings.copy_results_to == "":
        return

    if not os.path.exists(settings.copy_results_to):
        os.makedirs(settings.copy_results_to, exist_ok=True)

    if settings.results_filepath.is_file():
        dst_filename = settings.results_filepath.name
        if settings.copy_results_suffix != "":
            filepath = settings.results_filepath
            dst_filename = f"{filepath.stem}_{settings.copy_results_suffix}{filepath.suffix}"
        shutil.copy(
            settings.results_filepath.as_posix(),
            os.path.join(settings.copy_results_to, dst_filename),
        )
    if settings.mujoco_warnings_path.is_file():
        dst_filename = settings.mujoco_warnings_path.name
        if settings.copy_results_suffix != "":
            filepath = settings.mujoco_warnings_path
            dst_filename = f"{filepath.stem}_{settings.copy_results_suffix}{filepath.suffix}"
        shutil.copy(
            settings.mujoco_warnings_path.as_posix(),
            os.path.join(settings.copy_results_to, dst_filename),
        )
    if settings.could_not_process_path.is_file():
        dst_filename = settings.could_not_process_path.name
        if settings.copy_results_suffix != "":
            filepath = settings.could_not_process_path
            dst_filename = f"{filepath.stem}_{settings.copy_results_suffix}{filepath.suffix}"
        shutil.copy(
            settings.could_not_process_path.as_posix(),
            os.path.join(settings.copy_results_to, dst_filename),
        )


def run_lift_force_test(house_filepath: Path) -> tuple[bool, dict[str, Any], Counter]:
    global SETTINGS

    results = {"house_name": house_filepath.stem}
    success = True
    categories_counts = Counter()

    if SETTINGS is None:
        results["error"] = f"House '{house_filepath.stem}' - global settings is not initialized yet"
        return False, results, categories_counts

    try:
        # if True:
        categories_counts = Counter()
        # ------------------------------------------------------------------------------------------
        spec: mj.MjSpec = mj.MjSpec.from_file(house_filepath.as_posix())
        if HAS_SLEEP_ISLAND_SUPPORT and SETTINGS.use_sleep_island:
            spec.option.enableflags |= mj.mjtEnableBit.mjENBL_SLEEP

        for body in spec.worldbody.bodies:
            assert isinstance(body, mj.MjsBody)
            lemma = body.name.split("_")[0]
            categories_counts[lemma] += 1
        # ------------------------------------------------------------------------------------------
        model: mj.MjModel = spec.compile()
        data: mj.MjData = mj.MjData(model)
        mj.mj_forward(model, data)

        pickupable_bodies_ids = find_pickupable_bodies(model)

        results = {
            "house_name": house_filepath.stem,
            "num_total_bodies": model.nbody,
            "num_pickupable_bodies": len(pickupable_bodies_ids),
            "counts": {
                "success": {
                    "num": 0,
                    "names": [],
                    "masses": [],
                    "diff-height": [],
                    "distance-moved": [],
                },
                "failed": {
                    "num": 0,
                    "cant_lift": {
                        "num": 0,
                        "names": [],
                        "masses": [],
                        "diff-height": [],
                        "distance-moved": [],
                    },
                },
                "badqacc": [],
            },
            "all_pass": False,
            "warnings": [],
        }

        test_bodies_ids = []
        if SETTINGS.only_failed:
            test_bodies_ids = get_only_failed_bodies_from_previous_run(
                house_filepath.stem, SETTINGS, model
            )
        else:
            test_bodies_ids = pickupable_bodies_ids

        if len(test_bodies_ids) < 1:  # No bodies to process here
            results["all_pass"] = True
            return True, results, categories_counts

        bodies_lifted_mask, bodies_diff_height, bodies_distance_moved = [], [], []
        if SETTINGS.run_batch:
            (
                bodies_lifted_mask,
                bodies_diff_height,
                bodies_distance_moved,
                warnings,
                badqacc_bodies,
            ) = apply_lift_force_and_monitor_batch(model, data, test_bodies_ids, SETTINGS)
        else:
            (
                bodies_lifted_mask,
                bodies_diff_height,
                bodies_distance_moved,
                warnings,
                badqacc_bodies,
            ) = apply_lift_force_and_monitor_serial(model, data, test_bodies_ids, SETTINGS)

        bodies_cant_lift_mask = np.logical_not(bodies_lifted_mask)

        bodies_ids_lifted = test_bodies_ids[bodies_lifted_mask]
        bodies_ids_cant_lift = test_bodies_ids[bodies_cant_lift_mask]

        bodies_names_lifted = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_id)
            for body_id in bodies_ids_lifted
        ]
        bodies_names_cant_lift = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_id)
            for body_id in bodies_ids_cant_lift
        ]

        # Save the data for the ones that were lifted
        results["counts"]["success"]["num"] = np.count_nonzero(bodies_lifted_mask)
        results["counts"]["success"]["names"] = bodies_names_lifted
        results["counts"]["success"]["masses"] = model.body_subtreemass[bodies_ids_lifted].tolist()
        results["counts"]["success"]["diff-height"] = bodies_diff_height[
            bodies_lifted_mask
        ].tolist()
        results["counts"]["success"]["distance-moved"] = bodies_distance_moved[
            bodies_lifted_mask
        ].tolist()

        # Save the data for the ones that couldn't be lifted
        num_cant_lift = np.count_nonzero(bodies_cant_lift_mask)
        results["counts"]["failed"]["num"] = num_cant_lift
        results["counts"]["failed"]["cant_lift"]["num"] = num_cant_lift
        results["counts"]["failed"]["cant_lift"]["names"] = bodies_names_cant_lift
        results["counts"]["failed"]["cant_lift"]["masses"] = model.body_subtreemass[
            bodies_ids_cant_lift
        ].tolist()
        results["counts"]["failed"]["cant_lift"]["diff-height"] = bodies_diff_height[
            bodies_cant_lift_mask
        ].tolist()
        results["counts"]["failed"]["cant_lift"]["distance-moved"] = bodies_distance_moved[
            bodies_cant_lift_mask
        ].tolist()

        results["all_pass"] = num_cant_lift == 0

        results["warnings"].extend(warnings)
        results["counts"]["badqacc"].extend(badqacc_bodies)
    except Exception as e:
        results = {
            "house_name": house_filepath.stem,
            "error": f"Got an error while processing house '{house_filepath.stem}' - error: {e}",
        }
        success = False

    return success, results, categories_counts


def add_test_settings(settings: TestSettings) -> None:
    if settings.results_filepath.exists():
        data = {}
        with open(settings.results_filepath, "r") as fhandle:
            data = json.load(fhandle)
            data["settings"] = asdict(settings)

        with open(settings.results_filepath, "w") as fhandle:
            json.dump(data, fhandle, indent=4, default=json_serializer)


def show_results_summary(settings: TestSettings) -> None:
    if settings.results_filepath.is_file():
        sort_results_lift_force_test_by_scene_number(
            settings.results_filepath,
            settings.dataset,
            settings.split,
        )
        get_fail_info_lift_force_test(
            settings.results_filepath,
            settings.dataset,
            settings.split,
            settings.identifier,
        )

        (
            count_total_pickupable,
            cant_lift,
            num_scenes_pass,
            num_scenes_fail,
            count_total_badqacc,
        ) = get_lift_tests_results_info(settings.results_filepath)

        counts_categories, counts_overall = get_object_categories_failed_lift_test(
            settings.results_filepath
        )

        print("\n")
        print("#--------------------------------------------")
        print("LIFT TEST")
        print("\n")
        print("Setup:")
        print(f"dataset              : {settings.dataset}")
        print(f"split               : {settings.split}")
        print(f"identifier          : {settings.identifier}")
        print(f"mujoco version      : {mj.__version__}")
        print(f"use-sleep-island    : {settings.use_sleep_island}")
        print("\n")

        build_settings_filepath = settings.houses_folder / "housegen_build_settings.json"
        if build_settings_filepath.is_file():
            with open(build_settings_filepath, "r") as fhandle:
                data = json.load(fhandle)
            print("Parameters:")
            if "param_geom_margin" in data:
                print(f"geom-margin                 : {data['param_geom_margin']}")
            if "param_freejoint_damping" in data:
                print(f"freejoint-damping           : {data['param_freejoint_damping']}")
            if "param_freejoint_frictionloss" in data:
                print(f"freejoint-frictionloss      : {data['param_freejoint_frictionloss']}")
            if "use_sleep_island" in data:
                print(f"use-sleep-island            : {data['use_sleep_island']}")
            if "settle_time" in data:
                print(f"settle-time                 : {data['settle_time']}")
            if "mujoco-version" in data:
                print(f"mujoco-version-generated    : {data['mujoco-version']}")
            print(f"mujoco-version-run          : {mj.__version__}")
            print("\n")

        print(f"total bodies pickupable     : {count_total_pickupable}")
        print(f"total bodies can't lift     : {len(cant_lift)}")
        print(f"total bodies diverge        : {count_total_badqacc}")
        print(f"total number scenes pass    : {num_scenes_pass}")
        print(f"total number scenes fail    : {num_scenes_fail}")
        print("")
        print("Categories:")
        for category in counts_categories:
            print(f"\t{category:>10} : {counts_categories[category]} / {counts_overall[category]}")
        print("#--------------------------------------------")
        print("\n")


def save_scene_results(
    settings: TestSettings,
    results_all: dict[str, Any],
    scene_results: dict[str, Any],
    categories_counts: Counter,
) -> None:
    if "house_name" not in scene_results:
        return

    house_name = scene_results["house_name"]
    if not settings.only_failed:
        for cat, num in categories_counts.items():
            if cat not in results_all["categories"]:
                results_all["categories"][cat] = num
            else:
                results_all["categories"][cat] += num
        results_all["results"][house_name] = scene_results
    elif house_name in results_all["results"]:
        prev_scene_results = results_all["results"][house_name]

        # Replace fails info with the new scene results
        prev_scene_results["counts"]["failed"]["num"] = scene_results["counts"]["failed"]["num"]
        prev_scene_results["counts"]["failed"]["cant_lift"]["num"] = scene_results["counts"][
            "failed"
        ]["cant_lift"]["num"]
        prev_scene_results["counts"]["failed"]["cant_lift"]["names"] = scene_results["counts"][
            "failed"
        ]["cant_lift"]["names"]
        prev_scene_results["counts"]["failed"]["cant_lift"]["masses"] = scene_results["counts"][
            "failed"
        ]["cant_lift"]["masses"]
        prev_scene_results["counts"]["failed"]["cant_lift"]["diff-height"] = scene_results[
            "counts"
        ]["failed"]["cant_lift"]["diff-height"]
        prev_scene_results["counts"]["failed"]["cant_lift"]["distance-moved"] = scene_results[
            "counts"
        ]["failed"]["cant_lift"]["distance-moved"]
        prev_scene_results["all_pass"] = scene_results["all_pass"]
        prev_scene_results["warnings"] = scene_results["warnings"]
        prev_scene_results["counts"]["badqacc"] = scene_results["counts"]["badqacc"]

        # Combine the new success results, avoid duplicates
        prev_body_names = prev_scene_results["counts"]["success"]["names"]
        prev_body_masses = prev_scene_results["counts"]["success"]["masses"]
        prev_body_diff = prev_scene_results["counts"]["success"]["diff-height"]
        prev_body_dist = prev_scene_results["counts"]["success"]["distance-moved"]
        prev_body_map = {
            name: (mass, diff, dist)
            for name, mass, diff, dist in zip(
                prev_body_names, prev_body_masses, prev_body_diff, prev_body_dist
            )
        }

        new_body_names = scene_results["counts"]["success"]["names"]
        new_body_masses = scene_results["counts"]["success"]["masses"]
        new_body_diff = scene_results["counts"]["success"]["diff-height"]
        new_body_dist = scene_results["counts"]["success"]["distance-moved"]
        new_body_map = {
            name: (mass, diff, dist)
            for name, mass, diff, dist in zip(
                new_body_names, new_body_masses, new_body_diff, new_body_dist
            )
        }

        for body_name, (mass, diff, dist) in prev_body_map.items():
            if body_name not in new_body_map:
                new_body_map[body_name] = (mass, diff, dist)

        upt_body_names = []
        upt_body_masses = []
        upt_body_diff = []
        upt_body_dist = []
        for body_name, (mass, diff, dist) in new_body_map.items():
            upt_body_names.append(body_name)
            upt_body_masses.append(mass)
            upt_body_diff.append(diff)
            upt_body_dist.append(dist)

        prev_scene_results["counts"]["success"]["names"] = upt_body_names
        prev_scene_results["counts"]["success"]["masses"] = upt_body_masses
        prev_scene_results["counts"]["success"]["diff-height"] = upt_body_diff
        prev_scene_results["counts"]["success"]["distance-moved"] = upt_body_dist
        prev_scene_results["counts"]["success"]["num"] = len(upt_body_names)

        results_all["results"][house_name] = prev_scene_results

    with open(settings.results_filepath, "w") as fhandle:
        json.dump(results_all, fhandle, indent=4, default=json_serializer)


def save_scene_errors(settings: TestSettings, error_message: str) -> None:
    with open(settings.could_not_process_path, "a") as fhandle:
        fhandle.write(error_message + "\n")


def main() -> int:
    global SETTINGS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ithor",
        choices=["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"],
        help="The type of dataset to use for the test, either 'procthor' or 'ithor'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--house",
        type=str,
        default="",
        help="The house to load for the test",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="The number of parallel processes to use for the test",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Whether or not to only load failed bodies from a previous run",
    )
    parser.add_argument(
        "--run-batch",
        action="store_true",
        help="Whether or not to run the test with all bodies at once (in a batch)",
    )
    parser.add_argument(
        "--track-max",
        action="store_true",
        help="Whether or not to keep track of the max height difference for the test",
    )
    parser.add_argument(
        "--houses-folder",
        type=str,
        default="",
        help="The path to the folder that contains the houses to use for the test",
    )
    parser.add_argument(
        "--copy-results-to",
        type=str,
        default="",
        help="The path where to copy the results json file, if required",
    )
    parser.add_argument(
        "--copy-results-suffix",
        type=str,
        default="",
        help="An extra suffix to add to the results files, so we can run on a whole dataset in parts",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="The start index for procthor houses to test",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10000,
        help="The end index for procthor houses to test",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default=f"{datetime.now().strftime('%m%d%y')}" + ("_weka" if IS_BEAKER else ""),
        help="A suffix to use when saving the results json files",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="If enabled, only shows the results from the latest json file",
    )
    parser.add_argument(
        "--exclude-heavy",
        action="store_true",
        help="Whether or not to exclude objects whose weight is larger than the max. lift force",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="",
        help="A path to the file containing results for showing the summary",
    )
    parser.add_argument(
        "--distance-to-move",
        type=float,
        default=DISTANCE_TO_LIFT,
        help="The target distance the object has to move to be considered success",
    )
    parser.add_argument(
        "--use-sleep-island",
        action="store_true",
        help="Whether or not to enable sleeping island (requires 3.3.8)",
    )

    args = parser.parse_args()

    def get_houses_folder(args: Any) -> Path:
        match args.dataset:
            case "ithor" | "procthor-10k" | "procthor-objaverse" | "holodeck-objaverse":
                if args.houses_folder:
                    return Path(args.houses_folder)
                def_houses = (
                    DEFAULT_HOUSES_FOLDER_WEKA if IS_BEAKER else DEFAULT_HOUSES_FOLDER_LOCAL
                )
                def_houses_ithor = (
                    DEFAULT_HOUSES_FOLDER_ITHOR_WEKA
                    if IS_BEAKER
                    else DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL
                )
                houses_folder_str = def_houses if args.dataset != "ithor" else def_houses_ithor
                houses_folder_str = (
                    houses_folder_str.format(dataset=args.dataset, split=args.split)
                    if args.dataset != "ithor"
                    else houses_folder_str
                )
                return Path(houses_folder_str)
            case _:
                raise ValueError(f"Given dataset '{args.dataset}' is not a valid dataset")

    SETTINGS = TestSettings(
        dataset=args.dataset,
        split=args.split,
        identifier=args.identifier,
        results_filepath=ROOT_DIR
        / Path(
            HISTORY_RESULTS_INFO_PATH_TEMPLATE.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        could_not_process_path=ROOT_DIR
        / Path(
            COULD_NOT_PROCESS_PATH_TEMPLATE.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        mujoco_warnings_path=ROOT_DIR
        / Path(
            MUJOCO_WARNINGS_FILEPATH.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        houses_folder=get_houses_folder(args),
        only_failed=args.only_failed,
        run_batch=args.run_batch,
        track_max=args.track_max,
        exclude_heavy=args.exclude_heavy,
        copy_results_to=args.copy_results_to,
        copy_results_suffix=args.copy_results_suffix,
        use_sleep_island=args.use_sleep_island,
    )

    if args.summary:
        if args.results_file != "":
            SETTINGS.results_filepath = Path(args.results_file)
        show_results_summary(SETTINGS)
        return 0

    add_test_settings(SETTINGS)

    results_all = dict(results=dict(), settings=asdict(SETTINGS), categories=dict())
    if SETTINGS.results_filepath.is_file():
        with open(SETTINGS.results_filepath, "r") as fhandle:
            results_all = json.load(fhandle)

    if args.house != "":
        xml_file_path = Path(args.house)
        scene_success, scene_results, categories_counts = run_lift_force_test(xml_file_path)
        if scene_success:
            save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
        else:
            save_scene_errors(SETTINGS, scene_results.get("error", ""))
    else:
        xmls_to_test = []
        if SETTINGS.dataset in {"procthor-10k", "procthor-objaverse", "holodeck-objaverse"}:
            pattern = SETTINGS.split + r"_(\d+)"
            candidates_xmls_to_test = SETTINGS.houses_folder.glob(f"{SETTINGS.split}_*.xml")
            for candidate_xml_test in candidates_xmls_to_test:
                if "ceiling" in candidate_xml_test.stem:
                    continue
                if "_new.xml" in candidate_xml_test.name:
                    continue
                if "orig" in candidate_xml_test.stem:
                    continue
                if "non_settled" in candidate_xml_test.stem:
                    continue
                index = int(re.search(pattern, candidate_xml_test.stem).group(1))  # type: ignore
                if index < args.start or index >= args.end:
                    continue
                xmls_to_test.append(candidate_xml_test)
        elif SETTINGS.dataset == "ithor":
            xmls_to_test = list(SETTINGS.houses_folder.glob("*_physics.xml"))

        if SETTINGS.only_failed:
            xmls_to_test = get_only_failed_scenes_from_previous_run(xmls_to_test, SETTINGS)

        if args.max_workers > 1:
            results = p_uimap(run_lift_force_test, xmls_to_test, num_cpus=args.max_workers)
            for scene_success, scene_results, categories_counts in results:
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))
        else:
            for house_xml in tqdm(xmls_to_test):
                scene_success, scene_results, categories_counts = run_lift_force_test(house_xml)
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))

    show_results_summary(SETTINGS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
