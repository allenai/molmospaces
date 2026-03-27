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
    collect_warnings,
    get_articulation_tests_results_info,
    get_fail_info_articulation_force_test,
    get_object_categories_failed_articulation_test,
    sort_results_articulation_force_test_by_scene_number,
)
from tqdm import tqdm

from molmo_spaces.env.arena.scene_tweaks import (
    is_body_com_within_box_site,
    is_body_within_site_in_freespace,
)

ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_HOUSES_FOLDER_WEKA = (
    "/weka/prior-default/datasets/mujoco-thor/assets/scenes/{dataset}-{split}"
)
DEFAULT_HOUSES_FOLDER_LOCAL = "assets/scenes/{dataset}-{split}"
DEFAULT_HOUSES_FOLDER_ITHOR_WEKA = "/weka/prior-default/datasets/mujoco-thor/assets/scenes/ithor"
DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL = "assets/scenes/ithor"

results_filepath_TEMPLATE = "history_articulation_force_test_{dataset}_{split}_{identifier}.json"
INFO_BODIES_WITHIN_SITES_PATH_TEMPLATE = (
    "info_bodies_within_sites_{dataset}_{split}_{identifier}.json"
)
COULD_NOT_PROCESS_PATH_TEMPLATE = (
    "articulation_force_test_could_not_process_{dataset}_{split}_{identifier}.txt"
)
MUJOCO_WARNINGS_FILEPATH = (
    "articulation_force_test_mujoco_warnings_{dataset}_{split}_{identifier}.json"
)

IS_BEAKER = "BEAKER_JOB_ID" in os.environ

DEFAULT_TIMESTEP = 0.002
DEFAULT_MONITOR_TIME = 2
DEFAULT_MONITOR_STEPS = int(DEFAULT_MONITOR_TIME / DEFAULT_TIMESTEP)
DEFAULT_TEST_FORCE = 20.0

OPEN_RANGE_PERCENT = 0.667

MJC_VERSION = tuple(map(int, mj.__version__.split(".")))
HAS_SLEEP_ISLAND_SUPPORT = MJC_VERSION >= (3, 3, 8)
HAS_NEW_MJSPEC_SUPPORT = MJC_VERSION >= (3, 3, 8)

TDataset = Literal["ithor", "procthor-10k", "procthor-objaverse", "holodeck-objaverse"]
TSplit = Literal["train", "val", "test"]


@dataclass
class TestSettings:
    dataset: TDataset
    split: TSplit
    identifier: str
    houses_folder: Path

    results_filepath: Path
    info_bodies_within_sites_path: Path
    could_not_process_path: Path
    mujoco_warnings_path: Path

    is_beaker: bool = IS_BEAKER
    only_failed: bool = False
    run_batch: bool = True

    copy_results_to: str = ""
    copy_results_suffix: str = ""

    use_sleep_island: bool = HAS_SLEEP_ISLAND_SUPPORT


SETTINGS: TestSettings | None = None


def json_serializer(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def find_objects_within_inner_sites(
    spec: mj.MjSpec, model: mj.MjModel, data: mj.MjData
) -> dict[int, int]:
    mj.mj_forward(model, data)

    # Collect all root bodies from the model
    root_bodies: list[mj.MjsBody] = []
    current_body_spec = spec.worldbody.first_body()
    while current_body_spec:
        root_bodies.append(current_body_spec)
        current_body_spec = spec.worldbody.next_body(current_body_spec)

    bodies_detected: dict[int, int] = {}
    for body_spec in root_bodies:
        body_id = model.body(body_spec.name).id
        for site_id in range(model.nsite):
            if is_body_com_within_box_site(site_id, body_id, model, data):
                in_free_space, _, _ = is_body_within_site_in_freespace(
                    site_id, body_id, model, data
                )
                if not in_free_space:  # it's actually inside a drawer or similar
                    bodies_detected[site_id] = body_id
                    break
    return bodies_detected


def find_articulated_joints(model: mj.MjModel) -> np.ndarray:
    joints_ids = []
    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id].item()
        if jnt_type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue
        joints_ids.append(jnt_id)
    return np.array(joints_ids, dtype=np.int32)


def get_only_failed_joints_from_previous_run(
    house_name: str, settings: TestSettings, model: mj.MjModel
) -> np.ndarray:
    joints_ids = []

    if settings.results_filepath.exists():
        with open(settings.results_filepath, "r") as fhandle:
            data = json.load(fhandle)
        all_results = data.get("results", {})
        if house_name in all_results:
            joints_cant_open_names = all_results[house_name]["joints_cant_open"]
            for joint_name in joints_cant_open_names:
                joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, joint_name)
                if joint_id != -1:
                    joints_ids.append(joint_id)

    return np.array(joints_ids)


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
        count_failed = all_results[house_xml.stem]["num_joints_cant_open"]
        if count_failed > 0:
            filtered_houses_xmls.append(house_xml)

    return filtered_houses_xmls


def apply_force_and_monitor_serial(
    model: mj.MjModel,
    data: mj.MjData,
    joints_ids: np.ndarray,
    force_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    joints_success = []
    joints_open_percent = []
    warnings = []
    joints_done: set[int] = set()

    for joint_id in joints_ids:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        joint_id_int = joint_id.item()

        jnt_qposadr = model.jnt_qposadr[joint_id].item()
        jnt_dofadr = model.jnt_dofadr[joint_id].item()
        jnt_range_min, jnt_range_max = model.jnt_range[joint_id]
        jnt_range_diff = abs(jnt_range_max - jnt_range_min)

        jnt_qpos_start = data.qpos[jnt_qposadr].item()

        dist_to_low = abs(jnt_qpos_start - jnt_range_min)
        dist_to_high = abs(jnt_qpos_start - jnt_range_max)

        # if closer to the high value, must apply generalized force backwards
        sign = 1.0 if dist_to_low < dist_to_high else -1.0

        open_success = False
        force_to_apply = force_magnitude * sign
        open_percent = 0.0

        for _ in range(DEFAULT_MONITOR_STEPS):
            data.qfrc_applied[jnt_dofadr] = 0.0
            if joint_id_int not in joints_done:
                data.qfrc_applied[jnt_dofadr] = force_to_apply
            mj.mj_step(model, data)
            jnt_qpos_end = data.qpos[jnt_qposadr].item()
            open_percent = abs(jnt_qpos_end - jnt_qpos_start) / jnt_range_diff
            if open_percent >= OPEN_RANGE_PERCENT:
                joints_done.add(joint_id_int)
                data.qvel[jnt_dofadr] = 0.0
                open_success = True
                break

        warnings.extend(collect_warnings(model, data))

        joints_success.append(open_success)
        joints_open_percent.append(open_percent)

    return np.array(joints_success), np.array(joints_open_percent), warnings


def apply_force_and_monitor_batch(
    model: mj.MjModel,
    data: mj.MjData,
    joints_ids: np.ndarray,
    force_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    warnings = []

    joints_qposadr = model.jnt_qposadr[joints_ids]  # (n_batch, )
    joints_dofadr = model.jnt_dofadr[joints_ids]  # (n_batch, )
    joints_range_min = model.jnt_range[joints_ids, 0]  # (n_batch, )
    joints_range_max = model.jnt_range[joints_ids, 1]  # (n_batch, )
    joints_range_diff = np.abs(joints_range_max - joints_range_min)  # (n_batch, )

    joints_qpos_start = data.qpos[joints_qposadr].copy()
    joints_dist_to_low = np.abs(joints_qpos_start - joints_range_min)
    joints_dist_to_high = np.abs(joints_qpos_start - joints_range_max)

    joints_sign = np.zeros(joints_ids.shape, dtype=np.float32)
    joints_sign[joints_dist_to_low < joints_dist_to_high] = 1.0
    joints_sign[joints_dist_to_low >= joints_dist_to_high] = -1.0

    joints_forces = force_magnitude * joints_sign

    joints_done: set[int] = set()
    for _ in range(DEFAULT_MONITOR_STEPS):
        if len(joints_done) == len(joints_ids):
            break
        for idx, (joint_id, force) in enumerate(zip(joints_ids, joints_forces)):
            joint_id_int = joint_id.item()
            data.qfrc_applied[joints_dofadr[idx]] = 0.0
            if joint_id_int not in joints_done:
                data.qfrc_applied[joints_dofadr[idx]] = force
        mj.mj_step(model, data)
        joints_qpos_end = data.qpos[joints_qposadr]
        joints_open_percent = np.abs(joints_qpos_end - joints_qpos_start) / joints_range_diff
        for i, (joint_id, joint_dofadr) in enumerate(zip(joints_ids, joints_dofadr)):
            joint_id_int = joint_id.item()
            if joint_id_int in joints_done:
                continue
            if joints_open_percent[i] >= OPEN_RANGE_PERCENT:
                joints_done.add(joint_id_int)
                data.qvel[joint_dofadr] = 0.0

    warnings.extend(collect_warnings(model, data))

    joints_qpos_end = data.qpos[joints_qposadr]
    joints_open_percent = np.abs(joints_qpos_end - joints_qpos_start) / joints_range_diff

    return np.array([jid in joints_done for jid in joints_ids]), joints_open_percent, warnings


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
    if settings.info_bodies_within_sites_path.is_file():
        dst_filename = settings.info_bodies_within_sites_path.name
        if settings.copy_results_suffix != "":
            filepath = settings.info_bodies_within_sites_path
            dst_filename = f"{filepath.stem}_{settings.copy_results_suffix}{filepath.suffix}"
        shutil.copy(
            settings.info_bodies_within_sites_path.as_posix(),
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


def run_articulation_force_test(house_filepath: Path) -> tuple[bool, dict[str, Any], Counter]:
    global SETTINGS

    results = {"house_name": house_filepath.stem}
    success = True
    categories_counts = Counter()

    if SETTINGS is None:
        results["error"] = f"House '{house_filepath.stem}' - global settings is not initialized yet"
        return False, results, categories_counts

    try:
        # if True:
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

        site_to_body_within_map = find_objects_within_inner_sites(spec, model, data)

        articulable_joints_ids = find_articulated_joints(model)

        results = {
            "house_name": house_filepath.stem,
            "num_total_joints": model.njnt,
            "num_articulable_joints": len(articulable_joints_ids),
            "num_joints_cant_open": 0,
            "joints_cant_open": [],
            "joints_cant_open_percent": [],
            "objects_cant_open": [],
            "joints_do_open": [],
            "joints_do_open_percent": [],
            "max_force_applied": DEFAULT_TEST_FORCE,
            "all_pass": False,
            "bodies_make_stuck": [],
            "warnings": [],
        }

        test_joints_ids = []
        if SETTINGS.only_failed:
            test_joints_ids = get_only_failed_joints_from_previous_run(
                house_filepath.stem, SETTINGS, model
            )
        else:
            test_joints_ids = articulable_joints_ids

        if len(test_joints_ids) < 1:
            results["all_pass"] = True
            return success, results, categories_counts

        if SETTINGS.run_batch:
            joints_open_mask, joints_open_percent, warnings = apply_force_and_monitor_batch(
                model, data, test_joints_ids, DEFAULT_TEST_FORCE
            )
        else:
            joints_open_mask, joints_open_percent, warnings = apply_force_and_monitor_serial(
                model, data, test_joints_ids, DEFAULT_TEST_FORCE
            )
        joints_cant_open_mask = np.logical_not(joints_open_mask)

        joints_ids_opened = test_joints_ids[joints_open_mask]
        joints_ids_cant_open = test_joints_ids[joints_cant_open_mask]

        joints_percent_opened = joints_open_percent[joints_open_mask]
        joints_percent_cant_open = joints_open_percent[joints_cant_open_mask]

        joints_names_opened = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT.value, jnt_id)
            for jnt_id in joints_ids_opened
        ]
        joints_names_cant_open = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT.value, jnt_id)
            for jnt_id in joints_ids_cant_open
        ]

        # Get all objects that are within a site that belongs to the corresponding articulated
        # object and the specific joint (both site and joint should share same parent)
        bodies_within_sites: set[str] = set()
        for jnt_id in joints_ids_cant_open:
            body_id = model.jnt_bodyid[jnt_id].item()
            for candidate_site_id, body_root_id in site_to_body_within_map.items():
                if body_id == model.site_bodyid[candidate_site_id].item():
                    body_root_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_root_id)
                    bodies_within_sites.add(body_root_name)
                    break

        bodies_ids_cant_open = model.jnt_bodyid[joints_ids_cant_open]
        root_bodies_ids_cant_open = model.body_rootid[bodies_ids_cant_open]
        root_bodies_names_cant_open = [model.body(bid).name for bid in root_bodies_ids_cant_open]

        results["joints_do_open"] = joints_names_opened
        results["joints_do_open_percent"] = joints_percent_opened.tolist()
        results["joints_cant_open"] = joints_names_cant_open
        results["joints_cant_open_percent"] = joints_percent_cant_open.tolist()
        results["objects_cant_open"] = root_bodies_names_cant_open

        num_joints_cant_open = len(results["joints_cant_open"])
        results["num_joints_cant_open"] = num_joints_cant_open
        results["all_pass"] = num_joints_cant_open == 0
        results["bodies_make_stuck"] = list(bodies_within_sites)
        results["warnings"] = warnings
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
        sort_results_articulation_force_test_by_scene_number(
            settings.results_filepath,
            settings.dataset,
            settings.split,
        )
        get_fail_info_articulation_force_test(
            settings.results_filepath,
            settings.dataset,
            settings.split,
            settings.identifier,
        )

        count_total_joints, count_cant_open_joints, count_scenes_pass, count_scenes_fail = (
            get_articulation_tests_results_info(settings.results_filepath)
        )

        counts_categories = get_object_categories_failed_articulation_test(
            settings.results_filepath
        )

        print("\n")
        print("#--------------------------------------------")
        print("ARTICULATION TEST")
        print("\n")
        print("Setup:")
        print(f"dataset             : {settings.dataset}")
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

        print(f"total joints                : {count_total_joints}")
        print(f"total joints can't open     : {count_cant_open_joints}")
        print(f"total number scenes pass    : {count_scenes_pass}")
        print(f"total number scenes fail    : {count_scenes_fail}")
        print("\n")
        print("Categories: ")
        for category in counts_categories:
            print(f"\t{category:>10} : {counts_categories[category]}")
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
        prev_scene_results["joints_cant_open"] = scene_results["joints_cant_open"]
        prev_scene_results["joints_cant_open_percent"] = scene_results["joints_cant_open_percent"]
        prev_scene_results["objects_cant_open"] = scene_results["objects_cant_open"]
        prev_scene_results["num_joints_cant_open"] = scene_results["num_joints_cant_open"]
        prev_scene_results["all_pass"] = scene_results["all_pass"]
        prev_scene_results["bodies_make_stuck"] = scene_results["bodies_make_stuck"]
        prev_scene_results["warnings"] = scene_results["warnings"]

        # Combine the new success results, avoid duplicates
        prev_jnt_do_open = prev_scene_results["joints_do_open"]
        prev_jnt_do_open_percent = prev_scene_results["joints_do_open_percent"]
        prev_jnt_do_open_map = {
            name: percent for name, percent in zip(prev_jnt_do_open, prev_jnt_do_open_percent)
        }

        new_jnt_do_open = scene_results["joints_do_open"]
        new_jnt_do_open_percent = scene_results["joints_do_open_percent"]
        new_jnt_do_open_map = {
            name: percent for name, percent in zip(new_jnt_do_open, new_jnt_do_open_percent)
        }

        for jnt_name, jnt_open_percent in prev_jnt_do_open_map.items():
            if jnt_name not in new_jnt_do_open_map:
                new_jnt_do_open_map[jnt_name] = jnt_open_percent

        upt_jnt_do_open = []
        upt_jnt_do_open_percent = []
        for jnt_name, jnt_open_percent in new_jnt_do_open_map.items():
            upt_jnt_do_open.append(jnt_name)
            upt_jnt_do_open_percent.append(jnt_open_percent)

        prev_scene_results["joints_do_open"] = upt_jnt_do_open
        prev_scene_results["joints_do_open_percent"] = upt_jnt_do_open_percent
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
        "--run-batch",
        action="store_true",
        help="Whether or not to run the test with all joints at once (in a batch)",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Whether or not to only load failed joints from a previous run",
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
        "--results-file",
        type=str,
        default="",
        help="A path to the file containing results for showing the summary",
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
                if args.houses_folder != "":
                    return Path(args.houses_folder)
                def_houses = (
                    DEFAULT_HOUSES_FOLDER_WEKA if IS_BEAKER else DEFAULT_HOUSES_FOLDER_LOCAL
                ).format(dataset=args.dataset, split=args.split)
                def_houses_ithor = (
                    DEFAULT_HOUSES_FOLDER_ITHOR_WEKA
                    if IS_BEAKER
                    else DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL
                )
                houses_folder_str = def_houses if args.dataset != "ithor" else def_houses_ithor
                return Path(houses_folder_str)
            case _:
                raise ValueError(f"Given dataset '{args.dataset}' is not a valid dataset")

    SETTINGS = TestSettings(
        dataset=args.dataset,
        split=args.split,
        identifier=args.identifier,
        results_filepath=ROOT_DIR
        / Path(
            results_filepath_TEMPLATE.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        info_bodies_within_sites_path=ROOT_DIR
        / Path(
            INFO_BODIES_WITHIN_SITES_PATH_TEMPLATE.format(
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
        scene_success, scene_results, categories_counts = run_articulation_force_test(xml_file_path)
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
        else:
            print(
                f"Should give --dataset with either 'procthor' or 'ithor', but got {SETTINGS.dataset}"
            )
            return 1

        if SETTINGS.only_failed:
            xmls_to_test = get_only_failed_scenes_from_previous_run(xmls_to_test, SETTINGS)

        if args.max_workers > 1:
            results = p_uimap(run_articulation_force_test, xmls_to_test, num_cpus=args.max_workers)
            for scene_success, scene_results, categories_counts in results:
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))
        else:
            for house_xml in tqdm(xmls_to_test):
                scene_success, scene_results, categories_counts = run_articulation_force_test(
                    house_xml
                )
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))

    show_results_summary(SETTINGS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
