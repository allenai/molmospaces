import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import mujoco as mj
import numpy as np
from p_tqdm import p_uimap
from test_utils import (
    get_categories_failed_stability_test,
    get_fail_info_stability_test,
    get_stability_tests_results_info,
    sort_results_stability_test_by_scene_number,
)
from tqdm import tqdm

from molmo_spaces.env.arena.scene_tweaks import (
    is_body_com_within_box_site,
    is_body_within_site_in_freespace,
)

ROOT_DIR = Path(__file__).parent.parent.parent

RESULTS_FILEPATH = "stability_test_results_{dataset}_{split}_{identifier}.json"
WARNINGS_FILEPATH = "stability_test_warnings_{dataset}_{split}_{identifier}.json"
ERRORS_FILEPATH = "stability_test_errors_{dataset}_{split}_{identifier}.json"

DEFAULT_HOUSES_FOLDER_WEKA = (
    "/weka/prior-default/datasets/mujoco-thor/assets/scenes/{dataset}-{split}-refactor"
)
DEFAULT_HOUSES_FOLDER_LOCAL = "assets/scenes/{dataset}-{split}"
DEFAULT_HOUSES_FOLDER_ITHOR_WEKA = "/weka/prior-default/datasets/mujoco-thor/assets/scenes/ithor"
DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL = "assets/scenes/ithor"

DEFAULT_TIMESTEP = 0.002
DEFAULT_SETTLE_TIME = 1.0  # Settle for 1 seconds
DEFAULT_SETTLE_STEPS = int(DEFAULT_SETTLE_TIME / DEFAULT_TIMESTEP)  # 1000 steps @ dt = 0.002
DEFAULT_MONITOR_TIME = 3.0  # Monitor for 3 seconds
DEFAULT_MONITOR_STEPS = int(DEFAULT_MONITOR_TIME / DEFAULT_TIMESTEP)  # 1500 steps @ dt = 0.002

DEFAULT_DIST_THRESHOLD = 0.05  # detect motion as displacement of at least 5 cm
DEFAULT_JOINT_THRESHOLD = 0.1
DEFAULT_JOINT_PERCENT_THRESHOLD = 0.1
DEFAULT_JITTER_THRESHOLD = 0.1  # detect jitter if cummulative displacement is at least 10cm

IS_BEAKER = "BEAKER_JOB_ID" in os.environ

MJC_VERSION = tuple(map(int, mj.__version__.split(".")))
HAS_SLEEP_ISLAND_SUPPORT = MJC_VERSION >= (3, 3, 8)


def get_houses_folder(args: Any) -> Path:
    match args.dataset:
        case "ithor" | "procthor-10k" | "procthor-objaverse" | "holodeck-objaverse":
            if args.houses_folder != "":
                return Path(args.houses_folder)
            def_houses = (
                DEFAULT_HOUSES_FOLDER_WEKA if IS_BEAKER else DEFAULT_HOUSES_FOLDER_LOCAL
            ).format(dataset=args.dataset, split=args.split)
            def_houses_ithor = (
                DEFAULT_HOUSES_FOLDER_ITHOR_WEKA if IS_BEAKER else DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL
            )
            houses_folder_str = def_houses if args.dataset != "ithor" else def_houses_ithor
            return Path(houses_folder_str)
        case _:
            raise ValueError(f"Given dataset '{args.dataset}' is not a valid dataset")


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


@dataclass
class TestSettings:
    dataset: str
    split: str
    identifier: str
    houses_folder: Path

    results_filepath: Path
    warnings_filepath: Path
    errors_filepath: Path

    timestep: float = DEFAULT_TIMESTEP
    settle_time: float = DEFAULT_SETTLE_TIME
    settle_steps: int = DEFAULT_SETTLE_STEPS
    monitor_time: float = DEFAULT_MONITOR_TIME
    monitor_steps: int = DEFAULT_MONITOR_STEPS

    copy_results_to: str = ""
    copy_results_suffix: str = ""

    use_sleep_island: bool = HAS_SLEEP_ISLAND_SUPPORT
    body_drift_threshold: float = DEFAULT_DIST_THRESHOLD
    joint_drift_threshold_percent: float = DEFAULT_JOINT_PERCENT_THRESHOLD
    body_jitter_threshold: float = DEFAULT_JITTER_THRESHOLD


SETTINGS: TestSettings | None = None


def json_serializer(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class TestBodyResults:
    names: list[str] = field(default_factory=list)
    diff_dist: list[float] = field(default_factory=list)
    diff_rot: list[float] = field(default_factory=list)
    count_verts: list[int] = field(default_factory=list)
    count_faces: list[int] = field(default_factory=list)


@dataclass
class TestJointResults:
    names: list[str] = field(default_factory=list)
    diff_qpos: list[float] = field(default_factory=list)


class TestResults(TypedDict):
    bodies: TestBodyResults
    joints: TestJointResults


def is_free_body(model: mj.MjModel, body_id: int) -> bool:
    return body_id != 0 and model.body_dofnum[body_id].item() == 6


def get_free_bodies_ids(model: mj.MjModel) -> np.ndarray:
    return np.array([bid for bid in range(model.nbody) if is_free_body(model, bid)])


def is_articulation_joint(model: mj.MjModel, joint_id: int) -> bool:
    return model.jnt_type[joint_id].item() in {mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE}


def get_articulation_joints_ids(model: mj.MjModel) -> np.ndarray:
    return np.array([jid for jid in range(model.njnt) if is_articulation_joint(model, jid)])


def run_stability_test(house_filepath: Path) -> tuple[bool, dict[str, Any], Counter]:
    global SETTINGS

    results = {"house_name": house_filepath.stem}
    success = True
    categories_counts = Counter()

    if SETTINGS is None:
        results["error"] = f"House '{house_filepath.stem}' - global settings is not initialized yet"
        return False, results, categories_counts

    try:
        # if True:
        # Do some processing to the scene if needed ------------------------------------------------
        spec = mj.MjSpec.from_file(house_filepath.as_posix())
        spec.option.timestep = SETTINGS.timestep
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

        bodies_to_monitor_ids = get_free_bodies_ids(model)
        bodies_to_monitor_cumdist = np.zeros(bodies_to_monitor_ids.shape, dtype=np.float32)
        has_bodies = len(bodies_to_monitor_ids) > 0

        joints_to_monitor_ids = get_articulation_joints_ids(model)
        joints_to_monitor_qposadr = np.array([])
        joints_init_qpos = np.array([])
        joints_to_monitor_range_diff = np.array([])

        has_joints = len(joints_to_monitor_ids) > 0
        if has_joints:
            joints_to_monitor_qposadr = np.array(
                [model.jnt_qposadr[jid].item() for jid in joints_to_monitor_ids]
            )
            joints_to_monitor_range_min = model.jnt_range[joints_to_monitor_ids, 0]  # (n_batch, )
            joints_to_monitor_range_max = model.jnt_range[joints_to_monitor_ids, 1]  # (n_batch, )
            joints_to_monitor_range_diff = np.abs(
                joints_to_monitor_range_max - joints_to_monitor_range_min
            )
            joints_init_qpos = data.qpos[joints_to_monitor_qposadr].copy()

        mj.mj_step(model, data, nstep=SETTINGS.settle_steps)

        bodies_init_pos = np.array([])
        bodies_bef_pos = np.array([])
        if has_bodies:
            bodies_init_pos = data.xpos[bodies_to_monitor_ids, :3].copy()
            bodies_bef_pos = bodies_init_pos.copy()
            bodies_aft_pos = bodies_init_pos.copy()

        for _ in range(SETTINGS.monitor_steps):
            mj.mj_step(model, data)
            if has_bodies:
                bodies_aft_pos = data.xpos[bodies_to_monitor_ids, :3].copy()
                diff_dist = np.linalg.norm(bodies_aft_pos - bodies_bef_pos, axis=1)
                bodies_bef_pos = bodies_aft_pos.copy()
                bodies_to_monitor_cumdist += diff_dist

        bodies_detected_names = []
        bodies_detected_diff_pos = np.array([])
        bodies_jitter_names = []
        bodies_jitter_diff = np.array([])
        joints_detected_open_percent = np.array([])

        joints_detected_names = []
        joints_detected_diff_qpos = np.array([])
        bodies_within_sites: set[str] = set()

        if has_bodies:
            bodies_end_pos = data.xpos[bodies_to_monitor_ids, :3].copy()
            bodies_dist = np.linalg.norm(bodies_end_pos - bodies_init_pos, axis=1)
            # ref: https://math.stackexchange.com/questions/90081/quaternion-distance
            # bodies_quat_dist = 1 - np.dot(bodies_end_quat, bodies_init_quat)

            bodies_detected_mask = bodies_dist >= DEFAULT_DIST_THRESHOLD
            bodies_detected_ids = bodies_to_monitor_ids[bodies_detected_mask]
            bodies_detected_diff_pos = bodies_dist[bodies_detected_mask]
            bodies_detected_names = [
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bid) for bid in bodies_detected_ids
            ]

            bodies_jitter_mask = bodies_to_monitor_cumdist >= SETTINGS.body_jitter_threshold
            bodies_jitter_ids = bodies_to_monitor_ids[bodies_jitter_mask]
            bodies_jitter_diff = bodies_to_monitor_cumdist[bodies_jitter_mask]
            bodies_jitter_names = [
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bid) for bid in bodies_jitter_ids
            ]

        if has_joints:
            joints_end_qpos = data.qpos[joints_to_monitor_qposadr].copy()
            joints_delta_qpos = np.abs(joints_end_qpos - joints_init_qpos)
            joints_open_percent = joints_delta_qpos / joints_to_monitor_range_diff

            joints_detected_mask = joints_open_percent >= DEFAULT_JOINT_PERCENT_THRESHOLD
            joints_detected_ids = joints_to_monitor_ids[joints_detected_mask]
            joints_detected_diff_qpos = joints_delta_qpos[joints_detected_mask]
            joints_detected_open_percent = joints_open_percent[joints_detected_mask]
            joints_detected_names = [
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT.value, jid)
                for jid in joints_detected_ids
            ]

            # Get all objects that are within a site that belongs to the corresponding articulated
            # object and the specific joint (both site and joint should share same parent)
            for jnt_id in joints_detected_ids:
                body_id = model.jnt_bodyid[jnt_id].item()
                for candidate_site_id, body_root_id in site_to_body_within_map.items():
                    if body_id == model.site_bodyid[candidate_site_id].item():
                        body_root_name = mj.mj_id2name(
                            model, mj.mjtObj.mjOBJ_BODY.value, body_root_id
                        )
                        bodies_within_sites.add(body_root_name)
                        break

        results = {
            "house_name": house_filepath.stem,
            "bodies": {"names": [], "diff_dist": []},
            "joints": {"names": [], "diff_qpos": [], "open_percent": [], "bodies_make_drift": []},
            "jitter": {"names": [], "diff": []},
            "total": {"bodies": 0, "joints": 0},
        }
        if has_bodies:
            results["bodies"] = {
                "names": bodies_detected_names,
                "diff_dist": bodies_detected_diff_pos.tolist(),
            }
            results["jitter"] = {
                "names": bodies_jitter_names,
                "diff": bodies_jitter_diff.tolist(),
            }
            results["total"]["bodies"] = len(bodies_to_monitor_ids)
        if has_joints:
            results["joints"] = {
                "names": joints_detected_names,
                "diff_qpos": joints_detected_diff_qpos.tolist(),
                "open_percent": joints_detected_open_percent.tolist(),
                "bodies_make_drift": list(bodies_within_sites),
            }
            results["total"]["joints"] = len(joints_to_monitor_ids)
    except Exception as e:
        results = {
            "house_name": house_filepath.stem,
            "error": f"Got an error while processing house '{house_filepath.stem}' - error: {e}",
        }
        success = False

    return success, results, categories_counts


def show_results_summary(settings: TestSettings) -> None:
    if settings.results_filepath.is_file():
        sort_results_stability_test_by_scene_number(
            settings.results_filepath,
            settings.dataset,
            settings.split,
        )
        get_fail_info_stability_test(
            settings.results_filepath,
            settings.dataset,
            settings.split,
            settings.identifier,
        )

        (
            count_total_bodies,
            count_total_joints,
            count_bodies_fail,
            count_joints_fail,
            count_bodies_jitter,
            count_scenes_pass,
            count_scenes_fail,
        ) = get_stability_tests_results_info(settings.results_filepath)

        counts_cat_bodies, counts_cat_joints, counts_cat_jitter, counts_overall = (
            get_categories_failed_stability_test(settings.results_filepath)
        )

        print("\n")
        print("#--------------------------------------------")
        print("STABILITY TEST")
        print("\n")
        print("Setup:")
        print(f"dataset              : {settings.dataset}")
        print(f"split               : {settings.split}")
        print(f"identifier          : {settings.identifier}")
        print(f"mujoco version      : {mj.__version__}")
        print(f"use-sleep-island    : {settings.use_sleep_island}")
        print(f"jitter-threshold    : {settings.body_jitter_threshold}")
        print("\n")

        build_settings_filepath = settings.houses_folder / "housegen_build_settings.json"
        if build_settings_filepath.is_file():
            with open(build_settings_filepath, "r") as fhandle:
                data = json.load(fhandle)
            print("House build parameters:")
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
                print(f"mujoco-version              : {data['mujoco-version']}")
            print(f"mujoco-version-run          : {mj.__version__}")
            print("\n")

        print("Test settings:")

        print("\n")
        print(f"total bodies tested         : {count_total_bodies}")
        print(f"total joints tested         : {count_total_joints}")
        print(f"total bodies fail           : {count_bodies_fail}")
        print(f"total joints fail           : {count_joints_fail}")
        print(f"total bodies jitter         : {count_bodies_jitter}")
        print(f"total number scenes pass    : {count_scenes_pass}")
        print(f"total number scenes fail    : {count_scenes_fail}")
        print("")
        print("Categories (bodies): ")
        for category in counts_cat_bodies:
            print(f"\t{category:>10} : {counts_cat_bodies[category]} / {counts_overall[category]}")
        print("Categories (joints): ")
        for category in counts_cat_joints:
            print(f"\t{category:>10} : {counts_cat_joints[category]}")
        print("Categories (bodies-jitter): ")
        for category in counts_cat_jitter:
            print(f"\t{category:>10} : {counts_cat_jitter[category]} / {counts_overall[category]}")
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
    for cat, num in categories_counts.items():
        if cat not in results_all["categories"]:
            results_all["categories"][cat] = num
        else:
            results_all["categories"][cat] += num
    results_all["results"][house_name] = scene_results
    with open(settings.results_filepath, "w") as fhandle:
        json.dump(results_all, fhandle, indent=4, default=json_serializer)


def save_scene_errors(settings: TestSettings, error_message: str) -> None:
    with open(settings.errors_filepath, "a") as fhandle:
        fhandle.write(error_message + "\n")


def main() -> int:
    global SETTINGS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--house",
        type=str,
        default="",
        help="The path to a THOR mjcf house to be tested",
    )
    parser.add_argument(
        "--houses-folder",
        type=str,
        default="",
        help="The path to the folder containing all houses to be tested",
    )
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
        "--identifier",
        type=str,
        default=f"{datetime.now().strftime('%m%d%y')}" + ("_weka" if IS_BEAKER else ""),
        help="A suffix to use when saving the results json files",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="The number of parallel processes to use for the test",
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
        "--copy-results-to",
        type=str,
        default="",
        help="Path where to copy all test results files",
    )
    parser.add_argument(
        "--copy-results-suffix",
        type=str,
        default="",
        help="An extra suffix to add to the results files, so we can run on a whole dataset in parts",
    )
    parser.add_argument(
        "--jitter-threshold",
        type=float,
        default=DEFAULT_JITTER_THRESHOLD,
        help="The threshold to use to detect jitter (cummulative distance traveled)",
    )
    parser.add_argument(
        "--use-sleep-island",
        action="store_true",
        help="Whether or not to enable sleeping island (requires 3.3.8)",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=DEFAULT_SETTLE_TIME,
        help="The amount of time to settle the scene for, in seconds",
    )
    parser.add_argument(
        "--monitor-time",
        type=float,
        default=DEFAULT_MONITOR_TIME,
        help="The amount of time to monitor the test for, in seconds",
    )

    args = parser.parse_args()

    SETTINGS = TestSettings(
        dataset=args.dataset,
        split=args.split,
        identifier=args.identifier,
        houses_folder=get_houses_folder(args),
        results_filepath=Path(
            RESULTS_FILEPATH.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        warnings_filepath=Path(
            WARNINGS_FILEPATH.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        errors_filepath=Path(
            ERRORS_FILEPATH.format(
                dataset=args.dataset,
                split=args.split,
                identifier=args.identifier,
            )
        ),
        copy_results_to=args.copy_results_to,
        copy_results_suffix=args.copy_results_suffix,
        body_jitter_threshold=args.jitter_threshold,
        use_sleep_island=args.use_sleep_island,
        settle_time=args.settle_time,
        settle_steps=int(args.settle_time / DEFAULT_TIMESTEP),
        monitor_time=args.monitor_time,
        monitor_steps=int(args.monitor_time / DEFAULT_TIMESTEP),
    )

    if args.summary:
        if args.results_file != "":
            SETTINGS.results_filepath = Path(args.results_file)
        show_results_summary(SETTINGS)
        return 0

    results_all = dict(results=dict(), settings=asdict(SETTINGS), categories=dict())
    if SETTINGS.results_filepath.is_file():
        with open(SETTINGS.results_filepath, "r") as fhandle:
            results_all = json.load(fhandle)

    if args.house != "":
        house_filepath = Path(args.house)
        if not house_filepath.is_file():
            return 1
        scene_success, scene_results, categories_counts = run_stability_test(house_filepath)
        if scene_success:
            save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
        else:
            save_scene_errors(SETTINGS, scene_results.get("error", ""))

    elif SETTINGS.houses_folder.is_dir():
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
            print(f"Should give --dataset, but got invalid '{SETTINGS.dataset}'")
            return 1

        if args.max_workers > 1:
            results = p_uimap(run_stability_test, xmls_to_test, num_cpus=args.max_workers)
            for scene_success, scene_results, categories_counts in results:
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))
        else:
            for house_xml in tqdm(xmls_to_test):
                scene_success, scene_results, categories_counts = run_stability_test(house_xml)
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))

    show_results_summary(SETTINGS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
