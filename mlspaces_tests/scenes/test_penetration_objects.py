import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco as mj
from p_tqdm import p_uimap
from test_utils import (
    get_categories_failed_penetration_test,
    get_fail_info_penetration_test,
    get_penetration_tests_results_info,
    sort_results_penetration_test_by_scene_number,
)
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent

RESULTS_FILEPATH = "penetration_test_results_{dataset}_{split}_{identifier}.json"
WARNINGS_FILEPATH = "penetration_test_warnings_{dataset}_{split}_{identifier}.json"
ERRORS_FILEPATH = "penetration_test_errors_{dataset}_{split}_{identifier}.json"

DEFAULT_HOUSES_FOLDER_WEKA = (
    "/weka/prior-default/datasets/mujoco-thor/assets/scenes/{dataset}-{split}-refactor"
)
DEFAULT_HOUSES_FOLDER_LOCAL = "assets/scenes/{dataset}-{split}"
DEFAULT_HOUSES_FOLDER_ITHOR_WEKA = "/weka/prior-default/datasets/mujoco-thor/assets/scenes/ithor"
DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL = "assets/scenes/ithor"

DEFAULT_TIMESTEP = 0.002
DEFAULT_SETTLE_TIME = 1.0  # Settle for 1 seconds
DEFAULT_SETTLE_STEPS = int(DEFAULT_SETTLE_TIME / DEFAULT_TIMESTEP)  # 1000 steps @ dt = 0.002
DEFAULT_PENETRATION_THRESHOLD = -0.01

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

    copy_results_to: str = ""
    copy_results_suffix: str = ""

    use_sleep_island: bool = HAS_SLEEP_ISLAND_SUPPORT


SETTINGS: TestSettings | None = None


def json_serializer(obj):
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def is_free_body(model: mj.MjModel, body_id: int) -> bool:
    return body_id != 0 and model.body_dofnum[body_id].item() == 6


def get_free_bodies_ids(model: mj.MjModel) -> list[int]:
    return [bid for bid in range(model.nbody) if is_free_body(model, bid)]


def run_penetration_test(house_filepath: Path) -> tuple[bool, dict[str, Any], Counter]:
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

        freebodies_set = set(get_free_bodies_ids(model))
        bodies_counts: defaultdict[int, set[int]] = defaultdict(set)

        n_settle_steps = int(SETTINGS.settle_time / SETTINGS.timestep)

        mj.mj_step(model, data, nstep=n_settle_steps)

        bodies_detected_pairs: list[tuple[int, int]] = []
        geoms_detected_pairs: list[tuple[int, int]] = []
        geoms_detected_pairs_dist: list[float] = []
        for i_con in range(data.ncon):
            contact = data.contact[i_con]
            geom_id_1, geom_id_2 = contact.geom[0], contact.geom[1]
            body_id_1, body_id_2 = model.geom_bodyid[[geom_id_1, geom_id_2]]
            root_id_1, root_id_2 = model.body_rootid[[body_id_1, body_id_2]]

            if contact.dist >= DEFAULT_PENETRATION_THRESHOLD:
                continue

            bodies_detected_pairs.append((root_id_1, root_id_2))
            geoms_detected_pairs.append((geom_id_1, geom_id_2))
            geoms_detected_pairs_dist.append(contact.dist)

            bodies_counts[root_id_1].add(root_id_2)
            bodies_counts[root_id_2].add(root_id_1)

        bodies_pairs_a, bodies_pairs_b = [], []
        geoms_pairs_a, geoms_pairs_b = [], []
        for (body_a, body_b), (geom_a, geom_b) in zip(bodies_detected_pairs, geoms_detected_pairs):
            bodies_pairs_a.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_a))
            bodies_pairs_b.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, body_b))
            geoms_pairs_a.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, geom_a))
            geoms_pairs_b.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, geom_b))

        counts = {}
        catcounts = Counter()
        for bid, bcountset in bodies_counts.items():
            bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bid)
            bothernames = [
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bid) for bid in bcountset
            ]
            counts[bname] = dict(
                free=bid in freebodies_set, count=len(bothernames), other=list(bothernames)
            )
            blemma = bname.split("_")[0]
            catcounts[blemma] += 1
            for bothername in bothernames:
                botherlemma = bothername.split("_")[0]
                catcounts[botherlemma] += 1
        results = {
            "pairs": {
                "body-a": bodies_pairs_a,
                "body-b": bodies_pairs_b,
                "geom-a": geoms_pairs_a,
                "geom-b": geoms_pairs_b,
                "dist": geoms_detected_pairs_dist,
            },
            "counts": counts,
            "catcounts": dict(catcounts),
            "total": len(bodies_pairs_a),
            "house_name": house_filepath.stem,
        }

    except Exception as e:
        results = {
            "house_name": house_filepath.stem,
            "error": f"Got an error while processing house '{house_filepath.stem}' - error: {e}",
        }
        success = False

    return success, results, categories_counts


def show_results_summary(settings: TestSettings) -> None:
    if settings.results_filepath.is_file():
        sort_results_penetration_test_by_scene_number(
            settings.results_filepath,
            settings.dataset,
            settings.split,
        )
        get_fail_info_penetration_test(
            settings.results_filepath,
            settings.dataset,
            settings.split,
            settings.identifier,
        )

        count_pairs_fail, count_scenes_pass, count_scenes_fail = get_penetration_tests_results_info(
            settings.results_filepath
        )

        counts_cat_bodies, counts_overall = get_categories_failed_penetration_test(
            settings.results_filepath
        )

        print("\n")
        print("#--------------------------------------------")
        print("INTERSECTION TEST")
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
        print(f"total pairs fail           : {count_pairs_fail}")
        print(f"total number scenes pass    : {count_scenes_pass}")
        print(f"total number scenes fail    : {count_scenes_fail}")
        print("")
        print("Categories (bodies): ")
        for category in counts_cat_bodies:
            if category == "world":
                continue
            print(f"\t{category:>10} : {counts_cat_bodies[category]} / {counts_overall[category]}")
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
        "--copy-results-suffix",
        type=str,
        default="",
        help="An extra suffix to add to the results files, so we can run on a whole dataset in parts",
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
        "--use-sleep-island",
        action="store_true",
        help="Whether or not to enable sleeping island (requires 3.3.8)",
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
        use_sleep_island=args.use_sleep_island,
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
        scene_success, scene_results, categories_counts = run_penetration_test(house_filepath)
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
            results = p_uimap(run_penetration_test, xmls_to_test, num_cpus=args.max_workers)
            for scene_success, scene_results, categories_counts in results:
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))
        else:
            for house_xml in tqdm(xmls_to_test):
                scene_success, scene_results, categories_counts = run_penetration_test(house_xml)
                if scene_success:
                    save_scene_results(SETTINGS, results_all, scene_results, categories_counts)
                else:
                    save_scene_errors(SETTINGS, scene_results.get("error", ""))

    show_results_summary(SETTINGS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
