import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import trimesh

from molmo_spaces.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA as TYPE_TO_LEMMA,
)

WARNINGS_TO_CHECK = [
    mj.mjtWarning.mjWARN_BADQACC,
    mj.mjtWarning.mjWARN_BADQPOS,
    mj.mjtWarning.mjWARN_BADQVEL,
    mj.mjtWarning.mjWARN_CNSTRFULL,
    mj.mjtWarning.mjWARN_CONTACTFULL,
]
WARNINGS_NAMES = [
    "bad-qacc",
    "bad-qpos",
    "bad-qvel",
    "constraint-full",
    "contact-full",
]

ARTICULABLE_CATEGORIES_NAMES = [
    "Microwave",
    "Fridge",
    "Toaster",
    "Dresser",
    "Light_Switch",
    "Toilet",
    "Book",
    "Shelving_Unit",
    "Side_Table",
    "Coffee_Table",
    "Desk",
    "Laptop",
    "Doorway",
    "Laundry_Hamper",
    "Safe",
    "Box",
    "Bathroom_Faucet",
]
ARTICULABLE_CATEGORIES_LEMMAS = [
    TYPE_TO_LEMMA[category].replace("_", "")
    if category in TYPE_TO_LEMMA
    else category.lower().split("_")[0].replace("_", "")
    for category in ARTICULABLE_CATEGORIES_NAMES
]

LEMMA_TO_CATEGORY = {
    lemma: category
    for lemma, category in zip(ARTICULABLE_CATEGORIES_LEMMAS, ARTICULABLE_CATEGORIES_NAMES)
}

ARTICULABLE_CATEGORIES_PATTERNS = {
    lemma: re.compile(lemma + r"_\d+") for lemma in ARTICULABLE_CATEGORIES_LEMMAS
}

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


# --------------------------------------------------------------------------------------------------
#                                   COMMON HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------


def get_bodies_in_contact_with_body(body_id: int, model: mj.MjModel, data: mj.MjData) -> list[int]:
    bodies_in_contact = []
    for con_i in range(data.ncon):
        geom_id_0 = data.contact[con_i].geom[0]
        geom_id_1 = data.contact[con_i].geom[1]

        body_id_0 = model.geom_bodyid[geom_id_0].item()
        body_id_1 = model.geom_bodyid[geom_id_1].item()

        root_body_id_0 = model.body_rootid[body_id_0].item()
        root_body_id_1 = model.body_rootid[body_id_1].item()

        if root_body_id_0 == body_id:
            bodies_in_contact.append(root_body_id_1)
        elif root_body_id_1 == body_id:
            bodies_in_contact.append(root_body_id_0)

    return bodies_in_contact


def warn_info_to_str(model: mj.MjModel, data: mj.MjData, warn_id: mj.mjtWarning) -> str:
    if warn_id == mj.mjtWarning.mjWARN_BADQACC or warn_id == mj.mjtWarning.mjWARN_BADQVEL:
        dof_id = data.warning[warn_id].lastinfo
        body_id = model.dof_bodyid[dof_id].item()
        joint_id = model.dof_jntid[dof_id].item()
        root_body_id = model.body_rootid[body_id].item()
        root_body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, root_body_id)
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT.value, joint_id)
        return f"body-name: {root_body_name} - joint-name: {joint_name}"
    return ""


def collect_warnings(model: mj.MjModel, data: mj.MjData) -> list[str]:
    warnings = []
    for warn_id, warn_name in zip(WARNINGS_TO_CHECK, WARNINGS_NAMES):
        warn = data.warning[warn_id]
        if warn.number < 1:
            continue
        warning_str = mj.mju_warningText(warn_id.value, warn.lastinfo)
        msg = f"{warn_name} > {warning_str} - {warn_info_to_str(model, data, warn_id)} - number: {warn.number} - time: {data.time}"
        warnings.append(msg)

    return warnings


def collect_badqacc_body(model: mj.MjModel, data: mj.MjData) -> str:
    bad_body = ""
    warn = data.warning[mj.mjtWarning.mjWARN_BADQACC]
    if warn.number > 0:
        dof_id = warn.lastinfo
        body_id = model.dof_bodyid[dof_id].item()
        root_body_id = model.body_rootid[body_id].item()
        bad_body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, root_body_id)

    return bad_body


# --------------------------------------------------------------------------------------------------
#                              HELPER FUNCTIONS FOR LIFT FORCE TEST
# --------------------------------------------------------------------------------------------------


def sort_results_lift_force_test_by_scene_number(
    filepath: Path, flavor: str, split: str = "train"
) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    results_info = data["results"]
    pattern = r"FloorPlan(\d+)_physics" if flavor == "ithor" else (split + r"_(\d+)")

    matching_keys = [key for key in results_info if re.match(pattern, key)]
    sorted_keys = sorted(matching_keys, key=lambda x: int(re.search(pattern, x).group(1)))  # type: ignore

    ordered_results_info = {key: results_info[key] for key in sorted_keys}
    for key in ordered_results_info:
        if "all_pass" not in results_info[key]:
            results_info[key]["all_pass"] = results_info[key]["counts"]["failed"]["num"] == 0

    data["results"] = ordered_results_info

    data["info"] = {
        "num_scenes_pass": 0,
        "scenes_pass_names": [],
        "num_scenes_dont_pass": 0,
        "scenes_dont_pass_names": [],
    }

    for key in ordered_results_info:
        if ordered_results_info[key]["all_pass"]:
            data["info"]["num_scenes_pass"] += 1
            data["info"]["scenes_pass_names"].append(key)
        else:
            data["info"]["num_scenes_dont_pass"] += 1
            data["info"]["scenes_dont_pass_names"].append(key)

    with open(filepath, "w") as fhandle:
        json.dump(data, fhandle, indent=4)


def get_fail_info_lift_force_test(filepath: Path, flavor: str, split: str, identifier: str) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    fail_info: list[tuple[str, str]] = []  # [(scene_name, body_name)]
    for key in data["results"]:
        fail_info.extend(
            [(key, body) for body in data["results"][key]["counts"]["failed"]["cant_lift"]["names"]]
        )

    with open(f"stats_fails_lift_test_{flavor}_{split}_{identifier}.txt", "w") as fhandle:
        for scene_name, body_name in fail_info:
            body_idx = data["results"][scene_name]["counts"]["failed"]["cant_lift"]["names"].index(
                body_name
            )
            diff_height = data["results"][scene_name]["counts"]["failed"]["cant_lift"][
                "diff-height"
            ][body_idx]
            dist_moved = data["results"][scene_name]["counts"]["failed"]["cant_lift"][
                "distance-moved"
            ][body_idx]
            weight = (
                9.81
                * data["results"][scene_name]["counts"]["failed"]["cant_lift"]["masses"][body_idx]
            )
            msg_weight = "greater-than-max" if weight > 30 else "ok"
            fhandle.write(
                f"{scene_name}, {body_name}, w={weight}, {msg_weight}, {diff_height}, {dist_moved}\n"
            )


@dataclass
class CantLiftResult:
    house_name: str
    body_name: str
    body_mass: float
    body_diff_height: float
    body_distance_moved: float


def get_lift_tests_results_info(
    filepath: Path,
) -> tuple[int, list[CantLiftResult], int, int, int]:
    count_total_pickupable = 0
    count_total_badqacc = 0
    cant_lift: list[CantLiftResult] = []

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    results = data.get("results", {})
    for key in results:
        count_total_pickupable += results[key]["num_pickupable_bodies"]
        count_total_badqacc += len(results[key]["counts"]["badqacc"])
        for body, mass, diff_height, dist_moved in zip(
            results[key]["counts"]["failed"]["cant_lift"]["names"],
            results[key]["counts"]["failed"]["cant_lift"]["masses"],
            results[key]["counts"]["failed"]["cant_lift"]["diff-height"],
            results[key]["counts"]["failed"]["cant_lift"]["distance-moved"],
        ):
            cant_lift.append(CantLiftResult(key, body, mass, diff_height, dist_moved))

    num_scenes_pass = data["info"]["num_scenes_pass"]
    num_scenes_fail = data["info"]["num_scenes_dont_pass"]

    return (
        count_total_pickupable,
        cant_lift,
        num_scenes_pass,
        num_scenes_fail,
        count_total_badqacc,
    )


# --------------------------------------------------------------------------------------------------
#                             HELPER FUNCTIONS FOR ARTICULATION FORCE TEST
# --------------------------------------------------------------------------------------------------


def sort_results_articulation_force_test_by_scene_number(
    filepath: Path, flavor: str, split: str = "train"
) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    results_info = data["results"]
    pattern = r"FloorPlan(\d+)_physics" if flavor == "ithor" else (split + r"_(\d+)")

    matching_keys = [key for key in results_info if re.match(pattern, key)]
    sorted_keys = sorted(matching_keys, key=lambda x: int(re.search(pattern, x).group(1)))  # type: ignore

    for key in results_info:
        if "all_pass" not in results_info[key]:
            results_info[key]["all_pass"] = results_info[key]["num_joints_cant_open"] == 0

    ordered_data = {key: results_info[key] for key in sorted_keys}
    data["results"] = ordered_data

    data["info"] = {
        "num_scenes_pass": 0,
        "scenes_pass_names": [],
        "num_scenes_dont_pass": 0,
        "scenes_dont_pass_names": [],
    }

    for key in ordered_data:
        if ordered_data[key]["all_pass"]:
            data["info"]["num_scenes_pass"] += 1
            data["info"]["scenes_pass_names"].append(key)
        else:
            data["info"]["num_scenes_dont_pass"] += 1
            data["info"]["scenes_dont_pass_names"].append(key)

    with open(filepath, "w") as fhandle:
        json.dump(data, fhandle, indent=4)


def get_fail_info_articulation_force_test(
    filepath: Path, flavor: str, split: str, identifier: str
) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    fail_info: list[tuple[str, str]] = []
    results_info = data.get("results", {})
    for key in results_info:
        fail_info.extend([(key, joint) for joint in results_info[key]["joints_cant_open"]])

    with open(f"stats_fails_articulation_test_{flavor}_{split}_{identifier}.txt", "w") as fhandle:
        for scene_name, joint_name in fail_info:
            fhandle.write(f"{scene_name}, {joint_name}\n")


def get_articulation_tests_results_info(filepath: Path) -> tuple[int, int, int, int]:
    count_total_joints = 0
    count_cant_open_joints = 0

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    for scene_name in data["results"]:
        count_total_joints += data["results"][scene_name]["num_articulable_joints"]
        count_cant_open_joints += data["results"][scene_name]["num_joints_cant_open"]

    count_scenes_pass = data["info"]["num_scenes_pass"]
    count_scenes_fail = data["info"]["num_scenes_dont_pass"]

    return count_total_joints, count_cant_open_joints, count_scenes_pass, count_scenes_fail


def get_lemma_name_from_joint_name(joint_name: str) -> str:
    category_name = ""
    for category, lemma in ARTICULABLE_CATEGORIES_PATTERNS.items():
        if re.search(lemma, joint_name):
            category_name = category
            break
    return category_name


def get_object_categories_failed_articulation_test(filepath: Path) -> Counter:
    counts = Counter()
    if not filepath.exists():
        return counts

    data = {}
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)
    data_res = data.get("results", {})
    for scene_info in data_res.values():
        joints_cant_open = scene_info["joints_cant_open"]
        for joint_name in joints_cant_open:
            joint_lemma = joint_name.split("_")[0]
            counts[joint_lemma] += 1
    return counts


def get_object_categories_failed_lift_test(filepath: Path) -> tuple[Counter, Counter]:
    counts = Counter()
    counts_overall = Counter()

    if not filepath.is_file():
        return counts, counts_overall

    data = {}
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    if "categories" in data:
        counts_overall = Counter(data["categories"])

    data_res = data.get("results", {})
    for scene_info in data_res.values():
        bodies_cant_lift = scene_info["counts"]["failed"]["cant_lift"]["names"]
        for body_name in bodies_cant_lift:
            body_lemma = body_name.split("_")[0]
            counts[body_lemma] += 1
    return counts, counts_overall


# --------------------------------------------------------------------------------------------------
#                               HELPER FUNCTIONS FOR STABILITY TEST
# --------------------------------------------------------------------------------------------------


def sort_results_stability_test_by_scene_number(
    filepath: Path, flavor: str, split: str = "train"
) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    results_info = data["results"]
    pattern = r"FloorPlan(\d+)_physics" if flavor == "ithor" else (split + r"_(\d+)")

    matching_keys = [key for key in results_info if re.match(pattern, key)]
    sorted_keys = sorted(matching_keys, key=lambda x: int(re.search(pattern, x).group(1)))  # type: ignore

    for key in results_info:
        if "all_pass" not in results_info[key]:
            n_bodies = len(results_info[key]["bodies"]["names"])
            n_joints = len(results_info[key]["joints"]["names"])
            n_jitter = len(results_info[key]["jitter"]["names"])
            results_info[key]["all_pass"] = n_bodies == 0 and n_joints == 0 and n_jitter == 0

    ordered_data = {key: results_info[key] for key in sorted_keys}
    data["results"] = ordered_data

    data["info"] = {
        "num_scenes_pass": 0,
        "scenes_pass_names": [],
        "num_scenes_dont_pass": 0,
        "scenes_dont_pass_names": [],
    }

    for key in ordered_data:
        if ordered_data[key]["all_pass"]:
            data["info"]["num_scenes_pass"] += 1
            data["info"]["scenes_pass_names"].append(key)
        else:
            data["info"]["num_scenes_dont_pass"] += 1
            data["info"]["scenes_dont_pass_names"].append(key)

    with open(filepath, "w") as fhandle:
        json.dump(data, fhandle, indent=4)


def get_fail_info_stability_test(filepath: Path, flavor: str, split: str, identifier: str) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    fail_info: list[tuple[str, str]] = []
    results_info = data.get("results", {})
    for key in results_info:
        bodies_info = results_info[key]["bodies"]
        names, diffs = bodies_info["names"], bodies_info["diff_dist"]
        fail_info.extend([(key, f"body: {name}, diff: {diff}") for name, diff in zip(names, diffs)])
        joints_info = results_info[key]["joints"]
        names, percents = joints_info["names"], joints_info["open_percent"]
        fail_info.extend(
            [(key, f"joint: {name}, percent: {percent}") for name, percent in zip(names, percents)]
        )
        jitter_info = results_info[key]["jitter"]
        names, diffs = jitter_info["names"], jitter_info["diff"]
        fail_info.extend(
            [(key, f"body-jitter: {name}, diff: {diff}") for name, diff in zip(names, diffs)]
        )

    with open(f"stats_fails_stability_test_{flavor}_{split}_{identifier}.txt", "w") as fhandle:
        for scene_name, joint_name in fail_info:
            fhandle.write(f"{scene_name}, {joint_name}\n")


def get_stability_tests_results_info(filepath: Path) -> tuple[int, int, int, int, int, int, int]:
    count_bodies_fail = 0
    count_joints_fail = 0
    count_bodies_jitter = 0
    count_total_bodies = 0
    count_total_joints = 0

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    for scene_name in data["results"]:
        scene_results = data["results"][scene_name]
        count_bodies_fail += len(scene_results["bodies"]["names"])
        count_joints_fail += len(scene_results["joints"]["names"])
        count_bodies_jitter += len(scene_results["jitter"]["names"])
        count_total_bodies += scene_results["total"]["bodies"]
        count_total_joints += scene_results["total"]["joints"]

    count_scenes_pass = data["info"]["num_scenes_pass"]
    count_scenes_fail = data["info"]["num_scenes_dont_pass"]

    return (
        count_total_bodies,
        count_total_joints,
        count_bodies_fail,
        count_joints_fail,
        count_bodies_jitter,
        count_scenes_pass,
        count_scenes_fail,
    )


def get_categories_failed_stability_test(filepath: Path) -> tuple[Counter, Counter, Counter, Counter]:
    counts_bodies, counts_joints, counts_jitter = Counter(), Counter(), Counter()
    counts_overall = Counter()
    if not filepath.is_file():
        return counts_bodies, counts_joints, counts_jitter, counts_overall

    data = {}
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    if "categories" in data:
        counts_overall = Counter(data["categories"])

    data_res = data.get("results", {})
    for scene_info in data_res.values():
        bodies_unstable = scene_info["bodies"]["names"]
        for body_name in bodies_unstable:
            body_lemma = body_name.split("_")[0]
            counts_bodies[body_lemma] += 1

        joints_unstable = scene_info["joints"]["names"]
        for joint_name in joints_unstable:
            joint_lemma = joint_name.split("_")[0]
            counts_joints[joint_lemma] += 1

        bodies_jitter = scene_info["jitter"]["names"]
        for body_name in bodies_jitter:
            body_lemma = body_name.split("_")[0]
            counts_jitter[body_lemma] += 1
    return counts_bodies, counts_joints, counts_jitter, counts_overall


# --------------------------------------------------------------------------------------------------
#                                 HELPER FUNCTIONS FOR SPEED TEST
# --------------------------------------------------------------------------------------------------


@dataclass
class StatsRuntime:
    realtime_factor: float
    steptime: float
    pos_collision: float
    col_narrowphase: float


def get_stats_runtime_test(filepath: Path) -> dict[str, StatsRuntime]:
    stats: dict[str, StatsRuntime] = dict()
    if not filepath.is_file():
        return stats

    data = {}
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)
    data_res = data.get("results", {})
    for scene_name, scene_info in data_res.items():
        stats[scene_name] = StatsRuntime(
            realtime_factor=scene_info["real_time_factor"],
            steptime=scene_info["mjc_steptime"],
            pos_collision=scene_info["mjc_timers"]["pos_collision"],
            col_narrowphase=scene_info["mjc_timers"]["col_narrowphase"],
        )

    return stats


# --------------------------------------------------------------------------------------------------
#                             HELPER FUNCTIONS FOR PENETRATION TEST
# --------------------------------------------------------------------------------------------------


def sort_results_penetration_test_by_scene_number(
    filepath: Path, flavor: str, split: str = "train"
) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    results_info = data["results"]
    pattern = r"FloorPlan(\d+)_physics" if flavor == "ithor" else (split + r"_(\d+)")

    matching_keys = [key for key in results_info if re.match(pattern, key)]
    sorted_keys = sorted(matching_keys, key=lambda x: int(re.search(pattern, x).group(1)))  # type: ignore

    for key in results_info:
        if "all_pass" not in results_info[key]:
            n_pairs = len(results_info[key]["pairs"]["body-a"])
            results_info[key]["all_pass"] = n_pairs == 0

    ordered_data = {key: results_info[key] for key in sorted_keys}
    data["results"] = ordered_data

    data["info"] = {
        "num_scenes_pass": 0,
        "scenes_pass_names": [],
        "num_scenes_dont_pass": 0,
        "scenes_dont_pass_names": [],
    }

    for key in ordered_data:
        if ordered_data[key]["all_pass"]:
            data["info"]["num_scenes_pass"] += 1
            data["info"]["scenes_pass_names"].append(key)
        else:
            data["info"]["num_scenes_dont_pass"] += 1
            data["info"]["scenes_dont_pass_names"].append(key)

    with open(filepath, "w") as fhandle:
        json.dump(data, fhandle, indent=4)


def get_fail_info_penetration_test(
    filepath: Path, flavor: str, split: str, identifier: str
) -> None:
    if not filepath.exists():
        print(f"Results file `{filepath.as_posix()}` doesn't exist")
        return
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    fail_info: list[tuple[str, str]] = []
    results_info = data.get("results", {})
    for scene_name in results_info:
        counts_info = results_info[scene_name]["counts"]
        for body_name, body_info in counts_info.items():
            fail_info.append(
                (
                    scene_name,
                    f"body: {body_name}, count: {body_info['count']}, free: {body_info['free']}",
                )
            )

    with open(f"stats_fails_penetration_test_{flavor}_{split}_{identifier}.txt", "w") as fhandle:
        for scene_name, msg in fail_info:
            fhandle.write(f"{scene_name}, {msg}\n")


def get_categories_failed_penetration_test(filepath: Path) -> tuple[Counter, Counter]:
    counts_bodies = Counter()
    counts_overall = Counter()
    if not filepath.is_file():
        return counts_bodies, counts_overall

    data = {}
    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    if "categories" in data:
        counts_overall = Counter(data["categories"])

    data_res = data.get("results", {})
    for scene_info in data_res.values():
        counts_info = scene_info["counts"]
        for body_name in counts_info:
            lemma = body_name.split("_")[0]
            counts_bodies[lemma] += 1
    return counts_bodies, counts_overall


def get_penetration_tests_results_info(filepath: Path) -> tuple[int, int, int]:
    count_pairs_fail = 0

    with open(filepath, "r") as fhandle:
        data = json.load(fhandle)

    for scene_name in data["results"]:
        scene_results = data["results"][scene_name]
        count_pairs_fail += len(scene_results["counts"])

    count_scenes_pass = data["info"]["num_scenes_pass"]
    count_scenes_fail = data["info"]["num_scenes_dont_pass"]

    return count_pairs_fail, count_scenes_pass, count_scenes_fail


# --------------------------------------------------------------------------------------------------
#                                 MESH-GEOMETRY HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------


def get_geometry_counts(mesh_filepath: Path) -> tuple[int, int]:
    n_verts, n_faces = 0, 0
    mesh = trimesh.load_mesh(mesh_filepath.as_posix())
    n_verts = mesh.vertices.shape[0]
    n_faces = mesh.faces.shape[0]

    return n_verts, n_faces
