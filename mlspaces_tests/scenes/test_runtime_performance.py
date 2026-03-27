import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Manager
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Any, cast

import mujoco as mj
import numpy as np
from test_utils import get_stats_runtime_test
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent

DEFAULT_HOUSES_FOLDER_WEKA = (
    "/weka/prior-default/datasets/mujoco-thor/assets/scenes/{dataset}-{split}-refactor"
)
DEFAULT_HOUSES_FOLDER_LOCAL = "assets/scenes/{dataset}-{split}"
DEFAULT_HOUSES_FOLDER_ITHOR_WEKA = "/weka/prior-default/datasets/mujoco-thor/assets/scenes/ithor"
DEFAULT_HOUSES_FOLDER_ITHOR_LOCAL = "assets/scenes/ithor"

RESULTS_FILEPATH = "history_runtime_test_{dataset}_{split}_{identifier}.json"
WARNINGS_FILEPATH = "runtime_test_warnings_{dataset}_{split}_{identifier}.json"
ERRORS_FILEPATH = "runtime_test_errors_{dataset}_{split}_{identifier}.txt"

IS_BEAKER = "BEAKER_JOB_ID" in os.environ

DEFAULT_TIMESTEP = 0.002
DEFAULT_TEST_TIME = 10.0
DEFAULT_TEST_STEPS = int(DEFAULT_TEST_TIME / DEFAULT_TIMESTEP)
DEFAULT_TEST_RUNS = 5

MJC_VERSION = tuple(map(int, mj.__version__.split(".")))
HAS_SLEEP_ISLAND_SUPPORT = MJC_VERSION >= (3, 3, 8)

MUJOCO_WARNINGS_TO_CHECK = [
    mj.mjtWarning.mjWARN_BADQACC,
    mj.mjtWarning.mjWARN_BADQPOS,
    mj.mjtWarning.mjWARN_BADQVEL,
    mj.mjtWarning.mjWARN_CNSTRFULL,
    mj.mjtWarning.mjWARN_CONTACTFULL,
]
MUJOCO_WARNINGS_NAMES = [
    "bad-qacc",
    "bad-qpos",
    "bad-qvel",
    "constraint-full",
    "contact-full",
]


def collect_warnings(data: mj.MjData) -> list[str]:
    warnings = []
    for warn_id, warn_name in zip(MUJOCO_WARNINGS_TO_CHECK, MUJOCO_WARNINGS_NAMES):
        warn = data.warning[warn_id]
        if warn.number < 1:
            continue
        warning_str = mj.mju_warningText(warn_id.value, warn.lastinfo)
        msg = f"{warn_name} > {warning_str} - number: {warn.number} - time: {data.time}"
        warnings.append(msg)

    return warnings


MUJOCO_TIMERS_IDS = [
    # main API
    mj.mjtTimer.mjTIMER_STEP,  # step
    mj.mjtTimer.mjTIMER_FORWARD,  # forward
    mj.mjtTimer.mjTIMER_INVERSE,  # inverse
    # breakdown of step/forward
    mj.mjtTimer.mjTIMER_POSITION,  # fwdPosition
    mj.mjtTimer.mjTIMER_VELOCITY,  # fwdVelocity
    mj.mjtTimer.mjTIMER_ACTUATION,  # fwdActuation
    mj.mjtTimer.mjTIMER_CONSTRAINT,  # fwdConstraint
    mj.mjtTimer.mjTIMER_ADVANCE,  # mj_Euler, mj_implicit
    # breakdown of fwdPosition
    mj.mjtTimer.mjTIMER_POS_KINEMATICS,  # kinematics, com, tendon, transmission
    mj.mjtTimer.mjTIMER_POS_INERTIA,  # inertia computations
    mj.mjtTimer.mjTIMER_POS_COLLISION,  # collision detection
    mj.mjtTimer.mjTIMER_POS_MAKE,  # make constraints
    mj.mjtTimer.mjTIMER_POS_PROJECT,  # project constraints
    # breakdown of mj_collision
    mj.mjtTimer.mjTIMER_COL_BROAD,  # broadphase
    mj.mjtTimer.mjTIMER_COL_NARROW,  # narrowphase
]

MUJOCO_TIMERS_IDS_MAP = {mj.mjTIMERSTRING[timer_id]: timer_id for timer_id in MUJOCO_TIMERS_IDS}


@dataclass
class TestSettings:
    dataset: str
    split: str
    identifier: str
    houses_folder: Path

    results_path: Path
    warnings_path: Path
    errors_path: Path

    timestep: float = DEFAULT_TIMESTEP
    test_time: float = DEFAULT_TEST_TIME
    test_nsteps: int = DEFAULT_TEST_STEPS
    n_test_runs: int = DEFAULT_TEST_RUNS

    copy_results_to: str = ""

    use_sleep_island: bool = HAS_SLEEP_ISLAND_SUPPORT

    def to_dict(self) -> dict[str, Any]:
        return dict(
            timestep=self.timestep,
            test_time=self.test_time,
            results_path=self.results_path.as_posix(),
            warnings_path=self.warnings_path.as_posix(),
            errors_path=self.errors_path.as_posix(),
            dataset=self.dataset,
            identifier=self.identifier,
            use_sleep_island=self.use_sleep_island,
        )


@dataclass
class MjTimerInfo:
    duration: float = 0.0
    number: int = 0
    r_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        r_time = self.duration / self.number if self.number > 0 else 0.0
        return dict(duration=self.duration, number=self.number, time=r_time)


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


class ResultsCache:
    def __init__(self) -> None:
        self.house_name = ""
        self.num_bodies: int = 0
        self.num_geoms: int = 0
        self.num_joints: int = 0
        self.num_degrees_of_freedom: int = 0
        self.num_root_bodies: int = 0
        self.num_visual_geoms: int = 0
        self.num_collider_geoms: int = 0
        self.num_collider_geoms_mjt_structural: int = 0
        self.num_collider_geoms_mjt_dynamic: int = 0
        self.num_collider_geoms_mjt_articulable_dynamic: int = 0

        self.simtime: float = 0.0
        self.n_steps_per_second: int = 0
        self.real_time_factor: float = 0.0
        self.step_time: float = 0.0
        self.n_contacts_per_step: float = 0.0
        self.n_constraints_per_step: float = 0.0

        self.n_contacts_per_step_arr: list[int] = []
        self.n_constraints_per_step_arr: list[int] = []

        # Data obtained from the internal MuJoCo timers on the C++ side
        self.mjc_duration: float = 0.0
        self.mjc_steptime: float = 0.0
        self.mjc_timers: dict[mj.mjtTimer, MjTimerInfo] = {
            timer_id: MjTimerInfo() for timer_id in MUJOCO_TIMERS_IDS
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(
            # Properties that keep the same among all runs
            num_bodies=self.num_bodies,
            num_geoms=self.num_geoms,
            num_joints=self.num_joints,
            num_degrees_of_freedom=self.num_degrees_of_freedom,
            num_root_bodies=self.num_root_bodies,
            num_visual_geoms=self.num_visual_geoms,
            num_collider_geoms=self.num_collider_geoms,
            num_collider_geoms_mjt_structural=self.num_collider_geoms_mjt_structural,
            num_collider_geoms_mjt_dynamic=self.num_collider_geoms_mjt_dynamic,
            num_collider_geoms_mjt_articulable_dynamic=self.num_collider_geoms_mjt_articulable_dynamic,
            # Properties that change among all runs
            simtime=self.simtime,
            n_steps_per_second=self.n_steps_per_second,
            real_time_factor=self.real_time_factor,
            step_time=self.step_time,
            n_contacts_per_step=int(self.n_contacts_per_step),
            n_constraints_per_step=int(self.n_constraints_per_step),
            mjc_duration=self.mjc_duration,
            mjc_steptime=self.mjc_steptime,
            mjc_timers={
                mj.mjTIMERSTRING[timer_id]: self.mjc_timers[timer_id].to_dict()
                for timer_id in self.mjc_timers
            },
        )

    def from_dict(self, data: dict[str, Any]) -> None:
        self.num_bodies = data.get("num_bodies", 0)
        self.num_geoms = data.get("num_geoms", 0)
        self.num_joints = data.get("num_joints", 0)
        self.num_degrees_of_freedom = data.get("num_degrees_of_freedom", 0)
        self.num_root_bodies = data.get("num_root_bodies", 0)
        self.num_visual_geoms = data.get("num_visual_geoms", 0)
        self.num_collider_geoms = data.get("num_collider_geoms", 0)
        self.num_collider_geoms_mjt_structural = data.get("num_collider_geoms_mjt_structural", 0)
        self.num_collider_geoms_mjt_dynamic = data.get("num_collider_geoms_mjt_dynamic", 0)
        self.num_collider_geoms_mjt_articulable_dynamic = data.get(
            "num_collider_geoms_mjt_articulable_dynamic", 0
        )

        self.simtime = data.get("simtime", 0.0)
        self.n_steps_per_second = data.get("n_steps_per_second", 0)
        self.real_time_factor = data.get("real_time_factor", 0.0)
        self.step_time = data.get("step_time", 0.0)
        self.n_contacts_per_step = data.get("n_contacts_per_step", 0.0)
        self.n_constraints_per_step = data.get("n_constraints_per_step", 0.0)

        self.mjc_duration = data.get("mjc_duration", 0.0)
        self.mjc_steptime = data.get("mjc_steptime", 0.0)

        engine_timers_dict = data.get("mjc_timers", {})
        for timer_str in engine_timers_dict:
            duration = float(engine_timers_dict[timer_str].get("duration", 0.0))
            number = int(engine_timers_dict[timer_str].get("number", 0))
            r_time = duration / number if number > 0 else 0.0
            timer_id = MUJOCO_TIMERS_IDS_MAP[timer_str]
            self.mjc_timers[timer_id] = MjTimerInfo(duration, number, r_time)


def show_results_summary_from_cache(res: ResultsCache) -> None:
    print("--------------------------- Scene info ---------------------------")
    print(f"Number of bodies                    : {res.num_bodies}")
    print(f"Number of geoms                     : {res.num_geoms}")
    print(f"Number of joints                    : {res.num_joints}")
    print(f"Number of dof                       : {res.num_degrees_of_freedom}")
    print(f"Number of root bodies               : {res.num_root_bodies}")
    print(f"Number of visual geoms              : {res.num_visual_geoms}")
    print(f"Number of collider geoms            : {res.num_collider_geoms}")
    print(f"Number of MJT_STRUCTURAL            : {res.num_collider_geoms_mjt_structural}")
    print(f"Number of MJT_DYNAMIC               : {res.num_collider_geoms_mjt_dynamic}")
    print(f"Number of MJT_ARTICULABLE_DYNAMIC   : {res.num_collider_geoms_mjt_articulable_dynamic}")

    print("-------------------------- Runtime info --------------------------")
    print(f"Simulation time             : {res.simtime} s")
    print(f"Steps per second            : {res.n_steps_per_second}")
    print(f"Realtime factor             : {res.real_time_factor}")
    print(f"Time per step               : {res.mjc_steptime} ms")

    print(f"Contacts / step             : {res.n_contacts_per_step}")
    print(f"Constraints / step          : {res.n_constraints_per_step}")

    def timer_to_str(timer_id: mj.mjtTimer) -> str:
        res_duration = res.mjc_timers[timer_id].duration
        res_number = res.mjc_timers[timer_id].number
        res_time = res_duration / res_number if res_number > 0 else 0.0
        res_total = res.mjc_steptime
        res_percent = 100.0 * res_time / res_total
        return f"{1e3 * res_time:3.6f} ({res_percent:.2f} %)"

    print("Internal profiler, ms per step")
    print(f"                step : {timer_to_str(mj.mjtTimer.mjTIMER_STEP)}")
    print(f"             forward : {timer_to_str(mj.mjtTimer.mjTIMER_FORWARD)}")
    print(f"            position : {timer_to_str(mj.mjtTimer.mjTIMER_POSITION)}")
    print(f"            velocity : {timer_to_str(mj.mjtTimer.mjTIMER_VELOCITY)}")
    print(f"          constraint : {timer_to_str(mj.mjtTimer.mjTIMER_CONSTRAINT)}")
    print(f"             advance : {timer_to_str(mj.mjtTimer.mjTIMER_ADVANCE)}")
    print("")
    print(f"   position total    : {timer_to_str(mj.mjtTimer.mjTIMER_POSITION)}")
    print(f"     kinematics      : {timer_to_str(mj.mjtTimer.mjTIMER_POS_KINEMATICS)}")
    print(f"     inertia         : {timer_to_str(mj.mjtTimer.mjTIMER_POS_INERTIA)}")
    print(f"     collision       : {timer_to_str(mj.mjtTimer.mjTIMER_POS_COLLISION)}")
    print(f"        broadphase   : {timer_to_str(mj.mjtTimer.mjTIMER_COL_BROAD)}")
    print(f"        narrowphase  : {timer_to_str(mj.mjtTimer.mjTIMER_COL_NARROW)}")
    print(f"     make            : {timer_to_str(mj.mjtTimer.mjTIMER_POS_MAKE)}")
    print(f"     project         : {timer_to_str(mj.mjtTimer.mjTIMER_POS_PROJECT)}")


def show_results_summary_from_file(house_path: Path, settings: TestSettings) -> None:
    if not settings.results_path.exists():
        return

    results_info = dict(results=dict(), warnings=dict(), settings=dict())
    with open(settings.results_path, "r") as fhandle:
        results_info = json.load(fhandle)

    if house_path.stem not in results_info["results"]:
        return

    res = ResultsCache()
    res.from_dict(results_info["results"][house_path.stem])
    show_results_summary_from_cache(res)


def get_num_visual_geoms(spec: mj.MjSpec) -> int:
    count = 0
    for geom in spec.geoms:
        if geom.classname.name == "__VISUAL_MJT__":
            count += 1
            continue
        elif geom.contype == 0 and geom.conaffinity == 0:
            count += 1
            continue
    return count


def get_num_collider_geoms(spec: mj.MjSpec) -> dict[str, int]:
    counts = {"all": 0, "mjt_structural": 0, "mjt_dynamic": 0, "mjt_articulable_dynamic": 0}
    for geom in spec.geoms:
        if geom.classname.name == "__VISUAL_MJT__":
            continue
        if geom.contype == 0 and geom.conaffinity == 0:
            continue
        counts["all"] += 1
        if geom.classname.name == "__STRUCTURAL_MJT__":
            counts["mjt_structural"] += 1
        elif geom.classname.name == "__DYNAMIC_MJT__":
            counts["mjt_dynamic"] += 1
        elif geom.classname.name == "__ARTICULABLE_DYNAMIC_MJT__":
            counts["mjt_articulable_dynamic"] += 1
    return counts


def get_num_root_bodies_from_spec(spec: mj.MjSpec) -> int:
    count = 0
    body = spec.worldbody.first_body()
    while body is not None:
        count += 1
        body = spec.worldbody.next_body(body)
    return count


def run_performance_test_single(house_path: Path, settings: TestSettings) -> ResultsCache:

    spec = mj.MjSpec.from_file(house_path.as_posix())
    spec.option.timestep = settings.timestep
    if HAS_SLEEP_ISLAND_SUPPORT and settings.use_sleep_island:
        spec.option.enableflags |= mj.mjtEnableBit.mjENBL_SLEEP

    model: mj.MjModel = spec.compile()
    data: mj.MjData = mj.MjData(model)

    mj.mj_resetData(model, data)

    results = ResultsCache()
    results.house_name = house_path.stem
    results.num_bodies = model.nbody
    results.num_geoms = model.ngeom
    results.num_joints = model.njnt
    results.num_degrees_of_freedom = model.nv
    results.num_root_bodies = get_num_root_bodies_from_spec(spec)

    results.num_visual_geoms = get_num_visual_geoms(spec)
    colliders_geoms_counts = get_num_collider_geoms(spec)
    results.num_collider_geoms = colliders_geoms_counts["all"]
    results.num_collider_geoms_mjt_structural = colliders_geoms_counts["mjt_structural"]
    results.num_collider_geoms_mjt_dynamic = colliders_geoms_counts["mjt_dynamic"]
    results.num_collider_geoms_mjt_articulable_dynamic = colliders_geoms_counts[
        "mjt_articulable_dynamic"
    ]

    warnings = []
    for _ in range(settings.test_nsteps):
        mj.mj_step(model, data)
        warnings.extend(collect_warnings(data))

        results.n_contacts_per_step_arr.append(data.ncon)
        results.n_constraints_per_step_arr.append(data.nefc)

    mjc_duration = data.timer[mj.mjtTimer.mjTIMER_STEP].duration
    mjc_steptime = mjc_duration / data.timer[mj.mjtTimer.mjTIMER_STEP].number

    results.simtime = mjc_duration
    results.step_time = mjc_steptime
    results.n_steps_per_second = int(settings.test_nsteps / results.simtime)
    results.real_time_factor = settings.test_nsteps * settings.timestep / results.simtime
    results.n_contacts_per_step = sum(results.n_contacts_per_step_arr) / len(
        results.n_contacts_per_step_arr
    )
    results.n_constraints_per_step = sum(results.n_constraints_per_step_arr) / len(
        results.n_constraints_per_step_arr
    )

    results.mjc_duration = mjc_duration
    results.mjc_steptime = mjc_steptime

    for timer_id in MUJOCO_TIMERS_IDS:
        if data.timer[timer_id].number > 0:
            results.mjc_timers[timer_id].duration = data.timer[timer_id].duration
            results.mjc_timers[timer_id].number = data.timer[timer_id].number
            results.mjc_timers[timer_id].r_time = (
                data.timer[timer_id].duration / data.timer[timer_id].number
            )

    return results


def run_performance_test(house_path: Path, settings: TestSettings, lock: LockType | None) -> ResultsCache | None:
    results: ResultsCache | None = None
    try:
        # if True:
        results = run_performance_test_single(house_path, settings)

    except Exception as e:
        with lock if lock is not None else nullcontext():
            errors_dict = dict()
            if settings.errors_path.exists():
                with open(settings.errors_path, "r") as fhandle:
                    errors_dict = json.load(fhandle)
            errors_dict[house_path.stem] = f"Got an error while processing house: {e}"
            with open(settings.errors_path, "w") as fhandle:
                json.dump(errors_dict, fhandle, indent=4)

    return results


def compare_results(results_a: Path, results_b: Path) -> None:
    import matplotlib.pyplot as plt

    if not results_a.is_file() or not results_b.is_file():
        return

    stats_a = get_stats_runtime_test(results_a)
    stats_b = get_stats_runtime_test(results_b)

    rtfactors_a, rtfactors_b, rtfactors_rate = [], [], []
    max_name, max_ratio = "", 0.0
    for scene_name in stats_a:
        assert scene_name in stats_b
        rtf_a, rtf_b = stats_a[scene_name].realtime_factor, stats_b[scene_name].realtime_factor
        rtfactors_a.append(rtf_a)
        rtfactors_b.append(rtf_b)
        rtf_ratio = rtf_a / rtf_b
        if rtf_ratio < 1.0:
            print(f"wtf? scene: {scene_name}, ratio: {rtf_ratio}")
        rtfactors_rate.append(rtf_ratio)

        if rtf_ratio > max_ratio:
            max_ratio, max_name = rtf_ratio, scene_name

    print(f"max_name: {max_name} max-ratio: {max_ratio}")

    np_rtfactors_a = np.array(rtfactors_a)
    np_rtfactors_b = np.array(rtfactors_b)
    np_rtfactors_rate = np.array(rtfactors_rate)

    fig, ax = plt.subplots()

    ax.grid(True)
    ax.boxplot([np_rtfactors_a, np_rtfactors_b], tick_labels=["3.3.8", "3.3.2"])
    ax.set_title("Real time factors comparison (decimation vs no-decimation)")
    fig.savefig("realtime_factors_comparison_boxplot.png")

    fig, ax = plt.subplots()

    ax.grid(True)
    ax.boxplot([np_rtfactors_rate], tick_labels=["Ratio 3.3.8 / 3.3.2"])
    ax.set_title("Ratio of real time factors 3.3.8 vs 3.3.2")
    fig.savefig("realtime_factors_comparison_ratios.png")


def save_results(results_scene: ResultsCache, settings: TestSettings) -> None:
    results_all = dict(results=dict(), warnings=dict(), settings=dict())
    if settings.results_path.is_file():
        with open(settings.results_path, "r") as fhandle:
            results_all = json.load(fhandle)
    results_all["settings"] = settings.to_dict()
    results_all["results"][results_scene.house_name] = results_scene.to_dict()
    with open(settings.results_path, "w") as fhandle:
        json.dump(results_all, fhandle, indent=4)


def main() -> int:
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
        "--model",
        type=str,
        default="",
        help="The specific model to use for the test",
    )
    parser.add_argument(
        "--houses-folder",
        type=str,
        default="",
        help="The path to the folder that contains the houses to use for the test",
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
        "--timestep",
        type=float,
        default=DEFAULT_TIMESTEP,
        help="The timestep used for the simulations on the test",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=DEFAULT_TEST_STEPS,
        help="The number of steps for a single run of the test",
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
        help="The max number of parallel processes to use for the test",
    )
    parser.add_argument(
        "--copy-results-to",
        type=str,
        default="",
        help="The path where to copy the results json file, if required",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Whether or not to show only the summary of the last run for a given house",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Whether to compare two runs of the test",
    )
    parser.add_argument(
        "--compare-first",
        type=str,
        default="",
        help="First file to compare",
    )
    parser.add_argument(
        "--compare-second",
        type=str,
        default="",
        help="First file to compare",
    )
    parser.add_argument(
        "--use-sleep-island",
        action="store_true",
        help="Whether or not to enable sleeping island (requires 3.3.8)",
    )

    args = parser.parse_args()

    settings = TestSettings(
        dataset=args.dataset,
        split=args.split,
        identifier=args.identifier,
        houses_folder=get_houses_folder(args),
        results_path=ROOT_DIR
        / Path(
            RESULTS_FILEPATH.format(
                dataset=args.dataset, split=args.split, identifier=args.identifier
            )
        ),
        warnings_path=ROOT_DIR
        / Path(
            WARNINGS_FILEPATH.format(
                dataset=args.dataset, split=args.split, identifier=args.identifier
            )
        ),
        errors_path=ROOT_DIR
        / Path(
            ERRORS_FILEPATH.format(dataset=args.dataset, split=args.split, identifier=args.identifier)
        ),
        copy_results_to=args.copy_results_to,
        timestep=args.timestep,
        test_nsteps=args.nstep,
        test_time=args.nstep * args.timestep,
        n_test_runs=DEFAULT_TEST_RUNS,
        use_sleep_island=args.use_sleep_island,
    )

    if args.compare:
        if args.compare_first != "" and args.compare_second != "":
            filepath_first = Path(args.compare_first)
            filepath_second = Path(args.compare_second)
            compare_results(filepath_first, filepath_second)
        return 0

    if args.model != "":
        model_path = Path(args.model)
        if not model_path.is_file():
            print(f"[ERROR]: given model @ {model_path.as_posix()} is not a valid file")
            return 1
        if args.summary:
            show_results_summary_from_file(model_path, settings)
        else:
            results_scene = run_performance_test(model_path, settings, None)
            if results_scene:
                show_results_summary_from_cache(results_scene)
                save_results(results_scene, settings)
    else:
        xmls_to_test = []
        if settings.dataset in {"procthor-10k", "procthor-objaverse", "holodeck-objaverse"}:
            pattern = settings.split + r"_(\d+)"
            candidates_xmls_to_test = settings.houses_folder.glob(f"{settings.split}_*.xml")
            for candidate_xml_test in candidates_xmls_to_test:
                index = int(re.search(pattern, candidate_xml_test.stem).group(1))  # type: ignore
                if index < args.start or index >= args.end:
                    continue
                xmls_to_test.append(candidate_xml_test)
        elif settings.dataset == "ithor":
            pattern = r"FloorPlan(\d+)_physics.xml"
            candidates_xmls_to_test = settings.houses_folder.glob("*_physics.xml")
            for candidate_xml_test in candidates_xmls_to_test:
                index = int(re.search(pattern, candidate_xml_test.name).group(1))  # type: ignore
                if index < args.start or index >= args.end:
                    continue
                xmls_to_test.append(candidate_xml_test)
        else:
            print(
                f"[ERROR]: Should give --dataset with either 'procthor' or 'ithor', but got {settings.dataset}"
            )
            return 1

        if args.max_workers > 1:
            with tqdm(total=len(xmls_to_test)) as progressbar, Manager() as manager:
                m_lock: LockType = cast(LockType, manager.Lock())
                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = [
                        executor.submit(run_performance_test, fpath, settings, m_lock)
                        for fpath in xmls_to_test
                    ]
                    for future in as_completed(futures):
                        results_scene = future.result()
                        if results_scene:
                            save_results(results_scene, settings)
                        progressbar.update(1)

        else:
            for house_xml in tqdm(xmls_to_test):
                results_scene = run_performance_test(house_xml, settings, None)
                if results_scene:
                    save_results(results_scene, settings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
