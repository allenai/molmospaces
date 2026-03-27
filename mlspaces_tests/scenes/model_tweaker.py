import argparse
import json
from pathlib import Path

import mujoco as mj
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
THOR_ASSETS_DIR = ROOT_DIR / "assets" / "objects" / "thor"


def tweak_dressers_apply_offset() -> None:
    TWEAK_DRESSERS_OFFSET_Z_PLUS = 0.018

    candidates_xmls = THOR_ASSETS_DIR.rglob("*.xml")
    dressers_xmls = []
    for candidate_xml in candidates_xmls:
        if "Dresser_" in candidate_xml.stem:
            dressers_xmls.append(candidate_xml)

    for dresser_xml in tqdm(dressers_xmls):
        spec = mj.MjSpec.from_file(dresser_xml.as_posix())
        # NOTE(wilbert): here we're assumming that the body on which to apply the offset is the
        # first body child of the root body. It seems that thor assets follow this pattern
        root_body = spec.worldbody.first_body()
        target_body = root_body.first_body()
        target_body.pos[2] += TWEAK_DRESSERS_OFFSET_Z_PLUS

        model: mj.MjModel | None = None
        try:
            model = spec.compile()
        except Exception:
            print(f"There was an error tweaking dresser : {dresser_xml.stem}")

        if model is not None:
            with open(dresser_xml, "w") as fhandle:
                fhandle.write(spec.to_xml())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tweak-dressers-apply-offset",
        action="store_true",
        help="Whether or not to tweak the dressers by moving 0.05 in the +z direction",
    )

    args = parser.parse_args()

    status_file = THOR_ASSETS_DIR / "tweaks_status.json"
    if not status_file.exists():
        with open(status_file, "w") as fhandle:
            json.dump({"tweaks": {"apply_dressers_offset": False}}, fhandle, indent=4)

    with open(status_file, "r") as fhandle:
        status_data = json.load(fhandle)

    if args.tweak_dressers_apply_offset:
        if not status_data["tweaks"].get("apply_dressers_offset", False):
            tweak_dressers_apply_offset()

            status_data["tweaks"]["apply_dressers_offset"] = True
            with open(status_file, "w") as fhandle:
                json.dump(status_data, fhandle, indent=4)
            print("Finished applying offset tweak to all dressers")
        else:
            print("Dressers offset tweak already applied to thor assets")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
