"""Build the `g1_dex3` robot: the molmospaces G1 (g1_dex.xml) with a Unitree Dex3-1 hand on each wrist.

    python scripts/data/format_conversion/build_g1_dex3.py [--out_dir <ROBOTS_DIR>/g1_dex3]

The body, arms, cameras and walking-controller actuators are g1_dex.xml's, unchanged. The Dex1
parallel gripper on the right wrist is replaced, and a hand is added on the left (which had a
rubber dummy hand), using the Dex3-1 kinematics, inertials and meshes from MuJoCo Menagerie's
`unitree_g1/g1_with_hands.xml` (Unitree, BSD-3-Clause; downloaded on first use and its LICENSE
copied next to the meshes).

Per hand: 7 position-actuated joints (thumb_0..2, index_0..1, middle_0..1), a `{side}_grasp`
site at the extended index fingertip (z along the finger, y = palm normal), and a wrist camera
(`left_wrist_camera` mirrors the existing `right_wrist_camera`).
"""

import argparse
import copy
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.molmo_spaces_constants import ROBOTS_DIR

MENAGERIE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1"
HAND_LINKS = ["palm", "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
HAND_JOINTS = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
# xyaxes of the grasp site in the wrist-yaw frame: z = finger direction, y = palm normal
# (the side the fingers curl toward; the thumb side).
SITE_XYAXES = {"right": "0 0 -1 0 1 0", "left": "0 0 1 0 -1 0"}


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)


def find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=ROBOTS_DIR / "g1_dex3")
    ap.add_argument(
        "--src_dir", type=Path, default=ROBOTS_DIR / "g1", help="molmospaces g1 robot dir"
    )
    args = ap.parse_args()
    out, assets = args.out_dir, args.out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # ---- assets: the G1's own (symlinked), plus the hand meshes from Menagerie.
    for f in (args.src_dir / "assets").iterdir():
        link = assets / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    menagerie_xml = out / "_menagerie_g1_with_hands.xml"
    fetch(f"{MENAGERIE}/g1_with_hands.xml", menagerie_xml)
    fetch(f"{MENAGERIE}/LICENSE", assets / "LICENSE_unitree_menagerie")
    for side in ("left", "right"):
        for link in HAND_LINKS:
            fetch(
                f"{MENAGERIE}/assets/{side}_hand_{link}_link.STL",
                assets / f"{side}_hand_{link}_link.STL",
            )

    src = ET.parse(args.src_dir / "g1_dex.xml").getroot()
    hands = ET.parse(menagerie_xml).getroot()
    src.set("model", "g1_dex3")

    # ---- worldbody: drop the Dex1 gripper and the left dummy hand, graft the Dex3 hands.
    for side in ("left", "right"):
        wrist = find_body(src, f"{side}_wrist_yaw_link")
        for child in list(wrist):
            if child.tag == "body" and child.get("name") == "right_dex_base":
                wrist.remove(child)
            elif child.tag == "geom" and (
                child.get("name") == "offset" or child.get("mesh") == "left_rubber_hand"
            ):
                wrist.remove(child)
            elif child.tag == "site" and child.get("name") == f"{side}_grasp":
                wrist.remove(child)
        src_wrist = find_body(hands, f"{side}_wrist_yaw_link")
        for child in src_wrist:
            # the palm geoms and the three finger-root bodies; the wrist's own geoms/joint stay.
            if child.tag == "body" or (child.tag == "geom" and "hand" in (child.get("mesh") or "")):
                wrist.append(copy.deepcopy(child))
        for joint in wrist.iter("joint"):
            if "_hand_" in joint.get("name", ""):
                joint.set("armature", "0.002")
                joint.set("frictionloss", "0.005")
                joint.set("damping", "0.02")
        # Wrist camera: the right one exists; the left mirrors it (it sits ~on the wrist axis).
        if side == "left":
            for geom in list(find_body(src, "right_wrist_yaw_link").iter("geom")):
                if geom.get("name") in ("camera_box", "camera"):
                    wrist.append(copy.deepcopy(geom))
                    wrist[-1].set("name", f"left_{geom.get('name')}")
            cam = copy.deepcopy(
                next(c for c in find_body(src, "right_wrist_yaw_link").iter("camera"))
            )
            cam.set("name", "left_wrist_camera")
            wrist.append(cam)

    # ---- assets: hand meshes in, Dex1 meshes out.
    asset = src.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", "").startswith("dex_") or mesh.get("name") == "left_rubber_hand":
            asset.remove(mesh)
    for side in ("left", "right"):
        for link in HAND_LINKS:
            ET.SubElement(
                asset, "mesh",
                {"content_type": "model/stl", "file": f"{side}_hand_{link}_link.STL"},
            )  # fmt: skip

    # ---- Dex1 tendon / equality / contact excludes / actuator / keyframes go.
    for tag in ("tendon", "equality", "keyframe"):
        for el in src.findall(tag):
            src.remove(el)
    size = src.find("size")
    if size is not None:
        size.attrib.pop("nkey", None)
    contact = src.find("contact")
    for ex in list(contact):
        if any(
            "dex" in (ex.get(k) or "") or "Link" in (ex.get(k) or "") for k in ("body1", "body2")
        ):
            contact.remove(ex)
    actuator = src.find("actuator")
    for act in list(actuator):
        if act.get("name") == "right_grip":
            actuator.remove(act)
    hand_ranges = {}
    for joint in hands.iter("joint"):
        if "_hand_" in joint.get("name", ""):
            hand_ranges[joint.get("name")] = joint.get("range")
    for side in ("left", "right"):
        for j in HAND_JOINTS:
            name = f"{side}_hand_{j}_joint"
            lo, hi = hand_ranges[name].split()
            ET.SubElement(
                actuator, "position",
                {"name": name, "joint": name, "group": "2", "kp": "3", "kv": "0.1",
                 "ctrlrange": f"{lo} {hi}", "forcelimited": "true", "forcerange": "-1.4 1.4"},
            )  # fmt: skip

    # ---- grasp sites at the extended index fingertip (needs the compiled model).
    for side in ("left", "right"):
        wrist = find_body(src, f"{side}_wrist_yaw_link")
        ET.SubElement(
            wrist, "site",
            {"name": f"{side}_grasp", "pos": "0.2 0 0.0285", "size": "0.005", "group": "5",
             "xyaxes": SITE_XYAXES[side]},
        )  # fmt: skip
    path = out / "g1_dex3.xml"
    ET.indent(src)
    ET.ElementTree(src).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_index_1_link")
        geom = next(
            g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_type[g] == 7
        )
        adr, n = (
            model.mesh_vertadr[model.geom_dataid[geom]],
            model.mesh_vertnum[model.geom_dataid[geom]],
        )
        verts = model.mesh_vert[adr : adr + n]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        # fingertip: farthest vertex of the distal link along the finger (+x of the wrist frame)
        world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - data.xpos[wrist_id]) @ data.xmat[wrist_id].reshape(3, 3)
        tip = local[np.argmax(local[:, 0])]
        for site in find_body(src, f"{side}_wrist_yaw_link").iter("site"):
            if site.get("name") == f"{side}_grasp":
                site.set("pos", " ".join(f"{v:.4f}" for v in tip))
        print(f"{side} index fingertip in wrist frame: {tip}")
    ET.ElementTree(src).write(path)
    menagerie_xml.unlink()
    m2 = mujoco.MjModel.from_xml_path(str(path))
    print(f"wrote {path}: nq={m2.nq} nu={m2.nu} nbody={m2.nbody}")


if __name__ == "__main__":
    main()
