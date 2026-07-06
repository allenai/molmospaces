"""Delta failure-mode frequency: cluttered minus sparse, single panel.

Same data path as plot_occlusion_failure_modes_by_density.py, but instead of
side-by-side panels per density bucket, plots bars of
(cluttered_pct - sparse_pct) per failure-mode class, per policy.
"""

import collections
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

# Reuse benchmark helpers.
sys.path.insert(0, str(Path("/weka/prior/aguru/molmo-spaces/scripts/benchmarks").resolve()))
from create_json_benchmark import (
    compute_num_nearby_graspable,
    extract_frozen_config,
    parse_obs_scene,
)

plt.style.use(["science", "grid"])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
    "font.size": 200,
    "axes.titlesize": 200,
    "axes.labelsize": 200,
    "xtick.labelsize": 200,
    "ytick.labelsize": 200,
    "legend.fontsize": 200,
    # Scale axis spines, tick marks, and grid lines to stay visible against
    # the larger font size (matches the paper's other plots).
    "axes.linewidth": 6.3,
    "xtick.major.size": 36.0,
    "xtick.major.width": 6.3,
    "ytick.major.size": 36.0,
    "ytick.major.width": 6.3,
    "xtick.minor.size": 20.25,
    "xtick.minor.width": 4.5,
    "ytick.minor.size": 20.25,
    "ytick.minor.width": 4.5,
    "grid.linewidth": 3.6,
})

RADIUS_M = 0.30
DENSITY_THRESHOLD = 2  # split: <2 (sparse) vs >=2 (cluttered)

ANN_FILES = [
    (r"$\pi_{0.5}$",
     "/weka/prior/aguru/molmo-spaces/eval_output/occlusion/pi05/20260510_031126/human_annotations.jsonl",
     "#ECE133"),  # seaborn colorblind: yellow
    ("DreamZero",
     "/weka/prior/aguru/molmo-spaces/eval_output/occlusion/dreamzero/20260506_215036/human_annotations.jsonl",
     "#029E73"),  # seaborn colorblind: green
    ("MolmoAct2",
     "/weka/prior/aguru/molmo-spaces/eval_output/occlusion/molmoact2/20260510_031126/human_annotations.jsonl",
     "#CC78BC"),  # seaborn colorblind: pink
]

PATH_REMAPS = [
    ("/Users/arjunguru/weka/molmo-spaces", "/weka/prior/aguru/molmo-spaces"),
]

OUTCOME_CLASSES = [
    "no_grasp_attempt",
    "grasp_wrong_object",
    "grasp_right_object_fail_clutter",
    "grasp_right_object_fail_idiosyncratic",
    "grasp_right_object_success",
]
CLASS_LABELS = {
    "grasp_right_object_success":             "Success",
    "no_grasp_attempt":                       "No grasp\nattempt",
    "grasp_wrong_object":                     "Wrong\nobject",
    "grasp_right_object_fail_clutter":        "Clutter\ncollision",
    "grasp_right_object_fail_idiosyncratic":  "Grasping\nfailure",
}


def remap(video_path: str) -> str:
    for old, new in PATH_REMAPS:
        if video_path.startswith(old):
            return new + video_path[len(old):]
    return video_path


def locate_h5(video_path: str) -> Path | None:
    p = Path(video_path)
    m = re.search(r"_(batch_\d+_of_\d+)\.mp4$", p.name)
    return (p.parent / f"trajectories_{m.group(1)}.h5") if m else None


def episode_to_traj_key(episode_str: str) -> str:
    return f"traj_{int(episode_str.replace('episode_', ''))}"


def load_annotations(path: str) -> list[dict]:
    rows = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def density_per_episode(rows: list[dict]) -> dict:
    """Return {(house, episode) -> num_nearby_graspable or None}. Caches H5 opens."""
    cache: dict[Path, h5py.File] = {}
    out: dict = {}
    try:
        for rec in rows:
            key = (rec["house"], rec["episode"])
            if key in out:
                continue
            video = remap(rec["video_path"])
            h5_path = locate_h5(video)
            if h5_path is None or not h5_path.exists():
                out[key] = None
                continue
            if h5_path not in cache:
                try:
                    cache[h5_path] = h5py.File(h5_path, "r")
                except OSError:
                    out[key] = None
                    continue
            f5 = cache[h5_path]
            tk = episode_to_traj_key(rec["episode"])
            if tk not in f5:
                out[key] = None
                continue
            try:
                obs_scene = parse_obs_scene(f5[tk]["obs_scene"][()])
                fc = extract_frozen_config(obs_scene)
                out[key] = compute_num_nearby_graspable(fc, RADIUS_M)
            except Exception:
                out[key] = None
    finally:
        for f5 in cache.values():
            try:
                f5.close()
            except Exception:
                pass
    return out


# Load annotations from all 3 policies.
all_rows = [(name, load_annotations(path), color) for name, path, color in ANN_FILES]
ann_maps = [
    (name, {(r["house"], r["episode"]): r["annotation"] for r in rows}, color)
    for name, rows, color in all_rows
]
common_keys = set.intersection(*(set(m.keys()) for _, m, _ in ann_maps))
print(f"Common (house, episode) pairs: {len(common_keys)}")

density = density_per_episode(all_rows[0][1])
keys_with_density = {k for k in common_keys if density.get(k) is not None}
print(f"Episodes with computable density: {len(keys_with_density)}")

sparse_keys = {k for k in keys_with_density if density[k] < DENSITY_THRESHOLD}
cluttered_keys = {k for k in keys_with_density if density[k] >= DENSITY_THRESHOLD}
print(f"  sparse (<{DENSITY_THRESHOLD}): n={len(sparse_keys)}    "
      f"cluttered (>={DENSITY_THRESHOLD}): n={len(cluttered_keys)}")


def percents_for_bucket(keys: set) -> list[tuple[str, str, list[float]]]:
    out = []
    for name, mp, color in ann_maps:
        counts = collections.Counter(mp[k] for k in keys)
        n = max(len(keys), 1)
        pct = [100.0 * counts.get(c, 0) / n for c in OUTCOME_CLASSES]
        out.append((name, color, pct))
    return out


sparse_rows = percents_for_bucket(sparse_keys)
cluttered_rows = percents_for_bucket(cluttered_keys)

# delta = cluttered - sparse, per policy per class
deltas = []
for (name, color, pct_sparse), (_, _, pct_clut) in zip(sparse_rows, cluttered_rows):
    delta = [c - s for c, s in zip(pct_clut, pct_sparse)]
    deltas.append((name, color, delta))

n_groups = len(OUTCOME_CLASSES)
n_policies = len(ann_maps)
x = np.arange(n_groups) * 1.6  # widen group spacing so xtick labels don't collide at large font size
bar_width = 0.36

fig, ax = plt.subplots(figsize=(70.0, 58.0))

for i, (name, color, delta) in enumerate(deltas):
    offset = (i - (n_policies - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        delta,
        width=bar_width,
        label=name,
        color=color,
        edgecolor="black",
        linewidth=0.8,
    )

ax.axhline(0, color="black", linewidth=4.5)
ax.set_xticks(x)
ax.set_xticklabels([CLASS_LABELS[c] for c in OUTCOME_CLASSES], fontsize=200)
ax.set_ylabel(
    "Failure Mode Frequency Difference\n"
    r"(Cluttered Scene Frequency $-$"
    "\n"
    r"Sparse Scene Frequency)"
)
ax.set_title("Effect of Scene Clutter on Failure-mode Frequency")

ax.legend(
    frameon=False,
    ncol=n_policies,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.30),
)

fig.tight_layout()
fig.subplots_adjust(left=0.10, right=0.995, top=0.92, bottom=0.22)
ax.yaxis.labelpad = 63
ax.xaxis.labelpad = 72
out = "failure_modes_occlusion_density_delta.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.5)
fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.5)
print(f"Saved: {out}")
