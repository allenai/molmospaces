"""Failure-mode frequency on the pick-with-occlusion task, split by nearby-graspable density.

Buckets each common episode by `compute_num_nearby_graspable(radius=0.30 m)` from
its trajectories H5 (sourced from create_json_benchmark.py via the same code path as
analyze_outcomes_by_density.py). Produces a 2-subplot figure: one bucket per panel,
3 policy bars per failure-mode class.
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
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

RADIUS_M = 0.30
DENSITY_THRESHOLD = 2  # split: <2 vs >=2

ANN_FILES = [
    (r"$\pi_{0.5}$",
     "/weka/prior/aguru/molmo-spaces/eval_output/occlusion/pi05/20260510_031126/human_annotations.jsonl",
     "#ECE133"),
    ("DreamZero",
     "/weka/prior/aguru/molmo-spaces/eval_output/occlusion/dreamzero/20260506_215036/human_annotations.jsonl",
     "#029E73"),
    ("MolmoAct2",
     "/weka/prior/aguru/molmo-spaces/eval_output/occlusion/molmoact2/20260510_031126/human_annotations.jsonl",
     "#CC78BC"),
]

PATH_REMAPS = [
    ("/Users/arjunguru/weka/molmo-spaces", "/weka/prior/aguru/molmo-spaces"),
]

OUTCOME_CLASSES = [
    "grasp_right_object_success",
    "no_grasp_attempt",
    "grasp_wrong_object",
    "grasp_right_object_fail_clutter",
    "grasp_right_object_fail_idiosyncratic",
]
CLASS_LABELS = {
    "grasp_right_object_success":             "Success",
    "no_grasp_attempt":                       "No grasp\nattempt",
    "grasp_wrong_object":                     "Grasped\nwrong object",
    "grasp_right_object_fail_clutter":        "Moved toward right object,\ncollision with clutter",
    "grasp_right_object_fail_idiosyncratic":  "Right object,\ngrasping failure",
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

# Compute density from one policy's H5s (any of them — same houses/episodes).
density = density_per_episode(all_rows[0][1])
keys_with_density = {k for k in common_keys if density.get(k) is not None}
print(f"Episodes with computable density: {len(keys_with_density)}")

low_keys = {k for k in keys_with_density if density[k] < DENSITY_THRESHOLD}
high_keys = {k for k in keys_with_density if density[k] >= DENSITY_THRESHOLD}
print(f"  < {DENSITY_THRESHOLD}: n={len(low_keys)}    >= {DENSITY_THRESHOLD}: n={len(high_keys)}")


def percents_for_bucket(keys: set) -> list[tuple[str, str, list[float]]]:
    """Return [(policy_name, color, [pct_per_class])] for the given key set."""
    out = []
    for name, mp, color in ann_maps:
        counts = collections.Counter(mp[k] for k in keys)
        n = max(len(keys), 1)
        pct = [100.0 * counts.get(c, 0) / n for c in OUTCOME_CLASSES]
        out.append((name, color, pct))
    return out


n_groups = len(OUTCOME_CLASSES)
n_policies = len(ann_maps)
x = np.arange(n_groups)
bar_width = 0.26


def draw_panel(ax, keys: set, title: str):
    if not keys:
        ax.set_title(f"{title} (empty)")
        return
    rows = percents_for_bucket(keys)
    ymax = 0.0
    for i, (name, color, pct) in enumerate(rows):
        offset = (i - (n_policies - 1) / 2) * bar_width
        ymax = max(ymax, max(pct))
        ax.bar(
            x + offset,
            pct,
            width=bar_width,
            label=name,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in OUTCOME_CLASSES], fontsize= 20)
    ax.set_ylabel(r"\% of episodes in bucket")
    ax.set_title(title)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_ylim(0, 60)


fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), sharey=True)
draw_panel(
    axes[0], low_keys,
    rf"Low density: $<{DENSITY_THRESHOLD}$ nearby graspable ($n={len(low_keys)}$)",
)
draw_panel(
    axes[1], high_keys,
    rf"High density: $\geq {DENSITY_THRESHOLD}$ nearby graspable ($n={len(high_keys)}$)",
)
axes[1].set_ylabel("")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    frameon=False,
    ncol=n_policies,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.10),
)
fig.subplots_adjust(bottom=0.28)
fig.suptitle(
    rf"Failure-Mode Frequency by Nearby-Graspable Density "
    rf"(radius ${RADIUS_M:.2f}$ m)"
)

fig.tight_layout()
for ax in axes:
    ax.set_ylim(0, 60)  # re-assert after tight_layout
out = "failure_modes_occlusion_by_density.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
