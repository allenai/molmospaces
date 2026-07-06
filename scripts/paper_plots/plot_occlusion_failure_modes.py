"""Bar chart: failure-mode frequency on the pick-with-occlusion task,
grouped by failure-mode class with one bar per policy.

Restricted to the set of (house, episode) keys annotated in *all three* files.
"""

import collections
import json

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

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

# Outcome classes (success + failure modes).
FAILURE_CLASSES = [
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


def load_annotations(path: str) -> dict:
    """Return {(house, episode) -> annotation}. Later rows overwrite earlier ones."""
    out: dict = {}
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(d["house"], d["episode"])] = d["annotation"]
    return out


# Load each file and intersect keys.
annotations = [(name, load_annotations(path), color) for name, path, color in ANN_FILES]
common_keys = set.intersection(*(set(a.keys()) for _, a, _ in annotations))
n_common = len(common_keys)
print(f"Common (house, episode) pairs across all 3 policies: {n_common}")

per_policy = []
for name, ann, color in annotations:
    counts = collections.Counter(ann[k] for k in common_keys)
    pct = [100.0 * counts.get(c, 0) / n_common for c in FAILURE_CLASSES]
    per_policy.append((name, color, pct))
    print(f"  {name}: " + ", ".join(f"{c}={counts.get(c,0)}" for c in FAILURE_CLASSES))

n_groups = len(FAILURE_CLASSES)
n_policies = len(per_policy)
x = np.arange(n_groups)
bar_width = 0.26

fig, ax = plt.subplots(figsize=(8.6, 4.0))
ymax = 0.0
for i, (name, color, pct) in enumerate(per_policy):
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
ax.set_xticklabels([CLASS_LABELS[c] for c in FAILURE_CLASSES], fontsize= 20)
ax.set_xlabel("Failure mode")
ax.set_ylabel(r"\% of annotated episodes")
ax.set_title(
    "Failure-Mode Frequency on Pick-with-Occlusion Task "
    rf"($n={n_common}$ shared episodes)"
)
ax.set_yticks(np.arange(0, 51, 10))
ax.set_ylim(0, 50)
ax.legend(
    frameon=False,
    ncol=n_policies,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
)

fig.tight_layout()
ax.set_ylim(0, 50)  # re-assert after tight_layout
out = "failure_modes_occlusion.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
