"""Bar chart: policy success rate vs. nearby graspable count."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the style)

plt.style.use(["science", "grid"])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
    "font.size": 180,
    "axes.titlesize": 180,
    "axes.labelsize": 180,
    "xtick.labelsize": 180,
    "ytick.labelsize": 180,
    "legend.fontsize": 180,
    # Scale axis spines, tick marks, and grid lines to stay visible against
    # the larger font size (4.5x the 40pt baseline).
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

# Raw successes / trials per nearby-graspable bucket.
# NOTE: DreamZero[1] was provided as "19.91" — interpreted as 19/91.
tiptop = [37/90, 27/91, 28/91, 30/91, 23/91, 134/546]
pi05 = [20/90, 17/91, 12/91, 12/91, 11/91, 55/545]
dreamzero = [29/90, 19/91, 25/91, 20/91, 21/91, 66/546]
molmoact2 = [24/90, 23/91, 20/91, 14/91, 11/91, 56/546]

policies = [
    ("TiPToP",        tiptop,     "#0173B2"),  # sns colorblind: blue
    (r"$\pi_{0.5}$",  pi05,       "#ECE133"),
    ("DreamZero",     dreamzero,  "#029E73"),
    ("MolmoAct2",    molmoact2, "#CC78BC"),
]

# Adjust these if the last bucket is something other than the aggregate.
x_labels = ["0", "1", "2", "3", "4", r"$\geq 5$"]
n_groups = len(x_labels)
n_policies = len(policies)

group_spacing = 7.0  # gap between group centers; widened to fit the 4th bar
                     # (TiPToP) plus its extra inter-bar gap.
x = np.arange(n_groups) * group_spacing
bar_width = 0.98
# Extra horizontal gap (in data units) between TiPToP (leftmost) and the rest
# of the bars within each group.
tiptop_extra_gap = 0.5

# Bar offsets within each group. TiPToP (i=0) sits to the left of the rest
# with an extra gap; pi05 / DreamZero / MolmoAct stay evenly spaced.
shift = (tiptop_extra_gap + bar_width) / 2
bar_offsets = [
    -1.5 * bar_width - 0.5 * tiptop_extra_gap,  # TiPToP
    -1.0 * bar_width + shift,                    # pi0.5
    0.0 * bar_width + shift,                     # DreamZero
    1.0 * bar_width + shift,                     # MolmoAct2
]

fig, ax = plt.subplots(figsize=(63.0, 45.0))

for i, (name, vals, color) in enumerate(policies):
    offset = bar_offsets[i]
    ax.bar(
        x + offset,
        vals,
        width=bar_width,
        label=name,
        color=color,
        edgecolor="black",
        linewidth=0.8,
    )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=180)
ax.set_xlabel(r"Nearby graspable object count (radius $0.30$ m)")
ax.set_ylabel("Success rate")
ax.set_title("Pick with Occlusion Task: Policy Success Rates Decline with Increasing Clutter")
ax.set_ylim(0, max(max(v) for _, v, _ in policies) * 1.25)
ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.55),
)

fig.tight_layout()
fig.subplots_adjust(left=0.08, right=0.995, top=0.887, bottom=0.253)
ax.set_title(
    "Pick with Occlusion Task: Policy Success\nRates Decline with Increasing Clutter",
    pad=120,
)
ax.yaxis.labelpad = 63
ax.xaxis.labelpad = 72
out_path = "success_by_nearby_graspable.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
fig.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.5)
print(f"Saved: {out_path}")
