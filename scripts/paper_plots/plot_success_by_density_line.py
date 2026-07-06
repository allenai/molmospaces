"""Line chart: policy success rate vs. nearby graspable count.

Alternative rendering of ``plot_success_by_density.py`` — same data, axis
titles, text size, and color scheme, but drawn as one line per policy
instead of grouped bars.
"""

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

x_labels = ["0", "1", "2", "3", "4", r"$\geq 5$"]
n_groups = len(x_labels)
x = np.arange(n_groups)

fig, ax = plt.subplots(figsize=(63.0, 45.0))

for name, vals, color in policies:
    ax.plot(
        x,
        vals,
        label=name,
        color=color,
        linewidth=14.0,
        marker="o",
        markersize=60.0,
        markeredgecolor="black",
        markeredgewidth=4.0,
    )

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=180)
ax.set_xlabel(r"Nearby graspable object count (radius $0.30$ m)")
ax.set_ylabel("Success rate")
ax.set_title(
    "Pick with Occlusion Task: Policy Success\nRates Decline with Increasing Clutter",
    pad=120,
)
ax.set_ylim(0, max(max(v) for _, v, _ in policies) * 1.25)
ax.legend(
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.55),
)

fig.tight_layout()
fig.subplots_adjust(left=0.08, right=0.995, top=0.887, bottom=0.253)
ax.yaxis.labelpad = 63
ax.xaxis.labelpad = 72
out_path = "success_by_nearby_graspable_line.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
fig.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.5)
print(f"Saved: {out_path}")
