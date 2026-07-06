"""Bar chart: policy plain success rate vs. semantic prompt level."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

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

# Per-policy data across prompt levels (1, 2, 3).
# Plain: (mean, +/- error). Conditional: (mean, ci_low, ci_high).
# Lift: (mean, ci_low, ci_high) from the base pick task ("Pick up the {object}.").
pi05 = {
    "plain":       [(17.20, 3.56), (19.40, 3.70), (16.80, 3.53)],
    "conditional": [(49.14, 41.83, 56.50), (54.49, 47.15, 61.64), (51.22, 43.62, 58.76)],
    "lift":        (35.00, 30.95, 39.28),
}
dreamzero = {
    "plain":       [(23.60, 3.92), (26.60, 4.04), (25.00, 3.98)],
    "conditional": [(57.00, 50.18, 63.56), (63.03, 56.33, 69.26), (75.76, 68.67, 81.66)],
    "lift":        (41.40, 37.16, 45.77),
}
molmoact2 = {
    "plain":       [(25.00, 3.98), (23.60, 3.92), (25.00, 3.98)],
    "conditional": [(63.78, 56.83, 70.18), (61.46, 54.40, 68.06), (66.84, 59.81, 73.19)],
    "lift":        (39.20, 35.02, 43.55),
}

policies = [
    (r"$\pi_{0.5}$",   pi05,       "#ECE133"),
    ("DreamZero",      dreamzero,  "#029E73"),
    ("MolmoAct2",     molmoact2, "#CC78BC"),
]

prompt_labels = [
    "\n".join([
        r"\textit{\textquotedblleft Pick up}",
        r"\textit{the \{object\}.\textquotedblright}",
    ]),
    "\n".join([
        r"\textit{\textquotedblleft Pick up}",
        r"\textit{the \{object\}}",
        r"\textit{to give}",
        r"\textit{to someone.\textquotedblright}",
    ]),
    "\n".join([
        r"\textit{\textquotedblleft Pick up}",
        r"\textit{the \{object\}}",
        r"\textit{by the}",
        r"\textit{handle.\textquotedblright}",
    ]),
]
lift_label = "\n".join([
    r"\textit{\textquotedblleft Pick up}",
    r"\textit{the \{object\}.\textquotedblright}",
])
n_groups = len(prompt_labels)
n_policies = len(policies)
group_spacing = 4.5  # gap between group centers; > 1.0 widens the empty
                     # space between adjacent bar sets so tick labels can
                     # breathe.
x = np.arange(n_groups) * group_spacing
# Extra horizontal gap between the base-pick (lift) group and the prompt-level
# groups, on top of one group_spacing, so the dotted separator has room.
lift_extra_gap = 2.0
x_lift = -group_spacing - lift_extra_gap
bar_width = 0.98


fig, ax = plt.subplots(figsize=(70.0, 62.0))

ymax = 0.0
for i, (name, data, color) in enumerate(policies):
    offset = (i - (n_policies - 1) / 2) * bar_width
    means = np.array([m for m, _ in data["plain"]])
    errs = np.array([e for _, e in data["plain"]])
    ymax = max(ymax, float((means + errs).max()))
    ax.bar(
        x + offset,
        means,
        width=bar_width,
        label=name,
        color=color,
        edgecolor="black",
        linewidth=0.8,
        yerr=errs,
        capsize=20.0,
        error_kw={"elinewidth": 4.5, "ecolor": "black"},
    )

    # Lift-rate bar with asymmetric 95% CI error bars, drawn one group-width
    # to the left of the prompt-level groups (separated by a dotted line).
    lmean, llo, lhi = data["lift"]
    lyerr = np.array([[lmean - llo], [lhi - lmean]])
    ymax = max(ymax, float(lhi))
    ax.bar(
        x_lift + offset,
        lmean,
        width=bar_width,
        color=color,
        edgecolor="black",
        linewidth=0.8,
        yerr=lyerr,
        capsize=20.0,
        error_kw={"elinewidth": 4.5, "ecolor": "black"},
    )

# Solid jagged (zigzag) separator between the base-pick (lift) group and the
# prompt-level groups — drawn as a piecewise polyline alternating ±dx around
# the midpoint x so it reads as a hand-drawn tear/break.
_sep_x_center = (x_lift + x[0]) / 2
_sep_n_segments = 14
_sep_dx = 0.45
_sep_ys = np.linspace(0, 60, _sep_n_segments + 1)
_sep_xs = _sep_x_center + _sep_dx * (1 - 2 * (np.arange(len(_sep_ys)) % 2))
ax.plot(
    _sep_xs,
    _sep_ys,
    color="black",
    linewidth=10.0,
    solid_capstyle="round",
    solid_joinstyle="miter",
    zorder=5,
)

ax.set_xticks([x_lift, *x])
ax.set_xticklabels([lift_label, *prompt_labels], rotation=0)
ax.set_xlabel("Prompt level")

# Two grouped sub-axis labels placed ABOVE the axes — one over the
# base-pick bar, one centered over the three semantic-pick groups. Wrapped
# so each label keeps the visual width it had at the smaller font size,
# now at the regular text size.
_subaxis_label_y = 1.03  # axes fraction (above axes top)
_subaxis_blend = ax.get_xaxis_transform()
ax.text(
    x_lift,
    _subaxis_label_y,
    textwrap.fill("Base Pick Task with Standard Prompt", width=14),
    ha="center",
    va="bottom",
    transform=_subaxis_blend,
)
ax.text(
    float(np.mean(x)),
    _subaxis_label_y,
    textwrap.fill("Grasping Affordance Pick Task", width=12),
    ha="center",
    va="bottom",
    transform=_subaxis_blend,
)
ax.set_ylabel(r"Success rate (\%)")
ax.set_title(
    "Pick with Affordance Task: Policy Performance\n"
    "Does Not Improve With Decomposed Steps in Prompts",
    pad=800,
)
ax.set_ylim(0, 60)
ax.legend(
    frameon=False,
    ncol=n_policies,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.95),
)

fig.tight_layout()
fig.subplots_adjust(left=0.08, right=0.995, top=0.65, bottom=0.27)
ax.yaxis.labelpad = 63
ax.xaxis.labelpad = 72
out = "success_by_semprompt_level.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.5)
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight", pad_inches=0.5)
print(f"Saved: {out}")
