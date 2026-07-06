"""Bar chart for prompt level 2 only:
two x-axis groups (overall success rate, conditional-on-lift success rate),
3 policy bars each, with error bars.
"""

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

# Prompt level 2 only. Plain: (mean, +/- err). Conditional: (mean, ci_lo, ci_hi).
policies = [
    (r"$\pi_{0.5}$",   (19.40, 3.70), (54.49, 47.15, 61.64), "#ECE133"),
    ("DreamZero",      (26.60, 4.04), (63.03, 56.33, 69.26), "#029E73"),
    ("MolmoAct2",     (23.60, 3.92), (61.46, 54.40, 68.06), "#CC78BC"),
]

group_labels = ["Overall", r"Conditional on target lifted"]
n_groups = len(group_labels)
n_policies = len(policies)
x = np.arange(n_groups)
bar_width = 0.26

fig, ax = plt.subplots(figsize=(6.0, 3.8))

ymax = 0.0
for i, (name, plain, cond, color) in enumerate(policies):
    plain_mean, plain_err = plain
    cond_mean, cond_lo, cond_hi = cond

    means = np.array([plain_mean, cond_mean])
    yerr = np.array([
        [plain_err, cond_mean - cond_lo],   # lower
        [plain_err, cond_hi - cond_mean],   # upper
    ])
    ymax = max(ymax, plain_mean + plain_err, cond_hi)

    offset = (i - (n_policies - 1) / 2) * bar_width
    ax.bar(
        x + offset,
        means,
        width=bar_width,
        label=name,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        yerr=yerr,
        capsize=3.0,
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
    )

ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_xlabel("")
ax.set_ylabel(r"Success rate (\%)")
ax.set_title(r"Policy Success Rate, Prompt Level 2 "
             r"(\textit{``Pick up the \{object\} to give to someone.''})")
ax.set_ylim(0, 100)
ax.legend(
    frameon=False,
    ncol=n_policies,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
)

fig.tight_layout()
out_path = "success_semprompt_level2.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
