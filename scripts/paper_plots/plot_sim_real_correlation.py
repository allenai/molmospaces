"""Scatter plot: simulation vs. real-world success rate for the
pick-with-occlusion task, one point per policy."""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.stats import spearmanr

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
    "legend.fontsize": 18,
})

# (label, sim_mean, sim_err, real, color)
policies = [
    ("DreamZero",     18.00, 2.50, 65.5, "#029E73"),
    (r"$\pi_{0.5}$",  14.10, 2.30, 42.9, "#ECE133"),
    (r"$\pi_{0}$-FAST", 8.21, 1.87, 36.4, "#DE8F05"),
    (r"$\pi_{0}$",     2.81, 1.22, 34.0, "#0173B2"),
    ("MolmoAct2",    17.20, 2.47, 45.0, "#CC78BC"),
]

sim_means = np.array([p[1] for p in policies])
sim_errs  = np.array([p[2] for p in policies])
real_vals = np.array([p[3] for p in policies])

# Pearson and Spearman correlations
r_pearson = np.corrcoef(sim_means, real_vals)[0, 1]
rho_spearman, _ = spearmanr(sim_means, real_vals)

# Least-squares fit (for visual trend line, not weighted).
slope, intercept = np.polyfit(sim_means, real_vals, 1)
x_fit = np.linspace(0, sim_means.max() * 1.2, 100)
y_fit = slope * x_fit + intercept

fig, ax = plt.subplots(figsize=(5.0, 4.2))

ax.plot(
    x_fit, y_fit,
    linestyle="--",
    color="black",
    linewidth=1.0,
    alpha=0.5,
    zorder=1,
)

for name, sim_mean, sim_err, real, color in policies:
    ax.plot(
        sim_mean, real,
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=name,
        zorder=3,
    )

ax.set_xlabel(r"Simulation Success Rate (\%)", labelpad=8)
ax.set_ylabel(r"Real-World Success Rate (\%)", labelpad=8)
ax.set_title(
    "Sim-to-Real Correlation for Pick with Occlusion Task",
    pad=20,
    y=1.06,
)
ax.set_xlim(0, sim_means.max() * 1.10)
ax.set_ylim(0, real_vals.max() * 1.20)

# Linear-fit summary inside the axes (bottom-right) to keep the legend tight.
ax.text(
    0.97, 0.04,
    fr"Linear fit: Pearson $r={r_pearson:.2f}$,"
    "\n"
    fr"Spearman $\rho={rho_spearman:.2f}$",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=14,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="lightgray", linewidth=0.6, alpha=0.9),
)

ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=5,
    handletextpad=0.3,
    columnspacing=0.9,
    fontsize=14,
)

fig.tight_layout()
fig.subplots_adjust(top=0.86, bottom=0.24, left=0.15, right=0.98)
out_path = "sim_real_correlation.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
fig.savefig(out_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
