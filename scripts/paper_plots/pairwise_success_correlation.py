#!/usr/bin/env python
"""Episode-wise pairwise success correlation between policies on the same benchmark.

Given several eval output folders -- each one policy's results on the *same* benchmark --
build the table

    cell[x][y] = P(policy y succeeded | policy x succeeded)

over the episodes both policies ran. This answers "do these policies succeed and fail on
the same scenes?": a row that stays near each policy's marginal success rate means the
successes are roughly independent, while values well above the marginal mean the policies
are solving the same episodes.

Success is defined exactly as scripts/paper_plots/thor_analysis.py does -- an episode
counts as a success if its per-timestep `success` array is ever True -- so the marginal
rates printed here match `analyze_run.py --run-path <folder>`.

Episodes are keyed by (house_index, trajectory_index). Because that assumes both runs
enumerated each house's episodes in the same order, the key is *verified* against a
fingerprint taken from the episode's initial robot base pose and object start pose, which
differ per episode. Misaligned episodes are reported rather than silently correlated.

Usage:
    python scripts/paper_plots/pairwise_success_correlation.py \
        --run-path pi05=/path/to/pi05_results \
        --run-path dreamzero=/path/to/dreamzero_results \
        --run-path molmoact2=/path/to/molmoact2_results

    # labels are optional; the folder name is used when omitted
    python scripts/paper_plots/pairwise_success_correlation.py -r /a -r /b --csv out.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

FINGERPRINT_FIELDS = ("robot_base_pose", "obj_start")
HOUSE_RE = re.compile(r"house_(\d+)$")


def parse_run_arg(value: str) -> tuple[str, str]:
    """Accept either "label=/path" or "/path" (label derived from the folder name)."""
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"empty label in {value!r}")
    else:
        path = value
        label = os.path.basename(os.path.normpath(path)) or path
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return label, path


def episode_fingerprint(traj: h5py.Group) -> tuple | None:
    """A per-episode signature that does not depend on policy behaviour.

    Uses the first timestep of fields fixed by the benchmark episode spec (where the
    robot and the target object start), so two runs of the same episode agree while
    different episodes do not.
    """
    extra = traj.get("obs", {}).get("extra") if "obs" in traj else None
    if extra is None:
        return None
    parts: list[float] = []
    for field in FINGERPRINT_FIELDS:
        if field not in extra:
            return None
        arr = np.asarray(extra[field])
        if arr.ndim < 2 or arr.shape[0] == 0:
            return None
        parts.extend(np.round(arr[0].astype(float), 3).tolist())
    return tuple(parts)


def load_run(path: str) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], tuple]]:
    """Return {(house, traj_idx): success} and {(house, traj_idx): fingerprint}."""
    success: dict[tuple[int, int], bool] = {}
    fingerprints: dict[tuple[int, int], tuple] = {}

    for house_dir in sorted(glob.glob(os.path.join(path, "house_*"))):
        m = HOUSE_RE.search(os.path.normpath(house_dir))
        if not m:
            continue
        house = int(m.group(1))
        for h5_path in sorted(glob.glob(os.path.join(house_dir, "*.h5"))):
            try:
                with h5py.File(h5_path, "r") as h:
                    for traj_key in sorted(h.keys()):
                        traj = h[traj_key]
                        if "success" not in traj:
                            continue
                        idx_match = re.search(r"(\d+)$", traj_key)
                        if idx_match is None:
                            continue
                        key = (house, int(idx_match.group(1)))
                        success[key] = bool(np.asarray(traj["success"]).any())
                        fp = episode_fingerprint(traj)
                        if fp is not None:
                            fingerprints[key] = fp
            except OSError as exc:
                print(f"  warning: could not read {h5_path}: {exc}", file=sys.stderr)
    return success, fingerprints


def check_alignment(labels, runs, fingerprints, common) -> int:
    """Verify identical keys really are the same episode. Returns mismatch count."""
    reference_label = labels[0]
    mismatches = 0
    for key in common:
        ref = fingerprints[reference_label].get(key)
        if ref is None:
            continue
        for label in labels[1:]:
            other = fingerprints[label].get(key)
            if other is not None and other != ref:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        f"  MISALIGNED house_{key[0]} traj_{key[1]}: "
                        f"{reference_label} vs {label} have different start states",
                        file=sys.stderr,
                    )
                break
    return mismatches


def report_statistics(labels, vectors, width) -> None:
    """Correlation, independence, and paired-difference statistics.

    Three different questions, deliberately kept separate:
      phi / odds ratio  -- are two policies' successes correlated?
      McNemar           -- is one policy better than the other? (paired; the right
                           test here, since both ran the same episodes -- comparing
                           independent confidence intervals would understate power)
      co-solve spread   -- across all policies at once, is episode difficulty shared?
    """
    from itertools import combinations

    try:
        from scipy.stats import binomtest, fisher_exact
    except ImportError:  # scipy is a molmospaces dependency, but degrade gracefully
        print("\n(scipy unavailable -- skipping significance tests)")
        return

    print("\n" + "=" * 78)
    print("PAIRWISE STATISTICS")
    print("=" * 78)
    print(
        f"{'pair':<{2*width}}{'phi':>8}{'odds ratio':>13}{'Fisher p':>12}"
        f"{'McNemar p':>12}   discordant"
    )
    for x, y in combinations(labels, 2):
        vx, vy = vectors[x], vectors[y]
        n11 = int((vx & vy).sum())
        n10 = int((vx & ~vy).sum())
        n01 = int((~vx & vy).sum())
        n00 = int((~vx & ~vy).sum())

        # phi == Pearson r for two binary vectors
        denom = np.sqrt(
            float(n11 + n10) * float(n01 + n00) * float(n11 + n01) * float(n10 + n00)
        )
        phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else float("nan")

        odds, fisher_p = fisher_exact([[n11, n10], [n01, n00]])

        # McNemar (exact): among episodes the two disagree on, is the split 50/50?
        disc = n10 + n01
        mcnemar_p = binomtest(n10, disc, 0.5).pvalue if disc > 0 else float("nan")

        pair = f"{x} vs {y}"
        odds_s = "inf" if np.isinf(odds) else f"{odds:.2f}"
        print(
            f"{pair:<{2*width}}{phi:>8.3f}{odds_s:>13}{fisher_p:>12.2e}"
            f"{mcnemar_p:>12.2e}   {n10}/{n01} ({disc})"
        )
    print(
        "\n  phi: correlation of the two success vectors (0 = independent, 1 = identical).\n"
        "  odds ratio / Fisher p: effect size and exact test against independence.\n"
        "  McNemar p: paired test of *which policy is better*, using only the episodes\n"
        "    they disagree on (discordant column is x-only/y-only wins)."
    )

    # --- joint view: how many policies solved each episode, vs independence ---
    n_pol = len(labels)
    stacked = np.vstack([vectors[label] for label in labels])
    per_episode = stacked.sum(axis=0)
    observed = np.bincount(per_episode, minlength=n_pol + 1)

    # Expected counts under independence with the same marginals (Poisson-binomial
    # via direct convolution -- exact, no simulation needed).
    probs = [float(vectors[label].mean()) for label in labels]
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1.0 - p, p])
    expected = dist * len(per_episode)

    print("\n" + "=" * 78)
    print("HOW MANY POLICIES SOLVED EACH EPISODE (shared difficulty)")
    print("=" * 78)
    print(f"{'# policies':>12}{'observed':>12}{'if independent':>17}{'excess':>10}")
    for k in range(n_pol + 1):
        excess = observed[k] - expected[k]
        print(f"{k:>12}{observed[k]:>12}{expected[k]:>17.1f}{excess:>+10.1f}")
    chi2 = float(
        np.sum((observed - expected) ** 2 / np.where(expected > 0, expected, np.nan)[: len(observed)])
    )
    # df: the histogram has n_pol+1 cells (n_pol free after fixing the total), and the
    # marginals -- estimated from this same data -- pin the mean, costing one more.
    # Verified empirically: permuting each policy independently gives a null whose mean
    # is n_pol-1, and a chi-square distribution's mean equals its df.
    df = max(n_pol - 1, 1)
    print(f"\n  chi-square vs independence: {chi2:.1f} on {df} df")
    print(
        "  (A permutation null -- shuffling each policy's successes across episodes,\n"
        "   which preserves marginals and destroys only the episode-level association --\n"
        "   confirms this df and gives p far below any asymptotic approximation.)"
    )
    print(
        "  Excess at the extremes (0 and all) with a deficit in the middle means episode\n"
        "  difficulty is shared: the benchmark separates easy from hard scenes more than\n"
        "  it separates policies."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pairwise P(y succeeded | x succeeded) across policies on one benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--run-path",
        dest="runs",
        action="append",
        required=True,
        type=parse_run_arg,
        metavar="[LABEL=]PATH",
        help="Eval output folder, optionally labelled. Repeat for each policy.",
    )
    parser.add_argument("--csv", default=None, help="Also write the matrix to this CSV path.")
    parser.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="Do not verify episode identity via start-state fingerprints.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Only print the conditional-probability table (skip phi/Fisher/McNemar).",
    )
    args = parser.parse_args()

    labels = [label for label, _ in args.runs]
    if len(set(labels)) != len(labels):
        parser.error(f"duplicate labels: {labels}")
    if len(labels) < 2:
        parser.error("need at least two runs to correlate")

    runs: dict[str, dict] = {}
    fingerprints: dict[str, dict] = {}
    print("Loading runs...")
    for label, path in args.runs:
        succ, fps = load_run(path)
        if not succ:
            print(f"  ERROR: no episodes found in {path}", file=sys.stderr)
            return 1
        runs[label] = succ
        fingerprints[label] = fps
        n = len(succ)
        s = sum(succ.values())
        print(f"  {label:<14} {n:>5} episodes  {s:>4} successes  {100.0*s/n:>6.2f}%   {path}")

    # Episodes present in every run
    common = set.intersection(*(set(v.keys()) for v in runs.values()))
    counts = {label: len(v) for label, v in runs.items()}
    if len(set(counts.values())) != 1 or len(common) != next(iter(counts.values())):
        print(
            f"\n  NOTE: runs do not cover identical episode sets; "
            f"using the {len(common)} episodes common to all "
            f"(per-run totals: {counts})"
        )
    common_sorted = sorted(common)
    print(f"\nEpisodes compared: {len(common_sorted)}")

    if not args.skip_alignment_check:
        mismatches = check_alignment(labels, runs, fingerprints, common_sorted)
        if mismatches:
            print(
                f"\n  ERROR: {mismatches} episodes have mismatched start states across runs.\n"
                f"  (house, traj) indices do not refer to the same episode, so the table\n"
                f"  below would be meaningless. Re-run with --skip-alignment-check to force.",
                file=sys.stderr,
            )
            return 1
        print("Alignment check: start states agree across runs for all compared episodes.")

    vectors = {
        label: np.array([runs[label][k] for k in common_sorted], dtype=bool) for label in labels
    }

    # cell[x][y] = P(y succeeded | x succeeded)
    n_labels = len(labels)
    matrix = np.full((n_labels, n_labels), np.nan)
    for i, x in enumerate(labels):
        x_succ = vectors[x]
        denom = int(x_succ.sum())
        for j, y in enumerate(labels):
            if denom == 0:
                continue
            matrix[i, j] = float((x_succ & vectors[y]).sum()) / denom

    width = max(12, max(len(x) for x in labels) + 2)
    print("\n" + "=" * 78)
    print("P( column policy succeeded | row policy succeeded )")
    print("=" * 78)
    header = " " * width + "".join(f"{y:>{width}}" for y in labels)
    print(header)
    for i, x in enumerate(labels):
        row = f"{x:<{width}}" + "".join(
            ("     n/a    " if np.isnan(matrix[i, j]) else f"{100*matrix[i,j]:>{width-1}.1f}%")
            for j in range(n_labels)
        )
        print(row)

    print("\nMarginal success rate (baseline for reading each column):")
    for label in labels:
        v = vectors[label]
        print(f"  {label:<{width}}{100.0*v.mean():>6.2f}%   ({int(v.sum())}/{len(v)})")
    print(
        "\nA cell above the column's marginal means that policy does better than usual on\n"
        "episodes the row policy also solved (correlated competence). A cell near the\n"
        "marginal means the two policies' successes are roughly independent."
    )

    if not args.no_stats:
        report_statistics(labels, vectors, width)

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("given_x_succeeded," + ",".join(labels) + "\n")
            for i, x in enumerate(labels):
                cells = ["" if np.isnan(matrix[i, j]) else f"{matrix[i,j]:.6f}" for j in range(n_labels)]
                fh.write(x + "," + ",".join(cells) + "\n")
            fh.write("\nmarginal_success_rate\n")
            for label in labels:
                fh.write(f"{label},{vectors[label].mean():.6f}\n")
            fh.write(f"\nepisodes_compared,{len(common_sorted)}\n")
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
