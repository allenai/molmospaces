"""Compare a gold rollout trace against a ported/native one.

    python scripts/g1_molmo_port_comparison/check_gold_parity.py GOLD.txt OURS.txt
    python .../check_gold_parity.py GOLD.txt OURS.txt --strict

Normalizes only the cosmetic differences the ported stack is allowed to have
(the [gold]/[ported]/[ours] print prefix, warnings, asset paths), then compares.

Two modes, because only one of them is achievable across the two conda envs:

  default   Compare the DISCRETE invariants: episode selection, target object,
            spawn pose, per-episode step counts / sim_time / success. These
            MUST match. Continuous physics state is reported as drift, not
            failure.

  --strict  Require every trace line byte-identical. Only meaningful when both
            files come from the SAME environment -- e.g. ported-vs-ported
            across a refactor, which is the real regression gate. Gold-vs-ported
            can never pass --strict: the envs ship different MuJoCo versions
            (3.11.0 vs 3.5.0), which diverges continuous state at ~1e-3 over a
            couple thousand steps. That is not a port defect; the only lever is
            aligning the MuJoCo versions, a real dependency decision
            (molmo-spaces pins mujoco~=3.5.0).

Gold vs ported -- does the port still behave like the reference stack:

    cd ~/code/g1_molmo && conda run -n g1_molmo python \\
        molmospaces/scripts/g1_molmo_comparison/generate_gold_rollout.py \\
        --seed 0 > /tmp/gold.txt 2>&1
    cd ~/code/molmospaces && conda run -n mlspaces python \\
        scripts/g1_molmo_port_comparison/generate_ported_rollout.py \\
        --seed 0 > /tmp/ported.txt 2>&1
    conda run -n mlspaces python \\
        scripts/g1_molmo_port_comparison/check_gold_parity.py \\
        /tmp/gold.txt /tmp/ported.txt

Expect `PASS: 9/9 discrete invariants identical`, ending in `SUCCESS on
episode 4` (steps=2338 sim_time=12.04s), with the intermediate failures at 750
and 288 steps matching too. Continuous-state drift is reported, not failed.

Ported vs ported -- the regression gate to run around EVERY refactor step.
Record a baseline before touching anything, then compare:

    conda run -n mlspaces python \\
        scripts/g1_molmo_port_comparison/generate_ported_rollout.py \\
        --seed 0 > /tmp/baseline.txt 2>&1
    # ... make a change ...
    conda run -n mlspaces python \\
        scripts/g1_molmo_port_comparison/generate_ported_rollout.py \\
        --seed 0 > /tmp/after.txt 2>&1
    conda run -n mlspaces python \\
        scripts/g1_molmo_port_comparison/check_gold_parity.py \\
        /tmp/baseline.txt /tmp/after.txt --strict

Expect `PASS (strict): 212 trace lines byte-identical`. If it fails, stop and
fix before continuing.

Neither gate looks at pixels -- check_texture_parity.py in this directory does
that, and is only meaningful when the fetchman texture pack is present (watch
for build_thor_texture_pools()'s JORDI-TODO fallback warning in the log first).
Neither covers the native pick pipeline either; that is run_house_sweep.py.

Exit code 0 = pass, 1 = fail.
"""

import argparse
import re
import sys

SKIP_SUBSTRINGS = (
    "Attach conflict",
    "gravity: parent has",
    "nkey: parent has",
    "UserWarning",
    "self.model = spec.compile()",
    "saved end-of-episode reset state",
    "WARNING:",
    "Using SCENES_ROOT",
)

# Lines whose equality is non-negotiable regardless of physics drift.
INVARIANT_PATTERNS = (
    re.compile(r"=== episode \d+: target=\S+ robot_xy=\S+.*robot_yaw_rad=[-\d.]+"),
    re.compile(r"episode \d+ result: steps=\d+ sim_time=[\d.]+s success=\w+"),
    re.compile(r"SUCCESS on episode \d+"),
)


def normalize(path):
    out = []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or any(s in line for s in SKIP_SUBSTRINGS):
                continue
            line = re.sub(r"\[(gold|ported|ours)\]", "[X]", line)
            line = re.sub(r"\[(gold|ported|ours) phase", "[X phase", line)
            line = re.sub(r"\[(gold|ported|ours) obs", "[X obs", line)
            out.append(line)
    return out


def invariants(lines):
    return [ln for ln in lines if any(p.search(ln) for p in INVARIANT_PATTERNS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("ours")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="require every line byte-identical (same-env comparisons only)",
    )
    args = ap.parse_args()

    a, b = normalize(args.gold), normalize(args.ours)

    if args.strict:
        if a == b:
            print(f"PASS (strict): {len(a)} trace lines byte-identical")
            return 0
        print(f"FAIL (strict): {len(a)} vs {len(b)} lines")
        shown = 0
        for i in range(max(len(a), len(b))):
            x = a[i] if i < len(a) else "<missing>"
            y = b[i] if i < len(b) else "<missing>"
            if x != y:
                print(f"\n--- first diff at trace line {i} ---\n  gold: {x}\n  ours: {y}")
                shown += 1
                if shown >= 3:
                    break
        return 1

    ia, ib = invariants(a), invariants(b)
    if ia == ib:
        print(f"PASS: {len(ia)}/{len(ia)} discrete invariants identical")
        for ln in ia:
            print(f"    {ln}")
        drift = sum(1 for x, y in zip(a, b) if x != y)
        print(f"\n(continuous-state lines differing: {drift}/{min(len(a), len(b))} -- expected")
        print(" across envs; see this file's docstring on the MuJoCo version split)")
        return 0

    print(f"FAIL: discrete invariants differ ({len(ia)} vs {len(ib)})")
    for i in range(max(len(ia), len(ib))):
        x = ia[i] if i < len(ia) else "<missing>"
        y = ib[i] if i < len(ib) else "<missing>"
        if x != y:
            print(f"\n--- {i} ---\n  gold: {x}\n  ours: {y}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
