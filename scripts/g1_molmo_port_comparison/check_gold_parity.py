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
            across a refactor, which is the real regression gate (see
            NEXT_STEPS.md). Gold-vs-ported can never pass --strict: the envs
            ship different MuJoCo versions (3.11.0 vs 3.5.0), which diverges
            continuous state at ~1e-3 over a couple thousand steps.

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
