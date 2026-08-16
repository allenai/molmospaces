"""Gold-vs-ported scene-texture / render parity.

The port used to build its texture pools from THOR's material-database.json
(a different, larger, differently-categorized set) while gold sampled its
repo-local assets/textures/<Category>/ pack -- so the two stacks rendered the
same houses with different walls, floors and counters. The pack now lives in
the ResourceManager cache under textures/fetchman/<Category>/ and
build_thor_texture_pools() reads it there. This script checks that.

Run once per stack, then diff:

    cd /Users/maxa/code/g1_molmo
    conda run -n g1_molmo python \
        /Users/maxa/code/molmospaces/scripts/g1_molmo_port_comparison/check_texture_parity.py \
        --stack gold --out /tmp/tex_gold

    cd /Users/maxa/code/molmospaces
    conda run -n mlspaces python scripts/g1_molmo_port_comparison/check_texture_parity.py \
        --stack ported --out /tmp/tex_ported

    conda run -n mlspaces python scripts/g1_molmo_port_comparison/check_texture_parity.py \
        --compare /tmp/tex_gold /tmp/tex_ported

Each run writes <out>.json (pool sizes + the per-category texture basenames
actually applied to the scene) and <out>_<camera>.png.

What must match exactly: the pool contents and the applied texture basenames.
Renders are compared as a mean-absolute pixel difference rather than byte
equality -- the two conda envs ship different MuJoCo versions (3.11.0 vs
3.5.0), which perturbs rasterization slightly; see check_gold_parity.py's
docstring on the same split.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCENE = "scenes/procthor-10k-val/val_0.xml"


def run(stack, out_prefix, seed):
    if stack == "gold":
        sys.path.insert(0, str(Path.home() / "code" / "g1_molmo"))
        from molmospaces.configs.bowl_mixed_grasponly import get_config
        from molmospaces.env_g1ms import make_env
    else:
        from molmo_spaces.g1_molmo_port.configs.bowl_mixed_grasponly import get_config
        from molmo_spaces.g1_molmo_port.env_g1ms import make_env

    cfg = get_config().copy_and_resolve_references()
    cfg["scene"] = SCENE
    cfg["randomize_scene"] = False
    cfg["randomize_object"] = False
    cfg["launch_viewer"] = False
    cfg["seed"] = seed

    env = make_env(cfg.to_dict())
    raw = getattr(env, "env", env)
    env.reset()

    pools = raw._texture_pools
    applied = raw._scene_texture_paths
    record = {
        "stack": stack,
        "seed": seed,
        "scene": SCENE,
        "pool_sizes": {c: len(v) for c, v in sorted(pools.items())},
        "pool_basenames": {c: [os.path.basename(p) for p in v] for c, v in sorted(pools.items())},
        "applied_basenames": {
            c: [os.path.basename(p) for p in v] for c, v in sorted(applied.items())
        },
        "renders": {},
    }

    import imageio.v2 as imageio

    for name, img in raw.render_cameras().items():
        path = f"{out_prefix}_{name}.png"
        imageio.imwrite(path, img)
        record["renders"][name] = {"path": path, "shape": list(img.shape)}

    with open(f"{out_prefix}.json", "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    print(f"[{stack}] pools: {record['pool_sizes']}")
    print(f"[{stack}] applied: {json.dumps(record['applied_basenames'], sort_keys=True)}")
    print(f"[{stack}] wrote {out_prefix}.json + {len(record['renders'])} renders")
    return 0


def compare(a_prefix, b_prefix):
    import numpy as np

    with open(f"{a_prefix}.json") as fh:
        a = json.load(fh)
    with open(f"{b_prefix}.json") as fh:
        b = json.load(fh)

    failed = False
    for key in ("pool_basenames", "applied_basenames"):
        if a[key] == b[key]:
            n = sum(len(v) for v in a[key].values())
            print(f"PASS {key}: identical ({n} files across {len(a[key])} categories)")
        else:
            failed = True
            print(f"FAIL {key}:")
            for cat in sorted(set(a[key]) | set(b[key])):
                x, y = a[key].get(cat, []), b[key].get(cat, [])
                if x != y:
                    print(f"  {cat}: gold {len(x)} vs ours {len(y)}")
                    print(f"    gold-only: {sorted(set(x) - set(y))[:6]}")
                    print(f"    ours-only: {sorted(set(y) - set(x))[:6]}")

    import imageio.v2 as imageio

    for name in sorted(set(a["renders"]) & set(b["renders"])):
        ia = imageio.imread(a["renders"][name]["path"]).astype(np.int16)
        ib = imageio.imread(b["renders"][name]["path"]).astype(np.int16)
        if ia.shape != ib.shape:
            failed = True
            print(f"FAIL render {name}: shape {ia.shape} vs {ib.shape}")
            continue
        diff = np.abs(ia - ib)
        frac = float((diff > 8).mean())
        print(
            f"render {name}: mean|diff|={diff.mean():.3f} max={int(diff.max())} "
            f"frac(pixels>8)={frac:.4f}"
        )
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=("gold", "ported"))
    ap.add_argument("--out", default="/tmp/tex_parity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("GOLD_PREFIX", "OURS_PREFIX"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.stack:
        ap.error("--stack is required unless --compare is given")
    return run(args.stack, args.out, args.seed)


if __name__ == "__main__":
    sys.exit(main())
