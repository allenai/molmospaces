# benchmark_ithor_pick_hard_simple

Simplified **THOR-only** pick benchmark for [Isaac Lab Arena](../../scripts/run_arena_benchmark_episode.py): same **iTHOR `house_index`** set and **robot placement** style as the published **FrankaPickHardBench** (first episode per house as template), but the pickup is always **`Bowl_27`** with a **heuristic** world pose (fixed offset in the robot frame: default **0.55 m forward, 0.42 m up** (tuned so the bowl clears the Arena Franka base and typical counter height), transformed by `robot_base_pose`).

This is **not** MuJoCo pose parity: the bowl is not at the original sampled object pose. It is meant for **pipeline and visual** checks with **no Objaverse USDs**—only `objects/thor/` and iTHOR scene USDs under your install root.

## Regenerate `benchmark.json`

From the repo root (or anywhere), with `molmo_spaces_isaac` on `PYTHONPATH` or run from a venv where the package is installed:

```bash
python3 molmo_spaces_isaac/scripts/build_simple_ithor_thor_pick_benchmark.py \
  --source /path/to/FrankaPickHardBench_.../benchmark.json \
  --output molmo_spaces_isaac/examples/benchmark_ithor_pick_hard_simple/benchmark.json
```

Environment defaults for `--source`: `MOLMO_PICK_BENCHMARK_SOURCE` or `MOLMO_PICK_BENCHMARK_DIR` (file or directory containing `benchmark.json`).

Options: `--max-houses K` (cap episode count), `--offset-forward`, `--offset-lateral`, `--offset-up` (meters in robot frame).

## Assets

Use a single **`ms-download`** install with **thor** assets and **ithor** scenes, e.g.:

```bash
ms-download --type usd --install-dir /path/to/assets --assets thor --scenes ithor
```

Pass **`--assets_root /path/to/assets`** to the Arena runner. For this bundled benchmark, **`--scenes_root` is optional**: if omitted, the runner defaults it to **`--assets_root`** (or `MOLMO_ISAAC_ASSETS_ROOT`) so one root is enough.
