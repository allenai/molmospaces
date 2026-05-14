#!/usr/bin/env python3
"""Run Arena benchmark episodes as isolated Isaac processes.

Isaac/Kit can be brittle when many environments are created and torn down in a
single Python process. This wrapper keeps the lower-level episode runner as the
source of truth and launches one fresh process per benchmark episode, then
aggregates the per-episode JSON files into a benchmark summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_RUNNER = SCRIPT_DIR / "run_arena_benchmark_episode.py"


def _load_all_episode_dicts(benchmark_dir: Path) -> list[dict]:
    try:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

        episodes = load_all_episodes(benchmark_dir)
        return [ep.model_dump() if hasattr(ep, "model_dump") else ep for ep in episodes]
    except ImportError:
        bench_file = benchmark_dir / "benchmark.json"
        if not bench_file.is_file():
            raise SystemExit(f"No benchmark.json in {benchmark_dir}")
        with open(bench_file) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"{bench_file} does not contain a benchmark episode list")
        return data


def _parse_episode_indices(raw: str | None, n_total: int) -> list[int]:
    if raw is None:
        return [0]
    indices: list[int] = []
    for piece in raw.replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(piece))
    if not indices:
        raise SystemExit("--episode_indices did not contain any episode indices")

    for idx in indices:
        if idx < 0 or idx >= n_total:
            raise SystemExit(f"episode index {idx} out of range [0, {n_total - 1}]")
    return indices


def _episode_metadata(idx: int, episode_dict: dict) -> dict:
    task = episode_dict.get("task") or {}
    return {
        "idx": int(idx),
        "house_index": episode_dict.get("house_index"),
        "scene_dataset": episode_dict.get("scene_dataset"),
        "pickup_obj_name": task.get("pickup_obj_name"),
        "task_description": ((episode_dict.get("language") or {}).get("task_description")),
    }


def _stream_process(cmd: list[str], log_path: Path, cwd: Path | None) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return int(proc.wait())


def _read_episode_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run MolmoSpaces Arena benchmark episodes in isolated Isaac "
            "processes and aggregate result JSON."
        )
    )
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help="Comma-separated benchmark indices or ranges, e.g. '0,8,17-20'.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Run the first N episodes. Use 0 for all episodes. Mutually exclusive with --episode_indices.",
    )
    parser.add_argument("--results_json", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=None, help="CWD for child episode runs.")
    parser.add_argument(
        "--per_episode_dir",
        type=Path,
        default=None,
        help="Directory for per-episode result JSON and logs. Defaults next to --results_json.",
    )
    parser.add_argument(
        "--stop_on_failure",
        action="store_true",
        help="Stop after the first failed child process or failed episode.",
    )
    args, child_args = parser.parse_known_args()

    if args.max_episodes is not None and args.episode_indices is not None:
        raise SystemExit("Use either --max_episodes or --episode_indices, not both.")

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    if not benchmark_dir.is_dir():
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")

    episode_dicts = _load_all_episode_dicts(benchmark_dir)
    if not episode_dicts:
        raise SystemExit(f"No episodes found in {benchmark_dir}")

    if args.max_episodes is not None:
        n_run = len(episode_dicts) if args.max_episodes == 0 else min(args.max_episodes, len(episode_dicts))
        indices = list(range(n_run))
    else:
        indices = _parse_episode_indices(args.episode_indices, len(episode_dicts))

    results_json = args.results_json.expanduser().resolve()
    per_episode_dir = (
        args.per_episode_dir.expanduser().resolve()
        if args.per_episode_dir is not None
        else results_json.parent / f"{results_json.stem}_episodes"
    )
    per_episode_dir.mkdir(parents=True, exist_ok=True)

    work_dir = args.work_dir.expanduser().resolve() if args.work_dir is not None else None
    print(
        f"[molmospaces_arena_batch] Running {len(indices)} episode(s): {indices}",
        flush=True,
    )
    print(f"[molmospaces_arena_batch] Per-episode artifacts: {per_episode_dir}", flush=True)

    rows: list[dict] = []
    for ordinal, idx in enumerate(indices, start=1):
        episode_json = per_episode_dir / f"episode_{ordinal:03d}_idx_{idx:04d}.json"
        log_path = per_episode_dir / f"episode_{ordinal:03d}_idx_{idx:04d}.log"
        cmd = [
            sys.executable,
            str(EPISODE_RUNNER),
            "--benchmark_dir",
            str(benchmark_dir),
            "--episode_idx",
            str(idx),
            "--results_json",
            str(episode_json),
            *child_args,
        ]
        print(
            f"\n[molmospaces_arena_batch] Episode {idx} ({ordinal}/{len(indices)}): "
            f"{' '.join(cmd)}",
            flush=True,
        )
        exit_code = _stream_process(cmd, log_path, cwd=work_dir)
        episode_result = _read_episode_result(episode_json)
        row = {
            **_episode_metadata(idx, episode_dicts[idx]),
            "ordinal": int(ordinal),
            "status": "ran" if episode_result is not None else "process_failed",
            "exit_code": exit_code,
            "success": False,
            "step_count": None,
            "log_path": str(log_path),
            "result_json": str(episode_json) if episode_result is not None else None,
        }
        if episode_result is not None:
            episode_rows = episode_result.get("results") or []
            if episode_rows:
                row.update(
                    {
                        "success": bool(episode_rows[0].get("success")),
                        "step_count": episode_rows[0].get("step_count"),
                        "reason": episode_rows[0].get("reason"),
                    }
                )
            else:
                row["reason"] = "missing_episode_result_row"
        else:
            row["reason"] = f"child_exit_{exit_code}"
        rows.append(row)
        print(
            f"[molmospaces_arena_batch] Episode {idx}: "
            f"{'SUCCESS' if row['success'] else 'FAIL'} "
            f"(exit={exit_code}, steps={row.get('step_count')})",
            flush=True,
        )
        if args.stop_on_failure and (exit_code != 0 or not row["success"]):
            break

    success_count = sum(1 for row in rows if row.get("success"))
    payload = {
        "benchmark_dir": str(benchmark_dir),
        "episode_indices": indices,
        "completed_count": len(rows),
        "success_count": success_count,
        "total_count": len(rows),
        "success_rate": (float(success_count) / float(len(rows))) if rows else None,
        "per_episode_dir": str(per_episode_dir),
        "child_args": child_args,
        "results": rows,
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"\n[molmospaces_arena_batch] Complete: {success_count}/{len(rows)} successful "
        f"({100.0 * success_count / len(rows):.1f}%); wrote {results_json}",
        flush=True,
    )
    return 0 if rows and success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
