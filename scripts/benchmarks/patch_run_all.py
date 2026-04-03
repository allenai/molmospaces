import subprocess
import argparse


def detect_task_type(path: str) -> str | None:
    p = path.lower()
    if "close" in p:
        return "close"
    if "open" in p or "opening" in p:
        return "open"
    if "pick" in p or "pnp" in p:
        return "pick"
    return None


def find_benchmarks(root: str) -> list[str]:
    result = subprocess.run(
        ["find", root, "-iname", "benchmark.json"],
        capture_output=True, text=True, check=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_update_cmd(json_file: str, dry_run: bool = False) -> list[str]:
    cmd = ["python", "update_task_descriptions.py", json_file]
    task_type = detect_task_type(json_file)
    if task_type:
        cmd += ["--task-type", task_type]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def build_patch_cmd(root: str, dry_run: bool = False) -> list[str]:
    cmd = ["python", "patch_benchmarks.py", "--benchmarks_dir", root]
    if dry_run:
        cmd.append("--dry_run")
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory to search for benchmark.json files")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to all calls")
    parser.add_argument("--print-only", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    benchmarks = find_benchmarks(args.root)
    print(f"Found {len(benchmarks)} benchmark files\n")

    # --- patch_benchmarks.py (once, on the root dir) ---
    patch_cmd = build_patch_cmd(args.root, dry_run=args.dry_run)
    print("# patch_benchmarks.py")
    print(" ".join(patch_cmd))
    if not args.print_only:
        subprocess.run(patch_cmd, check=True)

    print()

    # --- update_task_descriptions.py (once per benchmark.json) ---
    print("# update_task_descriptions.py")
    for bench in benchmarks:
        cmd = build_update_cmd(bench, dry_run=args.dry_run)
        print(" ".join(cmd))
        if not args.print_only:
            subprocess.run(cmd, check=True)