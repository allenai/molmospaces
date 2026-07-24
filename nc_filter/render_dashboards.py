#!/usr/bin/env python3
"""
Render one or more live dashboard JSON files from run_dashboard.py (or legacy scans).

Examples::

    python render_dashboards.py experiments/*.dashboard.json
    python render_dashboards.py --watch 2 dashboards/
    watch -n 2 python render_dashboards.py procthor_licenses.dashboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


STATE_STYLES = {
    "running": "bold cyan",
    "scanning": "bold cyan",
    "complete": "bold green",
    "failed": "bold red",
    "error": "bold red",
    "interrupted": "bold yellow",
    "idle": "dim",
}


@dataclass
class NormalizedDashboard:
    source: Path
    name: str
    title: str
    state: str
    message: str | None
    updated_at: str | None
    current: int | None
    total: int | None
    unit: str | None
    completion_pct: float | None
    elapsed_seconds: float | None
    eta_seconds: float | None
    metrics: dict
    sections: list[dict]
    error: str | None = None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize_dashboard(path: Path, data: dict) -> NormalizedDashboard:
    if data.get("dashboard_version") == 1:
        task = data.get("task", {})
        status = data.get("status", {})
        progress = data.get("progress", {})
        return NormalizedDashboard(
            source=path,
            name=task.get("name") or path.stem,
            title=task.get("title") or path.stem,
            state=status.get("state", "unknown"),
            message=status.get("message"),
            updated_at=status.get("updated_at"),
            current=progress.get("current"),
            total=progress.get("total"),
            unit=progress.get("unit"),
            completion_pct=progress.get("completion_pct"),
            elapsed_seconds=progress.get("elapsed_seconds"),
            eta_seconds=progress.get("eta_seconds"),
            metrics=data.get("metrics", {}),
            sections=data.get("sections", []),
        )

    if "progress" in data and "status" in data:
        progress = data["progress"]
        status = data["status"]
        splits = data.get("splits", {})
        split_rows = []
        for split_name in sorted(splits):
            split = splits[split_name]
            split_rows.append(
                (
                    split_name,
                    f"{split.get('indices_scanned', '?')}/{split.get('indices_total', '?')} "
                    f"({split.get('completion_pct', '?')}%) | "
                    f"scenes {split.get('scenes_found', 0)} | "
                    f"NC {split.get('scenes_with_nc_pct', '?')}%",
                )
            )
        nc_stats = progress.get("nc_house_stats", {})
        nc_rows = []
        if nc_stats:
            nc_rows = [
                ("nc_houses", nc_stats.get("nc_houses")),
                ("avg_nc_per_nc_house", nc_stats.get("avg_nc_objects_per_nc_house")),
                ("max_nc_in_one_house", nc_stats.get("max_nc_objects_in_one_house")),
            ]
        sections = []
        if split_rows:
            sections.append(
                {
                    "title": "Splits",
                    "rows": [{"label": k, "value": v} for k, v in split_rows],
                }
            )
        if nc_rows:
            sections.append(
                {
                    "title": "NC house stats",
                    "rows": [{"label": k, "value": v} for k, v in nc_rows],
                }
            )
        train = splits.get("train", {})
        license_pcts = train.get("object_license_pcts", {})
        if license_pcts:
            sections.append(
                {
                    "title": "Object license % (train)",
                    "rows": [
                        {"label": k, "value": f"{v}%"} for k, v in sorted(license_pcts.items())
                    ],
                }
            )
        return NormalizedDashboard(
            source=path,
            name=path.name.replace(".dashboard.json", ""),
            title="procthor license scan",
            state=status.get("state", "unknown"),
            message=status.get("current_split"),
            updated_at=status.get("updated_at"),
            current=progress.get("indices_scanned"),
            total=progress.get("indices_total"),
            unit="indices",
            completion_pct=progress.get("completion_pct"),
            elapsed_seconds=progress.get("elapsed_seconds"),
            eta_seconds=progress.get("eta_seconds"),
            metrics={
                "scenes_found": progress.get("scenes_found"),
                "total_objects": progress.get("total_objects"),
                "avg_objects_per_house": progress.get("avg_objects_per_house"),
            },
            sections=sections,
        )

    return NormalizedDashboard(
        source=path,
        name=path.stem,
        title=path.stem,
        state="unknown",
        message="Unrecognized dashboard schema",
        updated_at=None,
        current=None,
        total=None,
        unit=None,
        completion_pct=None,
        elapsed_seconds=None,
        eta_seconds=None,
        metrics={},
        sections=[],
    )


def read_dashboard(path: Path) -> NormalizedDashboard:
    try:
        data = load_json(path)
        return normalize_dashboard(path, data)
    except Exception as exc:
        return NormalizedDashboard(
            source=path,
            name=path.stem,
            title=path.stem,
            state="error",
            message=str(exc),
            updated_at=None,
            current=None,
            total=None,
            unit=None,
            completion_pct=None,
            elapsed_seconds=None,
            eta_seconds=None,
            metrics={},
            sections=[],
            error=str(exc),
        )


def discover_paths(paths: list[str] | None) -> list[Path]:
    if not paths:
        paths = ["."]
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.glob("*.dashboard.json")))
            found.extend(sorted(path.glob("*dashboard.json")))
        elif path.is_file():
            found.append(path)
        elif path.exists():
            found.append(path)
        else:
            print(f"Warning: path not found, skipping: {path}", file=sys.stderr)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def progress_text(d: NormalizedDashboard) -> str:
    if d.current is None and d.total is None:
        return "-"
    unit = d.unit or "items"
    if d.total is not None:
        pct_suffix = f" ({d.completion_pct}%)" if d.completion_pct is not None else ""
        return f"{d.current or 0}/{d.total} {unit}{pct_suffix}"
    return f"{d.current or 0} {unit}"


def metrics_text(metrics: dict, limit: int = 3) -> str:
    if not metrics:
        return ""
    parts = [f"{k}={metrics[k]}" for k in sorted(metrics)[:limit]]
    return ", ".join(parts)


def render_plain(dashboards: list[NormalizedDashboard]) -> str:
    lines = []
    for d in dashboards:
        lines.append(
            f"{d.source.name} | {d.state} | {progress_text(d)} | "
            f"elapsed {format_duration(d.elapsed_seconds)} | "
            f"eta {format_duration(d.eta_seconds)} | {metrics_text(d.metrics)}"
        )
        for section in d.sections:
            lines.append(f"  [{section.get('title', 'Details')}]")
            for row in section.get("rows", []):
                lines.append(f"    {row.get('label')}: {row.get('value')}")
    return "\n".join(lines)


def render_rich(
    dashboards: list[NormalizedDashboard],
    *,
    max_section_rows: int | None = None,
) -> Group:
    summary = Table(title="Running tasks", expand=True, show_lines=False)
    summary.add_column("Task", style="bold", no_wrap=True)
    summary.add_column("State")
    summary.add_column("Progress")
    summary.add_column("Elapsed", justify="right")
    summary.add_column("ETA", justify="right")
    summary.add_column("Updated", no_wrap=True)
    summary.add_column("Metrics")

    for d in dashboards:
        state_style = STATE_STYLES.get(d.state, "white")
        summary.add_row(
            Text(d.title),
            Text(d.state, style=state_style),
            Text(progress_text(d)),
            Text(format_duration(d.elapsed_seconds)),
            Text(format_duration(d.eta_seconds)),
            Text(d.updated_at or "-"),
            Text(metrics_text(d.metrics, limit=4) or (d.message or "-")),
        )
        if d.message and d.state not in ("running", "scanning"):
            summary.caption = f"{d.title}: {d.message}"

    panels = [Panel(summary, border_style="blue")]
    for d in dashboards:
        if not d.sections:
            continue
        for section in d.sections:
            table = Table(
                title=f"{d.title} — {section.get('title', 'Details')}",
                expand=True,
                show_header=True,
            )
            table.add_column("Label", style="cyan", no_wrap=True)
            table.add_column("Value", overflow="fold")
            rows = section.get("rows", [])
            visible_rows = rows
            omitted = 0
            if max_section_rows is not None and len(rows) > max_section_rows:
                visible_rows = rows[:max_section_rows]
                omitted = len(rows) - max_section_rows
            for row in visible_rows:
                # Rich treats [train] etc. as markup in plain strings — use Text().
                table.add_row(
                    Text(str(row.get("label", "")), style="cyan"),
                    Text(str(row.get("value", ""))),
                )
            if omitted:
                table.add_row(
                    Text("…"),
                    Text(f"({omitted} more rows; use --max-section-rows to cap)"),
                )
            panels.append(Panel(table, border_style="dim"))
    return Group(*panels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render live dashboard JSON files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help=(
            "Dashboard JSON files and/or directories (default: current directory). "
            "Matches *.dashboard.json"
        ),
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Refresh every N seconds (rich live display if available).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Force plain-text output (no rich).",
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="Force rich output even when stdout is not a TTY.",
    )
    parser.add_argument(
        "--max-section-rows",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap on rows shown per section table (default: show all).",
    )
    args = parser.parse_args()

    dashboard_paths = discover_paths(args.paths)
    if not dashboard_paths:
        print(
            "No dashboard files found. Looked for *.dashboard.json in: "
            + ", ".join(args.paths or ["."]),
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Watching {len(dashboard_paths)} dashboard file(s): "
        + ", ".join(p.name for p in dashboard_paths),
        file=sys.stderr,
    )

    use_rich = RICH_AVAILABLE and not args.plain and (args.rich or sys.stdout.isatty())
    if args.rich and not RICH_AVAILABLE:
        print("rich is not installed; use: pip install rich", file=sys.stderr)
        sys.exit(1)

    def snapshot() -> list[NormalizedDashboard]:
        return [read_dashboard(path) for path in dashboard_paths]

    def render() -> Group | str:
        if use_rich:
            return render_rich(
                snapshot(),
                max_section_rows=args.max_section_rows,
            )
        return render_plain(snapshot())

    if args.watch is None:
        output = render()
        if use_rich:
            Console().print(output)
        else:
            print(output)
        return

    if use_rich:
        console = Console()
        with Live(
            console=console,
            refresh_per_second=4,
            screen=False,
            vertical_overflow="visible",
        ) as live:
            while True:
                live.update(
                    render_rich(
                        snapshot(),
                        max_section_rows=args.max_section_rows,
                    )
                )
                time.sleep(args.watch)
    else:
        while True:
            print(render_plain(snapshot()))
            print("-" * 72)
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
