"""
Shared live dashboard JSON for long-running scripts.

Each experiment writes a single ``*.dashboard.json`` file atomically while it runs.
Use ``render_dashboards.py`` to watch many dashboards at once.

Schema (dashboard_version=1)::

    {
      "dashboard_version": 1,
      "task": {"name": "...", "title": "..."},
      "status": {"state": "running", "message": null, "updated_at": "..."},
      "progress": {
        "current": 10, "total": 100, "unit": "houses",
        "completion_pct": 10.0, "elapsed_seconds": 3.2, "eta_seconds": 28.8
      },
      "metrics": {"episodes_kept": 42},
      "sections": [{"title": "By split", "rows": [{"label": "train", "value": "..."}]}]
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pct(current: int, total: int | None) -> float | None:
    if total is None or total <= 0:
        return None
    return round(100.0 * current / total, 2)


def eta_seconds(
    elapsed: float,
    current: int,
    total: int | None,
    *,
    baseline: int = 0,
) -> float | None:
    done_this_run = current - baseline
    if total is None or done_this_run <= 0:
        return None
    remaining = total - current
    if remaining <= 0:
        return 0.0
    return round(elapsed / done_this_run * remaining, 1)


class RunDashboard:
    """Write a versioned dashboard JSON file on a timer or step interval."""

    def __init__(
        self,
        path: Path,
        *,
        name: str,
        title: str | None = None,
        total: int | None = None,
        unit: str = "items",
        baseline_current: int = 0,
        interval: int = 10,
        seconds: float = 2.0,
    ) -> None:
        self.path = Path(path)
        self.name = name
        self.title = title or name
        self.total = total
        self.unit = unit
        self.baseline_current = max(0, baseline_current)
        self.interval = max(1, interval)
        self.seconds = seconds
        self.start_time = time.monotonic()
        self.last_write_time = 0.0
        self.current = 0
        self.state = "running"
        self.message: str | None = None
        self.metrics: dict[str, Any] = {}
        self.sections: list[dict[str, Any]] = []

    def set_total(self, total: int | None) -> None:
        self.total = total

    def set_status(self, state: str, message: str | None = None) -> None:
        self.state = state
        if message is not None:
            self.message = message

    def set_metrics(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def set_sections(self, sections: list[dict[str, Any]]) -> None:
        self.sections = sections

    def mark_progress_baseline(self) -> None:
        """Restart elapsed/ETA timing without changing current progress."""
        self.start_time = time.monotonic()
        self.last_write_time = 0.0

    def update(
        self,
        *,
        current: int | None = None,
        increment: int = 0,
        state: str | None = None,
        message: str | None = None,
        metrics: dict[str, Any] | None = None,
        sections: list[dict[str, Any]] | None = None,
        force: bool = False,
    ) -> None:
        if current is not None:
            self.current = current
        elif increment:
            self.current += increment
        if state is not None:
            self.state = state
        if message is not None:
            self.message = message
        if metrics:
            self.metrics.update(metrics)
        if sections is not None:
            self.sections = sections
        if force or self._should_write():
            self.write()

    def finish(
        self,
        state: str = "complete",
        message: str | None = None,
        *,
        metrics: dict[str, Any] | None = None,
        sections: list[dict[str, Any]] | None = None,
    ) -> None:
        if metrics:
            self.metrics.update(metrics)
        if sections is not None:
            self.sections = sections
        self.set_status(state, message)
        self.write(force=True)

    def _should_write(self) -> bool:
        if self.current > 0 and self.current % self.interval == 0:
            return True
        return time.monotonic() - self.last_write_time >= self.seconds

    def payload(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.start_time
        return {
            "dashboard_version": 1,
            "task": {
                "name": self.name,
                "title": self.title,
            },
            "status": {
                "state": self.state,
                "message": self.message,
                "updated_at": utc_now_iso(),
            },
            "progress": {
                "current": self.current,
                "total": self.total,
                "unit": self.unit,
                "completion_pct": pct(self.current, self.total),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": eta_seconds(
                    elapsed,
                    self.current,
                    self.total,
                    baseline=self.baseline_current,
                ),
            },
            "metrics": self.metrics,
            "sections": self.sections,
        }

    def write(self, *, force: bool = False) -> None:
        del force
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.payload(), f, indent=2, sort_keys=True)
        tmp_path.replace(self.path)
        self.last_write_time = time.monotonic()


def section(title: str, rows: dict[str, Any] | list[tuple[str, Any]]) -> dict[str, Any]:
    if isinstance(rows, dict):
        row_list = [{"label": k, "value": v} for k, v in rows.items()]
    else:
        row_list = [{"label": k, "value": v} for k, v in rows]
    return {"title": title, "rows": row_list}
