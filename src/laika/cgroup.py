"""Helpers for reading Linux cgroup memory counters."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _read_value(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value in {"", "max"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def memory_stats(root: Path | None = None) -> dict[str, Any] | None:
    """Read current, peak, limit and OOM event counters for this cgroup."""
    cgroup_v2 = root or Path("/sys/fs/cgroup")
    if (cgroup_v2 / "memory.peak").is_file():
        events: dict[str, int] = {}
        try:
            for line in (
                (cgroup_v2 / "memory.events").read_text(encoding="ascii").splitlines()
            ):
                key, value = line.split(maxsplit=1)
                events[key] = int(value)
        except (OSError, ValueError):
            events = {}
        return {
            "version": 2,
            "current_bytes": _read_value(cgroup_v2 / "memory.current"),
            "peak_bytes": _read_value(cgroup_v2 / "memory.peak"),
            "limit_bytes": _read_value(cgroup_v2 / "memory.max"),
            "events": events,
        }

    cgroup_v1 = root / "memory" if root is not None else Path("/sys/fs/cgroup/memory")
    if (cgroup_v1 / "memory.max_usage_in_bytes").is_file():
        limit = _read_value(cgroup_v1 / "memory.limit_in_bytes")
        if limit is not None and limit >= 1 << 60:
            limit = None
        return {
            "version": 1,
            "current_bytes": _read_value(cgroup_v1 / "memory.usage_in_bytes"),
            "peak_bytes": _read_value(cgroup_v1 / "memory.max_usage_in_bytes"),
            "limit_bytes": limit,
            "events": {},
        }
    return None
