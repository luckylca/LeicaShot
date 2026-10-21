from __future__ import annotations

from laika.benchmark import (
    _classify_cgroup_worker_outcome,
    _parse_cgroup_v2_memory_events,
    _parse_macos_peak_rss,
)


def test_parse_cgroup_v2_memory_events() -> None:
    assert _parse_cgroup_v2_memory_events(
        "low 0\nhigh 1\nmax 7\noom 2\noom_kill 1\ninvalid nope\ntruncated"
    ) == {"low": 0, "high": 1, "max": 7, "oom": 2, "oom_kill": 1}


def test_classify_cgroup_worker_outcome() -> None:
    before = {"memory_events": {"oom_kill": 0}}
    after = {"memory_events": {"oom_kill": 1}}

    assert _classify_cgroup_worker_outcome(0, before, after) == "success"
    assert _classify_cgroup_worker_outcome(2, before, after) == "oom"
    assert _classify_cgroup_worker_outcome(2, before, before) == "error"


def test_parse_macos_process_high_water_rss() -> None:
    assert _parse_macos_peak_rss("  601145344  maximum resident set size\n") == (
        601145344
    )
    assert _parse_macos_peak_rss("time output without the metric") is None
