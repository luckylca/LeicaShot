from __future__ import annotations

from laika.cgroup import memory_stats as _cgroup_memory


def test_cgroup_v2_counters_are_reported(tmp_path) -> None:
    (tmp_path / "memory.current").write_text("1024\n")
    (tmp_path / "memory.peak").write_text("2048\n")
    (tmp_path / "memory.max").write_text("4096\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 1\nmax 0\noom 0\n")

    assert _cgroup_memory(tmp_path) == {
        "version": 2,
        "current_bytes": 1024,
        "peak_bytes": 2048,
        "limit_bytes": 4096,
        "events": {"low": 0, "high": 1, "max": 0, "oom": 0},
    }


def test_cgroup_v1_unlimited_value_is_reported_as_no_limit(tmp_path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory.usage_in_bytes").write_text("1024\n")
    (memory / "memory.max_usage_in_bytes").write_text("2048\n")
    (memory / "memory.limit_in_bytes").write_text("9223372036854771712\n")

    assert _cgroup_memory(tmp_path) == {
        "version": 1,
        "current_bytes": 1024,
        "peak_bytes": 2048,
        "limit_bytes": None,
        "events": {},
    }


def test_cgroup_metrics_are_optional_on_unsupported_hosts(tmp_path) -> None:
    assert _cgroup_memory(tmp_path) is None
