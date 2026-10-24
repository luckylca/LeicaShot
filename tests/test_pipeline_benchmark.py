from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_benchmark_script_prefers_checkout_over_stale_pythonpath(tmp_path) -> None:
    stale_package = tmp_path / "stale-site" / "laika"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text(
        "raise RuntimeError('stale image package was imported')\n",
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_processing_pipeline.py"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stale_package.parent)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_process_tree_peak_rss_sampling_interval_is_configurable() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_processing_pipeline.py"
    )
    module = runpy.run_path(str(script))
    peak_type = module["ProcessTreePeak"]

    assert peak_type().interval_seconds == 0.1
    assert peak_type(0.25).interval_seconds == 0.25
    with pytest.raises(ValueError, match="interval must be positive"):
        peak_type(0)
