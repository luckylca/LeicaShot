from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_validate_task_count = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "benchmark_api_task.py")
)["_validate_task_count"]


def test_api_benchmark_task_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _validate_task_count(0, 2)


def test_api_benchmark_task_count_cannot_exceed_admission_capacity() -> None:
    with pytest.raises(ValueError, match="exceeds API admission capacity"):
        _validate_task_count(3, 2)


def test_api_benchmark_accepts_tasks_within_admission_capacity() -> None:
    _validate_task_count(2, 2)
