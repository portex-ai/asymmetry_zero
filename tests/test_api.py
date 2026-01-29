"""Tests for portex_eval programmatic API."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from portex_eval import (
    Benchmark,
    EvalResults,
    PortexEvalError,
    ReportPaths,
    create_benchmark,
)


def _write_benchmark_json(path: Path) -> None:
    payload = [
        {"task": "Question 1", "answer": "Answer 1", "reference_file": ""},
        {"task": "Question 2", "answer": "Answer 2", "reference_file": ""},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_create_benchmark_creates_bundle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "bench.json"
        _write_benchmark_json(input_path)

        benchmark = create_benchmark(str(input_path))
        bundle_path = Path(benchmark.path)
        assert benchmark.task_count == 2
        assert bundle_path.is_dir()

        tasks = json.loads((bundle_path / "tasks.json").read_text(encoding="utf-8"))
        answers = json.loads((bundle_path / "answers.json").read_text(encoding="utf-8"))
        assert tasks["version"] == 2
        assert len(tasks["prompts"]) == 2
        assert len(answers) == 2


def test_create_benchmark_rejects_non_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "bench.json"
        input_path.write_text(json.dumps({"task": "oops"}), encoding="utf-8")

        with pytest.raises(PortexEvalError, match="must be a list"):
            create_benchmark(str(input_path))


def test_benchmark_requires_path() -> None:
    with pytest.raises(ValueError, match="path is required"):
        Benchmark(path="", task_count=0)


def test_report_paths_fields() -> None:
    paths = ReportPaths(
        eval_level="/tmp/eval_level.csv",
        task_level="/tmp/task_level.csv",
        criterion_level="/tmp/criterion_level.csv",
        judgement_level="/tmp/judgement_level.csv",
    )
    assert paths.eval_level.endswith("eval_level.csv")


def test_eval_results_absolute_paths() -> None:
    results = EvalResults(
        logs=["./runs/log.eval"],
        reports=ReportPaths(
            eval_level="./reports/eval_level.csv",
            task_level="./reports/task_level.csv",
            criterion_level="./reports/criterion_level.csv",
            judgement_level="./reports/judgement_level.csv",
        ),
        rewards="./runs/rl_rewards.txt",
        run_id="run-1",
        output_dir="./runs/run-1",
    ).with_absolute_paths()
    assert results.output_dir.startswith("/")
