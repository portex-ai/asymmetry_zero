"""Tests for portex_eval programmatic API."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portex_eval import (
    AgentEvalBundle,
    AgentEvalResults,
    Benchmark,
    EvalResults,
    PortexEvalError,
    ReportPaths,
    create_benchmark,
    create_agent_eval,
)


def _write_benchmark_json(path: Path) -> None:
    """
    Write a predefined benchmark JSON file containing two tasks and their grading criteria.
    
    The file will be created (or overwritten) at the given path using UTF-8 encoding. The payload includes two tasks:
    - Task 1 with a single criterion using grader_type "ExactMatch".
    - Task 2 with a single criterion using grader_type "llm-judge" and a semantic prompt.
    
    Parameters:
        path (Path): Destination file path where the benchmark JSON will be written.
    """
    payload = [
        {
            "task": "Question 1",
            "reference_file": "",
            "criteria": [
                {
                    "id": "q1-exact",
                    "name": "Exact answer",
                    "weight": 100,
                    "grader_type": "ExactMatch",
                    "semanticPrompt": "Answer 1",
                }
            ],
        },
        {
            "task": "Question 2",
            "reference_file": "",
            "criteria": [
                {
                    "id": "q2-semantic",
                    "name": "Semantic correctness",
                    "weight": 100,
                    "grader_type": "llm-judge",
                    "semanticPrompt": "The answer should clearly say Answer 2.",
                }
            ],
        },
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
        assert "answer" not in answers[0]
        assert answers[0]["criteria"][0]["grader_type"] == "ExactMatch"
        assert answers[1]["criteria"][0]["grader_type"] == "llm-judge"


def test_create_benchmark_rejects_non_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "bench.json"
        input_path.write_text(json.dumps({"task": "oops"}), encoding="utf-8")

        with pytest.raises(PortexEvalError, match="must be a list"):
            create_benchmark(str(input_path))


def test_create_benchmark_rejects_missing_criteria() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "bench.json"
        input_path.write_text(
            json.dumps([{"task": "Question 1", "reference_file": "", "criteria": []}]),
            encoding="utf-8",
        )

        with pytest.raises(PortexEvalError, match="criteria must be a non-empty list"):
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


def test_eval_rejects_invalid_max_samples() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tasks.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "prompts": [{"task_id": "task-1", "task_prompt": "Question 1", "reference_file": ""}],
                }
            ),
            encoding="utf-8",
        )
        (bundle_dir / "answers.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": "task-1",
                        "reference_file": "",
                        "tools": [],
                        "criteria": [
                            {
                                "id": "c1",
                                "name": "Exact answer",
                                "weight": 100,
                                "grader_type": "ExactMatch",
                                "semanticPrompt": "Answer 1",
                            }
                        ],
                        "passThreshold": 100,
                    }
                ]
            ),
            encoding="utf-8",
        )

        from portex_eval import eval as run_eval

        with patch("portex_eval.benchmark.run.benchmark_one"):
            with pytest.raises(PortexEvalError, match="max_samples must be at least 1"):
                run_eval(
                    path=str(bundle_dir),
                    judges=["openrouter:openai/gpt-4o"],
                    candidates=["openrouter:openai/gpt-4o-mini"],
                    max_samples=0,
                )


def test_eval_threads_model_config_specs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tasks.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "prompts": [{"task_id": "task-1", "task_prompt": "Question 1", "reference_file": ""}],
                }
            ),
            encoding="utf-8",
        )
        (bundle_dir / "answers.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": "task-1",
                        "reference_file": "",
                        "tools": [],
                        "criteria": [
                            {
                                "id": "c1",
                                "name": "Exact answer",
                                "weight": 100,
                                "grader_type": "ExactMatch",
                                "semanticPrompt": "Answer 1",
                            }
                        ],
                        "passThreshold": 100,
                    }
                ]
            ),
            encoding="utf-8",
        )

        from portex_eval import eval as run_eval

        mock_result = MagicMock(
            run_id="run-1",
            output_dir="/tmp/out",
            eval_log="/tmp/out/log.eval",
        )

        with (
            patch("portex_eval.benchmark.run.benchmark_one", return_value=mock_result) as mock_benchmark,
            patch("portex_eval.reporting.tables.run"),
            patch("portex_eval.rewards.extract_rewards", return_value=[]),
            patch("portex_eval.rewards.write_rewards", return_value="/tmp/out/rl_rewards.json"),
            patch(
                "portex_eval.rewards.write_training_data",
                return_value="/tmp/out/rl_training_data.json",
            ),
            patch("portex_eval.rewards.build_rewards", return_value=MagicMock(task_ids=[])),
        ):
            run_eval(
                path=str(bundle_dir),
                judges=[{"provider": "anthropic", "model": "claude-sonnet-4-5"}],
                candidates=[
                    {
                        "provider": "vllm",
                        "model": "Qwen/Qwen3-VL-4B-Instruct",
                        "base_url": "https://modal.example/v1",
                    }
                ],
            )

        kwargs = mock_benchmark.call_args.kwargs
        assert kwargs["candidate_spec"]["provider"] == "vllm"
        assert kwargs["candidate_spec"]["base_url"] == "https://modal.example/v1"
        assert kwargs["judge_specs"][0]["provider"] == "anthropic"


def test_create_agent_eval_calls_adapter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tasks.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "prompts": [{"task_id": "task-1", "task_prompt": "Question 1", "reference_file": ""}],
                }
            ),
            encoding="utf-8",
        )
        (bundle_dir / "answers.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": "task-1",
                        "criteria": [
                            {
                                "id": "c1",
                                "name": "Exact answer",
                                "weight": 100,
                                "grader_type": "ExactMatch",
                                "semanticPrompt": "Answer 1",
                            }
                        ],
                        "passThreshold": 100,
                    }
                ]
            ),
            encoding="utf-8",
        )

        with patch(
            "portex_eval.api.create_agent_eval_bundle",
            return_value=AgentEvalBundle(
                path="/tmp/agent",
                datasets_dir="/tmp/agent/datasets",
                task_count=1,
            ),
        ) as mock_create:
            result = create_agent_eval(path=str(bundle_dir), output_dir="/tmp/agent")

        mock_create.assert_called_once()
        assert result.task_count == 1


def test_agent_eval_calls_harbor_runner() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir) / "agent"
        (task_root / "datasets").mkdir(parents=True)

        mock_result = AgentEvalResults(
            datasets_dir=str(task_root / "datasets"),
            jobs_dir=str(task_root / "jobs" / "run-1"),
            reports=ReportPaths(
                eval_level=str(task_root / "reports" / "eval_level.csv"),
                task_level=str(task_root / "reports" / "task_level.csv"),
                criterion_level=str(task_root / "reports" / "criterion_level.csv"),
                judgement_level=str(task_root / "reports" / "judgement_level.csv"),
            ),
            run_id="run-1",
            output_dir=str(task_root),
        )

        with (
            patch("portex_eval.api.run_harbor_tasks") as mock_run,
            patch("portex_eval.api.harbor_run_result_to_api", return_value=mock_result),
        ):
            from portex_eval.api import agent_eval

            result = agent_eval(
                task_root=str(task_root),
                judges=[{"provider": "openai", "model": "gpt-4o-mini"}],
                extra_args=["--model", "demo-agent"],
            )

        assert mock_run.call_args.kwargs["judges"][0]["provider"] == "openai"
        assert result.run_id == "run-1"
