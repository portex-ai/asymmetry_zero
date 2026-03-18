"""Tests for benchmark run provider-mode selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from portex_eval.benchmark.run import benchmark_one


def _write_bundle(bundle_dir: Path) -> None:
    (bundle_dir / "tasks.json").write_text(
        json.dumps({"version": 2, "prompts": [{"task_id": "task-1", "task_prompt": "Q1"}]}),
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
                            "semanticPrompt": "4",
                        }
                    ],
                    "passThreshold": 100,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_benchmark_one_uses_provider_task_for_custom_endpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)

        with (
            patch("portex_eval.benchmark.run.run_inspect_eval", return_value={"log_files": []})
            as mock_run,
            patch("portex_eval.benchmark.run._pick_eval_log", return_value="/tmp/fake.eval"),
        ):
            benchmark_one(
                bundle_dir=str(bundle_dir),
                index_root=tmpdir,
                eval_runs_root=tmpdir,
                candidate_spec={
                    "provider": "vllm",
                    "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "base_url": "https://modal.example/v1",
                },
                judge_specs=[{"provider": "anthropic", "model": "claude-sonnet-4-5"}],
            )

        kwargs = mock_run.call_args.kwargs
        assert kwargs["task_spec"].endswith("@portex_qa_eval_with_providers")
        assert kwargs["model"] is None
        assert "PORTEX_CANDIDATE_CONFIG" in kwargs["extra_env"]
        assert "PORTEX_JUDGE_CONFIGS" in kwargs["extra_env"]


def test_benchmark_one_threads_provider_logprob_env() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)

        with (
            patch("portex_eval.benchmark.run.run_inspect_eval", return_value={"log_files": []})
            as mock_run,
            patch("portex_eval.benchmark.run._pick_eval_log", return_value="/tmp/fake.eval"),
        ):
            benchmark_one(
                bundle_dir=str(bundle_dir),
                index_root=tmpdir,
                eval_runs_root=tmpdir,
                candidate_spec={
                    "provider": "vllm",
                    "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "base_url": "https://modal.example/v1",
                },
                judge_specs=[{"provider": "anthropic", "model": "claude-sonnet-4-5"}],
                logprobs=True,
                top_logprobs=5,
            )

        extra_env = mock_run.call_args.kwargs["extra_env"]
        assert extra_env["PORTEX_LOGPROBS"] == "true"
        assert extra_env["PORTEX_TOP_LOGPROBS"] == "5"


def test_benchmark_one_keeps_inspect_path_for_plain_openrouter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)

        with (
            patch("portex_eval.benchmark.run.run_inspect_eval", return_value={"log_files": []})
            as mock_run,
            patch("portex_eval.benchmark.run._pick_eval_log", return_value="/tmp/fake.eval"),
        ):
            benchmark_one(
                bundle_dir=str(bundle_dir),
                index_root=tmpdir,
                eval_runs_root=tmpdir,
                candidate_spec="openrouter:openai/gpt-4o-mini",
                judge_specs=["openrouter:openai/gpt-4o"],
            )

        kwargs = mock_run.call_args.kwargs
        assert kwargs["task_spec"].endswith("@portex_qa_eval")
        assert kwargs["model"] == "openrouter/openai/gpt-4o-mini"
        assert kwargs["extra_env"]["PORTEX_JUDGE_MODELS"] == "openrouter/openai/gpt-4o"
