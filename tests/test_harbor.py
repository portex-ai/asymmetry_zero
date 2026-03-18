"""Tests for Harbor-backed agent eval support."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from portex_eval.benchmark.harbor.adapter import create_agent_eval_bundle
from portex_eval.benchmark.harbor.results import write_harbor_artifacts
from portex_eval.benchmark.harbor.run import run_harbor_tasks


def _write_bundle(bundle_dir: Path) -> None:
    refs_dir = bundle_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "diagram.txt").write_text("reference", encoding="utf-8")
    (bundle_dir / "tasks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "prompts": [
                    {
                        "task_id": "task-1",
                        "task_prompt": "Use the reference file and answer the question.",
                        "reference_file": "diagram.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "answers.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "reference_file": "diagram.txt",
                    "tools": ["bash"],
                    "criteria": [
                        {
                            "id": "c1",
                            "name": "Exact answer",
                            "weight": 100,
                            "grader_type": "ExactMatch",
                            "semanticPrompt": "diagram",
                        }
                    ],
                    "passThreshold": 100,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_create_agent_eval_bundle_generates_harbor_task_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)

        result = create_agent_eval_bundle(
            bundle_dir=str(bundle_dir),
            output_dir=str(Path(tmpdir) / "agent-tasks"),
        )

        task_dir = Path(result.datasets_dir) / "portex_task-1"
        assert result.task_count == 1
        assert (task_dir / "task.toml").is_file()
        assert (task_dir / "instruction.md").is_file()
        assert (task_dir / "tests" / "portex_grade.py").is_file()
        assert (task_dir / "tests" / "runtime" / "portex_eval" / "grading" / "core.py").is_file()
        assert (task_dir / "environment" / "refs" / "diagram.txt").is_file()

        task_config = json.loads((task_dir / "tests" / "task_config.json").read_text(encoding="utf-8"))
        task_toml = (task_dir / "task.toml").read_text(encoding="utf-8")
        assert task_config["task_id"] == "task-1"
        assert task_config["reference_file"] == "diagram.txt"
        assert task_config["judge_models"]
        assert 'OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}"' in task_toml
        assert "OPENAI_API_KEY" not in task_toml
        assert "ANTHROPIC_API_KEY" not in task_toml


def test_run_harbor_tasks_builds_command_and_env() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as mock_subprocess,
            patch(
                "portex_eval.benchmark.harbor.results.write_harbor_artifacts",
                return_value=(
                    ("eval.csv", "task.csv", "criterion.csv", "judgement.csv"),
                    type("RewardsPayload", (), {"task_ids": ["task-1"], "reward": [100.0]})(),
                    "rl_rewards.json",
                    "rl_training_data.json",
                ),
            ) as mock_artifacts,
        ):
            result = run_harbor_tasks(
                task_root=str(task_root),
                judges=[{"provider": "openai", "model": "gpt-4o-mini"}],
                n_concurrent=2,
                env="local",
                extra_args=["--model", "demo-agent"],
            )

        cmd = mock_subprocess.call_args.args[0]
        env = mock_subprocess.call_args.kwargs["env"]
        assert cmd[:4] == [mock_subprocess.call_args.args[0][0], "-m", "harbor.cli.main", "run"]
        assert "--n-concurrent" in cmd
        assert "--env" in cmd
        assert env["PORTEX_JUDGE_MODELS"] == "openai:gpt-4o-mini"
        assert "PORTEX_JUDGE_CONFIGS" in env
        assert result.reports.eval_level == "eval.csv"
        mock_artifacts.assert_called_once()


def test_run_harbor_tasks_materializes_judge_api_keys() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as mock_subprocess,
            patch(
                "portex_eval.benchmark.harbor.results.write_harbor_artifacts",
                return_value=(
                    ("eval.csv", "task.csv", "criterion.csv", "judgement.csv"),
                    type("RewardsPayload", (), {"task_ids": ["task-1"], "reward": [100.0]})(),
                    "rl_rewards.json",
                    "rl_training_data.json",
                ),
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "openai-test-key"}, clear=False),
        ):
            run_harbor_tasks(
                task_root=str(task_root),
                judges=[{"provider": "openai", "model": "gpt-4o-mini"}],
            )

        env = mock_subprocess.call_args.kwargs["env"]
        judge_configs = json.loads(env["PORTEX_JUDGE_CONFIGS"])
        assert judge_configs[0]["provider"] == "openai"
        assert judge_configs[0]["model"] == "gpt-4o-mini"
        assert judge_configs[0]["api_key"] == "openai-test-key"
        assert "api_key_env" not in judge_configs[0]


def test_run_harbor_tasks_requires_harbor_install() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ModuleNotFoundError, match="uv sync --group harbor"):
                run_harbor_tasks(task_root=str(task_root))


def test_write_harbor_artifacts_emits_reports_and_rl_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "agent-run"
        jobs_dir = output_dir / "jobs" / "run-1" / "task-1" / "logs" / "verifier"
        jobs_dir.mkdir(parents=True)
        detail = {
            "task_id": "task-1",
            "question": "What is in the file?",
            "submission": "Answer: diagram",
            "reference_file": "diagram.txt",
            "pass_threshold": 100,
            "total_score": 1.0,
            "total_score_raw": 100.0,
            "passed": True,
            "grade": "C",
            "judge_names": ["ExactMatch"],
            "criteria_results": [
                {
                    "criterion_id": "c1",
                    "name": "Exact answer",
                    "prompt": "diagram",
                    "semanticPrompt": "diagram",
                    "grader_type": "ExactMatch",
                    "weight": 100,
                    "grade": "C",
                    "passed": True,
                    "awarded": 100.0,
                    "judges": [
                        {
                            "model": "ExactMatch",
                            "grade": "C",
                            "passed": True,
                            "awarded": 100.0,
                            "explanation": "matched",
                        }
                    ],
                }
            ],
            "error": None,
        }
        (jobs_dir / "portex_detail.json").write_text(json.dumps(detail), encoding="utf-8")

        report_paths, rewards_payload, rewards_path, training_data_path = write_harbor_artifacts(
            jobs_dir=str(output_dir / "jobs"),
            output_dir=str(output_dir),
            run_id="run-1",
            datasets_dir=str(output_dir / "datasets"),
            agent_model="demo-agent",
            harbor_args=["--model", "demo-agent"],
        )

        assert Path(report_paths[0]).is_file()
        assert Path(report_paths[1]).is_file()
        assert Path(report_paths[2]).is_file()
        assert Path(report_paths[3]).is_file()
        assert Path(rewards_path).is_file()
        assert Path(training_data_path).is_file()
        assert rewards_payload.task_ids == ["task-1"]

        training_data = json.loads(Path(training_data_path).read_text(encoding="utf-8"))
        assert training_data["records"][0]["completion"] == "Answer: diagram"
