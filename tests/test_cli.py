"""Tests for the CLI module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from portex_eval.cli import app

runner = CliRunner()


class TestFormatCommand:
    """Tests for the format command."""

    def test_format_help(self) -> None:
        """Test that format --help works."""
        result = runner.invoke(app, ["format", "--help"])
        assert result.exit_code == 0
        assert "Format a benchmark.json file" in result.output
        assert "INPUT_PATH" in result.output

    def test_format_success(self) -> None:
        """Test successful bundle creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "benchmark.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "task": "What is 2+2?",
                            "criteria": [
                                {
                                    "id": "math-exact",
                                    "name": "Exact answer",
                                    "weight": 100,
                                    "grader_type": "ExactMatch",
                                    "semanticPrompt": "4",
                                }
                            ],
                        },
                        {
                            "task": "What color is the sky?",
                            "criteria": [
                                {
                                    "id": "sky-answer",
                                    "name": "Expected answer",
                                    "weight": 100,
                                    "grader_type": "llm-judge",
                                    "semanticPrompt": "The answer should say the sky is blue.",
                                }
                            ],
                        },
                    ]
                )
            )

            result = runner.invoke(app, ["format", str(input_path)])

            assert result.exit_code == 0
            assert "Created bundle at:" in result.output
            assert "Task count: 2" in result.output

            created_path = result.output.split("Created bundle at: ", 1)[1].splitlines()[0].strip()
            output_dir = Path(created_path)
            assert output_dir.is_dir()
            assert (output_dir / "tasks.json").is_file()
            assert (output_dir / "answers.json").is_file()

    def test_format_nonexistent_file(self) -> None:
        """Test error handling for nonexistent file."""
        result = runner.invoke(app, ["format", "/nonexistent/file.json"])
        assert result.exit_code == 2
        assert "does not" in result.output and "exist" in result.output

    def test_format_invalid_json(self) -> None:
        """Test error handling for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "invalid.json"
            input_path.write_text("not valid json")

            result = runner.invoke(app, ["format", str(input_path)])

            assert result.exit_code == 1
            assert "Error:" in result.output


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self) -> None:
        """Test that run --help works."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run an evaluation benchmark" in result.output
        assert "--bundle" in result.output
        assert "--judge" in result.output
        assert "--candidate" in result.output

    def test_run_missing_bundle(self) -> None:
        """Test error when bundle is missing."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "--bundle is required" in result.output

    def test_run_missing_judge(self) -> None:
        """Test error when judge is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["run", "--bundle", tmpdir])
            assert result.exit_code == 1
            assert "--judge is required" in result.output

    def test_run_missing_candidate(self) -> None:
        """Test error when candidate is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["run", "--bundle", tmpdir, "--judge", "openrouter/openai/gpt-4o"]
            )
            assert result.exit_code == 1
            assert "--candidate is required" in result.output

    def test_run_passes_max_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_result = MagicMock()
            mock_result.run_id = "run-1"
            mock_result.output_dir = "/tmp/out"
            mock_result.logs = []
            mock_result.reports = None
            mock_result.rewards_path = ""
            mock_result.rewards = MagicMock(task_ids=[])

            with patch("portex_eval.cli.run_eval", return_value=mock_result) as mock_run_eval:
                result = runner.invoke(
                    app,
                    [
                        "run",
                        "--bundle",
                        tmpdir,
                        "--judge",
                        "openrouter:openai/gpt-4o",
                        "--candidate",
                        "openrouter:openai/gpt-4o-mini",
                        "--max-samples",
                        "4",
                    ],
                )

            assert result.exit_code == 0
            mock_run_eval.assert_called_once()
            assert mock_run_eval.call_args.kwargs["max_samples"] == 4

    def test_run_passes_logprob_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_result = MagicMock()
            mock_result.run_id = "run-1"
            mock_result.output_dir = "/tmp/out"
            mock_result.logs = []
            mock_result.reports = None
            mock_result.rewards_path = ""
            mock_result.training_data_path = ""
            mock_result.rewards = MagicMock(task_ids=[])

            with patch("portex_eval.cli.run_eval", return_value=mock_result) as mock_run_eval:
                result = runner.invoke(
                    app,
                    [
                        "run",
                        "--bundle",
                        tmpdir,
                        "--judge",
                        "openrouter:openai/gpt-4o",
                        "--candidate",
                        "openrouter:openai/gpt-4o-mini",
                        "--logprobs",
                        "--top-logprobs",
                        "5",
                    ],
                )

            assert result.exit_code == 0
            mock_run_eval.assert_called_once()
            assert mock_run_eval.call_args.kwargs["logprobs"] is True
            assert mock_run_eval.call_args.kwargs["top_logprobs"] == 5


class TestMainHelp:
    """Tests for the main CLI help."""

    def test_main_help(self) -> None:
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Lightweight evaluation framework" in result.output
        assert "format" in result.output
        assert "run" in result.output
        assert "agent-create" in result.output
        assert "agent-run" in result.output


class TestAgentCommands:
    def test_agent_create_help(self) -> None:
        result = runner.invoke(app, ["agent-create", "--help"])
        assert result.exit_code == 0
        assert "Convert a Portex bundle into Harbor task directories" in result.output

    def test_agent_create_calls_api(self) -> None:
        """
        Verifies the `agent-create` CLI command calls the agent creation API and prints the created task root.
        
        Creates a temporary bundle directory, patches `create_agent_eval` to return a mock result, invokes the `agent-create` command with bundle and output paths, and asserts the command exits successfully, `create_agent_eval` was called once, and the output contains "Task root:".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_dir.mkdir()

            mock_result = MagicMock(path="/tmp/agent", datasets_dir="/tmp/agent/datasets", task_count=1)
            with patch("portex_eval.cli.create_agent_eval", return_value=mock_result) as mock_create:
                result = runner.invoke(
                    app,
                    [
                        "agent-create",
                        "--bundle",
                        str(bundle_dir),
                        "--output",
                        str(Path(tmpdir) / "agent-out"),
                    ],
                )

            assert result.exit_code == 0
            mock_create.assert_called_once()
            assert "Task root:" in result.output

    def test_agent_run_calls_api(self) -> None:
        """
        Verifies the agent-run CLI command forwards judge identifiers and trailing extra arguments to the agent runner API.
        
        Sets up a temporary agent task root, patches `run_agent_eval` to return a mock result, invokes the CLI with a `--judge` and a `--` separator followed by extra arguments, and asserts the command exits successfully and that `run_agent_eval` received the expected `judges` list and `extra_args`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir) / "agent"
            task_root.mkdir()

            mock_result = MagicMock()
            mock_result.run_id = "run-1"
            mock_result.output_dir = "/tmp/out"
            mock_result.datasets_dir = "/tmp/out/datasets"
            mock_result.jobs_dir = "/tmp/out/jobs/run-1"
            mock_result.reports = None
            mock_result.rewards_path = ""
            mock_result.training_data_path = ""
            mock_result.rewards = MagicMock(task_ids=[])

            with patch("portex_eval.cli.run_agent_eval", return_value=mock_result) as mock_run:
                result = runner.invoke(
                    app,
                    [
                        "agent-run",
                        "--tasks",
                        str(task_root),
                        "--judge",
                        "openrouter:openai/gpt-4o-mini",
                        "--",
                        "--model",
                        "demo-agent",
                    ],
                )

            assert result.exit_code == 0
            assert mock_run.call_args.kwargs["judges"] == ["openrouter:openai/gpt-4o-mini"]
            assert mock_run.call_args.kwargs["extra_args"] == ["--model", "demo-agent"]
