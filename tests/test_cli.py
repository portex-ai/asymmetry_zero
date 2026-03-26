"""Tests for the CLI module."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from portex_eval.cli import app

runner = CliRunner()
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain_output(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


class TestFormatCommand:
    """Tests for the format command."""

    def test_format_help(self) -> None:
        """Test that format --help works."""
        result = runner.invoke(app, ["format", "--help"], color=False)
        output = _plain_output(result.output)
        assert result.exit_code == 0
        assert "Format a benchmark.json file" in output
        assert "INPUT_PATH" in output

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
        result = runner.invoke(app, ["run", "--help"], color=False)
        output = _plain_output(result.output)
        assert result.exit_code == 0
        assert "Run an evaluation benchmark" in output
        assert "--bundle" in output
        assert "--judge" in output
        assert "--candidate" in output

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
        result = runner.invoke(app, ["--help"], color=False)
        output = _plain_output(result.output)
        assert result.exit_code == 0
        assert "Lightweight evaluation framework" in output
        assert "format" in output
        assert "run" in output
        assert "agent-create" in output
        assert "agent-run" in output


class TestAgentCommands:
    def test_agent_create_help(self) -> None:
        result = runner.invoke(app, ["agent-create", "--help"], color=False)
        output = _plain_output(result.output)
        assert result.exit_code == 0
        assert "Convert a Portex bundle into Harbor task directories" in output

    def test_agent_create_calls_api(self) -> None:
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

    def test_agent_run_loads_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task_root = tmp_path / "agent"
            task_root.mkdir()
            config_path = tmp_path / "run_spec.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        f"tasks: {task_root.name}",
                        "output: jobs",
                        "judges:",
                        "  - openrouter:openai/gpt-4o-mini",
                        "n_concurrent: 4",
                        "env: modal",
                        "overwrite: false",
                        "harbor_args:",
                        "  - --agent",
                        "  - portex-multimodal",
                        "  - --model",
                        "  - demo-agent",
                    ]
                ),
                encoding="utf-8",
            )

            mock_result = MagicMock()
            mock_result.run_id = "run-1"
            mock_result.output_dir = "/tmp/out"
            mock_result.datasets_dir = "/tmp/out/datasets"
            mock_result.jobs_dir = "/tmp/out/jobs/run-1"

            with patch("portex_eval.cli.run_agent_eval", return_value=mock_result) as mock_run:
                result = runner.invoke(
                    app,
                    ["agent-run", "--config", str(config_path)],
                )

            assert result.exit_code == 0
            assert mock_run.call_args.kwargs["task_root"] == str(task_root.resolve())
            assert mock_run.call_args.kwargs["output_dir"] == str((tmp_path / "jobs").resolve())
            assert mock_run.call_args.kwargs["judges"] == ["openrouter:openai/gpt-4o-mini"]
            assert mock_run.call_args.kwargs["n_concurrent"] == 4
            assert mock_run.call_args.kwargs["env"] == "modal"
            assert mock_run.call_args.kwargs["overwrite"] is False
            assert mock_run.call_args.kwargs["extra_args"] == [
                "--agent",
                "portex-multimodal",
                "--model",
                "demo-agent",
            ]

    def test_agent_run_cli_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task_root = tmp_path / "agent"
            task_root.mkdir()
            override_root = tmp_path / "override-agent"
            override_root.mkdir()
            config_path = tmp_path / "run_spec.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        f"tasks: {task_root.name}",
                        "judges:",
                        "  - openrouter:openai/gpt-4o-mini",
                        "env: modal",
                        "harbor_args:",
                        "  - --model",
                        "  - from-config",
                    ]
                ),
                encoding="utf-8",
            )

            mock_result = MagicMock()
            mock_result.run_id = "run-1"
            mock_result.output_dir = "/tmp/out"
            mock_result.datasets_dir = "/tmp/out/datasets"
            mock_result.jobs_dir = "/tmp/out/jobs/run-1"

            with patch("portex_eval.cli.run_agent_eval", return_value=mock_result) as mock_run:
                result = runner.invoke(
                    app,
                    [
                        "agent-run",
                        "--config",
                        str(config_path),
                        "--tasks",
                        str(override_root),
                        "--judge",
                        "openrouter:google/gemini-2.5-flash",
                        "--env",
                        "docker",
                        "--overwrite",
                        "--",
                        "--model",
                        "from-cli",
                    ],
                )

            assert result.exit_code == 0
            assert mock_run.call_args.kwargs["task_root"] == str(override_root)
            assert mock_run.call_args.kwargs["judges"] == [
                "openrouter:openai/gpt-4o-mini",
                "openrouter:google/gemini-2.5-flash",
            ]
            assert mock_run.call_args.kwargs["env"] == "docker"
            assert mock_run.call_args.kwargs["overwrite"] is True
            assert mock_run.call_args.kwargs["extra_args"] == [
                "--model",
                "from-config",
                "--model",
                "from-cli",
            ]
