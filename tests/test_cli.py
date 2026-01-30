"""Tests for the CLI module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
                        {"task": "What is 2+2?", "answer": "4"},
                        {"task": "What color is the sky?", "answer": "Blue"},
                    ]
                )
            )

            result = runner.invoke(app, ["format", str(input_path)])

            assert result.exit_code == 0
            assert "Created bundle at:" in result.output
            assert "Task count: 2" in result.output

            output_dir = Path(tmpdir) / "benchmark"
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


class TestMainHelp:
    """Tests for the main CLI help."""

    def test_main_help(self) -> None:
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Lightweight evaluation framework" in result.output
        assert "format" in result.output
        assert "run" in result.output
