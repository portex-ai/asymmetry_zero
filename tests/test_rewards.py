"""Tests for portex_eval.rewards module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from portex_eval.rewards import extract_rewards, write_rewards


def test_extract_rewards_from_valid_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text(
            "task_id,score,other_col\ntask-1,87.5,foo\ntask-2,100.0,bar\ntask-3,0.0,baz\n",
            encoding="utf-8",
        )

        rewards = extract_rewards(str(csv_path))

        assert len(rewards) == 3
        assert rewards[0] == ("task-1", 87.5)
        assert rewards[1] == ("task-2", 100.0)
        assert rewards[2] == ("task-3", 0.0)


def test_extract_rewards_skips_empty_values() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text(
            "task_id,score\ntask-1,87.5\n,50.0\ntask-3,\ntask-4,75.0\n",
            encoding="utf-8",
        )

        rewards = extract_rewards(str(csv_path))

        assert len(rewards) == 2
        assert rewards[0] == ("task-1", 87.5)
        assert rewards[1] == ("task-4", 75.0)


def test_extract_rewards_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="task_level.csv not found"):
        extract_rewards("/nonexistent/path/task_level.csv")


def test_extract_rewards_missing_task_id_column() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text(
            "id,score\ntask-1,87.5\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing 'task_id' column"):
            extract_rewards(str(csv_path))


def test_extract_rewards_missing_score_column() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text(
            "task_id,points\ntask-1,87.5\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing 'score' column"):
            extract_rewards(str(csv_path))


def test_extract_rewards_invalid_score_value() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text(
            "task_id,score\ntask-1,not_a_number\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid score value 'not_a_number'"):
            extract_rewards(str(csv_path))


def test_extract_rewards_empty_csv() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Empty or invalid CSV"):
            extract_rewards(str(csv_path))


def test_write_rewards_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_rewards.txt"
        task_scores = [
            ("task-1", 87.5),
            ("task-2", 100.0),
            ("task-3", 0.0),
        ]

        result_path = write_rewards(task_scores, str(output_path))

        assert Path(result_path).is_file()
        content = output_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "task-1 87.5"
        assert lines[1] == "task-2 100.0"
        assert lines[2] == "task-3 0.0"


def test_write_rewards_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested" / "dir" / "rl_rewards.txt"
        task_scores = [("task-1", 50.0)]

        result_path = write_rewards(task_scores, str(output_path))

        assert Path(result_path).is_file()


def test_write_rewards_returns_absolute_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_rewards.txt"
        task_scores = [("task-1", 50.0)]

        result_path = write_rewards(task_scores, str(output_path))

        assert Path(result_path).is_absolute()


def test_write_rewards_empty_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_rewards.txt"

        result_path = write_rewards([], str(output_path))

        assert Path(result_path).is_file()
        content = output_path.read_text(encoding="utf-8")
        assert content == ""


def test_write_rewards_space_separated_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_rewards.txt"
        task_scores = [("kljakljsd-aklkjhl-1", 87.5)]

        write_rewards(task_scores, str(output_path))

        content = output_path.read_text(encoding="utf-8")
        assert "\t" not in content
        assert " " in content
        assert content == "kljakljsd-aklkjhl-1 87.5\n"


def test_integration_extract_and_write() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "task_level.csv"
        csv_path.write_text(
            "task_id,score,model_response\n"
            "id-1,90.0,response1\n"
            "id-2,75.5,response2\n"
            "id-3,100.0,response3\n",
            encoding="utf-8",
        )
        output_path = Path(tmpdir) / "rl_rewards.txt"

        task_scores = extract_rewards(str(csv_path))
        result_path = write_rewards(task_scores, str(output_path))

        content = Path(result_path).read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "id-1 90.0"
        assert lines[1] == "id-2 75.5"
        assert lines[2] == "id-3 100.0"
