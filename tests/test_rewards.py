"""Tests for portex_eval.rewards module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    Logprob,
    Logprobs,
    ModelOutput,
    TopLogprob,
)

from portex_eval.rewards import build_training_data, extract_rewards, write_rewards, write_training_data


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
        output_path = Path(tmpdir) / "rl_rewards.json"
        task_scores = [
            ("task-1", 87.5),
            ("task-2", 100.0),
            ("task-3", 0.0),
        ]

        result_path = write_rewards(task_scores, str(output_path))

        assert Path(result_path).is_file()
        assert output_path.read_text(encoding="utf-8") == (
            '{\n'
            '  "task_ids": [\n'
            '    "task-1",\n'
            '    "task-2",\n'
            '    "task-3"\n'
            '  ],\n'
            '  "reward": [\n'
            '    87.5,\n'
            '    100.0,\n'
            '    0.0\n'
            '  ]\n'
            '}'
        )


def test_write_rewards_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested" / "dir" / "rl_rewards.json"
        task_scores = [("task-1", 50.0)]

        result_path = write_rewards(task_scores, str(output_path))

        assert Path(result_path).is_file()


def test_write_rewards_returns_absolute_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_rewards.json"
        task_scores = [("task-1", 50.0)]

        result_path = write_rewards(task_scores, str(output_path))

        assert Path(result_path).is_absolute()


def test_write_rewards_empty_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_rewards.json"

        result_path = write_rewards([], str(output_path))

        assert Path(result_path).is_file()
        content = output_path.read_text(encoding="utf-8")
        assert content == '{\n  "task_ids": [],\n  "reward": []\n}'


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
        output_path = Path(tmpdir) / "rl_rewards.json"

        task_scores = extract_rewards(str(csv_path))
        result_path = write_rewards(task_scores, str(output_path))

        content = Path(result_path).read_text(encoding="utf-8")
        assert '"task_ids"' in content
        assert '"reward"' in content


def test_build_training_data_serializes_prompt_completion_and_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    eval_log = SimpleNamespace(
        samples=[
            SimpleNamespace(
                id="task-1",
                epoch=1,
                metadata={"reference_file": "image.png"},
                messages=[
                    ChatMessageSystem(content="System instructions"),
                    ChatMessageUser(
                        content=[
                            ContentText(text="What is in the image?"),
                            ContentImage(image="attachment://image.png"),
                        ]
                    ),
                    ChatMessageAssistant(content="Answer: cat"),
                ],
                output=ModelOutput(
                    model="openrouter/qwen/qwen3-4b-instruct",
                    completion="Answer: cat",
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(content="Answer: cat"),
                            logprobs=Logprobs(
                                content=[
                                    Logprob(
                                        token="Answer",
                                        logprob=-0.1,
                                        bytes=[65],
                                        top_logprobs=[
                                            TopLogprob(token="Answer", logprob=-0.1, bytes=[65])
                                        ],
                                    ),
                                    Logprob(token=": cat", logprob=-0.2, bytes=[58, 32, 99, 97, 116]),
                                ]
                            ),
                        )
                    ],
                ),
                scores={"portex_scorer": SimpleNamespace(metadata={"task_id": "task-1"})},
            )
        ]
    )

    def _fake_read_eval_log(path: str, header_only: bool = False) -> SimpleNamespace:
        assert path == "/tmp/fake.eval"
        assert header_only is False
        return eval_log

    monkeypatch.setattr("inspect_ai.log.read_eval_log", _fake_read_eval_log)

    payload = build_training_data([("task-1", 100.0)], "/tmp/fake.eval")

    assert payload["format"] == "portex-rl-training-data"
    assert payload["records"][0]["task_id"] == "task-1"
    assert payload["records"][0]["reward"] == 100.0
    assert payload["records"][0]["prompt_token_ids"] is None
    assert payload["records"][0]["completion_token_ids"] is None
    assert payload["records"][0]["completion_logprobs"][0]["token"] == "Answer"
    assert payload["records"][0]["completion_logprobs"][0]["top_logprobs"][0]["token"] == "Answer"
    assert "[image]" in payload["records"][0]["prompt_text"]


def test_write_training_data_creates_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "portex_eval.rewards.writer.build_training_data",
        lambda task_scores, eval_log_path: {
            "format": "portex-rl-training-data",
            "version": 1,
            "source": {"eval_log": eval_log_path},
            "records": [{"task_id": "task-1", "reward": 50.0}],
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "rl_training_data.json"
        result_path = write_training_data([("task-1", 50.0)], "/tmp/fake.eval", str(output_path))
        assert Path(result_path).is_file()
        assert '"format": "portex-rl-training-data"' in output_path.read_text(encoding="utf-8")
