"""Tests for portex_eval.bundle.formatter module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portex_eval import PortexEvalError
from portex_eval.bundle.formatter import (
    FormattedBundle,
    _normalize_criteria_weights,
    _parse_criteria_response,
    format_bundle,
    induce_criteria,
)


@pytest.fixture
def sample_bundle(tmp_path: Path) -> Path:
    """
    Create a sample input bundle on disk for tests.
    
    Writes a directory named "input_bundle" under `tmp_path` containing:
    - tasks.json: two tasks ("task-1" and "task-2") with prompts and reference_file entries.
    - answers.json: corresponding answer entries where each entry includes a single criterion (with `id`, `name`, `weight`, `grader_type` set to "ExactMatch", and `semanticPrompt`) and a `passThreshold`.
    - refs/math.txt: a reference file for the math task.
    
    Parameters:
        tmp_path (Path): Base temporary directory (pytest tmp_path fixture).
    
    Returns:
        Path: Path to the created "input_bundle" directory.
    """
    bundle_dir = tmp_path / "input_bundle"
    bundle_dir.mkdir()

    tasks = [
        {
            "task_id": "task-1",
            "task_prompt": "What is the capital of France?",
            "reference_file": "",
        },
        {
            "task_id": "task-2",
            "task_prompt": "Calculate 2 + 2",
            "reference_file": "math.txt",
        },
    ]

    answers = [
        {
            "task_id": "task-1",
            "reference_file": "",
            "tools": [],
            "criteria": [
                {
                    "id": "task-1-c1",
                    "name": "Exact capital",
                    "weight": 100,
                    "grader_type": "ExactMatch",
                    "semanticPrompt": "Paris",
                }
            ],
            "passThreshold": 100,
        },
        {
            "task_id": "task-2",
            "reference_file": "math.txt",
            "tools": [],
            "criteria": [
                {
                    "id": "task-2-c1",
                    "name": "Exact sum",
                    "weight": 100,
                    "grader_type": "ExactMatch",
                    "semanticPrompt": "4",
                }
            ],
            "passThreshold": 100,
        },
    ]

    (bundle_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (bundle_dir / "answers.json").write_text(json.dumps(answers), encoding="utf-8")

    refs_dir = bundle_dir / "refs"
    refs_dir.mkdir()
    (refs_dir / "math.txt").write_text("Reference for math task", encoding="utf-8")

    return bundle_dir


@pytest.fixture
def sample_bundle_v2(tmp_path: Path) -> Path:
    """
    Create a temporary v2-format input bundle for tests.
    
    Creates a directory containing:
    - tasks.json: a version 2 tasks file with one prompt (task-1).
    - answers.json: one answer entry for task-1 with a single criterion (`id`: "correctness", `grader_type`: "llm-judge", weight 100) and `passThreshold` 100.
    - refs/: an empty references directory.
    
    Parameters:
        tmp_path (Path): pytest temporary path used as the parent for the bundle.
    
    Returns:
        Path: Path to the created bundle directory.
    """
    bundle_dir = tmp_path / "input_bundle_v2"
    bundle_dir.mkdir()

    tasks = {
        "version": 2,
        "prompts": [
            {
                "task_id": "task-1",
                "task_prompt": "What is the capital of France?",
                "reference_file": "",
            },
        ],
    }

    answers = [
        {
            "task_id": "task-1",
            "reference_file": "",
            "tools": [],
            "criteria": [
                {
                    "id": "correctness",
                    "name": "Correctness",
                    "weight": 100,
                    "grader_type": "llm-judge",
                    "semanticPrompt": "The answer is correct",
                }
            ],
            "passThreshold": 100,
        },
    ]

    (bundle_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (bundle_dir / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    (bundle_dir / "refs").mkdir()

    return bundle_dir


@pytest.fixture
def answer_only_bundle(tmp_path: Path) -> Path:
    """
    Create a temporary evaluation bundle containing an answer with no criteria to trigger criteria induction.
    
    Creates a directory named "answer_only_bundle" under the provided tmp_path and writes:
    - tasks.json with a single task prompt,
    - answers.json with a matching answer whose "criteria" list is empty and "passThreshold" set to 100,
    - an empty refs/ directory.
    
    Parameters:
        tmp_path (Path): Base temporary directory in which the bundle directory will be created.
    
    Returns:
        Path: Path to the created bundle directory.
    """
    bundle_dir = tmp_path / "answer_only_bundle"
    bundle_dir.mkdir()

    tasks = [
        {
            "task_id": "task-1",
            "task_prompt": "What is the capital of France?",
            "reference_file": "",
        }
    ]
    answers = [
        {
            "task_id": "task-1",
            "answer": "Paris",
            "reference_file": "",
            "tools": [],
            "criteria": [],
            "passThreshold": 100,
        }
    ]

    (bundle_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (bundle_dir / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    (bundle_dir / "refs").mkdir()
    return bundle_dir


class TestFormatBundle:
    """Tests for format_bundle function."""

    def test_format_bundle_basic(self, sample_bundle: Path, tmp_path: Path) -> None:
        """Test basic bundle formatting without criteria induction."""
        output_dir = tmp_path / "output_bundle"

        result = format_bundle(sample_bundle, output_dir)

        assert isinstance(result, FormattedBundle)
        assert result.task_count == 2
        assert result.criteria_induced is False
        assert Path(result.output_dir).is_dir()

        tasks_path = Path(result.output_dir) / "tasks.json"
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert tasks["version"] == 2
        assert len(tasks["prompts"]) == 2
        assert tasks["prompts"][0]["task_id"] == "task-1"

        answers_path = Path(result.output_dir) / "answers.json"
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        assert len(answers) == 2
        assert answers[0]["task_id"] == "task-1"
        assert "answer" not in answers[0]
        assert answers[0]["criteria"][0]["grader_type"] == "ExactMatch"

    def test_format_bundle_copies_refs(self, sample_bundle: Path, tmp_path: Path) -> None:
        """Test that reference files are copied."""
        output_dir = tmp_path / "output_bundle"

        format_bundle(sample_bundle, output_dir)

        refs_file = output_dir / "refs" / "math.txt"
        assert refs_file.is_file()
        assert refs_file.read_text(encoding="utf-8") == "Reference for math task"

    def test_format_bundle_v2_input(self, sample_bundle_v2: Path, tmp_path: Path) -> None:
        """Test formatting a v2 input bundle."""
        output_dir = tmp_path / "output_bundle"

        result = format_bundle(sample_bundle_v2, output_dir)

        assert result.task_count == 1

        tasks_path = Path(result.output_dir) / "tasks.json"
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert tasks["version"] == 2

    def test_format_bundle_preserves_existing_criteria(
        self, sample_bundle_v2: Path, tmp_path: Path
    ) -> None:
        """Test that existing criteria are preserved."""
        output_dir = tmp_path / "output_bundle"

        result = format_bundle(sample_bundle_v2, output_dir, induce_criteria_flag=False)

        answers_path = Path(result.output_dir) / "answers.json"
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        assert len(answers[0]["criteria"]) == 1
        assert answers[0]["criteria"][0]["id"] == "correctness"

    def test_format_bundle_rejects_missing_input(self, tmp_path: Path) -> None:
        """Test that missing input directory raises error."""
        with pytest.raises(PortexEvalError, match="Input bundle directory not found"):
            format_bundle(tmp_path / "nonexistent", tmp_path / "output")

    def test_format_bundle_rejects_nonempty_output(
        self, sample_bundle: Path, tmp_path: Path
    ) -> None:
        """Test that non-empty output directory raises error."""
        output_dir = tmp_path / "output_bundle"
        output_dir.mkdir()
        (output_dir / "existing.txt").write_text("existing", encoding="utf-8")

        with pytest.raises(PortexEvalError, match="Output directory is not empty"):
            format_bundle(sample_bundle, output_dir)

    def test_format_bundle_rejects_invalid_tasks(self, tmp_path: Path) -> None:
        """Test that invalid tasks.json raises error."""
        bundle_dir = tmp_path / "bad_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tasks.json").write_text('{"not": "a list"}', encoding="utf-8")
        (bundle_dir / "answers.json").write_text("[]", encoding="utf-8")

        with pytest.raises(PortexEvalError, match="prompts must be a list"):
            format_bundle(bundle_dir, tmp_path / "output")

    def test_format_bundle_rejects_mismatched_task_ids(self, tmp_path: Path) -> None:
        """Test that answer referencing unknown task_id raises error."""
        bundle_dir = tmp_path / "mismatch_bundle"
        bundle_dir.mkdir()

        tasks = [{"task_id": "task-1", "task_prompt": "Test", "reference_file": ""}]
        answers = [{"task_id": "wrong-id", "answer": "Answer"}]

        (bundle_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
        (bundle_dir / "answers.json").write_text(json.dumps(answers), encoding="utf-8")

        with pytest.raises(PortexEvalError, match="unknown task_id"):
            format_bundle(bundle_dir, tmp_path / "output")


class TestCriteriaInduction:
    """Tests for criteria induction functionality."""

    def test_induce_criteria_with_mock_provider(self) -> None:
        """Test induce_criteria with mocked provider."""
        mock_response = MagicMock()
        mock_response.text = """[
            {"id": "accuracy", "name": "Accuracy", "weight": 60, "semanticPrompt": "Is accurate"},
            {"id": "clarity", "name": "Clarity", "weight": 40, "semanticPrompt": "Is clear"}
        ]"""

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch("portex_eval.bundle.formatter.get_provider", return_value=mock_provider):
            criteria = induce_criteria("Paris", "What is the capital of France?")

        assert len(criteria) == 2
        assert criteria[0]["id"] == "accuracy"
        assert sum(c["weight"] for c in criteria) == 100

    def test_format_bundle_with_criteria_induction(
        self, answer_only_bundle: Path, tmp_path: Path
    ) -> None:
        """
        Verifies that enabling criteria induction causes format_bundle to generate and persist induced criteria.
        
        Patches the provider to return a single induced criterion and asserts that:
        - result.criteria_induced is True
        - the output answers.json contains a non-empty `criteria` list for each answer
        - each induced criterion has `grader_type` set to "llm-judge"
        """
        mock_response = MagicMock()
        mock_response.text = """[
            {"id": "correct", "name": "Correct", "weight": 100, "semanticPrompt": "Is correct"}
        ]"""

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_response

        with patch("portex_eval.bundle.formatter.get_provider", return_value=mock_provider):
            output_dir = tmp_path / "output_bundle"
            result = format_bundle(
                answer_only_bundle, output_dir, induce_criteria_flag=True, judge_model="mock:model"
            )

        assert result.criteria_induced is True

        answers_path = Path(result.output_dir) / "answers.json"
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        for answer in answers:
            assert len(answer["criteria"]) > 0
            assert answer["criteria"][0]["grader_type"] == "llm-judge"

    def test_format_bundle_rejects_missing_criteria_without_induction(
        self, answer_only_bundle: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(PortexEvalError, match="criteria must be a non-empty list"):
            format_bundle(answer_only_bundle, tmp_path / "output_bundle")


class TestParseCriteriaResponse:
    """Tests for _parse_criteria_response helper."""

    def test_parse_plain_json(self) -> None:
        """Test parsing plain JSON array."""
        response = '[{"id": "test", "name": "Test", "weight": 100, "semanticPrompt": "Test"}]'
        criteria = _parse_criteria_response(response)
        assert len(criteria) == 1
        assert criteria[0]["id"] == "test"

    def test_parse_json_with_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        response = """Here are the criteria:
```json
[{"id": "test", "name": "Test", "weight": 100, "semanticPrompt": "Test"}]
```"""
        criteria = _parse_criteria_response(response)
        assert len(criteria) == 1

    def test_parse_json_with_surrounding_text(self) -> None:
        """Test parsing JSON with surrounding text."""
        response = """Some text before
[{"id": "test", "name": "Test", "weight": 100, "semanticPrompt": "Test"}]
Some text after"""
        criteria = _parse_criteria_response(response)
        assert len(criteria) == 1

    def test_parse_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises PortexEvalError."""
        with pytest.raises(PortexEvalError, match="Failed to parse criteria"):
            _parse_criteria_response("not valid json")

    def test_parse_missing_field_raises_error(self) -> None:
        """Test that missing required field raises PortexEvalError."""
        with pytest.raises(PortexEvalError, match="missing required field"):
            _parse_criteria_response('[{"id": "test", "name": "Test"}]')


class TestNormalizeCriteriaWeights:
    """Tests for _normalize_criteria_weights helper."""

    def test_weights_already_sum_to_100(self) -> None:
        """Test that weights summing to 100 are preserved."""
        criteria = [
            {"id": "a", "weight": 60},
            {"id": "b", "weight": 40},
        ]
        result = _normalize_criteria_weights(criteria)
        assert sum(c["weight"] for c in result) == 100

    def test_weights_normalized_to_100(self) -> None:
        """Test that weights are normalized to sum to 100."""
        criteria = [
            {"id": "a", "weight": 30},
            {"id": "b", "weight": 20},
        ]
        result = _normalize_criteria_weights(criteria)
        assert sum(c["weight"] for c in result) == 100

    def test_zero_weights_distributed_equally(self) -> None:
        """Test that zero weights are distributed equally."""
        criteria = [
            {"id": "a", "weight": 0},
            {"id": "b", "weight": 0},
            {"id": "c", "weight": 0},
        ]
        result = _normalize_criteria_weights(criteria)
        assert sum(c["weight"] for c in result) == 100

    def test_empty_criteria(self) -> None:
        """Test that empty criteria list returns empty."""
        result = _normalize_criteria_weights([])
        assert result == []
