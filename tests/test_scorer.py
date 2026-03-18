"""Tests for exact-match scoring helpers."""

from __future__ import annotations

from portex_eval.benchmark.inspect.scorer import (
    _extract_final_answer,
    _grade_criterion_exact_match,
)


def test_extract_final_answer_prefers_answer_line() -> None:
    submission = "Reasoning here.\nAnswer: The capital of France is Paris."
    assert _extract_final_answer(submission) == "The capital of France is Paris."


def test_extract_final_answer_falls_back_to_last_line() -> None:
    submission = "Reasoning here.\nFinal line"
    assert _extract_final_answer(submission) == "Final line"


def test_grade_criterion_exact_match_passes_on_inclusion() -> None:
    criterion = {
        "id": "capital",
        "name": "Exact capital",
        "weight": 100,
        "grader_type": "ExactMatch",
        "semanticPrompt": "Paris",
    }
    result = _grade_criterion_exact_match(
        criterion,
        "Reasoning...\nAnswer: The capital of France is Paris.",
    )
    assert result["passed"] is True
    assert result["awarded"] == 100
    assert result["grader_type"] == "ExactMatch"
    assert result["judges"][0]["model"] == "ExactMatch"


def test_grade_criterion_exact_match_fails_without_match() -> None:
    criterion = {
        "id": "capital",
        "name": "Exact capital",
        "weight": 100,
        "grader_type": "ExactMatch",
        "semanticPrompt": "Paris",
    }
    result = _grade_criterion_exact_match(
        criterion,
        "Reasoning...\nAnswer: The capital of France is Lyon.",
    )
    assert result["passed"] is False
    assert result["awarded"] == 0
