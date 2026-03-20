"""Tests for exact-match scoring helpers."""

from __future__ import annotations

import asyncio

from portex_eval.grading import core as grading_core
from portex_eval.providers.base import Response
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


def test_grade_criteria_with_providers_caps_concurrent_criteria() -> None:
    criteria = [{"id": f"c{i}"} for i in range(25)]
    state = {"current": 0, "peak": 0}

    async def fake_grade(
        criterion: dict[str, object],
        *,
        question: str,
        submission: str,
        judge_providers: dict[str, object],
    ) -> dict[str, object]:
        del criterion, question, submission, judge_providers
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.01)
        state["current"] -= 1
        return {"awarded": 0.0}

    original = grading_core.grade_criterion_with_providers
    grading_core.grade_criterion_with_providers = fake_grade
    try:
        asyncio.run(
            grading_core.grade_criteria_with_providers(
                criteria,
                question="q",
                submission="a",
                judge_providers={},
            )
        )
    finally:
        grading_core.grade_criterion_with_providers = original

    assert state["peak"] == grading_core.MAX_CONCURRENT_CRITERIA_GRADING


def test_parse_grade_from_response_tolerates_none() -> None:
    passed, grade = grading_core.parse_grade_from_response(None)

    assert passed is False
    assert grade == "I"


def test_grade_with_provider_captures_usage_latency_and_cost() -> None:
    class FakeProvider:
        provider_id = "openrouter"
        model_name = "test-model"

        async def agenerate(self, prompt: str) -> Response:  # noqa: ARG002
            return Response(
                text="Reasoning...\nGRADE: C",
                usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                latency=1.25,
                cost=0.0042,
            )

    result = asyncio.run(
        grading_core.grade_with_provider(
            FakeProvider(),  # type: ignore[arg-type]
            question="q",
            answer="a",
            criterion="c",
            weight=2.0,
        )
    )

    assert result["passed"] is True
    assert result["input_tokens"] == 11
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 18
    assert result["latency"] == 1.25
    assert result["cost"] == 0.0042
