"""Tests for portex_eval.reporting module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from portex_eval.reporting import load, tables


class TestHelperFunctions:
    """Unit tests for helper functions in tables module."""

    def test_parse_json_returns_dict_from_valid_json(self) -> None:
        result = tables._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_returns_empty_dict_for_none(self) -> None:
        result = tables._parse_json(None)
        assert result == {}

    def test_parse_json_returns_dict_directly(self) -> None:
        result = tables._parse_json({"existing": "dict"})
        assert result == {"existing": "dict"}

    def test_parse_json_returns_empty_for_invalid_json(self) -> None:
        result = tables._parse_json("not valid json")
        assert result == {}

    def test_parse_json_returns_empty_for_non_dict_json(self) -> None:
        result = tables._parse_json("[1, 2, 3]")
        assert result == {}

    def test_composite_id_extracts_parent_name(self) -> None:
        result = tables._composite_id("/some/path/bundle_name/tasks.json")
        assert result == "bundle_name"

    def test_composite_id_returns_none_for_empty(self) -> None:
        result = tables._composite_id(None)
        assert result is None

    def test_composite_id_returns_none_for_empty_string(self) -> None:
        result = tables._composite_id("")
        assert result is None

    def test_usage_for_models_aggregates_tokens(self) -> None:
        model_usage = {
            "model-a": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "model-b": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
        }
        result = tables._usage_for_models(model_usage, ["model-a", "model-b"])
        assert result["input_tokens"] == 300
        assert result["output_tokens"] == 150
        assert result["total_tokens"] == 450

    def test_usage_for_models_returns_none_when_no_models_requested(self) -> None:
        # No models to look up means found=False
        model_usage = {"model-a": {"input_tokens": 100}}
        result = tables._usage_for_models(model_usage, [])
        assert result["input_tokens"] is None
        assert result["output_tokens"] is None
        assert result["total_tokens"] is None

    def test_usage_for_models_returns_zero_for_missing_model(self) -> None:
        # When model is not in usage dict, returns 0s (not None)
        model_usage = {"model-a": {"input_tokens": 100}}
        result = tables._usage_for_models(model_usage, ["model-x"])
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_judge_models_from_metadata_extracts_models(self) -> None:
        metadata = {
            "criteria": [
                {"judges": [{"model": "judge-1"}, {"model": "judge-2"}]},
                {"judges": [{"model": "judge-1"}]},
            ]
        }
        result = tables._judge_models_from_metadata(metadata)
        assert result == ["judge-1", "judge-2"]

    def test_judge_models_from_metadata_handles_empty(self) -> None:
        result = tables._judge_models_from_metadata({})
        assert result == []

    def test_matches_criterion_true_for_matching_prompts(self) -> None:
        assert tables._matches_criterion("prompt text", "prompt text") is True
        assert tables._matches_criterion("  prompt text  ", "prompt text") is True

    def test_matches_criterion_false_for_different_prompts(self) -> None:
        assert tables._matches_criterion("prompt a", "prompt b") is False

    def test_matches_criterion_false_for_none(self) -> None:
        assert tables._matches_criterion(None, "prompt") is False
        assert tables._matches_criterion("prompt", None) is False


class TestSummarizeEvents:
    """Tests for event summarization."""

    def test_summarize_events_returns_none_for_empty_df(self) -> None:
        empty_df = pd.DataFrame(columns=["model_event_time", "model_event_cost"])
        result = tables._summarize_events(empty_df)
        assert result["latency"] is None
        assert result["cost"] is None

    def test_summarize_events_sums_latency_and_cost(self) -> None:
        df = pd.DataFrame(
            {
                "model_event_time": [1.5, 2.0, 0.5],
                "model_event_cost": [0.01, 0.02, 0.01],
            }
        )
        result = tables._summarize_events(df)
        assert result["latency"] == pytest.approx(4.0)
        assert result["cost"] == pytest.approx(0.04)


class TestLoadFunction:
    """Tests for the load helper function."""

    def test_load_reads_csv(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            f.flush()
            path = f.name

        try:
            df = load(path)
            assert len(df) == 2
            assert list(df.columns) == ["col1", "col2"]
        finally:
            Path(path).unlink()


class TestEventCallCost:
    """Tests for extracting cost from event calls."""

    def test_event_call_cost_from_response_usage(self) -> None:
        call = {"response": {"usage": {"cost": 0.05}}}
        result = tables._event_call_cost(call)
        assert result == pytest.approx(0.05)

    def test_event_call_cost_from_json_string(self) -> None:
        call = json.dumps({"response": {"usage": {"cost": 0.03}}})
        result = tables._event_call_cost(call)
        assert result == pytest.approx(0.03)

    def test_event_call_cost_returns_none_for_missing(self) -> None:
        call: dict[str, dict[str, float]] = {"response": {}}
        result = tables._event_call_cost(call)
        assert result is None

    def test_event_call_cost_returns_none_for_none(self) -> None:
        result = tables._event_call_cost(None)
        assert result is None


class TestEventCallCriterion:
    """Tests for extracting criterion from event calls."""

    def test_event_call_criterion_extracts_from_message(self) -> None:
        content = "Some prefix\n***\n[Criterion]: Test criterion name\n***\n[END DATA]"
        call = {"request": {"messages": [{"content": content}]}}
        result = tables._event_call_criterion(call)
        assert result == "Test criterion name"

    def test_event_call_criterion_returns_none_for_no_match(self) -> None:
        call = {"request": {"messages": [{"content": "No criterion here"}]}}
        result = tables._event_call_criterion(call)
        assert result is None

    def test_event_call_criterion_returns_none_for_none(self) -> None:
        result = tables._event_call_criterion(None)
        assert result is None
