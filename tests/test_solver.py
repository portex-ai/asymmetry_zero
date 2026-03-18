"""Tests for provider-backed solver helpers."""

from __future__ import annotations

from portex_eval.benchmark.inspect.solver import _provider_model_output
from portex_eval.providers import get_provider


def test_provider_model_output_returns_inspect_model_output() -> None:
    provider = get_provider(
        {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_url": "https://modal.example/v1",
        }
    )

    output = _provider_model_output(
        provider,
        "Answer: cat",
        {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )

    assert output.model == "vllm:Qwen/Qwen3-VL-4B-Instruct"
    assert output.completion == "Answer: cat"
    assert output.message.text == "Answer: cat"
    assert output.usage is not None
    assert output.usage.input_tokens == 10
    assert output.usage.output_tokens == 3
    assert output.usage.total_tokens == 13
