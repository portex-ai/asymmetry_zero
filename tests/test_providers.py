"""Tests for provider registry and model config parsing."""

from __future__ import annotations

from portex_eval.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    get_provider,
    model_config_from_spec,
)
from portex_eval.providers.base import ModelConfig


def test_model_config_from_spec_accepts_dict() -> None:
    config = model_config_from_spec(
        {
            "provider": "custom",
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_url": "https://example.com/v1",
            "api_key_env": "CUSTOM_API_KEY",
            "headers": {"X-Test": "1"},
            "options": {"temperature": 0},
        }
    )

    assert config.provider == "custom"
    assert config.model == "Qwen/Qwen3-VL-4B-Instruct"
    assert config.base_url == "https://example.com/v1"
    assert config.api_key_env == "CUSTOM_API_KEY"
    assert config.headers["X-Test"] == "1"
    assert config.options["temperature"] == 0


def test_get_provider_resolves_openai(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = get_provider("openai:gpt-4o-mini")

    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_id == "openai"
    assert provider.model_name == "gpt-4o-mini"


def test_get_provider_resolves_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = get_provider("anthropic:claude-sonnet-4-5")

    assert isinstance(provider, AnthropicProvider)
    assert provider.provider_id == "anthropic"
    assert provider.model_name == "claude-sonnet-4-5"


def test_get_provider_resolves_openai_compatible_alias() -> None:
    provider = get_provider(
        {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_url": "https://modal.example/v1",
        }
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.provider_id == "vllm"
    assert provider.model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_openrouter_parse_response_coerces_none_content(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = get_provider("openrouter:openai/gpt-4o-mini")

    response = provider._parse_response({"choices": [{"message": {"content": None}}]})  # type: ignore[attr-defined]

    assert response.text == ""


def test_openrouter_parse_response_extracts_cost(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = get_provider("openrouter:openai/gpt-4o-mini")

    response = provider._parse_response(  # type: ignore[attr-defined]
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.1234},
        }
    )

    assert response.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert response.cost == 0.1234


def test_openai_compatible_parse_response_coerces_none_content() -> None:
    provider = OpenAICompatibleProvider(
        ModelConfig(provider="custom", model="demo", base_url="https://example.com/v1")
    )

    response = provider._parse_response({"choices": [{"message": {"content": None}}]})  # type: ignore[attr-defined]

    assert response.text == ""
