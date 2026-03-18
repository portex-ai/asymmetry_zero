"""Tests for provider registry and model config parsing."""

from __future__ import annotations

from portex_eval.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    get_provider,
    model_config_from_spec,
)


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
