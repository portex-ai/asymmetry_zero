# Providers

This guide covers the model-spec formats accepted by `portex-eval`.

## Model Spec Forms

Most APIs accept a `ModelSpec`, which can be:

- A string such as `openrouter:openai/gpt-4o-mini`
- A `ModelConfig` instance
- A config object with provider settings

String form:

```text
provider:model
```

Examples:

- `openrouter:openai/gpt-4o-mini`
- `openai:gpt-4o-mini`
- `anthropic:claude-sonnet-4-5`
- `vllm:Qwen/Qwen3-VL-4B-Instruct`

Config-object form:

```python
{
    "provider": "vllm",
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "base_url": "https://my-endpoint.example.com/v1",
    "api_key_env": "VLLM_API_KEY",
    "headers": {"x-team": "evals"},
    "options": {"temperature": 0.0},
}
```

## Supported Providers

| Provider id | Auth | Notes |
| --- | --- | --- |
| `openrouter` | `OPENROUTER_API_KEY` | Good default for standard bundle evals |
| `openai` | `OPENAI_API_KEY` | Direct OpenAI provider |
| `anthropic` | `ANTHROPIC_API_KEY` | Direct Anthropic provider |
| `openai_compatible` | endpoint-specific | Shared backend for arbitrary OpenAI-compatible APIs |
| `openai-compatible` | endpoint-specific | Alias for `openai_compatible` |
| `vllm` | endpoint-specific | Alias for `openai_compatible` |
| `custom` | endpoint-specific | Alias for `openai_compatible` |

Use `portex-eval[providers]` or `portex-eval[all]` when you need direct provider objects or non-OpenRouter endpoints.

## Config Object Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `provider` | Yes | Provider id |
| `model` | Yes | Provider-specific model name |
| `base_url` | No | Custom API base URL |
| `api_key` | No | Inline API key |
| `api_key_env` | No | Environment variable name containing the API key |
| `headers` | No | Extra HTTP headers |
| `options` | No | Provider-specific options |

## Standard OpenRouter Run

```python
from portex_eval import eval

results = eval(
    path="./examples/simple_bundle",
    judges=[
        "openrouter:openai/gpt-4o-mini",
        "openrouter:anthropic/claude-3.5-sonnet",
    ],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)
```

## Mixed Providers

```python
from portex_eval import eval

results = eval(
    path="./examples/simple_bundle",
    judges=[
        "openrouter:openai/gpt-4o-mini",
        {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    ],
    candidates=[
        {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_url": "https://my-endpoint.example.com/v1",
            "api_key_env": "VLLM_API_KEY",
        }
    ],
)
```

## Using Providers Directly

```python
from portex_eval import get_provider

provider = get_provider("openrouter:openai/gpt-4o-mini")
response = provider.generate("What is 2 + 2?", max_tokens=32, temperature=0.0)

print(response.text)
print(response.usage)
```

The response object includes:

- `text`
- `usage`
- `latency`
- `cost`
- `raw`

## Discovering Provider IDs

```python
from portex_eval.providers import get_supported_providers

print(sorted(get_supported_providers()))
```

## Notes

- OpenRouter-only runs can use the standard Inspect-based runtime without provider config objects.
- Use config objects when you need a custom base URL, non-default auth, or custom headers.
- `judge_configs` and `candidate_configs` on the CLI accept either inline JSON or paths to JSON files.
