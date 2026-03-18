# Providers

This guide covers model provider configuration and usage in `portex-eval`.

## Overview

Providers wrap external LLM APIs and expose a uniform interface for generation calls. They are used both for:

1. **Judge models** - Evaluating candidate responses
2. **Candidate models** - Generating responses to be evaluated

## Model String Format

Models are specified using the format:

```
provider:model_id
```

Examples:
- `openrouter:openai/gpt-4o`
- `openrouter:anthropic/claude-3.5-sonnet`
- `openrouter:google/gemini-2.5-flash`
- `openrouter:meta-llama/llama-3.3-70b-instruct`
- `openai:gpt-4o-mini`
- `anthropic:claude-sonnet-4-5`
- `vllm:Qwen/Qwen3-VL-4B-Instruct`
- `custom:my-model`

For custom endpoints or per-model auth, you can also pass a config object instead of a plain string:

```python
{
    "provider": "vllm",
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "base_url": "https://portex--qwen3-vl-4b-instruct-vllm-baseline-serve.modal.run/v1",
    "api_key_env": "MODAL_VLLM_API_KEY"
}
```

## Supported Providers

### OpenRouter

[OpenRouter](https://openrouter.ai) provides unified access to models from multiple providers.

#### Setup

1. Get an API key from [OpenRouter](https://openrouter.ai/keys)
2. Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or create a `.env` file:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

#### Available Models

OpenRouter provides access to hundreds of models. Common choices for judges:

| Model | Identifier |
|-------|------------|
| GPT-4o | `openrouter:openai/gpt-4o` |
| Claude 3.5 Sonnet | `openrouter:anthropic/claude-3.5-sonnet` |
| Gemini 2.5 Flash | `openrouter:google/gemini-2.5-flash` |
| Gemini 2.5 Pro | `openrouter:google/gemini-2.5-pro` |

Common choices for candidates:

| Model | Identifier |
|-------|------------|
| Llama 3.3 70B | `openrouter:meta-llama/llama-3.3-70b-instruct` |
| Mixtral 8x22B | `openrouter:mistralai/mixtral-8x22b-instruct` |
| Qwen 2.5 72B | `openrouter:qwen/qwen-2.5-72b-instruct` |
| DeepSeek V3 | `openrouter:deepseek/deepseek-chat` |

See [OpenRouter models](https://openrouter.ai/models) for the full list.

#### Configuration Options

```python
from portex_eval import get_provider

provider = get_provider(
    "openrouter:google/gemini-2.5-flash",
    api_key="sk-or-v1-...",  # Override env var
    max_retries=5,           # Rate limit retries
    base_delay=1.0,          # Initial retry delay (seconds)
    max_delay=60.0,          # Maximum retry delay (seconds)
    jitter=0.5,              # Randomness factor for delays
    timeout=120.0,           # Request timeout (seconds)
)
```

### OpenAI

OpenAI models can be addressed directly without going through OpenRouter:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
provider = get_provider("openai:gpt-4o-mini")
```

### Anthropic

Anthropic models are supported as first-class providers:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
provider = get_provider("anthropic:claude-sonnet-4-5")
```

### OpenAI-Compatible Endpoints

`vllm` and `custom` are aliases for the shared OpenAI-compatible backend. Use them for Modal, local vLLM, or other deployments that expose `/v1/chat/completions`.

```python
provider = get_provider(
    {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "base_url": "https://portex--qwen3-vl-4b-instruct-vllm-baseline-serve.modal.run/v1",
    }
)
```

## Using Providers Directly

While `eval()` handles provider creation internally, you can use providers directly:

```python
from portex_eval import get_provider

# Create a provider
provider = get_provider("openrouter:google/gemini-2.5-flash")

# Synchronous generation
response = provider.generate(
    "What is the capital of France?",
    max_tokens=100,
    temperature=0.0,
)
print(response.text)

# Async generation
import asyncio

async def main():
    response = await provider.agenerate(
        "What is the capital of France?",
        max_tokens=100,
    )
    print(response.text)

asyncio.run(main())
```

### Response Object

```python
@dataclass
class Response:
    text: str                       # Generated text
    usage: dict[str, int] | None    # Token usage (prompt_tokens, completion_tokens)
    raw: Any                        # Raw API response
```

## Judge Panel Configuration

For robust evaluation, use a panel of 3 diverse judge models:

```python
from portex_eval import eval

results = eval(
    path="./mybenchmark",
    judges=[
        "openrouter:openai/gpt-4o",           # OpenAI
        "openrouter:anthropic/claude-3.5-sonnet",  # Anthropic
        "openrouter:google/gemini-2.5-flash",      # Google
    ],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)
```

This diversity helps reduce bias from any single model family.

You can also mix direct providers and custom endpoints in one run:

```python
results = eval(
    path="./mybenchmark",
    judges=[
        "openrouter:google/gemini-2.5-flash",
        {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    ],
    candidates=[
        {
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "base_url": "https://portex--qwen3-vl-4b-instruct-vllm-baseline-serve.modal.run/v1",
        }
    ],
)
```

## Rate Limiting

OpenRouter enforces rate limits. The provider automatically handles:

1. **Exponential backoff** - Doubles delay between retries
2. **Jitter** - Adds randomness to avoid thundering herd
3. **Retry-After headers** - Respects server-specified delays

Configure retry behavior:

```python
provider = get_provider(
    "openrouter:openai/gpt-4o",
    max_retries=10,    # More retries for high-volume runs
    max_delay=120.0,   # Longer maximum delay
)
```

## Error Handling

```python
from portex_eval import get_provider
from portex_eval.providers import RateLimitError

provider = get_provider("openrouter:openai/gpt-4o")

try:
    response = provider.generate("Hello")
except RateLimitError as e:
    print(f"Rate limited after max retries: {e}")
except ValueError as e:
    print(f"Provider error: {e}")
```

## Custom Providers

To add a custom provider, implement the `Provider` abstract class:

```python
from portex_eval.providers.base import Provider, Response

class MyProvider(Provider):
    def __init__(self, model_id: str, **kwargs):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def provider_id(self) -> str:
        return "myprovider"

    def generate(self, prompt: str, **kwargs) -> Response:
        # Your implementation
        return Response(text="...", usage=None, raw=None)

    async def agenerate(self, prompt: str, **kwargs) -> Response:
        # Your async implementation
        return Response(text="...", usage=None, raw=None)
```

Register the provider:

```python
from portex_eval.providers import register_provider

register_provider("myprovider", lambda config, **kw: MyProvider(config.model, **kw))

# Now use it
provider = get_provider("myprovider:my-model")
```

## Future Providers

Additional provider support is planned:

- Fireworks AI
- Together AI
- BYOK (Bring Your Own Key) for direct API access

See [CONTRIBUTING.md](../CONTRIBUTING.md) to help add new providers.
