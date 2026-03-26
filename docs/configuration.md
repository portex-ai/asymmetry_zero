# Configuration

This guide covers public configuration surfaces for `portex-eval`.

## Environment Variables

### Provider credentials

| Variable | Used for |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter model strings such as `openrouter:openai/gpt-4o-mini` |
| `OPENAI_API_KEY` | Direct OpenAI provider configs such as `openai:gpt-4o-mini` |
| `ANTHROPIC_API_KEY` | Direct Anthropic provider configs such as `anthropic:claude-sonnet-4-5` |

### Runtime directories

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORTEX_BUNDLES_DIR` | `./bundles` | Default bundle root used by `Config.resolve_bundle_path()` |
| `PORTEX_RUNS_DIR` | `./eval_runs` | Root directory under which standard eval runs are written |
| `PORTEX_CACHE_DIR` | `./.portex_cache` | Local cache directory |

`python-dotenv` is loaded automatically, so a local `.env` file works without extra setup.

## The `Config` Class

```python
from portex_eval import Config, eval

config = Config(
    bundles_dir="./bundles",
    runs_dir="./eval_runs",
    cache_dir="./.portex_cache",
)

results = eval(
    path="./examples/simple_bundle",
    judges=["openrouter:openai/gpt-4o-mini"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    config=config,
)
```

You can also build it from the environment or a dictionary:

```python
Config.from_env()
Config.from_dict({"runs_dir": "./custom-runs"})
```

## Standard Eval Runtime Options

The main public runtime knobs are:

| Argument | Meaning |
| --- | --- |
| `output_dir` | Root directory used for eval runs |
| `overwrite` | Allow overwriting an existing run directory |
| `max_samples` | Maximum number of bundle samples to run in parallel |
| `logprobs` | Request candidate completion logprobs when supported |
| `top_logprobs` | Number of alternative logprobs per completion token |

Example:

```python
results = eval(
    path="./examples/simple_bundle",
    judges=["openrouter:openai/gpt-4o-mini"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    output_dir="./custom-runs",
    max_samples=4,
    logprobs=True,
    top_logprobs=5,
)
```

`output_dir` is the runs root, not the final leaf directory. The finished run path is available in `results.output_dir`.

## Run Specs

Run specs are YAML files loaded with `load_run_spec()`:

```yaml
schema_version: 1
bundle_path: ./examples/simple_bundle
judges:
  - openrouter:openai/gpt-4o-mini
  - openrouter:anthropic/claude-3.5-sonnet
candidates:
  - openrouter:meta-llama/llama-3.3-70b-instruct
```

Programmatic use:

```python
from portex_eval import eval, load_run_spec

spec = load_run_spec("./examples/run_spec.yaml")
results = eval(
    path=spec.bundle_path,
    judges=spec.judges,
    candidates=spec.candidates,
)
```

You can also write specs back out with `write_run_spec()`.

## `agent-run` YAML Configs

`portex-eval agent-run` accepts a separate YAML config format via `--config`:

```yaml
schema_version: 1
tasks: /tmp/simple_bundle_agent
output: /tmp/simple_bundle_agent_run
judges:
  - openrouter:openai/gpt-4o-mini
n_concurrent: 4
env: modal
overwrite: false
harbor_args:
  - --agent
  - terminus-2
  - --model
  - openrouter/openai/gpt-4o-mini
```

Supported top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be `1` |
| `tasks` | Harbor task root created by `agent-create` |
| `output` | Optional Harbor results root |
| `judges` | Judge model strings |
| `judge_configs` | Judge model config JSON strings or JSON file paths |
| `n_concurrent` | Harbor task concurrency |
| `env` | Harbor environment profile name |
| `overwrite` | Whether to allow overwriting an existing output root |
| `harbor_args` | Extra arguments forwarded to `harbor run` |

## Provider Config Objects

Provider config objects are documented in [Providers](providers.md). Use them when you need custom base URLs, per-model headers, or non-default API key names.
