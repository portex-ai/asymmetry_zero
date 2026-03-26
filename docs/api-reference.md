# API Reference

This page documents the current public Python surface.

## Common Types

### `ModelSpec`

Most model-taking APIs accept:

```python
str | ModelConfig | dict[str, object]
```

Examples:

```python
"openrouter:openai/gpt-4o-mini"
{"provider": "anthropic", "model": "claude-sonnet-4-5"}
```

## Core Functions

### `create_benchmark(path: str) -> Benchmark`

Format a BYOB benchmark JSON file into a Portex bundle.

```python
from portex_eval import create_benchmark

benchmark = create_benchmark("./examples/benchmark.json")
print(benchmark.path)
print(benchmark.task_count)
```

Notes:

- The input file must be a JSON list.
- The output bundle is written next to the input file with a random suffix.
- Relative `reference_file` paths are resolved relative to the input JSON file.

### `eval(...) -> EvalResults`

```python
from portex_eval import eval

results = eval(
    path="./examples/simple_bundle",
    judges=["openrouter:openai/gpt-4o-mini"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    output_dir="./eval_runs",
    max_samples=4,
    logprobs=True,
    top_logprobs=5,
)
```

Key arguments:

| Argument | Meaning |
| --- | --- |
| `path` | Bundle directory path |
| `benchmark` | `Benchmark` returned by `create_benchmark()` |
| `judges` | Judge `ModelSpec` list |
| `candidates` | Candidate `ModelSpec` list |
| `output_dir` | Runs root for generated outputs |
| `config` | Optional `Config` instance |
| `task_spec` | Advanced Inspect task override |
| `max_samples` | Parallel sample count |
| `logprobs` | Request candidate completion logprobs |
| `top_logprobs` | Number of alternative token logprobs |
| `overwrite` | Allow overwriting an existing run directory |

Rules:

- Provide exactly one of `path` or `benchmark`.
- `judges` and `candidates` must both be non-empty.
- `results.output_dir` is the final timestamped run directory.

### `create_agent_eval(...) -> AgentEvalBundle`

Generate Harbor task directories from a bundle:

```python
from portex_eval import create_agent_eval

bundle = create_agent_eval(
    path="./examples/simple_bundle",
    output_dir="/tmp/simple_bundle_agent",
)
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `path` or `benchmark` | Source bundle |
| `output_dir` | Harbor task root to generate |
| `overwrite` | Allow replacing an existing output directory |

### `agent_eval(...) -> AgentEvalResults`

Run Harbor on generated Harbor tasks:

```python
from portex_eval import agent_eval

results = agent_eval(
    task_root="/tmp/simple_bundle_agent",
    judges=["openrouter:openai/gpt-4o-mini"],
    extra_args=["--agent", "terminus-2", "--model", "openrouter/openai/gpt-4o-mini"],
)
```

Key arguments:

| Argument | Meaning |
| --- | --- |
| `task_root` | Harbor task root created by `create_agent_eval()` |
| `judges` | Optional Harbor verifier judge models |
| `output_dir` | Optional Harbor results root |
| `n_concurrent` | Harbor task concurrency |
| `env` | Harbor environment profile |
| `extra_args` | Extra arguments forwarded to `harbor run` |
| `overwrite` | Allow overwriting an existing output root |

### `load_run_spec(path) -> RunSpec`

Load a YAML run spec:

```python
from portex_eval import load_run_spec

spec = load_run_spec("./examples/run_spec.yaml")
```

### `write_run_spec(path, spec) -> None`

Write a `RunSpec` back to YAML:

```python
from portex_eval.run_spec import RunSpec, write_run_spec

spec = RunSpec(
    bundle_path="./examples/simple_bundle",
    judges=["openrouter:openai/gpt-4o-mini"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)
write_run_spec("./run_spec.yaml", spec)
```

## Result Dataclasses

### `Benchmark`

| Field | Meaning |
| --- | --- |
| `path` | Absolute bundle directory |
| `task_count` | Number of tasks |

Methods:

- `resolve_path() -> Path`

### `ReportPaths`

| Field | Meaning |
| --- | --- |
| `eval_level` | Path to `eval_level.csv` |
| `task_level` | Path to `task_level.csv` |
| `criterion_level` | Path to `criterion_level.csv` |
| `judgement_level` | Path to `judgement_level.csv` |

### `Rewards`

| Field | Meaning |
| --- | --- |
| `task_ids` | Task ids in output order |
| `reward` | Numeric reward values |

### `EvalResults`

| Field | Meaning |
| --- | --- |
| `logs` | Absolute paths to generated `.eval` logs |
| `reports` | `ReportPaths` or `None` |
| `rewards` | `Rewards` payload |
| `rewards_path` | Path to `rl_rewards.json` |
| `training_data_path` | Path to `rl_training_data.json` |
| `run_id` | Timestamp-based run identifier |
| `output_dir` | Final run directory |

Methods:

- `with_absolute_paths() -> EvalResults`

### `AgentEvalBundle`

| Field | Meaning |
| --- | --- |
| `path` | Harbor task root |
| `datasets_dir` | Generated Harbor datasets directory |
| `task_count` | Number of generated tasks |

### `AgentEvalResults`

| Field | Meaning |
| --- | --- |
| `datasets_dir` | Harbor datasets directory |
| `jobs_dir` | Harbor jobs directory for the run |
| `reports` | `ReportPaths` or `None` |
| `rewards` | `Rewards` payload |
| `rewards_path` | Path to `rl_rewards.json` |
| `training_data_path` | Path to `rl_training_data.json` |
| `run_id` | Harbor run identifier |
| `output_dir` | Harbor results root |

Methods:

- `with_absolute_paths() -> AgentEvalResults`

## Configuration Dataclasses

### `Config`

```python
from portex_eval import Config

config = Config()
```

Fields:

| Field | Default |
| --- | --- |
| `bundles_dir` | `./bundles` |
| `runs_dir` | `./eval_runs` |
| `cache_dir` | `./.portex_cache` |

Helpers:

- `Config.from_env()`
- `Config.from_dict(data)`
- `resolve_bundle_path(bundle_name)`
- `ensure_runs_dir()`
- `ensure_cache_dir()`

### `RunSpec`

| Field | Meaning |
| --- | --- |
| `bundle_path` | Bundle directory path |
| `judges` | Judge model string list |
| `candidates` | Candidate model string list |
| `schema_version` | Must be `1` |

Helper:

- `validate_bundle_exists()`

## Provider Helpers

### `get_provider(model_spec, **kwargs) -> Provider`

```python
from portex_eval import get_provider

provider = get_provider("openrouter:openai/gpt-4o-mini")
response = provider.generate("Hello")
```

### `Response`

| Field | Meaning |
| --- | --- |
| `text` | Generated text |
| `usage` | Normalized token-usage dictionary or `None` |
| `latency` | Provider latency if available |
| `cost` | Provider cost if available |
| `raw` | Raw provider response |

### `ModelConfig`

| Field | Meaning |
| --- | --- |
| `provider` | Provider id |
| `model` | Provider model name |
| `base_url` | Custom API URL |
| `api_key` | Inline API key |
| `api_key_env` | API key env-var name |
| `headers` | Extra request headers |
| `options` | Provider-specific options |

### `get_supported_providers() -> set[str]`

```python
from portex_eval.providers import get_supported_providers

print(sorted(get_supported_providers()))
```

## Reporting Helpers

### `reports.load(path) -> pandas.DataFrame`

```python
from portex_eval import reports

task_df = reports.load(results.reports.task_level)
```

## Errors

Public APIs raise `PortexEvalError` for validation and runtime failures that should be shown to end users.
