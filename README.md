# portex-eval

`portex-eval` is a bundle-first evaluation toolkit for judge-panel and Harbor agent evals. It formats simple benchmark JSON into Portex bundles, runs Inspect-based evaluation workflows, and emits CSV plus JSON artifacts for analysis and post-training.

## Capabilities

- Format a simple benchmark JSON file into a reusable Portex bundle
- Run standard judge-panel evaluations against bundle tasks
- Mix OpenRouter strings, direct OpenAI/Anthropic configs, and OpenAI-compatible endpoints
- Emit `.eval` logs, CSV reports, `rl_rewards.json`, and `rl_training_data.json`
- Convert the same bundle into Harbor tasks for agent evaluation

## Installation

For most users:

```bash
pip install 'portex-eval[all]'
# or
uv tool install 'portex-eval[all]'
```

Available install targets:

| Package | Use case |
| --- | --- |
| `portex-eval` | Bundle formatting, schema helpers, report loading, and core types |
| `portex-eval[inspect]` | Standard bundle eval runs via Inspect AI |
| `portex-eval[providers]` | Direct provider adapters and custom OpenAI-compatible endpoints |
| `portex-eval[harbor]` | Harbor agent evals; Harbor support requires Python 3.12+ |
| `portex-eval[all]` | All extras above |

From source:

```bash
git clone https://github.com/portex-ai/portex-eval
cd portex-eval
uv sync
uv run portex-eval --help

# Optional Harbor stack
uv sync --group harbor
```

`uv sync` installs the normal development stack, including Inspect. Harbor is kept in its own dependency group so non-Harbor environments do not need its heavier dependency chain.

## Quick Start

The examples below use OpenRouter model strings, so they only need `OPENROUTER_API_KEY`:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Run an existing bundle:

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

print(results.output_dir)
print(results.rewards_path)
```

Format and run a BYOB benchmark:

```python
from portex_eval import create_benchmark, eval

benchmark = create_benchmark("./examples/benchmark.json")

results = eval(
    benchmark=benchmark,
    judges=["openrouter:openai/gpt-4o-mini"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)
```

Use the CLI:

```bash
portex-eval format examples/benchmark.json

portex-eval run \
  --bundle examples/simple_bundle \
  --judge openrouter:openai/gpt-4o-mini \
  --judge openrouter:anthropic/claude-3.5-sonnet \
  --candidate openrouter:meta-llama/llama-3.3-70b-instruct
```

The standard evaluator writes a timestamped run directory under the configured runs root. By default that root is `./eval_runs`.

## Harbor Agent Evals

Generate Harbor tasks from the same bundle:

```bash
portex-eval agent-create \
  --bundle examples/simple_bundle \
  --output /tmp/simple_bundle_agent
```

Run Harbor against those generated tasks:

```bash
portex-eval agent-run \
  --tasks /tmp/simple_bundle_agent \
  --judge openrouter:openai/gpt-4o-mini \
  -- \
  --env modal \
  --agent terminus-2 \
  --model openrouter/openai/gpt-4o-mini \
  --jobs-dir /tmp/simple_bundle_agent_jobs
```

Arguments after `--` are forwarded to `harbor run`.

## Outputs

A standard eval run produces:

```text
<results.output_dir>/
├── logs/
│   └── *.eval
├── manifest.json
├── reports/
│   ├── eval_level.csv
│   ├── task_level.csv
│   ├── criterion_level.csv
│   └── judgement_level.csv
├── rl_rewards.json
└── rl_training_data.json
```

Harbor runs expose the output root, `datasets_dir`, and `jobs_dir` separately through `AgentEvalResults`.

## Documentation

- [Getting Started](docs/getting-started.md)
- [Bundle Format](docs/bundle-format.md)
- [Configuration](docs/configuration.md)
- [Providers](docs/providers.md)
- [API Reference](docs/api-reference.md)
- [Outputs](docs/outputs.md)
- [Analysis](docs/analysis.md)
- [Development](docs/development.md)

## Examples

- [`examples/simple_bundle/`](examples/simple_bundle/) - Minimal valid bundle
- [`examples/benchmark.json`](examples/benchmark.json) - Simple BYOB benchmark input
- [`examples/run_spec.yaml`](examples/run_spec.yaml) - Standard run spec example

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
