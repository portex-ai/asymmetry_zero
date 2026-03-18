# portex-eval

**Open-source evaluation framework for LLM judges and candidates**

`portex-eval` exposes the grading methodology used at [Portex](https://portexai.com) without distributing private eval-bundles. It runs benchmark workflows on user-provided bundles, emits standardized artifacts, and outputs task-level rewards for RL training pipelines.

## Features

- **Transparent grading** - Same scoring logic used internally at Portex
- **Judge panel** - Multi-model evaluation with configurable judge models
- **RL-ready outputs** - Task-level reward scores (`rl_rewards.txt`) for reward modeling
- **Inspect integration** - Built on [Inspect AI](https://inspect.ai) for rich logging and analysis
- **Bring your own benchmark** - Convert simple JSON to Portex bundle format

## Installation

With [UV](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install portex-eval
```

Or from source (e.g. for development):

```bash
git clone https://github.com/portex-ai/portex-eval && cd portex-eval
uv sync
uv run portex-eval --help
```

With pip:

```bash
pip install portex-eval
```

For full functionality including providers and Inspect integration:

```bash
uv tool install 'portex-eval[all]'
# or: pip install portex-eval[all]
```

## Quick Start

### Option 1: Using an existing bundle

```python
from portex_eval import eval

results = eval(
    path="./mybenchmark",  # Directory with tasks.json, answers.json, refs/
    judges=[
        "openrouter:openai/gpt-4o",
        "openrouter:anthropic/claude-3.5-sonnet",
        "openrouter:google/gemini-2.5-flash",
    ],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)

# Access results
print(f"Run ID: {results.run_id}")
print(f"Rewards: {results.rewards}")
```

### Option 2: Bring your own benchmark (BYOB)

```python
from portex_eval import create_benchmark, eval

# Convert simple JSON to Portex bundle format
benchmark = create_benchmark("./mybench.json")

results = eval(
    benchmark=benchmark,
    judges=[
        "openrouter:openai/gpt-4o",
        "openrouter:anthropic/claude-3.5-sonnet",
        "openrouter:google/gemini-2.5-flash",
    ],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    max_samples=4,
)
```

### Using the CLI

If you installed with `uv tool install`, run `portex-eval` directly. If you used `uv sync` from source, use `uv run portex-eval`:

```bash
# Format a benchmark.json into a Portex bundle
portex-eval format mybench.json
# or from source: uv run portex-eval format mybench.json

# Run an evaluation
portex-eval run \
  --bundle examples/simple_bundle \
  --judge openrouter:openai/gpt-4o-mini \
  --judge openrouter:anthropic/claude-3.5-sonnet \
  --judge openrouter:google/gemini-2.5-flash \
  --candidate-config '{"provider":"vllm","model":"Qwen/Qwen3-VL-4B-Instruct","base_url":"https://portex--qwen3-vl-4b-instruct-vllm-baseline-serve.modal.run/v1"}'

# Mixed providers and a custom Modal/vLLM endpoint
portex-eval run \
  --bundle examples/simple_bundle \
  --judge openrouter:openai/gpt-4o-mini \
  --judge-config '{"provider":"anthropic","model":"claude-sonnet-4-5"}' \
  --candidate-config '{"provider":"vllm","model":"Qwen/Qwen3-VL-4B-Instruct","base_url":"https://portex--qwen3-vl-4b-instruct-vllm-baseline-serve.modal.run/v1"}'

# See all options
portex-eval --help
portex-eval run --help
```

Programmatic mixed-provider runs can also use config objects:

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

### Analyzing results

```python
from portex_eval import reports

# Load CSV reports
eval_df = reports.load(results.reports.eval_level)
task_df = reports.load(results.reports.task_level)
criterion_df = reports.load(results.reports.criterion_level)
judgement_df = reports.load(results.reports.judgement_level)

# Or use Inspect AI analysis helpers
from inspect_ai.analysis import eval_df, samples_df

samples = samples_df(results.logs)
evals = eval_df(results.logs)
```

## Configuration

Set environment variables or use the `Config` class:

```bash
export OPENROUTER_API_KEY="your-api-key"
export PORTEX_BUNDLES_DIR="./bundles"
export PORTEX_RUNS_DIR="./eval_runs"
```

See [Configuration Guide](docs/configuration.md) for all options.

## Bundle Format

A Portex bundle is a directory containing:

```
mybenchmark/
├── tasks.json      # Task prompts with IDs
├── answers.json    # Grading criteria and verifier config
└── refs/           # Optional reference files (images, etc.)
```

See [Bundle Format Guide](docs/bundle-format.md) for schema details.

## Outputs

Each evaluation run produces:

| Output | Description |
|--------|-------------|
| `eval_log.json` | Inspect AI evaluation log |
| `reports/eval_level.csv` | Aggregate metrics per evaluation |
| `reports/task_level.csv` | Per-task scores |
| `reports/criterion_level.csv` | Per-criterion breakdown |
| `reports/judgement_level.csv` | Individual judge verdicts |
| `rl_rewards.txt` | Task-level reward scores for RL |

See [Outputs Guide](docs/outputs.md) for format details.

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and first evaluation
- [Bundle Format](docs/bundle-format.md) - Eval bundle schema reference
- [API Reference](docs/api-reference.md) - Programmatic API documentation
- [Outputs](docs/outputs.md) - Output file formats and analysis
- [Providers](docs/providers.md) - Model provider configuration
- [Analysis](docs/analysis.md) - Inspecting and analyzing results
- [Configuration](docs/configuration.md) - Environment and runtime options
- [Development](docs/development.md) - Contributing and local setup

## Examples

See the [`examples/`](examples/) directory for:

- [`simple_bundle/`](examples/simple_bundle/) - Minimal valid bundle
- [`benchmark.json`](examples/benchmark.json) - BYOB input format
- [`run_spec.yaml`](examples/run_spec.yaml) - Batch run configuration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
