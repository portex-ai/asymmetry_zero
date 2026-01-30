# portex-eval

**Open-source evaluation framework for LLM judges and candidates**

`portex-eval` exposes the grading methodology used at [Portex](https://portex.ai) without distributing private eval-bundles. It runs benchmark workflows on user-provided bundles, emits standardized artifacts, and outputs task-level rewards for RL training pipelines.

## Features

- **Transparent grading** - Same scoring logic used internally at Portex
- **Judge panel** - Multi-model evaluation with configurable judge models
- **RL-ready outputs** - Task-level reward scores (`rl_rewards.txt`) for reward modeling
- **Inspect integration** - Built on [Inspect AI](https://inspect.ai) for rich logging and analysis
- **Bring your own benchmark** - Convert simple JSON to Portex bundle format

## Installation

```bash
pip install portex-eval
```

For full functionality including providers and Inspect integration:

```bash
pip install portex-eval[all]
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
)
```

### Using the CLI

```bash
# Format a benchmark.json into a Portex bundle
portex-eval format mybench.json

# Run an evaluation
portex-eval run --bundle ./mybenchmark \
    --judge openrouter/openai/gpt-4o \
    --judge openrouter/anthropic/claude-3.5-sonnet \
    --candidate openrouter/meta-llama/llama-3.3-70b-instruct

# See all options
portex-eval --help
portex-eval run --help
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
├── answers.json    # Reference answers and grading criteria
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
