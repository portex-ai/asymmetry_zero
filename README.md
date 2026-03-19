# portex-eval

**Open-source evaluation framework for LLM judges and candidates**

`portex-eval` exposes the grading methodology used at [Portex](https://portexai.com) without distributing private eval-bundles. It runs benchmark workflows on user-provided bundles, emits standardized artifacts, and outputs task-level rewards for RL training pipelines.

## Features

- **Transparent grading** - Same scoring logic used internally at Portex
- **Judge panel** - Multi-model evaluation with configurable judge models
- **RL-ready outputs** - Task-level reward scores (`rl_rewards.txt`) for reward modeling
- **Inspect integration** - Built on [Inspect AI](https://inspect.ai) for rich logging and analysis
- **Harbor agent evals** - Generate Harbor tasks and run agentic evals from the same Portex bundle
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

This default `uv sync` installs the standard development stack only. Harbor is kept out of the default dev environment so Linux workflows do not fail on Harbor's heavier optional dependency chain.

With pip:

```bash
pip install portex-eval
```

For full functionality including providers, Inspect, and Harbor:

```bash
uv tool install 'portex-eval[all]'
# or: pip install portex-eval[all]
```

If you only need Harbor-backed agent evals in addition to the base package:

```bash
pip install 'portex-eval[harbor]'
```

From source with `uv`, install Harbor explicitly when you need agent evals:

```bash
uv sync --group harbor
# or, to install the package extra as well:
uv sync --extra harbor
```

`portex-eval` pins the Harbor stack to the same known-good versions used in `harbor-portex-bench`:

- `harbor==0.1.42`
- `claude-agent-sdk==0.1.36`

## Quick Start

### Option 1: Using an existing bundle

```python
from portex_eval import eval

results = eval(
    path="./examples/simple_bundle",
    judges=[
        "openrouter:openai/gpt-4o",
        "openrouter:anthropic/claude-3.5-sonnet",
        "openrouter:google/gemini-2.5-flash",
    ],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    output_dir="/examples/jobs",

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
  --candidate openrouter:openai/gpt-5.2 \
  --output examples/jobs/simple_bundle

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

### Agent Evals With Harbor

Start from the same Portex bundle format, then generate Harbor tasks and run Harbor against them:

```bash
# Create Harbor task directories from a Portex bundle
portex-eval agent-create \
  --bundle examples/simple_bundle \
  --output examples/simple_bundle_agent

# Run Harbor on the generated tasks
portex-eval agent-run   \
    --tasks examples/simple_bundle_agent \
    --judge openrouter:openai/gpt-4o-mini  \
    --   \
    --env modal   \
    --agent terminus-2  \
    --model openrouter/openai/gpt-5.4  \
    --jobs-dir examples/jobs/simple_bundle_agent
```

For image-heavy tasks where we want a `terminus-2`-style Harbor agent to receive the
reference image directly on its first model turn instead of discovering it through shell
tools, use the built-in Portex multimodal Harbor agent:

```bash
portex-eval agent-run \
  --tasks examples/simple_bundle_img_agent \
  --judge openrouter:google/gemini-2.5-flash \
  -- \
  --env modal \
  --agent portex-multimodal \
  --model openrouter/google/gemini-3.1-pro-preview \
  --jobs-dir examples/jobs/simple_bundle_img_agent \
  --ak max_turns=10
```

You can also place the `agent-run` options in a YAML file and pass them with `--config`:

```bash
portex-eval agent-run --config examples/simple_bundle_agent/run_spec.yaml
```

Generated Harbor task instructions now include the exact `/app/refs/...` path, and
`agent-run` automatically injects known `model_info` metadata for supported custom vision
models when Harbor/LiteLLM needs it.

`agent-run` writes Harbor datasets and job outputs only. It does not generate the
Inspect-specific `reports/` CSVs or RL training artifacts.

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

Programmatic Harbor workflows are available too:

```python
from portex_eval import agent_eval, create_agent_eval

bundle = create_agent_eval(
    path="./examples/simple_bundle",
    output_dir="./agent_eval_tasks/simple_bundle",
)

results = agent_eval(
    task_root=bundle.path,
    judges=["openrouter:openai/gpt-4o-mini"],
    extra_args=["--model", "demo-agent"],
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
