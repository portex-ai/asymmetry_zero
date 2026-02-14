# Getting Started

This guide walks through installing `portex-eval` and running your first evaluation.

## Prerequisites

- Python 3.10 or later
- An [OpenRouter](https://openrouter.ai) API key

## Installation

### Basic installation

With [UV](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install portex-eval
```

With pip:

```bash
pip install portex-eval
```

### Full installation with all features

```bash
uv tool install 'portex-eval[all]'
# or: pip install portex-eval[all]
```

This includes:
- **providers** - HTTP client for OpenRouter API
- **inspect** - Inspect AI integration for logging and analysis

### Development installation

With UV (recommended):

```bash
git clone https://github.com/portex-ai/portex-eval.git
cd portex-eval
uv sync
uv run portex-eval --help
```

With pip:

```bash
git clone https://github.com/portex-ai/portex-eval.git
cd portex-eval
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Configuration

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your API key
```

## Your First Evaluation

### Option 1: Use the example bundle

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
)

print(f"Run completed: {results.run_id}")
print(f"Output directory: {results.output_dir}")
print(f"Rewards file: {results.rewards}")
```

### Option 2: Bring your own benchmark

Create a simple JSON file with your tasks:

```json
[
  {
    "task": "What is the capital of France?",
    "answer": "Paris",
    "reference_file": ""
  },
  {
    "task": "What is 2 + 2?",
    "answer": "4",
    "reference_file": ""
  }
]
```

Convert and evaluate:

```python
from portex_eval import create_benchmark, eval

# Convert to Portex bundle format
benchmark = create_benchmark("./mybench.json")
print(f"Created bundle with {benchmark.task_count} tasks at {benchmark.path}")

# Run evaluation
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

### Option 3: Use a run spec file

Create `run_spec.yaml`:

```yaml
schema_version: 1
bundle_path: ./examples/simple_bundle
judges:
  - openrouter:openai/gpt-4o
  - openrouter:anthropic/claude-3.5-sonnet
  - openrouter:google/gemini-2.5-flash
candidates:
  - openrouter:meta-llama/llama-3.3-70b-instruct
```

Load and run:

```python
from portex_eval import load_run_spec, eval

spec = load_run_spec("./run_spec.yaml")

results = eval(
    path=spec.bundle_path,
    judges=spec.judges,
    candidates=spec.candidates,
)
```

## Viewing Results

After an evaluation completes, explore the outputs:

```python
from portex_eval import reports

# Load task-level scores
task_df = reports.load(results.reports.task_level)
print(task_df[["task_id", "score"]])

# Load criterion-level breakdown
criterion_df = reports.load(results.reports.criterion_level)
print(criterion_df.head())
```

Read the reward file for RL training:

```python
with open(results.rewards) as f:
    for line in f:
        task_id, score = line.strip().split()
        print(f"{task_id}: {score}")
```

## Next Steps

- [Bundle Format](bundle-format.md) - Learn the full bundle schema
- [API Reference](api-reference.md) - Explore the programmatic API
- [Outputs](outputs.md) - Understand output formats
- [Providers](providers.md) - Configure model endpoints
- [Analysis](analysis.md) - Deep dive into evaluation results
