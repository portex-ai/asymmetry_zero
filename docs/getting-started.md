# Getting Started

This guide covers the fastest path from install to a successful eval run.

## 1. Install

For most users, install everything:

```bash
pip install 'portex-eval[all]'
```

If you only need standard bundle evals, `portex-eval[inspect]` is enough. If you only need bundle formatting and report loading, the base `portex-eval` package is enough.

## 2. Configure API Access

The simplest path is OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

`portex-eval` also auto-loads a local `.env` file.

## 3. Run the Example Bundle

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

The run creates a timestamped directory under `./eval_runs` unless you override the runs root.

## 4. Format a BYOB Benchmark

Start from the shipped example input:

```bash
portex-eval format examples/benchmark.json
```

Or programmatically:

```python
from portex_eval import create_benchmark

benchmark = create_benchmark("./examples/benchmark.json")
print(benchmark.path)
print(benchmark.task_count)
```

`create_benchmark()` writes a sibling directory with a random suffix such as `examples/benchmark_a1b2c3d4/`.

## 5. Run From the CLI

```bash
portex-eval run \
  --bundle examples/simple_bundle \
  --judge openrouter:openai/gpt-4o-mini \
  --judge openrouter:anthropic/claude-3.5-sonnet \
  --candidate openrouter:meta-llama/llama-3.3-70b-instruct
```

See all flags with:

```bash
portex-eval run --help
```

## 6. Run From a Run Spec

The repository includes [`examples/run_spec.yaml`](../examples/run_spec.yaml):

```python
from portex_eval import eval, load_run_spec

spec = load_run_spec("./examples/run_spec.yaml")

results = eval(
    path=spec.bundle_path,
    judges=spec.judges,
    candidates=spec.candidates,
)
```

Run specs currently accept model strings, not provider config objects.

## 7. Optional: Harbor Agent Evals

Generate Harbor tasks:

```bash
portex-eval agent-create \
  --bundle examples/simple_bundle \
  --output /tmp/simple_bundle_agent
```

Run Harbor:

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

Harbor support requires the Harbor dependency stack:

```bash
pip install 'portex-eval[harbor]'
```

## 8. Inspect the Outputs

After a standard eval, the main artifacts are:

- `logs/*.eval`
- `reports/*.csv`
- `rl_rewards.json`
- `rl_training_data.json`

See [Outputs](outputs.md) and [Analysis](analysis.md) for follow-up workflows.

## Next Steps

- [Bundle Format](bundle-format.md)
- [Configuration](configuration.md)
- [Providers](providers.md)
- [API Reference](api-reference.md)
