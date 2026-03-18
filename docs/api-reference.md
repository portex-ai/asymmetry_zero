# API Reference

This document describes the programmatic API for `portex-eval`.

## Core Functions

### eval()

Run an evaluation benchmark and return results.

```python
from portex_eval import eval

results = eval(
    path="./mybenchmark",       # Path to bundle directory
    judges=["openrouter:..."],  # Judge model strings or config objects
    candidates=["openrouter:..."],  # Candidate model strings or config objects
    output_dir=None,            # Optional output directory
    config=None,                # Optional Config instance
    task_spec=None,             # Optional task specification override
    overwrite=False,            # Allow overwriting existing outputs
)
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | One of path/benchmark | Path to the bundle directory |
| `benchmark` | `Benchmark` | One of path/benchmark | Benchmark instance from `create_benchmark()` |
| `judges` | `list[str | dict]` | Yes | List of judge model strings or config objects |
| `candidates` | `list[str | dict]` | Yes | List of candidate model strings or config objects |
| `output_dir` | `str` | No | Output directory. Defaults to `./eval_runs/<run_id>/` |
| `config` | `Config` | No | Runtime configuration. Defaults to `Config.from_env()` |
| `task_spec` | `str` | No | Task specification override |
| `overwrite` | `bool` | No | If True, allow overwriting existing outputs |

#### Returns

`EvalResults` - Results object with paths to logs, reports, and rewards.

#### Raises

- `PortexEvalError` - Validation or runtime errors

#### Example

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
```

Custom endpoints can be supplied per model:

```python
results = eval(
    path="./examples/simple_bundle",
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

---

### create_benchmark()

Create a Portex bundle from a BYOB JSON file.

```python
from portex_eval import create_benchmark

benchmark = create_benchmark("./mybench.json")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | Yes | Path to the input JSON file |

#### Returns

`Benchmark` - Benchmark descriptor with path and task count.

#### Raises

- `PortexEvalError` - If the input file is invalid or missing

#### Input Format

The input JSON must be a list of objects with `task`, `criteria`, and optional `reference_file`:

```json
[
  {
    "task": "What is 2+2?",
    "criteria": [
      {
        "id": "math-exact",
        "name": "Exact answer",
        "weight": 100,
        "grader_type": "ExactMatch",
        "semanticPrompt": "4"
      }
    ],
    "reference_file": ""
  }
]
```

#### Output

Creates a bundle directory adjacent to the input file:
- `./mybench.json` → `./mybench/tasks.json`, `./mybench/answers.json`, `./mybench/refs/`

---

### load_run_spec()

Load a RunSpec from a YAML file.

```python
from portex_eval import load_run_spec

spec = load_run_spec("./run_spec.yaml")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str | Path` | Yes | Path to the YAML run spec file |

#### Returns

`RunSpec` - Parsed run specification.

#### Raises

- `FileNotFoundError` - If the spec file does not exist
- `ValueError` - If the spec is malformed or missing required fields

---

## Types

### Benchmark

Returned by `create_benchmark()`.

```python
@dataclass(frozen=True)
class Benchmark:
    path: str        # Absolute path to the bundle directory
    task_count: int  # Number of tasks in the bundle
```

#### Methods

- `resolve_path() -> Path` - Return the absolute Path for the bundle

---

### EvalResults

Returned by `eval()`.

```python
@dataclass(frozen=True)
class EvalResults:
    logs: list[str]           # Paths to Inspect .log/.eval files
    reports: ReportPaths      # Paths to CSV reports
    rewards: str              # Path to rl_rewards.txt
    run_id: str               # Run identifier
    output_dir: str           # Output directory for this run
```

#### Methods

- `with_absolute_paths() -> EvalResults` - Return a copy with resolved absolute paths

---

### ReportPaths

Paths to CSV report files.

```python
@dataclass(frozen=True)
class ReportPaths:
    eval_level: str       # Path to eval_level.csv
    task_level: str       # Path to task_level.csv
    criterion_level: str  # Path to criterion_level.csv
    judgement_level: str  # Path to judgement_level.csv
```

---

### RunSpec

Specification for an evaluation run.

```python
@dataclass(frozen=True)
class RunSpec:
    bundle_path: str          # Path to the evaluation bundle
    judges: list[str]         # Judge model endpoints
    candidates: list[str]     # Candidate model endpoints
    schema_version: int = 1   # Schema version
```

#### Methods

- `validate_bundle_exists() -> Path` - Validate that the bundle path exists

---

### Config

Runtime configuration.

```python
@dataclass(frozen=True)
class Config:
    bundles_dir: str  # Directory for eval bundles (default: ./bundles)
    runs_dir: str     # Directory for run outputs (default: ./eval_runs)
    cache_dir: str    # Cache directory (default: ./.portex_cache)
```

#### Class Methods

- `from_env() -> Config` - Create from environment variables
- `from_dict(data: dict) -> Config` - Create from a dictionary

#### Methods

- `resolve_bundle_path(bundle_name: str) -> Path` - Resolve a bundle name to path
- `ensure_runs_dir() -> Path` - Ensure runs directory exists
- `ensure_cache_dir() -> Path` - Ensure cache directory exists

---

### PortexEvalError

Custom exception for validation and runtime errors.

```python
from portex_eval import PortexEvalError

try:
    results = eval(path="./missing", judges=[], candidates=[])
except PortexEvalError as e:
    print(f"Evaluation failed: {e}")
```

---

## Reports Module

Load CSV reports from evaluation results.

### reports.load()

```python
from portex_eval import reports

df = reports.load(results.reports.task_level)
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | `str` | Yes | Path to the CSV file |

#### Returns

`pandas.DataFrame` - Loaded report data.

---

## Providers Module

Access model provider abstractions.

### get_provider()

```python
from portex_eval import get_provider

provider = get_provider("openrouter:google/gemini-2.5-flash")
response = provider.generate("What is 2+2?")
print(response.text)
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_string` | `str` | Yes | Model identifier in format `provider:model_id` |
| `**kwargs` | | No | Additional provider-specific options |

#### Returns

`Provider` - Initialized provider instance.

### Provider (Abstract)

```python
class Provider(ABC):
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> Response: ...

    async def agenerate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> Response: ...

    @property
    def model_name(self) -> str: ...

    @property
    def provider_id(self) -> str: ...
```

### Response

```python
@dataclass
class Response:
    text: str                       # Generated text
    usage: dict[str, int] | None    # Token usage statistics
    raw: Any                        # Raw API response
```

---

## Rewards Module

Extract and write reward signals for RL training.

### extract_rewards()

```python
from portex_eval.rewards import extract_rewards

task_scores = extract_rewards("./reports/task_level.csv")
# Returns: [("task-001", 87.5), ("task-002", 100.0), ...]
```

### write_rewards()

```python
from portex_eval.rewards import write_rewards

path = write_rewards(task_scores, "./rl_rewards.txt")
```
