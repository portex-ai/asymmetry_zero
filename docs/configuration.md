# Configuration

This guide covers all configuration options for `portex-eval`.

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter provider |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `PORTEX_BUNDLES_DIR` | `./bundles` | Directory for eval bundles |
| `PORTEX_RUNS_DIR` | `./eval_runs` | Directory for run outputs |
| `PORTEX_CACHE_DIR` | `./.portex_cache` | Cache directory |
| `PORTEX_JUDGE_MODELS` | (none) | Comma-separated judge models (internal use) |

### Setting Variables

#### Shell export

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export PORTEX_BUNDLES_DIR="./my-bundles"
export PORTEX_RUNS_DIR="./my-runs"
```

#### .env file

Create a `.env` file in your project root:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-...

# Optional paths
PORTEX_BUNDLES_DIR=./bundles
PORTEX_RUNS_DIR=./eval_runs
PORTEX_CACHE_DIR=./.portex_cache
```

The library loads `.env` automatically via `python-dotenv`.

## Config Class

Use the `Config` class for programmatic configuration:

```python
from portex_eval import Config, eval

# Create from environment (default)
config = Config.from_env()

# Create with explicit values
config = Config(
    bundles_dir="./my-bundles",
    runs_dir="./my-runs",
    cache_dir="./.cache",
)

# Create from dictionary
config = Config.from_dict({
    "bundles_dir": "./my-bundles",
    "runs_dir": "./my-runs",
})

# Use in eval
results = eval(
    path="./mybenchmark",
    judges=["openrouter:openai/gpt-4o"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    config=config,
)
```

### Config Attributes

```python
@dataclass(frozen=True)
class Config:
    bundles_dir: str  # Directory for eval bundles
    runs_dir: str     # Directory for run outputs
    cache_dir: str    # Cache directory
```

### Config Methods

```python
# Resolve a bundle name to its full path
path = config.resolve_bundle_path("mybenchmark")
# Returns: Path("/absolute/path/to/bundles/mybenchmark")

# Ensure runs directory exists
runs_path = config.ensure_runs_dir()
# Creates directory if needed, returns Path

# Ensure cache directory exists
cache_path = config.ensure_cache_dir()
# Creates directory if needed, returns Path
```

## Run Spec Configuration

For batch runs, use YAML run spec files:

```yaml
schema_version: 1

bundle_path: ./mybenchmark

judges:
  - openrouter:openai/gpt-4o
  - openrouter:anthropic/claude-3.5-sonnet
  - openrouter:google/gemini-2.5-flash

candidates:
  - openrouter:meta-llama/llama-3.3-70b-instruct
```

### Loading Run Specs

```python
from portex_eval import load_run_spec, eval

spec = load_run_spec("./run_spec.yaml")

results = eval(
    path=spec.bundle_path,
    judges=spec.judges,
    candidates=spec.candidates,
)
```

### Run Spec Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | integer | Yes | Must be `1` |
| `bundle_path` | string | Yes | Path to the eval bundle |
| `judges` | list[string] | Yes | List of judge model identifiers |
| `candidates` | list[string] | Yes | List of candidate model identifiers |

## Provider Configuration

### OpenRouter

Configure the OpenRouter provider:

```python
from portex_eval import get_provider

provider = get_provider(
    "openrouter:google/gemini-2.5-flash",
    api_key="sk-or-v1-...",  # Override OPENROUTER_API_KEY
    max_retries=5,           # Rate limit retries (default: 5)
    base_delay=1.0,          # Initial backoff delay seconds (default: 1.0)
    max_delay=60.0,          # Maximum backoff delay seconds (default: 60.0)
    jitter=0.5,              # Jitter factor 0-1 (default: 0.5)
    timeout=120.0,           # Request timeout seconds (default: 120.0)
)
```

## Output Directory Configuration

Control where results are written:

```python
from portex_eval import eval

# Use default (./eval_runs/<run_id>/)
results = eval(
    path="./mybenchmark",
    judges=["openrouter:openai/gpt-4o"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)

# Specify custom output directory
results = eval(
    path="./mybenchmark",
    judges=["openrouter:openai/gpt-4o"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    output_dir="./custom-output",
)

# Allow overwriting existing outputs
results = eval(
    path="./mybenchmark",
    judges=["openrouter:openai/gpt-4o"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    output_dir="./existing-run",
    overwrite=True,
)
```

## Directory Layout

Default directory structure:

```
project/
├── .env                    # Environment variables
├── bundles/                # PORTEX_BUNDLES_DIR
│   └── mybenchmark/
│       ├── tasks.json
│       ├── answers.json
│       └── refs/
├── eval_runs/              # PORTEX_RUNS_DIR
│   └── 20240115_143022.../
│       ├── eval_log.json
│       ├── reports/
│       └── rl_rewards.txt
└── .portex_cache/          # PORTEX_CACHE_DIR
```

## Logging

`portex-eval` uses Python's standard logging. Configure as needed:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific loggers
logging.getLogger("portex_eval").setLevel(logging.INFO)
logging.getLogger("portex_eval.providers").setLevel(logging.WARNING)
```

Log messages include:
- Rate limit warnings with retry delays
- Request errors and retries
- Bundle validation messages

## Best Practices

### Production Configuration

```bash
# .env for production
OPENROUTER_API_KEY=sk-or-v1-...
PORTEX_BUNDLES_DIR=/data/eval-bundles
PORTEX_RUNS_DIR=/data/eval-runs
PORTEX_CACHE_DIR=/data/.portex-cache
```

### CI/CD Configuration

```yaml
# GitHub Actions example
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  PORTEX_RUNS_DIR: ./ci-runs
```

### Multiple Environments

```python
import os
from portex_eval import Config

# Development
if os.getenv("ENV") == "development":
    config = Config(
        bundles_dir="./dev-bundles",
        runs_dir="./dev-runs",
    )
# Production
else:
    config = Config(
        bundles_dir="/data/bundles",
        runs_dir="/data/runs",
    )
```
