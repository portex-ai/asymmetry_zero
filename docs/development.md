# Development

This guide covers setting up a development environment and contributing to `portex-eval`.

## Prerequisites

- Python 3.10 or later
- Git
- An [OpenRouter](https://openrouter.ai) API key for running tests

## Setup

### Clone the repository

```bash
git clone https://github.com/portex-ai/portex-eval.git
cd portex-eval
```

### Create a virtual environment and install dependencies

With [UV](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

UV creates a `.venv` and installs the project with dev dependencies (pytest, mypy, ruff, etc.). Run commands with `uv run`:

```bash
uv run pytest
uv run portex-eval --help
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

This installs:
- Core dependencies (pyyaml, pandas, python-dotenv)
- Provider dependencies (httpx)
- Inspect AI integration
- Development tools (pytest, mypy, ruff)

### Configure environment

```bash
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
```

## Development Workflow

### Running tests

If you use UV, prefix commands with `uv run` (e.g. `uv run pytest`).

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run specific test
pytest tests/test_api.py::test_eval_requires_path_or_benchmark

# Run with coverage
pytest --cov=portex_eval --cov-report=html
```

### Type checking

```bash
mypy .
```

### Linting

```bash
# Check for issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Formatting

```bash
ruff format .
```

### Run all checks

```bash
ruff format . && ruff check . && mypy . && pytest
```

## Project Structure

```
portex-eval/
├── portex_eval/           # Main package
│   ├── __init__.py        # Public API exports
│   ├── api.py             # High-level API (create_benchmark, eval)
│   ├── config.py          # Runtime configuration
│   ├── errors.py          # Custom exceptions
│   ├── run_spec.py        # YAML run spec parsing
│   ├── types.py           # Core data types
│   ├── benchmark/         # Benchmark runner (from portex-bench)
│   │   └── inspect/       # Inspect AI integration
│   ├── providers/         # Model provider adapters
│   │   ├── base.py        # Abstract Provider class
│   │   └── openrouter.py  # OpenRouter implementation
│   ├── reporting/         # CSV report generation
│   │   └── tables.py      # Report table generation
│   └── rewards/           # RL rewards extraction
│       └── writer.py      # Rewards file writer
├── portex-bench/          # Separate subproject (excluded from lint)
├── tests/                 # Test suite
├── examples/              # Example bundles and configs
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── CONTRIBUTING.md        # Contribution guidelines
```

## Code Style

### Python version

Target Python 3.10+ with type hints throughout.

### Formatting

- Line length: 100 characters
- Formatter: Ruff

### Linting rules

Ruff is configured with these rule sets:
- `E` - pycodestyle errors
- `F` - pyflakes
- `I` - isort (import sorting)
- `UP` - pyupgrade
- `B` - flake8-bugbear
- `C4` - flake8-comprehensions

### Type hints

All public APIs should be fully typed:

```python
def create_benchmark(path: str) -> Benchmark:
    """Create a benchmark from a JSON file."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def eval(
    *,
    path: str | None = None,
    benchmark: Benchmark | None = None,
    judges: list[str],
    candidates: list[str],
) -> EvalResults:
    """Run an evaluation benchmark.

    Args:
        path: Path to the bundle directory.
        benchmark: Benchmark instance from create_benchmark().
        judges: List of judge model identifiers.
        candidates: List of candidate model identifiers.

    Returns:
        EvalResults with paths to logs, reports, and rewards.

    Raises:
        PortexEvalError: If validation fails.
    """
    ...
```

### Naming conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_leading_underscore`

### Imports

Organize imports in this order (enforced by Ruff):
1. Standard library
2. Third-party packages
3. Local imports

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from portex_eval.config import Config
from portex_eval.errors import PortexEvalError
```

## Adding a Provider

To add support for a new model provider:

1. Create `portex_eval/providers/newprovider.py`:

```python
from portex_eval.providers.base import Provider, Response

class NewProvider(Provider):
    def __init__(self, model_id: str, *, api_key: str | None = None):
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("NEWPROVIDER_API_KEY")
        if not self._api_key:
            raise ValueError("API key required")

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def provider_id(self) -> str:
        return "newprovider"

    def generate(self, prompt: str, **kwargs) -> Response:
        # Implementation
        ...

    async def agenerate(self, prompt: str, **kwargs) -> Response:
        # Async implementation
        ...

def create_newprovider(model_id: str, **kwargs) -> NewProvider:
    return NewProvider(model_id, **kwargs)
```

2. Register in `portex_eval/providers/__init__.py`:

```python
from portex_eval.providers.newprovider import NewProvider, create_newprovider

_PROVIDER_REGISTRY: dict[str, ProviderFactory] = {
    "openrouter": create_openrouter_provider,
    "newprovider": create_newprovider,
}
```

3. Add tests in `tests/test_providers.py`

4. Update documentation in `docs/providers.md`

5. Add environment variable to `.env.example`

## Testing

### Test structure

Tests are in `tests/` with naming convention `test_*.py`:

```python
import pytest
from portex_eval import create_benchmark, eval
from portex_eval.errors import PortexEvalError

def test_create_benchmark_validates_input(tmp_path):
    """Test that create_benchmark validates input format."""
    ...

def test_eval_requires_judges():
    """Test that eval requires at least one judge."""
    with pytest.raises(PortexEvalError, match="At least one judge"):
        eval(path="./bundle", judges=[], candidates=["model"])
```

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def sample_bundle(tmp_path):
    """Create a minimal valid bundle for testing."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    # Create tasks.json, answers.json, refs/
    return str(bundle_dir)
```

### Mocking providers

For tests that don't need real API calls:

```python
from unittest.mock import Mock, patch

def test_eval_without_api(sample_bundle):
    with patch("portex_eval.api.benchmark_one") as mock_run:
        mock_run.return_value = Mock(...)
        results = eval(...)
```

## Releasing

1. Update version in `portex_eval/__init__.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v0.1.0`
4. Push: `git push origin main --tags`
5. Build: `python -m build`
6. Upload: `twine upload dist/*`

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Read the [API Reference](api-reference.md)
