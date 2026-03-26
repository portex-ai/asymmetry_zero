# Development

This guide covers local development for `portex-eval`.

## Setup

Clone the repository and install the default development stack:

```bash
git clone https://github.com/portex-ai/portex-eval
cd portex-eval
uv sync
```

That install includes the normal dev tools plus Inspect. Add Harbor only when you need it:

```bash
uv sync --group harbor
```

Editable pip install also works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Day-to-Day Commands

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pytest
```

The automated test suite does not require live provider keys. You only need API keys for manual end-to-end runs against real model providers.

## Project Layout

```text
portex_eval/
  api.py
  cli.py
  config.py
  run_spec.py
  types.py
  benchmark/
  bundle/
  grading/
  providers/
  reporting/
  rewards/
tests/
docs/
examples/
```

## Manual Smoke Tests

Standard CLI surface:

```bash
uv run portex-eval --help
uv run portex-eval run --help
uv run portex-eval agent-run --help
```

Example benchmark formatting:

```bash
uv run portex-eval format examples/benchmark.json
```

## GitHub CI

GitHub Actions now runs the unit test suite on Python 3.10, 3.11, and 3.12 via `.github/workflows/ci.yml`.

The workflow also runs a dedicated coverage job on Python 3.12 that:

- executes `pytest --cov=portex_eval`
- writes a coverage summary into the GitHub run summary
- uploads `coverage.xml`, `.coverage`, and `htmlcov/` as workflow artifacts

## Adding a Provider

If you add a new provider:

1. Implement the provider under `portex_eval/providers/`.
2. Register it in `portex_eval/providers/__init__.py`.
3. Add tests under `tests/`.
4. Update [Providers](providers.md) if the public config surface changed.

## Documentation

Keep the docs aligned with the current runtime, especially:

- Install extras and Python-version requirements
- Output artifact names and formats
- Example paths in `examples/`
- Public API fields in `types.py`
