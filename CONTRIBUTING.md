# Contributing to portex-eval

Thank you for your interest in contributing to `portex-eval`! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and inclusive. We welcome contributions from everyone.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- An [OpenRouter](https://openrouter.ai) API key for running tests

### Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/your-username/portex-eval.git
cd portex-eval
```

2. Create a virtual environment and install dependencies:

   With [UV](https://docs.astral.sh/uv/) (recommended):

   ```bash
   uv sync
   ```

   With pip:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. (pip only) If you used pip, activate the venv. With UV, use `uv run` for commands below.

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
```

5. Verify setup:

   ```bash
   uv run ruff format . && uv run ruff check . && uv run mypy . && uv run pytest
   ```
   (With pip and activated venv, omit `uv run`.)

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
Add OpenRouter rate limit retry logic

- Implement exponential backoff with jitter
- Add configurable retry parameters
- Handle Retry-After header from API
```

### Code Style

This project uses:
- **Ruff** for linting and formatting
- **mypy** for type checking

Before committing:

```bash
ruff format .       # Format code
ruff check --fix .  # Lint and auto-fix
mypy .              # Type check
pytest              # Run tests
```

### Type Hints

All public APIs must be fully typed:

```python
def create_benchmark(path: str) -> Benchmark:
    """Create a benchmark from a JSON file.

    Args:
        path: Path to the input JSON file.

    Returns:
        Benchmark with the generated bundle path.
    """
    ...
```

### Testing

Write tests for new functionality:

```python
def test_new_feature():
    """Test description of what this tests."""
    # Arrange
    ...
    # Act
    result = function_under_test(...)
    # Assert
    assert result.value == expected
```

Run tests:

```bash
pytest                           # All tests
pytest tests/test_api.py         # Specific file
pytest -k "test_name"            # By name pattern
pytest --cov=portex_eval         # With coverage
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass:
   ```bash
   ruff format . && ruff check . && mypy . && pytest
   ```
4. Update documentation if needed
5. Submit a pull request

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Type hints added for public APIs
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` on GitHub.

### Documentation

- Improve existing documentation
- Add examples and tutorials
- Fix typos and clarify explanations

### New Providers

Add support for additional model providers:

1. Create `portex_eval/providers/newprovider.py`
2. Implement the `Provider` abstract class
3. Register in `portex_eval/providers/__init__.py`
4. Add tests
5. Update `docs/providers.md`
6. Add environment variables to `.env.example`

See [docs/development.md](docs/development.md#adding-a-provider) for details.

### Bug Fixes

- Report bugs with clear reproduction steps
- Submit fixes with tests that verify the fix

### Feature Requests

Open an issue to discuss new features before implementing.

## Project Structure

```
portex-eval/
├── portex_eval/           # Main package
│   ├── __init__.py        # Public exports
│   ├── api.py             # Core API
│   ├── config.py          # Configuration
│   ├── errors.py          # Exceptions
│   ├── run_spec.py        # YAML parsing
│   ├── types.py           # Data types
│   ├── benchmark/         # Runner logic
│   ├── providers/         # Model providers
│   ├── reporting/         # CSV reports
│   └── rewards/           # RL rewards
├── tests/                 # Test suite
├── examples/              # Examples
├── docs/                  # Documentation
└── pyproject.toml         # Project config
```

## Questions?

- Open a GitHub issue for questions
- Check existing issues and discussions
- Read the [documentation](docs/)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
