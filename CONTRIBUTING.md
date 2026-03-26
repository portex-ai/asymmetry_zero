# Contributing to portex-eval

Thanks for contributing.

## Before You Open a PR

Set up the repo with the development workflow in [docs/development.md](docs/development.md), then run:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pytest
```

GitHub Actions will rerun the unit suite on supported Python versions and publish a coverage report for the branch or pull request.

## Expectations

- Add or update tests for behavioral changes.
- Update docs when public behavior changes.
- Keep examples in `examples/` runnable.
- Do not commit local environment files or generated artifacts.

## Scope

Small fixes can go straight to a PR. For larger feature work or API changes, open an issue or discussion first so the surface area is agreed before implementation.

## License

By contributing, you agree that your contributions will be released under the MIT License.
