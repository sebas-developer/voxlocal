# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

The pre-commit hooks run:
- **ruff** — linting and formatting
- **mypy** — type checking
- **pytest** — quick test run

## Checks

```bash
ruff check .
mypy voxlocal/ --ignore-missing-imports
pytest
cov run -m pytest tests/ --cov=voxlocal --cov-report=term-missing
python -m build
```

Tests must not require downloaded models, network access, an audio device, or a
developer-specific cache.
